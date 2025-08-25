# -*- coding: utf-8 -*-


import argparse
import json
from dataclasses import dataclass
from math import atan2, degrees
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import ezdxf
from ezdxf.entities import Line, LWPolyline, Polyline, Circle, Arc, Text, MText, Dimension
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

# ---------------------- 單位處理 ----------------------
INSUNITS_TO_MM = {
    0: None, 1: 25.4, 2: 304.8, 3: 1609344.0, 4: 1.0, 5: 10.0, 6: 1000.0, 7: 1_000_000.0,
    8: 0.0000254, 9: 0.0254, 10: 914.4, 11: 1e-7, 12: 1e-6, 13: 0.001, 14: 100.0,
    15: 10000.0, 16: 100000.0, 17: 1e12, 18: 1.495978707e14, 19: 9.4607e18, 20: 3.0857e19,
}

def get_unit_scale_mm(doc, assume_unit: Optional[str]) -> float:
    insunits = int(doc.header.get("$INSUNITS", 0) or 0)
    s = INSUNITS_TO_MM.get(insunits)
    if s is None:
        if assume_unit:
            u = assume_unit.lower()
            return {"mm":1.0, "cm":10.0, "m":1000.0, "in":25.4, "ft":304.8}.get(u, 1.0)
        return 1.0
    return s

# ---------------------- 資料結構 ----------------------
@dataclass
class Issue:
    code: str
    message: str
    at: Tuple[float, float]
    handles: Tuple[str, ...]
    layer: str

# ---------------------- 工具函式 ----------------------
def line_to_linestring(e: Line) -> LineString:
    p1 = (e.dxf.start.x, e.dxf.start.y)
    p2 = (e.dxf.end.x, e.dxf.end.y)
    return LineString([p1, p2])

def pline_to_linestrings(e: LWPolyline) -> List[LineString]:
    pts = [(v[0], v[1]) for v in e.get_points()]
    segs = []
    for i in range(len(pts) - 1):
        segs.append(LineString([pts[i], pts[i+1]]))
    if e.closed:
        segs.append(LineString([pts[-1], pts[0]]))
    return segs

def angle_of(p1, p2) -> float:
    return degrees(atan2(p2[1]-p1[1], p2[0]-p1[0]))

def nearly_parallel(a1: float, a2: float, ang_tol: float) -> bool:
    d = abs(((a1 - a2 + 180) % 360) - 180)
    return d <= ang_tol

def bbox_polygon_from_layer(msp, layer_name: str, tol: float) -> Optional[Polygon]:
    lines = []
    for e in msp.query(f"LINE[layer=='{layer_name}']"):
        lines.append(line_to_linestring(e))
    for e in msp.query(f"LWPOLYLINE[layer=='{layer_name}']"):
        for seg in pline_to_linestrings(e):
            lines.append(seg)
    if not lines:
        return None
    merged = unary_union(lines)
    minx, miny, maxx, maxy = merged.bounds
    return Polygon([(minx-tol, miny-tol), (maxx+tol, miny-tol),
                    (maxx+tol, maxy+tol), (minx-tol, maxy+tol)])

def classify_entity(e) -> Optional[str]:
    if isinstance(e, (Line, LWPolyline, Polyline)):
        ln = (e.dxf.layer or "").upper()
        if "WALL" in ln: return "WALL"
        if "DOOR" in ln: return "DOOR"
        return "LINEWORK"
    if isinstance(e, (Text, MText)): return "TEXT"
    if isinstance(e, Dimension): return "DIM"
    if isinstance(e, (Circle, Arc)): return "ARC"
    return None

# ---------------------- 規則實作 ----------------------
def rule_short_segments(msp, min_seg_len: float) -> List[Issue]:
    issues: List[Issue] = []
    for e in msp.query("LINE"):
        ls = line_to_linestring(e)
        if ls.length < min_seg_len:
            mid = ls.interpolate(0.5, normalized=True)
            issues.append(Issue("SHORT_SEG", f"線段過短 {ls.length:.3f}",
                                (mid.x, mid.y), (e.dxf.handle,), e.dxf.layer))
    for e in msp.query("LWPOLYLINE"):
        for seg in pline_to_linestrings(e):
            if seg.length < min_seg_len:
                mid = seg.interpolate(0.5, normalized=True)
                issues.append(Issue("SHORT_SEG", f"PLINE子段過短 {seg.length:.3f}",
                                    (mid.x, mid.y), (e.dxf.handle,), e.dxf.layer))
    return issues

# （其他規則如 DUP_LINE, OVERLAP_LINE, OPEN_PLINE, OUTSIDE, WRONG_LAYER... 同之前，不再贅述）
# 這裡略過規則程式碼，保留原本你的版本

# ---------------------- DXF 輸出 ----------------------
def export_errors_dxf(issues: List[Issue], out_path: Path, base_doc):
    doc = ezdxf.new(dxfversion=base_doc.dxfversion)
    msp = doc.modelspace()
    if "ERRORS" not in doc.layers:
        doc.layers.new("ERRORS", dxfattribs={"color": 1})
    for i, it in enumerate(issues, 1):
        x, y = it.at
        msp.add_circle(center=(x, y), radius=2.5, dxfattribs={"layer": "ERRORS"})
        txt = f"[{i}] {it.code}: {it.message}\nlayer={it.layer} handles={','.join(it.handles)}"
        msp.add_mtext(txt, dxfattribs={"layer": "ERRORS"}).set_location((x+5, y+5))
    doc.saveas(out_path)

# ---------------------- 主流程 ----------------------
def main():
    ap = argparse.ArgumentParser(description="DXF 清理檢查器（輸出只含錯誤標記的 DXF）")

    # 📌 有給參數就用參數，沒給就用預設檔案
    ap.add_argument(
        "dxf",
        nargs="?",
        default=r"C:\Users\u0913\OneDrive\桌面\Paper\202503\柱筋結構圖.dxf",
        help="DXF 檔案路徑（不給則使用預設檔案）"
    )

    ap.add_argument("--out", default="errors_only.dxf")
    ap.add_argument("--assume-unit", default=None, help="當 $INSUNITS 未設時假設單位（mm/cm/m/in/ft）")

    ap.add_argument("--min-seg-mm", type=float, default=1.0, help="線段太短的閾值(mm)")
    ap.add_argument("--tol-mm", type=float, default=0.5, help="幾何容差(mm)")
    ap.add_argument("--angle-tol-deg", type=float, default=0.5, help="共線/平行角度容差(度)")

    ap.add_argument("--check-dup-lines", action="store_true")
    ap.add_argument("--check-overlap-lines", action="store_true")
    ap.add_argument("--check-open-polylines", action="store_true")
    ap.add_argument("--check-outside-boundary", action="store_true")
    ap.add_argument("--boundary-layer", default="BOUNDARY")
    ap.add_argument("--check-wrong-layer", action="store_true")
    ap.add_argument("--layer-spec", default=None, help='JSON，如：{"WALL":["L-WALL","A-WALL"],"DIM":["A-DIMS"]}')

    args = ap.parse_args()
    src = Path(args.dxf)
    if not src.exists():
        raise SystemExit(f"找不到檔案：{src}")

    doc = ezdxf.readfile(src)
    msp = doc.modelspace()

    # ⚠️ 這裡繼續呼叫各種 rule_xxx(...) 檢查
    issues: List[Issue] = []
    issues += rule_short_segments(msp, args.min_seg_mm)
    # （依需要加上其他規則呼叫）

    outp = Path(args.out)
    export_errors_dxf(issues, outp, doc)

    print("\n=== 檢查結果 ===")
    if not issues:
        print("未發現問題；已輸出空的錯誤標記檔：", outp)
        return
    for i, it in enumerate(issues, 1):
        x, y = it.at
        print(f"[{i}] {it.code:<14} @({x:.2f},{y:.2f})  layer={it.layer}  handles={','.join(it.handles)}  | {it.message}")
    print(f"\n共 {len(issues)} 個問題；已輸出：{outp}")

if __name__ == "__main__":
    main()
