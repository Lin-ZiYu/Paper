# -*- coding: utf-8 -*-

import argparse
import json
from dataclasses import dataclass
from math import isclose, atan2, degrees
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import ezdxf
from ezdxf.entities import Line, LWPolyline, Polyline, Circle, Arc, Text, MText, Dimension
from ezdxf.lldxf.const import DXFValueError

from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

# ---------------------- 載入DXF檔案 ---------------------- 
doc = ezdxf.readfile(r'C:\Users\u0913\OneDrive\桌面\Paper\202503\柱筋結構圖.dxf')
msp = doc.modelspace()


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
    code: str                 # 規則代碼，如 SHORT_SEG, DUP_LINE...
    message: str              # 說明
    at: Tuple[float, float]   # 標註點 (x,y) in model units
    handles: Tuple[str, ...]  # 相關實體 handle
    layer: str                # 來源圖層

# ---------------------- 工具函式 ----------------------
def line_to_linestring(e: Line) -> LineString:
    p1 = (e.dxf.start.x, e.dxf.start.y)
    p2 = (e.dxf.end.x, e.dxf.end.y)
    return LineString([p1, p2])

def pline_to_linestrings(e: LWPolyline) -> List[LineString]:
    pts = [(v[0], v[1]) for v in e.get_points()]  # (x, y, [start_width, end_width, bulge])
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
    # 從指定圖層收集 LINE / LWPOLYLINE 作為邊界（取其 unary_union 的多邊形包絡）
    lines = []
    for e in msp.query(f"LINE[layer=='{layer_name}']"):
        lines.append(line_to_linestring(e))
    for e in msp.query(f"LWPOLYLINE[layer=='{layer_name}']"):
        for seg in pline_to_linestrings(e):
            lines.append(seg)
    if not lines:
        return None
    merged = unary_union(lines)
    # 取外包框（convex hull 雖然簡單，但會吃掉凹處；改取 buffer->polygonize 的方式較繁）
    # 這裡保守用 bounds 作矩形（實務若要精確外框，可自定規則）
    minx, miny, maxx, maxy = merged.bounds
    # 容差擴張一點
    return Polygon([(minx-tol, miny-tol), (maxx+tol, miny-tol),
                    (maxx+tol, maxy+tol), (minx-tol, maxy+tol)])

def classify_entity(e) -> Optional[str]:
    """簡單的類別分類，供 --layer-spec 規則使用；你可依你的圖規擴增。"""
    if isinstance(e, (Line, LWPolyline, Polyline)):
        # 這裡僅示意：實務上可依 layer 名稱關鍵字判斷，如含 WALL/DOOR...
        ln = (e.dxf.layer or "").upper()
        if "WALL" in ln:
            return "WALL"
        if "DOOR" in ln:
            return "DOOR"
        return "LINEWORK"
    if isinstance(e, (Text, MText)):
        return "TEXT"
    if isinstance(e, Dimension):
        return "DIM"
    if isinstance(e, (Circle, Arc)):
        return "ARC"
    return None

# ---------------------- 規則實作 ----------------------
def rule_short_segments(msp, min_seg_len: float) -> List[Issue]:
    issues: List[Issue] = []
    # LINE
    for e in msp.query("LINE"):
        ls = line_to_linestring(e)
        if ls.length < min_seg_len:
            mid = ls.interpolate(0.5, normalized=True)
            issues.append(Issue("SHORT_SEG",
                                f"線段過短 {ls.length:.3f}",
                                (mid.x, mid.y),
                                (e.dxf.handle,), e.dxf.layer))
    # LWPOLYLINE segments
    for e in msp.query("LWPOLYLINE"):
        for seg in pline_to_linestrings(e):
            if seg.length < min_seg_len:
                mid = seg.interpolate(0.5, normalized=True)
                issues.append(Issue("SHORT_SEG",
                                    f"PLINE子段過短 {seg.length:.3f}",
                                    (mid.x, mid.y),
                                    (e.dxf.handle,), e.dxf.layer))
    return issues

def rule_duplicate_lines(msp, tol: float, ang_tol: float) -> List[Issue]:
    issues: List[Issue] = []
    # 用端點比對：相同兩端（允許交換），且距離<tol、方向近似
    seen: Dict[Tuple[Tuple[int,int],Tuple[int,int]], str] = {}
    def qkey(p):  # 量化以 tol 網格降低浮點誤差
        return (int(round(p[0]/tol)), int(round(p[1]/tol)))

    for e in msp.query("LINE"):
        p1 = (e.dxf.start.x, e.dxf.start.y)
        p2 = (e.dxf.end.x, e.dxf.end.y)
        a = angle_of(p1, p2)
        k1, k2 = qkey(p1), qkey(p2)
        key = tuple(sorted([k1, k2]))
        if key in seen:
            # 再做角度確認
            other_handle = seen[key]
            a_ok = True  # 對應到相同兩端點通常就可判定為重複
            if a_ok:
                midx, midy = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
                issues.append(Issue("DUP_LINE",
                                    "重複線段（相同兩端點）",
                                    (midx, midy),
                                    (seen[key], e.dxf.handle),
                                    e.dxf.layer))
        else:
            seen[key] = e.dxf.handle
    return issues

def rule_overlapping_collinear_lines(msp, tol: float, ang_tol: float, min_overlap: float=0.1) -> List[Issue]:
    """共線且投影重疊（不要求端點完全相同）。O(n^2) 實作，適合中小圖面。"""
    issues: List[Issue] = []
    lines = [e for e in msp.query("LINE")]
    n = len(lines)
    for i in range(n):
        li = lines[i]
        p1i = (li.dxf.start.x, li.dxf.start.y)
        p2i = (li.dxf.end.x, li.dxf.end.y)
        ai = angle_of(p1i, p2i)
        lsi = line_to_linestring(li)
        if lsi.length < tol:  # 太短由別的規則處理
            continue
        for j in range(i+1, n):
            lj = lines[j]
            p1j = (lj.dxf.start.x, lj.dxf.start.y)
            p2j = (lj.dxf.end.x, lj.dxf.end.y)
            aj = angle_of(p1j, p2j)
            if not nearly_parallel(ai, aj, ang_tol):
                continue
            lsj = line_to_linestring(lj)
            # 距離：兩線最短距離小於 tol 視為共線（近似）
            if lsi.distance(lsj) > tol:
                continue
            inter = lsi.intersection(lsj)
            overlap_len = inter.length if hasattr(inter, "length") else 0.0
            if overlap_len >= max(min_overlap, tol):
                m = inter.interpolate(0.5, normalized=True) if hasattr(inter, "interpolate") else lsi.interpolate(0.5, normalized=True)
                issues.append(Issue("OVERLAP_LINE",
                                    f"共線重疊 ~ {overlap_len:.3f}",
                                    (m.x, m.y),
                                    (li.dxf.handle, lj.dxf.handle),
                                    li.dxf.layer))
    return issues

def rule_open_polylines(msp, tol: float) -> List[Issue]:
    issues: List[Issue] = []
    for e in msp.query("LWPOLYLINE"):
        pts = [(v[0], v[1]) for v in e.get_points()]
        if len(pts) < 2:
            continue
        if e.closed:
            # 有時標記為閉合，但端點不吻合（數值誤差）；這裡可略過
            continue
        p1, p2 = pts[0], pts[-1]
        dx, dy = p1[0]-p2[0], p1[1]-p2[1]
        gap = (dx*dx + dy*dy) ** 0.5
        if gap > tol:
            midx, midy = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            issues.append(Issue("OPEN_PLINE",
                                f"PLINE 未封口，端點距離 {gap:.3f}",
                                (midx, midy),
                                (e.dxf.handle,),
                                e.dxf.layer))
    return issues

def rule_outside_boundary(msp, boundary_layer: str, tol: float) -> List[Issue]:
    issues: List[Issue] = []
    poly = bbox_polygon_from_layer(msp, boundary_layer, tol)
    if poly is None:
        return issues
    # 檢查 LINE / LWPOLYLINE / CIRCLE / ARC 的外逸
    # 判定：幾何的外包框/中心點落在邊界外
    def out_pt(pt):
        return not poly.buffer(0).contains(Point(pt))
    for e in msp.query("LINE"):
        ls = line_to_linestring(e)
        if not poly.buffer(0).contains(ls):
            c = ls.interpolate(0.5, normalized=True)
            issues.append(Issue("OUTSIDE",
                                "線段部分超出邊界",
                                (c.x, c.y),
                                (e.dxf.handle,),
                                e.dxf.layer))
    for e in msp.query("LWPOLYLINE"):
        segs = pline_to_linestrings(e)
        for seg in segs:
            if not poly.buffer(0).contains(seg):
                c = seg.interpolate(0.5, normalized=True)
                issues.append(Issue("OUTSIDE",
                                    "PLINE 子段超出邊界",
                                    (c.x, c.y),
                                    (e.dxf.handle,),
                                    e.dxf.layer))
                break
    for e in msp.query("CIRCLE"):
        center = (e.dxf.center.x, e.dxf.center.y)
        if out_pt(center):
            issues.append(Issue("OUTSIDE",
                                "圓心超出邊界",
                                center,
                                (e.dxf.handle,),
                                e.dxf.layer))
    for e in msp.query("ARC"):
        center = (e.dxf.center.x, e.dxf.center.y)
        if out_pt(center):
            issues.append(Issue("OUTSIDE",
                                "圓弧中心超出邊界",
                                center,
                                (e.dxf.handle,),
                                e.dxf.layer))
    return issues

def rule_wrong_layer(msp, layer_spec: Dict[str, List[str]]) -> List[Issue]:
    issues: List[Issue] = []
    allow_map = {k.upper(): [x.upper() for x in v] for k, v in layer_spec.items()}
    for e in msp:
        cls = classify_entity(e)
        if not cls:
            continue
        allows = allow_map.get(cls.upper())
        if not allows:
            continue
        ln = (getattr(e.dxf, "layer", "") or "").upper()
        if ln not in allows:
            # 標在幾何的中點/中心
            at = (0.0, 0.0)
            if isinstance(e, Line):
                ls = line_to_linestring(e); c = ls.interpolate(0.5, normalized=True); at=(c.x, c.y)
            elif isinstance(e, LWPolyline):
                segs = pline_to_linestrings(e)
                if segs:
                    c = segs[0].interpolate(0.5, normalized=True); at=(c.x, c.y)
            elif isinstance(e, (Circle, Arc)):
                at = (e.dxf.center.x, e.dxf.center.y)
            elif isinstance(e, (Text, MText, Dimension)):
                at = (e.dxf.insert.x, e.dxf.insert.y) if hasattr(e.dxf, "insert") else (0.0, 0.0)
            issues.append(Issue("WRONG_LAYER",
                                f"{cls} 不在允許圖層：{allows}（實際：{ln}）",
                                at,
                                (e.dxf.handle,),
                                getattr(e.dxf, "layer", "")))
    return issues

# ---------------------- DXF 輸出（錯誤標記） ----------------------
def export_errors_dxf(issues: List[Issue], out_path: Path, base_doc):
    # 做一份新 DXF（保留相同版本），在 modelspace 畫小圓 + MTEXT
    doc = ezdxf.new(dxfversion=base_doc.dxfversion)
    msp = doc.modelspace()
    # 準備圖層
    if "ERRORS" not in doc.layers:
        doc.layers.new("ERRORS", dxfattribs={"color": 1})  # 紅色 ACI=1
    for i, it in enumerate(issues, 1):
        x, y = it.at
        # 小圓
        msp.add_circle(center=(x, y), radius=2.5, dxfattribs={"layer": "ERRORS"})
        # 說明
        txt = f"[{i}] {it.code}: {it.message}\nlayer={it.layer} handles={','.join(it.handles)}"
        msp.add_mtext(txt, dxfattribs={"layer": "ERRORS"}).set_location((x+5, y+5))
    doc.saveas(out_path)

# ---------------------- 主流程 ----------------------
def main():
    ap = argparse.ArgumentParser(description="DXF 清理檢查器（輸出只含錯誤標記的 DXF）")
    ap.add_argument("dxf")
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
    unit_scale = get_unit_scale_mm(doc, args.assume_unit)  # 目前我們用 mm 判定，若要等比縮放可自行調整
    msp = doc.modelspace()

    # 規則運行（門檻以 mm 為基礎；DXF 通常為圖面單位，這裡假設圖面=mm）
    issues: List[Issue] = []

    # 1) 短線段
    issues += rule_short_segments(msp, args.min_seg_mm)

    # 2) 重複線
    if args.check_dup_lines:
        issues += rule_duplicate_lines(msp, tol=args.tol_mm, ang_tol=args.angle_tol_deg)

    # 3) 共線重疊
    if args.check_overlap_lines:
        issues += rule_overlapping_collinear_lines(msp, tol=args.tol_mm, ang_tol=args.angle_tol_deg)

    # 4) 未封口 PLINE
    if args.check_open_polylines:
        issues += rule_open_polylines(msp, tol=args.tol_mm)

    # 5) 超出邊界
    if args.check_outside_boundary:
        issues += rule_outside_boundary(msp, boundary_layer=args.boundary_layer, tol=args.tol_mm)

    # 6) 圖層規範
    if args.check_wrong_layer:
        spec = {}
        if args.layer_spec:
            try:
                spec = json.loads(args.layer_spec)
            except Exception as e:
                print(f"layer-spec 解析失敗：{e}；略過此規則")
        if spec:
            issues += rule_wrong_layer(msp, layer_spec=spec)

    # 輸出錯誤 DXF
    outp = Path(args.out)
    if issues:
        export_errors_dxf(issues, outp, doc)
    else:
        # 仍產出空檔，方便你打開確認
        export_errors_dxf([], outp, doc)

    # 終端摘要
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
