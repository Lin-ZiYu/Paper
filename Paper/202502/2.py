import ezdxf
import pandas as pd
import math

#載入 DXF 檔案
doc = ezdxf.readfile(r'C:\Users\u0913\OneDrive\桌面\Paper\202502\柱筋結構圖.dxf')
msp = doc.modelspace()

# 儲存柱資訊
columns_info = []

# 幾何輔助函數
def get_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = round(max(xs) - min(xs), 1)
    height = round(max(ys) - min(ys), 1)
    center = (round(sum(xs) / len(xs), 1), round(sum(ys) / len(ys), 1))
    return width, height, center

# 擷取疑似柱子
for e in msp:
    if e.dxftype() in {"LWPOLYLINE", "POLYLINE"}:
        try:
            points = e.get_points()
        except:
            points = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices()]
        if len(points) >= 4:
            width, height, center = get_bbox(points)
            # 根據實際單位調整篩選條件
            if 30 <= width <= 200 and 30 <= height <= 200:
                columns_info.append({
                    "寬度(cm)": width,
                    "高度(cm)": height,
                    "中心點": center,
                    "柱名稱": "",
                    "主筋": "",
                    "箍筋": ""
                })

# 擷取所有文字物件
text_entities = []
for e in msp:
    if e.dxftype() == "TEXT":
        text = e.dxf.text
        pos = (e.dxf.insert.x, e.dxf.insert.y)
        text_entities.append((text, pos))
    elif e.dxftype() == "MTEXT":
        text = e.text
        pos = (e.dxf.insert.x, e.dxf.insert.y)
        text_entities.append((text, pos))

def find_nearby_text(center, radius=100):
    nearby = []
    for text, pos in text_entities:
        dist = math.hypot(pos[0] - center[0], pos[1] - center[1])
        if dist < radius:
            nearby.append(text)
    return nearby

for col in columns_info:
    texts = find_nearby_text(col["中心點"])
    for t in texts:
        if "#" in t or "D" in t:  # 主筋樣式
            col["主筋"] = t
        elif "@" in t or "箍" in t:  # 箍筋樣式
            col["箍筋"] = t
        elif len(t) <= 4:  # 短字串當作柱名稱
            col["柱名稱"] = t

df = pd.DataFrame(columns_info)
df.to_excel("柱子資訊總表.xlsx", index=False)
print("已輸出柱子資訊至：柱子資訊總表.xlsx")
