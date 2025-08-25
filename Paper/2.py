import os
import ezdxf
import pandas as pd

doc = ezdxf.readfile(r'C:\Users\u0913\OneDrive\桌面\Paper\0824柱筋結構圖.dxf')
msp = doc.modelspace()

# 假設鋼筋柱子的線圖層名稱為 "COLUMN"
target_layer = "COLUMN"
data = []

for line in msp.query('LINE'):
    if line.dxf.layer == target_layer:
        start = line.dxf.start
        end = line.dxf.end
        length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
        data.append({
            '圖層': line.dxf.layer,
            '起點': start,
            '終點': end,
            '長度': length
        })

df = pd.DataFrame(data)
df.to_excel('柱子線段尺寸.xlsx', index=False)