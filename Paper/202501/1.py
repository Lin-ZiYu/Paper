import ezdxf
import pandas as pd

try:
    doc = ezdxf.readfile(r'C:\Users\u0913\OneDrive\桌面\202501\0824柱筋結構圖.dxf')
    msp = doc.modelspace()
except IOError:
    print("檔案不存在或無法讀取。")
    exit()
except ezdxf.DXFError:
    print("無效的DXF檔案。")
    exit()

# 假設鋼筋資料是以文字儲存
columns = ['柱子編號', '主筋號數', '主筋數量', '箍筋號數', '箍筋數量', '繫筋號數', '繫筋數量']
data = []

for text in msp.query('TEXT MTEXT'):
    # 這裡需要根據你的圖檔內容解析文字
    # 例如：text.plain_text 可能是 "C1 主筋#6x8 箍筋#3@100 繫筋#4x2"
    # 你需要用正則表達式或字串分割來解析
    pass

df = pd.DataFrame(data, columns=columns)
df.to_excel('柱子鋼筋表.xlsx', index=False)

# 假設您已經解析出這些資料
column_data = [
    {'柱子編號': 'C1', '主筋號數': 'D16', '主筋數量': 8, '箍筋號數': 'D10', '箍筋數量': '200mm'},
    {'柱子編號': 'C2', '主筋號數': 'D22', '主筋數量': 12, '箍筋號數': 'D13', '箍筋數量': '150mm'},
    # ... 更多柱子資料
]

df = pd.DataFrame(column_data)
df.to_excel("柱子鋼筋表.xlsx", index=False)