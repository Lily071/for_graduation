import pandas as pd
import matplotlib.pyplot as plt

## 追加1
# グラフの描画先の準備
# plt（最初にインポートしたmatplotlib.pyplotというモジュール）にある
# figure()メソッドを、figに代入しておきます。
fig = plt.figure()
## 追加1

def graph(diffx, diffy):
    
    plt.title('matplotlib graph')
    plt.xlabel('Time (sec)')
    plt.ylabel('Diff')
    print(diffx, diffy)
    #plt.plot(x=diffx, y=diffy)
    plt.plot(diffx, diffy) 

## 追加2
# ファイルに保存
# fig（準備した描写先であるFigure(432x288)のようなデータが準備されています）に
# img.pngという画像を保存しています。
fig.savefig("img.png")
## 追加2

plt.show()
