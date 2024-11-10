import pandas as pd
import matplotlib.pyplot as plt

# Excelファイルのパス
file_path = './ice_cream_sales_data_split.xlsx'

# Excelファイルからデータを読み込み、A列のインデックスを 'Date' として日付型に変換
ice_cream_sales = pd.read_excel(file_path, index_col='Date', parse_dates=True)

# 売上データの取得
y = ice_cream_sales['Ice_Cream_Sales']

# グラフ作成：年ごとの月次データをm=12ごとに可視化
plt.figure(figsize=(12, 8))

# 年ごとにデータをグループ化し、各年のデータを月ごとにプロット
for year, data in y.groupby(y.index.year):
    # 各年の月ごとの売上データをプロット
    # data が1年分に絞られているため、売上データが1月から12月の順
    plt.plot(data.index.month, data.values, marker='o', label=str(year))

# グラフの設定
plt.title("Yearly Ice Cream Sales (m=12)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(range(1, 13))  # x軸に1～12月のラベルを表示
plt.legend(title="Year")  # 凡例に年を表示
plt.grid(True)  # グリッド表示
plt.show()
