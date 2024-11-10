"""
下の次数は、selefsarima_order.pyで出した。数値を入れている。
次数は以下
p_values = [1, 2] 
d_values = [1] 
q_values = [1, 2] 
P_values = [1] 
D_values = [1] 
Q_values = [1]
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# Excelファイルのパス
file_path = './ice_cream_sales_data_split.xlsx'

# Excelファイルから全てのデータを読み込み、A列のindexをDateという名前へ。加えて、parse_datesより日付型へ
ice_cream_sales = pd.read_excel(file_path, index_col='Date', parse_dates=True)

# 売上データの取得
y = ice_cream_sales['Ice_Cream_Sales']


# 以下が、selfsarima_orderで出した候補となるパラメータの範囲を設定
p_values = [1, 2]
d_values = [1]
q_values = [1, 2]
P_values = [1]
D_values = [1]
Q_values = [1]
s = 12  # 季節性の周期

# AICを記録するためのリスト
results = []

# すべてのパラメータの組み合わせでSARIMAモデルを構築し、AICを計算
for p in p_values:
    for d in d_values:
        for q in q_values:
            for P in P_values:
                for D in D_values:
                    for Q in Q_values:
                        try:
                            # SARIMAモデルの設定と適合
                            model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s))
                            results_obj = model.fit(disp=False)
                            
                            # パラメータとAICを記録
                            results.append(((p, d, q), (P, D, Q, s), results_obj.aic))
                        except Exception as e:
                            # エラーが発生した場合はスキップ
                            print(f"Error for parameters ({p}, {d}, {q}) x ({P}, {D}, {Q}, {s}): {e}")

# 最小のAICを持つモデルの選定
best_result = min(results, key=lambda x: x[2])

# 最適なパラメータとAICの表示
print("最適なパラメータ (p, d, q):", best_result[0])
print("季節パラメータ (P, D, Q, s):", best_result[1])
print("最小のAIC:", best_result[2])


"""

計算結果
最適なパラメータ (p, d, q): (1, 1, 2)
季節パラメータ (P, D, Q, s): (1, 1, 1, 12)
最小のAIC: 981.3547889666379

"""
