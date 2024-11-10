# 初めに実行するコード
# これにより、次数の決定できるもの（dとD）と、候補（p,d,P,Q）を絞る

 
# 必要なライブラリのインポート 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller, acf, pacf 
import warnings 
 
import pandas as pd
import warnings

warnings.filterwarnings("ignore") 

# Excelファイルのパス
file_path = './ice_cream_sales_data_split.xlsx'

# Excelファイルから全てのデータを読み込み、A列のindexをDateという名前へ。加えて、parse_datesより日付型へ
ice_cream_sales = pd.read_excel(file_path, index_col='Date', parse_dates=True)

# 売上データの取得
y = ice_cream_sales['Ice_Cream_Sales']
 
 
# 1. ADF検定で非季節成分のdを決定 
def adf_test(series, description=""): 
    result = adfuller(series) 
    print(f"{description}") 
    print('ADF Statistic:', result[0]) 
    print('p-value:', result[1]) 
    print('Critical Values:', result[4]) 
    if result[1] < 0.05: 
        print("=> データは定常性が確認されました。\n") 
        return True  # 定常データ 
    else: 
        print("=> データは非定常です。\n") 
        return False  # 非定常データ 
 
# ADF検定実施 
print("== 非季節成分のADF検定結果 ==") 
is_stationary = adf_test(y, description="非定常データ（元データ）のADF検定結果") 
if not is_stationary: 
    # 非定常の場合は差分を取る 
    y_diff = y.diff().dropna() 
    d = 1 
    # 差分後のADF検定結果を表示 
    adf_test(y_diff, description="定常データ（1回差分後）のADF検定結果") 
else: 
    y_diff = y 
    d = 0 
 
# 2. 非季節成分のPACFとACFからp, qの候補を確認 
# 非定常データのPACFとACFを表示 
non_seasonal_pacf_vals = pacf(y_diff, nlags=20) 
non_seasonal_acf_vals = acf(y_diff, nlags=20) 
 
# 非季節成分の PACFとACF の数値を出力 
print("\n== 非季節成分のPACFの結果 ==") 
for i, val in enumerate(non_seasonal_pacf_vals[:5]):  # 前の数値としてラグ5まで出力 
    print(f"Lag {i}: {val}") 
 
print("\n== 非季節成分のACFの結果 ==") 
for i, val in enumerate(non_seasonal_acf_vals[:5]):  # 前の数値としてラグ5まで出力 
    print(f"Lag {i}: {val}") 
 
# 3. 季節性を考慮したSARIMAのためのD, P, Qを求める（m=12として） 
# 季節差分 D の決定 
seasonal_diff = y.diff(12).dropna() 
print("\n== 季節成分のADF検定結果 (Dの決定) ==") 
is_seasonal_stationary = adf_test(seasonal_diff, description="非定常データ（季節差分前）のADF検定結果") 
if not is_seasonal_stationary: 
    D = 1 
    # 1回の季節差分を行い、定常性確認 
    seasonal_diff = seasonal_diff.diff().dropna() 
    adf_test(seasonal_diff, description="定常データ（1回季節差分後）のADF検定結果") 
else: 
    D = 0 
 
# 季節成分の P, Q を確認するための季節PACFと季節ACF 
seasonal_pacf_vals = pacf(seasonal_diff, nlags=20) 
seasonal_acf_vals = acf(seasonal_diff, nlags=20) 
 
# 季節成分の PACFとACF の数値を出力 
print("\n== 季節成分のPACFの結果 ==") 
for i, val in enumerate(seasonal_pacf_vals[:5]):  # 前の数値としてラグ5まで出力 
    print(f"Lag {i}: {val}") 
 
print("\n== 季節成分のACFの結果 ==") 
for i, val in enumerate(seasonal_acf_vals[:5]):  # 前の数値としてラグ5まで出力 
    print(f"Lag {i}: {val}") 
 
# 最終的に選定された d と D の値 
print(f"\n選定された差分回数 d = {d}, 季節差分回数 D = {D}")



"""

== 非季節成分のADF検定結果 ==
非定常データ（元データ）のADF検定結果
ADF Statistic: -1.184906286925508
p-value: 0.6800547127047087
Critical Values: {'1%': -3.492995948509562, '5%': -2.888954648057252, '10%': -2.58139291903223}
=> データは非定常です。

定常データ（1回差分後）のADF検定結果
ADF Statistic: -9.297346027244764
p-value: 1.1367166859156197e-15
Critical Values: {'1%': -3.492995948509562, '5%': -2.888954648057252, '10%': -2.58139291903223}
=> データは定常性が確認されました。


== 非季節成分のPACFの結果 ==
Lag 0: 1.0
Lag 1: 0.3182156395407116
Lag 2: 0.2534211037941782
Lag 3: -0.18089662977762164
Lag 4: -0.5080568286914531

== 非季節成分のACFの結果 ==
Lag 0: 1.0
Lag 1: 0.31554155853616767
Lag 2: 0.3234908161371751
Lag 3: 0.006503162221427122
Lag 4: -0.358970161743379

== 季節成分のADF検定結果 (Dの決定) ==
非定常データ（季節差分前）のADF検定結果
ADF Statistic: -2.5916736584303246
p-value: 0.09473039309785719
Critical Values: {'1%': -3.5019123847798657, '5%': -2.892815255482889, '10%': -2.583453861475781}
=> データは非定常です。

定常データ（1回季節差分後）のADF検定結果
ADF Statistic: -4.653947903620131
p-value: 0.00010261402984550784
Critical Values: {'1%': -3.502704609582561, '5%': -2.8931578098779522, '10%': -2.583636712914788}
=> データは定常性が確認されました。


== 季節成分のPACFの結果 ==
Lag 0: 1.0
Lag 1: -0.5462365431855574
Lag 2: -0.41449484606963916
Lag 3: -0.09839249898822754
Lag 4: -0.3708734026413674

== 季節成分のACFの結果 ==
Lag 0: 1.0
Lag 1: -0.5411315287632625
Lag 2: 0.007412950857670747
Lag 3: 0.1588200536392355
Lag 4: -0.28522081870731014

選定された差分回数 d = 1, 季節差分回数 D = 1
"""