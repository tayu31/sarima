import numpy as np
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Excelファイルのパス
file_path = './ice_cream_sales_data_split.xlsx'

# Excelファイルから全てのデータを読み込み、A列のindexをDateという名前へ。加えて、parse_datesより日付型へ
ice_cream_sales = pd.read_excel(file_path, index_col='Date', parse_dates=True)

# 売上データの取得
y = ice_cream_sales['Ice_Cream_Sales']

# auto_arimaを用いて最適なSARIMAモデルのパラメータを自動で探索
auto_model = auto_arima(
    y,
    seasonal=True, m=12,  # 季節性周期を12に設定
    start_p=1, max_p=2,
    start_q=1, max_q=2,
    d=None, D=1,  # 差分は手動で決めた値に固定
    start_P=1, max_P=1,
    start_Q=1, max_Q=1,
    trace=True,  # 各試行の結果を表示
    error_action="ignore",  # エラーを無視して次に進む
    suppress_warnings=True, 
    stepwise=True  # ステップワイズ探索
)

# 自動で決定したモデルのパラメータとAICを表示
print("最適なモデルの要約:")
print(auto_model.summary())
print("\n選定されたパラメータ:")
print("非季節成分 (p, d, q):", auto_model.order)
print("季節成分 (P, D, Q, s):", auto_model.seasonal_order)
print("AIC:", auto_model.aic())


"""

Performing stepwise search to minimize aic
 ARIMA(1,1,1)(1,1,1)[12]             : AIC=984.262, Time=0.51 sec
 ARIMA(0,1,0)(0,1,0)[12]             : AIC=1099.852, Time=0.02 sec
 ARIMA(1,1,0)(1,1,0)[12]             : AIC=1034.190, Time=0.16 sec
 ARIMA(0,1,1)(0,1,1)[12]             : AIC=981.228, Time=0.29 sec
 ARIMA(0,1,1)(0,1,0)[12]             : AIC=1020.597, Time=0.08 sec
 ARIMA(0,1,1)(1,1,1)[12]             : AIC=982.891, Time=0.71 sec
 ARIMA(0,1,1)(1,1,0)[12]             : AIC=996.401, Time=0.35 sec
 ARIMA(0,1,0)(0,1,1)[12]             : AIC=inf, Time=0.18 sec
 ARIMA(1,1,1)(0,1,1)[12]             : AIC=982.501, Time=0.90 sec
 ARIMA(0,1,2)(0,1,1)[12]             : AIC=982.456, Time=0.42 sec
 ARIMA(1,1,0)(0,1,1)[12]             : AIC=1016.761, Time=0.27 sec
 ARIMA(1,1,2)(0,1,1)[12]             : AIC=979.704, Time=0.88 sec
 ARIMA(1,1,2)(0,1,0)[12]             : AIC=inf, Time=0.24 sec
 ARIMA(1,1,2)(1,1,1)[12]             : AIC=981.355, Time=1.26 sec
 ARIMA(1,1,2)(1,1,0)[12]             : AIC=996.528, Time=0.73 sec
 ARIMA(2,1,2)(0,1,1)[12]             : AIC=986.035, Time=0.83 sec
 ARIMA(2,1,1)(0,1,1)[12]             : AIC=984.423, Time=1.08 sec
 ARIMA(1,1,2)(0,1,1)[12] intercept   : AIC=inf, Time=0.57 sec

Best model:  ARIMA(1,1,2)(0,1,1)[12]
Total fit time: 9.546 seconds
最適なモデルの要約:
                                      SARIMAX Results
============================================================================================
Dep. Variable:                                    y   No. Observations:                  120
Model:             SARIMAX(1, 1, 2)x(0, 1, [1], 12)   Log Likelihood                -484.852
Date:                              Sat, 02 Nov 2024   AIC                            979.704
Time:                                      05:16:23   BIC                            993.068
Sample:                                  01-31-2013   HQIC                           985.122
                                       - 12-31-2022
Covariance Type:                                opg
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.9175      0.077     11.909      0.000       0.767       1.069
ma.L1         -1.9297      0.057    -33.924      0.000      -2.041      -1.818
ma.L2          0.9442      0.054     17.339      0.000       0.837       1.051
ma.S.L12      -0.8412      0.172     -4.884      0.000      -1.179      -0.504
sigma2       433.5136     83.586      5.186      0.000     269.688     597.339
===================================================================================
Ljung-Box (L1) (Q):                   0.14   Jarque-Bera (JB):                 1.05
Prob(Q):                              0.71   Prob(JB):                         0.59
Heteroskedasticity (H):               0.83   Skew:                            -0.15
Prob(H) (two-sided):                  0.58   Kurtosis:                         2.61
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

選定されたパラメータ:
非季節成分 (p, d, q): (1, 1, 2)
季節成分 (P, D, Q, s): (0, 1, 1, 12)
AIC: 979.7042664587451

"""