import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# 1. Excelファイルからデータを読み込み
file_path = './ice_cream_sales_data_split.xlsx'  # Excelファイルのパス
ice_cream_sales = pd.read_excel(file_path, index_col='Date', parse_dates=True)

# 2. 売上データの取得
y = ice_cream_sales['Ice_Cream_Sales']

# 3. 最適なSARIMAパラメータでモデルをフィッティング（例として (1, 1, 2) x (1, 1, 1, 12)）
p, d, q = 1, 1, 2
P, D, Q, s = 1, 1, 1, 12

# SARIMAモデルをデータに適合
model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s))
model_fit = model.fit(disp=False)

# 4. 予測（例えば、12か月先まで）
forecast_steps = 12
forecast = model_fit.get_forecast(steps=forecast_steps)

# 予測値と信頼区間の取得
predicted_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# 5. 予測結果のみのグラフ（1枚目）
plt.figure(figsize=(10, 6))
plt.plot(predicted_values.index, predicted_values, color='orange', label="Forecast")
plt.fill_between(predicted_values.index, 
                 confidence_intervals.iloc[:, 0], 
                 confidence_intervals.iloc[:, 1], 
                 color='pink', alpha=0.3, label="Confidence Interval")
plt.title("Forecasted Ice Cream Sales (Prediction Only)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

# 6. 全体のグラフ（2枚目）
x = y.index
plt.figure(figsize=(12, 6))
plt.plot(x,y, label="Actual Data", color='blue')
plt.plot(predicted_values.index, predicted_values, color='orange', label="Forecast")
plt.fill_between(predicted_values.index, 
                 confidence_intervals.iloc[:, 0], 
                 confidence_intervals.iloc[:, 1], 
                 color='pink', alpha=0.3, label="Confidence Interval")

plt.title("Ice Cream Sales with Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()
