import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# アップロードされたExcelファイルから分割データを読み込み
split_file_path = 'ice_cream_sales_data_split.xlsx'  # ローカル環境でのファイルパスに置き換えてください
train_data = pd.read_excel(split_file_path, sheet_name='Train_Data', index_col='Date', parse_dates=True)
validation_data = pd.read_excel(split_file_path, sheet_name='Validation_Data', index_col='Date', parse_dates=True)
test_data = pd.read_excel(split_file_path, sheet_name='Test_Data', index_col='Date', parse_dates=True)

# 1. 訓練データでSARIMAモデルのパラメータ探索
train_data_values = train_data['Ice_Cream_Sales']
auto_model = auto_arima(
    train_data_values,
    seasonal=True, m=12,  # 季節性周期を12に設定
    start_p=1, max_p=2,
    start_q=1, max_q=2,
    d=None, D=1,  # 差分の次数
    start_P=1, max_P=1,
    start_Q=1, max_Q=1,
    trace=True,  # 各試行の結果を表示
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)

# 最適なSARIMAモデルのパラメータとAICを表示
print("最適なモデルの要約:")
print(auto_model.summary())
print("\n選定されたパラメータ:")
print("非季節成分 (p, d, q):", auto_model.order)
print("季節成分 (P, D, Q, s):", auto_model.seasonal_order)
print("AIC:", auto_model.aic())

# 2. 検証データで予測を行い、予測精度を確認
# モデルを訓練データ全体に適合
p, d, q = auto_model.order
P, D, Q, s = auto_model.seasonal_order

# SARIMAモデルをデータに適合
model = SARIMAX(train_data_values, order=(p, d, q), seasonal_order=(P, D, Q, s))
model_fit = model.fit(disp=False)

# 検証データに対して予測
forecast_steps_validation = len(validation_data)
forecast_validation = model_fit.forecast(steps=forecast_steps_validation)

# 検証データ期間の売上平均値を計算
validation_mean = validation_data['Ice_Cream_Sales'].mean()

# 予測精度の計算
rmse_validation = np.sqrt(mean_squared_error(validation_data['Ice_Cream_Sales'], forecast_validation))
mae_validation = mean_absolute_error(validation_data['Ice_Cream_Sales'], forecast_validation)

# 平均に対するRMSEとMAEの割合を計算
rmse_validation_percentage = (rmse_validation / validation_mean) * 100
mae_validation_percentage = (mae_validation / validation_mean) * 100

print("\n検証データでの予測精度:")
print("RMSE:", rmse_validation)
print("MAE:", mae_validation)
print("検証データの平均売上:", validation_mean)
print("RMSEの平均売上に対する割合:", f"{rmse_validation_percentage:.2f}%")
print("MAEの平均売上に対する割合:", f"{mae_validation_percentage:.2f}%")

# 3. テストデータに対して予測し、予測精度を確認
forecast_steps_test = len(test_data)
forecast_test = model_fit.forecast(steps=forecast_steps_test)

# テストデータ期間の売上平均値を計算
test_mean = test_data['Ice_Cream_Sales'].mean()

# テストデータでの予測精度を計算
rmse_test = np.sqrt(mean_squared_error(test_data['Ice_Cream_Sales'], forecast_test))
mae_test = mean_absolute_error(test_data['Ice_Cream_Sales'], forecast_test)

# 平均に対するRMSEとMAEの割合を計算
rmse_test_percentage = (rmse_test / test_mean) * 100
mae_test_percentage = (mae_test / test_mean) * 100

print("\nテストデータでの予測精度:")
print("RMSE:", rmse_test)
print("MAE:", mae_test)
print("テストデータの平均売上:", test_mean)
print("RMSEの平均売上に対する割合:", f"{rmse_test_percentage:.2f}%")
print("MAEの平均売上に対する割合:", f"{mae_test_percentage:.2f}%")

# 4. グラフで訓練データ、検証データ、テストデータ、予測値を表示
plt.figure(figsize=(12, 6))
# 訓練データ、検証データ、テストデータを連続してプロット
plt.plot(pd.concat([train_data, validation_data, test_data]).index,
         pd.concat([train_data['Ice_Cream_Sales'], validation_data['Ice_Cream_Sales'], test_data['Ice_Cream_Sales']]),
         label="Actual Data", color='blue')
plt.plot(validation_data.index, forecast_validation, label="Forecast (Validation)", color='orange')
plt.plot(test_data.index, forecast_test, label="Forecast (Test)", color='red')
plt.title("Ice Cream Sales Forecast (Train, Validation & Test)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()
