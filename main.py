import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ===========================
# 1. DATA LOADING & PREPROCESSING
# ===========================
@st.cache_data
def load_data():
    file1 = "https://raw.githubusercontent.com/damodar04/Demand-Forecasting-System/main/Transactional_data_retail_01.csv"
    file2 = "https://raw.githubusercontent.com/damodar04/Demand-Forecasting-System/main/Transactional_data_retail_02.csv"
    customers = "https://raw.githubusercontent.com/damodar04/Demand-Forecasting-System/main/CustomerDemographics.csv"
    products = "https://raw.githubusercontent.com/damodar04/Demand-Forecasting-System/main/ProductInfo.csv"

    transactions = pd.concat([pd.read_csv(file1), pd.read_csv(file2)], ignore_index=True)
    customers = pd.read_csv(r"c:\Users\a\Desktop\Usecase_DemandForecasting\Usecase_DemandForecasting\Data_Files\CustomerDemographics.csv")
    products = pd.read_csv(r"c:\Users\a\Desktop\Usecase_DemandForecasting\Usecase_DemandForecasting\Data_Files\ProductInfo.csv")

    transactions['InvoiceDate'] = pd.to_datetime(transactions['InvoiceDate'], format='mixed', dayfirst=True)

    df = transactions.merge(products, on='StockCode', how='left')
    df = df.merge(customers, on='Customer ID', how='left')

    df['TotalSales'] = df['Quantity'] * df['Price']

    df['Month'] = df['InvoiceDate'].dt.month
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Year'] = df['InvoiceDate'].dt.year

    return df

df = load_data()

# ===========================
# 2. IDENTIFY TOP 10 PRODUCTS
# ===========================
top_products = df.groupby('StockCode')['Quantity'].sum().nlargest(10).index
df_top = df[df['StockCode'].isin(top_products)]

# ===========================
# 3. SARIMA MODEL FOR FORECASTING
# ===========================
def forecast_sarima(stock_code, weeks=15):
    product_sales = df[df['StockCode'] == stock_code].groupby('InvoiceDate')['Quantity'].sum().reset_index()
    product_sales = product_sales.set_index('InvoiceDate')

    sarima_model = SARIMAX(product_sales, order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_results = sarima_model.fit()

    future_dates = pd.date_range(start=product_sales.index[-1], periods=weeks, freq='W')
    forecast = sarima_results.get_forecast(steps=weeks)
    forecast_values = forecast.predicted_mean

    # Plot Historical and Forecasted Demand
    plt.figure(figsize=(12,6))
    plt.plot(product_sales, label="Historical Demand", color="blue")
    plt.plot(future_dates, forecast_values, label="Forecasted Demand", color="red", linestyle="dashed")
    plt.legend()
    plt.title(f"Historical vs Forecasted Demand for {stock_code}")
    plt.xlabel("Date")
    plt.ylabel("Quantity Sold")
    plt.show()

    return product_sales, future_dates, forecast_values

# ===========================
# 4. MACHINE LEARNING MODEL (XGBoost)
# ===========================
def train_xgboost():
    features = ['Month', 'DayOfWeek', 'Year', 'Quantity']
    target = 'TotalSales'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model_xgb = XGBRegressor()
    model_xgb.fit(X_train, y_train)

    y_pred_train = model_xgb.predict(X_train)
    y_pred_test = model_xgb.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    return model_xgb, mae_train, mae_test, y_train, y_pred_train, y_test, y_pred_test

xgb_model, mae_train, mae_test, y_train, y_pred_train, y_test, y_pred_test = train_xgboost()

# ===========================
# 5. STREAMLIT WEB APP
# ===========================
st.title("📊 Demand Forecasting System")

product_selected = st.sidebar.selectbox("Select Stock Code:", top_products)

product_sales, future_dates, forecast_values = forecast_sarima(product_selected, 15)

# Extract last 15 weeks of actual data to align with the forecast
actual_dates = product_sales.index[-15:]
actual_values = product_sales[-15:]

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Demand": forecast_values
})
# ========== SHOW FORECAST TABLE ==========
st.subheader("📋 Forecasted Demand Table")
st.dataframe(forecast_df)

fig, ax = plt.subplots(figsize=(12, 6))

# Plot full historical demand
ax.plot(product_sales, label="Historical Demand", color="blue", marker='o')
ax.plot(actual_dates, actual_values, marker='o', linestyle='-', color='green', label="Actual Data (Last 15 Weeks)")
ax.plot(actual_dates, forecast_values[:15], marker='o', linestyle='dashed', color='red', label="Forecasted Data")
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Quantity Sold")
ax.set_title(f"Demand Forecast for {product_selected}")
st.pyplot(fig)


# ========== DOWNLOAD CSV BUTTON ==========
csv = forecast_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="📥 Download Forecasted Data",
    data=csv,
    file_name=f"forecast_{product_selected}.csv",
    mime="text/csv"
)
# ===========================
# 6. ERROR HISTOGRAM
# ===========================
# ========== ERROR HISTOGRAM (PLOTTED AUTOMATICALLY) ==========
st.subheader("📊 Error Distributions for Training and Test Data")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Training Error Histogram (Green)
sns.histplot(y_train - y_pred_train, bins=30, color="green", kde=True, ax=axes[0])
axes[0].set_title("Training Error Distribution", fontsize=12)
axes[0].set_xlabel("Error", fontsize=10)
axes[0].set_ylabel("Frequency", fontsize=10)
axes[0].grid(True)

# Testing Error Histogram (Red)
sns.histplot(y_test - y_pred_test, bins=30, color="red", kde=True, ax=axes[1])
axes[1].set_title("Testing Error Distribution", fontsize=12)
axes[1].set_xlabel("Error", fontsize=10)
axes[1].set_ylabel("Frequency", fontsize=10)
axes[1].grid(True)

plt.tight_layout()
st.pyplot(fig)
# ===========================
# 7. XGBoost Model Performance
# ===========================
st.sidebar.header("XGBoost Model Performance")
st.sidebar.write(f"Training MAE: {mae_train:.2f}")
st.sidebar.write(f"Test MAE: {mae_test:.2f}")
