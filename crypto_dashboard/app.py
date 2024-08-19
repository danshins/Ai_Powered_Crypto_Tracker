import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Set page configuration
st.set_page_config(
    page_title="Crypto Investment Dashboard",  # Title that appears on the browser tab
    page_icon="💰",  # Favicon (emoji or file path)
)

# Function to fetch cryptocurrency data from CoinGecko API
def fetch_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 20,
        "page": 1,
        "sparkline": False
    }
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data)[
        ['id', 'symbol', 'current_price', 'market_cap', 'total_volume', 'price_change_percentage_24h']]


# Function to fetch historical price data for LSTM model
def fetch_historical_data(crypto_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = [item[1] for item in data['prices']]
    return pd.DataFrame(prices, columns=["Price"])


# Function to preprocess data for LSTM model
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Create sequences
    sequence_length = 60
    X_train = []
    y_train = []

    for i in range(sequence_length, len(scaled_data)):
        X_train.append(scaled_data[i - sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, scaler


# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Function to predict future prices using LSTM model
def predict_prices(model, data, scaler):
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)

    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]


# Streamlit App Title
st.title("Crypto Investment Dashboard")
st.header("Real-Time Cryptocurrency Prices")

# Fetching and displaying real-time cryptocurrency data
crypto_df = fetch_crypto_data()
st.dataframe(crypto_df)

# Sidebar for Portfolio Management
st.sidebar.title("Portfolio Management")
selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", crypto_df['id'])
amount_held = st.sidebar.number_input("Amount Held (in units)", min_value=0.0, step=0.01)
add_to_portfolio = st.sidebar.button("Add to Portfolio")

# Store portfolio in session state
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []

# Add selected cryptocurrency to the portfolio
if add_to_portfolio:
    st.session_state['portfolio'].append({"crypto": selected_crypto, "amount": amount_held})
    st.sidebar.success(f"Added {amount_held} of {selected_crypto} to your portfolio!")

# Display the portfolio
portfolio_df = pd.DataFrame(st.session_state['portfolio'])
if not portfolio_df.empty:
    st.write("Your Portfolio:")
    st.dataframe(portfolio_df)

# Plot price trends for top 10 cryptocurrencies
st.subheader("Price Trend for Top 10 Cryptocurrencies")
fig = px.line(crypto_df, x=crypto_df['symbol'], y='current_price', labels={'x': 'Cryptocurrency', 'y': 'Price (USD)'})
st.plotly_chart(fig)


# Calculate and plot Simple Moving Averages (SMA)
def calculate_sma(data, window):
    return data.rolling(window=window).mean()


crypto_df['SMA_7'] = calculate_sma(crypto_df['current_price'], 7)
crypto_df['SMA_30'] = calculate_sma(crypto_df['current_price'], 30)
crypto_df['SMA_7'].fillna(method='ffill', inplace=True)
crypto_df['SMA_30'].fillna(method='ffill', inplace=True)

# Plot SMAs and current prices
fig = px.line(
    crypto_df,
    x=crypto_df['symbol'],
    y=['current_price', 'SMA_7', 'SMA_30'],
    labels={'x': 'Cryptocurrency', 'y': 'Price (USD)'}
)
st.plotly_chart(fig)

# LSTM Model Section in the Sidebar
st.sidebar.header("LSTM Price Prediction")

if st.sidebar.button("Train LSTM Model"):
    st.subheader(f"Training LSTM Model for {selected_crypto}...")

    # Fetch historical data and train the LSTM model
    try:
        historical_data = fetch_historical_data(selected_crypto, 365)  # Fetch 1 year of data
        X_train, y_train, scaler = preprocess_data(historical_data)
        model = build_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, batch_size=1, epochs=1)

        # Predict the next day's price
        predicted_price = predict_prices(model, historical_data, scaler)
        st.subheader(f"Predicted Price for Tomorrow: ${predicted_price:.2f}")
    except Exception as e:
        st.error(f"An error occurred during model training or prediction: {str(e)}")

# Set Price Alerts
st.sidebar.header("Set Price Alerts")
alert_threshold = st.sidebar.number_input("Alert if price drops below (USD)", min_value=0.0, step=0.01)
current_price = crypto_df.loc[crypto_df['id'] == selected_crypto, 'current_price'].values[0]

if current_price < alert_threshold:
    st.warning(f"Alert! {selected_crypto} has dropped below ${alert_threshold:.2f}")
