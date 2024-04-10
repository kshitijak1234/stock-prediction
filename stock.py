import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Function to fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to preprocess data and split into features and target
def preprocess_data(stock_data):
    X = stock_data[['Open', 'High', 'Low', 'Volume']]
    y = stock_data['Close']
    return X, y

# Function to train the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to predict next day's closing price
def predict_price(model, last_data):
    predicted_price = model.predict(last_data)
    return predicted_price

# Function to generate recommendations based on predicted price
def generate_recommendations(predicted_price, current_price):
    if predicted_price > current_price:
        return 'Buy'
    elif predicted_price < current_price:
        return 'Sell'
    else:
        return 'Hold'

# Streamlit app
def main():
    name = 'Stock Price Prediction'
    st.title(name)

    # Company selection
    symbol = st.selectbox('Select Company', ['AAPL', 'MSFT', 'GOOGL'])

    # Year selection using a slider
    current_year = datetime.now().year
    year = st.slider('Select Year', min_value=2010, max_value=current_year, value=current_year)

    # Fetching stock data for the selected year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    stock_data = fetch_stock_data(symbol, start_date, end_date)

    # Preprocessing data
    X, y = preprocess_data(stock_data)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    model = train_model(X_train, y_train)

    # Evaluating the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.write(f'Training Score: {train_score}')
    st.write(f'Testing Score: {test_score}')

    # Predicting next day's closing price
    last_data = X.tail(1)
    predicted_price = predict_price(model, last_data)

    # Predict the closing price for the next day
    predicted_price1 = model.predict(last_data)
    st.write(f"Predicted closing price for the next day: {predicted_price1[0]}")

    # Generate recommendations
    current_price = stock_data['Close'].iloc[-1]
    recommendation = generate_recommendations(predicted_price1[0], current_price)
    st.write(f"Recommendation: {recommendation}")

    # Plotting actual and predicted closing prices
    num_days = st.slider('Number of Days to Display', min_value=1, max_value=len(stock_data), value=30)
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index[-num_days:], stock_data['Close'].tail(num_days), label='Actual Closing Price', color='blue')
    plt.axvline(x=stock_data.index[-1], color='gray', linestyle='--', label='Predicted Day')
    plt.plot(stock_data.index[-1] + timedelta(days=1), predicted_price, marker='o', markersize=8, label='Predicted Closing Price', color='red')
    plt.title(name)
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)

    st.pyplot(plt)

    # Displaying recommendations using horizontal bar graph
    recommendations_count = {'Buy': 0, 'Sell': 0, 'Hold': 0}
    recommendations_count[recommendation] += 1
    total_recommendations = sum(recommendations_count.values())
    recommendations_percentage = {k: v / total_recommendations * 100 for k, v in recommendations_count.items()}
    recommendations_df = pd.DataFrame.from_dict(recommendations_percentage, orient='index', columns=['Percentage'])
    st.bar_chart(recommendations_df)

        # Plot raw data
    st.subheader('Line Plot')

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Candle Plot
    st.subheader('Candlestick Plot: Price Evolution')

    def plot_candle_data():
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], name='market data'))
        fig.update_layout(
            title='Stock share price evolution',
            yaxis_title='Stock Price (USD per Share)')
        st.plotly_chart(fig)

    plot_candle_data()
        
    # Predict forecast with Prophet.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet(interval_width=0.95)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period, freq='D')
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('3. Forecast data')
    st.write("The model is trained with the data and generates predictions.")
    st.write("Load a time series to activate it.")
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)



        
main()