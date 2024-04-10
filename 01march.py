import streamlit as st 
from datetime import datetime, date
import pandas as pd
import yfinance as yf 
from prophet import Prophet 
from prophet.plot import plot_plotly
import plotly.graph_objs as go 
from prophet.plot import add_changepoints_to_plot

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
    model = Prophet(interval_width=0.95)
    model.fit(X_train)
    return model

# Function to predict next day's closing price
def predict_price(model, last_data):
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    return forecast['yhat'].iloc[-1]

# Function to generate recommendations based on predicted price
def generate_recommendations(predicted_price, current_price):
    if predicted_price > current_price:
        return 'Buy'
    elif predicted_price < current_price:
        return 'Sell'
    else:
        return 'Hold'

def main():
    st.title('Stock Price Prediction App')

    
    
    st.write('This application allows you to generate predictions of stock prices for the most important stocks.')
    st.markdown('''The library used for prediction is **[Prophet](https://facebook.github.io/prophet/)**.''')

    stocks = ('GOOG', 'AAPL', 'MSFT', 'TSLA', 'FB', 'AMZN', 'BTC-USD', 'ETH-USD')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    st.subheader('1. Data Loading')

    data_load_state = st.text('loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    st.subheader('2. Exploratory Data Analysis')
    st.markdown('Once the data is loaded, we move on to exploratory analysis.')

    st.subheader('Raw data')
    st.markdown('Below are the last 5 observations of the stock.')
    st.write(data.tail())

    st.subheader('Descriptive Statistics')
    st.markdown('You can observe the maximum, minimum, standard deviation, and average price.')
    st.write(data.describe())

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

    m = train_model(df_train)

    # Show and plot forecast
    st.subheader('3. Forecast data')
    st.write("The model is trained with the data and generates predictions.")
    st.write("Load a time series to activate it.")
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)    
    st.subheader('3. Price Prediction')

    st.write('Forecasted Stock Prices:')
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    st.write('Forecast Plot:')
    fig = plot_plotly(m, forecast)
    st.plotly_chart(fig)


main()