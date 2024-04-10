import streamlit as st 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, date

import pandas as pd
import yfinance as yf 
from prophet import Prophet 
from prophet.plot import plot_plotly
import plotly.graph_objs as go 
from plotly.subplots import make_subplots
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from PIL import Image

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction')
image = Image.open('prediction.jpg')
st.image(image, caption='Stock')

stocks = ('AAPL','TSLA')
selected_stock = st.selectbox('Select dataset for prediction', stocks)


n_years = st.slider('Years of prediction:', 1, 10)
period = n_years * 365


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data



data = load_data(selected_stock)


#st.markdown('Once the data is loaded, we move on to exploratory analysis.')





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
