import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta ,date
import plotly.graph_objs as go
import time
from prophet import Prophet 
from prophet.plot import plot_plotly
from plotly.subplots import make_subplots
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric


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
        return 'Buy', current_price, predicted_price, predicted_price - current_price
    elif predicted_price < current_price:
        return 'Sell', current_price, predicted_price, current_price - predicted_price
    else:
        return 'Hold', current_price, predicted_price, 0

# Streamlit app
def main():
    st.title('Stock Price Prediction')
    st.sidebar.header('Stock Price Prediction')

    # Company selection
    symbol = st.selectbox('Select Company', ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'FB', 'AMZN', 'BTC-USD', 'ETH-USD','IBM','NESN.SW','SONY','SMSN.IL','HP','KO','TATAMOTORS.NS','WIPRO.NS','INFY','DELL','NOK',''])

    # Year selection using a slider
    current_year = datetime.now().year
    year = st.slider('Select Year', min_value=2010, max_value=current_year, value=current_year)

    # Fetching stock data for the selected year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    stock_data = fetch_stock_data(symbol, start_date, end_date)
    num_days = st.slider('Number of Days to Display', min_value=1, max_value=len(stock_data), value=30)

    # Preprocessing data
    X, y = preprocess_data(stock_data)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    model = train_model(X_train, y_train)

    # Evaluating the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    st.sidebar.write(f'Training Score: {train_score}')
    st.sidebar.write(f'Testing Score: {test_score}')

    # Predicting next day's closing price
    last_data = X.tail(1)
    predicted_price = predict_price(model, last_data)

    # Predict the closing price for the next day
    predicted_price1 = model.predict(last_data)
    st.sidebar.write(f"Predicted closing price for the next day: {predicted_price1[0]}")

    # Generate recommendations
    current_price = stock_data['Close'].iloc[-1]
    recommendation, price_before, price_after, profit = generate_recommendations(predicted_price1[0], current_price)
    st.sidebar.write(f"Recommendation: {recommendation}")
    st.sidebar.write(f"Price before action: {price_before}")
    st.sidebar.write(f"Price after action: {price_after}")
    st.sidebar.write(f"Profit: {profit}")

    # Calculate days required to get profit after prediction
    if profit > 0:
        days_to_profit = (stock_data.index[-1] - stock_data.index[-2]).days
        st.sidebar.write(f"Days required to get profit after prediction: {days_to_profit}")

    # Plotting actual and predicted closing prices
    trace_actual = go.Scatter(x=stock_data.index[-num_days:], y=stock_data['Close'].tail(num_days), mode='lines', name='Actual Closing Price', line=dict(color='blue'))
    trace_predicted = go.Scatter(x=[stock_data.index[-1], stock_data.index[-1] + timedelta(days=1)], y=[stock_data['Close'].iloc[-1], predicted_price[0]], mode='markers+lines', name='Predicted Closing Price', marker=dict(color='red'))

    # Add current time to the graph
    current_time = datetime.now().strftime('%H:%M:%S')
    layout = go.Layout(title='Stock Price Prediction', xaxis=dict(title='Date'), yaxis=dict(title='Closing Price'), legend=dict(x=0, y=1), annotations=[dict(x=stock_data.index[-1], y=stock_data['Close'].iloc[-1], xref='x', yref='y', text=f'Current Time: {current_time}', showarrow=True, arrowhead=7, ax=0, ay=-40)])

    fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)
    st.plotly_chart(fig)

main()








st.title('Stock Prediction')
with st.expander("Stock Prediction App"):
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title('Stock Prediction App')

    stocks = ('GOOG', 'AAPL', 'MSFT', 'TSLA', 'FB', 'AMZN', 'BTC-USD', 'ETH-USD')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    @st.cache_resource
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

with st.expander("Line Plot"):
    # Plot raw data
    st.subheader('Line Plot')

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()


with st.expander("Candlestick Plot: Price Evolution"):
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

with st.expander("3. Forecast data"):
   # Show and plot forecast
    st.subheader('3. Forecast data')
    st.write("The model is trained with the data and generates predictions.")
    st.write("Load a time series to activate it.")
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)


    st.write(forecast[[ 'yhat_lower']])




    st.subheader("Forecast components")
    st.write("Loading model components.")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
    st.markdown('The first graph shows information about the trend.')
    st.markdown('The second graph shows information about the weekly trend.')
    st.markdown('The last graph provides information about the yearly trend.')

    st.subheader('ChangePoints Plot')
    st.markdown('*Changepoints* are the date points where time series exhibit abrupt changes in trajectory.')
    st.markdown('By default, Prophet adds *25 changepoints* to the first 80% of the dataset.')

    fig3 = m.plot(forecast)
    a = add_changepoints_to_plot(fig3.gca(), m, forecast)
    st.write(fig3) 



