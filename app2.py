import streamlit as st 
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

st.title('Stock Prediction App')

image = Image.open('prediction.jpg')

st.image(image, caption='Magician')

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

st.subheader('4. Model Evaluation')
st.markdown('To analyze *MAE and RMSE*, we need to split the data into train and test and perform cross-validation.')
with st.expander("Explanation"):
    st.markdown("""The *Prophet* library allows us to split our historical data into training data and test data for cross-validation. The main parameters include:""")
    st.write("*Training data (initial)*: The amount of data for training. The parameter in the API is called initial.")
    st.write("*Horizon*: The data beyond the validation.")
    st.write("*Cutoff (period)*: A forecast is made for each observed point between the cutoff and cutoff + horizon.")

with st.expander("Cross validation"):
    initial = st.number_input(value=365, label="initial", min_value=30, max_value=1096)
    initial = str(initial) + "days"

    period = st.number_input(value=90, label="period", min_value=1, max_value=365)
    period = str(period) + "days"

    horizon = st.number_input(value=90, label="period", min_value=30, max_value=366)
    horizon = str(horizon) + "days"

with st.expander("Metrics"):
    df_cv = cross_validation(m, initial='1000 days', period='90 days', horizon='365 days')
    df_p = performance_metrics(df_cv)

    st.markdown('Metrics definition')
    st.write("*MSE: Mean Squared Error*")
    st.write("*RMSE: Root Mean Squared Error*")
    st.write("*MAE: Mean Absolute Error*")
    st.write("*MAPE: Mean Absolute Percentage Error*")
    st.write("*MdAPE: Median Absolute Percentage Error*")

    try:
        metrics = ['Choose a metric', 'mse', 'rmse', 'mae', 'mape', 'mdape', 'coverage']
        selected_metric = st.selectbox("Select metric to plot", options=metrics)
        fig4 = plot_cross_validation_metric(df_cv, metric=selected_metric)
        st.write(fig4)
    except:
        st.error("Please make sure that you select a metric")
        st.stop()

st.subheader('Authors')
st.write('*Sebastian Esponda* :sunglasses:' )
st.write('*Gary Martin* :wink:')
st.write('*Levi Vilchez* :stuck_out_tongue:')
st.write('*Javier Jimenez Pena* :laughing:')
st.write('*Yoursef Ouabi*:smile')