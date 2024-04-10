import streamlit as st 
from datetime import datetime


import pandas as pandas
import yfinance as yf 
from prophet import Prophet 
from prophet.plot import plot_plotly
from plotly import graph_objs as go 
from plotly.subplots import make_subplots
import plotly as px 
from prophet.plot import add_changepoints_to_plot
from prophet.daignostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from PIL import Image



START = "2015-01-01"
TODAY = date.today().strftile("%Y-%m-%d")

st.title('Stock Prediction App')

image = Image.open('prediction.jpg')

st.image(image, caption='Magician')

st.write('Esta application to permite generar predictions del precio de las acciones de los stocks mas importances .')
st.markdown(''' '' '''La libreria usada para el predicting es **[Prophet](https://facebook.github.io/prophet/)**.''' '' ''')

stocks = ('GOOG','AAPL','MSFT','TSLA','FB','AMZN','BTC-USD','ETH-USD')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:',1,4)
period = n_years* 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

st.subheader('1.Data Loading')

data_load_state =st.text('loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('2.Exploratory Data Analysis')
st.markdown('Una vez cargaddos los datos pasamos al analysis exploratorio')

st.subheader('Raw data')
st.markdown('Debajo tenemos las ultimas 5 observaciones del stock')
st.write(data.tail())

st.subheader('Descriptive Statistics')
st.markdown('Se pueden observer los maximize,mminimize, desviacion estandar, precio medio')
st.write(data.describe())

#Plot raw data
st.subheader(' Line Plot')

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.layout.update(title_text='Time Series datawith Rangeslider', xaxis _rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Candle Plot
st.subheader(' Candlestick Plot: Price Evolution')

def plot_candle_data():
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Date'],
               open=data['Open'],
               high=data['High'],
               low=data['Low'],
               close=data['Close'], name = 'market data'))
    fig.update_layout(
        title='Stock share price evolution',
        yaxis_title='Stock Price (USD per Share)')
        st.plotly_chart(fig)

plot_candle_data()



# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet(interval_width=0.95)
m.fit(df_train)
future = m.make_future_dataframe(periods,freq = 'D')
forecast = m.predict(future)

# Show and plot forecast
st.subheader('3.Forecast data')
st.write("E1 modelo se entrena con los datos y genera predicciones.")
st.write("Carga una seria temporal para activarlo.")
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader("Forecast componenets")
st.write("Cargamos los components del modelo.")
fig2 = m.plot_components(forecast)
st.write(fig2)
st.markdown(' E1 primer grafico muestra information sobre la tendencia.')
st.markdown(' E1 segundo grafico muestra information sobre la tendencia semanal.')
st.markdown(' E1 ultimo grafico nos aporta informacion acerca de la tenencia anual.')

st.subheader('ChangePoints Plot')
st.markdown('Los **Changepoints** son los puntos de fecha en los qoe las series temporales prwsentan cambios bruscos en la trayectoria.')
st.markdown('Por defecto, Prophet anade **25 puntos** de cambio al 80% inicial del conjunto de datos.')

fig3 = m.plot(forecast)
a = add_changepoints_to_plot(fig3.gca(), m, forecast)
st.write(fig3)


st.subheader('4.Model Evaluation')
st.markdown('Para analizar el **MAE y RMSE**, debemos divir los datos en train y test y hacer una validacion cruzada ')
with st.expander("Explication"):
    st.markdown("""La libreria **Prophet** permite dividir nuestro historico de datos en data de entrenamiento y datos test para hacer una validacion cruzada. Las prin )
    st.write("**Training data (initial)**: La cantidad de datos para el entranmiento. E1 parametro en la APIse llama inicial.)
    st.write("**Horizon**: Los datos aparte de la validacion.")
    st.write("**Cutoff (period )**: se realiza un forecast para cada punto observado entre el corte y el corte + horizonte.""")

with st.expander("Cross validation"):
    initial = st.number_input(value= 365,label="initial",min_value=30,max_value=1096)
    initial = str(initial) + "days"

    period = st.number_input(value= 90,label="period",main_value=1,max_value=365)
    period = str(period) + "days"

    horizon = st.number_input(value= 90,label="period",main_value=30,max_value=366)
    horizon = str(period) + "days"

with st.expander("Metrics"):


    df_cv = cross_validation(m, initial='1000 days', period='90 days', horizon = '365 days')
    df_p= performance_metrics(df_cv)

    #st.write(df_p)

    st.markdown('Metrics definition')
    st.write("**Mse: mean absolute error**")
    st.write("**Mae: Mean average error**")
    st.write("**Mape: Mean average percentage error")
    st.write("**Mse: mean absolute error**")
    st.write("**Mdape: Median average percentage error**")


    try:
        metrics = ['Choose a metric','mse','rmse','mae','mape','mdape','coverage']
        selected_metrics = st.selectbox("Select metric to plot",options=metrics)
        fig4 = plot_cross_validation_metric(df_cv, metric=selected_metric)
        st.write(fig4)
    except:
        st.error("Please make sure that you select a metric")
        st.stop()




st.subheader('Authors')
st.write('**Sebastian Esponda** :sunglasses:' )
st.write('**Gary Martin** :wink:')
st.write('**Levi Vilchez** :stuck_out_tongue:')
st.write('**Javier Jimenez Pena** :laughing:')
st.write('**Yoursef Ouabi**:smile')





