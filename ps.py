import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf

# Define the stock symbol and time range for historical data
stock_symbol = 'AAPL'
start_date = '2021-01-01'
end_date = '2022-01-01'

# Fetch historical stock price data using yfinance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Extracting features (input) and target (output) variables
X = stock_data[['Open', 'High', 'Low', 'Volume']]  # Using Open, High, Low, Volume as features
y = stock_data['Close']  # Predicting the closing price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Using 100 trees
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training Score: {train_score}")
print(f"Testing Score: {test_score}")

# Predict the closing price for the next day
last_data = X.tail(1)
predicted_price = model.predict(last_data)
print(f"Predicted closing price for the next day: {predicted_price[0]}")

# Plotting actual and predicted closing prices
plt.figure(figsize=(10, 6))
plt.plot(y.index, y, label='Actual Closing Price', color='blue')
plt.axvline(x=last_data.index, color='gray', linestyle='--', label='Predicted Day')
plt.plot(last_data.index, predicted_price, marker='o', markersize=8, label='Predicted Closing Price', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()