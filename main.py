import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# read and print df containing csv
df = pd.read_csv("data.csv")

# clean price column
df['Price'] = df['Price'].str.replace(',','').astype(float)

# convert to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# rename colums
df = df.rename(columns={'Date':'ds','Price':'y'})

# initialize prophet model

model = Prophet(daily_seasonality=True)
model.fit(df)

# future predictions
future = model.make_future_dataframe(periods=365)

forecast = model.predict(future)


# plot the forecast 
model.plot(forecast)
plt.title("XAUUSD Price Forecast")
plt.show()
