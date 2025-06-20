import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar

# read and print df containing csv
df = pd.read_csv("data.csv")

# clean price column
df['Price'] = df['Price'].str.replace(',','').astype(float)

# convert to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# rename colums
df = df.rename(columns={'Date':'ds','Price':'y'})

# define holidays
calendar = USFederalHolidayCalendar()
holidays = calendar.holidays(start=df['ds'].min(), end=df['ds'].max())
holiday_df = pd.DataFrame({'holiday': 'us_holiday', 'ds':pd.to_datetime(holidays)})

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
