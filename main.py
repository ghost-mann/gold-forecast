import pandas as pd
import prophet
import matplotlib as plt

# read and print df containing csv
df = pd.read_csv("data.csv")

# initialize prophet model

model = prophet()
model.fit(df)

# future predictions
future = model.make_future_dataframe(period=365)

forecast = model.predict(future)

