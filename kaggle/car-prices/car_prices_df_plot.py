#!/usr/bin/env python


import matplotlib.pyplot as plt
import pandas as pd

# Import the data set
data_frame = pd.read_csv("car_prices.csv")
data_frame = data_frame.dropna()

# Plot using the `DataFrame.plot`
data_frame.plot(kind='scatter', x='year', y='sellingprice', title="Car Price by Year", xlabel="Year",
                ylabel="Car Price")
plt.show()

# Plot car price based on mileage
data_frame.plot(kind='scatter', x='odometer', y='sellingprice', title="Car Price based on mileage", xlabel="Mileage",
                ylabel="Car Price")
plt.show()

# Plot car price based on mileage
data_frame.value_counts('make').plot(kind='bar', title="Number of Cars by Make", xlabel="Make",
                                     ylabel="Count", rot=90, figsize=(10, 20))
plt.show()

