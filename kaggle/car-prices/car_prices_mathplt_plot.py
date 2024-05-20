#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd

# Import the data set
data_frame = pd.read_csv("car_prices.csv")


X_train = pd.Series(data_frame.year)
y_train = pd.Series(data_frame.sellingprice)


plt.scatter(X_train, y_train)
plt.title("Car Price by Year")
plt.xlabel("Year")
plt.ylabel("Car Price")
plt.show()

X_train = pd.Series(data_frame.odometer)

plt.scatter(X_train, y_train)
plt.title("Car Price based on mileage")
plt.xlabel("Mileage")
plt.ylabel("Car Price")
plt.show()


# Count the number of car sales per make
plt.figure(figsize=(20,40))
X_train = pd.Series(data_frame.make)

X_train.value_counts().plot(kind='bar', width=0.4)
plt.title('Number of Cars by Make')
plt.xlabel('Make')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
