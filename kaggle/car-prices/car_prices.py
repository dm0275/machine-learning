#!/usr/bin/env python

import os
import sys
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Import the data set
data_frame = pd.read_csv("car_prices.csv")

# X_train = pd.DataFrame(data_frame.loc[:, data_frame.columns != 'sellingprice']).to_numpy()

X_train = pd.Series(data_frame.year)

y_train = pd.Series(data_frame.sellingprice)
features = ['size(sqft)','bedrooms','floors','age']

# print(y_train[0])
# print(X_train[0])
# print(type(X_train))
# print(y_train.to_frame())
# print(data_frame.columns)

# fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X_train[:,i],y_train)
#     ax[i].set_xlabel(features[i])
# ax[0].set_ylabel("Price (1000's)")
# plt.show()

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

X_train = pd.Series(data_frame.make)

# plt.bar(y_train, X_train)
# plt.tight_layout()
# plt.show()

first_10_rows = pd.DataFrame(data_frame.iloc[:20])
print(first_10_rows)

name = data_frame['make']#.head(20)
price = data_frame['sellingprice']#.head(20)

# Figure Size
fig = plt.figure(figsize =(10, 7))

# Horizontal Bar Plot
# plt.bar(name, price)
plt.bar(name[0:20], price[0:20])

plt.show()

print(name[0:20])

# ax = first_10_rows.plot.bar(x="make", y="sellingprice", rot=0)
# plt.tight_layout()
# plt.show()