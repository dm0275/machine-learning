#!/usr/bin/env python

import pandas as pd
import numpy as np

df = pd.read_csv("car_prices.csv")

print(df.head())

print(df.info())
print(df.describe())