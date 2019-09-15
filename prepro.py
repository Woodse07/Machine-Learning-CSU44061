import pandas as pd

data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')

print("Shape: ", data.shape)

print("\nFeatures: ", data.columns)
print("\n")

features = data[data.columns[:-1]]

incomes = data[data.columns[-1]]

print(features.head())
