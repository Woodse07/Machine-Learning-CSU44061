import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
train_data = train_data.fillna(train_data.mean())

print(train_data.head(3))
print(train_data.min())
print(train_data.max())

# Functions
def simplify_ages(df):
	df.Age = df.Age.fillna(-0.5)
	bins = (-1,0,5,12,18,25,35,45,60,120)
	group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Young Adult 2', 'Adult', 'Senior']
	df.Age = pd.cut(df.Age, bins, labels=group_names)
	return df

def simplify_city_size(df):
	df['Size of City'] = df['Size of City'].fillna(-0.5)
	bins = (-1,0,50000, 200000, 500000, 1000000, 1500000, 10000000)
	group_names = ['Unknown', 'Empty', 'Small', 'Medium', 'Medium-Big', 'Big', 'Huge']
	df['Size of City'] = pd.cut(df['Size of City'], bins, labels=group_names)
	return df

def drop_features(df):
	df = df.drop('Hair Color', axis=1)
	df = df.drop('Wears Glasses', axis=1)
	df = df.drop('Instance', axis=1)
	df = df.drop('University Degree', axis=1)
	df = df.drop('Gender', axis=1)
	return df


# Applying Transformations
train_data = simplify_ages(train_data)
train_data = simplify_city_size(train_data)

# Dropping useless features
train_data = drop_features(train_data)

print(train_data.head(3))
