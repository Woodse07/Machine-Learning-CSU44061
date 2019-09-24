import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import csv
from sklearn.ensemble import RandomForestRegressor

def simplify_ages(df):
	df.Age = df.Age.fillna(-0.5)
	bins = (0,15,25,36,50,120)
	group_names = ['0-15', '15-25', '25-36', '36-50', '50-120']
	df.Age = pd.cut(df.Age, bins, labels=group_names)
	return df

def simplify_city_size(df):
	df['Size of City'] = df['Size of City'].fillna(-0.5)
	bins = (-1,0,72734,286000,506092,750000,1184501,25000000,50000000)
	group_names = ['Unknown', '1_quartile', '1b_quartile', '2_quartile', '2_quartileb', '3_quartile', '3_quartileb', '4_quartile']
	df['Size of City'] = pd.cut(df['Size of City'], bins, labels=group_names)
	return df

def simplify_yor(df):
	df['Year of Record'] = df['Year of Record'].fillna(-0.5)
	bins = (-1,1979,1990,2000,2010,2020)
	group_names = ['unknown', '1980s', '1990s', '2000s', '2010s']
	df['Year of Record'] = pd.cut(df['Year of Record'], bins, labels=group_names)
	return df

def simplify_height(df):
	df['Body Height [cm]'] = df['Body Height [cm]'].fillna(-0.5)
	bins = (-1, 90, 160, 175, 191, 270)
	group_names = ['Unknown', '90-160', '160-175', '175-191', '191-270']
	df['Body Height [cm]'] = pd.cut(df['Body Height [cm]'], bins, labels=group_names)
	return df

def drop_features(df):
	df = df.drop('Hair Color', axis=1)
	df = df.drop('Wears Glasses', axis=1)
	df = df.drop('Instance', axis=1)
	#df = df.drop('University Degree', axis=1)
	#df = df.drop('Gender', axis=1)
	#df = df.drop('Profession', axis=1)
	return df

def encode_features(df):
	features = ['University Degree', 'Age', 'Size of City', 'Body Height [cm]', 'Year of Record', 'Gender', 'Profession']
	for feature in features:
		le = preprocessing.LabelEncoder()
		df[feature] = le.fit_transform(df[feature].astype(str))
	return df

# Loading data and dealing with NAs
train_data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
train_data[(train_data['Gender'] == '0')] = None
train_data[(train_data['University Degree'] == '0')] = None
train_data = train_data.fillna(train_data.median())
train_data = train_data.fillna("unknown")


# Applying Transformations
train_data = simplify_ages(train_data)
train_data = simplify_city_size(train_data)
train_data = simplify_yor(train_data)
train_data = simplify_height(train_data)

# Dropping useless features
train_data = drop_features(train_data)

# Encoding Features
train_data = encode_features(train_data)

# Extracting Features & income
X = train_data[train_data.columns[:-1]]
Y = train_data[train_data.columns[-1]]

# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) 

# Random Forest Regression
#rf = RandomForestRegressor(n_estimators = 100, random_state = 42, max_depth=7)
#rf.fit(X_train, y_train)
#y_pred = rf.predict(X_test)
#print(rf.score(X_test, y_test))

# Linear Regression
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print("\n")
print(regr.score(X_test, y_test))

print(y_test.head())
print("\n")
for i in range(5):
	print(y_pred[i])

