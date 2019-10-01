import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import csv
from sklearn.base import TransformerMixin
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

# Functions
def test_model(X, Y, country_grouped):
	test_data = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
	test_data[(test_data['Gender'] == '0')] = None
	test_data[(test_data['University Degree'] == '0')] = None
	test_data = test_data.fillna(test_data.mean())
	test_data = test_data.fillna("unknown")

	#print(test_data['University Degree'].value_counts())

	print(test_data['Country'].head())


	# Applying Transformations
	test_data = simplify_ages(test_data)
	test_data = simplify_city_size(test_data)
	test_data = simplify_yor(test_data)
	test_data = simplify_height(test_data)
	#test_data = simplify_country(test_data)
	for i in range(len(test_data)):
			try:
				test_data['Country'][i] = country_grouped[test_data['Country'][i]]
			except:
				test_data['Country'][i] = 0


	# Dropping useless features
	test_data = drop_features(test_data)

	# Encoding Features
	test_data.drop(['Income'], axis=1, inplace=True)
	test_data = encode_features(test_data)


	# Normalising Data
	#print(test_data.head())
	#test_data=(test_data-test_data.min())/(test_data.max()-test_data.min())
	#print(test_data.head())

	# Linear Regression
	#regr = linear_model.LinearRegression()
	#regr.fit(X, Y)
	#y_pred = regr.predict(test_data)

	# Random Forest Regression
	rf = RandomForestRegressor(n_estimators = 1000, max_depth=8)
	rf.fit(X, Y)
	y_pred = rf.predict(test_data)

	with open('tcd ml 2019-20 income prediction submission file.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		with open('tcd ml 2019-20 income prediction submission file done.csv', 'w') as write_file:
			writer = csv.writer(write_file)
			for row in csv_reader:
				if line_count == 0:
					line_count += 1
					writer.writerow(row)
				else:
					row[1] = y_pred[line_count-1]
					line_count += 1
					writer.writerow(row)


def simplify_ages(df):
	df.Age = df.Age.fillna(-0.5)
	bins = (0,15,20,25,30,35,40,50,60,120)
	group_names = ['0-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-50', '50-60', '60-120']
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

def simplify_country(df):
	grouped = df.groupby('Country')
	grouped = grouped['Income in EUR'].agg(np.mean)
	for i in range(len(df)):
		train_data['Country'][i] = grouped[train_data['Country'][i]]
	return df, grouped

def drop_features(df):
	df = df.drop('Hair Color', axis=1)
	df = df.drop('Wears Glasses', axis=1)
	df = df.drop('Instance', axis=1)
	#df = df.drop('University Degree', axis=1)
	#df = df.drop('Gender', axis=1)
	#df = df.drop('Profession', axis=1)
	return df

def encode_features(df):
	features = ['University Degree', 'Age', 'Size of City', 'Body Height [cm]', 'Year of Record', 'Profession']
	for feature in features:
		le = preprocessing.LabelEncoder()
		df[feature] = le.fit_transform(df[feature].astype(str))
	df.loc[df['Gender'] == 'other', 'Gender'] = 'unknown'
	df = pd.concat([df,pd.get_dummies(df['Gender'], prefix='Gender')], axis=1)
	df.drop(['Gender'], axis=1, inplace=True)
	df.drop(['Gender_male'], axis=1, inplace=True)
	return df


train_data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
train_data[(train_data['Gender'] == '0')] = None
train_data[(train_data['University Degree'] == '0')] = None
train_data = train_data.fillna(train_data.mean())
train_data = train_data.fillna("unknown")

# Applying Transformations
train_data = simplify_ages(train_data)
train_data = simplify_city_size(train_data)
train_data = simplify_yor(train_data)
train_data = simplify_height(train_data)
train_data, country_grouped = simplify_country(train_data)

# Dropping useless features
train_data = drop_features(train_data)

# Encoding Features
train_data = encode_features(train_data)

# Extracting Features & Income
Y = train_data['Income in EUR']
train_data.drop(['Income in EUR'], axis=1, inplace=True)
X = train_data

# Normalising Professions
#print(X.head())
#X=(X-X.min())/(X.max()-X.min())
#print(X.head())

# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) 

# Linear Regression
#regr = linear_model.LinearRegression()
#regr.fit(X_train, y_train)
#y_pred = regr.predict(X_test)
#print("\n")
#print(regr.score(X_test, y_test))

# Lasso
#regr = linear_model.Lasso()
#regr.fit(X_train, y_train)
#y_pred = regr.predict(X_test)
#print(regr.score(X_test, y_test))

# Random Forest Regression
rf = RandomForestRegressor(n_estimators = 100, max_depth=8)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(rf.score(X_test, y_test))

# SGD Regressor
#regr = linear_model.SGDRegressor()
#regr.fit(X_train, y_train)
#y_pred = regr.predict(X_test)
#print(regr.score(X_test, y_test))

print(y_test.head())
print("\n")
for i in range(5):
	print(y_pred[i])


test_model(X, Y, country_grouped)













