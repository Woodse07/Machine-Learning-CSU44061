import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import neural_network
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score

def simplify_ages(df):
	df.Age = df.Age.fillna(-0.5)
	bins = (0,15,20,25,30,35,40,50,60,120)
	group_names = ['0-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-50', '50-60', '60-120']
	df.Age = pd.cut(df.Age, bins, labels=group_names)
	return df

def simplify_city_size(df):
	df['Size of City'] = df['Size of City'].fillna(-0.5)
	bins = (0,72734,506092,1184501,50000000)
	group_names = ['1_quartile', '2_quartile', '3_quartile', '4_quartile']
	df['Size of City'] = pd.cut(df['Size of City'], bins, labels=group_names)
	return df

def simplify_yor(df):
	df['Year of Record'] = df['Year of Record'].fillna(-0.5)
	bins = (1979,1990,2000,2010,2020)
	group_names = ['1980s', '1990s', '2000s', '2010s']
	df['Year of Record'] = pd.cut(df['Year of Record'], bins, labels=group_names)
	grouped = df.groupby('Year of Record')
	grouped = grouped['Income in EUR'].agg(np.mean)
#	print(grouped.head(10))
	#year_avgs = []
	#for i in range(len(df)):
#		year_avgs.append(grouped[df['Year of Record'][i]])
#	df['Yr_Record_avg'] = year_avgs
#	print(df['Yr_Record_avg'].head())
	return df

def simplify_height(df):
	df['Body Height [cm]'] = df['Body Height [cm]'].fillna(-0.5)
	bins = (-1, 130, 160, 175, 191, 270)
	group_names = ['unknown', '90-160', '160-175', '175-191', '191-270']
	df['Body Height [cm]'] = pd.cut(df['Body Height [cm]'], bins, labels=group_names)
	return df

def simplify_country(df):
	grouped = df.groupby('Country')
	grouped = grouped['Income in EUR'].agg(np.mean)
	for i in range(len(df)):
		train_data['Country'][i] = grouped[train_data['Country'][i]]
	return df

def simplify_profession(df):
	grouped = df.groupby('Profession')
	grouped = grouped['Income in EUR'].agg(np.mean)
	for i in range(len(df)):
		train_data['Profession'][i] = grouped[train_data['Profession'][i]]
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
	features = ['University Degree', 'Age', 'Size of City', 'Body Height [cm]', 'Year of Record', 'Profession']
	for feature in features:
		le = preprocessing.LabelEncoder()
		df[feature] = le.fit_transform(df[feature].astype(str))
	df = pd.concat([df,pd.get_dummies(df['Gender'], prefix='Gender')], axis=1)
	df = pd.concat([df,pd.get_dummies(df['Age'], prefix='Age')], axis=1)
	df = pd.concat([df,pd.get_dummies(df['Year of Record'], prefix='YOR')], axis=1)
	df = pd.concat([df,pd.get_dummies(df['Size of City'], prefix='SOC')], axis=1)
	df = pd.concat([df,pd.get_dummies(df['University Degree'], prefix='Degree')], axis=1)
	df = pd.concat([df,pd.get_dummies(df['Body Height [cm]'], prefix='Height')], axis=1)
	df.drop(['Gender'], axis=1, inplace=True)
	df.drop(['Gender_male'], axis=1, inplace=True)
	df.drop(['Age'], axis=1, inplace=True)
	df.drop(['Year of Record'], axis=1, inplace=True)
	df.drop(['Size of City'], axis=1, inplace=True)
	df.drop(['University Degree'], axis=1, inplace=True)
	df.drop(['Body Height [cm]'], axis=1, inplace=True)
	df['Country'] = df['Country'] / df['Country'].max()
	df['Profession'] = df['Profession'] / df['Profession'].max()
	return df

def info(df):
	print("")
	for i in df.columns:
		if type(df[i][1])== str:
			print(df[i].value_counts())
			print("")
	print("")

def plot(df):
	fig, ((a,b),(c,d),(e,f)) = plt.subplots(3,2,figsize=(15,20))
	plt.xticks(rotation=45)
	sns.countplot(df['Country'],hue=df['Income in EUR'],ax=f)
	sns.countplot(df['Age'],hue=df['Income in EUR'],ax=b)
	sns.countplot(df['Gender'],hue=df['Income in EUR'],ax=c)
	sns.countplot(df['Year of Record'],hue=df['Income in EUR'],ax=d)
	sns.countplot(df['Body Height [cm]'],hue=df['Income in EUR'],ax=e)
	sns.countplot(df['University Degree'],hue=df['Income in EUR'],ax=a)


# Loading data and dealing with NAs
train_data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
#plot(train_data)
#info(train_data)
train_data[(train_data['Gender'] == '0')] = None
train_data[(train_data['Gender'] == 'other')] = None
train_data[(train_data['University Degree'] == '0')] = None
train_data = train_data.fillna(train_data.mean())
train_data = train_data.fillna("unknown")


# Applying Transformations
train_data = simplify_ages(train_data)
train_data = simplify_city_size(train_data)
train_data = simplify_yor(train_data)
train_data = simplify_height(train_data)
train_data = simplify_country(train_data)
#train_data = simplify_profession(train_data)

# Dropping useless features
train_data = drop_features(train_data)

#info(train_data)

# Encoding Features
train_data = encode_features(train_data)

# Extracting Features & income
Y = train_data['Income in EUR']
train_data.drop(['Income in EUR'], axis=1, inplace=True)
X = train_data

# Doesn't Work
# Normalising Professions
#print(X.head())
#X=(X-X.min())/(X.max()-X.min())
#print(X.head())

# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) 

# Random Forest Regression
#rf = RandomForestRegressor(n_estimators = 1000, max_depth=10, n_jobs=-1)
#rf.fit(X_train, y_train)
#y_pred = rf.predict(X_test)
#print(rf.score(X_test, y_test))
#rms = sqrt(mean_squared_error(y_test, y_pred))
#print("")
#print(rms)
#print("")

# Linear Regression
#regr = linear_model.LinearRegression()
#regr.fit(X_train, y_train)
#y_pred = regr.predict(X_test)
#print("\n")
#print(regr.score(X_test, y_test))

# Neural Net
#regr = neural_network.MLPRegressor(solver = 'lbfgs', learning_rate = 'constant', activation = 'relu', verbose = True, shuffle = False, hidden_layer_sizes=(100,100,100), early_stopping = True)
#regr.fit(X_train, y_train)
#y_pred = regr.predict(X_test)
#print(regr.score(X_test, y_test))
#rms = sqrt(mean_squared_error(y_test, y_pred))
#print("")
#print(rms)
#print("")

# Xgboost
X_train['Country'] = pd.to_numeric(X_train['Country'])
X_test['Country'] = pd.to_numeric(X_test['Country'])
print(X_train.dtypes)
xgboost = XGBRegressor()
xgboost.fit(X_train, y_train)
print(xgboost)
y_pred = xgboost.predict(X_test)
accuracy = xgboost.score(X_test, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print(X_test.head())
print(y_test.head())
print("\n")
for i in range(5):
	print(y_pred[i])


