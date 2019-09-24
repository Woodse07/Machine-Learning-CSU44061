import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from math import sqrt
import matplotlib.pyplot as plt

# Functions
def simplify_ages(df):
	df.Age = df.Age.fillna(-0.5)
	bins = (-1,0,5,12,18,25,35,60,120)
	group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
	df.Age = pd.cut(df.Age, bins, labels=group_names)
	return df



# Training Data
train_data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
train_data = train_data.fillna(train_data.mean())

# Deleting rows I think don't matter
train_data = train_data.drop('Hair Color', axis=1)
train_data = train_data.drop('Wears Glasses', axis=1)
train_data = train_data.drop('Instance', axis=1)
train_data = train_data.drop('University Degree', axis=1)
train_data = train_data.drop('Gender', axis=1)
#train_data = train_data.drop('Body Height [cm]', axis=1)

# Transforming features to avoid overfitting
train_data = simplify_ages(train_data)

print("\n")
print("Data after transformation: ")
print(train_data.head())
print("\n")

# Test Data
test_data = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')

# Seperating features and using get_dummies to deal with categorical data
features = pd.get_dummies(train_data[train_data.columns[:-1]], drop_first=False)

# Seperating incomes and casting them to ints (Might have to come back to this)
incomes = train_data[train_data.columns[-1]]

# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(features, incomes, test_size=0.3, random_state=1) 

# Just for clarity
print("\n")
print("Training Data:")
print("\n")
print(X_train.head())
print(y_train.head())
print("\n")

# Just for clarity
print("\n")
print("Testing Data:")
print("\n")
print(X_test.head(5))
print(y_test.head(10))
print("\n")


# Trying K nearest neighbors first
#knn = KNeighborsClassifier(n_neighbors=3)
#knn.fit(X_train, y_train)

# Trying Linear Regression
regr = linear_model.LinearRegression()

# Trying Lasso
#regr = linear_model.Lasso()

regr.fit(X_train, y_train)

# Testing Data
#test_features = pd.get_dummies(test_data[test_data.columns], drop_first=False)

#y_pred = knn.predict(X_test)
#print(y_pred)

y_pred = regr.predict(X_test)
for i in range(10):
	print(y_pred[i])
print("\n")


#rms = sqrt(mean_squared_error(y_pred, y_test))
#print(rms)
print(regr.score(X_test, y_test))
#print("kNN model accuracy:", metrics.accuracy_score(y_test, y_pred)) 







