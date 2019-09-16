import csv
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 

# Training Data
train_data = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
train_data = train_data.fillna(train_data.mean())

# Test Data
test_data = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
test_features = pd.get_dummies(train_data[train_data.columns[:-1]], drop_first=False)


# Deleting rows I think don't matter
#train_data = train_data.drop('Hair Color', axis=1)
#train_data = train_data.drop('Wears Glasses', axis=1)
#train_data = train_data.drop('Instance', axis=1)
#train_data = train_data.drop('Body Height [cm]', axis=1)

# Seperating features and using get_dummies to deal with categorical data
features = pd.get_dummies(train_data[train_data.columns[:-1]], drop_first=False)

# Seperating incomes and casting them to ints (Might have to come back to this)
incomes = train_data[train_data.columns[-1]]

regr = linear_model.LinearRegression()
regr.fit(features, incomes)

predicted_incomes = regr.predict(test_features)

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
				row[1] = predicted_incomes[line_count-1]
				line_count += 1
				writer.writerow(row)
