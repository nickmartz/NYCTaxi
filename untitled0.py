
# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Importing the dataset
dataset = pd.read_csv('nycTaxiDataset.csv')

dataset['Jan'] = np.where(dataset['pickup_date'] == 1, 1, 0)
dataset['Feb'] = np.where(dataset['pickup_date'] == 2, 1, 0)
dataset['Mar'] = np.where(dataset['pickup_date'] == 3, 1, 0)
dataset['Apr'] = np.where(dataset['pickup_date'] == 4, 1, 0)
dataset['May'] = np.where(dataset['pickup_date'] == 5, 1, 0)
dataset['Jun'] = np.where(dataset['pickup_date'] == 6, 1, 0)
dataset = dataset.drop(columns=['pickup_date'])

# Format purposes, target at the end
dataset['tripDuration'] = dataset['trip_duration']
dataset = dataset.drop(columns=['trip_duration'])
dataset.head()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

optDataset = dataset.drop(dataset[dataset.tripDuration > 20000].index)
optDataset = optDataset.drop(optDataset[optDataset.tripDuration < 300].index)
print(optDataset.shape)

optimalX = optDataset.iloc[:, :-1].values
optimalY = optDataset.iloc[:, -1].values

# Lets try it without outliers using stdev/mean
mean = np.mean(optimalY)
std = np.std(optimalY)

# low threshold is at least five minutes
lowThresh = mean - (2*std)
highThresh = mean + (2*std)
print('Low threshold:', lowThresh)
print('High threshold:', highThresh)

# Indexes to delete
optDataset = optDataset[optDataset.tripDuration < highThresh]
optDataset = optDataset[optDataset.tripDuration > lowThresh]

print(optDataset.shape)

optimalX = optDataset.iloc[:, :-1].values
optimalY = optDataset.iloc[:, -1].values
print(optimalX.shape)
print(optimalY.shape)

X_train, X_test, y_train, y_test = train_test_split(optimalX, optimalY, test_size = 1/10, random_state = 0)

# Fitting Random Forest Regression to the dataset
# Import RandomForestRegressor
# n_estimators is the number of trees in the forest
# I pick 277 to get closest to the estimate of 160k, what the candidate claims he/she got
# Too many trees causes overfitting
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

result = np.sqrt(mean_squared_error(y_test, y_pred))

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()