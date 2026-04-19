import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xg
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as mpl

data_frame = pd.read_csv('Cleaned_Dataset_2yr.csv')

data_frame.columns

#Bed Bath Ratio
# adding 0.5 to the denominator because avoids dividing by zero

data_frame['bed_bath_ratio'] = (data_frame['BedroomsTotal']) / (data_frame['BathroomsTotalInteger'] + 0.5)
data_frame['bed_bath_ratio'] = data_frame['bed_bath_ratio'].round(2)

#Cleaning up dataset( Dropping columns, Dropping NA values, and Encoding)
data_frame = data_frame.drop(columns = ['ListingId', 'CloseDate'])
data_frame = data_frame.dropna()

category = data_frame.select_dtypes(include=['object', 'category']).columns
for x in category: #encoding for the category values
  data_frame[x] = LabelEncoder().fit_transform(data_frame[x])

#Train Test Split
X = data_frame.drop(columns = ['ClosePrice'])
y = data_frame['ClosePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

print(data_frame)

# Simple LinearRegressions
md = LinearRegression()
md.fit(X_train, y_train)

# R^2
r_squared = r2_score(y_test, md.predict(X_test))
print()
print('r^2 = ', r_squared)

print()

# MSE
mse = mean_squared_error(y_test, md.predict(X_test))
print('MSE = ', mse)

print()

# RMSE
rmse = np.sqrt(mse)
print('RMSE = ', rmse)


#Random Forest Implementation (100 Trees)
rand_for = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rand_for.fit(X_train, y_train)

y_pred = rand_for.predict(X_test) #predicted value

print()

#r2 value 
r_2 = r2_score(y_test, y_pred)
print(f'R^2 value = {r_2}')

#mdAPE value 
val = np.abs(y_test - y_pred) / y_test
print(f'mdAPE value = {np.median(val * 100)}')

#Meaningful Features
features = pd.Series(rand_for.feature_importances_, index=X_train.columns)
features = features.sort_values(ascending = False)

mpl.figure(figsize = (12,6))
features.plot(kind='bar')
mpl.title('Important Features')
mpl.show()

print()

feature_table = features.reset_index()
feature_table.columns = ['Feature', 'Value of Importance']
feature_table['Value of Importance'] = (feature_table['Value of Importance'] * 100).round(2)
feature_table = feature_table.sort_values(by = 'Value of Importance', ascending = False)
print(feature_table)

# XGBoost Model
mod = XGBRegressor(
    n_estimators = 200,
    learning_rate = 0.1,
    max_depth = 3,
    subsample = 0.8,
    random_state = 42
)
mod.fit(X_train, y_train)

print(f' r^2 value: {r2_score(y_test, mod.predict(X_test))}')

print()

value = np.abs(y_test - mod.predict(X_test)) / y_test
print(f'mdAPE value = {np.median(value * 100)}')

# model evaluation
features = pd.Series(mod.feature_importances_, index=X_train.columns)
features = features.sort_values(ascending = False)

mpl.figure(figsize = (12,6))
features.plot(kind='bar')
mpl.title('Important Features')
mpl.show()

print()

feature_table = features.reset_index()
feature_table.columns = ['Feature', 'Value of Importance']
feature_table['Value of Importance'] = (feature_table['Value of Importance'] * 100).round(2)
feature_table = feature_table.sort_values(by = 'Value of Importance', ascending = False)
print(feature_table)

