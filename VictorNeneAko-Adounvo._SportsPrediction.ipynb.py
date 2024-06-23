# -*- coding: utf-8 -*-
"""Homework 2 AI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vAl5qo8eDzZqjEIm3lgcoklPFGifSOTi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,VotingRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold,GridSearchCV
from google.colab import drive
import tensorflow as tf
drive.mount('/content/drive')
tf.config.list_physical_devices('GPU')

path_22 = '/content/drive/My Drive/AI work/players_22.csv'
path_21 = '/content/drive/My Drive/AI work/male_players (legacy).csv'


players22 = pd.read_csv(path_22)
players21 = pd.read_csv(path_21)

players21

players21.hist(bins=50, figsize = (20,15))
plt.show()

players21df = pd.DataFrame(players21)
players22df = pd.DataFrame(players22)
threshold = 0.3*len(players21df)

# Drop columns with more than 30% missing values
players21df = players21df.dropna(thresh=threshold, axis=1)

players21df

players21df.head()
players21df.info()

print(players21df.columns[:99])

irrelevant_columns = ['player_url','player_positions','nationality_id',"player_face_url",'fifa_update_date',"club_jersey_number","club_joined_date","club_contract_valid_until_year","long_name","dob","short_name","club_team_id","player_url","real_face","club_team_id","club_name","league_name","nationality_id","nationality_name","club_position",'ls', 'st', 'rs',
       'lw', 'lf', 'cf','fifa_version', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm',
       'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb','player_traits',
       'rcb', 'rb', 'gk','league_id']
players21df = players21df.drop(irrelevant_columns,axis = 1)
numeric_data = players21df.select_dtypes(include = np.number)
non_numeric = players21df.select_dtypes(include = ['object'])
numeric_data = numeric_data.fillna(numeric_data.mean())
print(non_numeric.columns)
non_numeric = pd.get_dummies(non_numeric).astype(int)

threshold = 0.3*len(players21df)

# Drop columns with more than 30% missing values
non_numeric = non_numeric.dropna(thresh=threshold, axis=1)



print(non_numeric.columns)

#multivariate imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter = 10, random_state = 0)
numeric_data = pd.DataFrame(np.round(imp.fit_transform(numeric_data)),columns = numeric_data.columns)

X = pd.concat([numeric_data,non_numeric],axis = 1)
X = X.fillna(X.mean())

corr_matrix = X.corr()
relevant_columns = corr_matrix["overall"].sort_values(ascending=False).index[:11]

X = X[relevant_columns]
X = X.sample(n=20000, random_state=42)

y = X["overall"]
x = X.drop("overall",axis = 1)
from sklearn.preprocessing import StandardScaler

"""Scaling

"""

scale = StandardScaler()
scaled = scale.fit_transform(x)
scaled

x = pd.DataFrame(scaled,columns = x.columns)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 42)

from sklearn.linear_model import LinearRegression
l = LinearRegression()
l.fit(xtrain,ytrain)
y_pred = l.predict(xtest)

intercept = l.intercept_
coefficients = l.coef_

import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error,r2_score

print(f""" Mean Absolute Error = {mean_absolute_error(y_pred,ytest)},
        Mean Squared Error = {mean_squared_error(y_pred,ytest)}
        Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,ytest))},
        R2 score = {r2_score(y_pred,ytest)}""""")

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 3)
x_poly_train = poly.fit_transform(xtrain)
x_poly_test = poly.fit_transform(xtest)

model = LinearRegression()
model.fit(x_poly_train,ytrain)
y_pred = model.predict(x_poly_test)
print(f""" Mean Absolute Error = {mean_absolute_error(y_pred,ytest)},
        Mean Squared Error = {mean_squared_error(y_pred,ytest)}
        Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,ytest))},
        R2 score = {r2_score(y_pred,ytest)}""""")

from sklearn.linear_model import ElasticNet, Lasso,Ridge

model = Ridge()
model.fit(xtrain,ytrain)

y_pred = model.predict(xtest)
print(f""" Mean Absolute Error = {mean_absolute_error(y_pred,ytest)},
        Mean Squared Error = {mean_squared_error(y_pred,ytest)}
        Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,ytest))},
        R2 score = {r2_score(y_pred,ytest)}""""")

model = ElasticNet()
model.fit(xtrain,ytrain)

y_pred = model.predict(xtest)
print(f""" Mean Absolute Error = {mean_absolute_error(y_pred,ytest)},
        Mean Squared Error = {mean_squared_error(y_pred,ytest)}
        Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,ytest))},
        R2 score = {r2_score(y_pred,ytest)}""""")

model = Lasso()
model.fit(xtrain,ytrain)

y_pred = model.predict(xtest)
print(f""" Mean Absolute Error = {mean_absolute_error(y_pred,ytest)},
        Mean Squared Error = {mean_squared_error(y_pred,ytest)}
        Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,ytest))},
        R2 score = {r2_score(y_pred,ytest)}""""")







num_folds = 10
kf  = KFold(n_splits = num_folds, shuffle = True, random_state = 42)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
dtree = DecisionTreeRegressor()

param_grid = {
    'max_depth': [2, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1.0, 'sqrt', 'log2']
}

grid_search = GridSearchCV(dtree,param_grid,cv = kf,scoring = 'neg_mean_squared_error')
GS_score = cross_val_score(dtree,xtrain,ytrain,cv = kf,scoring = 'neg_mean_squared_error')
grid_search.fit(xtrain, ytrain)
best_params = grid_search.best_params_
best_dt_model = grid_search.best_estimator_

score = best_dt_model.score(xtest, ytest)
mse = mean_squared_error(ytest, best_dt_model.predict(xtest))
rmse = np.sqrt(mse)

print("RMSE: %.4f" % rmse)
print("size of prediction: ", len(best_dt_model.predict(xtest)))
print("prediction: \n", best_dt_model.predict(xtest))
print("test score: {0:.4f}\n".format(score))

feature_names = xtrain.columns

#####Random Forest Regressor
RF_R = RandomForestRegressor(n_estimators = 100,random_state = 42)

RFscore = cross_val_score(RF_R,xtrain,ytrain,cv = kf,scoring = 'neg_mean_squared_error')
RMSE_for_RF_R = np.sqrt(-RFscore.mean())
RF_R.fit(xtrain,ytrain)

#predict and evaluate
y_pred = RF_R.predict(xtest)
score = RF_R.score(xtest,ytest)
mse = mean_squared_error(ytest,y_pred)
rmse = np.sqrt(mse)


print("RMSE: %.4f" % rmse)
print("size of prediction: ", len(y_pred))
print("prediction: \n", y_pred)
print("test score: {0:.4f}\n".format(score))


feature_names = xtrain.columns

##Gradient Boosting Regressor
GB_R = GradientBoostingRegressor(n_estimators = 500,random_state = 42,max_depth = 5, min_samples_split=2, learning_rate=0.01)
GBscore = cross_val_score(GB_R,xtrain,ytrain,cv = kf,scoring = 'neg_mean_squared_error')
RMSE_for_GB_R = np.sqrt(-GBscore.mean())
GB_R.fit(xtrain,ytrain)

#predict and evaluate
y_pred = GB_R.predict(xtest)
score = GB_R.score(xtest,ytest)
mse = mean_squared_error(ytest,y_pred)
rmse = np.sqrt(mse)


print("RMSE: %.4f" % rmse)
print("size of prediction: ", len(y_pred))
print("prediction: \n", y_pred)
print("test score: {0:.4f}\n".format(score))



feature_names = xtrain.columns

#XGBoost Regressor
xgb_R = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgbscore = cross_val_score(xgb_R,xtrain,ytrain,cv = kf,scoring = 'neg_mean_squared_error')
RMSE_for_xgb_R = np.sqrt(-xgbscore.mean())
xgb_R.fit(xtrain,ytrain)

#predict and evaluate
y_pred = xgb_R.predict(xtest)
score = xgb_R.score(xtest,ytest)
mse = mean_squared_error(ytest,y_pred)
rmse = np.sqrt(mse)

feature_names = xtrain.columns



print("RMSE: %.4f" % rmse)
print("size of prediction: ", len(y_pred))
print("prediction: \n", y_pred)
print("test score: {0:.4f}\n".format(score))

"""Optimization

Gradient boost
"""

params = {'n_estimators': 10000, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'squared_error'}
optimized_GB = GradientBoostingRegressor(**params)
optimized_GB.fit(xtrain,ytrain)
score = optimized_GB.score(xtest,ytest)
y_pred = optimized_GB.predict(xtest)
mse = mean_squared_error(ytest,y_pred)
rmse = np.sqrt(mse)

print("RMSE: %.4f" % rmse)
print("size of prediction: ", len(y_pred))
print("prediction: \n", y_pred)
print("test score: {0:.4f}\n".format(score))

feature_names = xtrain.columns

"""optimized XGboost Regressor"""

params = {'n_estimators': 10000, 'max_depth': 5, 'subsample': 0.8,
          'learning_rate': 0.01}

optimized_xgb = xgb.XGBRegressor(**params)
optimized_xgb.fit(xtrain,ytrain)

score = optimized_xgb.score(xtest,ytest)
y_pred = optimized_xgb.predict(xtest)
mse = mean_squared_error(ytest,y_pred)
rmse = np.sqrt(mse)

feature_names = xtrain.columns


print("RMSE: %.4f" % rmse)
print("size of prediction: ", len(y_pred))
print("prediction: \n", y_pred)
print("test score: {0:.4f}\n".format(score))

"""Optimized Random Forest Regressor"""

params = {
    'n_estimators': 100,
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'auto',
    'bootstrap': True,
    'random_state': 42
}

# Instantiate the model with the defined hyperparameters
optimized_rf = RandomForestRegressor(**params)

# Train the model
optimized_rf.fit(xtrain, ytrain)

# Predict and evaluate
y_pred = optimized_rf.predict(xtest)
score = optimized_rf.score(xtest, ytest)
mse = mean_squared_error(ytest, y_pred)
rmse = np.sqrt(mse)

print("RMSE: %.4f" % rmse)
print("size of prediction: ", len(y_pred))
print("prediction: \n", y_pred)
print("test score: {0:.4f}\n".format(score))

feature_names = xtrain.columns


#param_grid = {
#    'n_estimators': [100, 200, 500],
#    'max_depth': [None, 5, 10, 20],
#   'min_samples_split': [2, 5, 10],
#  'min_samples_leaf': [1, 2, 4],
#    'max_features': [1.0, 'sqrt', 'log2'],
#    'bootstrap': [True, False],
#    'random_state': [42,40]
#}

# Instantiate the model
#rf = RandomForestRegressor(random_state=42)

# Instantiate the grid search
#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=1, verbose=2, scoring='neg_mean_squared_error')

# Fit the grid search to the data
#grid_search.fit(xtrain, ytrain)

# Get the best parameters
#best_params = grid_search.best_params_
#print("Best parameters found: ", best_params)

# Train the final model with the best parameters
#optimized_rf = RandomForestRegressor(**best_params)
#optimized_rf.fit(xtrain, ytrain)

# Predict and evaluate
#y_pred = optimized_rf.predict(xtest)
#score = optimized_rf.score(xtest, ytest)
#mse = mean_squared_error(ytest, y_pred)
#rmse = np.sqrt(mse)

#print("RMSE: %.4f" % rmse)
#print("size of prediction: ", len(y_pred))
#print("prediction: \n", y_pred)
#print("test score: {0:.4f}\n".format(score))

"""Ensembling"""

ensemble = VotingRegressor(estimators=[('rf', optimized_rf), ('gb', optimized_GB), ('xgb', optimized_xgb)])


ensemble.fit(xtrain,ytrain)

y_pred = ensemble.predict(xtest)
score = ensemble.score(xtest,ytest)
mse = mean_squared_error(ytest,y_pred)
rmse = np.sqrt(mse)

feature_names = xtrain.columns


print("RMSE: %.4f" % rmse)
print("size of prediction: ", len(y_pred))
print("prediction: \n", y_pred)
print("test score: {0:.4f}\n".format(score))

def preprocessing(data,feature_names):
  # Drop columns with more than 30% missing values
  threshold = 0.3*len(data)
  data = data.dropna(thresh=threshold, axis=1)

  relevant_columns = ['overall','movement_reactions','potential','passing','wage_eur','mentality_composure','value_eur','dribbling','attacking_short_passing','mentality_vision','international_reputation']
  data = data[relevant_columns]

  imp = IterativeImputer(max_iter = 10, random_state = 0)
  data = pd.DataFrame(np.round(imp.fit_transform(data)),columns = data.columns)
  #combine both numeric and non-numeric dataframes
  X = data
  #fill missing values with mean
  X = X.fillna(X.mean())


# Subset data
  X = X[relevant_columns]
  y = X["overall"]
  x = X.drop("overall", axis=1)
  X = X[feature_names]

  # Scale the features
  scale = StandardScaler()
  scaled = scale.fit_transform(x)
  x = pd.DataFrame(scaled, columns=x.columns)

  X = pd.concat([x, y.reset_index(drop=True)], axis=1)

  return X



data_22 = preprocessing(players22df,feature_names)

X_22 = data_22.drop(columns=['overall'])
y_22 = data_22['overall']




X_22.head()

score = ensemble.score(X_22,y_22)

y_pred = ensemble.predict(X_22)
mse = mean_squared_error(y_22,y_pred)
rmse = np.sqrt(mse)
print("RMSE: %.4f" % np.sqrt(mse))
print("size of prediction: ", len(ensemble.predict(X_22)))
print("prediction: \n", ensemble.predict(X_22))
print("test score: {0:.4f}\n".format(score))

score = optimized_GB.score(X_22, y_22)


# Make predictions using the ensemble model
y_pred_22 = optimized_GB.predict(X_22)

# Calculate RMSE (Root Mean Squared Error)
mse = mean_squared_error(y_22, y_pred_22)

print("MSE: %.4f" % np.sqrt(mse))
print("size of prediction: ", len(optimized_GB.predict(X_22)))
print("prediction: \n", optimized_GB.predict(X_22))
print("test score: {0:.4f}\n".format(score))

score = optimized_rf.score(X_22, y_22)


# Make predictions using the ensemble model
y_pred_22 = optimized_rf.predict(X_22)

# Calculate RMSE (Root Mean Squared Error)
mse = mean_squared_error(y_22, y_pred_22)

print("MSE: %.4f" % np.sqrt(mse))
print("size of prediction: ", len(y_pred_22))
print("prediction: \n", y_pred_22)
print("test score: {0:.4f}\n".format(score))

score = optimized_xgb.score(X_22, y_22)


# Make predictions using the ensemble model
y_pred_22 = optimized_xgb.predict(X_22)

# Calculate RMSE (Root Mean Squared Error)
mse = mean_squared_error(y_22, y_pred_22)

print("MSE: %.4f" % np.sqrt(mse))
print("size of prediction: ", len(y_pred_22))
print("prediction: \n", y_pred_22)
print("test score: {0:.4f}\n".format(score))

import joblib

# Save the model to a file
joblib.dump(ensemble, 'ensemble_model.pkl')
