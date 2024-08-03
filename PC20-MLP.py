import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


# Defining the MLP model
mlp = MLPRegressor()

# Defining the parameter grid
param_grid = {
    'hidden_layer_sizes': [(50,), (50, 50), (100, 50), (100,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'max_iter': [500, 1000, 1500],
}

# Parameter Search with GridSearchCV
random_state = 42
kf = KFold(n_splits=5, random_state=random_state, shuffle=True)
grid_search = GridSearchCV(mlp, param_grid, cv=kf, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Output optimal parameters and scoring
print("Optimal parameter:", grid_search.best_params_)
print("Best rating:", grid_search.best_score_)
# Model construction using optimal parameters
best_mlp = grid_search.best_estimator_
# training model
best_mlp.fit(X, y)

# Projected target data
y_pred = best_mlp.predict(X)

# Assessment of projected results
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

# Output of projected results and assessment indicators
print('Outputs:', y_pred)
print('R2 coefficient:', r2)
print('MSE:', mse)

# Save predictions
predicted_data = pd.DataFrame({'Predicted_result': y_pred})
predicted_data.to_csv('MLP_predicted_results.csv', index=False)

# Reading test set data
test_data = pd.read_csv('Valid_dataset.csv')
# Extracting test set feature data
X_test = test_data
# Predictions on the test set
y_test_pred = best_mlp.predict(X_test)
# Output prediction results
print('Test Set Outputs:', y_test_pred)
# Save predictions
predicted_data = pd.DataFrame({'Predicted_result': y_test_pred})
predicted_data.to_csv('MLP_test_predicted_results.csv', index=False)