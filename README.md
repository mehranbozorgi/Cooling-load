# Cooling-load
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from bayes_opt import BayesianOptimization
import joblib

# Define evaluation metrics
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def rae(y_true, y_pred):
    return np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true - np.mean(y_true)))

def mase(y_true, y_pred, y_train):
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

def mbd(y_true, y_pred):
    return np.mean(y_pred - y_true)

# Load the cleaned data
cleaned_data = pd.read_csv('...\\cleaned_data.csv')

# Trim column names to remove leading/trailing whitespace
cleaned_data.columns = cleaned_data.columns.str.strip()

# Check for NaN values
print(cleaned_data.isnull().sum())

# Handle NaN values by filling them with the mean of each column
cleaned_data.fillna(cleaned_data.mean(), inplace=True)

# Ensure all values in 'Internal loads' are positive before applying log transformation
cleaned_data['Internal loads'] = cleaned_data['Internal loads'].apply(lambda x: x if x > 0 else 1e-9)

# Feature Engineering: Create interaction terms and polynomial features
cleaned_data['T_amb_T_in'] = cleaned_data['T_amb'] * cleaned_data['T_in']
cleaned_data['T_amb_Internal_loads'] = cleaned_data['T_amb'] * cleaned_data['Internal loads']
cleaned_data['T_in_Internal_loads'] = cleaned_data['T_in'] * cleaned_data['Internal loads']
cleaned_data['T_amb_squared'] = cleaned_data['T_amb'] ** 2
cleaned_data['T_in_squared'] = cleaned_data['T_in'] ** 2
cleaned_data['log_Internal_loads'] = np.log1p(cleaned_data['Internal loads'])  # Safe log transformation

# Add polynomial features for other relevant variables
cleaned_data['RH_in_squared'] = cleaned_data['RH_in'] ** 2
cleaned_data['RH_amb_squared'] = cleaned_data['RH_amb'] ** 2
cleaned_data['G_squared'] = cleaned_data['G'] ** 2
cleaned_data['u_wind_squared'] = cleaned_data['u_wind'] ** 2
cleaned_data['u_direction_squared'] = cleaned_data['u_direction'] ** 2
cleaned_data['Heat_Transfer_squared'] = cleaned_data['Heat Transfer'] ** 2
cleaned_data['Occupants_squared'] = cleaned_data['Occupants'] ** 2
cleaned_data['Area_squared'] = cleaned_data['Area'] ** 2

# Experiment with additional transformations
cleaned_data['T_amb_cubed'] = cleaned_data['T_amb'] ** 3
cleaned_data['T_in_cubed'] = cleaned_data['T_in'] ** 3

# Apply sqrt transformation safely
cleaned_data['sqrt_T_amb'] = cleaned_data['T_amb'].apply(lambda x: np.sqrt(x) if x >= 0 else 0)
cleaned_data['sqrt_T_in'] = cleaned_data['T_in'].apply(lambda x: np.sqrt(x) if x >= 0 else 0)

# Separate features and target
X = cleaned_data.drop(columns=['Cooling Load']).values
y = cleaned_data['Cooling Load'].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the objective function for Bayesian Optimization
def xgb_evaluate(n_estimators, learning_rate, max_depth, min_child_weight, gamma, subsample, colsample_bytree):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    )
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    
    for train_index, val_index in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
    
    return -np.mean(mse_scores)

# Set the range for hyperparameters
params = {
    'n_estimators': (100, 1000),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 10),
    'min_child_weight': (1, 10),
    'gamma': (0, 1),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1)
}

# Perform Bayesian Optimization with increased iterations
optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds=params,
    random_state=42,
    verbose=2
)
optimizer.maximize(init_points=15, n_iter=100)

# Get the best parameters
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])

# Train the final model with the best hyperparameters
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []
mae_scores = []
r2_scores = []
rmse_scores = []
smape_scores = []
rae_scores = []
mase_scores = []
mbd_scores = []
y_true_all = []
y_pred_all = []

for train_index, val_index in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = xgb.XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mse)
    smape_value = smape(y_val, y_pred)
    rae_value = rae(y_val, y_pred)
    mase_value = mase(y_val, y_pred, y_train)
    mbd_value = mbd(y_val, y_pred)
    
    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    smape_scores.append(smape_value)
    rae_scores.append(rae_value)
    mase_scores.append(mase_value)
    mbd_scores.append(mbd_value)
    
    y_true_all.extend(y_val)
    y_pred_all.extend(y_pred)

print(f'Final Mean MSE: {np.mean(mse_scores)}, Std MSE: {np.std(mse_scores)}')
print(f'Final Mean MAE: {np.mean(mae_scores)}, Std MAE: {np.std(mae_scores)}')
print(f'Final Mean R2: {np.mean(r2_scores)}, Std R2: {np.std(r2_scores)}')
print(f'Final Mean RMSE: {np.mean(rmse_scores)}, Std RMSE: {np.std(rmse_scores)}')
print(f'Final Mean SMAPE: {np.mean(smape_scores)}, Std SMAPE: {np.std(smape_scores)}')
print(f'Final Mean RAE: {np.mean(rae_scores)}, Std RAE: {np.std(rae_scores)}')
print(f'Final Mean MASE: {np.mean(mase_scores)}, Std MASE: {np.std(mase_scores)}')
print(f'Final Mean MBD: {np.mean(mbd_scores)}, Std MBD: {np.std(mbd_scores)}')

# Calculate SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X_scaled, check_additivity=False)

# Plot SHAP summary
shap.summary_plot(shap_values, X_scaled, feature_names=cleaned_data.drop(columns=['Cooling Load']).columns)

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_true_all, y_pred_all, alpha=0.3)
plt.plot([min(y_true_all), max(y_true_all)], [min(y_true_all), max(y_true_all)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.show()

# Residual plot
residuals = np.array(y_true_all) - np.array(y_pred_all)
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Count')
plt.title('Residual Histogram')
plt.show()

# Q-Q plot
import scipy.stats as stats
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
train_scores_mean = -train_scores.mean(axis=1)
val_scores_mean = -val_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, val_scores_mean, label='Validation error')
plt.xlabel('Training set size')
plt.ylabel('MSE')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Feature Importance Plot
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='weight', max_num_features=20)
plt.title('Feature Importance')
plt.show()

# Save the final model
joblib.dump(model, 'cooling_load_model_xgboost_bayesian.pkl')

