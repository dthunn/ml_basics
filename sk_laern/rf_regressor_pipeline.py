import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform
import joblib

# --- Load dataset ---
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# --- Identify numeric and categorical features ---
# For California Housing, all features are numeric, but we include categorical logic for generalization
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# --- Preprocessing ---
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- Create pipeline ---
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# --- Hyperparameter search space ---
param_dist = {
    'regressor__n_estimators': randint(100, 500),
    'regressor__max_depth': randint(3, 20),
    'regressor__min_samples_split': randint(2, 20),
    'regressor__min_samples_leaf': randint(1, 10),
    'regressor__max_features': ['auto', 'sqrt', 'log2']
}

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- RandomizedSearchCV ---
random_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    verbose=1,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    random_state=42
)

# --- Train the model ---
random_search.fit(X_train, y_train)

# --- Best parameters ---
print("Best Hyperparameters:", random_search.best_params_)

# --- Evaluate on test set ---
y_pred = random_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")
print(f"Test R^2: {r2:.4f}")

# --- Feature importance ---
importances = random_search.best_estimator_.named_steps['regressor'].feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': numeric_features,
    'importance': importances
}).sort_values(by='importance', ascending=False)
print(feature_importance_df)

# --- Save the model for production ---
joblib.dump(random_search.best_estimator_, "california_rf_model.pkl")
