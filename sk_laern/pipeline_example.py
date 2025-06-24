import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Sample data
data = pd.DataFrame({
    'color': ['red', 'green', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'XL', 'M', 'S'],
    'weight': [1.2, 2.4, 2.1, 3.5, 2.2, 1.9],
    'label': [0, 1, 0, 1, 0, 1]
})

X = data.drop('label', axis=1)
y = data['label']

categorical_cols = ['color', 'size']
numeric_cols = ['weight']

# Preprocessor for categorical + numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

# Create pipeline with preprocessing and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define parameter grid for tuning
param_grid = {
    'classifier__n_estimators': [10, 50, 100],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.1, 0.01]
}

# Setup GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)

# Run grid search
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best parameters:", grid_search.best_params_)

# Evaluate best model on test set
y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}")
