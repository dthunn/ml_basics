import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from scipy.stats import randint, uniform

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

# Define parameter distributions for RandomizedSearchCV
param_dist = {
    'classifier__n_estimators': randint(10, 200),        # random integers from 10 to 199
    'classifier__max_depth': randint(3, 10),             # random integers from 3 to 9
    'classifier__learning_rate': uniform(0.01, 0.3)      # uniform distribution between 0.01 and 0.31
}

# Setup RandomizedSearchCV with 10 iterations and 3-fold CV
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    random_state=42,
    verbose=1
)

# Run random search
random_search.fit(X_train, y_train)

# Best parameters found
print("Best parameters:", random_search.best_params_)

# Evaluate best model on test set
y_pred = random_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}")


