import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import time

# Load dataset
df = pd.read_csv("../dataset/GLOBAL_WEATHER_DATA.csv")  # Replace with your actual dataset

# Input features and targets
X = df[['dew_point', 'apparent_temperature', 'precipitation', 'rain',
        'wind_direction', 'hour', 'month', 'city']]
y = df[['temperature', 'relative_humidity', 'wind_speed']]

# Preprocessing: scale numerical and encode categorical
numeric_features = ['dew_point', 'apparent_temperature', 'precipitation',
                    'rain', 'wind_direction', 'hour', 'month']
categorical_features = ['city']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Ridge regression inside a MultiOutput wrapper
ridge = Ridge(alpha=1.0)
multi_ridge = MultiOutputRegressor(ridge)

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', multi_ridge)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("[INFO] Starting model training...")
start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()
print(f"[INFO] Model training complete in {end_train - start_train:.2f} seconds.")

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"[INFO] MAE: {mae:.4f}")
print(f"[INFO] R2 Score: {r2:.4f}")

# Save the model (no compression for fast loading)
model_filename = "ridge_weather_model.joblib"
joblib.dump(model, model_filename, compress=0)
print(f"[INFO] Model saved as {model_filename}")
