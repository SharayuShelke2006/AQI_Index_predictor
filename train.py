
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("city_day.csv")

# Select input columns
features = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3"
]

target = "AQI"

# Remove rows where AQI is missing
data = data.dropna(subset=[target])

X = data[features]
y = data[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "SVR": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf"))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

best_model = None
best_score = -float("inf")
best_name = ""

# Train and compare
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\nModel:", name)
    print("R2 Score:", round(r2, 4))
    print("RMSE:", round(rmse, 2))

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

# Save best model
joblib.dump({
    "model": best_model,
    "features": features,
    "name": best_name
}, "model.pkl")

print("\nBest Model:", best_name)
print("Best R2 Score:", round(best_score, 4))
print("Model saved successfully!")
