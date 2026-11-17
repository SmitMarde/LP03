
# Uber Fare Prediction
# Run this file directly in VS Code

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

FILE_PATH = "uber.csv"

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"'{FILE_PATH}' not found. "
        "Place uber.csv in the same folder as this script."
    )

data = pd.read_csv(FILE_PATH)
df = data.copy()

print("\n========== DATASET OVERVIEW ==========")
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())


# --------------------------------------------------
# 2. Data Cleaning
# --------------------------------------------------

# Remove unnecessary ID columns
df.drop(columns=["Unnamed: 0", "key"], errors="ignore", inplace=True)

# Convert pickup_datetime into datetime format
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"],
    errors="coerce",
    utc=True
)

# Remove missing values
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Keep only valid fares
df = df[df["fare_amount"] > 0]

# Remove invalid passenger counts
df = df[df["passenger_count"].between(1, 6)]

# Keep geographically plausible New York City-area coordinates
df = df[
    df["pickup_longitude"].between(-75, -72)
    & df["dropoff_longitude"].between(-75, -72)
    & df["pickup_latitude"].between(40, 42)
    & df["dropoff_latitude"].between(40, 42)
].copy()

# Remove fare outliers using the 1st and 99th percentiles
q_low = df["fare_amount"].quantile(0.01)
q_high = df["fare_amount"].quantile(0.99)

df = df[
    df["fare_amount"].between(q_low, q_high)
].copy()

if df.empty:
    raise ValueError(
        "No records remain after cleaning. "
        "Check your dataset and filtering conditions."
    )

print("\n========== CLEANED DATA ==========")
print("Rows remaining:", len(df))
print("\nMissing values:")
print(df.isnull().sum())

print("\nFare amount after cleaning:")
print(df["fare_amount"].describe())


# --------------------------------------------------
# 3. Feature Engineering
# --------------------------------------------------

# Extract useful information from pickup_datetime
df["pickup_hour"] = df["pickup_datetime"].dt.hour
df["pickup_day"] = df["pickup_datetime"].dt.day
df["pickup_month"] = df["pickup_datetime"].dt.month
df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek

# Calculate distance between pickup and dropoff
# using the Haversine formula

def calculate_distance(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


df["distance_km"] = calculate_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)

# Remove impossible zero-distance trips
# (This is a modeling choice; some real trips may be short.)
df = df[df["distance_km"] > 0].copy()

if df.empty:
    raise ValueError("No valid trips remain after distance calculation.")

print("\n========== FEATURE ENGINEERING ==========")
print(df[
    [
        "fare_amount",
        "distance_km",
        "pickup_hour",
        "pickup_day",
        "pickup_month",
        "pickup_dayofweek",
        "passenger_count"
    ]
].head())


# --------------------------------------------------
# 4. Correlation Analysis
# --------------------------------------------------

numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation.round(3))

plt.figure(figsize=(10, 7))
plt.imshow(correlation, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)
plt.yticks(range(len(correlation.columns)), correlation.columns)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 5. Visualize Fare Outliers
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.boxplot(df["fare_amount"])
plt.title("Box Plot of Cleaned Fare Amount")
plt.ylabel("Fare Amount")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 6. Prepare Features and Target
# --------------------------------------------------

# X contains predictor variables
# y contains the fare amount to predict

feature_columns = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_hour",
    "pickup_day",
    "pickup_month",
    "pickup_dayofweek",
    "distance_km"
]

X = df[feature_columns]
y = df["fare_amount"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=1
)

print("\n========== TRAINING DATA ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X.shape[1])


# --------------------------------------------------
# 7. Linear Regression
# --------------------------------------------------

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_rmse = np.sqrt(
    mean_squared_error(y_test, lr_predictions)
)

lr_r2 = r2_score(y_test, lr_predictions)

print("\n========== LINEAR REGRESSION ==========")
print(f"RMSE: {lr_rmse:.4f}")
print(f"R² Score: {lr_r2:.4f}")


# --------------------------------------------------
# 8. Random Forest Regressor
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=101,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_predictions)
)

rf_r2 = r2_score(y_test, rf_predictions)

print("\n========== RANDOM FOREST ==========")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R² Score: {rf_r2:.4f}")


# --------------------------------------------------
# 9. Compare Model Performance
# --------------------------------------------------

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "RMSE": [lr_rmse, rf_rmse],
    "R2 Score": [lr_r2, rf_r2]
})

print("\n========== MODEL COMPARISON ==========")
print(results.to_string(index=False))

best_model_name = (
    "Linear Regression"
    if lr_rmse < rf_rmse
    else "Random Forest"
)

print("\nModel with lower RMSE:", best_model_name)


# --------------------------------------------------
# 10. Actual vs Predicted Fare Plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    lr_predictions,
    alpha=0.4,
    label="Linear Regression"
)

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.4,
    label="Random Forest"
)

min_fare = min(y_test.min(), lr_predictions.min(), rf_predictions.min())
max_fare = max(y_test.max(), lr_predictions.max(), rf_predictions.max())

plt.plot(
    [min_fare, max_fare],
    [min_fare, max_fare],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Uber Fares")
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. Feature Importance (Random Forest)
# --------------------------------------------------

importance = pd.Series(
    rf_model.feature_importances_,
    index=feature_columns
).sort_values(ascending=True)

plt.figure(figsize=(9, 6))
importance.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

print("\n========== PROJECT COMPLETED ==========")
