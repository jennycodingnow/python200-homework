import requests
import pandas as pd
import joblib
import json
import sklearn
import sys
import os

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)


os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Step 1: Fetch the Data ---

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 37.566,
    "longitude": 126.9784,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "Asia/Seoul",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)
print(f"Data fetched successfully. Number of records: {len(df)}")
print(f"Summary of the dataset for Seoul, South Korea weather:")
print(df.head())
print(df.describe())

# --- Step 2: Engineer Labels ---

def label_run_day(row):
    """Return 1 if conditions are good for an outdoor run, 0 otherwise."""
    temp_ok    = 7 <= row["temperature_2m_max"] <= 26   # 45–79°F
    above_freeze = row["temperature_2m_min"] >= 0        # above freezing at dawn
    dry        = row["precipitation_sum"] < 3.0          # light rain or less
    not_windy  = row["wind_speed_10m_max"] < 30          # under 30 km/h
    return int(temp_ok and above_freeze and dry and not_windy)

df["good_for_running"] = df.apply(label_run_day, axis=1)

print("Class distribution:")
print(df["good_for_running"].value_counts())
print(f"\nFraction of good days: {df['good_for_running'].mean():.2%}")

# Fraction of good days is reasonable given the climate of Seoul, South Korea. 
# The city experiences hot summers and cold snowing winters, with a significant amount of rainfall during the monsoon season. 
# Therefore, it is expected that not all days will be suitable for outdoor running.

# --- Step 3: Train and Tune ---

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[FEATURES]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])

param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print(f"Best C for Logistic Regression:      {grid_search.best_params_['clf__C']}")
print(f"Best CV AUC for Logistic Regression: {grid_search.best_score_:.3f}")

best_lr = grid_search.best_estimator_

y_pred  = best_lr.predict(X_test)
y_probs = best_lr.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_probs)

print(classification_report(y_test, y_pred))
print(f"Test AUC: {test_auc:.3f}")

fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_probs)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr_lr, tpr=tpr_lr).plot(ax=ax, name=f"Logistic Regression (AUC={test_auc:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
ax.set_title("ROC curve for the best estimator (Logistic Regression)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/weather_roc.png")
plt.show()
plt.close()

# --- Step 4: Reflect on Evaluation ---

# The model achieved a cross-validation AUC of 0.765 and a test AUC of 0.849, 
# both of which are well above the random baseline of 0.5. This indicates that 
# the model has good ability to distinguish between days that are good and not 
# good for running. This performance is about what I expected because temperature, 
# precipitation, and wind speed are strong indicators of running conditions. 
# Looking at the classification report, the model has much lower recall (0.25) 
# for the "good for running" class than for the "not good for running" class (0.92), 
# meaning false negatives are more common than false positives. In practice, this means 
# the app is more likely to miss some good running days than recommend running on a poor weather day. 
# I think this is an acceptable trade-off because recommending a run in unsafe or uncomfortable
# weather could negatively affect the user experience. For a real application, 
# I would keep the default threshold of 0.5 since I prefer a more conservative recommendation 
# that prioritizes safety over suggesting every possible running opportunity.

# --- Step 5: Save the Model ---

joblib.dump(best_lr, "models/weather_classifier.pkl")

metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "features": FEATURES,
    "best_params": grid_search.best_params_,
    "test_auc": round(test_auc, 4),
    "city": "Seoul, South Korea",
    "latitude": 37.566,
    "longitude": 126.9784,
    "label_thresholds": {
        "temperature_2m_max": "7 to 26°C",
        "temperature_2m_min": ">=0°C",
        "precipitation_sum": "<3 mm",
        "wind_speed_10m_max": "<30 km/h"
    }
}
with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model and metadata successfully saved.")
