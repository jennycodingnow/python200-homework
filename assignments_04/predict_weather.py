
# --- Task 1: Load and Verify ---

import joblib
import json
import pandas as pd

loaded_clf_weather = joblib.load("models/weather_classifier.pkl")


with open("models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)

print("Metadata:")
print(f"City: {metadata['city']}")
print(f"Features: {metadata['features']}")
print(f"Test AUC: {metadata['test_auc']}")


# --- Task 2: Predict on New Data ---
new_days = pd.DataFrame({
    "temperature_2m_max": [18.0, 2.0, 28.0],
    "temperature_2m_min": [10.0, -2.0, 19.0],
    "precipitation_sum":  [0.0,   0.0, 12.0],
    "wind_speed_10m_max": [18.0,  8.0, 35.0],
})