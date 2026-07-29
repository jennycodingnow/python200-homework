import joblib
import json
import pandas as pd

# --- Task 1: Load and Verify ---

clf_weather = joblib.load("models/weather_classifier.pkl")

with open("models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)

print("Metadata:")
print(f"City: {metadata['city']}")
print(f"Features: {metadata['features']}")
print(f"Test AUC: {metadata['test_auc']}")

print(f"Classes: {clf_weather.classes_}")
print(f"Metadata: {metadata}")

# --- Task 2: Predict on New Data ---

new_days = pd.DataFrame({
    "temperature_2m_max": [20.2, 22.0, 16.3, 5.0, 30.0, 18.0],
    "temperature_2m_min": [2.5, 4.8, 2.9, -2.0, 22.0, 15.0],
    "precipitation_sum":  [0.0, 0.0, 0.0, 1.0, 0.0, 20.0],
    "wind_speed_10m_max": [14.6, 17.1, 16.0, 20.0, 10.0, 35.0],
})

print(metadata["features"])
print(new_days.columns.tolist())

predictions = clf_weather.predict(new_days)
print(f"predictions: {predictions}")
probabilities = clf_weather.predict_proba(new_days)[:, 1]


labels = {0: "skip", 1: "good"}

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Day {i+1}: {labels[pred]} ({prob:.2f} probability)")


# --- Task 3: Reflect ---

# My borderline case was Day 5, which was classified as skip with a 0.49 
# probability of being good for running. Although the model predicted skip, 
# the probability was very close to the 0.50 threshold, so the model was not 
# strongly confident. A day with a probability of 0.52 would also be borderline 
# because it is only slightly above the cutoff.

# The prediction script would fail if someone ran predict_weather.py before
# train_weather_classifier.py because the saved model file and metadata file
# would not exist yet. The joblib.load() step would raise an error because
# models/weather_classifier.pkl has not been created. I would make the error
# message more helpful by checking whether the model file exists before loading
# it and displaying a message explaining that the user needs to run
# train_weather_classifier.py first.

# To support daily production predictions, predict_weather.py would need to
# retrieve tomorrow's weather forecast automatically from a weather API instead
# of using manually created hypothetical days. The script would need to format
# the forecast data using the same feature names and order used during training,
# then pass the data into the saved pipeline to generate a prediction. In a
# production system, this script could also be scheduled to run automatically
# each day and save or send the prediction result to users.
