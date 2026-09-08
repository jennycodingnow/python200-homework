# https://youtu.be/Q1iaSCRPT8M

# ================================================
# Part 2: Project — The Double-Transform Pipeline
# ================================================

import os
import json
import pandas as pd
import joblib
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# ------------------------------------------------
# Step 1: Incremental Read
# ------------------------------------------------

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Load metadata
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]

response = supabase.table("weather_raw").select("*").execute()
raw_rows = response.data

enriched_response = supabase.table("weather_enriched").select("date").execute()
already_done = {row["date"] for row in enriched_response.data}

to_classify = [row for row in raw_rows if row["date"] not in already_done]

# Summary
print(f"Raw records: {len(raw_rows)}")
print(f"Already enriched: {len(already_done)}")
print(f"Will classify: {len(to_classify)}")

if not to_classify:
    print("All records already enriched, nothing to do.")
    exit()

# ------------------------------------------------
# Step 2: ML Transform
# ------------------------------------------------

clf = joblib.load("models/weather_classifier.pkl")
df = pd.DataFrame(to_classify)
X = df[FEATURES]
# ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max']

predictions   = clf.predict(X)
probabilities = clf.predict_proba(X)[:, 1]

enrichment_records = []
for i, row in enumerate(to_classify):
    enrichment_records.append({
        "date":             row["date"],
        "good_for_running": bool(predictions[i]),
        "confidence":       round(float(probabilities[i]), 4),
    })


print(f"Predictions Summary: ")
print(f"Good days predicted: {predictions.sum()} / {len(predictions)}")
print(f"Confidence range: {probabilities.min():.2f} - {probabilities.max():.2f}")

# ------------------------------------------------
# Step 3: LLM Transform
# ------------------------------------------------

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

def create_user_message(row, good_for_running, confidence):
    prediction_text = "good for running" if good_for_running else "not ideal for running"
    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°C, Low: {row['temperature_2m_min']}°C\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
    )


records_by_date = {
    r["date"]: r
    for r in to_classify
}

for i, record in enumerate(enrichment_records):
    raw_row = records_by_date.get(record["date"])

    if raw_row is None:
        print(f"  No matching data for {record['date']}")
        record["llm_summary"] = "Recommendation unavailable."
        continue

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": create_user_message(
                    raw_row, record["good_for_running"], record["confidence"]
                )},
            ],
            max_tokens=100,
        )
        raw_summary  = response.choices[0].message.content.strip()
        record["llm_summary"] = raw_summary or "Recommendation unavailable."
    except Exception as e:
        print(f"  API error on {record['date']}: {e}")
        record["llm_summary"] = "Recommendation unavailable."

    # Progress print every 50 records    
    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1} / {len(enrichment_records)}")

# ------------------------------------------------
# Step 4: Load
# ------------------------------------------------

db_response = (
    supabase.table("weather_enriched")
    .upsert(enrichment_records, on_conflict="date")
    .execute()
)
print(f"Upserted {len(db_response.data)} rows into weather_enriched")

# ------------------------------------------------
# Step 5: Verify
# ------------------------------------------------

# Get total row count
row_count = supabase.table("weather_enriched").select("date", count="exact").execute()
print(f"Total rows in weather_enriched: {row_count.count}")

# Get 5 sample of rows
sample = supabase.table("weather_enriched").select("*").limit(5).execute()

print("\nFive sample rows:")
for row in sample.data:
    print(f"\n{row['date']} | good={row['good_for_running']} | conf={row['confidence']:.2f}")
    print(f"  {row['llm_summary']}")
    print()

# Get number of good-for-running days
good_count = (
    supabase.table("weather_enriched")
    .select("date", count="exact")
    .eq("good_for_running", True)
    .execute()
)
print(f"Good-for-running days in weather_enriched: {good_count.count}")

# Questions:
# Do they accurately reflect the weather features and the model's prediction? 
# Pick one you think is particularly good and one that seems off — what might have caused the weaker one?

# Answer: 
# The LLM summaries generally reflect the weather features and the model's predictions accurately.
# One particularly good summary clearly connected cold or freezing temperatures with the prediction
# good_for_running=False. The confidence score was also below 0.5, which was consistent with the 
# LLM's recommendation to avoid running.
# One weaker summary was less informative because, in some cases, it provided the exact temperature, 
# while in others it only described the temperature as "cold" or "freezing" without giving the actual value. 
# This may have made the summary less precise and harder to evaluate against the model's input features.

# ------------------------------------------------
# Step 6: Reflect
# ------------------------------------------------

# Questions:
# Add a comment block (at least 5-6 sentences) addressing:

# 1) The ML classifier was trained on Charlotte, NC data. If you loaded weather data for 
# a different city in Week 9, do you expect the classifier's predictions to be accurate? Why or why not?

# Answer:
# The ML classifier may not be accurate for a different city Charlotte, NC and Seoul, South Korea because 
# it was trained on data specific to Charlotte, NC. The weather patterns, temperature ranges, and precipitation levels can vary significantly 
# between cities, which may lead to misclassifications when applying the model to a new location. 
# Seoul has substantially different seasonal patterns, humidity, precipitation, temperature ranges, and 
# monsoon behavior than Charlotte, NC. Therefore, the model may not generalize well to the new city, 
# and its predictions could be less reliable.


# 2) The LLM recommendations are generated from the model's prediction and the weather features. 
# Does the LLM have any ability to "override" the classifier, or is it purely additive? 
# What are the implications of that?

# Answer:
# The LLM does not have the ability to override the classifier's predictions; it is purely additive. 
# The LLM generates recommendations based on the classifier's output because we passed it to the LLM 
# along with the weather features provided. This means that if the classifier makes an incorrect prediction, 
# the LLM will still generate a recommendation based on that potentially flawed prediction. 
# The implication is that the overall accuracy of the recommendations is very dependent 
# on the classifier's performance. If the classifier is inaccurate, the LLM's recommendations may also be misleading, which could 
# affect user trust and decision-making. Therefore, it is important to ensure that the classifier is 
# well-trained and performs well on the data it will be applied to. 


# 3) If you ran this pipeline on 50,000 records instead of 365, what would be your main 
# concern: cost, latency, or something else? How would you address it?

# Answer:
# If we ran this pipeline on 50,000 records instead of 365, my main concern would be latency. 
# Each API call takes roughly 0.5 to 2 seconds. Calling an LLM sequentially for 10,000 
# records takes two to six hours of wall-clock time according to the lesson materials. For 50,000 records,
# it would take five times that amount, or 10 to 30 hours.
# To address this, we could implement batching, concurrency, or OpenAI's Batch API 
# (which processes requests asynchronously at reduced cost) to reduce the overall time taken according 
# to the lesson materials.