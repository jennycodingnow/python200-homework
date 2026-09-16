
# https://youtu.be/dAtnjL6Ad8A

# ================================================
# Part 2: Project — Full ETL Pipeline
# ================================================

import os
import json
import pandas as pd
import joblib
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
import requests
from prefect import flow, task

# ------------------------------------------------
# Extract Task
# ------------------------------------------------

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url:
    raise ValueError("Supabase URL is missing.")

if not supabase_key:
    raise ValueError("Supabase API key is missing.")

supabase = create_client(supabase_url, supabase_key)
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
FEATURES = metadata["features"]

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

CITY = "San Francisco, America"
TIMEZONE = "America/Los_Angeles"
LATITUDE = 37.77
LONGITUDE = -122.43
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"


@task(name="Extract 2023 weather", retries=2, retry_delay_seconds=10)
def extract() -> list:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": FEATURES,
        "timezone": TIMEZONE,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "daily" not in data:
        raise ValueError("Open-Meteo response missing 'daily' data.")

    missing = [f for f in FEATURES if f not in data["daily"]]
    if missing:
        raise ValueError(f"Missing daily features: {missing}")
    

    print("Open-Meteo 2023 weather data received successfully.")
    print(f"City: {CITY}")
    print(f"Latitude: {data['latitude']}")
    print(f"Longitude: {data['longitude']}")
    print(f"Timezone: {data['timezone']}")
    print(f"Date range: {data['daily']['time'][0]} to {data['daily']['time'][-1]}")
    print(f"Number of daily records: {len(data['daily']['time'])}")
    print(f"Daily variables: {FEATURES}")

    daily = data["daily"]

    records = []
    for i in range(len(daily["time"])):
        record = {
            "date": daily["time"][i],
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum": daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
        }

        records.append(record)

    print(f"Extracted {len(records)} daily records from Open-Meteo")
    return records


@task(name="Upsert data to weather_raw", retries=2, retry_delay_seconds=5)
def load_raw(records: list) -> None:
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows into weather_raw")


@task(name="Transform data")
def transform(raw_records: list) -> list:
    enriched_rows = (
        supabase.table("weather_enriched")
        .select("date")
        .execute()
        .data
    )
    already_enriched = {row["date"] for row in enriched_rows}

    to_process = [r for r in raw_records if r["date"] not in already_enriched]

    skip_count = len(raw_records) - len(to_process)
    print(
        f"Records to transform: {len(to_process)} "
        f"(skipping {skip_count} already enriched)"
    )

    if not to_process:
        print("All records already enriched — nothing to do.")
        return []

    # --- ML classify ---
    clf = joblib.load("models/weather_classifier.pkl")
    df  = pd.DataFrame(to_process)

    missing_features = [feature for feature in FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    X = df[FEATURES]

    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)[:, 1]

    print(f"ML classification completed. Good days for running: {int(predictions.sum())} / {len(predictions)}")

    enrichment_records = []
    for i in range(len(to_process)):
        record = {
            "date": to_process[i]["date"],
            "good_for_running": bool(predictions[i]),
            "confidence": round(float(probabilities[i]), 4),
            "llm_summary": None,
        }
        enrichment_records.append(record)


    # --- LLM enrich ---
    for i, record in enumerate(enrichment_records):
        raw_row = to_process[i]
        prediction_text = "good for running" if record["good_for_running"] else "not good for running"
        user_message = (
            f"Date: {raw_row['date']}\n"
            f"High: {raw_row['temperature_2m_max']}°C, Low: {raw_row['temperature_2m_min']}°C\n"
            f"Precipitation: {raw_row['precipitation_sum']} mm\n"
            f"Max wind speed: {raw_row['wind_speed_10m_max']} km/h\n"
            f"Model prediction: {prediction_text} (confidence: {record['confidence']:.0%})"
        )
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=100,
                timeout=30.0,
            )
            record["llm_summary"] = response.choices[0].message.content.strip() or "Recommendation unavailable."
        except Exception as e:
            print(f"  LLM error on {record['date']}: {e}")
            record["llm_summary"] = "Recommendation unavailable."

        if (i + 1) % 50 == 0:
            print(f"  LLM enriched {i + 1} / {len(enrichment_records)} records")

    print(f"Transform complete: {len(enrichment_records)} records enriched")
    return enrichment_records


@task(name="Upsert enriched data to weather_enriched", retries=2, retry_delay_seconds=5)
def load_enriched(enrichment_records: list) -> None:
    if not enrichment_records:
        print("No new enrichment records to load.")
        return

    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )
    print(
    f"Successfully upserted {len(response.data)} "
    f"enrichment records into weather_enriched."
    )


@flow(name="Weather ETL Pipeline", log_prints=True)
def etl_pipeline():
    raw_records = extract()
    load_raw(raw_records)
    enrichment_records = transform(raw_records)
    load_enriched(enrichment_records)
    print("Pipeline complete.")

if __name__ == "__main__":
    etl_pipeline()
