
# ================================================
# Part 2: Project — Extract + Load Pipeline
# ================================================
# https://youtu.be/-QtR8ufFUGk


import requests
import os
from dotenv import load_dotenv
from datetime import date
from supabase import create_client

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url:
    raise ValueError("Supabase URL is missing.")

if not supabase_key:
    raise ValueError("Supabase API key is missing.")

supabase = create_client(supabase_url, supabase_key)

LATITUDE = 37.566
LONGITUDE = 126.9784

# ------------------------------------------------
# Step 1: Extract
# ------------------------------------------------

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
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

data = response.json()

print("Open-Meteo response received successfully.")
print(f"Latitude: {data['latitude']}")
print(f"Longitude: {data['longitude']}")
print(f"Timezone: {data['timezone']}")
print(f"Number of daily records: {len(data['daily']['time'])}")
print(f"Variables returned: {list(data['daily'].keys())}")

daily = data["daily"]

# ------------------------------------------------
# Step 2: Transform
# ------------------------------------------------


records = [
    {
        "date":               daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum":  daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
    }
    for i in range(len(daily["time"]))
]

print(f"Expected 365 records; received {len(records)} records.")
print("First record:", records[0])
print("Last record:", records[-1])

# How many records do you expect for a full year, and how many did you get? If the numbers differ, 
# what might explain the discrepancy?
#
# Answer: 
# I expect 365 records for a full year (2023) because 2023 was not a leap year, and I got 365 records. If the numbers differed, 
# it could be due to missing dates in the API response or an issue with the requested date range.

# ------------------------------------------------
# Step 3: Load
# ------------------------------------------------

response = (
    supabase.table("weather_raw")
    .upsert(records, on_conflict="date")
    .execute()
)

print(f"Upserted {len(response.data)} weather records into weather_raw.")

# What does this tell you about idempotency?
#
# Answer:
# This tells me that the upsert operation is idempotent, meaning that if 
# I run the same operation multiple times with the same data, it will not 
# create duplicate records. Instead, it will update existing records if they 
# already exist, ensuring that the database remains consistent and avoids duplicates.

# ------------------------------------------------
# Step 4: Verify
# ------------------------------------------------

# After upserting, run a verification query that:

# Prints the total number of rows in weather_raw
# Prints the earliest and latest dates in the table
# Prints the row for 2023-07-04 (or the nearest date if that date is missing)

count_response = (supabase.table("weather_raw").select("date", count="exact").order("date").execute())
print(f"Total rows in weather_raw: {count_response.count}")

rows = count_response.data

if rows:    
    print(f"Earliest date: {rows[0]['date']}")
    print(f"Latest date: {rows[-1]['date']}")

    july_4_response = (
        supabase
        .table("weather_raw")
        .select("*")
        .eq("date", "2023-07-04")
        .execute()
    )

    if july_4_response.data:
        print(f"Row for 2023-07-04:") 
        print(july_4_response.data)
    else:
        print("No data found for 2023-07-04. Fetching nearest date...")
        previous_response = (
            supabase
            .table("weather_raw")
            .select("*")
            .lt("date", "2023-07-04")
            .order("date", desc=True)
            .limit(1)
            .execute()
            )

        next_response = (
            supabase
            .table("weather_raw")
            .select("*")
            .gt("date", "2023-07-04")
            .order("date", desc=False)
            .limit(1)
            .execute()
            )
        

        previous_row = previous_response.data[0] if previous_response.data else None
        next_row = next_response.data[0] if next_response.data else None

        if previous_row and next_row:

            target_date = date.fromisoformat("2023-07-04")
            previous_date = date.fromisoformat(previous_row["date"])
            next_date = date.fromisoformat(next_row["date"])

            if (target_date - previous_date) <= (next_date - target_date):
                print("Nearest date row:")
                print(previous_row)
            else:
                print("Nearest date row:")
                print(next_row)

        elif previous_row:
            print("Nearest date row:")
            print(previous_row)

        elif next_row:
            print("Nearest date row:")
            print(next_row)

        else:
            print("No data found for dates around 2023-07-04.")

else:
    print("weather_raw table is empty.")