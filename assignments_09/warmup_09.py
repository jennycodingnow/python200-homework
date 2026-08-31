from datetime import datetime
import os
from dotenv import load_dotenv
from supabase import create_client


# ================================================
# Part 1: Warmup
# ================================================

# ------------------------------------------------
# Supabase Connection 
# ------------------------------------------------

# Q1

# what are the two pieces of information supabase-py needs to connect to your project?
# Where do you find them in the Supabase dashboard, and why should they never be hardcoded in a Python script?
#
# Answer: 
# The two pieces of information supabase-py needs to connect to your project are:
# 1. The URL of your Supabase project
# 2. The public API key (Publishable Key) for your Supabase project
# You find them in dashboard under the "Settings" section, specifically in the "API" tab. But also, 
# you can find the URL in the project settings and the API key (Publishable Key) near the your project name in the 
# copy drop down menu.
# They should never be hardcoded in a Python script because it can lead to security vulnerabilities,
# as anyone with access to the code could potentially misuse the credentials. Instead, 
# they should be stored in environment variables.


# Q2

def get_client():
    """
    Returns a Supabase client object.
    """
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url:
        raise ValueError("Supabase URL is missing.")

    if not supabase_key:
        raise ValueError("Supabase API key is missing.")

    supabase = create_client(supabase_url, supabase_key)

    return supabase

# Q3
# what is Row Level Security (RLS), and why did you disable it on your tables for this course? 
# In what kind of real-world application would you want to keep it enabled?
#
# Answer:
# Row Level Security (RLS) is a database or production feature that allows you to define fine-grained access 
# policies and controls directly inside the database. 
# I disabled it on my tables for this course to simplify the learning process and avoid the complexity of
# managing and maintaining access control rules. This allows my scripts to read and write freely without 
# configuring access policies. In a real-world application, you would want to keep it 
# enabled to ensure that users can only access the data they are authorized to see, which is important for 
# maintaining data privacy and compliance. For example, in a HR application, you would want to ensure that employees 
# can only access their own records and not those of other employees.


# ------------------------------------------------
# supabase-py CRUD
# ------------------------------------------------

# Q1

def insert_test_record(supabase):
    """
    Inserts a test weather record into weather_raw.
    """
    data = {
        "date": datetime.now().date().isoformat(),
        "temperature_2m_max": 25.5,
        "temperature_2m_min": 17.3,
        "precipitation_sum":  3.5,
        "wind_speed_10m_max": 18.7,
    }
    response = supabase.table("weather_raw").upsert(data).execute()
    return response

supabase = get_client()

response = insert_test_record(supabase)
print(f"The weather record inserted: {response}")


# what would happen if you ran the function twice? 
# How would you change the call to make it safe to run multiple times?
#
# Answer:
# When I ran the function twice, I received an error indicating that "duplicate key value 
# violates unique constraint". This is because the function attempts 
# to insert a new record with the same date, which violates the unique constraint 
# on the date column. To make it safe to run multiple times, I would modify the 
# function to check if a record with the same date already exists before attempting 
# to insert a new one by using upsert instead of insert. Upsert inserts a new row if the key is 
# new and updates the existing row if it is not.

# Q2
def get_records_by_date_range(supabase, start, end):
    """
    Returns all records in weather_raw between start and end dates.
    """
    response = supabase.table("weather_raw").select("*").gte("date", start).lte("date", end).execute()
    return response.data

result = get_records_by_date_range(supabase, "2026-08-26", "2026-08-26")

# Q3

# Explain the difference between insert and upsert in supabase-py.
# Give a concrete example of when you would choose each.
#
# Answer:
# Insert is used to insert new records into a table. If a record with the same primary key 
# already exists, it will raise an error if you set a unique constraint on the primary key. 
# If you didn't set a unique constraint, it would simply insert a duplicate new record.
# Upsert is used to insert a new record or update an existing record if it already exists instead of 
# duplicating it. You would choose insert when you are certain that the record you are adding is new and does not already 
# exist in the table. And you would choose upsert when you want to ensure that the record is added if 
# it doesn't exist or updated if it does exist, which is useful for scenarios where you want to avoid 
# duplicates and maintain data integrity. Example of insert: When setup a new project with new tables 
# and data, you would use insert to add the initial records.  Example of upsert: When you are receiving 
# a customer's update to their profile information, you would use upsert to ensure that the existing record 
# is updated with the new information instead of creating a duplicate record.

def safe_upsert(supabase, records):
    """
    Safely upserts a list of records into weather_raw.
    """
    total_rows = 0

    for record in records:
        response = (supabase.table("weather_raw").upsert(record, on_conflict="date").execute())

        total_rows += len(response.data)
    print(f"Upserted {total_rows} records into weather_raw.")


records = [
    {
        "date": "2026-08-25",
        "temperature_2m_max": 25.5,
        "temperature_2m_min": 18.3,
        "precipitation_sum": 3.5,
        "wind_speed_10m_max": 18.7,
    },
    {
        "date": "2026-08-26",
        "temperature_2m_max": 28.0,
        "temperature_2m_min": 19.0,
        "precipitation_sum": 2.0,
        "wind_speed_10m_max": 15.0,
    },
]

safe_upsert(supabase, records)

# ------------------------------------------------
# Idempotency
# ------------------------------------------------

# Q1
# Explain why idempotency matters for a data pipeline. Give one concrete 
# example of what goes wrong in a non-idempotent pipeline when the script 
# crashes halfway through and is restarted.
#
# Answer:
# Idempotency matters for a data pipeline because it ensures that the same operation 
# can be performed multiple times produces the same result as running it once. 
# This is important in data pipelines because failures can occur, and if a 
# pipeline is not idempotent, re-running it after a failure could lead to 
# duplicate records, inconsistent data, or other consequences.
