import pandas as pd
import numpy as np
from prefect import task, flow

# Pipelines Q2
# Task functions
data_arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(data_arr):
    return pd.Series(data_arr, name="values")

@task
def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series
@task
def summarize_data(cleaned_series):
    series_summary = {
        "mean": cleaned_series.mean(),
        "median": cleaned_series.median(),
        "std": cleaned_series.std(),
        "mode": cleaned_series.mode()[0]}
    return series_summary

# Prefect flow
@flow
def pipeline_flow(data_arr):
    series = create_series(data_arr)
    cleaned_series = clean_data(series)
    result = summarize_data(cleaned_series)
    return result


if __name__ == "__main__":  
    result = pipeline_flow(data_arr)



# 1) Prefect is more overhead here because the dataset is very small and the tasks are simple. 
# Using a Prefect flow for this small dataset is overkill because it requires extra setup, learning, 
# and runtime overhead, even though the work can be done with normal Python functions.

# 2) Prefect is useful for larger or real world projects or tasks, such as data pipelines that run every day,
# machine learning workflows, or jobs that need scheduling, retries, and monitoring. Even if the tasks are simple, 
# Prefect helps manage the overall workflow.
