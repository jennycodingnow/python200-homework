# ================================================
# Part 1: Warmup
# ================================================

import time
from prefect import task
from prefect.logging import get_run_logger

# ------------------------------------------------
# Prefect Orchestration
# ------------------------------------------------

#------Prefect Q1------
# Questions:
# what is the difference between a @task and a @flow in Prefect? You have a helper 
# function that converts a temperature from Celsius to Fahrenheit — a pure, in-memory 
# calculation with no I/O. Would you decorate it with @task? Why or why not?

# Answers: 
# A @task is the smallest unit of work in a pipeline such as loading data, cleaning a dataset, 
# or sending a notification. It is created with @task() decorator in fron of a function in Python.
# A prefect automatically tracks its state (running, successful, or failed) are being tracked, 
# logs its output and retry that tasks if it encounters an error.  While a @flow is higher-level 
# workflow logic that connects multiple tasks together into a full pipeline. A flow orchestrates 
# how and when tasks run, and serves as the single entry point for running the entire pipeline. 
# It is created with @flow() decorator. 

# I would decorate the temperature conversion function with a @task because I would put it in a pipelien so 
# a Prefect can track, log and retry if it fails to convert a temperature. This makes a task efficient and robust.

#------Prefect Q2------
# Questions: 
# Write just the decorator line for a task named call_api that retries up to 3 times with a 30-second 
# delay between attempts.

@task (name="call_api", retries=3, retry_delay_seconds=30)


#------Prefect Q3------

# Questions: 
# You run your pipeline and the Prefect UI shows: extract is Completed, load_raw is 
# Completed, transform is Failed, load_enriched never ran. In a comment block, describe: 
# where in the UI do you look to understand what went wrong, and what specific information 
# would you expect to find there?

# It is in the Dashboard in UI that you will find what went wrong. The Dashboard displays 
# all your active and completed flow runs, giving you a quick overview of what’s happening. The specific
# information you would expect to find there when clicked on a specific flow run opens detailed information 
# about its individual tasks and logs. Each log entry is structured with timestamps and log 
# levels such as INFO, WARNING, and ERROR, and all logs are stored for future review.

# ------------------------------------------------
# Production Patterns
# ------------------------------------------------

#------Production Q1------

# Questions: 
# In a comment block, explain what raise_for_status() does and why it is better than writing 
# if response.status_code != 200: print("error") in a pipeline task. What happens to downstream 
# tasks in each case when the API returns a 500 error?

# Answers: 
# Raise_for_status() raises an exception immediately if the server returns a 4xx or 5xx HTTP response. 
# It is better than writing if response.status_code != 200: print("error") in a pipeline task because 
# it makes the failure explicit instead of allowing the pipeline to continue with bad or missing data.
# In Prefect, if raise_for_status() encounters a 500 error, the exception causes the task to be marked 
# as Failed. Unless the exception is caught, downstream tasks that depend on that task will not run. 
# The failure is also clearly shown in the logs, making the problem easier to debug.
# With if response.status_code != 200: print("error"), printing an error does not cause the task to fail. 
# The task continues running and may try to process the unsuccessful response as if it were valid data. 
# This can result in malformed or incorrect data being passed to downstream tasks, or the pipeline 
# appearing successful even though the API request failed.

#------Production Q2------

# Questions: 
# Your load_raw task uses upsert with on_conflict="date" instead of insert. The pipeline crashes halfway
# through the transform step. You fix the bug and re-run from the beginning. In a comment block, explain: 
# what does upsert protect you from in this scenario, and what would happen if you had used plain insert 
# instead?

# Answers: 
# The upsert protects you from duplicate records when you rerun the pipeline after a crash.
# With on_conflict="date", if a record for that date already exists, it is updated instead 
# of inserting another copy. If plain insert were used instead, re-running the pipeline 
# could fail with a duplicate/conflict error (or create duplicates if duplicates were allowed), 
# potentially leaving the data inconsistent and inaccurate.

#------Production Q3------
# Questions: 
# Write a task stub — just the function signature, decorator, and a single log line — that uses 
# get_run_logger() to log an INFO message saying how many enrichment records were upserted.
# The function should accept enrichment_records (a list) as its argument.


@task
def load_enriched(enrichment_records: list)-> None:
    logger = get_run_logger()
    logger.info(f"Upserted {len(enrichment_records)} enrichment records")


#------Production Q4------

# Questions:
# In a comment block, explain how the incremental processing check in the transform task 
# contributes to idempotency. If you removed it and the pipeline ran the ML and LLM steps on 
# all 365 records every time, what would be the practical consequences (in terms of cost, time, 
# and data correctness)?

# Answers: 
# The incremental processing check contributes to idempotency by ensuring that records
# that have already been processed are not processed again when the pipeline is re-run.
# This helps the pipeline produce the same effective result regardless of how many times
# it is run.

# If the check were removed and the ML and LLM steps ran on all 365 records every time,
# the pipeline would repeatedly redo work that was already completed. This would increase
# cost because the ML and LLM services would be called unnecessarily, and increase runtime
# because all 365 records would have to be processed each time. It could also affect data
# correctness by potentially creating duplicate results or overwriting existing results
# with new or inconsistent outputs.