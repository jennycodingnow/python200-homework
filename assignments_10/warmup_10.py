# ================================================
# Part 1: Warmup
# ================================================

import os
from time import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ------------------------------------------------
# ML vs. LLM in Pipelines
# ------------------------------------------------

# ML/LLM Q1

# Explain the difference between what the ML classifier produces and what 
# the LLM produces in this week's pipeline. Why does each tool do what it does? 
# What would go wrong if you tried to swap them — using the LLM to make the 
# binary good/skip prediction and the ML model to write the recommendation?

# Answer: 
# The ML classifier produces a binary output (good/skip) based on the features it was trained on. 
# It is designed to make consistent, fast, and data-driven classification decisions. The LLM, on 
# the other hand, generates natural-language recommendations based on the context and information 
# provided to it. Each tool is used for a different task because they are optimized for different 
# types of problems. The ML model is specialized for classification, while the LLM is specialized 
# for understanding context and generating coherent, contextually relevant language. If we swapped 
# the tools, the LLM could potentially make a binary prediction, but it would likely be less consistent, 
# less efficient, and less reliable than the dedicated ML classifier. The ML classifier, meanwhile, is 
# not designed to generate open-ended natural-language recommendations; it would only produce the types 
# of outputs it was trained to predict. Therefore, using the ML model for the recommendation and the LLM 
# for the binary prediction would make the pipeline less effective.

# ML/LLM Q2

# Converting a date string like "2023-07-04" to day-of-week
# I would use deterministic code because it's a straightforward transformation that doesn't require 
# learning or interpretation and always produces a repeatable result.

# Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text
# I would use an LLM for classifying a job posting as "entry-level", "mid-level", or "senior" 
# based on freeform text because it requires understanding the context and interpreting 
# unstructured language.

# Predicting customer churn given 15 numeric features and a labeled training dataset
# I would use a trained ML model for predicting customer churn given 15 numeric features and a 
# labeled training dataset because it can learn patterns from the data to make predictions.

# Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form
# I would use deterministic code because the known variations can be mapped to a canonical city name 
# using predefined rules or a lookup table.

# Summing a column of revenue figures
# I would use deterministic code because it's a simple arithmetic operation that doesn't require 
# learning or interpretation.

# ML/LLM Q3

# What is incremental processing, and why is it important for this pipeline? What would 
# happen — in terms of cost and data correctness — if the transform script re-processed 
# all 365 records every time it ran?

# Answer:
# Incremental processing is a method of processing only the new or changed data since the 
# last run, rather than redoing completed work. It is important for this pipeline because 
# it allows for efficient use of resources and reduces processing time and costs. It also 
# ensures that there are no duplicate records or inconsistencies in the data. 
# If the transform script re-processed all 365 records every time it ran, it would lead to 
# increased costs due to unnecessary computation and storage. Additionally, it could introduce
# data correctness issues, such as duplicate entries or overwriting of existing records,
# leading to inaccurate results and potential confusion in the processes. It would also 
# make the pipeline less efficient and slower, as it would be doing redundant work instead 
# of focusing on the new or updated data.

# ------------------------------------------------
# Prompt Design
# ------------------------------------------------

# Prompt Q1
# Part A
# Write an alternative system prompt that asks for a two-sentence recommendation where the
# first sentence states the prediction and the second sentence explains the reasoning.

# Answer:
# SYSTEM_PROMPT = (
#     "You are writing a two-sentence running recommendation for a daily weather summary app. "
#     "The first sentence must state the prediction, and the second sentence must explain the reasoning. "
#     "You will receive weather conditions for a single day and a machine learning prediction "
#     "about whether the day is good for running. "
#     "Write exactly two sentences—direct, practical, and specific to the conditions and reasoning. "
#     "Do not use bullet points, headers, or phrases like 'Based on the data'."
# )


# Part B
# What would you need to change in the validation logic to accommodate two sentences instead of one?
# Answer:
# To accommodate two sentences instead of one, the validation logic would need to be updated to 
# verify that the model output contains exactly two sentences. The validator should also ensure 
# that the first sentence contains the prediction and the second sentence provides the reasoning.

# Prompt Q2


def call_with_retry(client, messages, max_retries=3):

    SYSTEM_PROMPT = (
    "You are writing a two-sentence running recommendation for a daily weather summary app. "
    "The first sentence must state the prediction, and the second sentence must explain the reasoning. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly two sentences—direct, practical, and specific to the conditions and reasoning. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
    )

    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                system=SYSTEM_PROMPT,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(2)
    return None


# Describe when you would use this in a production pipeline.
# Answer:
# In a production pipeline, I would use this to make API calls more robust and resilient
# to temporary network errors, API failures, or other transient problems. Setting a maximum 
# number of retries and adding a delay between attempts helps prevent repeated requests 
# from happening too quickly. This is especially important in a production environment 
# where reliability and uptime are important. If all retry attempts fail, returning None allows
# the pipeline to handle the failure gracefully.
