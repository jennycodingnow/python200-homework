# --- Completions API ---  

# API Q1
import json
from dotenv import load_dotenv
from openai import OpenAI



load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print(f"Response: {response.choices[0].message.content}")
print(f"Model: {response.model}")
print(f"Total token used: {response.usage.total_tokens}")

# API Q2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"Temperature: {temp}, Response: {response.choices[0].message.content}")


# Temperature 0 and 0.7 came back with the same name , while temperature 1.5 with a 
# different name and longer response with variation in explanations. Lower temperatures tend to produce more consistent 
# and predictable outputs, while higher temperatures produce more randomness and creativity.
# If I want a consistent, reproducible output, I would use lower temperature 0.

# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices):
    print(f"Choice {i+1}: {choice.message.content}")

# API Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

print(f"Response: {response.choices[0].message.content}")

# The response was truncated/cut off when max_tokens was set to 15 because it limits the maximum number of tokens generated.
# In real applications, max_tokens helps control costs, response length, and prevents unnecessarily long outputs.

# --- System Messages and Personas ---  

# System Q1

# First personality
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print(f"First Personality Response: {response.choices[0].message.content}")

# Second personality
messages = [
    {"role": "system", "content":(
    "You are a busy colleague who expects others to research on their own. "
    "You don't have time to explain everything in detail and always provide a concise answer.")},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print(f"Second Personality Response: {response.choices[0].message.content}")

# The first personality explains the concept in details and ends with encouragement,
# while the second personality gives a shorter, more direct answer without encouragement.

# System Q2

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

print(f"Response for System Q2: {response.choices[0].message.content}")

# The model knows Jordan's name even though it's stateless because
# the previous messages are included in the messages list. This gives
# the model the conversation's context within the same API request.

# --- Prompt Engineering ---  

# Prompt Q1

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = "Classify the sentiment of each review as positive, negative, or mixed.\n"

for i, review in enumerate(reviews,1):
    prompt += f"Review {i}: {review}\n"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
)

print("Zero-Shot Results:")
print(response.choices[0].message.content)

# Prompt Q2

prompt = """
Classify the sentiment of each review as positive, negative, or mixed.
Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
\n"""

for i, review in enumerate(reviews,1):
    prompt += f"Review {i}: {review}\n"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
)

print("One-Shot Results:")
print(response.choices[0].message.content)

# The one-shot example did not significantly change the output in this case,
# but examples can help guide formatting and consistency.
# It's still able to classify the sentiment of each review correctly. 

# Prompt Q3

prompt = """
Classify the sentiment of each review as positive, negative, or mixed.
Example 1:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
Example 2:
Review: "His flight was delayed and the airline lost his luggage."
Sentiment: negative
Example 3:
Review: "She found a job that she loves right after graduation."
Sentiment: positive

Reviews:
"""

for i, review in enumerate(reviews,1):
    prompt += f"Review {i}: {review}\n"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt}
    ],
)

print("Few-Shot Results:")
print(response.choices[0].message.content)

# Zero-shot works well for simple tasks where the instructions are clear.
# One-shot works well when you want to show the model the desired output format.
# Few-shot works well for more complex tasks or when you want the model to
# follow a specific pattern or improve consistency.

# Prompt Q4

prompt = f"""
Show your step-by-step reasoning, then give the final answer on its own line labelled: Final answer: <value>

Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}], 
    temperature=0,
)

print(response.choices[0].message.content)

# Asking the model to reason step by step tends to improve accuracy because
# it breaks the problem into smaller calculations, reducing the chance of
# mistakes and making the solution easier to verify.

# Prompt Q5

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."


prompt = f"""
Analyze the sentiment of this customer review and respond only with valid JSON.
Return keys: sentiment (positive/negative/mixed), confidence (a float from 0 to 1), reason (one sentence).

Review : {review}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}], 
    temperature=0,
)

raw_response = response.choices[0].message.content
print("Raw response:", raw_response)

try:
    result = json.loads(raw_response)
    print("Parsed sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")
    print("Raw response:", raw_response)


# Prompt Q6

# First prompt for Prompt Q6
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}], 
    temperature=0,
)

print(response.choices[0].message.content)

# Second prompt for Prompt Q6
user_text = (
    "Today is hot, so we ate ice cream on the porch."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}], 
    temperature=0,
)

print(response.choices[0].message.content)

# Delimiters help separate user-provided text from instructions,
# preventing the model from confusing the content being analyzed
# with the instructions it should follow.

# --- Local Models with Ollama ---  


# OpenAI
# The Responses API is the newer API style
response = client.responses.create(
    model="gpt-5",
    input="Explain what a large language model is in two sentences."
)

print(response.output_text)

"""
OpenAI output:
A large language model is a type of neural network trained on vast amounts of text to predict the next word 
(token) in a sequence, learning statistical patterns, structure, and knowledge embedded in language. 
Using these learned representations, it can generate coherent text, answer questions, translate, summarize, 
and perform other language-based tasks from input prompts.

"""

"""
Ollama output:
A large language model is an AI model trained on vast amounts of text data, enabling it to understand and 
generate human-like language. These models can perform tasks like translation, customer 
service, and content generation, making them valuable tools across industries.

"""

# Differences:
# Both explained the same concept correctly. The OpenAI response used slightly different wording that was more technical,
# while the Ollama response was more concise and less technical.

# One advantage of running a model locally:
# It keeps your data on your own computer and can work without an internet
# connection after the model is downloaded.

# One disadvantage of running a model locally:
# Local models often require significant CPU/GPU resources and smaller
# models may produce lower-quality responses than larger cloud-hosted models.
# And it takes more storage space.