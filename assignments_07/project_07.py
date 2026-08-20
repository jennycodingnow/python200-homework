# Part 2: Mini-Project — World Happiness Agent
from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")
api_key = os.getenv("OPENAI_API_KEY")

df = None
DATA_PATH = Path("resources/merged_happiness.csv")

os.makedirs("outputs/", exist_ok=True)


# ------------------------------------------------
# Task 1: Define Your Tools
# ------------------------------------------------

#Tool 1: load_happiness_data

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into global dataframe.
    
    Returns:
        A dictionary containing the shape of the loaded data and a list
        of the column names.
    """
    global df
    if DATA_PATH.exists():
        yearly_df = pd.read_csv(DATA_PATH)
        yearly_df = yearly_df.rename(columns={ 
            "Ladder score": "happiness_score",
            "Happiness score": "happiness_score",
            "Country": "country",
            "Regional indicator": "region",
            "GDP per capita": "gdp_per_capita"
        })

        yearly_df.columns = (
            yearly_df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        df = yearly_df

    else: 
        data_dir = Path("resources/happiness_project/")
        all_files = list(data_dir.glob("world_happiness_*.csv"))

        if not all_files:
            return {"error": "No World Happiness CSV were found."}

        dataframes = []

        for file in all_files:
            yearly_df = pd.read_csv(file, sep=";", decimal=",")
            yearly_df = yearly_df.rename(columns={
                "Ladder score": "happiness_score",
                "Happiness score": "happiness_score",
                "Country": "country",
                "Regional indicator": "region",
                "GDP per capita": "gdp_per_capita"
            })

            yearly_df.columns = (
                yearly_df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
            )
            
            year = file.split("_")[-1].replace(".csv", "")
            yearly_df["year"] = int(year)


            dataframes.append(yearly_df)

        df = pd.concat(dataframes, ignore_index=True)


    return {
        "shape": df.shape,
        "columns": list(df.columns)
    }


#Tool 2: summarize_column
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.
    
    Args:
        column: Name of the column to summarize.
    
    Returns:
        A dictionary of descriptive statistics from pandas.describe(),
        or an error dictionary if the data is not loaded or the column
        does not exist.
    """
    if df is None:
        return {"error": "Data has not been loaded"}

    if column not in df.columns:
        return {"error": f"Column '{column}' not found in the dataset"}

    summary = df[column].describe().to_dict()
    return summary


#Tool 3: compute_correlation
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.
    
    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.

    Returns:
        A dictionary containing the column names, Pearson correlation
        coefficient, and p-value, or an error dictionary if the data
        or columns are unavailable.
        
    """

    if df is None:
        return {"error": "No CSV is loaded."}

    if col1 not in df.columns:
        return {"error": f"Column '{col1}' is not in the data."}

    if col2 not in df.columns:
        return {"error": f"Column '{col2}' is not in the data."}

    r, p = pearsonr(df[col1], df[col2])

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(r, 4),
        "p_value": round(p, 4)
    }

#Tool 4: get_top_n_countries
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.
    Args:
        column: The name of the column used to rank the countries.
        year: The year for which to retrieve the rankings.
        n: The number of top countries to return. Defaults to 5.

    Returns:
        A list of dictionaries containing each country's name and
        its value for the requested column. Returns an error dictionary
        if no data is loaded, the column does not exist, or the year
        is not available.
    """
    if df is None:
        return {"error": "No CSV is loaded."}

    if column not in df.columns:
        return {"error": f"Column '{column}' is not in the data."}

    if year not in df["year"].unique():
        return {"error": f"Year '{year}' is not in the data."}

    top_happiness = df[df["year"] == year].sort_values(by=column, ascending=False).head(n)
    return top_happiness[["country", column]].to_dict("records")

# ------------------------------------------------
# Task 2: Build the Agent
# ------------------------------------------------

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).

For plots, use matplotlib.pyplot with a non-interactive backend and save the figure
directly to the assignments_07/outputs/. 

Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# ------------------------------------------------
# Running the Project
# ------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------
    # Task 3: Run Guided Queries
    # ------------------------------------------------

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(f"Response: {response}")


    # ------------------------------------------------
    # Task 4: Your Own Questions
    # ------------------------------------------------


    # My query 1
    my_query_1 = "What is the average healthy_life_expectancy, and what are its minimum and maximum values?"   

    response_1 = agent.run(my_query_1, reset=False)
    print(f"\n Response query1: {response_1}")
    # Comment: Did this trigger tool use, code generation, or both?
    # This triggered tool use, the summarize_column.

    # My query 2
    my_query_2 ="Which region had the largest improvement in average happiness_score between 2015 and 2024?"

    response_2 = agent.run(my_query_2, reset=False)
    print(f"\n Response query2: {response_2}")
    # Comment: Did this trigger tool use, code generation, or both?
    # This triggered code generation because no existing tool directly calculated the 
    # change in regional averages between two years.


# ------------------------------------------------
# Task 5: Reflection
# ------------------------------------------------

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
#
# The agent reported a Pearson correlation of 0.6313 and a p-value of 0.0.
# It correctly identified the correlation as statistically significant because
# the p-value was below the conventional 0.05 significance threshold.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#
# Yes, I was surprised me that the agent was capable to recognize that it needed custom code, 
# but it also demonstrated a limitation: after failing to access the real data, it generated 
# mock data and produced an answer from that mock data.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
#
# An additional tool that would make the agent more useful would be a
# regional statistics tool. It could calculate statistics such as the average
# happiness score for each region for a selected year or compare regional
# averages between two years. This would help the agent answer questions such
# as "Which region had the largest improvement in happiness between 2015 and
# 2024?" without having to generate custom code or mock data for the calculation.
