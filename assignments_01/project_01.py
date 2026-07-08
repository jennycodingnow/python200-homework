from apprise import logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prefect import flow, task, get_run_logger
from scipy import stats
from scipy.stats import pearsonr


#Task 1
files_list = [
    "data/world_happiness_2015.csv",
    "data/world_happiness_2016.csv",
    "data/world_happiness_2017.csv",
    "data/world_happiness_2018.csv",
    "data/world_happiness_2019.csv",
    "data/world_happiness_2020.csv",
    "data/world_happiness_2021.csv",
    "data/world_happiness_2022.csv",
    "data/world_happiness_2023.csv",
    "data/world_happiness_2024.csv"
]


@task(retries=3, retry_delay_seconds=2)
def load_data(files_list):

    logger = get_run_logger()

    dataframes = []

    for file in files_list:

        df = pd.read_csv(file, sep=";", decimal=",")
        df = df.rename(columns={"Ladder score": "Happiness score"})

        year = file.split("_")[-1].replace(".csv", "")
        df["Year"] = int(year)

        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)

    merged_df.to_csv("outputs/merged_happiness.csv", index=False)

    logger.info("Datasets merged and saved")

    return merged_df

#Task 2
@task
def statistics_data(merged_df):

    logger = get_run_logger()
    
    happiness_statistics = {
        "mean": float(merged_df["Happiness score"].mean()),
        "median": float(merged_df["Happiness score"].median()),
        "std": float(merged_df["Happiness score"].std())
    }
    
    mean_by_year = merged_df.groupby("Year")["Happiness score"].mean()
    mean_by_region = merged_df.groupby("Regional indicator")["Happiness score"].mean()

    logger.info(f"Overall mean: {happiness_statistics['mean']:.2f}")
    logger.info(f"Overall median: {happiness_statistics['median']:.2f}")
    logger.info(f"Overall std: {happiness_statistics['std']:.2f}")

    logger.info(f"Mean happiness by year: {mean_by_year}")

    logger.info(f"Mean happiness by region: {mean_by_region}")
    return {
        "statistics": happiness_statistics,
        "mean_by_year": mean_by_year,
        "mean_by_region": mean_by_region
    }

#Task 3

@task
def create_visualizations(merged_df):
    logger = get_run_logger()

    #histogram 
    plt.figure(figsize=(8,5))
    plt.hist(merged_df["Happiness score"], bins=30, color='purple', alpha=0.7)
    plt.title("Happiness Score Distribution")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.savefig("outputs/happiness_histogram.png")
    plt.close()
    logger.info("Happiness histogram saved")

    #boxplot
    plt.figure(figsize=(18,16))
    sns.boxplot(data=merged_df, x="Year", y="Happiness score")
    plt.title("Happiness Score by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.xticks(rotation=45)
    plt.savefig("outputs/happiness_by_year.png")
    plt.close()
    logger.info("Happiness by year boxplot saved")

    #scatter plot
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=merged_df, x="GDP per capita",y="Happiness score")
    plt.title("GDP per capita vs. Happiness Score")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness Score")
    plt.savefig("outputs/gdp_vs_happiness.png")
    plt.close()
    logger.info("Happiness vs. GDP scatter plot saved")

    #correlation heatmap
    plt.figure(figsize=(20,20))
    numeric_cols = merged_df.select_dtypes(include="number")
    corr_matrix = numeric_cols.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/correlation_heatmap.png")
    plt.close()
    logger.info("Correlation heatmap saved")

#Task 4
@task
def hypothesis_testing(merged_df):
    logger = get_run_logger()
    #independent samples t-test-1
    hscore_2019 = merged_df[merged_df["Year"] == 2019]["Happiness score"]
    hscore_2020 = merged_df[merged_df["Year"] == 2020]["Happiness score"]

    t_stat_1, p_value_1 = stats.ttest_ind(hscore_2019, hscore_2020)
    mean_2019 = hscore_2019.mean()
    mean_2020 = hscore_2020.mean()

    logger.info(f"Happiness 2019 Mean: {mean_2019:.2f}")
    logger.info(f"Happiness 2020 Mean: {mean_2020:.2f}")

    logger.info(f"Test 1 T-statistic: {t_stat_1:.3f}")
    logger.info(f"Test 1 P-value: {p_value_1:.4f}")

    if p_value_1 < 0.05:
        interpretation = (
            "The difference in happiness scores between 2019 and 2020 "
            "is statistically significant. This suggests that global "
            "happiness levels changed in 2020."
        )
    else:
        interpretation = (
            "The difference in happiness scores between 2019 and 2020 is not statistically significant. "
            "This suggests that global happiness levels did not change significantly in 2020."
        )
    logger.info(f"Interpretation: {interpretation}")
    
    #independent samples t-test-2
    region_e_2020 = merged_df[
    (merged_df["Regional indicator"] == "East Asia") &
    (merged_df["Year"] == 2020)]["Happiness score"]

    region_w_2020 = merged_df[
        (merged_df["Regional indicator"] == "Western Europe") &
        (merged_df["Year"] == 2020)
    ]["Happiness score"]

    t_stat_2, p_value_2 = stats.ttest_ind(
        region_e_2020,
        region_w_2020,
        equal_var=False
    )

    mean_e = region_e_2020.mean()
    mean_w = region_w_2020.mean()

    logger.info(f"Happiness East Asia Mean in 2020: {mean_e:.2f}")
    logger.info(f"Happiness Western Europe Mean in 2020: {mean_w:.2f}")

    logger.info(f"Test 2 T-statistic for 2020: {t_stat_2:.3f}")
    logger.info(f"Test 2 P-value for 2020: {p_value_2:.4f}")

    if p_value_2 < 0.05:
        interpretation = (
            "The difference in happiness scores between East Asia and Western Europe in 2020 "
            "is statistically significant. This suggests that there is a significant "
            "difference in happiness levels between these two regions."
        )
    else:
        interpretation = (
            "The difference in happiness scores between East Asia and Western Europe in 2020 is not statistically significant. "
            "This suggests that there is no significant difference in happiness levels between these two regions."
        )
    logger.info(f"Interpretation: {interpretation}")

    hypothesis = {
    "2019_vs_2020": {
        "t_stat": t_stat_1,
        "p_value": p_value_1
    },
    "east_asia_vs_western_europe_2020": {
        "t_stat": t_stat_2,
        "p_value": p_value_2
    }
    }

    return hypothesis

#Task 5
@task
def correlation_testing(merged_df):
    logger = get_run_logger()
    numeric_cols_df = merged_df.select_dtypes(include="number")
    happiness_col = numeric_cols_df["Happiness score"]
    correlation_dict = {}
    factors = numeric_cols_df.drop(columns=["Ranking", "Happiness score", "Year"]).columns
    for column in factors:
        coefficient, p_value = pearsonr(
            numeric_cols_df[column],
            happiness_col
        )

        correlation_dict[column] = {
            "coefficient": coefficient,
            "p_value": p_value
        }

    number_of_tests = len(factors)
    adjusted_alpha = 0.05 / number_of_tests

    logger.info(f"Number of tests: {number_of_tests}")
    logger.info(f"Adjusted alpha: {adjusted_alpha:.5f}")

    for column, value in correlation_dict.items():
        if value["p_value"] < 0.05:
            before = "significant"
        else:
            before = "not significant"

        if value["p_value"] < adjusted_alpha:
            after = "significant after Bonferroni"
        else:
            after = "not significant after Bonferroni"

        correlation_dict[column]["before_bonferroni"] = before
        correlation_dict[column]["after_bonferroni"] = after
        
        logger.info(
            f"{column}: correlation={value['coefficient']:.3f}, "
            f"p-value={value['p_value']:.4f}, "
            f"before={before}, after={after}"
        )
    return correlation_dict

#Task 6
@task
def summary_report(merged_df, hypothesis_data, correlation_data):
    logger = get_run_logger()
    number_of_countries = merged_df["Country"].nunique()
    number_of_years = merged_df["Year"].nunique()
    mean_by_region = merged_df.groupby("Regional indicator")["Happiness score"].mean()
    top_regions = mean_by_region.sort_values(ascending=False).head(3)
    bottom_regions = mean_by_region.sort_values().head(3)
    happiness_factors = [ factor for factor, result in correlation_data.items() if result["after_bonferroni"] == "significant after Bonferroni"]

    logger.info("Total number of countries and years in the merged dataset:")
    logger.info(f"Number of countries: {number_of_countries}")
    logger.info(f"Number of years: {number_of_years}")

    logger.info("Top 3 regions by average happiness score:")
    for region, score in top_regions.items():
        logger.info(f"{region}: {score:.3f}")

    logger.info("Bottom 3 regions by average happiness score:")
    for region, score in bottom_regions.items():
        logger.info(f"{region}: {score:.3f}")

    logger.info("Hypothesis testing results:")
    logger.info(f"The P-value for 2019 vs 2020 is {hypothesis_data['2019_vs_2020']['p_value']:.4f}, so we can conclude that the average global happiness was not statistically significant.")
    logger.info(f"The P-value for East Asia vs Western Europe in 2020 is {hypothesis_data['east_asia_vs_western_europe_2020']['p_value']:.4f}, so we can conclude there is a significant difference in happiness scores between these regions in 2020.")
    
    logger.info("Correlation testing results:")
    logger.info(f"Factors correlated with happiness using the Bonferroni correction: {happiness_factors}")

@flow
def happiness_pipeline(files_list):

    merged_df = load_data(files_list)
    statistics_data(merged_df)
    create_visualizations(merged_df)
    hypothesis_testing(merged_df)
    correlation_testing(merged_df)
    summary_report(merged_df, hypothesis_testing(merged_df), correlation_testing(merged_df))


if __name__ == "__main__":  
    happiness_pipeline(files_list)
    