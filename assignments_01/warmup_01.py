import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats
from scipy import stats
from scipy.stats import pearsonr



# --- Pandas ---

# Pandas Q1

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print("First 3 rows of the dataset:")
print(df.head(3))

print("Shape of the dataset:")
print(df.shape)

print("Data types of the dataset:") 
print(df.dtypes)

# Pandas Q2
print("Students who passed the course:")
passed_students = df[(df["passed"]==True) & (df["grade"]>80)]
print(passed_students)

# Pandas Q3
df["grade_curved"] = df["grade"] + 5
print("Dataset with new column 'grade_curved':")
print(df)

# Pandas Q4
df["name_upper"] = df["name"].str.upper()
print("Name with uppercase:")
print(df[["name", "name_upper"]])

# Pandas Q5
city_grades = df.groupby("city")["grade"].mean()
print("Mean grade for each city: ")
print(city_grades)

# Pandas Q6
df["city"] = df["city"].replace("Austin", "Houston")
print("Replace Austin with Houston:")
print(df[["name", "city"]])

# Pandas Q7
df.sort_values(by="grade", ascending=False, inplace=True)
print("First 3 rows after sorting by grade:")
print(df.head(3))

# --- NumPy ---

# NumPy Q1
arr1 = np.array([10, 20, 30, 40, 50])

print("1D Array shape: ", arr1.shape)
print("1D Array data type: ", arr1.dtype)
print("1D Array dimensions: ", arr1.ndim)

# NumPy Q2
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print("2D Array shape: ", arr.shape)
print("2D Array size: ", arr.size)

# NumPy Q3
result =  arr[0:2,0:2]
print("2D Array slice: ", result)

# NumPy Q4
zeros_arr = np.zeros((3,4), dtype=int)
print("Array of zeros: ")
print(zeros_arr)

ones_arr = np.ones((2,5), dtype=int)
print("Array of ones: ")
print(ones_arr)

# NumPy Q5
arr3 = np.arange(0, 50, 5)
print("Array: ", arr3)
print("Shape: ", arr3.shape)
print("Mean: ", arr3.mean())
print("Sum: ", arr3.sum())
print("Standard Deviation: ", arr3.std())

# NumPy Q6
arr4 = np.random.normal(loc=0, scale=1, size=200)
print("Array of random numbers from a normal distribution: ")
print(arr4)

print("Mean: ", np.mean(arr4))
print("Standard Deviation: ", np.std(arr4))

# --- Matplotlib ---

# Matplotlib Q1 - Line Plot
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x, y, marker='o', linestyle='-', color='blue')
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q2 - Bar Plot
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores, color=['green', 'orange', 'blue', 'red'])
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplotlib Q3 - Scatter Plot
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color='red', marker='o', label='Dataset 1')
plt.scatter(x2, y2, color='green', marker='s', label='Dataset 2')
plt.title("Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Matplotlib Q4  - Subplots

fig, axs = plt.subplots(1,2)

#left subplot:
axs[0].plot(x, y, marker='o', linestyle='-', color='blue')
axs[0].set_title("Squares")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")

#right subplot:
axs[1].bar(subjects, scores, color=['green', 'orange', 'blue', 'red'])
axs[1].set_title("Subject Scores")
axs[1].set_xlabel("Subjects")
axs[1].set_ylabel("Scores")

plt.show()

# --- Descriptive Statistics ---

# Descriptive Stats Q1

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
arr_ds = np.array(data)

print(f"Mean: {np.mean(arr_ds)}")
print(f"Median: {np.median(arr_ds)}")
print(f"Variance: {np.var(arr_ds)}")
print(f"Standard Deviation: {np.std(arr_ds)}")

# Descriptive Stats Q2
random_data = np.random.normal(65, 10, 500)
plt.hist(random_data, bins=20, color='purple', alpha=0.7)
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Q3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], tick_labels = ["Group A", "Group B"])
plt.ylabel("Score")
plt.title("Score Comparison")
plt.show()

# Descriptive Stats Q4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.boxplot([normal_data, skewed_data], tick_labels = ["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.show()

# The exponential distribution is more skewed (not symmetrical) than the normal distribution (more balanced).
# The exponential distribution has a longer upper whisker, indicating that there might be some outliers above the whisker 
# and mean is not centered in the middle of the box, which indicates that the data is skewed, so
# the median is approriate to use as a measure of central tendency.
# While the normal distribution is almost symmetrical, the mean is appropriate to use as a measure of central tendency.

# Descriptive Stats Q5

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print("Data 1")
print("Mean: ", np.mean(data1)) 
print("Median: ", np.median(data1)) 
mode1 = stats.mode(data1)
print("Mode:", mode1.mode)

print("\nData 2")
print("Mean: ", np.mean(data2)) 
print("Median: ", np.median(data2)) 
mode2 = stats.mode(data2)
print("Mode: ", mode2.mode)

#The median is the same for both datasets but the mean is so different because the outlier of number 150 in data2 skews the mean. 


# --- Hypothesis Testing ---

# Hypothesis Q1

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

result1= stats.ttest_ind(group_a, group_b)
print("Q1")
print(f"t-statistic: {result1.statistic:.3f}")
print(f"p-value: {result1.pvalue:.6f}")

# Hypothesis Q2

if result1.pvalue < 0.05:
    print("Significant")
else:
    print("Not significant")

# Hypothesis Q3

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

result2 = stats.ttest_rel(before, after)

# Rounding to float
print("\nQ3")
print(f"t-statistic: {result2.statistic:.3f}")
print(f"p-value: {result2.pvalue:.6f}")

if result2.pvalue < 0.05:
    print("Significant")
else:
    print("Not significant")

# Hypothesis Q4

scores = [72, 68, 75, 70, 69, 74, 71, 73]
result3 = stats.ttest_1samp(scores, 70)
print("\nQ4")
print(f"t-statistic: {result3.statistic:.3f}")
print(f"p-value: {result3.pvalue:.6f}")

if result3.pvalue < 0.05:
    print("Significant")
else:
    print("Not significant")
# The sample mean score was 71.5 which is slightly higher than the benchmark 70. 
# But this one-sample t-test didn't show that the difference is not statistically significant  (p-value > 0.05), 
# meaning that the difference might be due to random variation.

# Hypothesis Q5
# One-tailed: is group_a scores less than group_b scores?
result4 = stats.ttest_ind(group_a, group_b, alternative="less")
print("\nQ5")
print(f"t-statistic: {result4.statistic:.3f}")
print(f"p-value: {result4.pvalue:.6f}")

if result4.pvalue < 0.05:
    print("Significant")
else:
    print("Not significant")

# Hypothesis Q6

print("In conclusion, the results of Q1 show that Group A scored lower than Group B, "
"and the difference is statistically significant. This difference is unlikely to be due to random chance.")

# --- Correlation ---

# Correlation Q1

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print("Correlation matrix:")
print(corr_matrix)
print("Correlation coefficient:") 
print(corr_matrix[0, 1])
# The correlation coefficient is 0.99999 which is near 1. 
# This indicates a positive linear relationship between x and y.

# Correlation Q2

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

r, p = pearsonr(x, y)
print("Correlation coefficient:", round(r, 2))
print("p-value:", round(p, 4)) 
# A small p-value (0.0) suggests the relationship is unlikely to be due to random noise


# Correlation Q3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
print("Correlation using DataFrame:")
corr = df.corr()
print(corr)

# Correlation Q4 - scatter plot

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y, color='teal')
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Correlation Q5 - heatmap using Q3 dataframe

sns.heatmap(corr, annot=True,cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- Pipelines---

# Pipelines Q1


def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series

def summarize_data(cleaned_series):
    series_summary = {
        "mean": cleaned_series.mean(),
        "median": cleaned_series.median(),
        "std": cleaned_series.std(),
        "mode": cleaned_series.mode()[0]}
    return series_summary

def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    result = summarize_data(cleaned_series)
    return result


if __name__ == "__main__":  
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    result = data_pipeline(arr)

    print("Summary for Pipeline Q1:")
    for key, value in result.items():
        print(f"{key}: {value}")


    