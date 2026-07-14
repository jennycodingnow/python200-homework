from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

# The CSV file uses semicolons (;) as the field separator,
# so I need to specify sep=";" when reading it.

# Task 1
df = pd.read_csv('data/student_performance_math.csv', sep=";")
print(df.shape)
print(df.head(5))
print(df.dtypes)

# print(df['G3'].value_counts().sort_index())

plt.hist(df['G3'], bins=21, color='purple', alpha=0.7)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Score (G3)")
plt.ylabel("Frequency")
plt.savefig("outputs/g3_distribution.png")
plt.show()

# Task 2
print("Before filtering:", df.shape)
df_clean = df[df['G3'] != 0]

print("After filtering:", df_clean.shape)
# Remove rows where G3 equals 0 because these represent students who did not take the final exam, 
# not students who earned a zero. Keeping these rows would distort the model's grade predictions by 
# treating exam absence as poor academic performance.
print (df_clean.columns)

yn_cols = ["schoolsup", "internet", "higher", "activities"]


for col in yn_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

print("These data types: ", df_clean.dtypes)
print(df_clean.head(8))

corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])
print("Original Correlation:", round(corr_original, 2))
print("Filtered Correlation:", round(corr_filtered, 2))


# Filtering changes the result because the original dataset contains rows with
# G3 = 0 that students were absent from the final exam, which distort the relationship between absences and final grades.
# These zero values weaken the correlation because they represent a different
# group of students and affect the calculation of the linear relationship.
# After removing these rows, the correlation better reflects the relationship
# among students with actual final grades, showing a stronger negative
# association between absences and G3.

#Task 3
numeric_cols = df_clean.select_dtypes(include="number")
correlations = numeric_cols.corr()["G3"].drop("G3").sort_values()

print("Correlations:")
print(correlations)

print("Strongest positive correlations with G3:")
print(correlations.tail(1))
print("Strongest negative correlations with G3:")
print(correlations.head(1))

# Results are as expected. Usually positive variables will have a positive correlation with G3, 
# and negative variables will have a negative correlation with G3.
# The strongest positive correlation with G3 is "G2" (the second period grade), 
# which makes sense because students who perform well in the second period are likely to 
# perform well in the final exam. 
# The strongest negative correlation with G3 is "failures," indicating that students with more failures 
# tend to have lower final grades.

#Scatter plot G2 vs G3

plt.scatter(df_clean["G2"], df_clean["G3"])
plt.xlabel("G2 (Second Period Grade)")
plt.ylabel("G3 (Final Grade)")
plt.title("G2 vs G3")
plt.savefig("outputs/g2_vs_g3.png")
plt.show()

# G2 has the strongest positive correlation with G3 (r = 0.965583).
# Students with higher second-period grades tend to have higher final grades.
# This relationship is expected because G2 measures academic performance
# close to the final grade.

#Box plot failures vs G3

df_clean.boxplot(column="G3", by="failures")
plt.xlabel("Number of Past Failures")
plt.ylabel("G3 (Final Period Grade)")
plt.title("G3 Distribution by Number of Past Failures")
plt.suptitle("") 
plt.savefig("outputs/failures_vs_g3_boxplot.png")
plt.show()

# Failures has the strongest negative correlation with G3 (r = -0.293831). 
# The box plot shows that the median G3 tends to decrease as the number of past failures increases. This suggests that
# students with more previous failures are more likely to have lower final
# grades.

#Task 4
X = df_clean[["failures"]].values
y = df_clean["G3"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
print("Slope:", model.coef_[0])
print("Intercept: ", model.intercept_)

y_pred = model.predict(X_test)
#Below is the manual way, which I can import built-in function: 
#rmse = np.sqrt(mean_squared_error(y_test, y_pred)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print("RMSE for failures-only:", rmse)

r2_failures_only = model.score(X_test, y_test)
print("R2 for failures-only:", r2_failures_only)

# The slope is about -1.43, indicating that for each additional past failure,
# the model predicts the final grade (G3) decreases by about 1.43 points on average.
# The RMSE is about 2.96, which means the model's predictions are typically off
# by about 3 grade points. This is a relatively large error considering G3 is
# measured on a 0–20 scale.
# The R² is about 0.09, meaning the model explains only about 9% of the
# variation in final grades. This suggests that using failures alone is not
# very accurate, and other factors also influence student performance.
# This low R2 is consistent with the exploratory analysis, where failures had
# only a moderate negative correlation with G3 rather than a strong one.

#Task 5
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df_clean[feature_cols].values
y = df_clean["G3"].values
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_all_train.shape)
print("X_test shape:", X_all_test.shape)
print("y_train shape:", y_all_train.shape)
print("y_test shape:", y_all_test.shape)

model = LinearRegression()
model.fit(X_all_train, y_all_train)

y_all_pred = model.predict(X_all_test)
rmse_all = np.sqrt(np.mean((y_all_pred - y_all_test) ** 2))
r2_train = model.score(X_all_train, y_all_train)
r2_test = model.score(X_all_test, y_all_test)


print("RMSE with all features:", rmse_all)
print("Train R2:", r2_train)
print("Test R2:", r2_test)

print("Baseline R2:", r2_failures_only)
print("Full model R2:", r2_test)
print("Adding more features improves R2 by:", r2_test - r2_failures_only)

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# The negative coefficient for schoolsup is surprising because we might expect
# school support to improve grades. A possible explanation is that school support
# is provided mainly to students who are already struggling academically,
# so the model captures this association. 

# The baseline model has an R2 of 0.09, while the full model has an R2 of 0.15.
# This means that adding more features improves the model's ability to explain
# the variation in final grades, but the improvement is minor. The full model
# still explains only about 15% of the variation in G3, indicating that other
# factors not included in the model also influence student performance.

# If I were deploying this model in production, I would keep features with
# relatively large coefficients because they appear to contribute more to the
# predictions. These include failures, studytime, higher, internet, Medu,
# and Fedu. I would consider dropping features such as activities and
# freetime because their coefficients are very close to zero, suggesting
# they add little predictive value. I would keep failures even though its
# coefficient is negative, because it is an important predictor of G3.

#Task 6


plt.scatter(y_all_pred, y_all_test, alpha=0.6)

min_val = min(y_all_pred.min(), y_all_test.min())
max_val = max(y_all_pred.max(), y_all_test.max())

plt.plot([min_val, max_val], [min_val, max_val], color="red")

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Grade (G3)")
plt.ylabel("Actual Grade (G3)")
# Need to name the graph differently than the warmup assignment, 
# So I can save it in the outputs folder without overwriting the previous graph.
plt.savefig("outputs/predicted_vs_actual_students.png")
plt.show()

# The error is roughly uniform across grade levels
# If a point is above the diagonal, the actual final grade (G3) is higher than the predicted grade (Actual > Predicted),
# while if a point is below the diagonal, the actual final grade is lower than the predicted grade (Actual < Predicted). 
# The points are scattered around the diagonal, indicating that the model's predictions are generally close to the actual grades, 
# but there is still some error. 

#Summary: 
# 1) The size of the filtered dataset and the test set:
# The filtered dataset contains 357 rows and 18 columns. The full model test
# set contains 72 student records, with 11 input features used to predict G3.

# 2) The RMSE and R2 of your best model in plain language -- on a 0-20 scale, what does a typical prediction error actually mean?:
# The best model has an RMSE of 2.855, meaning the model's predictions are
# typically off by about 3 points on a 0-20 grading scale. The test R2 of 0.15
# means the model explains about 15% of the variation in final grades, so
# most of the differences in student performance are not explained by these
# features alone.

# 3) Which two features have the largest positive and largest negative coefficients, and what those mean:
# The largest positive coefficient is higher (0.610), indicating that students
# who plan to pursue higher education tend to have higher predicted final grades.
# The largest negative coefficient is failures (-1.145), indicating that each
# additional past failure is associated with lower predicted final grades.

# 4) One result that surprised you:
# A surprising result was that adding more features only increased the test R2
# from 0.09 in the baseline model to 0.15 in the full model. This suggests
# that the additional features provide some predictive value, but they do not
# explain most of the variation in final grades.

#Neglected Feature: The Power of G1

feature_cols_g1 = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", 
                "internet", "sex", "freetime", "activities", "traveltime", "G1"]

X_g1 = df_clean[feature_cols_g1].values
y_g1 = df_clean["G3"].values

X_g1_train, X_g1_test, y_g1_train, y_g1_test = train_test_split(
    X_g1, y_g1, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_g1_train, y_g1_train)

y_g1_pred = model_g1.predict(X_g1_test)

rmse_g1 = np.sqrt(np.mean((y_g1_pred - y_g1_test) ** 2))
r2_g1_train = model_g1.score(X_g1_train, y_g1_train)
r2_g1_test = model_g1.score(X_g1_test, y_g1_test)

print("RMSE with G1:", rmse_g1)
print("Train R2 with G1:", r2_g1_train)
print("Test R2 with G1:", r2_g1_test)

# 1) Does a high R2 here mean G1 is causing G3? 
# A high R2 does not mean that G1 causes G3. It means that G1 is a strong
# predictor of G3 because students who perform well in the first period often
# continue to perform well later. The relationship shows that G1 is a useful predictor of G3, 
# but it does not establish a causal relationship.

# 2) Is this a useful model for identifying students who might struggle? 
# Yes, this model can be useful for identifying students who might struggle
# after G1 becomes available. Students with lower G1 scores can be identified
# as needing additional support before the final grade is determined.

# 3) What might educators need to do if they wanted to intervene early, before G1 is even available?
# Before G1 is available, educators would need to rely on other early indicators
# such as attendance, study habits, participation, homework completion, and
# previous academic performance. These signals may be less accurate than G1,
# but they could help identify students who need support earlier.