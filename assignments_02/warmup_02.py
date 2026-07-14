
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split


# --- scikit-learn API ---

# Q1

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()         # 1. create
model.fit(years, salary)           # 2. fit
predicted = model.predict(years)   # 3. predict


print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print(f"Predict the salary for 4 years of experience: {model.predict([[4]])[0]:.2f}")
print(f"Predict the salary for 8 years of experience: {model.predict([[8]])[0]:.2f}")

# Q2

# x = np.array([10, 20, 30, 40, 50])

x_values = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
print("x_values shape:", x_values.shape)

# X needs to be 2D for scikit-learn because scikit-learn expects the input features to be in a 2D array
# regardless the number of features for consistency, uses the same input format for every machine learning algorithm so no ambiguity.

# Q3

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

print(X_clusters.shape)

kmeans = KMeans(n_clusters=3, random_state=42)            # 1. Create the model
kmeans.fit(X_clusters)                                    # 2. Fit -- find cluster centers
labels = kmeans.predict(X_clusters)                       # 3. Predict a label for each point
print((kmeans.cluster_centers_))
print(np.bincount(labels)) 
plt.figure(figsize=(6, 5)) 

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis", s=60, alpha=0.7)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="black", marker="X", s=200)

plt.title("K-Means clusters")
plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/kmeans_clusters.png")
plt.show()


# --- Linear Regression ---

# Q1

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)


plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.savefig("outputs/cost_vs_age.png")
plt.show()

# Observation: The scatter plot shows two distinct groups in scatter plot based on smoker status. 
# Smokers (red) generally have higher medical costs than non-smokers (blue) at similar ages.
# This suggests that smoker status is an important variable that affects medical costs.
# Both groups show an increasing trend as age increases.

# Q2

X = age.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, cost, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3

model = LinearRegression()
model.fit(X_train, y_train)
print("Slope:", model.coef_[0])
print("Intercept: ", model.intercept_)

y_pred = model.predict(X_test)

#Below is the manual way, which I can import built-in function: rmse = np.sqrt(mean_squared_error(y_test, y_pred)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

print("RMSE:", rmse)

r2_age_only = model.score(X_test, y_test)
print("r2 for age-only model:", r2_age_only)

# The slope means that for each additional year of age, the predicated medical cost increases by approximately $196.57. 


#Q4

X_full = np.column_stack([age, smoker])

X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(X_full, cost, test_size=0.2, random_state=42)
print("X_train shape:", X_full_train.shape)
print("X_test shape:", X_full_test.shape)
print("y_train shape:", y_full_train.shape)
print("y_test shape:", y_full_test.shape)

model_full = LinearRegression()
model_full.fit(X_full_train, y_full_train)
print("Slope:", model_full.coef_[0])
print("Intercept: ", model_full.intercept_)

y_full_pred = model_full.predict(X_full_test)
print(y_full_pred.shape)
print(y_full_test.shape)

rmse = np.sqrt(np.mean((y_full_pred - y_full_test) ** 2))

print("RMSE:", rmse)

r2_full = model_full.score(X_full_test, y_full_test)
print("r2 for Age and Smoker model:", r2_full)

print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])


# The R2 increases after adding smoker status, from 0.0695 with age alone to 0.773 with age and smoker status.
# This shows that smoker status is an important factor in predicting medical costs.
# Adding smoker status helps the model make better predictions.

# The smoker coefficient means that smokers are expected to have higher medical costs than non-smokers, 
# even if they are the same age.

#Q5

plt.scatter(y_full_pred, y_full_test)

min_val = min(y_full_pred.min(), y_full_test.min())
max_val = max(y_full_pred.max(), y_full_test.max())

plt.plot([min_val, max_val], [min_val, max_val], color="red")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig("outputs/predicted_vs_actual.png")
plt.show()

# If a point is above the diagonal, the actual medical cost is higher than the predicted cost (Actual > Predicted),
# so the model underestimated the cost.

# If a point is below the diagonal, the predicted medical cost is higher than the actual cost (Predicted > Actual),
# so the model overestimated the cost.