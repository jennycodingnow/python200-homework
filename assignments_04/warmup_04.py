import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    f1_score,
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- ROC and AUC --- 
# Q1

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

log_reg_y_proba = log_reg.predict_proba(X_test)[:, 1]

log_reg_auc = roc_auc_score(y_test, log_reg_y_proba)
print(f"AUC for Logistic Regression(unscaled): {log_reg_auc:.3f}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
knn_y_proba = knn_scaled.predict_proba(X_test_scaled)[:, 1]

knn_auc = roc_auc_score(y_test, knn_y_proba)
print(f"AUC for K-Nearest Neighbors(scaled): {knn_auc:.3f}")

# K-Nearest Neighbors has the higher AUC, indicating it separates the two
# classes better overall. A higher AUC means it distinguishes between the
# positive and negative classes more effectively across all possible
# classification thresholds, rather than at any single threshold.

# Q2

fpr_log_reg, tpr_log_reg, thresholds_log_reg = roc_curve(y_test, log_reg_y_proba)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, knn_y_proba)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr_log_reg, tpr=tpr_log_reg).plot(ax=ax, name=f"Logistic Regression (AUC={log_reg_auc:.2f})")
RocCurveDisplay(fpr=fpr_knn, tpr=tpr_knn).plot(ax=ax, name=f"KNN k=5 (AUC={knn_auc:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
ax.set_title("ROC Comparison — Classifiers")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/roc_comparison.png")
plt.show()
plt.close()

# At the point on each curve where TPR about 0.80, KNN model has a lower FPR (about 0.01) than Logistic Regression (about 0.06).
# This means that if we wanted to correctly identify about 80% of the positive
# cases, KNN would produce fewer false positives (false alarms) than Logistic Regression.

# Q3

optimum_f1 = 0
optimum_threshold = 0
optimum_index = 0
for i, threshold in enumerate(thresholds_log_reg):
    y_pred = (log_reg_y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    if f1 > optimum_f1:
        optimum_f1 = f1
        optimum_threshold = threshold
        optimum_index = i

print(f"Best threshold: {optimum_threshold:.3f}")
print(f"Best F1 score: {optimum_f1:.3f}")
print(f"TPR at best threshold: {tpr_log_reg[optimum_index]:.3f}")
print(f"FPR at best threshold: {fpr_log_reg[optimum_index]:.3f}")

# The optimal threshold (0.276) is lower than the default threshold of 0.5.
# This means the model classifies more observations as positive, increasing
# recall at the expense of more false positives.
#
# In real-world applications, you might choose a threshold below 0.5 when
# missing a positive case is more costly than generating extra false alarms,
# such as in medical diagnosis, fraud detection, or disease screening.

# --- GridSearchCV --- 

# Q1

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])

param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print(f"Best C for Logistic Regression:      {grid_search.best_params_['clf__C']}")
print(f"Best CV AUC for Logistic Regression: {grid_search.best_score_:.3f}")

best_pipe = grid_search.best_estimator_

y_pred  = best_pipe.predict(X_test)         
y_probs = best_pipe.predict_proba(X_test)[:, 1]
best_auc = roc_auc_score(y_test, y_probs)

print(f"Test AUC for the best model: {best_auc:.3f}")



default_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
])
default_pipe.fit(X_train, y_train)
default_y_probs = default_pipe.predict_proba(X_test)[:, 1]
default_auc = roc_auc_score(y_test, default_y_probs)

print(f"Default C=1.0 Test AUC: {default_auc:.3f}")

change = best_auc - default_auc
print(f"Change in AUC: {change:.3f}")

# The grid search selected C=100.0 instead of the default C=1.0. 
# The test AUC did not change (0.706) for both models, 
# suggesting that the increased model flexibility from weaker regularization 
# did not improve the overall performance on unseen data."

# Q2

pipe = Pipeline([
    ("clf",    DecisionTreeClassifier(random_state=42)),
])

param_grid = {
    "clf__max_depth": [2, 3, 5, 8, None]
}

grid_search_dt = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search_dt.fit(X_train, y_train)

print(f"Best max_depth for Decision Tree: {grid_search_dt.best_params_['clf__max_depth']}")
print(f"Best CV AUC for Decision Tree: {grid_search_dt.best_score_:.3f}")

best_pipe_dt = grid_search_dt.best_estimator_

y_pred_dt  = best_pipe_dt.predict(X_test)         
y_probs_dt = best_pipe_dt.predict_proba(X_test)[:, 1]
best_auc_dt = roc_auc_score(y_test, y_probs_dt)

print(f"Test AUC for the best Decision Tree: {best_auc_dt:.3f}")

# Logistic Regression achieved a test AUC of 0.773, while Decision Tree achieved
# a test AUC of 0.917. I would choose the Decision Tree because it has a higher AUC.
# This indicates that it separates the two classes better across different classification thresholds.
# However, AUC would not be the only factor I would consider when selecting a
# model for further development. I would also evaluate metrics such as precision,
# recall, and F1-score, as well as factors like interpretability and the impact
# of false positives and false negatives.

# Q3

results = grid_search_dt.cv_results_

print("Keys in the results dictionary:")
print(results.keys())

print("Detailed results for each combination of parameters:")
for rank, depth, mean, std in sorted(
    zip(
        results["rank_test_score"], 
        results["param_clf__max_depth"], 
        results["mean_test_score"], 
        results["std_test_score"]
        )
    ):
        print(f"Rank: {rank}, Depth: {depth}, Mean: {mean:.3f}, Std: {std:.3f}")

# Max_depth=5 and max_depth=3 have similar mean CV AUC scores (0.917 and
# 0.902, respectively), with low standard deviations (0.021 and 0.019).
# I would choose max_depth=5 because it achieves a higher mean AUC while
# only having a slightly higher standard deviation, making it the better
# overall trade-off between performance and consistency.


# --- joblib --- 

# Q1

joblib.dump(best_pipe, "models/warmup_model.pkl")
print("Model saved.")

loaded_clf = joblib.load("models/warmup_model.pkl")

original_preds = best_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# If only the Logistic Regression model were saved without the StandardScaler,
# calling predict(X_test) on the unscaled test data could produce incorrect
# predictions because the model was trained on scaled features. Saving the
# entire pipeline preserves both the scaler and the classifier, ensuring that
# new data is transformed consistently before predictions are made.

# Q2
# --- Simulated prediction script ---

clf_joblib = joblib.load("models/warmup_model.pkl")

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

predictions = clf_joblib.predict(new_samples)
probabilities = clf_joblib.predict_proba(new_samples)[:, 1]

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = "class 1" if pred == 1 else "class 0"
    print(f"Day {i+1}: {label} ({prob:.2f} probability)")
    
# The all-zeros row is predicted as class 1 with a probability of about 0.65.
# This happens because the model uses the patterns
# it learned during training to make predictions. A row with all features equal
# to zero does not always mean the prediction will be neutral because the model
# also considers the learned feature weights and the intercept.