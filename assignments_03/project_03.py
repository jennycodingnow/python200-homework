import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Task 1

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print(f"Data shape: {df.shape}")
print("Data info:")
df.info()
print(f"First 5 rows:\n{df.head()}")
print(f" Count of spam vs. non-spam emails:\n{df['spam_label'].value_counts()}")

# The dataset contains 4601 emails.
# There are 2788 non-spam emails (60.6%) and 1813(39.4%) spam emails so there a slight class imbalance in the dataset. Accuracy alone
# may not be a good metric to evaluate the model's performance. Other metrics like precision, recall, and F1-score will need to 
# be considered to evaluate the model's performance. The heavy skew toward zero means that most of the emails
# do not have the features that are indicative of spam. 
# The numeric scale vary so dramatically across features because
# there extremem outliers in the dataset. For example, the feature "capital_run_length_longest" can reach values in the thousands, while most frequency features are tiny fractions. 
# It matters for some of models because it will skew the model's learning process, causing it to focus on the bigger one and ignore
# the smaller ones.  

non_spam = df[df["spam_label"] == 0]
spam = df[df["spam_label"] == 1]
features = ["word_freq_free", "char_freq_!", "capital_run_length_total"]

for feature in features:
    plt.figure(figsize=(6, 4))
    plt.boxplot([non_spam[feature], spam[feature]], tick_labels = ["Non-Spam", "Spam"])
    plt.ylabel(feature)
    plt.title(f"Distribution of {feature} by Spam Label")
    plt.savefig(f"outputs/{feature}_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# I notice that the distributions of the features "word_freq_free", "char_freq_!", and "capital_run_length_total" 
# are quite different between spam and non-spam emails. 
# For example, spam emails tend to have higher frequencies of the word "free" and 
# exclamation marks, as well as longer runs of capital letters. 
# These features could be useful for distinguishing between spam and non-spam emails but 
# they overlap considerably and have big outliers that need further analysis.


# Task 2
# Prepare Your Data

X = df.drop("spam_label", axis=1)
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# In this dataset, most frequency features are tiny fractions, while the feature
# "capital_run_length_longest" can reach values in the thousands.
# PCA searches for directions of maximum variance.
# Without scaling, the model will be biased towards features with much larger numeric ranges.
# These features would dominate the main components. Therefore, the data are standardized before PCA.

# PCA preprocessing

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

pca = PCA()

pca.fit(X_train_scaled)

cumula_explained = np.cumsum(pca.explained_variance_ratio_ * 100)
plt.plot(range(1, len(cumula_explained) + 1), cumula_explained, marker='o')
plt.xlabel("Number Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(y=90, color='red', linestyle='--', label='90% Variance')
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pca_variance_explained_90.png")
plt.show()
plt.close()

n = np.argmax(cumula_explained >= 90) + 1
print("Number of components that reach 90% variance:", n)

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]

# Task 3

print("------------Classification Models------------")


# KNN Classifier-Unscaled Data
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)

predict_unscaled = knn_unscaled.predict(X_test)

print("Accuracy for unscaled data Knn:", f"{accuracy_score(y_test, predict_unscaled):.4f}")
print("Classification report for unscaled data KNN:")
print(classification_report(y_test, predict_unscaled))

# KNN Classifier-Scaled Data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
predict_scaled = knn_scaled.predict(X_test_scaled)

# KNN Classifier-PCA Data
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
predict_pca = knn_pca.predict(X_test_pca)

print("Accuracy for scaled data KNN:", f"{accuracy_score(y_test, predict_scaled):.4f}")
print("Classification report for scaled data KNN:")
print(classification_report(y_test, predict_scaled))

print("Accuracy for PCA data KNN:", f"{accuracy_score(y_test, predict_pca):.4f}")
print("Classification report for PCA data KNN:")
print(classification_report(y_test, predict_pca))

# Scaled KNN and PCA KNN have similar accuracy and better than unscaled KNN.
# The Scaled KNN has a slightly better accuracy (0.9077) than PCA KNN (0.9066).

# Decision Tree Classifier

max_depths = [3, 5, 10, None]
for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    predict_train = tree.predict(X_train)
    predict_test = tree.predict(X_test)

    accuracy_train = accuracy_score(y_train, predict_train)
    accuracy_test = accuracy_score(y_test, predict_test)


    print(f"Accuracy for Decision Tree with max_depth={depth}:")
    print(f"  Training: {accuracy_train:.4f}")
    print(f"  Test: {accuracy_test:.4f}")

# As the maximum depth increases, training accuracy consistently increases and test accuracy
# also increases but more slowly. This indicates that as the tree grows deeper, it can
# capture more patterns in the data, improving performance on both the training and test sets.
# However, the gap between the training and test accuracy also grows larger, suggesting that
# deeper trees are beginning to overfit the training data.

# I would choose a maximum depth of 10 to use in production because this depth provides a good
# balance between model complexity and performance. A smaller depth may underfit the data by
# not capturing enough patterns, while larger depths increase the risk of overfitting.

best_depth = 10
best_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
best_tree.fit(X_train, y_train)
best_tree_predict = best_tree.predict(X_test)
best_tree_accuracy = accuracy_score(y_test, best_tree_predict)
print(f"Best Decision Tree Accuracy with max_depth={best_depth}: {best_tree_accuracy:.4f}")
print(f"Classification report for Decision Tree with max_depth={best_depth}:")
print(classification_report(y_test, best_tree_predict))

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
predict_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, predict_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print("Classification report for Random Forest:")
print(classification_report(y_test, predict_rf))


#Logistic Regression Classifier-Scaled Data

log_reg =LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg.fit(X_train_scaled, y_train)

predict_log = log_reg.predict(X_test_scaled)

print(f"Logistic Regression scaled data Accuracy: {accuracy_score(y_test, predict_log):.4f}")
print("Classification report for Logistic Regression:")
print(classification_report(y_test, predict_log))

#Logistic Regression Classifier-PCA-Reduced Data

log_regr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_regr_pca.fit(X_train_pca, y_train)

predict_log_pca = log_regr_pca.predict(X_test_pca)

accuracy_log_pca = accuracy_score(y_test, predict_log_pca)

print(f"Logistic Regression (PCA) Accuracy: {accuracy_log_pca:.4f}")
print("Classification Report:")
print(classification_report(y_test, predict_log_pca))


# Random Forest: 0.9457 
# Logistic Regression: 0.9294
# Decision Tree: 0.9088
# KNN (scaled): 0.9077

# The Random Forest model performs the best with accuracy 0.9457. 
# The Logistic Regression model is the second best to perform with high accuracy above 0.929. 
# The Logistic Regression model trained on the scaled features performed slightly better than the model trained on the PCA-reduced features. Although PCA reduced 
# the dimensionality of the data, it also removed some information that was useful for classification. In this case, 
# scaling alone provided the best performance. PCA vs. non-PCA, PCA worked better. Yes, it does that match the hypothesis from Task 2 because
# after scaling, it improves the performance of the model. 
# The Decision Tree with max_depth 10 and max_depth None have high accuracy on training data but lower accuracy on 
# test data, indicating overfitting.

# For spam filter, I would prefer minimizing false negatives because phishing and malware 
# emails can pose serious security risks. Allowing spam into the inbox could expose users to scams, 
# so I would rather block more suspicious emails even if it occasionally results in legitimate emails being flagged.
# Users can check their spam folder if they didn't receive an expected email. 


cm_rf = confusion_matrix(y_test, predict_rf)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["Not Spam", "Spam"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix for the Best Model (Random Forest)")
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.show()
plt.close()


# Based on the confusion matrix, the Random Forest classifier made more false negatives (32 spam emails that get through) than
# false positives (18 legitimate emails marked as spam). Since I prioritized minimizing false negatives for a spam filter, this result 
# shows that the model still allows some spam messages through. Although the number of false positives is 
# lower, reducing false negatives would be important because missed spam or phishing emails could create 
# security risks for users.


print("Feature importances for the best Decision Tree:")
importances_dt = best_tree.feature_importances_
feature_importance_dt = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances_dt
})

feature_importance_dt = feature_importance_dt.sort_values(by="Importance", ascending=False)
print(feature_importance_dt.head(10))


print("\nFeature importances for the Random Forest:")
importances_rt = rf.feature_importances_
feature_importance_rt = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances_rt
})

feature_importance_rt = feature_importance_rt.sort_values(by="Importance", ascending=False)
print(feature_importance_rt.head(10))

plt.bar(feature_importance_rt['Feature'].head(10), feature_importance_rt['Importance'].head(10), color=['green', 'orange', 'blue', 'red'])
plt.title("Top 10 Feature Importances from Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.show()
plt.close()

# These models do not agree on the most important features. The Decision Tree model 
# indicates that "char_freq_$" is the most important feature (0.388731), 
# while the Random Forest model indicates that "char_freq_!" is the most important. 
# These results do not match my intuition about what makes an email spam because
# I thought the most important feature would be "capital_run_length_total". A lot of 
# spam emails have capital letters to grab attention in emails.

#Task 4

print("------------Cross-Validation------------")

# KNeighborsClassifier Scaled Data
knn_unscaled_scores = cross_val_score(knn_unscaled, X_train, y_train, cv=5)

print(f"Unscaled KNN Scores: {knn_unscaled_scores}")           
print(f"Mean: {knn_unscaled_scores.mean():.3f}")
print(f"Std:  {knn_unscaled_scores.std():.3f}")

# KNeighborsClassifier Scaled Data
knn_scaled_scores = cross_val_score(knn_scaled, X_train_scaled, y_train, cv=5)

print(f"Scaled KNN Scores: {knn_scaled_scores}")
print(f"Mean: {knn_scaled_scores.mean():.3f}")
print(f"Std:  {knn_scaled_scores.std():.3f}")

# KNeighborsClassifier PCA Data
knn_pca_scores = cross_val_score(knn_pca, X_train_pca, y_train, cv=5)

print(f"PCA KNN Scores: {knn_pca_scores}")
print(f"Mean: {knn_pca_scores.mean():.3f}")
print(f"Std:  {knn_pca_scores.std():.3f}")

# Decision Tree Classifier best_depth=10
dt_scores = cross_val_score(best_tree, X_train, y_train, cv=5)

print(f"Decision Tree Scores: {dt_scores}")
print(f"Mean: {dt_scores.mean():.3f}")
print(f"Std:  {dt_scores.std():.3f}")

# RandomForestClassifier
rf_scores = cross_val_score(rf, X_train, y_train, cv=5)

print(f"Random Forest Scores: {rf_scores}")
print(f"Mean: {rf_scores.mean():.3f}")
print(f"Std:  {rf_scores.std():.3f}")

# LogisticRegression-scaled Data
log_reg_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=5)

print(f"Logistic Regression Scores: {log_reg_scores}")
print(f"Mean: {log_reg_scores.mean():.3f}")
print(f"Std:  {log_reg_scores.std():.3f}")

# LogisticRegression-PCA Data
log_reg_pca_scores = cross_val_score(log_regr_pca, X_train_pca, y_train, cv=5)

print(f"Logistic Regression (PCA) Scores: {log_reg_pca_scores}")
print(f"Mean: {log_reg_pca_scores.mean():.3f}")
print(f"Std:  {log_reg_pca_scores.std():.3f}")

# Random Forest is the most accurate model. It has the highest mean accuracy (0.954) and modest standard deviation (0.013).
# The most stable model is the Logistic Regression (PCA). It has the lowest standard deviation of 0.003, indicating the 
# lowest variance across folds. Yes, the ranking match the accuracy ranking from Task 3. The Random Forest model is the most 
# accurate.

#Task 5

print("------------Pipelines------------")
#Best tree-based classifier pipeline
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(random_state=42))
])

rf_pipeline.fit(X_train, y_train)
predict_rf_p = rf_pipeline.predict(X_test)

print(f"Accuracy score for the best tree-based classifier: {accuracy_score(y_test, predict_rf_p):.4f}")
print(classification_report(y_test, predict_rf_p))

#Best non-tree classifier pipeline
log_pipeline = Pipeline([
    ("scaler",     StandardScaler()), 
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

log_pipeline.fit(X_train, y_train)
predict_log_p = log_pipeline.predict(X_test)

print(f"Accuracy score for best non-tree classifier (Logistic Regression): {accuracy_score(y_test, predict_log_p):.4f}")
print(classification_report(y_test, predict_log_p))


# PCA did not improve the performance of the non-tree model, so it was not included in the Logistic Regression pipeline.
# PCA is not used in the Random Forest pipeline because tree-based models are not sensitive to feature scaling so PCA is unnecessary.
# The pipeline results matched the earlier manual implementation, producing the same accuracy and classification report.
# The two pipelines have different structures because Random Forest is a tree-based model that does not require feature
# scaling, while Logistic Regression performs better when the features are scaled and standardized.
# Using pipelines combines preprocessing and model training into a single workflow, making the code more organized,
# reproducible, and easier to maintain, share with others, and deploy.