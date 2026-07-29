import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target


# --- Preprocessing ---

# Q1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled  = scaler.transform(X_test)  

print("X_train_scaled mean:", X_train_scaled.mean(axis=0))
# We fit the scaler only on X_train to prevent information from the test set from leaking into the training process.

# --- KNN ---

# Q1

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds_unscaled = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds_unscaled))
print(classification_report(y_test, preds_unscaled))

# Q2

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

preds_scaled = knn.predict(X_test_scaled)

print("Accuracy for scaled data:", f"{accuracy_score(y_test, preds_scaled):.4f}")

# Scaling makes all features contribute equally to the distance calculation.
# For the Iris dataset, the features are already on similar scales, so scaling
# makes little difference and slightly lowers the model's accuracy in this case.

# Q3
# note: test set is never used for cross-validation.
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print(cv_scores)           
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")

# This result is more trustworthy than a single train/test split because it ensures 
# that every training example participates in evaluation at some point, the averaged 
# score is more stable than any single split. The standard deviation indicates how consistent the model's 
# performance is across the folds, a lower std means more consistent results across folds.

# Q4

k_values = [1, 3, 5, 7, 9, 11, 13, 15]  

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}")

# I would choose k=5 because it's first of two values with the highest mean accuracy, indicating consistent performance across folds. 

# --- Classifier Evaluation ---

# Q1

cm = confusion_matrix(y_test, preds_unscaled)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.show()
plt.close()

# The model for this dataset has no confusion. Since the Iris dataset is relatively simple 
# and well-separated, the KNN model can classify the species with high accuracy, 
# leading to a confusion matrix that shows perfect classification for all classes and 
# off diagonal elements are zero, indicating no misclassifications.

# --- The sklearn API: Decision Trees ---

# Q1

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

preds_tree = tree.predict(X_test)

print("Accuracy for Decision Tree:", f"{accuracy_score(y_test, preds_tree):.4f}")
print(classification_report(y_test, preds_tree))

print("Accuracy for KNN:", f"{accuracy_score(y_test, preds_unscaled):.4f}")

# Decision Trees do not rely on distance calculations, so scaling the features
# generally does not affect their performance. KNN is distance-based,
# so feature scaling is important for its accuracy.

# --- Logistic Regression and Regularization ---

# Q1
values = [0.01, 1.0, 100]

# Can't use "logistic regression" directly for multi-class classification, 
# so I use "OneVsRestClassifier"to handle the multi-class case or there will be
# an error at run-time.

for c in values:
    log_reg = OneVsRestClassifier(
        LogisticRegression(
            C=c,
            max_iter=1000,
            solver="liblinear"
        )
    )
    log_reg.fit(X_train_scaled, y_train)

    coef_sum = np.abs(np.vstack([est.coef_ for est in log_reg.estimators_])).sum()
    print(f"C={c}: total coefficient magnitude = {coef_sum:.3f}")


# --- PCA ---

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# Q1

print("X digits shape:", X_digits.shape)
print("Images shape:", images.shape)

fig, axes = plt.subplots(1, 10, figsize=(15, 3))

first_indices = {}

for i, label in enumerate(y_digits):
    if label not in first_indices:
        first_indices[label] = i

# One-row subplot showing one example of each digit (0–9).
for digit in range(10):
    index = first_indices[digit]
    axes[digit].imshow(images[index], cmap='gray_r')
    axes[digit].set_title(f"Digit: {digit}")
    axes[digit].axis('off')

plt.savefig("outputs/sample_digits.png")
plt.show()
plt.close()

# Q2
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA 2D Projection of Digits Dataset')
plt.savefig("outputs/pca_2d_projection.png")
plt.show()
plt.close()

# Yes, images of the same digit generally cluster together, although some overlap occurs.

# Q3

cumula_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(range(1, len(cumula_variance) + 1), cumula_variance, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(y=0.8, color='red', linestyle='--', label='80% Variance')
plt.legend()
plt.savefig("outputs/pca_variance_explained.png")
plt.show()
plt.close()


# Approximately 13 components are needed to explain 80% of the variance in the digits dataset, 
# as indicated by the point where the cumulative explained variance curve crosses the 0.8 line.

# Q4


def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for col in range(5):
    axes[0, col].imshow(images[col], cmap='gray')
    axes[0, col].set_title(f"Digit: {col}")
    axes[0, col].axis("off")

components = [2, 5, 15, 40]
for row, n in enumerate(components, start=1):
    for col in range(5):
        reconstructed = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(reconstructed, cmap='gray')
        axes[row, col].axis("off")

row_labels = ["Original", "2 PCs", "5 PCs", "15 PCs", "40 PCs"]

for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, rotation=90, size=12)

plt.tight_layout(rect=[0.08, 0, 1, 1])
plt.savefig("outputs/pca_reconstructions.png")
plt.show()
plt.close()


# Reconstructions improve as more components are added.
# With 2 PCs, images are blurry, but around 15 PCs the digits become somewhat recognizable.
# Using 40 PCs captures more details and produces images closer to the originals.
# This matches the variance curve, where most information is captured before it levels off.