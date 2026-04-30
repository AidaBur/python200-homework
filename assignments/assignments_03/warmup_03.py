import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier


from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

os.makedirs("outputs", exist_ok=True)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---
# Q1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q2

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nMean of each column in X_train_scaled:")
print(X_train_scaled.mean(axis=0))

# Comment:
# We fit the scaler only on X_train so the test set stays unseen until evaluation.

# --- KNN ---
# Q1

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("\nKNN Q1: Unscaled Data")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification report:")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))


# Q2

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

print("\nKNN Q2: Scaled Data")
print("Accuracy:", accuracy_score(y_test, y_pred_knn_scaled))

# Scaling hurt performance slightly here: unscaled KNN scored 1.0, while scaled
# KNN scored about 0.93. This can happen because the Iris features are already
# on similar scales, and scaling changes the distance relationships between points.



# Q3

cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print("\nKNN Q3: 5-Fold Cross-Validation")
print("Fold scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())

# Comment:
# Cross-validation is more trustworthy than a single train/test split because it
# evaluates the model across multiple different validation folds.


# Q4

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = None
best_score = 0

print("\nKNN Q4: Compare k Values")

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_train, y_train, cv=5)
    mean_score = scores.mean()

    print("k:", k, "mean CV score:", mean_score)

    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print("Chosen k:", best_k)

# Comment:
# I would choose k=5 because it has the highest mean cross-validation score.
# k=7 has the same score, but k=5 is slightly simpler because it uses fewer neighbors.


# --- Classifier Evaluation ---
# Q1

cm = confusion_matrix(y_test, y_pred_knn)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix")
plt.savefig("outputs/knn_confusion_matrix.png")

# Comment:
# The confusion matrix shows which Iris species were classified correctly or incorrectly.
# If there are mistakes, they are most likely between versicolor and virginica,
# because those two species have more similar measurements than setosa.

# --- Decision Trees ---
# Q1

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

print("\nDecision Trees Q1")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Classification report:")
print(classification_report(y_test, y_pred_tree, target_names=iris.target_names))

# Comment:
# The Decision Tree accuracy is about 0.97, which is slightly lower than the
# unscaled KNN accuracy. Both models still perform well on the Iris dataset.

# Comment:
# Scaling should not affect a Decision Tree much because trees split features
# using thresholds and do not rely on distance calculations like KNN.

# --- Logistic Regression and Regularization ---
# Q1

c_values = [0.01, 1.0, 100]

print("\nLogistic Regression Q1")

for c in c_values:
    log_model = OneVsRestClassifier(
        LogisticRegression(
            C=c,
            max_iter=1000,
            solver="liblinear"
        )
    )

    log_model.fit(X_train_scaled, y_train)

    coef_size = np.abs(log_model.estimators_[0].coef_).sum()
    coef_size += np.abs(log_model.estimators_[1].coef_).sum()
    coef_size += np.abs(log_model.estimators_[2].coef_).sum()

    print("C:", c)
    print("Total coefficient magnitude:", coef_size)

# Comment:
# As C increases from 0.01 to 100, the total coefficient magnitude increases.
# This shows that smaller C means stronger regularization, which keeps the
# coefficients smaller. Larger C means weaker regularization, so the model is
# allowed to use larger coefficients.

# --- PCA ---

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images = digits.images    # same data shaped as 8x8 images for plotting


# Q1

print("\nPCA Q1")
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

plt.figure(figsize=(10, 2))

for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]

    plt.subplot(1, 10, digit + 1)
    plt.imshow(images[idx], cmap="gray_r")
    plt.title(str(digit))
    plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_digits.png")


# Q2

pca = PCA()
pca.fit(X_digits)

scores = pca.transform(X_digits)

plt.figure()
scatter = plt.scatter(
    scores[:, 0],
    scores[:, 1],
    c=y_digits,
    cmap="tab10",
    s=10
)
plt.colorbar(scatter, label="Digit")
plt.title("PCA 2D Projection of Digits")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("outputs/pca_2d_projection.png")

# Comment:
# Same-digit images tend to form loose clusters in the 2D PCA space.
# The clusters are not perfectly separated because two components cannot capture
# all variation in the 64 original pixel features.

# Q3

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(
    np.arange(1, len(cumulative_variance) + 1),
    cumulative_variance
)
plt.axhline(0.80, linestyle="--")
plt.title("PCA Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.savefig("outputs/pca_variance_explained.png")

components_80 = np.argmax(cumulative_variance >= 0.80) + 1

print("\nPCA Q3")
print("Components needed for 80% variance:", components_80)

# Comment:
# The plot shows that about 13 principal components are needed to explain
# 80% of the variance in the digit images.

# Q4

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


n_values = [2, 5, 15, 40]

plt.figure(figsize=(10, 8))

# Original row
for col in range(5):
    plt.subplot(len(n_values) + 1, 5, col + 1)
    plt.imshow(images[col], cmap="gray_r")
    plt.title("Original")
    plt.axis("off")

# Reconstruction rows
for row, n in enumerate(n_values):
    for col in range(5):
        reconstructed = reconstruct_digit(col, scores, pca, n)

        plot_index = (row + 1) * 5 + col + 1
        plt.subplot(len(n_values) + 1, 5, plot_index)
        plt.imshow(reconstructed, cmap="gray_r")
        plt.title("n=" + str(n))
        plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")

# Comment:
# The digits become clearly recognizable around n=15 components.
# This matches the variance plot because around that range the curve has already
# captured most of the important structure and begins to level off.
