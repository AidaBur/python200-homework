# --- scikit-learn API ---
# Q1

import numpy as np
from sklearn.linear_model import LinearRegression

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

predictions = model.predict([[4], [8]])

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediction for 4 years:", predictions[0])
print("Prediction for 8 years:", predictions[1])


# Q2

x = np.array([10, 20, 30, 40, 50])

print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)

print("Reshaped shape:", x_2d.shape)

# Comment:
# scikit-learn requires X to be 2D because it treats input as a table
# with rows (samples) and columns (features).


# Q3

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)

labels = kmeans.predict(X_clusters)

print("Cluster centers:")
print(kmeans.cluster_centers_)

print("Points per cluster:")
print(np.bincount(labels))

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker='x'
)

plt.title("KMeans Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.savefig("outputs/kmeans_clusters.png")

# Comment:
# KMeans groups data into clusters by finding cluster centers
# and assigning each point to the nearest center.

# --- Linear Regression ---
# Q1

import os
from sklearn.model_selection import train_test_split

os.makedirs("outputs", exist_ok=True)

np.random.seed(42)
num_patients = 100

age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

plt.figure()
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")
plt.savefig("outputs/cost_vs_age.png")

# Comment:
# The scatter plot shows two visible groups of patients.
# Smokers generally have much higher medical costs than non-smokers.
# This suggests that smoker status is an important feature for predicting cost.


# Q2

X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nLinear Regression Q2:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Q3

model_age = LinearRegression()
model_age.fit(X_train, y_train)

y_pred = model_age.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2_age = model_age.score(X_test, y_test)

print("\nLinear Regression Q3:")
print("Slope:", model_age.coef_[0])
print("Intercept:", model_age.intercept_)
print("RMSE:", rmse)
print("R^2:", r2_age)

# Comment:
# The slope means that for each additional year of age, the model predicts
# medical cost will increase by about this many dollars.
# Since this model only uses age, it misses the important difference between
# smokers and non-smokers.


# Q4

X_full = np.column_stack([age, smoker])

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

y_pred_full = model_full.predict(X_test_full)
r2_full = model_full.score(X_test_full, y_test_full)

print("\nLinear Regression Q4:")
print("Age-only R^2:", r2_age)
print("Age + smoker R^2:", r2_full)
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# Comment:
# Adding the smoker flag helps because smoker status has a large effect on cost.
# The smoker coefficient represents the estimated increase in annual medical cost
# for smokers compared with non-smokers, holding age constant.


# Q5

plt.figure()
plt.scatter(y_pred_full, y_test_full)
plt.plot(
    [y_test_full.min(), y_test_full.max()],
    [y_test_full.min(), y_test_full.max()]
)
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig("outputs/predicted_vs_actual.png")

# Comment:
# A point above the diagonal means the actual cost was higher than predicted,
# so the model underpredicted that patient's cost.
# A point below the diagonal means the actual cost was lower than predicted,
# so the model overpredicted that patient's cost.
