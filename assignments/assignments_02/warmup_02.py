import numpy as np
from sklearn.linear_model import LinearRegression

# Data
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# 1. Create model
model = LinearRegression()

# 2. Fit model
model.fit(years, salary)

# 3. Predict
predictions = model.predict([[4], [8]])

# 4. Get parameters
slope = model.coef_[0]
intercept = model.intercept_

# 5. Print results
print("Slope:", slope)
print("Intercept:", intercept)
print("Prediction for 4 years:", predictions[0])
print("Prediction for 8 years:", predictions[1])

# --- scikit-learn API ---
# Q2

import numpy as np

x = np.array([10, 20, 30, 40, 50])

print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)

print("Reshaped to 2D:", x_2d.shape)

# Q3

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate data
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# Create model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit model
kmeans.fit(X_clusters)

# Predict clusters
labels = kmeans.predict(X_clusters)

# Print centers
print("Cluster centers:")
print(kmeans.cluster_centers_)

# Print counts
print("Points per cluster:")
print(np.bincount(labels))

# Plot points
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels)

# Plot centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker='x'
)

# Labels
plt.title("KMeans Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Save
plt.savefig("outputs/kmeans_clusters.png")

# --- Linear Regression ---
# Q1

import os
import numpy as np
import matplotlib.pyplot as plt

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

# Q2

from sklearn.model_selection import train_test_split

X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3

from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression()

model.fit(X_train, y_train)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print("RMSE:", rmse)

r2 = model.score(X_test, y_test)
print("R^2:", r2)

# Q4

X_full = np.column_stack([age, smoker])

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y,
    test_size=0.2,
    random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)

r2_full = model_full.score(X_test, y_test)
print("R^2 with smoker:", r2_full)

print("age coefficient:   ", model_full.coef_[0])
print("smoker coefficient:", model_full.coef_[1])

# Q5

y_pred = model_full.predict(X_test)

plt.scatter(y_pred, y_test)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/predicted_vs_actual.png")
