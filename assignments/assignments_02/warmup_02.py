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