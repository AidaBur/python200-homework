import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("student_performance_math.csv", sep=";")

print("Shape:", df.shape)
print(df.head())
print(df.dtypes)

# Histogram
plt.hist(df["G3"], bins=21)

plt.title("Distribution of Final Math Grades")
plt.xlabel("Grade")
plt.ylabel("Count")

plt.savefig("outputs/g3_distribution.png")


# Task 2 

# Remove G3 = 0
df_clean = df[df["G3"] > 0]

print("Original shape:", df.shape)
print("Filtered shape:", df_clean.shape)

# Convert yes/no
yes_no_cols = ["schoolsup", "internet", "higher", "activities"]

for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

# Convert sex
df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

# Correlation check
print("Correlation before filtering:", df["absences"].corr(df["G3"]))
print("Correlation after filtering:", df_clean["absences"].corr(df_clean["G3"]))

# Task 3

import matplotlib.pyplot as plt

# 1. Correlations with G3
numeric_cols = df_clean.select_dtypes(include=["int64", "float64"])

correlations = numeric_cols.corr()["G3"].sort_values()

print("\nCorrelations with G3:")
print(correlations)


# 2. Plot 1: Failures vs G3
plt.figure()

plt.scatter(df_clean["failures"], df_clean["G3"])
plt.title("Failures vs G3")
plt.xlabel("Failures")
plt.ylabel("Final Grade")

plt.savefig("outputs/failures_vs_g3.png")


# 3. Plot 2: Studytime vs G3
plt.figure()

plt.scatter(df_clean["studytime"], df_clean["G3"])
plt.title("Study Time vs G3")
plt.xlabel("Study Time")
plt.ylabel("Final Grade")

plt.savefig("outputs/studytime_vs_g3.png")

# Task 4

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Feature and target
X = df_clean[["failures"]].values
y = df_clean["G3"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print("\nBaseline Model:")
print("Slope:", model.coef_[0])
print("RMSE:", rmse)
print("R^2:", r2)

# Task 5

feature_cols = [
    "failures", "Medu", "Fedu", "studytime",
    "higher", "schoolsup", "internet", "sex",
    "freetime", "activities", "traveltime"
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Metrics
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

print("\nFull Model:")
print("Train R^2:", train_r2)
print("Test R^2:", test_r2)
print("RMSE:", rmse)

# Coefficients
print("\nFeature coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Final Comment:
# The full model performs better than the baseline but still has limited predictive power.
# Train and test R² are close, indicating no overfitting.
# Failures have a strong negative impact on grades.
# Study time, internet access, and intention to pursue higher education positively affect performance.
# The negative coefficient for school support likely reflects that struggling students are more likely to receive help.

# Task 6

import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

plt.figure()
plt.scatter(y_pred, y_test)

# diagonal line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/predicted_vs_actual.png")

# Comment:
# Points close to the diagonal indicate accurate predictions.
# Points above the line mean the actual value is higher than predicted.
# Points below the line mean the model overestimated the value.