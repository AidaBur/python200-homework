import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Pre-preprocessing:
# The CSV file uses a semicolon (;) as a separator instead of a comma,
# so we must pass sep=";" to pd.read_csv().

# --- Task 1: Load and Explore ---

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

# Comment:
# The histogram shows a group of students with G3 = 0,
# which represents students who did not take the final exam.

# --- Task 2: Preprocess the Data ---

df_clean = df[df["G3"] > 0]

print("Original shape:", df.shape)
print("Filtered shape:", df_clean.shape)

# Comment:
# We remove G3 = 0 because these students did not take the exam.
# Keeping them would distort the model since 0 is not a real grade.

# Convert yes/no to 1/0
yes_no_cols = ["schoolsup", "internet", "higher", "activities"]

for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

# Convert sex
df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

# Correlation comparison
print("Correlation before filtering:", df["absences"].corr(df["G3"]))
print("Correlation after filtering:", df_clean["absences"].corr(df_clean["G3"]))

# Comment:
# Before filtering, students with G3=0 had high absences,
# which distorted the relationship.
# After filtering, absences show a clearer negative relationship with grades.

# --- Task 3: EDA ---

numeric_cols = df_clean.select_dtypes(include=["int64", "float64"])
correlations = numeric_cols.corr()["G3"].sort_values()

print("\nCorrelations with G3:")
print(correlations)

# Comment:
# Failures have the strongest negative correlation with G3.
# Study time and parents' education show positive relationships.
# G1 and G2 have very strong correlation with G3.

# Plot 1
plt.figure()
plt.scatter(df_clean["failures"], df_clean["G3"])
plt.title("Failures vs G3")
plt.xlabel("Failures")
plt.ylabel("Final Grade")
plt.savefig("outputs/failures_vs_g3.png")

# Comment:
# Students with more past failures tend to have lower grades.

# Plot 2
plt.figure()
plt.scatter(df_clean["studytime"], df_clean["G3"])
plt.title("Study Time vs G3")
plt.xlabel("Study Time")
plt.ylabel("Final Grade")
plt.savefig("outputs/studytime_vs_g3.png")

# Comment:
# More study time is associated with slightly higher grades.

# --- Task 4: Baseline Model ---

X = df_clean[["failures"]].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print("\nBaseline Model:")
print("Slope:", model.coef_[0])
print("RMSE:", rmse)
print("R^2:", r2)

# Comment:
# The model using only failures performs poorly.
# R² is very low, meaning failures alone cannot explain student performance well.
# RMSE (~3) means predictions are off by about 3 points on a 0–20 scale.

# --- Task 5: Full Model ---

feature_cols = [
    "failures", "Medu", "Fedu", "studytime",
    "higher", "schoolsup", "internet", "sex",
    "freetime", "activities", "traveltime"
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

print("\nFull Model:")
print("Train R^2:", train_r2)
print("Test R^2:", test_r2)
print("RMSE:", rmse)

print("\nFeature coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Comment:
# The full model performs better than the baseline.
# Train and test R² are close, indicating no overfitting.
# Failures strongly decrease grades.
# Study time and internet access improve performance.
# The negative schoolsup coefficient likely reflects that struggling students receive extra support.

# --- Task 6: Evaluation Plot ---

plt.figure()
plt.scatter(y_pred, y_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/predicted_vs_actual.png")

# Comment:
# Points close to the diagonal indicate accurate predictions.

# --- FINAL REQUIRED STEP: Add G1 ---

feature_cols_with_g1 = feature_cols + ["G1"]

X = df_clean[feature_cols_with_g1].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

test_r2_g1 = model.score(X_test, y_test)

print("\nModel with G1:")
print("Test R^2:", test_r2_g1)

# Comment:
# Adding G1 dramatically increases R² because previous grades strongly predict final grades.
# However, this does not mean G1 causes G3 — it is simply a very strong indicator.
# This model is less useful for early intervention, since G1 is already a later-stage outcome.