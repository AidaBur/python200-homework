import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

os.makedirs("outputs", exist_ok=True)

# Pre-preprocessing:
# This dataset is separated by semicolons, not commas.
# Because of that, I need to use sep=";" when reading the CSV file.

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

df_clean = df[df["G3"] > 0].copy()

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
# R^2 is very low, meaning failures alone cannot explain student performance well.
# RMSE (~3) means predictions are off by about 3 points on a 0-20 scale.

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
# Points close to the diagonal line mean the model predicted the grade accurately.
# Points above the diagonal mean the actual grade was higher than predicted,
# so the model underpredicted the student's grade.
# Points below the diagonal mean the actual grade was lower than predicted,
# so the model overpredicted the student's grade.
# The errors appear spread across both low and high grades instead of clustering
# only at one end, so the model does not seem to struggle only with low grades
# or only with high grades.


# --- Neglected Feature: The Power of G1 ---

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
# Adding G1 dramatically increases R² because G1 is the student's first-period grade.
# Since G1 and G3 are both grades for the same student in the same class,
# they are very strongly related.
#
# This does not mean G1 causes G3. It means G1 is a strong predictor of G3.
# A student who performs well early in the course is likely to perform well later,
# but the model does not prove that the earlier grade directly caused the final grade.
#
# A model that relies on G1 can help flag students after the first-period grade
# is available, but it is less useful for very early intervention.
# To intervene before G1 exists, educators would need earlier information such as
# absences, past failures, study time, school support, family background, or
# earlier diagnostic assessments collected before the first grading period.


# --- Final Summary ---
# The original dataset has 395 students. After filtering out students with G3 = 0,
# the dataset has 357 students. With an 80/20 train-test split, the test set has
# 72 students. I removed the G3 = 0 rows because a final grade of 0 most likely
# means the student did not take the final exam, rather than earning a real score of 0.

#
# RMSE tells us the average size of the model's prediction error.
# Since G3 is measured on a 0-20 scale, an RMSE around 3 means the model's
# predictions are usually off by about 3 grade points.
#
# R^2 tells us how much of the variation in final grades the model explains.
# A low R^2 means the model does not explain student grades very well.
# A higher R^2 means the model captures more of the patterns in the data.
#
# The best model is the model that includes G1. Its test R^2 is about 0.75.
# In plain language, this means the model explains about 75% of the variation
# in final grades on the test set. This is much higher than the full model
# without G1 because first-period grades are closely related to final grades.


# In the full model, the largest positive coefficient is higher.
# This means students who want to pursue higher education tend to have higher
# final grades, holding the other model features constant.
#
# The largest negative coefficient is failures, making it the strongest negative
# predictor in this model. This means students with more past class failures
# tend to have lower final grades, holding the other model features constant.

#
# However, these coefficients show relationships, not definite causes.
# For example, a feature with a positive coefficient may be connected to other
# advantages in a student's life, and school support may have a negative coefficient
# because struggling students are more likely to receive extra support.

#
# One surprising result is that schoolsup has a negative coefficient.
# At first, I expected school support to be linked with higher grades.
# A likely explanation is that students who receive school support may already
# be struggling, so this feature reflects existing academic difficulty rather
# than support causing lower grades.

