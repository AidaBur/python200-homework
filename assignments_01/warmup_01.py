import pandas as pd

# --- Pandas ---

# Pandas Q1

# Create DataFrame
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)

# First 3 rows
print("First 3 rows:")
print(df.head(3))

# Shape
print(f"Shape: {df.shape}")

# Data types
print("Data types:")
print(df.dtypes)

# Pandas Q2

# Filter students who passed and have grade > 80
filtered_df = df[(df["passed"] == True) & (df["grade"] > 80)]

print("Students who passed and have grade > 80:")
print(filtered_df)

# Pandas Q3

# New column with curved grades
df["grade_curved"] = df["grade"] + 5

print("DataFrame with grade_curved:")
print(df)

# Pandas Q4

# Uppercase names column
df["name_upper"] = df["name"].str.upper()

print("Name and name_upper columns:")
print(df[["name", "name_upper"]])

# Pandas Q5

# Group by city and calculate mean grade
grouped = df.groupby("city")["grade"].mean()

print("Average grade by city:")
print(grouped)

# Pandas Q6

# Replace Austin with Houston in city column
df["city"] = df["city"].replace("Austin", "Houston")

print("Updated name and city columns:")
print(df[["name", "city"]])

# Pandas Q7

# Sort by grade descending and get top 3
top_students = df.sort_values("grade", ascending=False).head(3)

print("Top 3 students by grade:")
print(top_students)

# --- NumPy ---

# NumPy Q1

import numpy as np

# 1D array
arr = np.array([10, 20, 30, 40, 50])

print("Array:", arr)
print("Shape:", arr.shape)
print("Data type:", arr.dtype)
print("Number of dimensions:", arr.ndim)

# NumPy Q2

# 2D array
arr_2d = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("2D Array:")
print(arr_2d)

print("Shape:", arr_2d.shape)
print("Size:", arr_2d.size)

# NumPy Q3

# Slice top-left 2x2 block
slice_block = arr_2d[0:2, 0:2]

print("Top-left 2x2 block:")
print(slice_block)

# NumPy Q4

# 3x4 array of zeros
zeros_array = np.zeros((3, 4))

# 2x5 array of ones
ones_array = np.ones((2, 5))

print("3x4 zeros array:")
print(zeros_array)

print("2x5 ones array:")
print(ones_array)

# NumPy Q5

arr_range = np.arange(0, 50, 5)

print("Array:", arr_range)
print("Shape:", arr_range.shape)
print("Mean:", np.mean(arr_range))
print("Sum:", np.sum(arr_range))
print("Standard Deviation:", np.std(arr_range))

# NumPy Q6

random_arr = np.random.normal(0, 1, 200)

print("Random array:")
print(random_arr)

print("Mean:", np.mean(random_arr))
print("Standard Deviation:", np.std(random_arr))

import matplotlib.pyplot as plt

# --- Matplotlib ---

# Matplotlib Q1

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)

plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Matplotlib Q2

subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

plt.bar(subjects, scores)

plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")

plt.show()

# Matplotlib Q3

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color="blue", label="Dataset 1")
plt.scatter(x2, y2, color="red", label="Dataset 2")

plt.xlabel("x")
plt.ylabel("y")

plt.legend()

plt.show()

# Matplotlib Q4

fig, axes = plt.subplots(1, 2)

# LEFT plot (line)
axes[0].plot(x, y)
axes[0].set_title("Squares")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# RIGHT plot (bar)
axes[1].bar(subjects, scores)
axes[1].set_title("Subject Scores")
axes[1].set_xlabel("Subjects")
axes[1].set_ylabel("Scores")

plt.tight_layout()
plt.show()

# --- Descriptive Statistics ---

# Descriptive Stats Q1

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Variance:", np.var(data))
print("Standard Deviation:", np.std(data))


# Descriptive Stats Q2

# Generate random data
scores = np.random.normal(65, 10, 500)

# Plot histogram
plt.hist(scores, bins=20)

plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")

plt.show()

# Descriptive Stats Q3

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])

plt.title("Score Comparison")

plt.ylabel("Scores")

plt.show()

# Descriptive Stats Q4

# Generate data
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

# Boxplots
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])

plt.title("Distribution Comparison")
plt.ylabel("Values")

plt.show()


# Descriptive Stats Q5

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

# Convert to numpy arrays
arr1 = np.array(data1)
arr2 = np.array(data2)

# For mode 
from collections import Counter

def get_mode(data):
    return Counter(data).most_common(1)[0][0]

print("Data1:")
print("Mean:", np.mean(arr1))
print("Median:", np.median(arr1))
print("Mode:", get_mode(data1))

print("\nData2:")
print("Mean:", np.mean(arr2))
print("Median:", np.median(arr2))
print("Mode:", get_mode(data2))

# --- Hypothesis Testing ---

# Hypothesis Q1

from scipy import stats

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# t-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)

print("T-statistic:", t_stat)
print("P-value:", p_value)

# Hypothesis Q2

alpha = 0.05

if p_value < alpha:
    print("The result is statistically significant.")
else:
    print("The result is not statistically significant.")

# Hypothesis Q3

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

# paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

print("Paired T-statistic:", t_stat)
print("Paired P-value:", p_value)

# Hypothesis Q4

scores = [72, 68, 75, 70, 69, 74, 71, 73]

# t-test
t_stat, p_value = stats.ttest_1samp(scores, 70)

print("One-sample T-statistic:", t_stat)
print("One-sample P-value:", p_value)

# Hypothesis Q5

# group_a < group_b
t_stat, p_value = stats.ttest_ind(group_a, group_b, alternative="less")

print("One-tailed P-value:", p_value)

# Hypothesis Q6

print("Group B scores are higher than Group A, and this difference is unlikely due to chance.")

# --- Correlation ---

# Correlation Q1

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)

print("Correlation matrix:")
print(corr_matrix)

print("Correlation coefficient:", corr_matrix[0, 1])

# Correlation Q2

from scipy.stats import pearsonr

x = [1,2,3,4,5,6,7,8,9,10]
y = [10,9,7,8,6,5,3,4,2,1]

corr, p_value = pearsonr(x, y)

print("Correlation:", corr)
print("P-value:", p_value)

# Correlation Q3

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age": [25, 30, 22, 35, 28]
}

df_corr = pd.DataFrame(people)

print("Correlation matrix:")
print(df_corr.corr())

# Correlation Q4

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y)

plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Correlation Q5

import seaborn as sns

corr_matrix = df_corr.corr()

sns.heatmap(corr_matrix, annot=True)

plt.title("Correlation Heatmap")

plt.show()

# --- Pipeline ---

# Pipeline Q1

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    """Convert a NumPy array to a pandas Series named 'values'."""
    return pd.Series(arr, name="values")

def clean_data(series):
    """Remove missing values from the Series."""
    return series.dropna()

def summarize_data(series):
    """Return summary statistics for the cleaned Series."""
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

def data_pipeline(arr):
    """Run the full data pipeline from raw array to summary statistics."""
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

result = data_pipeline(arr)

print("Pipeline summary:")
for key, value in result.items():
    print(f"{key}: {value}")