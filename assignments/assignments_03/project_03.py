import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

os.makedirs("outputs", exist_ok=True)

data_path = "../resources/spambase/spambase.data"

# --- Task 1: Load and Explore ---

column_names = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
    "char_freq_$", "char_freq_#", "capital_run_length_average",
    "capital_run_length_longest", "capital_run_length_total", "spam_label"
]

data_path = "../resources/spambase/spambase.data"

df = pd.read_csv(data_path, header=None, names=column_names)

print("Dataset shape:", df.shape)
print(df.head())
print(df.dtypes)

print("\nClass counts:")
print(df["spam_label"].value_counts())

print("\nClass proportions:")
print(df["spam_label"].value_counts(normalize=True))

# Comment:
# The dataset has one row per email. The target spam_label is 1 for spam and 0 for ham.
# Because the classes are not perfectly balanced, accuracy alone can be misleading.
# A model could look accurate while still missing many spam emails or incorrectly
# sending legitimate emails to spam.

features_to_plot = [
    "word_freq_free",
    "char_freq_!",
    "capital_run_length_total"
]

for feature in features_to_plot:
    plt.figure()
    df.boxplot(column=feature, by="spam_label")
    plt.title(feature + " by Spam Label")
    plt.suptitle("")
    plt.xlabel("Spam Label (0 = ham, 1 = spam)")
    plt.ylabel(feature)
    plt.savefig("outputs/" + feature.replace("!", "exclamation") + "_boxplot.png")

# Comment:
# The boxplots show that spam emails often have higher values for features like
# word_freq_free, char_freq_!, and capital_run_length_total.
# The differences are visible, but the distributions are also skewed with many
# values near zero.

print("\nFeature scale summary:")
print(df.drop(columns=["spam_label"]).describe().T[["mean", "std", "min", "max"]])

# Comment:
# Many word-frequency features are zero for most emails because most messages do
# not contain specific words like "free".
# The feature scales vary dramatically: word and character frequencies are small
# percentages, while capital_run_length_total can be very large.
# This matters for models such as KNN, PCA, and logistic regression because those
# methods are sensitive to feature magnitude. Tree-based models are less affected
# because they split on feature thresholds.

# --- Task 2: Prepare Your Data ---

X = df.drop(columns=["spam_label"])
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nTrain/test shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Comment:
# I use stratify=y so the train and test sets keep a similar spam/ham balance.
# I keep unscaled data for tree-based models because trees do not need scaling.
# I also create scaled data for KNN, PCA, and logistic regression because those
# methods are sensitive to feature magnitude.

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(
    np.arange(1, len(cumulative_variance) + 1),
    cumulative_variance
)
plt.axhline(0.90, linestyle="--")
plt.title("PCA Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.savefig("outputs/spambase_pca_variance.png")

n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print("\nPCA preprocessing:")
print("Components needed for 90% variance:", n_components_90)

X_train_pca = pca.transform(X_train_scaled)[:, :n_components_90]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components_90]

print("X_train_pca:", X_train_pca.shape)
print("X_test_pca:", X_test_pca.shape)

# Comment:
# PCA is fit only on the scaled training data to avoid leaking information from
# the test set. The data must be scaled first because PCA is based on variance,
# and unscaled large-valued features would dominate the components
#
# --- Task 3: Classifier Comparison ---

# KNN on unscaled data

knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)

y_pred_knn_unscaled = knn_unscaled.predict(X_test)

print("\nKNN Unscaled")
print("Accuracy:", accuracy_score(y_test, y_pred_knn_unscaled))
print("Classification report:")
print(classification_report(y_test, y_pred_knn_unscaled))


# KNN on scaled data

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

print("\nKNN Scaled")
print("Accuracy:", accuracy_score(y_test, y_pred_knn_scaled))
print("Classification report:")
print(classification_report(y_test, y_pred_knn_scaled))


# KNN on PCA-reduced data

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)

y_pred_knn_pca = knn_pca.predict(X_test_pca)

print("\nKNN PCA")
print("Accuracy:", accuracy_score(y_test, y_pred_knn_pca))
print("Classification report:")
print(classification_report(y_test, y_pred_knn_pca))

# Comment:
# Scaling helps KNN a lot on this dataset: accuracy improves from about 0.80
# unscaled to about 0.91 scaled. This makes sense because KNN uses distances,
# and Spambase features have very different numeric ranges.
# PCA performs almost the same as scaled KNN, but slightly lower here, so PCA
# does not improve KNN performance in this split.

# Decision Tree depth comparison

depth_values = [3, 5, 10, None]

print("\nDecision Tree Depth Comparison")

for depth in depth_values:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_accuracy = tree.score(X_train, y_train)
    test_accuracy = tree.score(X_test, y_test)

    print("max_depth:", depth)
    print("Train accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)

# Comment:
# As tree depth increases, training accuracy usually increases because the tree
# can fit more detailed rules. If test accuracy stops improving or drops while
# training accuracy keeps rising, that is evidence of overfitting.

# Chosen Decision Tree

chosen_tree_depth = 10

tree_model = DecisionTreeClassifier(max_depth=chosen_tree_depth, random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

print("\nChosen Decision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Classification report:")
print(classification_report(y_test, y_pred_tree))

# Comment:
# I would choose max_depth=10 for production. The unlimited tree has slightly
# higher test accuracy, but its training accuracy is almost perfect, which suggests
# overfitting. max_depth=10 gives nearly the same test performance with a simpler tree.

# Random Forest

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification report:")
print(classification_report(y_test, y_pred_rf))

# Logistic Regression on scaled data

log_scaled = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
log_scaled.fit(X_train_scaled, y_train)

y_pred_log_scaled = log_scaled.predict(X_test_scaled)

print("\nLogistic Regression Scaled")
print("Accuracy:", accuracy_score(y_test, y_pred_log_scaled))
print("Classification report:")
print(classification_report(y_test, y_pred_log_scaled))


# Logistic Regression on PCA-reduced data

log_pca = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
log_pca.fit(X_train_pca, y_train)

y_pred_log_pca = log_pca.predict(X_test_pca)

print("\nLogistic Regression PCA")
print("Accuracy:", accuracy_score(y_test, y_pred_log_pca))
print("Classification report:")
print(classification_report(y_test, y_pred_log_pca))

# Comment:
# Logistic regression performs better on the full scaled data than on the
# PCA-reduced data: about 0.93 vs about 0.92 accuracy.
# PCA did not improve performance here, likely because the original spam features
# contain useful individual signals that are slightly weakened by dimensionality reduction.

# Task 3 summary and best model confusion matrix

# Comment:
# The Random Forest performs best on the test set, with accuracy around 0.945.
# KNN improves greatly after scaling, which matches the expectation that distance-based
# models are sensitive to feature scale. PCA does not improve KNN or logistic regression
# here; the full scaled features work slightly better.
#
# For a spam filter, accuracy is useful but not enough. I would pay close attention
# to false positives, because sending a legitimate email to spam can cause someone
# to miss an important message. False negatives also matter because spam getting
# through is annoying or risky, but false positives are usually more costly.

best_model = rf_model
best_model_name = "Random Forest"
y_pred_best = y_pred_rf

cm_best = confusion_matrix(y_test, y_pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_best)
disp.plot()
plt.title("Best Model Confusion Matrix: " + best_model_name)
plt.savefig("outputs/best_model_confusion_matrix.png")

print("\nBest model confusion matrix:")
print(cm_best)

# Comment:
# The best model makes more false negatives than false positives.
# It lets 33 spam emails through as ham, while 18 legitimate ham emails are
# incorrectly marked as spam. Since I think false positives are more costly for
# users, this is a reasonable tradeoff, but both error types still matter.

# Feature importances

tree_importances = pd.Series(
    tree_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

rf_importances = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Decision Tree Feature Importances:")
print(tree_importances.head(10))

print("\nTop 10 Random Forest Feature Importances:")
print(rf_importances.head(10))

plt.figure(figsize=(10, 6))
rf_importances.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")

# Comment:
# The Decision Tree and Random Forest agree on several important spam signals,
# including char_freq_!, char_freq_$, word_freq_remove, word_freq_free, and
# capital-letter run features. This matches intuition because spam often uses
# urgent punctuation, money symbols, promotional words, and unusual capitalization.
# The Decision Tree puts most importance on a few features, while the Random Forest
# spreads importance across more features because it averages many trees.

# --- Task 4: Cross-Validation ---

models_for_cv = {
    "KNN unscaled": KNeighborsClassifier(n_neighbors=5),
    "KNN scaled": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "KNN PCA": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components_90)),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=chosen_tree_depth, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression scaled": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
    ]),
    "Logistic Regression PCA": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components_90)),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
    ])
}

print("\nCross-Validation Results")

for name, model_cv in models_for_cv.items():
    scores = cross_val_score(model_cv, X_train, y_train, cv=5)

    print(name)
    print("Mean accuracy:", scores.mean())
    print("Standard deviation:", scores.std())

# Comment:
# Cross-validation shows Random Forest is the most accurate model, with mean
# accuracy about 0.954. Logistic Regression PCA is the most stable by a tiny
# margin, with the lowest standard deviation, but Logistic Regression scaled is
# almost identical in stability and has a higher mean accuracy.
# The ranking mostly matches the single train/test split: Random Forest is still
# the strongest overall model.


# --- Task 5: Building Prediction Pipelines ---

tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
])

tree_pipeline.fit(X_train, y_train)
non_tree_pipeline.fit(X_train, y_train)

y_pred_tree_pipeline = tree_pipeline.predict(X_test)
y_pred_non_tree_pipeline = non_tree_pipeline.predict(X_test)

print("\nTree-Based Pipeline: Random Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_tree_pipeline))
print("Classification report:")
print(classification_report(y_test, y_pred_tree_pipeline))

print("\nNon-Tree Pipeline: Scaled Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_non_tree_pipeline))
print("Classification report:")
print(classification_report(y_test, y_pred_non_tree_pipeline))

# Comment:
# The two pipelines do not have the same structure. The Random Forest pipeline
# only needs the classifier because tree-based models do not require scaling.
# The Logistic Regression pipeline includes StandardScaler because logistic
# regression is sensitive to feature scale.
#
# Packaging preprocessing and modeling into a pipeline is useful because it keeps
# the steps together. This makes the model easier to reuse, hand off, or deploy,
# and it reduces the risk of forgetting to apply the same preprocessing at prediction time.


