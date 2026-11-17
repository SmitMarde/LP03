
# Email Spam Detection using SVM and KNN
# Run directly in VS Code

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

FILE_PATH = "email.csv"

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        "'emails.csv' not found. "
        "Place it in the same folder as this Python file."
    )

df = pd.read_csv(FILE_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum().sum())


# --------------------------------------------------
# 2. Data Preprocessing
# --------------------------------------------------

# Remove rows with missing target labels
df.dropna(subset=["Prediction"], inplace=True)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Separate features and target
# Email No. is an identifier, not a useful predictor.

X = df.drop(
    columns=["Email No.", "Prediction"],
    errors="ignore"
)

# Ensure all feature columns are numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace missing feature values with zero
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Target: 0 = Not Spam, 1 = Spam
Y = pd.to_numeric(df["Prediction"], errors="coerce")

valid_rows = Y.notna()
X = X.loc[valid_rows].copy()
Y = Y.loc[valid_rows].astype(int)

if not set(Y.unique()).issubset({0, 1}):
    raise ValueError("Prediction must contain only 0 and 1.")

if Y.nunique() != 2:
    raise ValueError("The dataset must contain both classes: 0 and 1.")

if len(X) < 2:
    raise ValueError("Not enough valid records to train the models.")

print("\n========== PREPROCESSED DATA ==========")
print("Number of emails:", len(X))
print("Number of word-frequency features:", X.shape[1])

print("\nClass distribution:")
print(Y.value_counts().sort_index().rename(
    index={0: "Not Spam", 1: "Spam"}
))


# --------------------------------------------------
# 3. Outlier Detection using IQR
# --------------------------------------------------

# Outliers are analyzed only in numeric input features.
# Word-frequency outliers are not automatically removed.

df_numeric = X.select_dtypes(include=[np.number])

Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outlier_mask = (
    (df_numeric < lower) | (df_numeric > upper)
)

outlier_counts = outlier_mask.sum().sort_values(
    ascending=False
)

topN = min(12, len(outlier_counts))

top_features = outlier_counts.head(topN).index.tolist()

print("\n========== OUTLIER ANALYSIS ==========")
print("Top features by outlier count:")
print(outlier_counts.head(topN))

if top_features:
    plt.figure(figsize=(16, 6))

    sns.boxplot(data=df_numeric[top_features])

    plt.title(f"Boxplots for Top {topN} Features by Outlier Count")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Word Frequency")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 4. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.25,
    random_state=42,
    stratify=Y
)

print("\n========== TRAIN-TEST SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. Helper Function for Evaluation
# --------------------------------------------------

results = {}


def evaluate_model(name, model, X_eval, y_true):
    predictions = model.predict(X_eval)

    accuracy = accuracy_score(y_true, predictions)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    print(f"\n========== {name} ==========")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Not Spam", "Spam"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(cm)

    results[name] = accuracy

    return predictions, cm


# --------------------------------------------------
# 6. Support Vector Machine (SVM)
# --------------------------------------------------

# Retains the original notebook's SVC configuration.
# SVM can also be evaluated with scaled features.

svc = SVC(
    C=1.0,
    kernel="rbf",
    gamma="auto"
)

print("\nTraining SVM...")
svc.fit(X_train, y_train)

svc_pred, svc_cm = evaluate_model(
    "SVM",
    svc,
    X_test,
    y_test
)


# --------------------------------------------------
# 7. K-Nearest Neighbors (KNN) - Original Features
# --------------------------------------------------

knn = KNeighborsClassifier(n_neighbors=7)

print("\nTraining KNN (k=7, unscaled features)...")
knn.fit(X_train, y_train)

knn_pred, knn_cm = evaluate_model(
    "KNN (k=7, unscaled)",
    knn,
    X_test,
    y_test
)


# --------------------------------------------------
# 8. Standardize Features
# --------------------------------------------------

# Fit the scaler only on training data to avoid data leakage.

scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("\nFeature scaling completed.")


# --------------------------------------------------
# 9. Scaled KNN for Different K Values
# --------------------------------------------------

ks = [1, 3, 5]

for k in ks:
    print(f"\nTraining scaled KNN (k={k})...")

    knn_scaled = KNeighborsClassifier(
        n_neighbors=k,
        n_jobs=-1
    )

    knn_scaled.fit(X_train_s, y_train)

    knn_scaled_pred, knn_scaled_cm = evaluate_model(
        f"KNN (k={k}, scaled)",
        knn_scaled,
        X_test_s,
        y_test
    )


# --------------------------------------------------
# 10. Compare Model Accuracy
# --------------------------------------------------

print("\n========== MODEL ACCURACY COMPARISON ==========")

results_df = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Accuracy"]
)

results_df["Accuracy (%)"] = results_df["Accuracy"] * 100

print(results_df.to_string(index=False, float_format="%.4f"))

best_model = max(results, key=results.get)

print(f"\nHighest accuracy: {best_model}")
print(f"Accuracy: {results[best_model] * 100:.2f}%")


# --------------------------------------------------
# 11. Accuracy Comparison Chart
# --------------------------------------------------

plt.figure(figsize=(11, 6))

bars = plt.bar(
    results_df["Model"],
    results_df["Accuracy (%)"]
)

plt.title("Email Spam Detection - Model Accuracy")
plt.xlabel("Classification Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=25, ha="right")

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 12. Confusion Matrix Visualization
# --------------------------------------------------

confusion_matrices = {
    "SVM": svc_cm,
    "KNN (k=7, unscaled)": knn_cm
}

# Add the scaled KNN models to the visualization.
for k in ks:
    model_name = f"KNN (k={k}, scaled)"

    model = KNeighborsClassifier(
        n_neighbors=k,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    predictions = model.predict(X_test_s)

    confusion_matrices[model_name] = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

for name, cm in confusion_matrices.items():
    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Spam", "Spam"],
        yticklabels=["Not Spam", "Spam"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 13. Project Completed
# --------------------------------------------------

print("\n========== PROJECT COMPLETED ==========")
