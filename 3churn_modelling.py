
# Customer Churn Prediction using Artificial Neural Network (ANN)
# Run directly in VS Code

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

FILE_PATH = "Churn_Modelling.csv"

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"'{FILE_PATH}' not found. "
        "Place the CSV file in the same folder as this script."
    )

df = pd.read_csv(FILE_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 3. Data Preprocessing
# --------------------------------------------------

# Remove rows with missing values
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove identifier and name columns.
# These columns are not used as predictive features.

X = df.drop(
    columns=[
        "RowNumber",
        "CustomerId",
        "Surname",
        "Exited"
    ],
    errors="ignore"
).copy()

y = df["Exited"].astype(int)

if y.nunique() != 2 or not set(y.unique()).issubset({0, 1}):
    raise ValueError("'Exited' must contain binary labels 0 and 1.")

# One-hot encode Geography.
# This avoids assigning an artificial order to countries.

X = pd.get_dummies(
    X,
    columns=["Geography"],
    drop_first=True,
    dtype=int
)

# Encode Gender as a binary variable
X["Gender"] = X["Gender"].map({
    "Female": 0,
    "Male": 1
})

if X.isnull().any().any():
    raise ValueError(
        "Missing or unexpected categorical values remain "
        "after preprocessing. Check the input CSV."
    )

print("\n========== PREPROCESSED DATA ==========")
print("Feature shape:", X.shape)
print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts().rename(
    index={0: "Stayed", 1: "Exited"}
))


# --------------------------------------------------
# 4. Split Data into Training, Validation and Test Sets
# --------------------------------------------------

# First reserve 20% for final testing.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

# Reserve 20% of the remaining training data for validation.
# Overall: 64% training, 16% validation, 20% testing.

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=SEED,
    stratify=y_train_full
)

print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 5. Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

# Fit only on training data to prevent data leakage.
X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)

print("\n========== NORMALIZATION ==========")

X_train_df = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

print("\nSample of normalized training data:")
print(X_train_df.head())

print("\nMean of scaled features:")
print(X_train_df.mean().round(4))

print("\nStandard deviation of scaled features:")
print(X_train_df.std().round(4))


# --------------------------------------------------
# 6. Build Artificial Neural Network
# --------------------------------------------------

model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),

    Dense(6, activation="relu"),

    Dropout(0.3),

    Dense(6, activation="relu"),

    Dense(1, activation="sigmoid")
])

print("\n========== ANN MODEL SUMMARY ==========")
model.summary()


# --------------------------------------------------
# 7. Compile Model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 8. Train the Model
# --------------------------------------------------

print("\n========== MODEL TRAINING ==========")

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    verbose=1
)


# --------------------------------------------------
# 9. Print Accuracy for Every Epoch
# --------------------------------------------------

train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

print("\n========== ACCURACY PER EPOCH ==========")

for i in range(len(train_acc)):
    print(
        f"Epoch {i + 1:02d}: "
        f"Training Accuracy = {train_acc[i] * 100:.2f}%, "
        f"Validation Accuracy = {val_acc[i] * 100:.2f}%"
    )


# --------------------------------------------------
# 10. Plot Training and Validation Accuracy
# --------------------------------------------------

epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    train_acc,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_acc,
    marker="o",
    label="Validation Accuracy"
)

plt.title("ANN Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(list(epochs))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. Plot Training and Validation Loss
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("ANN Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 12. Evaluate on Unseen Test Data
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

# Generate probability predictions, then convert to classes.
y_prob = model.predict(
    X_test_scaled,
    verbose=0
).ravel()

y_pred = (y_prob >= 0.5).astype(int)

final_accuracy = accuracy_score(y_test, y_pred)

print("\n========== FINAL TEST RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Classification Accuracy: {final_accuracy * 100:.2f}%")


# --------------------------------------------------
# 13. Classification Report
# --------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Stayed", "Exited"],
        zero_division=0
    )
)


# --------------------------------------------------
# 14. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

print("\nMatrix interpretation:")
print("True Negatives :", cm[0, 0])
print("False Positives:", cm[0, 1])
print("False Negatives:", cm[1, 0])
print("True Positives :", cm[1, 1])


# --------------------------------------------------
# 15. Visualize Confusion Matrix
# --------------------------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Customer Churn - Confusion Matrix")
plt.colorbar()

class_labels = ["Stayed", "Exited"]

plt.xticks([0, 1], class_labels)
plt.yticks([0, 1], class_labels)

threshold = cm.max() / 2 if cm.max() > 0 else 0

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 16. Project Completed
# --------------------------------------------------

print("\n========== PROJECT COMPLETED ==========")
