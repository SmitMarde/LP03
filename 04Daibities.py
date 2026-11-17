# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def run_knn_diabetes():
    # 1. Load the dataset
    # Ensure 'diabetes.csv' is in the same directory as this script/notebook
    try:
        df = pd.read_csv('diabetes.csv')
        print("Dataset loaded successfully!\n")
    except FileNotFoundError:
        print("Error: 'diabetes.csv' not found. Please ensure the file is uploaded or in the correct directory.")
        return

    # 2. Separate features (X) and target label (y)
    # The target column in this Kaggle dataset is usually named 'Outcome'
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 3. Split the data into Training and Testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Feature Scaling 
    # (KNN is distance-based, so scaling features to have mean=0 and variance=1 is crucial)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Initialize and Train the KNN Classifier
    # We use k=5 as a standard starting point
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # 6. Make Predictions on the test set
    y_pred = knn.predict(X_test_scaled)

    # 7. Compute the requested metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1.0 - accuracy
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # 8. Display the results
    print("-" * 30)
    print("K-NEAREST NEIGHBORS RESULTS")
    print("-" * 30)
    print("1. Confusion Matrix:")
    print(conf_matrix)
    print("\n   [True Negatives (TN) | False Positives (FP)]")
    print("   [False Negatives (FN)| True Positives (TP) ]")
    print(f"\n2. Accuracy   : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"3. Error Rate : {error_rate:.4f} ({error_rate * 100:.2f}%)")
    print(f"4. Precision  : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"5. Recall     : {recall:.4f} ({recall * 100:.2f}%)")
    print("-" * 30)

# Execute the function
if __name__ == "__main__":
    run_knn_diabetes()