import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix
)
import joblib
import os

# Load data
df = pd.read_csv("data/creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
yPred = rfc.predict(xTest)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(rfc, "model/random_forest_model.pkl")

# Evaluation
print("Random Forest Evaluation:")
print(f"Accuracy: {accuracy_score(yTest, yPred):.4f}")
print(f"Precision: {precision_score(yTest, yPred):.4f}")
print(f"Recall: {recall_score(yTest, yPred):.4f}")
print(f"F1: {f1_score(yTest, yPred):.4f}")
print(f"MCC: {matthews_corrcoef(yTest, yPred):.4f}")

# Confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(yTest, yPred), annot=True, fmt="d",
            xticklabels=["Clean", "Fraud"],
            yticklabels=["Clean", "Fraud"],
            cmap="Blues")
plt.title("Random Forest - Confusion Matrix")
plt.show()
