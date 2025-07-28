import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from collections import Counter
import joblib
import os

df = pd.read_csv("data/creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("After SMOTE:", Counter(y_train_sm))

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_sm, y_train_sm)
y_pred = model.predict(X_test)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/logistic_model.pkl")

# Results
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
            xticklabels=["Clean", "Fraud"],
            yticklabels=["Clean", "Fraud"],
            cmap="Oranges")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()
