from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd

def balance_with_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print("Resampled dataset shape:", Counter(y_res))
    return X_res, y_res
