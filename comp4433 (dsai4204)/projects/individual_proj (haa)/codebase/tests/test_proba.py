import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

# Dummy data
X = np.random.rand(20, 5)
y = np.random.randint(0, 2, 20)
# Ensure both classes are present
y[0] = 0
y[1] = 1

models = {
    "logreg": LogisticRegression(random_state=42, max_iter=10000),
    "xgboost": XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss'),
    "rf": RandomForestClassifier(criterion='entropy', random_state=42, n_estimators=100),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "tabpfn": TabPFNClassifier(device='cpu', n_estimators=2, random_state=42) # reduced n_estimators for speed, cpu for safety
}

print(f"{'Model':<10} | {'Has predict_proba':<18} | {'Output Shape':<15} | {'First Sample Output'}")
print("-" * 80)

for name, model in models.items():
    try:
        model.fit(X, y)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            has_proba = "Yes"
            shape = str(proba.shape)
            first_sample = str(proba[0])
        else:
            has_proba = "No"
            shape = "N/A"
            first_sample = "N/A"
    except Exception as e:
        has_proba = f"Error: {e}"
        shape = "N/A"
        first_sample = "N/A"
    
    print(f"{name:<10} | {has_proba:<18} | {shape:<15} | {first_sample}")