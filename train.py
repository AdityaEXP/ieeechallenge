import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed_train.csv")

X = df.drop(columns=["Label"])
y = df["Label"]

print(X.info())
TO_KEEP = [
    'GPS Spoofing',
    'Activities Declared',
    'Is App Taking Backup',
    'Screen Logging',
]
# TO_KEEP = ['Content Providers Declared',
#            'Metadata Elements',
#            'Duplicate Permissions Requested', 
#            'Target SDK Version', 
#            'Is App Taking Backup',
#            'Permissions Requested',
#            'Version Code',
#            ]
X = X[TO_KEEP]

print(X.info())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"train shape: {X_train.shape} & test shape: {X_test.shape}")

rf = RandomForestClassifier(
    n_estimators=1200,       
    max_depth=None,       
    min_samples_split=4,    
    class_weight='balanced',
    random_state=42,
    n_jobs=-1       
)

# rf = xgb = XGBClassifier(
#     n_estimators=500,
#     max_depth=16,          # deeper trees for complex patterns
#     learning_rate=0.05,   # smaller learning rate = smoother learning
#     subsample=0.8,
#     colsample_bytree=0.8,
#     reg_lambda=1,
#     random_state=42,
#     n_jobs=-1
# )
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)



print(f"acc: {acc:.4f}")
print(f"precision: {prec:.4f}")
print(f"recall: {rec:.4f}")
print(f"f1: {f1:.4f}")
print(f"rocauc: {roc:.4f}")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n=== Top 15 Important Features ===")
print(importances.head(15))

plt.figure(figsize=(10, 6))
importances.head(15).plot(kind='barh')
plt.title("Top 15 Feature Importances (Random Forest)")
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

rf = rf.fit(X, y)
import joblib
joblib.dump(rf, "models/random_forest_model.pkl")
