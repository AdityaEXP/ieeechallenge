import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/main.csv")
print(df.info())
NULL_COLUMNS = [
    'Duplicate Permissions Requested', 'Permissions Requested', 'Services Declared',
    'Broadcast Receivers', 'Content Providers Declared', 'Metadata Elements',
    'Version Code', 'Target SDK Version'
]

df[NULL_COLUMNS] = df[NULL_COLUMNS].fillna(df[NULL_COLUMNS].median())
df.drop(columns=['sha256'], inplace=True)

float_cols = [
    'Duplicate Permissions Requested', 'Permissions Requested', 'Activities Declared',
    'Services Declared', 'Broadcast Receivers', 'Content Providers Declared',
    'Metadata Elements', 'Version Code', 'Target SDK Version'
]

for col in float_cols:
    if abs(skew(df[col])) > 1:
        df[col] = np.log1p(df[col].clip(upper=df[col].quantile(0.99)))
    else:
        df[col] = df[col].clip(upper=df[col].quantile(0.99))

scaler = StandardScaler()
df[float_cols] = scaler.fit_transform(df[float_cols])
import joblib
joblib.dump(scaler, "models/scaler.pkl")

binary_cols = [c for c in df.select_dtypes('int').columns if c != 'Label']
low_var_cols = [col for col in binary_cols if df[col].nunique() == 1]
df.drop(columns=low_var_cols, inplace=True)

corr_matrix = df[binary_cols].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
high_corr_features.remove('Contact Information Theft')
to_not_remove = [
    'GPS Spoofing',
    'Activities Declared',
    'Is App Taking Backup',
    'Screen Logging',
    # 'Permissions Requested',
    # 'Services Declared',
    # 'Broadcast Receivers'
]
# high_corr_features.remove('GPS spooking')
# high_corr_features.remove('GPS spooking')
df.drop(columns=high_corr_features, inplace=True)
for tnr in to_not_remove:
    if tnr in high_corr_features:
        high_corr_features.remove(tnr)

corr_target = df.corrwith(df['Label']).sort_values(ascending=False)

print(corr_target.head(5))
print(corr_target.tail(5))

import matplotlib.pyplot as plt
corr_target.drop('Label').sort_values().plot(kind='barh', figsize=(12, 8))
plt.title("Feature Correlation with Target Label")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ==============================
# # 💾 Save Final Clean Data
# # ==============================
df.to_csv("data/processed_train.csv", index=False)