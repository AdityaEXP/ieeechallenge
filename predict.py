import pandas as pd
import numpy as np
import joblib
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

model = joblib.load("models/random_forest_model.pkl")
test_df = pd.read_csv("data/test.csv")
sha256_col = None
if 'sha256' in test_df.columns:
    sha256_col = test_df['sha256'].copy()

NULL_COLUMNS = [
    'Duplicate Permissions Requested', 'Permissions Requested', 'Services Declared',
    'Broadcast Receivers', 'Content Providers Declared', 'Metadata Elements',
    'Version Code', 'Target SDK Version'
]

float_cols = [
    'Duplicate Permissions Requested', 'Permissions Requested', 'Activities Declared',
    'Services Declared', 'Broadcast Receivers', 'Content Providers Declared',
    'Metadata Elements', 'Version Code', 'Target SDK Version'
]

test_df[NULL_COLUMNS] = test_df[NULL_COLUMNS].fillna(test_df[NULL_COLUMNS].median())

if 'sha256' in test_df.columns:
    test_df.drop(columns=['sha256'], inplace=True)

for col in float_cols:
    if abs(skew(test_df[col])) > 1:
        test_df[col] = np.log1p(test_df[col].clip(upper=test_df[col].quantile(0.99)))
    else:
        test_df[col] = test_df[col].clip(upper=test_df[col].quantile(0.99))

scaler = joblib.load("models/scaler.pkl")
test_df[float_cols] = scaler.transform(test_df[float_cols])

model_features = [
    'Duplicate Permissions Requested', 'Permissions Requested', 'Activities Declared',
    'Services Declared', 'Broadcast Receivers', 'Content Providers Declared',
    'Metadata Elements', 'Version Code', 'Target SDK Version', 'Is App Taking Backup'
]
TO_KEEP = [
    'GPS Spoofing',
    'Activities Declared',
    'Is App Taking Backup',
    'Screen Logging',
]
for col in model_features:
    if col not in test_df.columns:
        test_df[col] = 0

X_test = test_df[TO_KEEP]
# X_test = test_df[model_features]

pred_labels = model.predict(X_test)


output = pd.DataFrame({
    'sha256': sha256_col,
    'Label': pred_labels
})

output.to_csv("data/{sno}.csv".format(sno=input("Enter the serial number for the output file: ")), index=False)
print(output.head(10))
