import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor

# Load dataset
df = pd.read_csv("combined_file.csv")

# Drop unnecessary columns
drop_cols = ['Unnamed: 9', 'Valuation', 'MarketFee', 'ProgArrivals', 'AmcName', 'Arrivals']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Keep only valid rows
df = df[df['Model'] > 0]
df = df[df['Minimum'] > 0]
df = df[df['Maximum'] >= df['Minimum']]

# Add Year, Month, Day (simulate if Date not available)
if 'Date' not in df.columns:
    df['Date'] = pd.date_range(start='2022-01-01', periods=len(df), freq='D')

df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month
df['Day'] = pd.to_datetime(df['Date']).dt.day
df.drop(columns=['Date'], inplace=True, errors='ignore')

# Encode categorical features
label_encoders = {}
for col in ['YardName', 'CommName', 'VarityName']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & target
features = ['YardName', 'CommName', 'VarityName',
            'Minimum', 'Maximum', 'PriceRange', 'AvgPrice',
            'Year', 'Month', 'Day']
target = 'Model'

# Sample at least 200k if available
df_sampled = df.sample(n=min(200000, len(df)), random_state=42)
X = df_sampled[features]
y = df_sampled[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CatBoost
model = CatBoostRegressor(verbose=100)
model.fit(X_train, y_train)

# Save model + encoders
pickle.dump(model, open("CatBoost.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

print("✅ Training complete. Files saved: CatBoost.pkl, label_encoders.pkl")
