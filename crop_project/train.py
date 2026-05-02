"""
train.py
========
This script loads the dataset, trains a Random Forest model,
evaluates it, and saves it to disk.

Run this FIRST before running app.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# BASE DIRECTORY (always relative to this file)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# STEP 1: Load the dataset
# ─────────────────────────────────────────────
print("=" * 55)
print("  AI-Based Smart Agriculture Crop Recommendation")
print("  Model Training Script")
print("=" * 55)

data_path = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")  # ✅ Fixed: absolute path

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"\n❌ Dataset not found at: {data_path}"
        f"\n   Please make sure 'Crop_recommendation.csv' is inside the 'data/' folder."
    )

df = pd.read_csv(data_path)

print(f"\n✅ Dataset loaded successfully!")
print(f"   Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

# ─────────────────────────────────────────────
# STEP 2: Explore the dataset
# ─────────────────────────────────────────────
print("\n📊 Feature Summary:")
print(df.describe().round(2))

print(f"\n🌾 Crops in dataset ({df['label'].nunique()} total):")
print(", ".join(sorted(df['label'].unique())))

# ─────────────────────────────────────────────
# STEP 3: Check for missing values
# ─────────────────────────────────────────────
missing = df.isnull().sum().sum()
print(f"\n🔍 Missing values: {missing}  ({'None – dataset is clean!' if missing == 0 else 'Needs handling!'})")

# ─────────────────────────────────────────────
# STEP 4: Feature selection
# ─────────────────────────────────────────────
feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[feature_columns]
y = df['label']

print(f"\n✅ Features selected: {feature_columns}")
print(f"   Target column : 'label'")

# ─────────────────────────────────────────────
# STEP 5: Encode labels (text → numbers)
# ─────────────────────────────────────────────
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

# ─────────────────────────────────────────────
# STEP 6: Train-test split (80% train, 20% test)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)
print(f"\n📂 Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 7: Build and train the model
# ─────────────────────────────────────────────
print("\n🌲 Training Random Forest Classifier ...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("   Training complete ✅")

# ─────────────────────────────────────────────
# STEP 8: Evaluate the model
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📈 Model Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ─────────────────────────────────────────────
# STEP 9: Feature importance
# ─────────────────────────────────────────────
importances = model.feature_importances_
feat_imp = sorted(zip(feature_columns, importances), key=lambda x: -x[1])
print("🔑 Feature Importance (most → least impactful):")
for feat, imp in feat_imp:
    bar = "█" * int(imp * 50)
    print(f"   {feat:<12} {bar}  {imp:.4f}")

# ─────────────────────────────────────────────
# STEP 10: Save model and encoder with pickle
# ─────────────────────────────────────────────
model_dir    = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)           # ✅ Fixed: creates 'model/' folder if missing

model_path   = os.path.join(model_dir, "crop_model.pkl")
encoder_path = os.path.join(model_dir, "label_encoder.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(encoder_path, "wb") as f:
    pickle.dump(label_encoder, f)

print(f"\n💾 Model saved   → {model_path}")
print(f"💾 Encoder saved → {encoder_path}")
print("\n🎉 Training complete! You can now run: streamlit run app.py")
print("=" * 55)