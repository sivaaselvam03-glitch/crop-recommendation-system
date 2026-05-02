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
# STEP 1: Load the dataset
# ─────────────────────────────────────────────
print("=" * 55)
print("  AI-Based Smart Agriculture Crop Recommendation")
print("  Model Training Script")
print("=" * 55)

data_path = os.path.join("data", "Crop_recommendation.csv")
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
# Features (X): the soil and weather measurements
# Target  (y): the crop label we want to predict
feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[feature_columns]
y = df['label']

print(f"\n✅ Features selected: {feature_columns}")
print(f"   Target column : 'label'")

# ─────────────────────────────────────────────
# STEP 5: Encode labels (text → numbers)
#   RandomForest can handle string labels directly,
#   but encoding lets us recover probabilities per class.
# ─────────────────────────────────────────────
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # e.g. 'rice' → 18
class_names = label_encoder.classes_        # array of crop names

# ─────────────────────────────────────────────
# STEP 6: Train-test split (80% train, 20% test)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,       # 20% held out for evaluation
    random_state=42,      # fixed seed → reproducible results
    stratify=y_encoded    # keep class proportions equal in both splits
)
print(f"\n📂 Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 7: Build and train the model
#
# Why Random Forest?
#   ✔ Handles both numerical and categorical data well
#   ✔ Robust to outliers and noisy data
#   ✔ Provides feature importance scores
#   ✔ Rarely overfits with enough trees
#   ✔ Works great on tabular agriculture data
# ─────────────────────────────────────────────
print("\n🌲 Training Random Forest Classifier ...")

model = RandomForestClassifier(
    n_estimators=100,      # 100 decision trees in the forest
    max_depth=None,        # trees grow until pure leaves
    random_state=42,       # reproducible
    n_jobs=-1              # use all CPU cores for speed
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
model_path   = os.path.join("model", "crop_model.pkl")
encoder_path = os.path.join("model", "label_encoder.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(encoder_path, "wb") as f:
    pickle.dump(label_encoder, f)

print(f"\n💾 Model saved   → {model_path}")
print(f"💾 Encoder saved → {encoder_path}")
print("\n🎉 Training complete! You can now run: streamlit run app.py")
print("=" * 55)
