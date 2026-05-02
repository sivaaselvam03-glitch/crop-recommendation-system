# setup.py
# This runs automatically to create the model if it doesn't exist
import setup
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_and_save():
    """Train model if not already saved."""
    
    if os.path.exists("model/crop_model.pkl"):
        return  # already trained, skip
    
    print("Training model for first time...")
    os.makedirs("model", exist_ok=True)
    
    df = pd.read_csv("E:\data science\pooject\agriculture crop prediction web app\crop_project\data\Crop_recommendation.csv")
    
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    with open("model/crop_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open("model/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    
    print("Model trained and saved!")

train_and_save()