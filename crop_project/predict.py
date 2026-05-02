"""
predict.py
==========
Standalone prediction module.
Import this in app.py  OR  run directly to test predictions.
 
Usage:
    python predict.py
"""
 
import pickle
import os
import numpy as np
 
# ─────────────────────────────────────────────
# Crop Profitability Dictionary
# Prices are approximate Indian market prices (₹ per kg)
# Source: APMC / Mandi rates (2024 approximate)
# ─────────────────────────────────────────────
CROP_PRICES = {
    "rice":        22,      # ₹22/kg
    "maize":       18,      # ₹18/kg
    "chickpea":    55,      # ₹55/kg
    "kidneybeans": 90,      # ₹90/kg
    "pigeonpeas":  65,      # ₹65/kg
    "mothbeans":   60,      # ₹60/kg
    "mungbean":    75,      # ₹75/kg
    "blackgram":   70,      # ₹70/kg
    "lentil":      65,      # ₹65/kg
    "pomegranate": 80,      # ₹80/kg
    "banana":      15,      # ₹15/kg
    "mango":       40,      # ₹40/kg
    "grapes":      50,      # ₹50/kg
    "watermelon":  10,      # ₹10/kg
    "muskmelon":   15,      # ₹15/kg
    "apple":       80,      # ₹80/kg
    "orange":      25,      # ₹25/kg
    "papaya":      12,      # ₹12/kg
    "coconut":     25,      # ₹25/kg
    "cotton":      65,      # ₹65/kg
    "jute":        45,      # ₹45/kg
    "coffee":      180,     # ₹180/kg
}
 
# Typical yield per hectare (kg) for a good Indian season
CROP_YIELD = {
    "rice":        4500,
    "maize":       5500,
    "chickpea":    1200,
    "kidneybeans": 1500,
    "pigeonpeas":  1000,
    "mothbeans":   800,
    "mungbean":    1000,
    "blackgram":   900,
    "lentil":      1200,
    "pomegranate": 12000,
    "banana":      35000,
    "mango":       10000,
    "grapes":      15000,
    "watermelon":  25000,
    "muskmelon":   20000,
    "apple":       20000,
    "orange":      15000,
    "papaya":      25000,
    "coconut":     14000,
    "cotton":      2000,
    "jute":        2500,
    "coffee":      1500,
}
 
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
 
def load_model(model_dir=None):
    """Load the trained model and label encoder from disk."""
    if model_dir is None:
        model_dir = os.path.join(BASE_DIR, "model")
 
    model_path   = os.path.join(model_dir, "crop_model.pkl")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")
 
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found! Please run 'python train.py' first."
        )
 
    with open(model_path, "rb") as f:
        model = pickle.load(f)
 
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
 
    return model, label_encoder
 
 
def predict_crop(N, P, K, temperature, humidity, ph, rainfall, model_dir=None):
    """
    Predict the best crop for given soil & weather conditions.
 
    Parameters
    ----------
    N           : Nitrogen content in soil (mg/kg)
    P           : Phosphorus content in soil (mg/kg)
    K           : Potassium content in soil (mg/kg)
    temperature : Temperature in °C
    humidity    : Relative humidity in %
    ph          : Soil pH (0-14 scale)
    rainfall    : Annual rainfall in mm
 
    Returns
    -------
    dict with:
        recommended_crop  - best crop name
        confidence        - model confidence score (0-100%)
        top_3_crops       - list of (crop, confidence%) tuples
        profitability     - dict with profit details
    """
    model, label_encoder = load_model(model_dir)
 
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    import pandas as pd
    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=feature_cols)
 
    pred_encoded   = model.predict(features)[0]
    probabilities  = model.predict_proba(features)[0]
 
    recommended_crop = label_encoder.inverse_transform([pred_encoded])[0]
    confidence       = probabilities[pred_encoded] * 100
 
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top_3_crops  = [
        (label_encoder.classes_[i], round(probabilities[i] * 100, 2))
        for i in top3_indices
    ]
 
    profitability = estimate_profitability(top_3_crops)
 
    return {
        "recommended_crop": recommended_crop,
        "confidence":        round(confidence, 2),
        "top_3_crops":       top_3_crops,
        "profitability":     profitability,
    }
 
 
def estimate_profitability(top_3_crops):
    """
    For each of the top-3 predicted crops, calculate:
        estimated_revenue = price_per_kg x yield_per_hectare (kg)
 
    Returns a list of dicts sorted by confidence (highest first).
    """
    results = []
    for crop, confidence in top_3_crops:
        price  = CROP_PRICES.get(crop, 30)     # default ₹30/kg if unknown
        yield_ = CROP_YIELD.get(crop, 2000)    # default 2000 kg/ha if unknown
        revenue = price * yield_               # ₹ per hectare
 
        results.append({
            "crop":              crop,
            "confidence_pct":    confidence,
            "price_per_kg":      price,
            "yield_per_hectare": yield_,       # in kg
            "estimated_revenue": revenue,
        })
 
    # Sort by confidence so recommended crop stays first
    results.sort(key=lambda x: -x["confidence_pct"])
    return results
 
 
# ─────────────────────────────────────────────
# Quick test when running this file directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌱 Testing Prediction System ...")
    print("-" * 45)
 
    test_input = dict(N=90, P=42, K=43,
                      temperature=21, humidity=82,
                      ph=6.5, rainfall=203)
 
    print("Input conditions:")
    for k, v in test_input.items():
        print(f"   {k:<12}: {v}")
 
    result = predict_crop(**test_input)
 
    print(f"\n✅ Recommended Crop : {result['recommended_crop'].upper()}")
    print(f"   Confidence       : {result['confidence']}%")
 
    print("\n🏆 Top 3 Predictions:")
    for i, (crop, conf) in enumerate(result["top_3_crops"], 1):
        print(f"   {i}. {crop:<15}  {conf}% confidence")
 
    print("\n💰 Profitability Estimate (per hectare):")
    for row in result["profitability"]:
        print(f"   {row['crop']:<15}  ₹{row['estimated_revenue']:,.0f}  "
              f"(₹{row['price_per_kg']}/kg × {row['yield_per_hectare']}kg)")