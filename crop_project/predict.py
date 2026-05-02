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
# Prices are approximate Indian market prices (₹ per tonne)
# Source: APMC / Mandi rates (2024 approximate)
# ─────────────────────────────────────────────
CROP_PRICES = {
    "rice":        22000,    # ₹22,000/tonne
    "maize":       18000,    # ₹18,000/tonne
    "chickpea":    55000,    # ₹55,000/tonne
    "kidneybeans": 90000,    # ₹90,000/tonne
    "pigeonpeas":  65000,    # ₹65,000/tonne (Toor dal)
    "mothbeans":   60000,    # ₹60,000/tonne
    "mungbean":    75000,    # ₹75,000/tonne (Moong)
    "blackgram":   70000,    # ₹70,000/tonne (Urad)
    "lentil":      65000,    # ₹65,000/tonne (Masoor)
    "pomegranate": 80000,    # ₹80,000/tonne
    "banana":      15000,    # ₹15,000/tonne
    "mango":       40000,    # ₹40,000/tonne
    "grapes":      50000,    # ₹50,000/tonne
    "watermelon":  10000,    # ₹10,000/tonne
    "muskmelon":   15000,    # ₹15,000/tonne
    "apple":       80000,    # ₹80,000/tonne
    "orange":      25000,    # ₹25,000/tonne
    "papaya":      12000,    # ₹12,000/tonne
    "coconut":     25000,    # ₹25,000/tonne
    "cotton":      65000,    # ₹65,000/tonne
    "jute":        45000,    # ₹45,000/tonne
    "coffee":      180000,   # ₹1,80,000/tonne
}

# Typical yield per hectare (tonnes) for a good Indian season
CROP_YIELD = {
    "rice":        4.5,
    "maize":       5.5,
    "chickpea":    1.2,
    "kidneybeans": 1.5,
    "pigeonpeas":  1.0,
    "mothbeans":   0.8,
    "mungbean":    1.0,
    "blackgram":   0.9,
    "lentil":      1.2,
    "pomegranate": 12.0,
    "banana":      35.0,
    "mango":       10.0,
    "grapes":      15.0,
    "watermelon":  25.0,
    "muskmelon":   20.0,
    "apple":       20.0,
    "orange":      15.0,
    "papaya":      25.0,
    "coconut":     14.0,
    "cotton":      2.0,
    "jute":        2.5,
    "coffee":      1.5,
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


def predict_crop(N, P, K, temperature, humidity, ph, rainfall, model_dir="model"):
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

    # Build input as DataFrame so feature names match training
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    import pandas as pd
    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=feature_cols)

    # Get predicted class index and probabilities
    pred_encoded   = model.predict(features)[0]
    probabilities  = model.predict_proba(features)[0]

    # Convert class index back to crop name
    recommended_crop = label_encoder.inverse_transform([pred_encoded])[0]
    confidence       = probabilities[pred_encoded] * 100

    # Top-3 crops by probability
    top3_indices = np.argsort(probabilities)[::-1][:3]
    top_3_crops  = [
        (label_encoder.classes_[i], round(probabilities[i] * 100, 2))
        for i in top3_indices
    ]

    # Profitability estimate
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
        estimated_revenue = price_per_tonne x yield_per_hectare

    Returns a list of dicts sorted by revenue (highest first).
    """
    results = []
    for crop, confidence in top_3_crops:
        price  = CROP_PRICES.get(crop, 30000)   # default Rs.30,000 if unknown
        yield_ = CROP_YIELD.get(crop, 2.0)      # default 2t if unknown
        revenue = price * yield_                 # Rs. per hectare

        results.append({
            "crop":              crop,
            "confidence_pct":    confidence,
            "price_per_tonne":   price,
            "yield_per_hectare": yield_,
            "estimated_revenue": revenue,
        })

    # Sort by revenue descending
    results.sort(key=lambda x: -x["estimated_revenue"])
    return results


# ─────────────────────────────────────────────
# Quick test when running this file directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌱 Testing Prediction System ...")
    print("-" * 45)

    # Sample input: typical rice-growing conditions
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
              f"(₹{row['price_per_tonne']:,}/t × {row['yield_per_hectare']}t)")
