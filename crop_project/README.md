# 🌾 AI-Based Smart Agriculture Crop Recommendation System

A machine learning system that recommends the best crop to grow
based on soil nutrients and weather conditions.

---

## 📁 Project Structure

```
crop_project/
│
├── data/
│   └── Crop_recommendation.csv    ← Dataset (2200 rows, 22 crops)
│
├── model/
│   ├── crop_model.pkl             ← Trained Random Forest model (auto-created)
│   └── label_encoder.pkl          ← Label encoder (auto-created)
│
├── train.py                       ← Train the model (run this first!)
├── predict.py                     ← Prediction + profitability logic
├── app.py                         ← Streamlit web app
├── requirements.txt               ← Python dependencies
└── README.md                      ← This file
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
This will:
- Load and explore the dataset
- Train a Random Forest model
- Print accuracy and feature importance
- Save the model to `model/`

### 3. (Optional) Test prediction logic
```bash
python predict.py
```

### 4. Launch the web app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

---

## 📊 Dataset Features

| Feature | Description | Unit |
|---------|-------------|------|
| N | Nitrogen content in soil | mg/kg |
| P | Phosphorus content in soil | mg/kg |
| K | Potassium content in soil | mg/kg |
| temperature | Average temperature | °C |
| humidity | Relative humidity | % |
| ph | Soil pH value | 0–14 |
| rainfall | Annual rainfall | mm |
| **label** | **Crop to grow (target)** | — |

---

## 🌾 Crops Supported (22 total)

rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean,
blackgram, lentil, pomegranate, banana, mango, grapes, watermelon,
muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

---

## 🤖 Model Details

- **Algorithm**: Random Forest Classifier
- **Trees**: 100
- **Train/Test split**: 80% / 20%
- **Expected accuracy**: ~99%

---

## 🎤 Interview Explanation

### Problem Statement
Farmers often don't know which crop is best suited for their land.
The wrong crop choice leads to poor yield and financial loss.

### Approach
We use 7 measurable soil and weather features to train a machine
learning model that can classify the best crop among 22 options.

### Model Used
**Random Forest** — an ensemble of 100 decision trees. Each tree
votes, and the majority wins. It handles tabular agricultural data
very well and rarely overfits.

### Output
- Recommended crop with confidence score
- Top-3 crop alternatives
- Revenue estimate per hectare for profitability comparison

### Future Improvements
- Add real-time weather API integration
- Add soil test image recognition with CNNs
- Support regional price databases
- Add irrigation and fertilizer recommendations
- Build a mobile app version
