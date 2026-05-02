"""
app.py
======
Streamlit — AI-Based Smart Agriculture Crop Recommendation System.
UI v3: Dark futuristic theme, 3D card effects, neon-green accents,
        high-contrast fonts, animated hero, glowing buttons.
Core ML logic is UNCHANGED.
 
Run with:  streamlit run app.py
"""
 
import streamlit as st
import pandas as pd
import os
from predict import predict_crop, CROP_PRICES, CROP_YIELD
 
# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 AgriSense AI",
    page_icon="🌱",
    layout="wide",
)
 
# ── MASTER CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════
   FONTS
═══════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;500;600;700&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    font-size: 16px;
}
 
/* ═══════════════════════════════════════════════
   BACKGROUND — deep space + green nebula
═══════════════════════════════════════════════ */
.stApp {
    background:
        radial-gradient(ellipse at 20% 20%, rgba(0,255,120,0.08) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 80%, rgba(0,200,80,0.06) 0%, transparent 55%),
        linear-gradient(160deg, #020c06 0%, #050f08 40%, #040d06 100%);
    min-height: 100vh;
}
 
/* ═══════════════════════════════════════════════
   HIDE STREAMLIT CHROME
═══════════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
 
/* ═══════════════════════════════════════════════
   HERO BANNER
═══════════════════════════════════════════════ */
.hero {
    position: relative;
    background: linear-gradient(120deg, #001a0a 0%, #003318 50%, #001a0a 100%);
    border: 1px solid rgba(0,255,100,0.25);
    border-radius: 24px;
    padding: 3rem 3.5rem 2.5rem;
    margin-bottom: 2.2rem;
    overflow: hidden;
    box-shadow:
        0 2px 0   #00ff6644,
        0 8px 0   #00cc5022,
        0 20px 60px rgba(0,0,0,0.7),
        inset 0 1px 0 rgba(0,255,100,0.15);
    transform: perspective(900px) rotateX(1deg);
}
.hero::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, transparent 0%, rgba(0,255,80,0.04) 50%, transparent 100%);
    animation: scanline 4s linear infinite;
    pointer-events: none;
}
@keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100%); }
}
.hero::before {
    content: '🌾';
    font-size: 11rem;
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    opacity: 0.07;
    pointer-events: none;
    filter: grayscale(1);
}
.hero-tag {
    display: inline-block;
    background: rgba(0,255,100,0.12);
    color: #00ff64;
    border: 1px solid rgba(0,255,100,0.4);
    border-radius: 30px;
    padding: 0.25rem 1rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    color: #ffffff;
    margin: 0 0 0.4rem;
    line-height: 1.1;
    text-shadow: 0 0 40px rgba(0,255,100,0.4);
    letter-spacing: -1px;
}
.hero h1 span { color: #00ff64; }
.hero p {
    color: #a0ffb8;
    font-size: 1.1rem;
    font-weight: 400;
    max-width: 580px;
    line-height: 1.65;
    margin: 0;
}
 
/* ═══════════════════════════════════════════════
   3D SECTION CARDS
═══════════════════════════════════════════════ */
.card3d {
    background: linear-gradient(145deg, #0a1f0f, #071509);
    border: 1px solid rgba(0,255,100,0.18);
    border-radius: 20px;
    padding: 1.7rem 2rem 1.4rem;
    margin-bottom: 1.4rem;
    position: relative;
    box-shadow:
        0 1px 0   rgba(0,255,100,0.3),
        0 4px 0   rgba(0,200,70,0.15),
        0 8px 0   rgba(0,150,50,0.08),
        0 25px 50px rgba(0,0,0,0.6),
        inset 0 1px 0 rgba(255,255,255,0.04);
    transform: perspective(800px) rotateX(0.5deg);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card3d:hover {
    transform: perspective(800px) rotateX(0deg) translateY(-4px);
    box-shadow:
        0 1px 0   rgba(0,255,100,0.5),
        0 6px 0   rgba(0,200,70,0.2),
        0 12px 0  rgba(0,150,50,0.1),
        0 35px 70px rgba(0,0,0,0.65),
        inset 0 1px 0 rgba(255,255,255,0.06);
}
.card-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: #00ff64;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid rgba(0,255,100,0.2);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
 
/* ═══════════════════════════════════════════════
   INPUT LABELS — high contrast white
═══════════════════════════════════════════════ */
label, .stSlider label, .stNumberInput label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] {
    color: #e0ffe8 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}
 
/* ═══════════════════════════════════════════════
   NUMBER INPUTS
═══════════════════════════════════════════════ */
input[type="number"] {
    background: #0d1f10 !important;
    border: 1.5px solid rgba(0,255,100,0.3) !important;
    border-radius: 12px !important;
    color: #00ff64 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
}
input[type="number"]:focus {
    border-color: #00ff64 !important;
    box-shadow: 0 0 0 3px rgba(0,255,100,0.15), 0 0 20px rgba(0,255,100,0.1) !important;
}
 
/* ═══════════════════════════════════════════════
   SLIDERS
═══════════════════════════════════════════════ */
div[data-baseweb="slider"] > div {
    background: rgba(0,255,100,0.2) !important;
}
div[data-baseweb="slider"] [role="slider"] {
    background: #00ff64 !important;
    border-color: #00ff64 !important;
    box-shadow: 0 0 12px rgba(0,255,100,0.6) !important;
    width: 20px !important;
    height: 20px !important;
}
 
/* ═══════════════════════════════════════════════
   GLOW BUTTON
═══════════════════════════════════════════════ */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00401a, #00802e, #00401a) !important;
    color: #ffffff !important;
    border: 1.5px solid #00ff64 !important;
    border-radius: 14px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 0.85rem 2.5rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    box-shadow:
        0 0 20px rgba(0,255,100,0.3),
        0 4px 0 #003010,
        0 8px 20px rgba(0,0,0,0.5) !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow:
        0 0 35px rgba(0,255,100,0.55),
        0 4px 0 #004020,
        0 10px 30px rgba(0,0,0,0.5) !important;
    transform: translateY(-3px) !important;
    color: #00ff64 !important;
}
 
/* ═══════════════════════════════════════════════
   RESULT HERO
═══════════════════════════════════════════════ */
.result-hero {
    background: linear-gradient(135deg, #001a0a, #003318, #001a0a);
    border: 1px solid rgba(0,255,100,0.35);
    border-radius: 24px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 2px 0   rgba(0,255,100,0.5),
        0 6px 0   rgba(0,180,60,0.2),
        0 12px 0  rgba(0,120,40,0.1),
        0 30px 80px rgba(0,0,0,0.7),
        inset 0 1px 0 rgba(0,255,100,0.15);
    transform: perspective(900px) rotateX(1deg);
}
.result-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: 50%;
    transform: translateX(-50%);
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,255,100,0.1) 0%, transparent 70%);
    pointer-events: none;
}
.result-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.72rem;
    color: #00cc50;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.crop-name-3d {
    font-family: 'Orbitron', monospace;
    font-size: 3.2rem;
    font-weight: 900;
    color: #ffffff;
    text-shadow:
        0 0 20px rgba(0,255,100,0.6),
        0 0 60px rgba(0,255,100,0.3),
        0 4px 8px rgba(0,0,0,0.5);
    letter-spacing: -1px;
    margin: 0.3rem 0 0.8rem;
    line-height: 1;
}
.conf-pill {
    display: inline-block;
    background: rgba(0,255,100,0.12);
    border: 1.5px solid rgba(0,255,100,0.5);
    border-radius: 30px;
    padding: 0.4rem 1.4rem;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #00ff64;
    letter-spacing: 1px;
    box-shadow: 0 0 16px rgba(0,255,100,0.2);
}
 
/* ═══════════════════════════════════════════════
   RANK CARDS
═══════════════════════════════════════════════ */
.rank3d {
    background: linear-gradient(135deg, #0c1e0f, #071409);
    border: 1px solid rgba(0,255,100,0.2);
    border-radius: 14px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow:
        0 2px 0 rgba(0,200,70,0.25),
        0 5px 0 rgba(0,150,50,0.1),
        0 12px 30px rgba(0,0,0,0.5);
    transition: transform 0.2s, box-shadow 0.2s;
}
.rank3d:hover {
    transform: translateY(-3px);
    box-shadow:
        0 4px 0 rgba(0,200,70,0.35),
        0 8px 0 rgba(0,150,50,0.15),
        0 20px 40px rgba(0,0,0,0.55);
}
.rank-medal { font-size: 1.8rem; min-width: 2.2rem; }
.rank-name  { font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 1.1rem; color: #e0ffe8; flex: 1; }
.rank-bar-bg { flex: 2; height: 10px; background: rgba(0,255,100,0.1); border-radius: 99px; overflow: hidden; border: 1px solid rgba(0,255,100,0.15); }
.rank-bar-fill { height: 100%; background: linear-gradient(90deg, #00802e, #00ff64); border-radius: 99px; box-shadow: 0 0 10px rgba(0,255,100,0.5); }
.rank-pct { font-family: 'Orbitron', monospace; font-size: 0.9rem; font-weight: 700; color: #00ff64; min-width: 3.5rem; text-align: right; }
 
/* ═══════════════════════════════════════════════
   PROFIT BOX
═══════════════════════════════════════════════ */
.profit3d {
    background: linear-gradient(135deg, #1a1000, #2a1a00, #1a1000);
    border: 1.5px solid rgba(255,200,0,0.35);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
    box-shadow:
        0 2px 0   rgba(255,200,0,0.4),
        0 6px 0   rgba(200,150,0,0.15),
        0 20px 50px rgba(0,0,0,0.6),
        inset 0 1px 0 rgba(255,220,0,0.1);
    transform: perspective(800px) rotateX(0.5deg);
}
.profit3d .p-label  { font-family: 'Orbitron', monospace; font-size: 0.7rem; color: #ffcc00; letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 0.5rem; }
.profit3d .p-crop   { font-family: 'Rajdhani', sans-serif; font-size: 1.3rem; font-weight: 700; color: #fff8e0; margin-bottom: 0.3rem; }
.profit3d .p-amount { font-family: 'Orbitron', monospace; font-size: 2.4rem; font-weight: 900; color: #ffcc00; text-shadow: 0 0 30px rgba(255,200,0,0.5); letter-spacing: -1px; }
.profit3d .p-detail { font-size: 0.88rem; color: #ccaa55; margin-top: 0.4rem; font-weight: 500; }
 
/* ═══════════════════════════════════════════════
   METRIC SUMMARY CARDS
═══════════════════════════════════════════════ */
.met3d {
    background: linear-gradient(135deg, #0c1e0f, #071409);
    border: 1px solid rgba(0,255,100,0.15);
    border-radius: 12px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 3px 0 rgba(0,150,50,0.12), 0 8px 20px rgba(0,0,0,0.4);
    transition: transform 0.15s;
}
.met3d:hover { transform: translateX(4px); }
.met-label { font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; font-weight: 600; color: #66bb88; letter-spacing: 0.5px; }
.met-value { font-family: 'Orbitron', monospace; font-size: 1rem; font-weight: 700; color: #00ff64; text-shadow: 0 0 10px rgba(0,255,100,0.35); }
.met-unit  { font-size: 0.72rem; color: #449966; margin-left: 0.25rem; }
 
/* ═══════════════════════════════════════════════
   pH BADGE
═══════════════════════════════════════════════ */
.ph-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    margin-top: 0.4rem;
    letter-spacing: 0.3px;
}
 
/* ═══════════════════════════════════════════════
   ERROR BANNER
═══════════════════════════════════════════════ */
.err3d {
    background: #1a0505;
    border: 1.5px solid rgba(255,80,80,0.5);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    color: #ff8888;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 0 20px rgba(255,0,0,0.1);
}
 
/* ═══════════════════════════════════════════════
   FOOTER
═══════════════════════════════════════════════ */
.footer3d {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    color: #336644;
    font-size: 0.8rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    border-top: 1px solid rgba(0,255,100,0.08);
    margin-top: 2rem;
}
 
/* ═══════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020c06, #030e07) !important;
    border-right: 1px solid rgba(0,255,100,0.12) !important;
}
[data-testid="stSidebar"] * { color: #a0ffb8 !important; font-family: 'Rajdhani', sans-serif !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #00ff64 !important; font-family: 'Orbitron', monospace !important; font-size: 0.85rem !important; letter-spacing: 1.5px !important; }
[data-testid="stSidebar"] hr { border-color: rgba(0,255,100,0.15) !important; }
[data-testid="stSidebar"] table td,
[data-testid="stSidebar"] table th { color: #80cc99 !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] table th { color: #00ff64 !important; }
 
/* ═══════════════════════════════════════════════
   DATAFRAME & CAPTION
═══════════════════════════════════════════════ */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; border: 1px solid rgba(0,255,100,0.2) !important; }
.stCaption, small { color: #66aa77 !important; font-size: 0.88rem !important; }
[data-testid="stSpinner"] p { color: #00ff64 !important; }
 
/* ═══════════════════════════════════════════════
   RESPONSIVE
═══════════════════════════════════════════════ */
@media (max-width: 768px) {
    .hero h1 { font-size: 2rem; }
    .crop-name-3d { font-size: 2rem; }
    .profit3d .p-amount { font-size: 1.8rem; }
}
</style>
""", unsafe_allow_html=True)
 
 
# ── HERO BANNER ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">🤖 AI-Powered &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; ~99% Accuracy</div>
    <h1>AgriSense <span>AI</span></h1>
    <p>Enter your soil nutrients, weather data, and pH — the AI engine will
    recommend the optimal crop and project your profitability per hectare.</p>
</div>
""", unsafe_allow_html=True)
 
 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")

# FIXED
if not os.path.exists(MODEL_PATH):
    import subprocess, sys
    st.info("⚙️ Model not found. Training now — please wait ~30 seconds...")
    result = subprocess.run([sys.executable, os.path.join(BASE_DIR, "train.py")], 
                            capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"Training failed:\n{result.stderr}")
        st.stop()
    st.success("✅ Model trained successfully! Reloading...")
    st.rerun()
 
# ── MAIN LAYOUT ────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([2, 1], gap="large")
 
with left_col:
 
    # SECTION 1 — Soil Nutrients
    st.markdown('<div class="card3d"><div class="card-title">⚗️ &nbsp;Soil Nutrients</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        N = st.number_input("Nitrogen (N) mg/kg", min_value=0, max_value=200, value=90,
                            help="Supports leafy growth.")
    with c2:
        P = st.number_input("Phosphorus (P) mg/kg", min_value=0, max_value=200, value=42,
                            help="Roots & flower development.")
    with c3:
        K = st.number_input("Potassium (K) mg/kg", min_value=0, max_value=250, value=43,
                            help="Overall plant health & disease resistance.")
    st.markdown('</div>', unsafe_allow_html=True)
 
    # SECTION 2 — Weather
    st.markdown('<div class="card3d"><div class="card-title">🌤️ &nbsp;Weather Conditions</div>', unsafe_allow_html=True)
    w1, w2 = st.columns(2)
    with w1:
        temperature = st.slider("Temperature (°C)", 5.0, 50.0, 21.0, 0.5)
    with w2:
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 82.0, 0.5)
    rainfall = st.slider("Annual Rainfall (mm)", 20.0, 300.0, 203.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)
 
    # SECTION 3 — pH
    st.markdown('<div class="card3d"><div class="card-title">🧫 &nbsp;Soil pH</div>', unsafe_allow_html=True)
    ph = st.slider("pH Value", 3.0, 10.0, 6.5, 0.1)
    if ph < 5:
        ph_color, ph_bg, ph_text = "#ff6b6b", "rgba(255,80,80,0.12)", "Strongly Acidic"
    elif ph < 6:
        ph_color, ph_bg, ph_text = "#ffaa44", "rgba(255,150,50,0.12)", "Mildly Acidic"
    elif ph < 7:
        ph_color, ph_bg, ph_text = "#00ff64", "rgba(0,255,100,0.12)", "Neutral"
    elif ph < 8:
        ph_color, ph_bg, ph_text = "#44aaff", "rgba(50,150,255,0.12)", "Mildly Alkaline"
    else:
        ph_color, ph_bg, ph_text = "#cc88ff", "rgba(180,100,255,0.12)", "Strongly Alkaline"
    st.markdown(f"""
    <div class="ph-badge" style="background:{ph_bg};color:{ph_color};border:1.5px solid {ph_color}55;">
        pH {ph:.1f} — {ph_text}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
 
# RIGHT PANEL — live summary
with right_col:
    st.markdown('<div class="card3d"><div class="card-title">📡 &nbsp;Live Parameters</div>', unsafe_allow_html=True)
    params = [
        ("🌿 Nitrogen",     f"{N}",           "mg/kg"),
        ("🌸 Phosphorus",   f"{P}",           "mg/kg"),
        ("💎 Potassium",    f"{K}",           "mg/kg"),
        ("🌡️ Temperature", f"{temperature}", "°C"),
        ("💧 Humidity",     f"{humidity}",    "%"),
        ("⚗️ pH",          f"{ph}",          ""),
        ("🌧️ Rainfall",    f"{rainfall}",    "mm"),
    ]
    for label, val, unit in params:
        st.markdown(f"""
        <div class="met3d">
            <span class="met-label">{label}</span>
            <span>
                <span class="met-value">{val}</span>
                <span class="met-unit">{unit}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
 
# ── PREDICT BUTTON ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_mid, _ = st.columns([1, 2, 1])
with btn_mid:
    predict_btn = st.button("⚡ ANALYSE & RECOMMEND CROP", use_container_width=True, type="primary")
 
 
# ── RESULTS ────────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("⚙️  Running AI inference — analysing 7 parameters…"):
        try:
            result = predict_crop(N=N, P=P, K=K,
                                  temperature=temperature,
                                  humidity=humidity,
                                  ph=ph,
                                  rainfall=rainfall)
        except FileNotFoundError as e:
            st.markdown(f'<div class="err3d">❌ {e}</div>', unsafe_allow_html=True)
            st.stop()
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Big result hero
    st.markdown(f"""
    <div class="result-hero">
        <div class="result-label">✅ &nbsp; Recommended Crop</div>
        <div class="crop-name-3d">{result['recommended_crop'].upper()}</div>
        <div class="conf-pill">🎯 &nbsp; {result['confidence']}% Confidence</div>
    </div>
    """, unsafe_allow_html=True)
 
    r_left, r_right = st.columns(2, gap="large")
 
    with r_left:
        st.markdown('<div class="card3d"><div class="card-title">🏆 &nbsp;Top 3 Predictions</div>', unsafe_allow_html=True)
        medals = ["🥇", "🥈", "🥉"]
        for i, (crop, conf) in enumerate(result["top_3_crops"]):
            bar = int(float(conf))
            st.markdown(f"""
            <div class="rank3d">
                <div class="rank-medal">{medals[i]}</div>
                <div class="rank-name">{crop.capitalize()}</div>
                <div class="rank-bar-bg">
                    <div class="rank-bar-fill" style="width:{bar}%"></div>
                </div>
                <div class="rank-pct">{conf}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
 
    with r_right:
        st.markdown('<div class="card3d"><div class="card-title">💰 &nbsp;Profitability Analysis</div>', unsafe_allow_html=True)
        st.caption("Estimated revenue per hectare · market price × average yield")
 
        best = result["profitability"][0]
        st.markdown(f"""
        <div class="profit3d">
            <div class="p-label">💡 Most Profitable Option</div>
            <div class="p-crop">{best['crop'].capitalize()}</div>
            <div class="p-amount">₹{best['estimated_revenue']:,.0f}</div>
            <div class="p-detail">
                ₹{best['price_per_kg']:,} / kg &nbsp;×&nbsp; {best['yield_per_hectare']:,} kg/ha yield
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        profit_df = pd.DataFrame(result["profitability"][1:])
        cols = [c for c in ["crop", "price_per_kg", "yield_per_hectare", "estimated_revenue"]
                if c in profit_df.columns]
        profit_df = profit_df[cols]
        profit_df.columns = [{"crop": "Crop", "price_per_tonne": "₹/kg",
                               "yield_per_hectare": "Yield kg/ha",
                               "estimated_revenue": "Est. Revenue ₹"}.get(c, c) for c in cols]
        profit_df["Crop"] = profit_df["Crop"].str.capitalize()
        profit_df.index = ["🥈 2nd", "🥉 3rd"]
        st.dataframe(profit_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
 
 
# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer3d">
    🌱 &nbsp; AgriSense AI &nbsp;·&nbsp; Random Forest Classifier
    &nbsp;·&nbsp; Built with Streamlit &nbsp;·&nbsp; v3.0
</div>
""", unsafe_allow_html=True)
 
 
# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 Feature Guide")
    st.markdown("""
| Feature | Unit | Meaning |
|---------|------|---------|
| **N** | mg/kg | Nitrogen – leafy growth |
| **P** | mg/kg | Phosphorus – roots & flowers |
| **K** | mg/kg | Potassium – plant health |
| **Temp** | °C | Avg air temperature |
| **Humidity** | % | Moisture in air |
| **pH** | 0–14 | Soil acidity / alkalinity |
| **Rainfall** | mm | Annual precipitation |
""")
    st.divider()
    st.markdown("## 🌲 Model Info")
    st.markdown("""
- **Algorithm**: Random Forest
- **Dataset**: 2,200 samples · 22 crops
- **Accuracy**: ~99%
- **Input Features**: 7
""")
    st.divider()
    st.markdown("## ⚗️ pH Reference")
    st.markdown("""
- **3–5** → Strongly acidic
- **5–6** → Mildly acidic *(rice, maize)*
- **6–7** → Neutral *(most veg)*
- **7–8** → Mildly alkaline *(wheat)*
- **8+**  → Strongly alkaline
""")
    st.divider()
    st.markdown("""
<div style="text-align:center;color:#225533;font-size:0.7rem;
            font-family:'Orbitron',monospace;padding-top:0.5rem;letter-spacing:1px;">
    AGRISENSE AI · v3.0
</div>
""", unsafe_allow_html=True)