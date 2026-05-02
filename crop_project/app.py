"""
app.py
======
Streamlit — AI-Based Smart Agriculture Crop Recommendation System.
UI v4: Organic-luxury green theme, botanical aesthetics, glassmorphism cards,
        leaf-inspired typography, lush gradients, animated nature effects.

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
   FONTS — Botanical Luxury
═══════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
  --green-950: #021008;
  --green-900: #052210;
  --green-800: #083d1c;
  --green-700: #0d5c28;
  --green-600: #157a34;
  --green-500: #1fa646;
  --green-400: #2dc95a;
  --green-300: #5ddd82;
  --green-200: #9aedb2;
  --green-100: #d4f7de;
  --green-50:  #f0fdf4;
  --gold:      #c9a84c;
  --gold-light:#e8c97a;
  --cream:     #faf7f0;
  --text-main: #e8f5ec;
  --text-muted:#7db890;
  --glass-bg:  rgba(5, 34, 16, 0.72);
  --glass-border: rgba(45, 201, 90, 0.18);
  --card-shadow: 0 8px 32px rgba(0,0,0,0.45), 0 2px 8px rgba(0,0,0,0.3);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
}

/* ═══════════════════════════════════════════════
   BACKGROUND — Deep forest + light rays
═══════════════════════════════════════════════ */
.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 10% 0%, rgba(21,122,52,0.22) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(13,92,40,0.18) 0%, transparent 55%),
        radial-gradient(ellipse 40% 60% at 50% 50%, rgba(8,61,28,0.12) 0%, transparent 70%),
        linear-gradient(170deg, #021008 0%, #031a0c 35%, #021008 65%, #041409 100%);
    min-height: 100vh;
}

/* Subtle leaf pattern overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(circle at 15% 25%, rgba(29,201,72,0.03) 0%, transparent 25%),
        radial-gradient(circle at 85% 75%, rgba(29,201,72,0.04) 0%, transparent 30%),
        radial-gradient(circle at 50% 50%, rgba(13,92,40,0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ═══════════════════════════════════════════════
   HIDE STREAMLIT CHROME
═══════════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
}

/* ═══════════════════════════════════════════════
   HERO BANNER — Magazine editorial
═══════════════════════════════════════════════ */
.hero {
    position: relative;
    background:
        linear-gradient(135deg, rgba(5,34,16,0.95) 0%, rgba(8,61,28,0.9) 50%, rgba(5,34,16,0.95) 100%);
    border: 1px solid rgba(45,201,90,0.2);
    border-top: 3px solid var(--green-400);
    border-radius: 0 0 32px 32px;
    padding: 3.5rem 4rem 3rem;
    margin-bottom: 2.5rem;
    overflow: hidden;
    backdrop-filter: blur(20px);
    box-shadow:
        0 20px 60px rgba(0,0,0,0.6),
        0 1px 0 rgba(45,201,90,0.4),
        inset 0 0 80px rgba(13,92,40,0.15);
}

/* Decorative circles */
.hero::before {
    content: '';
    position: absolute;
    right: -80px;
    top: -80px;
    width: 350px;
    height: 350px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(21,122,52,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    left: -50px;
    bottom: -60px;
    width: 250px;
    height: 250px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(45,201,90,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-eyebrow {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(45,201,90,0.1);
    color: var(--green-300);
    border: 1px solid rgba(45,201,90,0.3);
    border-radius: 4px;
    padding: 0.3rem 0.9rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}
.hero-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green-400);
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.7); }
}

.hero-title-wrap {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 900;
    color: #ffffff;
    margin: 0;
    line-height: 1;
    letter-spacing: -2px;
}
.hero h1 em {
    font-style: italic;
    color: var(--green-300);
    font-weight: 400;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--green-400);
    letter-spacing: 3px;
    text-transform: uppercase;
    border-left: 2px solid var(--green-500);
    padding-left: 0.8rem;
    margin: 0 0 1.2rem;
}
.hero p {
    color: var(--text-muted);
    font-size: 1rem;
    font-weight: 300;
    max-width: 520px;
    line-height: 1.75;
    margin: 0;
    letter-spacing: 0.2px;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(45,201,90,0.12);
}
.hero-stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--green-300);
    line-height: 1;
}
.hero-stat-lbl {
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 0.2rem;
    font-weight: 500;
}
.hero-leaf {
    position: absolute;
    right: 4rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 8rem;
    opacity: 0.06;
    filter: blur(2px);
    animation: leaf-sway 6s ease-in-out infinite;
    pointer-events: none;
}
@keyframes leaf-sway {
    0%, 100% { transform: translateY(-50%) rotate(-5deg); }
    50% { transform: translateY(-55%) rotate(5deg); }
}

/* ═══════════════════════════════════════════════
   GLASS CARDS
═══════════════════════════════════════════════ */
.card-glass {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 1.6rem 1.8rem 1.4rem;
    margin-bottom: 1.3rem;
    backdrop-filter: blur(16px);
    box-shadow: var(--card-shadow);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}
.card-glass::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(45,201,90,0.5), transparent);
}
.card-glass:hover {
    border-color: rgba(45,201,90,0.3);
    box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(45,201,90,0.1);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.3rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid rgba(45,201,90,0.12);
}
.card-icon {
    width: 32px; height: 32px;
    background: rgba(45,201,90,0.1);
    border: 1px solid rgba(45,201,90,0.25);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--green-300);
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

/* ═══════════════════════════════════════════════
   INPUT LABELS
═══════════════════════════════════════════════ */
label, .stSlider label, .stNumberInput label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] {
    color: var(--green-200) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.4px !important;
}

/* ═══════════════════════════════════════════════
   NUMBER INPUTS
═══════════════════════════════════════════════ */
input[type="number"] {
    background: rgba(3,20,10,0.8) !important;
    border: 1.5px solid rgba(45,201,90,0.25) !important;
    border-radius: 10px !important;
    color: var(--green-300) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
input[type="number"]:focus {
    border-color: var(--green-400) !important;
    box-shadow: 0 0 0 3px rgba(45,201,90,0.12), 0 0 16px rgba(45,201,90,0.08) !important;
}

/* ═══════════════════════════════════════════════
   SLIDERS — organic style
═══════════════════════════════════════════════ */
div[data-baseweb="slider"] > div {
    background: rgba(45,201,90,0.15) !important;
    border-radius: 99px !important;
    height: 6px !important;
}
div[data-baseweb="slider"] [role="slider"] {
    background: var(--green-400) !important;
    border: 2px solid var(--green-300) !important;
    box-shadow: 0 0 0 4px rgba(45,201,90,0.15), 0 0 16px rgba(45,201,90,0.4) !important;
    width: 18px !important;
    height: 18px !important;
    border-radius: 50% !important;
    transition: box-shadow 0.2s !important;
}
div[data-baseweb="slider"] [role="slider"]:hover {
    box-shadow: 0 0 0 6px rgba(45,201,90,0.2), 0 0 24px rgba(45,201,90,0.5) !important;
}
[data-testid="stSlider"] p {
    color: var(--green-200) !important;
}

/* ═══════════════════════════════════════════════
   PREDICT BUTTON — premium feel
═══════════════════════════════════════════════ */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0d5c28 0%, #1fa646 50%, #0d5c28 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(45,201,90,0.5) !important;
    border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    padding: 0.9rem 2.5rem !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    box-shadow:
        0 4px 20px rgba(21,122,52,0.4),
        0 1px 0 rgba(93,221,130,0.3),
        inset 0 1px 0 rgba(255,255,255,0.08) !important;
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    width: 100% !important;
    position: relative;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #157a34 0%, #2dc95a 50%, #157a34 100%) !important;
    box-shadow:
        0 6px 30px rgba(21,122,52,0.6),
        0 1px 0 rgba(93,221,130,0.5),
        inset 0 1px 0 rgba(255,255,255,0.12) !important;
    transform: translateY(-2px) scale(1.005) !important;
}
div.stButton > button[kind="primary"]:active {
    transform: translateY(0) scale(0.998) !important;
}

/* ═══════════════════════════════════════════════
   LIVE PARAM CARDS (right panel)
═══════════════════════════════════════════════ */
.param-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.5rem;
    background: rgba(3,20,10,0.5);
    border: 1px solid rgba(45,201,90,0.1);
    border-radius: 10px;
    transition: background 0.2s, border-color 0.2s, transform 0.15s;
}
.param-item:hover {
    background: rgba(13,92,40,0.3);
    border-color: rgba(45,201,90,0.2);
    transform: translateX(3px);
}
.param-label {
    font-size: 0.82rem;
    color: var(--text-muted);
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.param-value {
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--green-300);
}
.param-unit {
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-left: 0.2rem;
}

/* ═══════════════════════════════════════════════
   pH INDICATOR
═══════════════════════════════════════════════ */
.ph-meter {
    margin-top: 1rem;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.3s ease;
}
.ph-scale {
    width: 100%;
    height: 6px;
    border-radius: 99px;
    background: linear-gradient(90deg, #ff4444 0%, #ff8844 15%, #ffcc44 30%, #44cc44 45%, #44aaff 65%, #8844ff 80%, #cc44ff 100%);
    margin: 0.8rem 0 0.4rem;
    position: relative;
    box-shadow: 0 0 12px rgba(0,0,0,0.4);
}
.ph-marker {
    position: absolute;
    top: -4px;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: white;
    border: 2px solid rgba(0,0,0,0.3);
    box-shadow: 0 0 8px rgba(0,0,0,0.5);
    transform: translateX(-50%);
    transition: left 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* ═══════════════════════════════════════════════
   RESULT HERO — editorial reveal
═══════════════════════════════════════════════ */
.result-hero {
    background:
        radial-gradient(ellipse at 30% 0%, rgba(21,122,52,0.25) 0%, transparent 50%),
        linear-gradient(135deg, rgba(5,34,16,0.98), rgba(8,61,28,0.95));
    border: 1px solid rgba(45,201,90,0.3);
    border-top: 4px solid var(--green-400);
    border-radius: 24px;
    padding: 3rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    box-shadow:
        0 24px 80px rgba(0,0,0,0.65),
        0 0 0 1px rgba(45,201,90,0.08),
        inset 0 1px 0 rgba(45,201,90,0.2);
    animation: result-in 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
}
@keyframes result-in {
    from { opacity: 0; transform: translateY(20px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.result-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--green-400);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    opacity: 0.9;
}
.crop-name {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 900;
    color: #ffffff;
    letter-spacing: -2px;
    margin: 0.2rem 0 1rem;
    line-height: 1;
    text-shadow: 0 0 60px rgba(45,201,90,0.35), 0 4px 20px rgba(0,0,0,0.4);
}
.conf-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(45,201,90,0.1);
    border: 1.5px solid rgba(45,201,90,0.4);
    border-radius: 30px;
    padding: 0.45rem 1.4rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--green-300);
    box-shadow: 0 0 20px rgba(45,201,90,0.15);
}

/* ═══════════════════════════════════════════════
   RANK CARDS
═══════════════════════════════════════════════ */
.rank-item {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.7rem;
    background: rgba(3,20,10,0.6);
    border: 1px solid rgba(45,201,90,0.12);
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    transition: all 0.2s ease;
    animation: rank-in 0.4s ease both;
}
.rank-item:nth-child(1) { animation-delay: 0.05s; }
.rank-item:nth-child(2) { animation-delay: 0.12s; }
.rank-item:nth-child(3) { animation-delay: 0.19s; }
@keyframes rank-in {
    from { opacity: 0; transform: translateX(-16px); }
    to   { opacity: 1; transform: translateX(0); }
}
.rank-item:hover {
    background: rgba(13,92,40,0.3);
    border-color: rgba(45,201,90,0.22);
    transform: translateX(4px);
}
.rank-medal { font-size: 1.5rem; min-width: 2rem; }
.rank-name  {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600; font-size: 1rem;
    color: var(--green-100); flex: 1;
    letter-spacing: 0.3px;
}
.rank-bar-bg {
    flex: 2; height: 8px;
    background: rgba(45,201,90,0.08);
    border-radius: 99px;
    overflow: hidden;
    border: 1px solid rgba(45,201,90,0.1);
}
.rank-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--green-700), var(--green-400));
    border-radius: 99px;
    box-shadow: 0 0 8px rgba(45,201,90,0.4);
    transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.rank-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem; font-weight: 700;
    color: var(--green-300); min-width: 3.5rem; text-align: right;
}

/* ═══════════════════════════════════════════════
   PROFIT CARD
═══════════════════════════════════════════════ */
.profit-card {
    background: linear-gradient(135deg, #161000, #221800, #161000);
    border: 1px solid rgba(201,168,76,0.3);
    border-top: 3px solid var(--gold);
    border-radius: 18px;
    padding: 1.6rem;
    margin-bottom: 1rem;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.5),
        0 0 40px rgba(201,168,76,0.05),
        inset 0 1px 0 rgba(232,201,122,0.1);
    animation: profit-in 0.5s ease 0.2s both;
}
@keyframes profit-in {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.profit-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--gold);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    opacity: 0.8;
}
.profit-crop {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem; font-weight: 600;
    color: #efe8d0; margin-bottom: 0.3rem;
}
.profit-amount {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem; font-weight: 700;
    color: var(--gold-light);
    text-shadow: 0 0 30px rgba(201,168,76,0.35);
    letter-spacing: -1px; line-height: 1;
}
.profit-detail {
    font-size: 0.8rem; color: #998855;
    margin-top: 0.5rem; font-weight: 400;
}

/* ═══════════════════════════════════════════════
   METRIC SUMMARY CARDS (right panel)
═══════════════════════════════════════════════ */
.met-row {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.45rem;
    background: rgba(3,20,10,0.5);
    border: 1px solid rgba(45,201,90,0.1);
    border-radius: 9px;
    transition: all 0.15s ease;
}
.met-row:hover { border-color: rgba(45,201,90,0.2); transform: translateX(3px); }
.met-lbl { font-size: 0.8rem; color: var(--text-muted); font-weight: 500; }
.met-val {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem; font-weight: 700; color: var(--green-300);
}
.met-unit { font-size: 0.65rem; color: var(--text-muted); margin-left: 0.2rem; }

/* ═══════════════════════════════════════════════
   SECTION DIVIDER
═══════════════════════════════════════════════ */
.section-divider {
    display: flex; align-items: center; gap: 1rem;
    margin: 1.8rem 0 1.4rem;
}
.divider-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(45,201,90,0.2), transparent);
}
.divider-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; color: var(--text-muted);
    letter-spacing: 2px; text-transform: uppercase;
}

/* ═══════════════════════════════════════════════
   ERROR BANNER
═══════════════════════════════════════════════ */
.err-banner {
    background: rgba(26,5,5,0.9);
    border: 1px solid rgba(255,80,80,0.35);
    border-left: 3px solid #ff5050;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #ff9999;
    font-weight: 500;
    font-size: 0.92rem;
    box-shadow: 0 4px 20px rgba(255,0,0,0.1);
}

/* ═══════════════════════════════════════════════
   FOOTER
═══════════════════════════════════════════════ */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    border-top: 1px solid rgba(45,201,90,0.08);
    margin-top: 2.5rem;
}
.footer-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1rem; font-style: italic;
    color: var(--green-600);
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.footer-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem; color: rgba(45,201,90,0.25);
    letter-spacing: 2px; text-transform: uppercase;
}

/* ═══════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020e05, #031209) !important;
    border-right: 1px solid rgba(45,201,90,0.1) !important;
}
[data-testid="stSidebar"] * {
    color: #7db890 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--green-300) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(45,201,90,0.1) !important;
}
[data-testid="stSidebar"] table td,
[data-testid="stSidebar"] table th {
    color: #6aaa80 !important; font-size: 0.8rem !important;
}
[data-testid="stSidebar"] table th {
    color: var(--green-300) !important;
}

/* ═══════════════════════════════════════════════
   DATAFRAME
═══════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(45,201,90,0.15) !important;
}
.stCaption, small {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
}
[data-testid="stSpinner"] p {
    color: var(--green-300) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 1px !important;
}

/* ═══════════════════════════════════════════════
   RESPONSIVE
═══════════════════════════════════════════════ */
@media (max-width: 768px) {
    .hero { padding: 2rem 1.5rem; }
    .hero h1 { font-size: 2.5rem; }
    .crop-name { font-size: 2.5rem; }
    .profit-amount { font-size: 2rem; }
    .hero-stats { gap: 1.5rem; }
    .hero-leaf { display: none; }
}
</style>
""", unsafe_allow_html=True)


# ── HERO BANNER ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-leaf">🌿</div>
    <div class="hero-eyebrow">
        <div class="hero-dot"></div>
        <span class="hero-tag">AI-Powered · Random Forest · 22 Crops</span>
    </div>
    <div class="hero-title-wrap">
        <h1>Agri<em>Sense</em></h1>
    </div>
    <p class="hero-sub">Intelligent Crop Recommendation Engine</p>
    <p>Enter your soil composition, local climate data, and pH level — the AI engine will analyse 7 parameters to recommend the optimal crop and project profitability per hectare.</p>
    <div class="hero-stats">
        <div>
            <div class="hero-stat-val">~99%</div>
            <div class="hero-stat-lbl">Accuracy</div>
        </div>
        <div>
            <div class="hero-stat-val">22</div>
            <div class="hero-stat-lbl">Crop Types</div>
        </div>
        <div>
            <div class="hero-stat-val">2.2K</div>
            <div class="hero-stat-lbl">Training Samples</div>
        </div>
        <div>
            <div class="hero-stat-val">7</div>
            <div class="hero-stat-lbl">Input Parameters</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")

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
    st.markdown("""
    <div class="card-glass">
        <div class="card-header">
            <div class="card-icon">⚗️</div>
            <div class="card-title">Soil Nutrients</div>
        </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        N = st.number_input("Nitrogen (N) mg/kg", min_value=0, max_value=200, value=90,
                            help="Supports leafy growth and chlorophyll production.")
    with c2:
        P = st.number_input("Phosphorus (P) mg/kg", min_value=0, max_value=200, value=42,
                            help="Essential for root development and flowering.")
    with c3:
        K = st.number_input("Potassium (K) mg/kg", min_value=0, max_value=250, value=43,
                            help="Regulates overall plant health and disease resistance.")
    st.markdown('</div>', unsafe_allow_html=True)

    # SECTION 2 — Weather
    st.markdown("""
    <div class="card-glass">
        <div class="card-header">
            <div class="card-icon">🌤️</div>
            <div class="card-title">Climate Conditions</div>
        </div>
    """, unsafe_allow_html=True)
    w1, w2 = st.columns(2)
    with w1:
        temperature = st.slider("Temperature (°C)", 5.0, 50.0, 21.0, 0.5)
    with w2:
        humidity = st.slider("Relative Humidity (%)", 10.0, 100.0, 82.0, 0.5)
    rainfall = st.slider("Annual Rainfall (mm)", 20.0, 300.0, 203.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # SECTION 3 — pH
    st.markdown("""
    <div class="card-glass">
        <div class="card-header">
            <div class="card-icon">🧫</div>
            <div class="card-title">Soil pH</div>
        </div>
    """, unsafe_allow_html=True)
    ph = st.slider("pH Value", 3.0, 10.0, 6.5, 0.1)

    # pH classification
    if ph < 5:
        ph_color, ph_bg, ph_text, ph_emoji = "#ff6b6b", "rgba(255,80,80,0.1)", "Strongly Acidic", "🔴"
    elif ph < 6:
        ph_color, ph_bg, ph_text, ph_emoji = "#ff9944", "rgba(255,150,50,0.1)", "Mildly Acidic", "🟠"
    elif ph < 7:
        ph_color, ph_bg, ph_text, ph_emoji = "#2dc95a", "rgba(45,201,90,0.1)", "Neutral — Optimal", "🟢"
    elif ph < 8:
        ph_color, ph_bg, ph_text, ph_emoji = "#44aaff", "rgba(50,150,255,0.1)", "Mildly Alkaline", "🔵"
    else:
        ph_color, ph_bg, ph_text, ph_emoji = "#cc88ff", "rgba(180,100,255,0.1)", "Strongly Alkaline", "🟣"

    # pH scale marker position (3.0–10.0 range → 0–100%)
    marker_pct = ((ph - 3.0) / 7.0) * 100

    st.markdown(f"""
    <div class="ph-meter" style="background:{ph_bg};border:1px solid {ph_color}33;">
        <span style="color:{ph_color};font-weight:700;font-size:1rem;">{ph_emoji} pH {ph:.1f}</span>
        <span style="color:{ph_color};font-size:0.85rem;font-weight:600;opacity:0.85;">{ph_text}</span>
    </div>
    <div class="ph-scale">
        <div class="ph-marker" style="left:{marker_pct}%;background:{ph_color};border-color:{ph_color}88;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:#336644;font-family:'Space Mono',monospace;margin-top:0.2rem;">
        <span>3.0 Acid</span><span>7.0 Neutral</span><span>10.0 Alkaline</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# RIGHT PANEL — live summary
with right_col:
    st.markdown("""
    <div class="card-glass">
        <div class="card-header">
            <div class="card-icon">📡</div>
            <div class="card-title">Live Parameters</div>
        </div>
    """, unsafe_allow_html=True)

    params = [
        ("🌿", "Nitrogen",     f"{N}",           "mg/kg"),
        ("🌸", "Phosphorus",   f"{P}",           "mg/kg"),
        ("💎", "Potassium",    f"{K}",           "mg/kg"),
        ("🌡️", "Temperature", f"{temperature}", "°C"),
        ("💧", "Humidity",     f"{humidity}",    "%"),
        ("⚗️", "pH Level",    f"{ph}",          ""),
        ("🌧️", "Rainfall",    f"{rainfall}",    "mm"),
    ]
    for icon, label, val, unit in params:
        st.markdown(f"""
        <div class="param-item">
            <span class="param-label">{icon} {label}</span>
            <span>
                <span class="param-value">{val}</span>
                <span class="param-unit">{unit}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Nutrient balance visual
    st.markdown("""
    <div class="card-glass" style="padding:1.3rem 1.6rem;">
        <div class="card-header" style="margin-bottom:1rem;">
            <div class="card-icon">📊</div>
            <div class="card-title">Nutrient Balance</div>
        </div>
    """, unsafe_allow_html=True)

    total = max(N + P + K, 1)
    for nutrient, val, color in [("N", N, "#2dc95a"), ("P", P, "#44aaff"), ("K", K, "#ffcc44")]:
        pct = int((val / max(total, 1)) * 100)
        st.markdown(f"""
        <div style="margin-bottom:0.6rem;">
            <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem;">
                <span style="font-size:0.75rem;color:#7db890;font-weight:600;">{nutrient}</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.72rem;color:{color};">{val} mg/kg</span>
            </div>
            <div style="height:5px;background:rgba(255,255,255,0.05);border-radius:99px;overflow:hidden;">
                <div style="width:{min(pct*2,100)}%;height:100%;background:{color};border-radius:99px;box-shadow:0 0 8px {color}66;transition:width 0.5s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── PREDICT BUTTON ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_mid, _ = st.columns([1, 2, 1])
with btn_mid:
    predict_btn = st.button("⚡  Analyse & Recommend Crop", use_container_width=True, type="primary")


# ── RESULTS ────────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Running AI inference — analysing 7 parameters…"):
        try:
            result = predict_crop(N=N, P=P, K=K,
                                  temperature=temperature,
                                  humidity=humidity,
                                  ph=ph,
                                  rainfall=rainfall)
        except FileNotFoundError as e:
            st.markdown(f'<div class="err-banner">❌ {e}</div>', unsafe_allow_html=True)
            st.stop()

    st.markdown("<br>", unsafe_allow_html=True)

    # Big result hero
    st.markdown(f"""
    <div class="result-hero">
        <div class="result-eyebrow">✦ &nbsp; AI Recommendation &nbsp; ✦</div>
        <div class="crop-name">{result['recommended_crop'].capitalize()}</div>
        <div class="conf-tag">
            <span style="width:7px;height:7px;border-radius:50%;background:#2dc95a;display:inline-block;box-shadow:0 0 8px #2dc95a;"></span>
            {result['confidence']}% Confidence
        </div>
    </div>
    """, unsafe_allow_html=True)

    r_left, r_right = st.columns(2, gap="large")

    with r_left:
        st.markdown("""
        <div class="card-glass">
            <div class="card-header">
                <div class="card-icon">🏆</div>
                <div class="card-title">Top 3 Predictions</div>
            </div>
        """, unsafe_allow_html=True)
        medals = ["🥇", "🥈", "🥉"]
        for i, (crop, conf) in enumerate(result["top_3_crops"]):
            bar = int(float(conf))
            st.markdown(f"""
            <div class="rank-item">
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
        st.markdown("""
        <div class="card-glass">
            <div class="card-header">
                <div class="card-icon">💰</div>
                <div class="card-title">Profitability Analysis</div>
            </div>
        """, unsafe_allow_html=True)
        st.caption("Estimated revenue per hectare · market price × average yield")

        best = result["profitability"][0]
        st.markdown(f"""
        <div class="profit-card">
            <div class="profit-label">💡 Most Profitable Option</div>
            <div class="profit-crop">{best['crop'].capitalize()}</div>
            <div class="profit-amount">₹{best['estimated_revenue']:,.0f}</div>
            <div class="profit-detail">
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
<div class="footer">
    <div class="footer-brand">AgriSense AI</div>
    <div class="footer-sub">Random Forest Classifier · Built with Streamlit · v4.0</div>
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
| **Temp** | °C | Average air temperature |
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
- **6–7** → Neutral *(most vegetables)*
- **7–8** → Mildly alkaline *(wheat)*
- **8+** → Strongly alkaline
""")
    st.divider()
    st.markdown("""
<div style="text-align:center;color:#225533;font-size:0.65rem;
            font-family:'Space Mono',monospace;padding-top:0.5rem;letter-spacing:2px;">
    AGRISENSE AI · v4.0
</div>
""", unsafe_allow_html=True)
