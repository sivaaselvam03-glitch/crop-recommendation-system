"""
app.py
======
Streamlit — AI-Based Smart Agriculture Crop Recommendation System.
UI v5: Premium Nature / Eco-Agriculture theme.
        Deep forest greens, earthy tones, amber/gold accents.
        Clean bold layout with organic warmth.
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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=Nunito+Sans:wght@300;400;600;700&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito Sans', sans-serif;
    font-size: 15px;
}

/* ═══════════════════════════════════════════════
   CSS VARIABLES
═══════════════════════════════════════════════ */
:root {
    --forest:        #064e3b;
    --deep:          #0b3d2e;
    --mid-green:     #065f46;
    --leaf:          #10b981;
    --sage:          #6ee7b7;
    --amber:         #f59e0b;
    --amber-dark:    #d97706;
    --amber-light:   #fcd34d;
    --earth:         #78350f;
    --cream:         #f0fdf4;
    --white:         #ffffff;
    --card-bg:       #ffffff;
    --card-alt:      #ecfdf5;
    --text-dark:     #052e16;
    --text-body:     #374151;
    --text-muted:    #6b7280;
    --border:        rgba(16,185,129,0.2);
    --shadow-sm:     0 4px 16px rgba(6,78,59,0.12);
    --shadow-md:     0 8px 32px rgba(6,78,59,0.18);
    --shadow-lg:     0 20px 60px rgba(6,78,59,0.25);
    --shadow-amber:  0 8px 30px rgba(245,158,11,0.3);
    --radius-sm:     10px;
    --radius-md:     16px;
    --radius-lg:     20px;
    --radius-xl:     28px;
}

/* ═══════════════════════════════════════════════
   BACKGROUND
═══════════════════════════════════════════════ */
.stApp {
    background:
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%2310b981' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E"),
        linear-gradient(160deg, #f0fdf4 0%, #ecfdf5 40%, #d1fae5 100%);
    min-height: 100vh;
}

/* ═══════════════════════════════════════════════
   HIDE STREAMLIT CHROME
═══════════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}

/* ═══════════════════════════════════════════════
   NAVBAR
═══════════════════════════════════════════════ */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 3rem;
    background: var(--deep);
    border-bottom: 2px solid var(--mid-green);
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 4px 20px rgba(6,46,32,0.35);
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}
.navbar-icon {
    font-size: 1.5rem;
    filter: drop-shadow(0 0 8px rgba(16,185,129,0.5));
}
.navbar-wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--cream);
    letter-spacing: -0.5px;
}
.navbar-wordmark em {
    font-style: normal;
    color: var(--amber);
}
.navbar-links {
    display: flex;
    align-items: center;
    gap: 2.5rem;
    font-family: 'Manrope', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: rgba(240,253,244,0.6);
    letter-spacing: 0.5px;
}
.navbar-links span { cursor: default; transition: color 0.2s ease; }
.navbar-links span:hover { color: var(--sage); }
.navbar-cta {
    background: var(--amber);
    color: var(--earth) !important;
    padding: 0.5rem 1.4rem;
    border-radius: 30px;
    font-weight: 700;
    color: #fff;
    cursor: default;
    box-shadow: var(--shadow-amber);
    transition: all 0.25s ease;
}
.navbar-cta:hover {
    background: var(--amber-dark) !important;
    transform: scale(1.04);
}

/* ═══════════════════════════════════════════════
   HERO SECTION
═══════════════════════════════════════════════ */
.hero-wrap {
    background: linear-gradient(130deg, var(--deep) 0%, var(--forest) 55%, #065f46 100%);
    padding: 4.5rem 3rem 4rem;
    position: relative;
    overflow: hidden;
}
.hero-bg-leaf {
    position: absolute;
    right: -60px;
    top: -40px;
    font-size: 22rem;
    opacity: 0.04;
    line-height: 1;
    pointer-events: none;
    transform: rotate(-20deg);
    filter: blur(2px);
}
.hero-bg-leaf2 {
    position: absolute;
    left: -40px;
    bottom: -60px;
    font-size: 16rem;
    opacity: 0.03;
    pointer-events: none;
    transform: rotate(30deg);
    filter: blur(1px);
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.4);
    color: var(--sage);
    border-radius: 30px;
    padding: 0.35rem 1.1rem;
    font-family: 'Manrope', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
}
.hero-badge-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--leaf);
    box-shadow: 0 0 8px var(--leaf);
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
.hero-h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    font-weight: 900;
    color: #f0fdf4;
    line-height: 1.08;
    letter-spacing: -1.5px;
    margin: 0 0 1rem;
    max-width: 680px;
}
.hero-h1 .accent { color: var(--amber); }
.hero-sub {
    color: rgba(240,253,244,0.65);
    font-size: 1.05rem;
    max-width: 520px;
    line-height: 1.75;
    margin-bottom: 2.2rem;
    font-weight: 400;
}
.hero-actions {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.btn-hero {
    display: inline-block;
    background: var(--amber);
    color: #1c1917;
    font-family: 'Manrope', sans-serif;
    font-size: 0.9rem;
    font-weight: 800;
    padding: 0.85rem 2.2rem;
    border-radius: 14px;
    letter-spacing: 0.5px;
    box-shadow: var(--shadow-amber);
    cursor: default;
    transition: all 0.25s ease;
}
.btn-hero:hover {
    background: var(--amber-dark);
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 12px 40px rgba(245,158,11,0.45);
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    margin-top: 3rem;
    flex-wrap: wrap;
}
.h-stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: #f0fdf4;
    line-height: 1;
}
.h-stat-num em {
    font-style: normal;
    color: var(--amber);
}
.h-stat-lbl {
    font-size: 0.72rem;
    color: rgba(240,253,244,0.45);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.2rem;
}
.hero-divider {
    width: 1px;
    height: 40px;
    background: rgba(255,255,255,0.1);
    align-self: center;
}

/* ═══════════════════════════════════════════════
   FEATURE STRIP
═══════════════════════════════════════════════ */
.feature-strip {
    background: var(--mid-green);
    padding: 1.5rem 3rem;
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.feat-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Manrope', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: rgba(240,253,244,0.8);
    letter-spacing: 0.5px;
}
.feat-item .fi-icon { font-size: 1.1rem; }

/* ═══════════════════════════════════════════════
   CONTENT WRAPPER
═══════════════════════════════════════════════ */
.content-pad {
    padding: 2.5rem 3rem;
    max-width: 1300px;
    margin: 0 auto;
}

/* ═══════════════════════════════════════════════
   SECTION LABEL
═══════════════════════════════════════════════ */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Manrope', sans-serif;
    font-size: 0.72rem;
    font-weight: 800;
    color: var(--leaf);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text-dark);
    letter-spacing: -0.8px;
    margin-bottom: 1.5rem;
    line-height: 1.2;
}

/* ═══════════════════════════════════════════════
   INPUT CARD
═══════════════════════════════════════════════ */
.input-card {
    background: var(--card-bg);
    border: 1.5px solid rgba(16,185,129,0.2);
    border-radius: var(--radius-lg);
    padding: 1.8rem 2rem 1.6rem;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
}
.input-card:hover {
    box-shadow: var(--shadow-md);
    border-color: rgba(16,185,129,0.35);
}
.input-card-title {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-family: 'Manrope', sans-serif;
    font-size: 0.72rem;
    font-weight: 800;
    color: var(--forest);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.3rem;
    padding-bottom: 0.75rem;
    border-bottom: 1.5px solid #d1fae5;
}
.input-card-title .ict-icon {
    width: 24px; height: 24px;
    background: linear-gradient(135deg, var(--forest), var(--leaf));
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
}

/* ═══════════════════════════════════════════════
   OVERRIDE STREAMLIT INPUTS
═══════════════════════════════════════════════ */
label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] {
    color: #374151 !important;
    font-family: 'Nunito Sans', sans-serif !important;
    font-size: 0.87rem !important;
    font-weight: 700 !important;
}
input[type="number"] {
    background: #f9fafb !important;
    border: 1.5px solid #d1fae5 !important;
    border-radius: 10px !important;
    color: var(--text-dark) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
input[type="number"]:focus {
    border-color: var(--leaf) !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.12) !important;
}

/* ── SLIDERS ── */
div[data-baseweb="slider"] > div {
    background: #d1fae5 !important;
}
div[data-baseweb="slider"] [role="slider"] {
    background: linear-gradient(135deg, var(--forest), var(--leaf)) !important;
    border-color: var(--leaf) !important;
    box-shadow: 0 0 12px rgba(16,185,129,0.5) !important;
    width: 20px !important;
    height: 20px !important;
}

/* ═══════════════════════════════════════════════
   SUMMARY PANEL
═══════════════════════════════════════════════ */
.summary-panel {
    background: linear-gradient(160deg, var(--deep) 0%, var(--forest) 100%);
    border-radius: var(--radius-xl);
    padding: 1.8rem 1.6rem;
    box-shadow: var(--shadow-lg);
    position: sticky;
    top: 90px;
}
.sp-title {
    font-family: 'Manrope', sans-serif;
    font-size: 0.68rem;
    font-weight: 800;
    color: var(--sage);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(110,231,183,0.15);
}
.sp-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.sp-row:last-child { border-bottom: none; }
.sp-label {
    font-size: 0.82rem;
    color: rgba(240,253,244,0.5);
    font-weight: 500;
}
.sp-value {
    font-family: 'Manrope', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: #6ee7b7;
}
.sp-unit {
    font-size: 0.7rem;
    color: rgba(110,231,183,0.5);
    margin-left: 3px;
}

/* ═══════════════════════════════════════════════
   ANALYSE BUTTON
═══════════════════════════════════════════════ */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #b45309 0%, var(--amber) 50%, var(--amber-light) 100%) !important;
    color: #1c1917 !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 800 !important;
    padding: 0.9rem 2.5rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    box-shadow: var(--shadow-amber) !important;
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    width: 100% !important;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 14px 40px rgba(245,158,11,0.45) !important;
    filter: brightness(1.05) !important;
}
div.stButton > button[kind="primary"]:active {
    transform: translateY(-1px) scale(0.99) !important;
}

/* ═══════════════════════════════════════════════
   RESULT AREA
═══════════════════════════════════════════════ */
.result-banner {
    background: linear-gradient(130deg, var(--deep) 0%, var(--forest) 60%, #065f46 100%);
    border-radius: var(--radius-xl);
    padding: 3rem 2.5rem 2.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-lg);
    border: 1px solid rgba(16,185,129,0.2);
}
.result-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--amber), var(--leaf), var(--amber));
}
.rb-eyebrow {
    font-family: 'Manrope', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--sage);
    margin-bottom: 0.6rem;
}
.rb-cropname {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 900;
    color: #f0fdf4;
    letter-spacing: -2px;
    line-height: 1;
    margin: 0.3rem 0 1rem;
}
.rb-cropname .accent { color: var(--amber); }
.confidence-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(16,185,129,0.12);
    border: 1.5px solid rgba(16,185,129,0.35);
    border-radius: 30px;
    padding: 0.45rem 1.6rem;
    font-family: 'Manrope', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--sage);
    box-shadow: 0 0 20px rgba(16,185,129,0.1);
}

/* ═══════════════════════════════════════════════
   RANK CARDS
═══════════════════════════════════════════════ */
.rank-card {
    background: var(--card-bg);
    border: 1.5px solid #d1fae5;
    border-radius: var(--radius-md);
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.rank-card:hover {
    transform: translateX(5px);
    box-shadow: var(--shadow-md);
    border-color: var(--leaf);
}
.rc-medal { font-size: 1.6rem; min-width: 2rem; }
.rc-name  { font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 0.98rem; color: var(--text-dark); flex: 1; }
.rc-bar-bg { flex: 2; height: 7px; background: #d1fae5; border-radius: 99px; overflow: hidden; }
.rc-bar-fill { height: 100%; background: linear-gradient(90deg, var(--forest), var(--leaf)); border-radius: 99px; box-shadow: 0 0 8px rgba(16,185,129,0.4); }
.rc-pct { font-family: 'Manrope', sans-serif; font-size: 0.9rem; font-weight: 800; color: var(--forest); min-width: 3.5rem; text-align: right; }

/* ═══════════════════════════════════════════════
   PROFIT CARD
═══════════════════════════════════════════════ */
.profit-card {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 2px solid rgba(245,158,11,0.3);
    border-radius: var(--radius-lg);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 28px rgba(245,158,11,0.12);
    transition: box-shadow 0.25s ease, border-color 0.25s ease;
    position: relative;
    overflow: hidden;
}
.profit-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--amber), var(--amber-light), var(--amber));
}
.profit-card:hover {
    box-shadow: 0 12px 40px rgba(245,158,11,0.22);
    border-color: rgba(245,158,11,0.5);
}
.pc-eyebrow { font-family: 'Manrope', sans-serif; font-size: 0.68rem; color: var(--earth); letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 0.4rem; font-weight: 800; }
.pc-crop    { font-family: 'Manrope', sans-serif; font-size: 1.05rem; font-weight: 700; color: #44200a; margin-bottom: 0.3rem; }
.pc-amount  { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 900; color: #92400e; letter-spacing: -1px; line-height: 1.1; }
.pc-detail  { font-size: 0.8rem; color: #a16207; margin-top: 0.4rem; font-weight: 600; }

/* ═══════════════════════════════════════════════
   PH BADGE
═══════════════════════════════════════════════ */
.ph-badge {
    display: inline-block;
    padding: 0.35rem 1.2rem;
    border-radius: 20px;
    font-family: 'Manrope', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    margin-top: 0.6rem;
    letter-spacing: 0.3px;
    border: 1.5px solid;
}

/* ═══════════════════════════════════════════════
   CTA SECTION
═══════════════════════════════════════════════ */
.cta-section {
    background: linear-gradient(130deg, var(--deep) 0%, #065f46 100%);
    border-radius: var(--radius-xl);
    padding: 3.5rem 2rem;
    text-align: center;
    margin: 3rem 0 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
}
.cta-section::before {
    content: '🌱';
    position: absolute;
    font-size: 14rem;
    left: -2rem;
    top: -3rem;
    opacity: 0.04;
    pointer-events: none;
}
.cta-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: #f0fdf4;
    letter-spacing: -0.8px;
    margin-bottom: 0.7rem;
}
.cta-sub {
    color: rgba(240,253,244,0.55);
    font-size: 0.95rem;
    margin-bottom: 1.8rem;
    max-width: 440px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.7;
}
.cta-btn {
    display: inline-block;
    background: var(--amber);
    color: #1c1917;
    font-family: 'Manrope', sans-serif;
    font-size: 0.88rem;
    font-weight: 800;
    padding: 0.8rem 2.4rem;
    border-radius: 12px;
    letter-spacing: 0.5px;
    box-shadow: var(--shadow-amber);
    cursor: default;
    transition: all 0.25s ease;
}
.cta-btn:hover {
    background: var(--amber-dark);
    transform: translateY(-2px);
    box-shadow: 0 10px 36px rgba(245,158,11,0.45);
}

/* ═══════════════════════════════════════════════
   INFO FEATURE CARDS (3-col strip)
═══════════════════════════════════════════════ */
.info-card {
    background: var(--card-bg);
    border: 1.5px solid #d1fae5;
    border-radius: var(--radius-lg);
    padding: 1.8rem 1.6rem;
    text-align: left;
    box-shadow: var(--shadow-sm);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    height: 100%;
}
.info-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-md);
}
.info-icon {
    font-size: 2.2rem;
    margin-bottom: 1rem;
    display: block;
}
.info-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--text-dark);
    margin-bottom: 0.5rem;
}
.info-body {
    font-size: 0.87rem;
    color: var(--text-muted);
    line-height: 1.7;
}

/* ═══════════════════════════════════════════════
   ERROR BANNER
═══════════════════════════════════════════════ */
.err-banner {
    background: #fef2f2;
    border: 1.5px solid #fca5a5;
    border-radius: var(--radius-md);
    padding: 1rem 1.4rem;
    color: #991b1b;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 4px 16px rgba(239,68,68,0.08);
}

/* ═══════════════════════════════════════════════
   FOOTER
═══════════════════════════════════════════════ */
.site-footer {
    background: var(--deep);
    padding: 2.2rem 3rem;
    border-top: 2px solid var(--mid-green);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    margin-top: 3rem;
}
.footer-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #ecfdf5;
}
.footer-brand em { font-style: normal; color: var(--amber); }
.footer-links {
    display: flex;
    gap: 2rem;
    font-family: 'Manrope', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: rgba(240,253,244,0.4);
    letter-spacing: 0.5px;
}
.footer-copy {
    font-size: 0.75rem;
    color: rgba(240,253,244,0.25);
    font-family: 'Nunito Sans', sans-serif;
}

/* ═══════════════════════════════════════════════
   DIVIDER
═══════════════════════════════════════════════ */
.eco-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #d1fae5, transparent);
    margin: 2rem 0;
}

/* ═══════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--deep), var(--forest)) !important;
    border-right: 1px solid rgba(16,185,129,0.15) !important;
}
[data-testid="stSidebar"] * {
    color: rgba(240,253,244,0.55) !important;
    font-family: 'Nunito Sans', sans-serif !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--sage) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(16,185,129,0.15) !important; }
[data-testid="stSidebar"] table td,
[data-testid="stSidebar"] table th { color: rgba(240,253,244,0.45) !important; font-size: 0.8rem !important; }
[data-testid="stSidebar"] table th { color: var(--sage) !important; }

/* ═══════════════════════════════════════════════
   DATAFRAME
═══════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1.5px solid #d1fae5 !important;
}
.stCaption, small { color: var(--text-muted) !important; font-size: 0.82rem !important; }
[data-testid="stSpinner"] p { color: var(--leaf) !important; }

/* ═══════════════════════════════════════════════
   RESPONSIVE
═══════════════════════════════════════════════ */
@media (max-width: 768px) {
    .hero-h1 { font-size: 2.4rem; }
    .rb-cropname { font-size: 2.6rem; }
    .pc-amount { font-size: 2rem; }
    .hero-stats { gap: 1.2rem; }
    .navbar-links { display: none; }
    .site-footer { flex-direction: column; text-align: center; }
}
</style>
""", unsafe_allow_html=True)


# ── NAVBAR ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <div class="navbar-icon">🌿</div>
        <div class="navbar-wordmark">Agri<em>Sense</em></div>
    </div>
    <div class="navbar-links">
        <span>About</span>
        <span>Solutions</span>
        <span>Insights</span>
        <span class="navbar-cta">Start Now</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-bg-leaf">🌾</div>
    <div class="hero-bg-leaf2">🌿</div>

    <div class="hero-badge">
        <div class="hero-badge-dot"></div>
        AI-Powered · 22 Crops · 99% Accuracy
    </div>

    <h1 class="hero-h1">Join The <span class="accent">Green</span><br>Revolution</h1>
    <p class="hero-sub">
        Input your soil and climate data — our Random Forest AI analyses 7
        parameters to recommend the ideal crop and estimate your profit per hectare.
    </p>
    <div class="hero-actions">
        <div class="btn-hero">🌱 &nbsp;Start Analysing</div>
        <span style="color:rgba(240,253,244,0.4);font-size:0.85rem;font-weight:600;">Trusted by 40K+ farmers</span>
    </div>

    <div class="hero-stats">
        <div>
            <div class="h-stat-num">40<em>K+</em></div>
            <div class="h-stat-lbl">Active Members</div>
        </div>
        <div class="hero-divider"></div>
        <div>
            <div class="h-stat-num">99<em>%</em></div>
            <div class="h-stat-lbl">Accuracy</div>
        </div>
        <div class="hero-divider"></div>
        <div>
            <div class="h-stat-num">22</div>
            <div class="h-stat-lbl">Crop Classes</div>
        </div>
        <div class="hero-divider"></div>
        <div>
            <div class="h-stat-num">2200<em>+</em></div>
            <div class="h-stat-lbl">Training Samples</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── FEATURE STRIP ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="feature-strip">
    <div class="feat-item"><span class="fi-icon">🧬</span> Random Forest Classifier</div>
    <div class="feat-item"><span class="fi-icon">⚡</span> Real-Time Inference</div>
    <div class="feat-item"><span class="fi-icon">💰</span> Profit Estimation</div>
    <div class="feat-item"><span class="fi-icon">🌡️</span> 7 Input Features</div>
    <div class="feat-item"><span class="fi-icon">🏆</span> Top-3 Crop Ranking</div>
</div>
""", unsafe_allow_html=True)


# ── MODEL CHECK ────────────────────────────────────────────────────────────────
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


# ── INFO FEATURE CARDS ─────────────────────────────────────────────────────────
st.markdown('<div style="padding: 2.5rem 3rem 0;">', unsafe_allow_html=True)
st.markdown('<div class="section-label">🌿 &nbsp; Why AgriSense</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Precision Agriculture, Simplified</div>', unsafe_allow_html=True)

fc1, fc2, fc3 = st.columns(3, gap="medium")
with fc1:
    st.markdown("""
    <div class="info-card">
        <span class="info-icon">🧠</span>
        <div class="info-title">AI-Driven Intelligence</div>
        <div class="info-body">Our Random Forest model trained on 2,200+ samples delivers ~99% accuracy across 22 crop types — giving you confidence in every decision.</div>
    </div>
    """, unsafe_allow_html=True)
with fc2:
    st.markdown("""
    <div class="info-card">
        <span class="info-icon">📊</span>
        <div class="info-title">Profit Forecasting</div>
        <div class="info-body">Get estimated revenue per hectare instantly. Compare top 3 crops side-by-side to choose the most financially rewarding option for your land.</div>
    </div>
    """, unsafe_allow_html=True)
with fc3:
    st.markdown("""
    <div class="info-card">
        <span class="info-icon">🌱</span>
        <div class="info-title">Soil & Climate Aware</div>
        <div class="info-body">Input 7 real-world parameters — N, P, K levels, temperature, humidity, pH and rainfall — to receive hyper-personalised crop recommendations.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── MAIN INPUT AREA ────────────────────────────────────────────────────────────
st.markdown('<div style="padding: 2rem 3rem 0;">', unsafe_allow_html=True)
st.markdown('<div class="section-label">⚗️ &nbsp; Soil & Climate Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Enter Your Field Parameters</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([2, 1], gap="large")

with left_col:

    # SOIL NUTRIENTS
    st.markdown("""
    <div class="input-card">
        <div class="input-card-title">
            <div class="ict-icon">⚗️</div>
            Soil Nutrients
        </div>
    """, unsafe_allow_html=True)
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

    # WEATHER
    st.markdown("""
    <div class="input-card">
        <div class="input-card-title">
            <div class="ict-icon">🌤️</div>
            Weather Conditions
        </div>
    """, unsafe_allow_html=True)
    w1, w2 = st.columns(2)
    with w1:
        temperature = st.slider("Temperature (°C)", 5.0, 50.0, 21.0, 0.5)
    with w2:
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 82.0, 0.5)
    rainfall = st.slider("Annual Rainfall (mm)", 20.0, 300.0, 203.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # SOIL pH
    st.markdown("""
    <div class="input-card">
        <div class="input-card-title">
            <div class="ict-icon">🧫</div>
            Soil pH
        </div>
    """, unsafe_allow_html=True)
    ph = st.slider("pH Value", 3.0, 10.0, 6.5, 0.1)
    if ph < 5:
        ph_color, ph_bg, ph_border, ph_text = "#dc2626", "#fef2f2", "#fca5a5", "Strongly Acidic"
    elif ph < 6:
        ph_color, ph_bg, ph_border, ph_text = "#ea580c", "#fff7ed", "#fdba74", "Mildly Acidic"
    elif ph < 7:
        ph_color, ph_bg, ph_border, ph_text = "#16a34a", "#f0fdf4", "#86efac", "Neutral — Ideal"
    elif ph < 8:
        ph_color, ph_bg, ph_border, ph_text = "#0284c7", "#f0f9ff", "#7dd3fc", "Mildly Alkaline"
    else:
        ph_color, ph_bg, ph_border, ph_text = "#7c3aed", "#faf5ff", "#c4b5fd", "Strongly Alkaline"
    st.markdown(f"""
    <div class="ph-badge" style="background:{ph_bg};color:{ph_color};border-color:{ph_border};">
        pH {ph:.1f} — {ph_text}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# RIGHT PANEL — live summary
with right_col:
    st.markdown(f"""
    <div class="summary-panel">
        <div class="sp-title">📡 &nbsp; Live Field Data</div>
        <div class="sp-row">
            <span class="sp-label">🌿 Nitrogen</span>
            <span><span class="sp-value">{N}</span><span class="sp-unit">mg/kg</span></span>
        </div>
        <div class="sp-row">
            <span class="sp-label">🌸 Phosphorus</span>
            <span><span class="sp-value">{P}</span><span class="sp-unit">mg/kg</span></span>
        </div>
        <div class="sp-row">
            <span class="sp-label">💎 Potassium</span>
            <span><span class="sp-value">{K}</span><span class="sp-unit">mg/kg</span></span>
        </div>
        <div class="sp-row">
            <span class="sp-label">🌡️ Temperature</span>
            <span><span class="sp-value">{temperature}</span><span class="sp-unit">°C</span></span>
        </div>
        <div class="sp-row">
            <span class="sp-label">💧 Humidity</span>
            <span><span class="sp-value">{humidity}</span><span class="sp-unit">%</span></span>
        </div>
        <div class="sp-row">
            <span class="sp-label">⚗️ pH Level</span>
            <span><span class="sp-value">{ph}</span></span>
        </div>
        <div class="sp-row">
            <span class="sp-label">🌧️ Rainfall</span>
            <span><span class="sp-value">{rainfall}</span><span class="sp-unit">mm</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── ANALYSE BUTTON ─────────────────────────────────────────────────────────────
st.markdown("<div style='padding: 0 3rem;'>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
_, btn_mid, _ = st.columns([1, 2, 1])
with btn_mid:
    predict_btn = st.button("🌾 Analyse & Recommend Crop", use_container_width=True, type="primary")
st.markdown("</div>", unsafe_allow_html=True)


# ── RESULTS ────────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("🌱  Running AI inference — analysing 7 parameters…"):
        try:
            result = predict_crop(N=N, P=P, K=K,
                                  temperature=temperature,
                                  humidity=humidity,
                                  ph=ph,
                                  rainfall=rainfall)
        except FileNotFoundError as e:
            st.markdown(f'<div class="err-banner">❌ {e}</div>', unsafe_allow_html=True)
            st.stop()

    st.markdown("<div style='padding: 0 3rem;'>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Result hero banner
    name = result['recommended_crop'].upper()
    # Make the last letter the accent colour
    st.markdown(f"""
    <div class="result-banner">
        <div class="rb-eyebrow">✦ &nbsp; AI Recommended Crop &nbsp; ✦</div>
        <div class="rb-cropname">{name[:-1]}<span class="accent">{name[-1]}</span></div>
        <div class="confidence-pill">🎯 &nbsp; {result['confidence']}% Confidence</div>
    </div>
    """, unsafe_allow_html=True)

    r_left, r_right = st.columns(2, gap="large")

    with r_left:
        st.markdown("""
        <div style="background:#fff;border:1.5px solid #d1fae5;border-radius:20px;padding:1.6rem 1.8rem;box-shadow:0 4px 16px rgba(6,78,59,0.12);">
            <div style="font-family:'Manrope',sans-serif;font-size:0.72rem;font-weight:800;color:#065f46;letter-spacing:2px;text-transform:uppercase;margin-bottom:1.2rem;padding-bottom:0.7rem;border-bottom:1.5px solid #d1fae5;">
                🏆 &nbsp; Top 3 Predictions
            </div>
        """, unsafe_allow_html=True)
        medals = ["🥇", "🥈", "🥉"]
        for i, (crop, conf) in enumerate(result["top_3_crops"]):
            bar = int(float(conf))
            st.markdown(f"""
            <div class="rank-card">
                <div class="rc-medal">{medals[i]}</div>
                <div class="rc-name">{crop.capitalize()}</div>
                <div class="rc-bar-bg">
                    <div class="rc-bar-fill" style="width:{bar}%"></div>
                </div>
                <div class="rc-pct">{conf}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r_right:
        st.markdown("""
        <div style="background:#fff;border:1.5px solid #d1fae5;border-radius:20px;padding:1.6rem 1.8rem;box-shadow:0 4px 16px rgba(6,78,59,0.12);">
            <div style="font-family:'Manrope',sans-serif;font-size:0.72rem;font-weight:800;color:#065f46;letter-spacing:2px;text-transform:uppercase;margin-bottom:1.2rem;padding-bottom:0.7rem;border-bottom:1.5px solid #d1fae5;">
                💰 &nbsp; Profitability Analysis
            </div>
        """, unsafe_allow_html=True)
        st.caption("Estimated revenue per hectare · market price × average yield")

        best = result["profitability"][0]
        st.markdown(f"""
        <div class="profit-card">
            <div class="pc-eyebrow">💡 Most Profitable Option</div>
            <div class="pc-crop">{best['crop'].capitalize()}</div>
            <div class="pc-amount">₹{best['estimated_revenue']:,.0f}</div>
            <div class="pc-detail">
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

    st.markdown("</div>", unsafe_allow_html=True)


# ── CTA SECTION ────────────────────────────────────────────────────────────────
st.markdown('<div style="padding: 0 3rem;">', unsafe_allow_html=True)
st.markdown("""
<div class="cta-section">
    <div class="cta-title">Ready to Grow Smarter? 🌿</div>
    <div class="cta-sub">
        Join 40,000+ farmers using AgriSense AI to make data-driven crop decisions
        and maximise their yield every season.
    </div>
    <div class="cta-btn">🌾 &nbsp; Join the Green Revolution</div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
    <div class="footer-brand">Agri<em>Sense</em> AI</div>
    <div class="footer-links">
        <span>About</span>
        <span>Solutions</span>
        <span>Insights</span>
        <span>Privacy</span>
    </div>
    <div class="footer-copy">Random Forest · Streamlit · v5.0 &nbsp; © 2025 AgriSense</div>
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
<div style="text-align:center;color:rgba(110,231,183,0.3);font-size:0.68rem;
            font-family:'Manrope',sans-serif;padding-top:0.5rem;letter-spacing:1.5px;text-transform:uppercase;">
    AgriSense AI · v5.0
</div>
""", unsafe_allow_html=True)
