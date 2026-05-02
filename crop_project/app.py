"""
app.py
======
Streamlit — AI-Based Smart Agriculture Crop Recommendation System.
UI v4: Premium nature-inspired agriculture theme — organic greens, earthy beige,
        warm amber accents, soft shadows, editorial layout.
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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --green-deep:    #064e3b;
    --green-mid:     #065f46;
    --green-dark:    #14532d;
    --green-light:   #bbf7d0;
    --green-pale:    #dcfce7;
    --green-muted:   #6ee7b7;
    --beige:         #f5deb3;
    --beige-light:   #fdf6ec;
    --beige-mid:     #e6ccb2;
    --beige-warm:    #f9f0e1;
    --amber:         #f59e0b;
    --amber-dark:    #d97706;
    --amber-light:   #fde68a;
    --text-dark:     #1a2e1f;
    --text-mid:      #374151;
    --text-muted:    #6b7280;
    --white:         #ffffff;
    --card-shadow:   0 4px 24px rgba(6,78,59,0.10), 0 1px 4px rgba(6,78,59,0.07);
    --card-hover:    0 12px 40px rgba(6,78,59,0.16), 0 2px 8px rgba(6,78,59,0.10);
    --radius-sm:     12px;
    --radius-md:     16px;
    --radius-lg:     20px;
    --radius-xl:     28px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    color: var(--text-dark);
}

/* ═══════════════════════════════════════════════
   BACKGROUND
═══════════════════════════════════════════════ */
.stApp {
    background:
        radial-gradient(ellipse at 0% 0%, rgba(187,247,208,0.35) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%, rgba(245,222,179,0.25) 0%, transparent 50%),
        linear-gradient(160deg, #f0fdf4 0%, #fdf6ec 50%, #f0fdf4 100%);
    min-height: 100vh;
}

/* ═══════════════════════════════════════════════
   HIDE STREAMLIT CHROME
═══════════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1200px !important; }

/* ═══════════════════════════════════════════════
   NAVBAR
═══════════════════════════════════════════════ */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 2.5rem;
    background: rgba(255,255,255,0.88);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1.5px solid rgba(6,78,59,0.10);
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 20px rgba(6,78,59,0.07);
    margin-bottom: 0;
}
.nav-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    font-weight: 900;
    color: var(--green-deep);
    letter-spacing: -0.5px;
}
.nav-logo .leaf { font-size: 1.5rem; }
.nav-logo .brand-dot { color: var(--amber); }
.nav-links {
    display: flex;
    align-items: center;
    gap: 2.2rem;
    list-style: none;
    margin: 0;
    padding: 0;
}
.nav-links a {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-mid);
    text-decoration: none;
    letter-spacing: 0.2px;
    transition: color 0.2s;
}
.nav-links a:hover { color: var(--green-deep); }
.nav-cta {
    background: var(--amber);
    color: var(--green-deep) !important;
    padding: 0.5rem 1.4rem;
    border-radius: 99px;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.3px !important;
    transition: background 0.25s, transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 2px 12px rgba(245,158,11,0.30);
}
.nav-cta:hover {
    background: var(--amber-dark) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 18px rgba(245,158,11,0.40) !important;
}

/* ═══════════════════════════════════════════════
   HERO SECTION
═══════════════════════════════════════════════ */
.hero-wrap {
    background:
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23064e3b' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E"),
        linear-gradient(135deg, #f0fdf4 0%, #fdf6ec 60%, #f0fdf4 100%);
    padding: 5rem 3rem 4rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '🌿';
    position: absolute;
    font-size: 18rem;
    right: -2rem;
    top: 50%;
    transform: translateY(-50%) rotate(-15deg);
    opacity: 0.05;
    pointer-events: none;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--green-pale);
    color: var(--green-deep);
    border: 1.5px solid var(--green-muted);
    border-radius: 99px;
    padding: 0.3rem 1rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    font-weight: 900;
    color: var(--green-deep);
    line-height: 1.1;
    letter-spacing: -2px;
    margin: 0 0 1.2rem;
    max-width: 640px;
}
.hero h1 .accent { color: var(--amber); }
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-muted);
    line-height: 1.7;
    max-width: 500px;
    margin-bottom: 2rem;
    font-weight: 400;
}
.hero-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--green-deep);
    color: #ffffff;
    padding: 0.85rem 2.2rem;
    border-radius: 99px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.2px;
    text-decoration: none;
    box-shadow: 0 4px 20px rgba(6,78,59,0.30);
    transition: all 0.3s ease;
}
.hero-btn:hover {
    background: var(--green-mid);
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(6,78,59,0.38);
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    margin-top: 2.5rem;
    padding-top: 2rem;
    border-top: 1.5px solid rgba(6,78,59,0.10);
}
.stat-item {}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--green-deep);
    line-height: 1;
}
.stat-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    font-weight: 500;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ═══════════════════════════════════════════════
   SECTION WRAPPERS
═══════════════════════════════════════════════ */
.section-pad {
    padding: 3rem 3rem 2rem;
}
.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--amber-dark);
    margin-bottom: 0.4rem;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--green-deep);
    letter-spacing: -0.8px;
    margin: 0 0 0.6rem;
}
.section-sub {
    font-size: 0.95rem;
    color: var(--text-muted);
    line-height: 1.65;
    max-width: 540px;
    margin-bottom: 2rem;
}

/* ═══════════════════════════════════════════════
   FEATURE CARDS
═══════════════════════════════════════════════ */
.feat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.2rem;
    margin-bottom: 2.5rem;
}
.feat-card {
    background: var(--white);
    border: 1.5px solid rgba(6,78,59,0.09);
    border-radius: var(--radius-lg);
    padding: 1.6rem 1.5rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    box-shadow: var(--card-shadow);
    position: relative;
    overflow: hidden;
}
.feat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--green-deep), var(--amber));
    opacity: 0;
    transition: opacity 0.3s ease;
}
.feat-card:hover {
    transform: translateY(-6px);
    box-shadow: var(--card-hover);
    border-color: rgba(6,78,59,0.18);
}
.feat-card:hover::before { opacity: 1; }
.feat-icon {
    font-size: 2rem;
    margin-bottom: 1rem;
    display: block;
    width: 52px;
    height: 52px;
    background: var(--green-pale);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.feat-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--green-deep);
    margin-bottom: 0.5rem;
}
.feat-desc {
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.6;
    font-weight: 400;
}

/* ═══════════════════════════════════════════════
   INSIGHT SECTION
═══════════════════════════════════════════════ */
.insight-wrap {
    background: linear-gradient(135deg, var(--green-deep) 0%, var(--green-dark) 100%);
    border-radius: var(--radius-xl);
    padding: 3rem 3.5rem;
    margin: 0 3rem 2.5rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3rem;
    align-items: center;
    position: relative;
    overflow: hidden;
}
.insight-wrap::after {
    content: '🌾';
    position: absolute;
    font-size: 14rem;
    right: -1rem;
    bottom: -2rem;
    opacity: 0.06;
    pointer-events: none;
}
.insight-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--green-muted);
    margin-bottom: 0.6rem;
}
.insight-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 900;
    color: var(--white);
    line-height: 1.15;
    letter-spacing: -0.8px;
    margin-bottom: 1rem;
}
.insight-text {
    font-size: 0.92rem;
    color: rgba(255,255,255,0.72);
    line-height: 1.75;
    font-weight: 400;
}
.insight-stats {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
}
.insight-stat {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: background 0.25s;
}
.insight-stat:hover { background: rgba(255,255,255,0.13); }
.insight-stat-icon {
    font-size: 1.6rem;
    width: 48px;
    height: 48px;
    background: rgba(255,255,255,0.10);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.insight-stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 900;
    color: var(--amber-light);
    line-height: 1;
}
.insight-stat-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.6);
    margin-top: 0.15rem;
    font-weight: 500;
}

/* ═══════════════════════════════════════════════
   INPUT SECTION TITLE
═══════════════════════════════════════════════ */
.input-section-wrap {
    padding: 2rem 3rem 1rem;
}
.input-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2rem;
}
.input-divider-line {
    flex: 1;
    height: 1.5px;
    background: linear-gradient(90deg, rgba(6,78,59,0.12), transparent);
}

/* ═══════════════════════════════════════════════
   FORM CARDS
═══════════════════════════════════════════════ */
.form-card {
    background: var(--white);
    border: 1.5px solid rgba(6,78,59,0.09);
    border-radius: var(--radius-lg);
    padding: 1.8rem 2rem 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--card-shadow);
    transition: box-shadow 0.3s ease;
}
.form-card:hover { box-shadow: var(--card-hover); }
.form-card-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--green-deep);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1.5px solid var(--green-pale);
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

/* ═══════════════════════════════════════════════
   INPUT OVERRIDES
═══════════════════════════════════════════════ */
label, .stSlider label, .stNumberInput label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] {
    color: var(--text-mid) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1px !important;
}
input[type="number"] {
    background: var(--beige-warm) !important;
    border: 1.5px solid rgba(6,78,59,0.18) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--green-deep) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}
input[type="number"]:focus {
    border-color: var(--green-deep) !important;
    box-shadow: 0 0 0 3px rgba(6,78,59,0.10) !important;
}
div[data-baseweb="slider"] > div {
    background: var(--green-light) !important;
}
div[data-baseweb="slider"] [role="slider"] {
    background: var(--green-deep) !important;
    border-color: var(--green-deep) !important;
    box-shadow: 0 0 0 3px rgba(6,78,59,0.15) !important;
    width: 20px !important;
    height: 20px !important;
}

/* ═══════════════════════════════════════════════
   LIVE PARAMS CARD
═══════════════════════════════════════════════ */
.param-card {
    background: var(--green-deep);
    border-radius: var(--radius-lg);
    padding: 1.8rem 1.6rem;
    box-shadow: 0 8px 40px rgba(6,78,59,0.25);
    position: sticky;
    top: 90px;
}
.param-card-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--green-muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.10);
}
.param-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    transition: background 0.15s;
}
.param-row:last-child { border-bottom: none; }
.param-row:hover { background: rgba(255,255,255,0.04); border-radius: 8px; padding-left: 0.4rem; padding-right: 0.4rem; }
.param-key {
    font-size: 0.83rem;
    font-weight: 500;
    color: rgba(255,255,255,0.65);
}
.param-val {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--amber-light);
}
.param-unit {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.35);
    margin-left: 0.2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 400;
}

/* ═══════════════════════════════════════════════
   pH BADGE
═══════════════════════════════════════════════ */
.ph-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 1rem;
    border-radius: 99px;
    font-size: 0.82rem;
    font-weight: 700;
    margin-top: 0.6rem;
    letter-spacing: 0.2px;
    transition: all 0.3s ease;
}

/* ═══════════════════════════════════════════════
   ANALYSE BUTTON
═══════════════════════════════════════════════ */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--amber) 0%, #e88b00 100%) !important;
    color: var(--green-deep) !important;
    border: none !important;
    border-radius: 99px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 800 !important;
    padding: 0.9rem 2.8rem !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 24px rgba(245,158,11,0.35), 0 2px 0 rgba(0,0,0,0.08) !important;
    transition: all 0.28s ease !important;
    width: 100% !important;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #e88b00 0%, var(--amber-dark) 100%) !important;
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 10px 35px rgba(245,158,11,0.45), 0 2px 0 rgba(0,0,0,0.10) !important;
}

/* ═══════════════════════════════════════════════
   CTA SECTION
═══════════════════════════════════════════════ */
.cta-wrap {
    background: linear-gradient(135deg, var(--beige-warm) 0%, var(--beige) 100%);
    border-top: 1.5px solid rgba(6,78,59,0.08);
    border-bottom: 1.5px solid rgba(6,78,59,0.08);
    padding: 3rem;
    text-align: center;
    margin: 1rem 0;
}
.cta-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--green-deep);
    letter-spacing: -0.8px;
    margin-bottom: 0.6rem;
}
.cta-sub {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-bottom: 1.8rem;
    line-height: 1.65;
}
.cta-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--amber);
    color: var(--green-deep);
    padding: 0.85rem 2.4rem;
    border-radius: 99px;
    font-weight: 800;
    font-size: 0.92rem;
    letter-spacing: 0.3px;
    text-decoration: none;
    box-shadow: 0 4px 20px rgba(245,158,11,0.30);
    transition: all 0.3s ease;
    text-transform: uppercase;
}
.cta-btn:hover {
    background: var(--amber-dark);
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(245,158,11,0.42);
}

/* ═══════════════════════════════════════════════
   RESULT SECTION
═══════════════════════════════════════════════ */
.result-hero {
    background: linear-gradient(135deg, var(--green-deep) 0%, var(--green-dark) 100%);
    border-radius: var(--radius-xl);
    padding: 2.8rem 2.5rem 2.2rem;
    text-align: center;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 50px rgba(6,78,59,0.30), 0 2px 0 rgba(6,78,59,0.20);
}
.result-hero::before {
    content: '';
    position: absolute;
    top: -80px; left: 50%;
    transform: translateX(-50%);
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(245,158,11,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.result-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--green-muted);
    margin-bottom: 0.5rem;
}
.crop-name {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    font-weight: 900;
    color: var(--white);
    letter-spacing: -2px;
    line-height: 1;
    margin: 0.3rem 0 0.9rem;
    text-shadow: 0 2px 20px rgba(0,0,0,0.20);
}
.conf-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(245,158,11,0.15);
    border: 1.5px solid rgba(245,158,11,0.45);
    border-radius: 99px;
    padding: 0.4rem 1.4rem;
    font-size: 1rem;
    font-weight: 700;
    color: var(--amber-light);
    letter-spacing: 0.3px;
}

/* ═══════════════════════════════════════════════
   RANK CARDS
═══════════════════════════════════════════════ */
.res-card {
    background: var(--white);
    border: 1.5px solid rgba(6,78,59,0.09);
    border-radius: var(--radius-lg);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--card-shadow);
    transition: box-shadow 0.3s ease;
}
.res-card:hover { box-shadow: var(--card-hover); }
.res-card-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--green-deep);
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1.5px solid var(--green-pale);
}
.rank-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--green-pale);
    transition: padding 0.2s ease;
}
.rank-row:last-child { border-bottom: none; }
.rank-row:hover { padding-left: 0.4rem; }
.rank-medal { font-size: 1.5rem; min-width: 1.8rem; }
.rank-crop { font-weight: 700; font-size: 1rem; color: var(--text-dark); flex: 1; }
.rank-bar-bg {
    flex: 2;
    height: 8px;
    background: var(--green-pale);
    border-radius: 99px;
    overflow: hidden;
}
.rank-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--green-mid), var(--amber));
    border-radius: 99px;
}
.rank-pct {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--green-deep);
    min-width: 3rem;
    text-align: right;
    font-family: 'Playfair Display', serif;
}

/* ═══════════════════════════════════════════════
   PROFIT BOX
═══════════════════════════════════════════════ */
.profit-hero {
    background: linear-gradient(135deg, var(--beige-warm), var(--beige));
    border: 1.5px solid var(--beige-mid);
    border-radius: var(--radius-md);
    padding: 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(245,158,11,0.12);
}
.profit-hero::before {
    content: '₹';
    position: absolute;
    right: 1rem;
    top: 50%;
    transform: translateY(-50%);
    font-family: 'Playfair Display', serif;
    font-size: 6rem;
    font-weight: 900;
    color: rgba(245,158,11,0.08);
    pointer-events: none;
}
.profit-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--amber-dark);
    margin-bottom: 0.3rem;
}
.profit-crop {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-dark);
    margin-bottom: 0.3rem;
}
.profit-amount {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 900;
    color: var(--green-deep);
    letter-spacing: -1.5px;
    line-height: 1;
}
.profit-detail {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 0.45rem;
}

/* ═══════════════════════════════════════════════
   ERROR BANNER
═══════════════════════════════════════════════ */
.err-card {
    background: #fff5f5;
    border: 1.5px solid rgba(220,38,38,0.25);
    border-radius: var(--radius-md);
    padding: 1rem 1.4rem;
    color: #b91c1c;
    font-weight: 600;
    font-size: 0.95rem;
}

/* ═══════════════════════════════════════════════
   FOOTER
═══════════════════════════════════════════════ */
.footer {
    background: var(--green-deep);
    color: rgba(255,255,255,0.55);
    text-align: center;
    padding: 2rem;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    margin-top: 3rem;
}
.footer strong { color: rgba(255,255,255,0.85); }
.footer-links {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 0.8rem;
    font-size: 0.78rem;
}
.footer-links a {
    color: rgba(255,255,255,0.4);
    text-decoration: none;
    transition: color 0.2s;
}
.footer-links a:hover { color: var(--amber-light); }

/* ═══════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--beige-warm) !important;
    border-right: 1.5px solid rgba(6,78,59,0.10) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-mid) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--green-deep) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1rem !important;
    letter-spacing: -0.3px !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(6,78,59,0.10) !important; }
[data-testid="stSidebar"] table td { color: var(--text-mid) !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] table th { color: var(--green-deep) !important; font-weight: 700 !important; }

/* ═══════════════════════════════════════════════
   DATAFRAME & MISC
═══════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-sm) !important;
    overflow: hidden !important;
    border: 1.5px solid rgba(6,78,59,0.10) !important;
}
.stCaption, small { color: var(--text-muted) !important; font-size: 0.82rem !important; }
[data-testid="stSpinner"] p { color: var(--green-deep) !important; }

/* ═══════════════════════════════════════════════
   RESPONSIVE
═══════════════════════════════════════════════ */
@media (max-width: 768px) {
    .hero h1 { font-size: 2.3rem; }
    .crop-name { font-size: 2.4rem; }
    .feat-grid { grid-template-columns: 1fr 1fr; }
    .insight-wrap { grid-template-columns: 1fr; padding: 2rem; }
    .hero-wrap { padding: 3rem 1.5rem; }
    .section-pad, .input-section-wrap { padding: 1.5rem 1rem; }
}
</style>
""", unsafe_allow_html=True)


# ── NAVBAR ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-logo">
        <span class="leaf">🌱</span>
        Agri<span class="brand-dot">Sense</span>
    </div>
    <ul class="nav-links">
        <li><a href="#">Home</a></li>
        <li><a href="#">About</a></li>
        <li><a href="#">Solutions</a></li>
        <li><a href="#">Insights</a></li>
        <li><a href="#" class="nav-cta">Get Started</a></li>
    </ul>
</div>
""", unsafe_allow_html=True)


# ── HERO ───────────────────────────────────────────────────────────────────────
hero_left, hero_right = st.columns([3, 2], gap="large")
with hero_left:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-tag">🤖 AI-Powered · Random Forest · ~99% Accuracy</div>
        <h1 class="hero">Empowering <span class="accent">Smart</span><br>Agriculture</h1>
        <p class="hero-sub">Enter your soil nutrients, weather conditions, and pH — our AI
        engine recommends the optimal crop and projects your profitability per hectare.</p>
        <a class="hero-btn" href="#">⚡ Analyse Your Soil →</a>
        <div class="hero-stats">
            <div class="stat-item">
                <div class="stat-num">40K+</div>
                <div class="stat-label">Farmers Helped</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">22</div>
                <div class="stat-label">Crop Varieties</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">99%</div>
                <div class="stat-label">Model Accuracy</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    st.markdown("""
    <div style="padding: 3rem 1.5rem 2rem; display:flex; align-items:center; justify-content:center; height:100%;">
        <div style="background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-radius: 24px; padding: 2.5rem;
                    border: 1.5px solid rgba(6,78,59,0.12); box-shadow: 0 8px 40px rgba(6,78,59,0.10);
                    text-align:center;">
            <div style="font-size: 6rem; margin-bottom: 1rem;">🌾</div>
            <div style="font-family:'Playfair Display',serif; font-size:1.3rem; font-weight:900;
                        color:#064e3b; margin-bottom:0.5rem;">AI Crop Intelligence</div>
            <div style="font-size:0.85rem; color:#6b7280; line-height:1.6;">
                7 soil & weather parameters analysed<br>instantly by our ML model.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── FEATURES ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-pad">
    <div class="section-label">What We Offer</div>
    <div class="section-title">Precision Farming, Simplified</div>
    <div class="section-sub">Our AI platform gives every farmer access to data-driven recommendations
    that were once only available to large agribusinesses.</div>
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon">🧬</div>
            <div class="feat-title">Soil Analysis</div>
            <div class="feat-desc">Deep NPK profiling to understand your soil's nutritional
            composition and what it needs most.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🌦️</div>
            <div class="feat-title">Weather Matching</div>
            <div class="feat-desc">We factor in local temperature, humidity, and annual rainfall
            to recommend climate-compatible crops.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">💰</div>
            <div class="feat-title">Profit Projection</div>
            <div class="feat-desc">Estimate revenue per hectare using live market prices and
            crop yield data — before you sow a single seed.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🤖</div>
            <div class="feat-title">AI Recommendations</div>
            <div class="feat-desc">Random Forest classifier trained on 2,200 samples gives you
            top-3 crop rankings with confidence scores.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── INSIGHTS SECTION ───────────────────────────────────────────────────────────
st.markdown("""
<div class="insight-wrap">
    <div>
        <div class="insight-label">Why AgriSense</div>
        <div class="insight-title">Data-Driven Farming for the Modern Age</div>
        <p class="insight-text">
            Traditional farming relies on intuition passed down through generations —
            but soil, climate, and markets have changed dramatically. AgriSense AI bridges
            that gap, delivering actionable crop intelligence in seconds. No expertise required.
            Just enter your field data and let the model do the rest.
        </p>
    </div>
    <div class="insight-stats">
        <div class="insight-stat">
            <div class="insight-stat-icon">📈</div>
            <div>
                <div class="insight-stat-num">3.2×</div>
                <div class="insight-stat-label">Average yield improvement reported</div>
            </div>
        </div>
        <div class="insight-stat">
            <div class="insight-stat-icon">⏱️</div>
            <div>
                <div class="insight-stat-num">&lt; 2s</div>
                <div class="insight-stat-label">AI inference time per recommendation</div>
            </div>
        </div>
        <div class="insight-stat">
            <div class="insight-stat-icon">🌱</div>
            <div>
                <div class="insight-stat-num">22 crops</div>
                <div class="insight-stat-label">Supported across all major categories</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── MODEL AUTO-TRAIN ───────────────────────────────────────────────────────────
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


# ── INPUT SECTION ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="input-section-wrap">
    <div class="input-divider">
        <div class="section-label" style="margin:0;">🔬 Analyse Your Field</div>
        <div class="input-divider-line"></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='padding: 0 3rem;'>", unsafe_allow_html=True)
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:

    # SOIL NUTRIENTS
    st.markdown('<div class="form-card"><div class="form-card-title">⚗️ Soil Nutrients</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="form-card"><div class="form-card-title">🌤️ Weather Conditions</div>', unsafe_allow_html=True)
    w1, w2 = st.columns(2)
    with w1:
        temperature = st.slider("Temperature (°C)", 5.0, 50.0, 21.0, 0.5)
    with w2:
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 82.0, 0.5)
    rainfall = st.slider("Annual Rainfall (mm)", 20.0, 300.0, 203.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # pH
    st.markdown('<div class="form-card"><div class="form-card-title">🧫 Soil pH</div>', unsafe_allow_html=True)
    ph = st.slider("pH Value", 3.0, 10.0, 6.5, 0.1)
    if ph < 5:
        ph_color, ph_bg, ph_text = "#dc2626", "rgba(220,38,38,0.08)", "Strongly Acidic"
    elif ph < 6:
        ph_color, ph_bg, ph_text = "#d97706", "rgba(217,119,6,0.10)", "Mildly Acidic"
    elif ph < 7:
        ph_color, ph_bg, ph_text = "#065f46", "rgba(6,95,70,0.08)", "Neutral"
    elif ph < 8:
        ph_color, ph_bg, ph_text = "#1d4ed8", "rgba(29,78,216,0.08)", "Mildly Alkaline"
    else:
        ph_color, ph_bg, ph_text = "#7c3aed", "rgba(124,58,237,0.08)", "Strongly Alkaline"
    st.markdown(f"""
    <div class="ph-badge" style="background:{ph_bg};color:{ph_color};border:1.5px solid {ph_color}33;">
        ⚗️ &nbsp; pH {ph:.1f} — {ph_text}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT PANEL — live summary
with right_col:
    st.markdown('<div class="param-card"><div class="param-card-title">📡 Live Parameters</div>', unsafe_allow_html=True)
    params = [
        ("🌿 Nitrogen",      f"{N}",           "mg/kg"),
        ("🌸 Phosphorus",    f"{P}",           "mg/kg"),
        ("💎 Potassium",     f"{K}",           "mg/kg"),
        ("🌡️ Temperature",  f"{temperature}", "°C"),
        ("💧 Humidity",      f"{humidity}",    "%"),
        ("⚗️ pH",           f"{ph}",          ""),
        ("🌧️ Rainfall",     f"{rainfall}",    "mm"),
    ]
    for label, val, unit in params:
        st.markdown(f"""
        <div class="param-row">
            <span class="param-key">{label}</span>
            <span>
                <span class="param-val">{val}</span>
                <span class="param-unit">{unit}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ── PREDICT BUTTON ─────────────────────────────────────────────────────────────
st.markdown("<div style='padding: 1.5rem 3rem 0;'>", unsafe_allow_html=True)
_, btn_mid, _ = st.columns([1, 2, 1])
with btn_mid:
    predict_btn = st.button("⚡ Analyse & Recommend Crop", use_container_width=True, type="primary")
st.markdown("</div>", unsafe_allow_html=True)


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
            st.markdown(f'<div class="err-card">❌ {e}</div>', unsafe_allow_html=True)
            st.stop()

    st.markdown("<div style='padding: 2rem 3rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-hero">
        <div class="result-eyebrow">✅ &nbsp; Recommended Crop</div>
        <div class="crop-name">{result['recommended_crop'].upper()}</div>
        <div class="conf-pill">🎯 &nbsp; {result['confidence']}% Confidence</div>
    </div>
    """, unsafe_allow_html=True)

    r_left, r_right = st.columns(2, gap="large")

    with r_left:
        st.markdown('<div class="res-card"><div class="res-card-title">🏆 Top 3 Predictions</div>', unsafe_allow_html=True)
        medals = ["🥇", "🥈", "🥉"]
        for i, (crop, conf) in enumerate(result["top_3_crops"]):
            bar = int(float(conf))
            st.markdown(f"""
            <div class="rank-row">
                <div class="rank-medal">{medals[i]}</div>
                <div class="rank-crop">{crop.capitalize()}</div>
                <div class="rank-bar-bg">
                    <div class="rank-bar-fill" style="width:{bar}%"></div>
                </div>
                <div class="rank-pct">{conf}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r_right:
        st.markdown('<div class="res-card"><div class="res-card-title">💰 Profitability Analysis</div>', unsafe_allow_html=True)
        st.caption("Estimated revenue per hectare · market price × average yield")

        best = result["profitability"][0]
        st.markdown(f"""
        <div class="profit-hero">
            <div class="profit-eyebrow">💡 Most Profitable Option</div>
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

    st.markdown("</div>", unsafe_allow_html=True)


# ── CTA SECTION ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cta-wrap">
    <div class="cta-title">Ready to Grow Smarter?</div>
    <p class="cta-sub">Join thousands of farmers who are already using AI to make better<br>
    crop decisions, reduce waste, and increase yield.</p>
    <a class="cta-btn" href="#">🌱 Start for Free Today</a>
</div>
""", unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>AgriSense AI</strong> &nbsp;·&nbsp; Random Forest Classifier &nbsp;·&nbsp; Built with Streamlit &nbsp;·&nbsp; v4.0
    <div class="footer-links">
        <a href="#">Privacy</a>
        <a href="#">Terms</a>
        <a href="#">Contact</a>
        <a href="#">Docs</a>
    </div>
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
<div style="text-align:center;color:#9ca3af;font-size:0.72rem;padding-top:0.5rem;">
    AgriSense AI · v4.0
</div>
""", unsafe_allow_html=True)
