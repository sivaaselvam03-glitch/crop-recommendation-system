"""
app.py
======
Streamlit — AI-Based Smart Agriculture Crop Recommendation System.
UI v4: Premium Glassmorphism Dark Blue theme — high-end SaaS aesthetic.
        Deep navy/midnight backgrounds, frosted glass cards, electric blue
        and violet glow accents, fluid typography, cinematic depth.
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
}

/* ═══════════════════════════════════════════════
   CSS VARIABLES
═══════════════════════════════════════════════ */
:root {
    --blue-primary:   #3b82f6;
    --blue-bright:    #60a5fa;
    --blue-glow:      rgba(59,130,246,0.35);
    --violet:         #7c3aed;
    --violet-glow:    rgba(124,58,237,0.25);
    --cyan:           #06b6d4;
    --cyan-glow:      rgba(6,182,212,0.2);
    --glass-bg:       rgba(255,255,255,0.04);
    --glass-border:   rgba(255,255,255,0.08);
    --glass-hover-bg: rgba(255,255,255,0.07);
    --text-primary:   #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted:     #475569;
    --surface-1:      rgba(15,20,40,0.8);
    --surface-2:      rgba(10,14,30,0.9);
    --gold:           #f59e0b;
    --gold-glow:      rgba(245,158,11,0.3);
}

/* ═══════════════════════════════════════════════
   BACKGROUND — deep space midnight
═══════════════════════════════════════════════ */
.stApp {
    background:
        radial-gradient(ellipse at 10% 0%,   rgba(59,130,246,0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 10%,  rgba(124,58,237,0.10) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 90%,  rgba(6,182,212,0.07)  0%, transparent 50%),
        linear-gradient(160deg, #050812 0%, #080d1e 45%, #060a18 100%);
    min-height: 100vh;
}

/* ═══════════════════════════════════════════════
   HIDE STREAMLIT CHROME
═══════════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
}

/* ═══════════════════════════════════════════════
   NAVBAR
═══════════════════════════════════════════════ */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 2rem;
    margin: 0 -2rem 2rem;
    background: rgba(8,13,30,0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(59,130,246,0.12);
    position: sticky;
    top: 0;
    z-index: 100;
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.navbar-logo {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #3b82f6, #7c3aed);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    box-shadow: 0 0 16px rgba(59,130,246,0.4);
}
.navbar-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.5px;
}
.navbar-name span { color: #60a5fa; }
.navbar-badge {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #7c3aed;
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
}
.navbar-links {
    display: flex;
    gap: 2rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: #94a3b8;
}
.navbar-links span { cursor: default; }
.navbar-links .active { color: #60a5fa; }

/* ═══════════════════════════════════════════════
   HERO SECTION
═══════════════════════════════════════════════ */
.hero {
    position: relative;
    padding: 3.5rem 3rem 3rem;
    margin-bottom: 2.5rem;
    overflow: hidden;
    border-radius: 24px;
    background:
        radial-gradient(ellipse at 70% 50%, rgba(59,130,246,0.12) 0%, transparent 60%),
        radial-gradient(ellipse at 20% 80%, rgba(124,58,237,0.10) 0%, transparent 50%),
        rgba(8,13,30,0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(59,130,246,0.15);
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.04) inset,
        0 40px 80px rgba(0,0,0,0.5),
        0 0 60px rgba(59,130,246,0.06);
}
.hero-noise {
    position: absolute;
    inset: 0;
    opacity: 0.025;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
    border-radius: 24px;
    pointer-events: none;
}
.hero-grid {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(59,130,246,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59,130,246,0.05) 1px, transparent 1px);
    background-size: 40px 40px;
    border-radius: 24px;
    pointer-events: none;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(59,130,246,0.1);
    color: #60a5fa;
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 30px;
    padding: 0.3rem 1rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-tag::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #3b82f6;
    box-shadow: 0 0 8px #3b82f6;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3.4rem;
    font-weight: 800;
    color: #f1f5f9;
    margin: 0 0 0.5rem;
    line-height: 1.05;
    letter-spacing: -2px;
}
.hero h1 .grad {
    background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 560px;
    line-height: 1.7;
    margin-bottom: 2rem;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
}
.hero-stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1;
}
.hero-stat-val span { color: #60a5fa; }
.hero-stat-lbl {
    font-size: 0.75rem;
    color: #475569;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.2rem;
}
.hero-orb {
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    box-shadow:
        0 0 80px rgba(59,130,246,0.15),
        0 0 160px rgba(124,58,237,0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 5rem;
    animation: float 6s ease-in-out infinite;
    filter: drop-shadow(0 0 30px rgba(59,130,246,0.3));
}
@keyframes float {
    0%, 100% { transform: translateY(-50%) translateY(0px); }
    50%       { transform: translateY(-50%) translateY(-12px); }
}

/* ═══════════════════════════════════════════════
   GLASS CARDS
═══════════════════════════════════════════════ */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 1.6rem 1.8rem 1.4rem;
    margin-bottom: 1.4rem;
    position: relative;
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.03) inset,
        0 20px 40px rgba(0,0,0,0.35),
        0 0 30px rgba(59,130,246,0.04);
    transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1),
                box-shadow 0.3s ease,
                border-color 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    border-color: rgba(59,130,246,0.2);
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.05) inset,
        0 30px 60px rgba(0,0,0,0.4),
        0 0 40px rgba(59,130,246,0.08);
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.3), transparent);
    border-radius: 20px 20px 0 0;
}
.card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: #60a5fa;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(59,130,246,0.12);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ═══════════════════════════════════════════════
   INPUT LABELS
═══════════════════════════════════════════════ */
label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] {
    color: #cbd5e1 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
}

/* ═══════════════════════════════════════════════
   NUMBER INPUTS
═══════════════════════════════════════════════ */
input[type="number"] {
    background: rgba(15,23,42,0.6) !important;
    border: 1.5px solid rgba(59,130,246,0.2) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
input[type="number"]:focus {
    border-color: rgba(59,130,246,0.6) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1), 0 0 20px rgba(59,130,246,0.08) !important;
}

/* ═══════════════════════════════════════════════
   SLIDERS
═══════════════════════════════════════════════ */
div[data-baseweb="slider"] > div {
    background: rgba(59,130,246,0.18) !important;
}
div[data-baseweb="slider"] [role="slider"] {
    background: linear-gradient(135deg, #3b82f6, #7c3aed) !important;
    border-color: #60a5fa !important;
    box-shadow: 0 0 14px rgba(59,130,246,0.6) !important;
    width: 20px !important;
    height: 20px !important;
}

/* ═══════════════════════════════════════════════
   GLOW BUTTON
═══════════════════════════════════════════════ */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 50%, #6d28d9 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(99,102,241,0.5) !important;
    border-radius: 14px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    padding: 0.9rem 2.5rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.08) inset,
        0 0 25px rgba(59,130,246,0.3),
        0 8px 24px rgba(0,0,0,0.4) !important;
    transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1) !important;
    width: 100% !important;
    position: relative !important;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.12) inset,
        0 0 45px rgba(99,102,241,0.5),
        0 0 80px rgba(59,130,246,0.2),
        0 12px 32px rgba(0,0,0,0.45) !important;
    transform: translateY(-3px) scale(1.01) !important;
    border-color: rgba(139,92,246,0.7) !important;
}

/* ═══════════════════════════════════════════════
   RESULT HERO
═══════════════════════════════════════════════ */
.result-hero {
    background:
        radial-gradient(ellipse at 50% 0%,  rgba(59,130,246,0.18) 0%, transparent 65%),
        radial-gradient(ellipse at 80% 100%, rgba(124,58,237,0.12) 0%, transparent 50%),
        rgba(8,13,30,0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 24px;
    padding: 2.8rem 2rem 2.4rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.04) inset,
        0 40px 80px rgba(0,0,0,0.5),
        0 0 60px rgba(59,130,246,0.08);
}
.result-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #3b82f6 30%, #7c3aed 70%, transparent);
    border-radius: 24px 24px 0 0;
}
.result-label {
    font-family: 'Space Grotesk', monospace;
    font-size: 0.72rem;
    color: #60a5fa;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    font-weight: 600;
}
.crop-name-3d {
    font-family: 'Syne', sans-serif;
    font-size: 3.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f1f5f9 0%, #93c5fd 50%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -2px;
    margin: 0.3rem 0 0.9rem;
    line-height: 1;
}
.conf-pill {
    display: inline-block;
    background: rgba(59,130,246,0.1);
    border: 1.5px solid rgba(59,130,246,0.35);
    border-radius: 30px;
    padding: 0.4rem 1.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #93c5fd;
    letter-spacing: 1px;
    box-shadow: 0 0 20px rgba(59,130,246,0.15);
}

/* ═══════════════════════════════════════════════
   RANK CARDS
═══════════════════════════════════════════════ */
.rank3d {
    background: rgba(15,23,42,0.5);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 14px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.rank3d:hover {
    transform: translateX(4px);
    border-color: rgba(59,130,246,0.25);
    box-shadow: 0 4px 30px rgba(59,130,246,0.08);
}
.rank-medal { font-size: 1.6rem; min-width: 2rem; }
.rank-name  { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1rem; color: #e2e8f0; flex: 1; }
.rank-bar-bg { flex: 2; height: 6px; background: rgba(59,130,246,0.1); border-radius: 99px; overflow: hidden; border: 1px solid rgba(59,130,246,0.15); }
.rank-bar-fill { height: 100%; background: linear-gradient(90deg, #1d4ed8, #7c3aed, #06b6d4); border-radius: 99px; box-shadow: 0 0 10px rgba(59,130,246,0.4); }
.rank-pct { font-family: 'Space Grotesk', monospace; font-size: 0.9rem; font-weight: 700; color: #60a5fa; min-width: 3.5rem; text-align: right; }

/* ═══════════════════════════════════════════════
   PROFIT BOX
═══════════════════════════════════════════════ */
.profit3d {
    background:
        radial-gradient(ellipse at 0% 0%, rgba(245,158,11,0.08) 0%, transparent 60%),
        rgba(20,14,5,0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.03) inset,
        0 20px 50px rgba(0,0,0,0.4),
        0 0 30px rgba(245,158,11,0.05);
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.profit3d:hover {
    border-color: rgba(245,158,11,0.35);
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.04) inset,
        0 25px 60px rgba(0,0,0,0.45),
        0 0 50px rgba(245,158,11,0.1);
}
.profit3d::before {
    content: '';
    display: block;
    width: 100%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(245,158,11,0.4), transparent);
    margin-bottom: 1.2rem;
    margin-top: -0.2rem;
}
.profit3d .p-label  { font-family: 'Space Grotesk', sans-serif; font-size: 0.68rem; color: #fbbf24; letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 700; }
.profit3d .p-crop   { font-family: 'DM Sans', sans-serif; font-size: 1.1rem; font-weight: 600; color: #fef3c7; margin-bottom: 0.3rem; }
.profit3d .p-amount { font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800; background: linear-gradient(135deg, #fbbf24, #f59e0b, #d97706); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -1px; line-height: 1.1; }
.profit3d .p-detail { font-size: 0.82rem; color: #92400e; margin-top: 0.4rem; font-weight: 500; }

/* ═══════════════════════════════════════════════
   METRIC SUMMARY CARDS
═══════════════════════════════════════════════ */
.met3d {
    background: rgba(15,23,42,0.4);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(59,130,246,0.1);
    border-radius: 12px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.met3d:hover {
    transform: translateX(4px);
    border-color: rgba(59,130,246,0.22);
}
.met-label { font-family: 'DM Sans', sans-serif; font-size: 0.82rem; font-weight: 500; color: #64748b; }
.met-value { font-family: 'Space Grotesk', sans-serif; font-size: 0.98rem; font-weight: 700; color: #93c5fd; }
.met-unit  { font-size: 0.7rem; color: #334155; margin-left: 0.2rem; }

/* ═══════════════════════════════════════════════
   pH BADGE
═══════════════════════════════════════════════ */
.ph-badge {
    display: inline-block;
    padding: 0.3rem 1.1rem;
    border-radius: 20px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.5rem;
    letter-spacing: 0.3px;
    backdrop-filter: blur(4px);
}

/* ═══════════════════════════════════════════════
   DIVIDER
═══════════════════════════════════════════════ */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.15), transparent);
    margin: 2rem 0;
}

/* ═══════════════════════════════════════════════
   ERROR BANNER
═══════════════════════════════════════════════ */
.err3d {
    background: rgba(30,5,5,0.7);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    color: #fca5a5;
    font-weight: 500;
    font-size: 0.95rem;
    box-shadow: 0 0 20px rgba(239,68,68,0.06);
}

/* ═══════════════════════════════════════════════
   FOOTER
═══════════════════════════════════════════════ */
.footer3d {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    color: #1e3a5f;
    font-size: 0.75rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    border-top: 1px solid rgba(59,130,246,0.07);
    margin-top: 3rem;
}
.footer3d .foot-brand { color: #1d4ed8; }

/* ═══════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #05080f, #070b18) !important;
    border-right: 1px solid rgba(59,130,246,0.1) !important;
}
[data-testid="stSidebar"] * {
    color: #64748b !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #3b82f6 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(59,130,246,0.1) !important; }
[data-testid="stSidebar"] table td,
[data-testid="stSidebar"] table th { color: #475569 !important; font-size: 0.8rem !important; }
[data-testid="stSidebar"] table th { color: #3b82f6 !important; }

/* ═══════════════════════════════════════════════
   DATAFRAME & CAPTION
═══════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(59,130,246,0.15) !important;
    background: rgba(8,13,30,0.6) !important;
}
.stCaption, small { color: #334155 !important; font-size: 0.82rem !important; }
[data-testid="stSpinner"] p { color: #60a5fa !important; }

/* ═══════════════════════════════════════════════
   RESPONSIVE
═══════════════════════════════════════════════ */
@media (max-width: 768px) {
    .hero h1 { font-size: 2.2rem; }
    .hero-orb { display: none; }
    .crop-name-3d { font-size: 2.2rem; }
    .profit3d .p-amount { font-size: 2rem; }
    .hero-stats { flex-wrap: wrap; gap: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)


# ── NAVBAR ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <div class="navbar-logo">🌱</div>
        <span class="navbar-name">Agri<span>Sense</span></span>
        <span class="navbar-badge">AI Platform</span>
    </div>
    <div class="navbar-links">
        <span class="active">Dashboard</span>
        <span>Crops</span>
        <span>Analytics</span>
        <span>Docs</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── HERO BANNER ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-noise"></div>
    <div class="hero-grid"></div>
    <div class="hero-tag">Random Forest · ~99% Accuracy · 22 Crops</div>
    <h1>Smart Crop <span class="grad">Intelligence</span></h1>
    <p class="hero-sub">
        Input your soil nutrients, weather conditions and pH — our AI engine
        analyses 7 parameters to surface the optimal crop and project profitability per hectare.
    </p>
    <div class="hero-stats">
        <div>
            <div class="hero-stat-val">2,200<span>+</span></div>
            <div class="hero-stat-lbl">Training Samples</div>
        </div>
        <div>
            <div class="hero-stat-val">99<span>%</span></div>
            <div class="hero-stat-lbl">Model Accuracy</div>
        </div>
        <div>
            <div class="hero-stat-val">22</div>
            <div class="hero-stat-lbl">Crop Classes</div>
        </div>
        <div>
            <div class="hero-stat-val">7</div>
            <div class="hero-stat-lbl">Input Features</div>
        </div>
    </div>
    <div class="hero-orb">🌾</div>
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
    st.markdown('<div class="glass-card"><div class="card-title">⚗️ &nbsp;Soil Nutrients</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="glass-card"><div class="card-title">🌤️ &nbsp;Weather Conditions</div>', unsafe_allow_html=True)
    w1, w2 = st.columns(2)
    with w1:
        temperature = st.slider("Temperature (°C)", 5.0, 50.0, 21.0, 0.5)
    with w2:
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 82.0, 0.5)
    rainfall = st.slider("Annual Rainfall (mm)", 20.0, 300.0, 203.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # SECTION 3 — pH
    st.markdown('<div class="glass-card"><div class="card-title">🧫 &nbsp;Soil pH</div>', unsafe_allow_html=True)
    ph = st.slider("pH Value", 3.0, 10.0, 6.5, 0.1)
    if ph < 5:
        ph_color, ph_bg, ph_text = "#f87171", "rgba(239,68,68,0.1)", "Strongly Acidic"
    elif ph < 6:
        ph_color, ph_bg, ph_text = "#fb923c", "rgba(251,146,60,0.1)", "Mildly Acidic"
    elif ph < 7:
        ph_color, ph_bg, ph_text = "#34d399", "rgba(52,211,153,0.1)", "Neutral"
    elif ph < 8:
        ph_color, ph_bg, ph_text = "#60a5fa", "rgba(96,165,250,0.1)", "Mildly Alkaline"
    else:
        ph_color, ph_bg, ph_text = "#c084fc", "rgba(192,132,252,0.1)", "Strongly Alkaline"
    st.markdown(f"""
    <div class="ph-badge" style="background:{ph_bg};color:{ph_color};border:1.5px solid {ph_color}33;">
        pH {ph:.1f} — {ph_text}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# RIGHT PANEL — live summary
with right_col:
    st.markdown('<div class="glass-card"><div class="card-title">📡 &nbsp;Live Parameters</div>', unsafe_allow_html=True)
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
    predict_btn = st.button("⚡ Analyse & Recommend Crop", use_container_width=True, type="primary")


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

    # Result hero
    st.markdown(f"""
    <div class="result-hero">
        <div class="result-label">✦ &nbsp; AI Recommended Crop &nbsp; ✦</div>
        <div class="crop-name-3d">{result['recommended_crop'].upper()}</div>
        <div class="conf-pill">🎯 &nbsp; {result['confidence']}% Confidence</div>
    </div>
    """, unsafe_allow_html=True)

    r_left, r_right = st.columns(2, gap="large")

    with r_left:
        st.markdown('<div class="glass-card"><div class="card-title">🏆 &nbsp;Top 3 Predictions</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="glass-card"><div class="card-title">💰 &nbsp;Profitability Analysis</div>', unsafe_allow_html=True)
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
    <span class="foot-brand">AgriSense AI</span>
    &nbsp;·&nbsp; Random Forest Classifier
    &nbsp;·&nbsp; Built with Streamlit
    &nbsp;·&nbsp; v4.0
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
<div style="text-align:center;color:#1e3a5f;font-size:0.68rem;
            font-family:'Space Grotesk',sans-serif;padding-top:0.5rem;letter-spacing:1.5px;text-transform:uppercase;">
    AgriSense AI · v4.0
</div>
""", unsafe_allow_html=True)
