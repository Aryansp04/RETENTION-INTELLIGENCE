"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
Cinematic Scroll-Triggered Experience Inspired by Maryl Gonzalez (marylgonzalez.com)
Intro Animations, GSAP-Style Scroll Reveals, 3D Perspective Artifacts,
Kinetic Typography, Sparkle Accents & Interactive Case Studio
Author: Aryan (Internship Portfolio Project)
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

# Append src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_processing import load_and_preprocess_data, calculate_portfolio_kpis
from src.analytics import run_statistical_hypothesis_tests, train_behavioral_models, compute_segment_exposure
from src.simulator import simulate_retention_interventions

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Retention Intelligence · Design, Technology & a Little Magic",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Full Aesthetic CSS + Intro Animations + Scroll Reveal Classes
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ========== CORE CANVAS ========== */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F8F6F2;
    scroll-behavior: smooth;
}

.stApp {
    background-color: #171323;
    background-image:
        radial-gradient(circle at 10% 10%, rgba(102, 88, 255, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 90% 20%, rgba(215, 198, 255, 0.12) 0%, transparent 45%),
        radial-gradient(circle at 50% 90%, rgba(179, 230, 219, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(244, 63, 94, 0.08) 0%, transparent 40%);
    background-attachment: fixed;
}

/* ========== INTRO ANIMATION KEYFRAMES ========== */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(50px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.92); }
    to { opacity: 1; transform: scale(1); }
}

@keyframes blurReveal {
    from { opacity: 0; filter: blur(12px); transform: translateY(20px); }
    to { opacity: 1; filter: blur(0); transform: translateY(0); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-60px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(60px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes floatGently {
    0%, 100% { transform: rotateX(8deg) rotateY(-4deg) translateY(0); }
    50% { transform: rotateX(6deg) rotateY(-2deg) translateY(-12px); }
}

@keyframes spinSparkle {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 30px rgba(102, 88, 255, 0.4); }
    50% { box-shadow: 0 0 60px rgba(215, 198, 255, 0.6); }
}

@keyframes ribbonDraw {
    from { stroke-dashoffset: 2000; }
    to { stroke-dashoffset: 0; }
}

/* ========== INTRO ANIMATION UTILITY CLASSES ========== */
.intro-hidden {
    opacity: 0;
}

.intro-fade-up {
    animation: fadeInUp 0.9s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.intro-fade-down {
    animation: fadeInDown 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.intro-blur-reveal {
    animation: blurReveal 1.1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.intro-scale-in {
    animation: scaleIn 0.9s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.intro-d1 { animation-delay: 0.1s; }
.intro-d2 { animation-delay: 0.3s; }
.intro-d3 { animation-delay: 0.5s; }
.intro-d4 { animation-delay: 0.7s; }
.intro-d5 { animation-delay: 0.9s; }
.intro-d6 { animation-delay: 1.1s; }
.intro-d7 { animation-delay: 1.3s; }

/* ========== SCROLL REVEAL SYSTEM ========== */
.scroll-reveal {
    opacity: 0;
    transform: translateY(45px);
    transition: opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1),
                transform 0.8s cubic-bezier(0.16, 1, 0.3, 1),
                filter 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    filter: blur(4px);
}

.scroll-reveal.visible {
    opacity: 1;
    transform: translateY(0);
    filter: blur(0);
}

.scroll-reveal-left {
    opacity: 0;
    transform: translateX(-50px);
    transition: opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1),
                transform 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

.scroll-reveal-left.visible {
    opacity: 1;
    transform: translateX(0);
}

.scroll-reveal-right {
    opacity: 0;
    transform: translateX(50px);
    transition: opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1),
                transform 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

.scroll-reveal-right.visible {
    opacity: 1;
    transform: translateX(0);
}

.scroll-reveal-scale {
    opacity: 0;
    transform: scale(0.9);
    transition: opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1),
                transform 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

.scroll-reveal-scale.visible {
    opacity: 1;
    transform: scale(1);
}

.sr-delay-1 { transition-delay: 0.1s; }
.sr-delay-2 { transition-delay: 0.2s; }
.sr-delay-3 { transition-delay: 0.35s; }
.sr-delay-4 { transition-delay: 0.5s; }
.sr-delay-5 { transition-delay: 0.65s; }

/* ========== NAVIGATION ========== */
.maryl-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 36px;
    background: rgba(23, 19, 35, 0.85);
    border: 1px solid rgba(215, 198, 255, 0.12);
    border-radius: 9999px;
    backdrop-filter: blur(20px);
    margin-bottom: 30px;
    position: sticky;
    top: 15px;
    z-index: 1000;
    box-shadow: 0 10px 30px rgba(10, 6, 18, 0.5);
    animation: fadeInDown 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.05s both;
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #FFFFFF;
    text-decoration: none;
}

.nav-sparkle {
    color: #F5D061;
    font-size: 16px;
    animation: spinSparkle 8s linear infinite;
    display: inline-block;
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 24px;
    font-size: 13px;
    font-weight: 500;
}

.nav-link {
    color: #C9C2E0;
    text-decoration: none;
    transition: color 0.2s ease;
}
.nav-link:hover { color: #D7C6FF; }

.nav-contact-btn {
    background: rgba(215, 198, 255, 0.12);
    border: 1px solid rgba(215, 198, 255, 0.35);
    color: #FFFFFF !important;
    padding: 8px 18px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.2s ease;
}
.nav-contact-btn:hover {
    background: #6658FF;
    border-color: #6658FF;
    box-shadow: 0 0 20px rgba(102, 88, 255, 0.5);
}

/* ========== HERO ========== */
.hero-container {
    padding: 60px 20px 40px 20px;
    text-align: center;
    position: relative;
}

.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #D7C6FF;
    margin-bottom: 20px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(215, 198, 255, 0.08);
    padding: 6px 16px;
    border-radius: 9999px;
    border: 1px solid rgba(215, 198, 255, 0.2);
}

.hero-title-giant {
    font-size: clamp(40px, 7vw, 76px);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.04em;
    color: #FFFFFF;
    margin: 0 auto;
    max-width: 1000px;
}

.hero-title-giant em {
    font-family: 'Playfair Display', Georgia, serif;
    font-style: italic;
    font-weight: 400;
    background: linear-gradient(135deg, #D7C6FF 0%, #EAC5ED 40%, #B3E6DB 80%, #FFFFFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.hero-sub-p {
    font-size: 18px;
    color: #C9C2E0;
    max-width: 720px;
    margin: 24px auto 36px auto;
    line-height: 1.6;
    font-weight: 400;
}

/* ========== 3D FLOATING ARTIFACT ========== */
.artifact-3d-wrapper {
    perspective: 1200px;
    margin: 40px auto;
    max-width: 680px;
}

.artifact-3d-card {
    background: linear-gradient(135deg, rgba(35, 23, 51, 0.85) 0%, rgba(20, 16, 30, 0.95) 100%);
    border: 1px solid rgba(215, 198, 255, 0.25);
    border-radius: 28px;
    padding: 36px 40px;
    box-shadow:
        0 30px 80px -15px rgba(10, 6, 18, 0.9),
        0 0 50px -10px rgba(102, 88, 255, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    animation: floatGently 6s ease-in-out infinite;
}
.artifact-3d-card:hover {
    animation-play-state: paused;
    transform: rotateX(0deg) rotateY(0deg) translateY(-8px);
    box-shadow:
        0 40px 100px -10px rgba(10, 6, 18, 1),
        0 0 70px rgba(215, 198, 255, 0.35);
}

.artifact-glow-ring {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(102, 88, 255, 0.4) 0%, rgba(23, 19, 35, 0) 70%);
    border: 1px solid rgba(215, 198, 255, 0.4);
    font-size: 36px;
    margin-bottom: 16px;
    animation: pulseGlow 3s ease-in-out infinite;
}

/* ========== STORY CHAPTERS ========== */
.chapter-section {
    padding: 50px 0;
    border-top: 1px solid rgba(215, 198, 255, 0.1);
}

.chapter-grid {
    display: grid;
    grid-template-columns: 1fr 1.4fr;
    gap: 40px;
    align-items: center;
}

.chapter-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #D7C6FF;
    margin-bottom: 8px;
}

.chapter-title {
    font-size: 38px;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.03em;
    color: #FFFFFF;
    margin: 0;
}

.chapter-title em {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-weight: 400;
    color: #D7C6FF;
}

.chapter-text {
    font-size: 16px;
    color: #C9C2E0;
    line-height: 1.7;
    margin-top: 14px;
}

/* ========== MANIFESTO ========== */
.manifesto-box {
    text-align: center;
    padding: 80px 20px;
    position: relative;
}

.manifesto-title {
    font-size: clamp(32px, 5vw, 56px);
    font-weight: 700;
    line-height: 1.2;
    letter-spacing: -0.02em;
    color: #FFFFFF;
    max-width: 900px;
    margin: 0 auto;
}

.manifesto-title span {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    color: #D7C6FF;
}

.manifesto-star-svg {
    display: inline-block;
    width: 38px;
    height: 38px;
    vertical-align: middle;
    color: #F5D061;
    margin: 0 10px;
    filter: drop-shadow(0 0 12px rgba(245, 208, 97, 0.7));
    animation: spinSparkle 12s linear infinite;
}

.manifesto-note {
    font-size: 16px;
    color: #A59DBF;
    margin-top: 18px;
    font-style: italic;
}

.ribbon-svg {
    width: 100%;
    max-width: 800px;
    height: 70px;
    margin: 20px auto;
    display: block;
    color: rgba(215, 198, 255, 0.25);
}

.ribbon-svg path {
    stroke-dasharray: 2000;
    stroke-dashoffset: 2000;
}

.ribbon-svg.visible path {
    animation: ribbonDraw 2.5s ease forwards;
}

/* ========== SHOWCASE CARDS ========== */
.showcase-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 24px;
    margin-top: 30px;
}

.showcase-card {
    background: linear-gradient(145deg, rgba(35, 23, 51, 0.7) 0%, rgba(20, 16, 30, 0.85) 100%);
    border: 1px solid rgba(215, 198, 255, 0.16);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 16px 36px rgba(10, 6, 18, 0.6);
    backdrop-filter: blur(16px);
    transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    position: relative;
}
.showcase-card:hover {
    transform: translateY(-5px) scale(1.01);
    border-color: rgba(215, 198, 255, 0.4);
    box-shadow: 0 24px 50px rgba(102, 88, 255, 0.25);
}

.showcase-card-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #D7C6FF;
    margin-bottom: 8px;
}

.showcase-card-h {
    font-size: 22px;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 8px 0;
}

.showcase-card-p {
    font-size: 13px;
    color: #C9C2E0;
    line-height: 1.6;
    margin: 0;
}

/* ========== STATUS PILLS ========== */
.pill-rose {
    display: inline-block;
    background: rgba(244, 63, 94, 0.15);
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: #FB7185;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}

.pill-lavender {
    display: inline-block;
    background: rgba(215, 198, 255, 0.12);
    border: 1px solid rgba(215, 198, 255, 0.35);
    color: #D7C6FF;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}

.pill-mint {
    display: inline-block;
    background: rgba(179, 230, 219, 0.12);
    border: 1px solid rgba(179, 230, 219, 0.35);
    color: #B3E6DB;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
}

/* ========== CONTACT SECTION ========== */
.contact-section {
    background: linear-gradient(145deg, rgba(35, 23, 51, 0.85) 0%, rgba(18, 14, 28, 0.95) 100%);
    border: 1px solid rgba(215, 198, 255, 0.22);
    border-radius: 28px;
    padding: 48px 56px;
    margin-top: 60px;
    box-shadow: 0 30px 80px rgba(10, 6, 18, 0.8);
    position: relative;
}

.contact-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(215, 198, 255, 0.12);
    padding-bottom: 20px;
    margin-bottom: 28px;
}

.contact-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #D7C6FF;
    text-transform: uppercase;
    margin: 0;
}

.contact-spark-text {
    font-size: 12px;
    color: #A59DBF;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.contact-title-big {
    font-family: 'Playfair Display', serif;
    font-size: clamp(36px, 5.5vw, 64px);
    color: #FFFFFF;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 16px;
    line-height: 1.1;
    transition: color 0.2s ease;
    margin: 10px 0 30px 0;
}
.contact-title-big em {
    font-style: italic;
    color: #D7C6FF;
}
.contact-title-big:hover em {
    color: #B3E6DB;
}

.contact-bottom {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(215, 198, 255, 0.12);
    padding-top: 24px;
    flex-wrap: wrap;
    gap: 16px;
}

.contact-email-link {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 16px;
    font-weight: 600;
    color: #FFFFFF;
    text-decoration: none;
    border-bottom: 1px solid #D7C6FF;
    padding-bottom: 2px;
}

/* ========== STREAMLIT OVERRIDES ========== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(23, 19, 35, 0.85);
    padding: 6px;
    border-radius: 16px;
    border: 1px solid rgba(215, 198, 255, 0.14);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 20px;
    color: #A59DBF;
    font-weight: 600;
    font-size: 13px;
    background: transparent;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: rgba(102, 88, 255, 0.3) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: 1px solid rgba(215, 198, 255, 0.4) !important;
    box-shadow: 0 4px 15px rgba(102, 88, 255, 0.25) !important;
}

section[data-testid="stSidebar"] {
    background-color: #120E1C;
    border-right: 1px solid rgba(215, 198, 255, 0.08);
}

/* ========== INSIGHT CALLOUT ========== */
.insight-callout {
    background: rgba(35, 23, 51, 0.7);
    border: 1px solid rgba(215, 198, 255, 0.2);
    border-radius: 14px;
    padding: 16px 22px;
    font-size: 14px;
    color: #C9C2E0;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Plotly Chart Styling Helper
# ---------------------------------------------------------
def style_magic_chart(fig, height=390, title=None):
    """
    Styles Plotly charts with velvet night, lavender, and mint aesthetic
    without any duplicate keyword argument collisions on axes.
    """
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans, sans-serif', color='#C9C2E0', size=12),
        height=height,
        margin=dict(l=20, r=20, t=55 if title else 25, b=25)
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(family='Plus Jakarta Sans', size=14, color='#F8F6F2')))
    fig.update_xaxes(
        gridcolor='rgba(215, 198, 255, 0.08)',
        zerolinecolor='rgba(215, 198, 255, 0.15)',
        tickfont=dict(family='JetBrains Mono, monospace', size=11, color='#9D96B8')
    )
    fig.update_yaxes(
        gridcolor='rgba(215, 198, 255, 0.08)',
        zerolinecolor='rgba(215, 198, 255, 0.15)',
        tickfont=dict(family='JetBrains Mono, monospace', size=11, color='#9D96B8')
    )
    return fig

# ---------------------------------------------------------
# Data Ingestion
# ---------------------------------------------------------
@st.cache_data
def get_clean_data():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Churn_Modelling.csv")
    if not os.path.exists(data_path):
        data_path = "Churn_Modelling.csv"
    return load_and_preprocess_data(data_path)

raw_df = get_clean_data()

# ---------------------------------------------------------
# Scroll-Trigger JavaScript (IntersectionObserver on parent)
# ---------------------------------------------------------
components.html("""
<script>
(function() {
    const parentDoc = window.parent.document;

    function initScrollReveal() {
        const revealSelectors = [
            '.scroll-reveal',
            '.scroll-reveal-left',
            '.scroll-reveal-right',
            '.scroll-reveal-scale'
        ];

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, {
            threshold: 0.12,
            rootMargin: '0px 0px -40px 0px'
        });

        revealSelectors.forEach(sel => {
            parentDoc.querySelectorAll(sel).forEach(el => {
                observer.observe(el);
            });
        });

        // Also trigger ribbon-draw animation
        parentDoc.querySelectorAll('.ribbon-svg').forEach(el => {
            const ribbonObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                    }
                });
            }, { threshold: 0.3 });
            ribbonObserver.observe(el);
        });
    }

    // Run after a brief delay to let Streamlit render
    setTimeout(initScrollReveal, 600);
    // Re-run periodically in case new elements are rendered (tab changes, etc.)
    setInterval(initScrollReveal, 2000);
})();
</script>
""", height=0)

# ---------------------------------------------------------
# Navigation (Animated on load)
# ---------------------------------------------------------
st.markdown("""
<div class="maryl-nav">
<div class="nav-brand">
<span class="nav-sparkle">✦</span>
<span>RETENTION INTELLIGENCE</span>
</div>
<div class="nav-links">
<a href="#work" class="nav-link">Analytical Folios <span>7</span></a>
<a href="#story" class="nav-link">The Story</a>
<a href="#manifesto" class="nav-link">Manifesto</a>
<a href="#contact" class="nav-contact-btn">Contact / Inquire ↗</a>
</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Hero Section with Staggered Intro Animations
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container" id="top">
<div class="hero-eyebrow intro-fade-up intro-d1" style="opacity:0;">
<span>✦</span> THOUGHTFUL BY DATA. RETENTION BY DESIGN. A LITTLE MAGIC.
</div>
<h1 class="hero-title-giant intro-blur-reveal intro-d2" style="opacity:0;">
Thoughtful by <em>design.</em><br>
Predictive by <em>data.</em><br>
A little <em>magic.</em>
</h1>
<p class="hero-sub-p intro-fade-up intro-d3" style="opacity:0;">
An empirical investigation into 10,000 European retail banking accounts across France, Germany, and Spain.
Evaluating customer stickiness, product depth, and silent capital flight.
</p>
<div class="artifact-3d-wrapper intro-scale-in intro-d5" style="opacity:0;">
<div class="artifact-3d-card">
<div class="artifact-glow-ring">🏛️</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 2px; color: #D7C6FF; text-transform: uppercase;">
EUROPEAN CENTRAL BANK BENCHMARK &bull; AUDITED PORTFOLIO
</div>
<div style="font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 600; color: #FFFFFF; margin: 8px 0;">
€764.86 Million <em style="color: #B3E6DB; font-style: italic;">Monitored.</em>
</div>
<p style="font-size: 14px; color: #C9C2E0; line-height: 1.6; margin: 0;">
Demonstrating how active transacting velocity and the optimal 2-product anchor protect over <strong>€50.12 Million in deposit capital</strong> from silent flight.
</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Story Chapters with Scroll-Triggered Reveals
# ---------------------------------------------------------
st.markdown("""
<div id="story">
<div class="chapter-section scroll-reveal">
<div class="chapter-grid">
<div class="scroll-reveal-left sr-delay-1">
<div class="chapter-eyebrow">CHAPTER 01 &bull; BEHAVIOR OVER DEMOGRAPHICS</div>
<h2 class="chapter-title">Human by<br><em>design.</em></h2>
</div>
<div class="chapter-text scroll-reveal-right sr-delay-2">
Traditional banking routinely confuses ledger solvency with customer commitment.
A customer maintaining six figures who never transacts is not a loyal customer—they are
an unengaged depositor quietly awaiting the next yield shift. We model banking through genuine
human engagement.
</div>
</div>
</div>

<div class="chapter-section scroll-reveal">
<div class="chapter-grid">
<div class="scroll-reveal-left sr-delay-1">
<div class="chapter-eyebrow">CHAPTER 02 &bull; ECONOMETRIC RIGOR</div>
<h2 class="chapter-title">Precision by<br><em>proof.</em></h2>
</div>
<div class="chapter-text scroll-reveal-right sr-delay-2">
From Pearson's Chi-Square Contingency tests to Mann-Whitney U distributions and Random Forest
permutation rankings: empirical proof that behavioral metrics dominate traditional salary
and credit scores by an order of magnitude.
</div>
</div>
</div>

<div class="chapter-section scroll-reveal">
<div class="chapter-grid">
<div class="scroll-reveal-left sr-delay-1">
<div class="chapter-eyebrow">CHAPTER 03 &bull; CAPITAL PRESERVATION</div>
<h2 class="chapter-title">More than<br><em>numbers.</em></h2>
</div>
<div class="chapter-text scroll-reveal-right sr-delay-2">
Safeguarding over <strong>€50.12 Million</strong> in liquidity reserves and protecting
<strong>€1.25 Million</strong> in recurring annual Net Interest Margin. Aligning retail banking
with Basel III and European Central Bank prudential liquidity standards.
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Manifesto with Scroll-Triggered Sparkle & Ribbon Draw
# ---------------------------------------------------------
st.markdown("""
<div class="manifesto-box scroll-reveal" id="manifesto">
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 2px; color: #D7C6FF; text-transform: uppercase; margin-bottom: 12px;">
GOOD BANKING MAKES SENSE. GREAT RETENTION UNDERSTANDS BEHAVIOR.
</div>
<div class="manifesto-title">
Equal parts <span>intention,</span> intuition
<svg class="manifesto-star-svg" viewBox="0 0 100 100" fill="currentColor">
<path d="M50 0C56 32 68 44 100 50C68 56 56 68 50 100C44 68 32 56 0 50C32 44 44 32 50 0Z"></path>
</svg>
&amp; a little <span>unexpected.</span>
</div>
<div class="manifesto-note">
From the first deposit sweep to the deepest relationship anchor.<br>
We model it. We simulate it. We safeguard it.
</div>
<svg viewBox="0 0 1000 200" class="ribbon-svg">
<path d="M-20 150C180-20 250 220 440 75S600 200 490 155S700-30 1020 130" fill="none" stroke="currentColor" stroke-width="2.5"></path>
</svg>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Portfolio Showcase Cards with Staggered Scroll Reveals
# ---------------------------------------------------------
st.markdown("""
<div class="scroll-reveal" style="margin-bottom: 20px;">
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 2px; color: #D7C6FF; text-transform: uppercase;">
PORTFOLIO SHOWCASE
</div>
<div style="font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 600; color: #FFFFFF; margin-top: 4px;">
Core Empirical Discoveries.
</div>
</div>

<div class="showcase-grid">
<div class="showcase-card scroll-reveal-scale sr-delay-1">
<div class="showcase-card-num">FOLIO 01</div>
<div class="showcase-card-h">The 2-Product Golden Zone</div>
<p class="showcase-card-p">
Moving from 1 to 2 products plunges churn from <strong>27.71% down to 7.58%</strong> (a 72.6% relative collapse in attrition).
</p>
</div>
<div class="showcase-card scroll-reveal-scale sr-delay-2">
<div class="showcase-card-num">FOLIO 02</div>
<div class="showcase-card-h">High-Balance Flight Paradox</div>
<p class="showcase-card-p">
Customers with &gt;€100k balance control <strong>€312.27 Million</strong> (40.8% of portfolio), yet 49.1% are dormant with a 32.8% churn flight risk.
</p>
</div>
<div class="showcase-card scroll-reveal-scale sr-delay-3">
<div class="showcase-card-num">FOLIO 03</div>
<div class="showcase-card-h">The German Banking Anomaly</div>
<p class="showcase-card-p">
100% of German accounts carry positive balances (€119,730 average), yet experience <strong>32.44% churn</strong>—double France and Spain.
</p>
</div>
<div class="showcase-card scroll-reveal-scale sr-delay-4">
<div class="showcase-card-num">FOLIO 04</div>
<div class="showcase-card-h">Relationship Strength Index (RSI)</div>
<p class="showcase-card-p">
A composite 0–100 behavioral scoring model that proves retained customers score <strong>60.0</strong> vs <strong>42.8</strong> for departing accounts.
</p>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Interactive Work Studio Header
# ---------------------------------------------------------
st.markdown("""
<div id="work" class="scroll-reveal" style="margin-bottom: 25px;">
<div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 15px;">
<div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 2px; color: #D7C6FF; text-transform: uppercase;">
✦ INTERACTIVE WORK STUDIO
</div>
<div style="font-family: 'Playfair Display', serif; font-size: 36px; font-weight: 600; color: #FFFFFF; margin-top: 4px;">
Live Analytical Folios.
</div>
</div>
<div style="font-size: 13px; color: #A59DBF;">
Toggle below to explore spatial 3D models, simulation engines &amp; lead rosters.
</div>
</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Filter Controls
with st.sidebar:
    st.markdown("### ✦ Tactical Cohort Filtering")
    geo_list = sorted(raw_df['Geography'].unique().tolist())
    selected_geos = st.multiselect("Geography", geo_list, default=geo_list)

    gender_list = sorted(raw_df['Gender'].unique().tolist())
    selected_genders = st.multiselect("Gender", gender_list, default=gender_list)

    activity_filter = st.radio("Activity State", ["All Accounts", "Active Members Only", "Inactive Members Only"])

    all_prods = sorted(raw_df['NumOfProducts'].unique().tolist())
    selected_prods = st.multiselect("Products Held", all_prods, default=all_prods)

    bal_min, bal_max = int(raw_df['Balance'].min()), int(raw_df['Balance'].max())
    bal_range = st.slider("Ledger Balance (€)", bal_min, bal_max, (bal_min, bal_max), step=5000)

    sal_min, sal_max = int(raw_df['EstimatedSalary'].min()), int(raw_df['EstimatedSalary'].max())
    sal_range = st.slider("Estimated Salary (€)", sal_min, sal_max, (sal_min, sal_max), step=5000)

filtered_df = raw_df[
    (raw_df['Geography'].isin(selected_geos)) &
    (raw_df['Gender'].isin(selected_genders)) &
    (raw_df['NumOfProducts'].isin(selected_prods)) &
    (raw_df['Balance'] >= bal_range[0]) &
    (raw_df['Balance'] <= bal_range[1]) &
    (raw_df['EstimatedSalary'] >= sal_range[0]) &
    (raw_df['EstimatedSalary'] <= sal_range[1])
]

if activity_filter == "Active Members Only":
    filtered_df = filtered_df[filtered_df['IsActiveMember'] == 1]
elif activity_filter == "Inactive Members Only":
    filtered_df = filtered_df[filtered_df['IsActiveMember'] == 0]

if len(filtered_df) == 0:
    st.error("No customer records match your filter criteria. Adjust the sidebar sliders.")
    st.stop()

kpis = calculate_portfolio_kpis(filtered_df)

# ---------------------------------------------------------
# Primary Navigation Tabs
# ---------------------------------------------------------
tab_command, tab_3d_hologram, tab_engagement, tab_products, tab_vip_radar, tab_simulation, tab_methodology = st.tabs([
    "✦ Executive Telemetry",
    "🪐 3D Relationship Space",
    "⚡ Engagement Dynamics",
    "📦 The 2-Product Golden Zone",
    "🎯 VIP Account Radar",
    "📈 Capital Preservation Engine",
    "📑 Econometric & Policy Tests"
])

# =========================================================
# TAB 1: EXECUTIVE TELEMETRY
# =========================================================
with tab_command:
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(f"""<div class="showcase-card">
<div class="showcase-card-num">METRIC I</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #9D96B8;">PORTFOLIO CHURN</div>
<div style="font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 600; color: #FFFFFF; margin: 4px 0;">{kpis['overall_churn_rate']*100:.2f}%</div>
<span class="pill-rose">ERR</span>
</div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""<div class="showcase-card">
<div class="showcase-card-num">METRIC II</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #9D96B8;">PRODUCT DEPTH</div>
<div style="font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 600; color: #FFFFFF; margin: 4px 0;">{kpis['product_depth_index']:.3f}</div>
<span class="pill-lavender">PDI</span>
</div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""<div class="showcase-card">
<div class="showcase-card-num">METRIC III</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #9D96B8;">HIGH-BAL DISENGAGEMENT</div>
<div style="font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 600; color: #FFFFFF; margin: 4px 0;">{kpis['high_balance_disengagement_rate']*100:.1f}%</div>
<span class="pill-rose">HBDR</span>
</div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""<div class="showcase-card">
<div class="showcase-card-num">METRIC IV</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #9D96B8;">CREDIT CARD STICKINESS</div>
<div style="font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 600; color: #FFFFFF; margin: 4px 0;">{kpis['credit_card_stickiness_score']*100:.1f}%</div>
<span class="pill-mint">CCSS</span>
</div>""", unsafe_allow_html=True)

    with m5:
        st.markdown(f"""<div class="showcase-card">
<div class="showcase-card-num">METRIC V</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #9D96B8;">RELATIONSHIP STRENGTH</div>
<div style="font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 600; color: #FFFFFF; margin: 4px 0;">{kpis['avg_relationship_strength']:.1f}</div>
<span class="pill-lavender">RSI / 100</span>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dial & Sunburst
    col_g1, col_g2, col_g3 = st.columns(3)

    with col_g1:
        churn_val = kpis['overall_churn_rate'] * 100
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_val,
            title={'text': "<b>Portfolio Churn Rate</b><br><span style='font-size:12px;color:#A59DBF'>Alert Zone > 25%</span>", 'font': {'family': 'Playfair Display, serif', 'color': '#FFFFFF'}},
            gauge={
                'axis': {'range': [0, 40], 'tickcolor': '#A59DBF'},
                'bar': {'color': '#FB7185'},
                'bgcolor': 'rgba(35, 23, 51, 0.6)',
                'borderwidth': 1,
                'bordercolor': 'rgba(215, 198, 255, 0.2)',
                'steps': [
                    {'range': [0, 15], 'color': 'rgba(179, 230, 219, 0.3)'},
                    {'range': [15, 25], 'color': 'rgba(245, 208, 97, 0.25)'},
                    {'range': [25, 40], 'color': 'rgba(251, 113, 133, 0.35)'}
                ],
            }
        ))
        fig_g1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#FFFFFF'), height=260, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_g1, width='stretch')

    with col_g2:
        avg_rsi = filtered_df['RelationshipStrengthIndex'].mean()
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_rsi,
            title={'text': "<b>Relationship Strength (RSI)</b><br><span style='font-size:12px;color:#A59DBF'>Scale: 0 to 100</span>", 'font': {'family': 'Playfair Display, serif', 'color': '#FFFFFF'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#A59DBF'},
                'bar': {'color': '#B3E6DB'},
                'bgcolor': 'rgba(35, 23, 51, 0.6)',
                'borderwidth': 1,
                'bordercolor': 'rgba(215, 198, 255, 0.2)',
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(251, 113, 133, 0.3)'},
                    {'range': [40, 70], 'color': 'rgba(245, 208, 97, 0.25)'},
                    {'range': [70, 100], 'color': 'rgba(179, 230, 219, 0.3)'}
                ],
            }
        ))
        fig_g2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#FFFFFF'), height=260, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_g2, width='stretch')

    with col_g3:
        sunburst_sample = filtered_df.sample(min(3000, len(filtered_df)), random_state=42).copy()
        sunburst_sample['Activity'] = sunburst_sample['IsActiveMember'].map({1: 'Active', 0: 'Inactive'})
        sunburst_sample['ProductTier'] = sunburst_sample['NumOfProducts'].apply(lambda x: f"{x} Prod")
        sunburst_sample['Status'] = sunburst_sample['Exited'].map({1: 'Exited', 0: 'Retained'})

        fig_sun = px.sunburst(
            sunburst_sample,
            path=['Geography', 'ProductTier', 'Activity', 'Status'],
            color='Status',
            color_discrete_map={'Retained': '#B3E6DB', 'Exited': '#FB7185'},
            title="Portfolio Structural Decomposition"
        )
        fig_sun.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#FFFFFF'), height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_sun, width='stretch')

    st.markdown("---")

    # Segment Exposure Breakdown
    st.markdown("### 🏛️ Capital Exposure by Engagement Profile")
    cohort_df = compute_segment_exposure(filtered_df)

    c_bar, c_tree = st.columns([1.3, 1])

    with c_bar:
        fig_cohort = px.bar(
            cohort_df,
            x='EngagementProfile',
            y='ChurnRate',
            color='AvgBalance',
            color_continuous_scale=['#493B58', '#6658FF', '#D7C6FF', '#FB7185'],
            text=[f"{v*100:.1f}%" for v in cohort_df['ChurnRate']],
            labels={'ChurnRate': 'Observed Churn', 'EngagementProfile': 'Cohort', 'AvgBalance': 'Avg Balance (€)'},
            title="Observed Churn Rate by Behavioral Profile (Color = Mean Balance €)"
        )
        fig_cohort.update_traces(textposition='outside')
        style_magic_chart(fig_cohort, height=360)
        fig_cohort.update_layout(xaxis_tickangle=-15)
        st.plotly_chart(fig_cohort, width='stretch')

    with c_tree:
        fig_tree = px.treemap(
            cohort_df,
            path=['EngagementProfile'],
            values='TotalBalance',
            color='ChurnRate',
            color_continuous_scale=['#B3E6DB', '#D7C6FF', '#FB7185'],
            title="Deposit Volume Share by Segment"
        )
        fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#FFFFFF'), height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_tree, width='stretch')

# =========================================================
# TAB 2: 3D RELATIONSHIP SPACE
# =========================================================
with tab_3d_hologram:
    st.subheader("🪐 3D Spatial Relationship Landscape")
    st.caption("Interact, zoom, and rotate in 3D: Customer Age vs. Ledger Balance vs. Product Depth.")

    col_3d_ctrl1, col_3d_ctrl2, col_3d_ctrl3 = st.columns(3)
    with col_3d_ctrl1:
        color_by_3d = st.selectbox("Color Coordinate By", ["Exited (Retention vs Churn)", "EngagementProfile", "RelationshipStrengthIndex"])
    with col_3d_ctrl2:
        sample_size_3d = st.slider("3D Sampling Density", min_value=500, max_value=min(3000, len(filtered_df)), value=min(1500, len(filtered_df)), step=250)
    with col_3d_ctrl3:
        camera_angle = st.selectbox("Perspective Viewpoint", ["Isometric Perspective", "Top View (Age vs Balance)", "Side View (Products vs Balance)"])

    hologram_sample = filtered_df.sample(sample_size_3d, random_state=42).copy()

    if color_by_3d == "Exited (Retention vs Churn)":
        hologram_sample['Outcome'] = hologram_sample['Exited'].map({0: 'Retained Account', 1: 'Exited Account'})
        fig_3d = px.scatter_3d(
            hologram_sample,
            x='Age',
            y='Balance',
            z='NumOfProducts',
            color='Outcome',
            size='RelationshipStrengthIndex',
            size_max=12,
            opacity=0.85,
            color_discrete_map={'Retained Account': '#B3E6DB', 'Exited Account': '#FB7185'},
            hover_name='Surname',
            hover_data={
                'CustomerId': True,
                'Age': True,
                'Balance': ':.2f',
                'NumOfProducts': True,
                'RelationshipStrengthIndex': ':.1f',
                'Geography': True
            },
            title="3D Customer Space (Size = Relationship Strength Index)"
        )
    else:
        fig_3d = px.scatter_3d(
            hologram_sample,
            x='Age',
            y='Balance',
            z='NumOfProducts',
            color=color_by_3d,
            size='RelationshipStrengthIndex',
            size_max=12,
            opacity=0.85,
            color_continuous_scale=['#493B58', '#6658FF', '#D7C6FF', '#FB7185'],
            hover_name='Surname',
            hover_data={'CustomerId': True, 'Age': True, 'Balance': ':.2f', 'NumOfProducts': True},
            title=f"3D Space Colored by {color_by_3d}"
        )

    if camera_angle == "Top View (Age vs Balance)":
        cam_eye = dict(x=0, y=0, z=2.2)
    elif camera_angle == "Side View (Products vs Balance)":
        cam_eye = dict(x=2.2, y=0, z=0)
    else:
        cam_eye = dict(x=1.4, y=-1.4, z=1.2)

    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans, sans-serif', color='#C9C2E0'),
        scene=dict(
            xaxis=dict(title='Age (Years)', backgroundcolor='#1E192E', gridcolor='#2E2345', showbackground=True),
            yaxis=dict(title='Balance (€)', backgroundcolor='#1E192E', gridcolor='#2E2345', showbackground=True),
            zaxis=dict(title='Product Count', tickvals=[1, 2, 3, 4], backgroundcolor='#1E192E', gridcolor='#2E2345', showbackground=True),
            camera=dict(eye=cam_eye)
        ),
        height=620,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_3d, width='stretch')

    st.markdown("""<div class="insight-callout">
<span style="color: #D7C6FF; font-weight: 700;">✦ Spatial Behavioral Insight:</span>
Observe the distinct cluster of <span style="color: #FB7185; font-weight: 700;">rose churned points</span> concentrated at
<strong>Age 45–60 with balances above €100,000 and 1 or 3+ products</strong>.
This cluster accounts for over sixty percent of total retail capital flight.
</div>""", unsafe_allow_html=True)

# =========================================================
# TAB 3: ENGAGEMENT DYNAMICS
# =========================================================
with tab_engagement:
    st.subheader("⚡ Behavioral Engagement Dynamics & Attrition")
    st.caption("Active transacting frequency acts as the primary protective moat against account attrition.")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        eng_split = filtered_df.groupby('IsActiveMember')['Exited'].agg(
            Count='count',
            ChurnRate='mean'
        ).reset_index()
        eng_split['Status'] = eng_split['IsActiveMember'].map({0: 'Inactive / Dormant', 1: 'Active Member'})
        eng_split['RetentionRate'] = 1 - eng_split['ChurnRate']

        fig_eng = go.Figure()
        fig_eng.add_trace(go.Bar(
            name='Retained',
            x=eng_split['Status'],
            y=eng_split['RetentionRate'],
            marker_color='#B3E6DB',
            text=[f"{v*100:.1f}%" for v in eng_split['RetentionRate']],
            textposition='inside'
        ))
        fig_eng.add_trace(go.Bar(
            name='Exited',
            x=eng_split['Status'],
            y=eng_split['ChurnRate'],
            marker_color='#FB7185',
            text=[f"{v*100:.1f}%" for v in eng_split['ChurnRate']],
            textposition='inside'
        ))
        style_magic_chart(fig_eng, height=360, title="Retention vs. Churn by Active Member Status")
        fig_eng.update_layout(barmode='stack', yaxis_tickformat='.0%')
        st.plotly_chart(fig_eng, width='stretch')

    with col_e2:
        geo_split = filtered_df.groupby(['Geography', 'IsActiveMember'])['Exited'].mean().reset_index()
        geo_split['Activity'] = geo_split['IsActiveMember'].map({0: 'Inactive', 1: 'Active'})

        fig_geo = px.bar(
            geo_split,
            x='Geography',
            y='Exited',
            color='Activity',
            barmode='group',
            labels={'Exited': 'Churn Rate'},
            color_discrete_map={'Inactive': '#FB7185', 'Active': '#B3E6DB'},
            title="Geographic Churn Comparison Across Jurisdictions"
        )
        style_magic_chart(fig_geo, height=360)
        fig_geo.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_geo, width='stretch')

    st.markdown("---")

    # Financial Commitment Violin & Tenure Heatmap
    st.markdown("### 💰 Financial Commitment & Tenure Survival")
    v1, v2 = st.columns(2)

    with v1:
        fig_violin = px.violin(
            filtered_df,
            x='IsActiveMember',
            y='Balance',
            color='Exited',
            box=True,
            points=False,
            color_discrete_map={0: '#B3E6DB', 1: '#FB7185'},
            labels={'IsActiveMember': 'Activity (0 = Inactive, 1 = Active)', 'Balance': 'Ledger Balance (€)'},
            title="Balance Dispersion by Engagement & Churn Outcome"
        )
        style_magic_chart(fig_violin, height=380)
        st.plotly_chart(fig_violin, width='stretch')

    with v2:
        tenure_heat = filtered_df.groupby(['Tenure', 'NumOfProducts'])['Exited'].mean().unstack()
        fig_heat = px.imshow(
            tenure_heat,
            labels=dict(x="Number of Products", y="Tenure (Years)", color="Churn Rate"),
            color_continuous_scale=['#231733', '#6658FF', '#D7C6FF', '#FB7185'],
            title="Tenure vs. Product Count Churn Heatmap"
        )
        style_magic_chart(fig_heat, height=380)
        st.plotly_chart(fig_heat, width='stretch')

# =========================================================
# TAB 4: THE 2-PRODUCT GOLDEN ZONE
# =========================================================
with tab_products:
    st.subheader("📦 Product Depth Optimization: The 2-Product Golden Zone")
    st.caption("Quantitative proof of non-linear retention dynamics across product count.")

    p_col1, p_col2 = st.columns([1.2, 1])

    with p_col1:
        prod_kpis = filtered_df.groupby('NumOfProducts')['Exited'].agg(
            Count='count',
            ChurnRate='mean'
        ).reset_index()

        fig_golden = go.Figure()
        fig_golden.add_trace(go.Bar(
            x=prod_kpis['NumOfProducts'],
            y=prod_kpis['Count'],
            name='Customer Accounts',
            yaxis='y',
            marker_color='rgba(215, 198, 255, 0.35)',
            marker_line=dict(color='#D7C6FF', width=1.5)
        ))
        fig_golden.add_trace(go.Scatter(
            x=prod_kpis['NumOfProducts'],
            y=prod_kpis['ChurnRate'],
            name='Churn Rate (%)',
            yaxis='y2',
            mode='lines+markers+text',
            text=[f"{v*100:.1f}%" for v in prod_kpis['ChurnRate']],
            textposition='top center',
            line=dict(color='#FB7185', width=3),
            marker=dict(size=10, color='#FF007A')
        ))

        fig_golden.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Plus Jakarta Sans, sans-serif', color='#C9C2E0', size=12),
            title=dict(text="<b>The 2-Product Golden Zone vs. Multi-Product Friction Curve</b>", font=dict(color='#FFFFFF', size=14)),
            xaxis=dict(
                title='Products Held per Customer',
                tickvals=[1, 2, 3, 4],
                gridcolor='rgba(215, 198, 255, 0.08)',
                tickfont=dict(family='JetBrains Mono', color='#A59DBF')
            ),
            yaxis=dict(
                title='Customer Volume',
                side='left',
                gridcolor='rgba(215, 198, 255, 0.08)',
                tickfont=dict(family='JetBrains Mono', color='#A59DBF')
            ),
            yaxis2=dict(
                title='Churn Rate',
                side='right',
                overlaying='y',
                tickformat='.0%',
                range=[0, 1.15],
                tickfont=dict(family='JetBrains Mono', color='#FB7185')
            ),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(x=0.03, y=0.95, bgcolor='rgba(35, 23, 51, 0.8)')
        )
        st.plotly_chart(fig_golden, width='stretch')

    with p_col2:
        cc_active = filtered_df.groupby(['HasCrCard', 'IsActiveMember'])['Exited'].agg(
            Count='count',
            Churn='mean'
        ).reset_index()
        cc_active['Segment'] = cc_active.apply(
            lambda r: f"{'Card' if r['HasCrCard']==1 else 'No Card'} & {'Active' if r['IsActiveMember']==1 else 'Inactive'}", axis=1
        )

        fig_cc_active = px.bar(
            cc_active,
            x='Segment',
            y='Churn',
            color='Churn',
            color_continuous_scale=['#B3E6DB', '#D7C6FF', '#FB7185'],
            text=[f"{v*100:.1f}%" for v in cc_active['Churn']],
            title="Credit Card Retention Conditioned on Activity"
        )
        fig_cc_active.update_traces(textposition='outside')
        style_magic_chart(fig_cc_active, height=400)
        fig_cc_active.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig_cc_active, width='stretch')

    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""<div class="showcase-card">
<div class="showcase-card-num">1 PRODUCT VULNERABILITY</div>
<div style="font-family: 'Playfair Display', serif; font-size: 26px; font-weight: 600; color: #FB7185; margin: 4px 0;">27.71% Churn</div>
<p style="font-size: 13px; color: #C9C2E0; margin: 0;">50.8% of customers hold no secondary anchors, resulting in near-zero switching resistance.</p>
</div>""", unsafe_allow_html=True)
    with b2:
        st.markdown("""<div class="showcase-card" style="border-color: rgba(179, 230, 219, 0.4); box-shadow: 0 0 25px rgba(179, 230, 219, 0.12);">
<div class="showcase-card-num" style="color: #B3E6DB;">THE 2-PRODUCT GOLDEN ZONE</div>
<div style="font-family: 'Playfair Display', serif; font-size: 26px; font-weight: 600; color: #B3E6DB; margin: 4px 0;">7.58% Churn (&minus;72.6%)</div>
<p style="font-size: 13px; color: #C9C2E0; margin: 0;">The mathematical optimum: pairing checking with savings or mortgage maximizes loyalty.</p>
</div>""", unsafe_allow_html=True)
    with b3:
        st.markdown("""<div class="showcase-card" style="border-color: rgba(251, 113, 133, 0.35);">
<div class="showcase-card-num" style="color: #FB7185;">MULTI-PRODUCT HOCKEY STICK</div>
<div style="font-family: 'Playfair Display', serif; font-size: 26px; font-weight: 600; color: #FB7185; margin: 4px 0;">82.7% to 100% Churn</div>
<p style="font-size: 13px; color: #C9C2E0; margin: 0;">Forced bundling beyond 2 products triggers customer alienation, fee fatigue, and immediate flight.</p>
</div>""", unsafe_allow_html=True)

# =========================================================
# TAB 5: VIP ACCOUNT RADAR & DOSSIER
# =========================================================
with tab_vip_radar:
    st.subheader("🎯 VIP Lead Detector & Individual Customer Radar Dossier")
    st.caption("Identify vulnerable high-balance accounts and inspect customer profiles with radar benchmarks.")

    c_lead1, c_lead2, c_lead3 = st.columns(3)
    with c_lead1:
        vip_min_bal = st.number_input("VIP Balance Threshold (€)", value=100000, step=10000)
    with c_lead2:
        vip_tier_filter = st.multiselect(
            "Retention Tier Filter",
            ['Critical Disengagement (Flight Risk)', 'Vulnerable', 'Moderate Engagement', 'Sticky Anchor (Very High Loyalty)'],
            default=['Critical Disengagement (Flight Risk)', 'Vulnerable']
        )
    with c_lead3:
        sort_order = st.selectbox("Sort Priority", ["Balance: High to Low", "RSI: Low to High", "Salary: High to Low"])

    vip_leads = filtered_df[
        (filtered_df['Balance'] >= vip_min_bal) &
        (filtered_df['RetentionTier'].isin(vip_tier_filter))
    ].copy()

    if sort_order == "Balance: High to Low":
        vip_leads = vip_leads.sort_values(by='Balance', ascending=False)
    elif sort_order == "RSI: Low to High":
        vip_leads = vip_leads.sort_values(by='RelationshipStrengthIndex', ascending=True)
    else:
        vip_leads = vip_leads.sort_values(by='EstimatedSalary', ascending=False)

    col_vm1, col_vm2, col_vm3 = st.columns(3)
    with col_vm1:
        st.metric("Identified At-Risk VIP Accounts", f"{len(vip_leads):,} folios")
    with col_vm2:
        st.metric("Total Vulnerable Deposit Volume", f"€{vip_leads['Balance'].sum():,.2f}")
    with col_vm3:
        st.metric("Average Relationship Strength (RSI)", f"{vip_leads['RelationshipStrengthIndex'].mean():.1f} / 100")

    st.markdown("### 📡 Individual Customer Radar Dossier")
    if len(vip_leads) > 0:
        dossier_options = [f"{row['CustomerId']} - {row['Surname']} (€{row['Balance']:,.0f})" for _, row in vip_leads.head(50).iterrows()]
        selected_dossier_str = st.selectbox("Select Customer to Inspect Dossier", dossier_options)
        selected_cid = int(selected_dossier_str.split(" - ")[0])
        cust_record = vip_leads[vip_leads['CustomerId'] == selected_cid].iloc[0]

        r_left, r_right = st.columns([1.2, 1])

        with r_left:
            radar_categories = ['Credit Score', 'Product Depth', 'Activity', 'Tenure', 'Salary Level']

            c_credit = min(cust_record['CreditScore'] / 8.5, 100)
            c_prod = 100 if cust_record['NumOfProducts'] == 2 else (50 if cust_record['NumOfProducts'] == 1 else 15)
            c_act = 100 if cust_record['IsActiveMember'] == 1 else 10
            c_tenure = min((cust_record['Tenure'] / 10.0) * 100, 100)
            c_salary = min((cust_record['EstimatedSalary'] / 200000.0) * 100, 100)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[c_credit, c_prod, c_act, c_tenure, c_salary, c_credit],
                theta=radar_categories + [radar_categories[0]],
                fill='toself',
                fillcolor='rgba(251, 113, 133, 0.25)',
                line=dict(color='#FB7185', width=2),
                name=f"{cust_record['Surname']} Profile"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[80, 100, 100, 75, 70, 80],
                theta=radar_categories + [radar_categories[0]],
                fill='none',
                line=dict(color='#B3E6DB', width=1.5, dash='dash'),
                name="Target Benchmark"
            ))
            fig_radar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#C9C2E0'),
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='#2E2345'),
                    angularaxis=dict(gridcolor='#2E2345')
                ),
                height=340,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, width='stretch')

        with r_right:
            cust_activity = 'Active Member' if cust_record['IsActiveMember'] == 1 else 'Inactive / Dormant'
            cust_card = 'Yes' if cust_record['HasCrCard'] == 1 else 'No'
            st.markdown(f"""<div class="showcase-card">
<div class="showcase-card-num">CUSTOMER PROFILE #{cust_record['CustomerId']}</div>
<div style="font-family: 'Playfair Display', serif; font-size: 26px; font-weight: 600; color: #FFFFFF; margin: 4px 0;">
{cust_record['Surname']}, {cust_record['Geography']}
</div>
<div style="margin: 8px 0;">
<span class="pill-rose">{cust_record['RetentionTier']}</span>
<span class="pill-lavender">RSI: {cust_record['RelationshipStrengthIndex']} / 100</span>
</div>
<div style="font-size: 13px; color: #C9C2E0; line-height: 1.6; margin-top: 10px;">
&bull; <strong>Ledger Balance:</strong> €{cust_record['Balance']:,.2f}<br>
&bull; <strong>Estimated Salary:</strong> €{cust_record['EstimatedSalary']:,.2f}<br>
&bull; <strong>Bank Products:</strong> {cust_record['NumOfProducts']} product(s)<br>
&bull; <strong>Activity Status:</strong> {cust_activity}<br>
&bull; <strong>Credit Card Held:</strong> {cust_card}
</div>
<div style="margin-top: 14px; font-size: 12px; color: #FB7185; background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.3); padding: 10px; border-radius: 8px;">
⚠️ <strong>Recommended Action:</strong> Proactively reach out with personalized wealth advisory. Offer 2-product fixed term or automated savings sweeps to avoid capital drain.
</div>
</div>""", unsafe_allow_html=True)

    # Lead Table
    st.markdown("#### Prioritized Account Roster")
    st.dataframe(
        vip_leads[[
            'CustomerId', 'Surname', 'Geography', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'RelationshipStrengthIndex', 'RetentionTier', 'Exited'
        ]].rename(columns={
            'CustomerId': 'ID',
            'NumOfProducts': 'Products',
            'HasCrCard': 'CreditCard',
            'IsActiveMember': 'Active',
            'RelationshipStrengthIndex': 'RSI Score',
            'RetentionTier': 'Risk Tier',
            'Exited': 'Historical Churn'
        }),
        width='stretch',
        hide_index=True
    )

    csv_data = vip_leads.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Prioritized Outreach Roster (CSV for CRM Direct Outreach)",
        data=csv_data,
        file_name="high_value_disengaged_leads.csv",
        mime="text/csv"
    )

# =========================================================
# TAB 6: WHAT-IF STRATEGY SIMULATOR
# =========================================================
with tab_simulation:
    st.subheader("📈 Retention Strategy What-If Simulator & Capital Preservation Engine")
    st.caption("Model the projected capital protection and churn reduction from three targeted behavioral interventions.")

    col_sim_l, col_sim_r = st.columns([1, 1.2])

    with col_sim_l:
        st.markdown("### 🎛️ Intervention Levers")
        sim_reengage = st.slider(
            "1. Digital Re-engagement Rate (%)",
            0.0, 50.0, 20.0, 5.0,
            help="% of inactive customers converted into active transactors via automated app notifications and fee rebates."
        )
        sim_cross_sell = st.slider(
            "2. Single-to-Optimal-2 Cross-Sell Rate (%)",
            0.0, 40.0, 15.0, 5.0,
            help="% of 1-product customers migrated into the 2-product golden zone (e.g. automated term deposits)."
        )
        sim_concierge = st.slider(
            "3. High-Balance Concierge Churn Reduction (%)",
            0.0, 50.0, 25.0, 5.0,
            help="Relative churn risk reduction achieved through dedicated wealth relationship managers contacting VIPs with >€100k balance."
        )
        sim_nim = st.number_input(
            "Bank Net Interest Margin / Deposit Yield (%)",
            0.5, 6.0, 2.5, 0.25,
            help="Net annualized interest margin earned by the bank on retained customer deposits."
        )

    sim_res = simulate_retention_interventions(
        filtered_df,
        reengagement_conversion_pct=sim_reengage,
        single_to_multi_cross_sell_pct=sim_cross_sell,
        high_bal_concierge_retention_lift=sim_concierge,
        annual_nim_pct=sim_nim
    )

    with col_sim_r:
        st.markdown("### 🎯 Projected Financial Outcomes")

        o1, o2 = st.columns(2)
        with o1:
            st.markdown(f"""<div class="showcase-card">
<div class="showcase-card-num">PROJECTED OUTCOME</div>
<div style="font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 600; color: #FFFFFF; margin: 4px 0;">{sim_res['simulated_churn_rate']*100:.2f}%</div>
<span class="pill-mint">Down from {sim_res['baseline_churn_rate']*100:.2f}% (&minus;{sim_res['churn_rate_reduction_bps']:.0f} bps)</span>
</div>""", unsafe_allow_html=True)
        with o2:
            st.markdown(f"""<div class="showcase-card">
<div class="showcase-card-num">CLIENT PROTECTION</div>
<div style="font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 600; color: #FFFFFF; margin: 4px 0;">{sim_res['total_customers_saved']:,}</div>
<span class="pill-lavender">Accounts Prevented from Exiting</span>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        o3, o4 = st.columns(2)
        with o3:
            st.markdown(f"""<div class="showcase-card">
<div class="showcase-card-num">LIQUIDITY PRESERVED</div>
<div style="font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 600; color: #FFFFFF; margin: 4px 0;">€{sim_res['total_capital_preserved']/1e6:.2f}M</div>
<span class="pill-mint">Protected from Liquidity Run-Off</span>
</div>""", unsafe_allow_html=True)
        with o4:
            st.markdown(f"""<div class="showcase-card">
<div class="showcase-card-num">NET INTEREST REVENUE</div>
<div style="font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 600; color: #FFFFFF; margin: 4px 0;">€{sim_res['annual_revenue_protected']/1e6:.2f}M</div>
<span class="pill-mint">At {sim_nim:.2f}% Annual Spread</span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Strategy Attribution Bar Chart
    st.markdown("### 📊 Strategy Attribution: Capital Preserved Across Levers")
    strat_data = pd.DataFrame([
        {'Strategy': k, 'Customers Saved': v['customers_saved'], 'Capital Saved (€M)': v['capital_saved']/1e6}
        for k, v in sim_res['strategy_breakdown'].items()
    ])

    fig_waterfall = px.bar(
        strat_data,
        x='Strategy',
        y='Capital Saved (€M)',
        color='Capital Saved (€M)',
        color_continuous_scale=['#493B58', '#6658FF', '#B3E6DB'],
        text=[f"€{v:.2f}M" for v in strat_data['Capital Saved (€M)']],
        title="Attribution of Capital Preserved Across Intervention Levers (€ Millions)"
    )
    fig_waterfall.update_traces(textposition='outside')
    style_magic_chart(fig_waterfall, height=360)
    st.plotly_chart(fig_waterfall, width='stretch')

# =========================================================
# TAB 7: EMPIRICAL RESEARCH TESTS
# =========================================================
with tab_methodology:
    st.subheader("📑 Empirical Methodology, Statistical Hypothesis Tests & Models")
    st.caption("Formal statistical validations supporting all conclusions and predictive modeling benchmarks.")

    hyp_tests = run_statistical_hypothesis_tests(raw_df)
    hyp_rows = []
    for k, v in hyp_tests.items():
        hyp_rows.append({
            'Hypothesis Test': v['name'],
            'Statistical Methodology': v['test_type'],
            'Test Statistic': v['statistic'],
            'p-Value': f"{v['p_value']:.4e}" if v['p_value'] < 0.0001 else f"{v['p_value']:.4f}",
            'Significant (α = 0.01)': "✅ Statistically Significant" if v['significant'] else "❌ Not Significant",
            'Empirical Interpretation': v['interpretation']
        })
    st.dataframe(pd.DataFrame(hyp_rows), width='stretch', hide_index=True)

    st.markdown("---")

    # Feature Importance Models
    models_dict = train_behavioral_models(raw_df)
    col_rf, col_or = st.columns(2)

    with col_rf:
        fig_rf = px.bar(
            models_dict['rf_importance'],
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=['#493B58', '#B3E6DB'],
            title="Random Forest Gini Feature Importance"
        )
        style_magic_chart(fig_rf, height=380)
        fig_rf.update_layout(yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_rf, width='stretch')

    with col_or:
        fig_or = px.bar(
            models_dict['odds_ratios'],
            x='OddsRatio',
            y='Feature',
            orientation='h',
            color='Direction',
            color_discrete_map={'Risk Factor (+)': '#FB7185', 'Retention Factor (-)': '#B3E6DB'},
            title="Logistic Regression Odds Ratios"
        )
        fig_or.add_vline(x=1.0, line_dash='dash', line_color='#A59DBF')
        style_magic_chart(fig_or, height=380)
        fig_or.update_layout(yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_or, width='stretch')

# ---------------------------------------------------------
# Contact Section with Scroll Reveal
# ---------------------------------------------------------
st.markdown("""
<div class="contact-section scroll-reveal" id="contact">
<div class="contact-top">
<p class="contact-eyebrow">HAVE SOMETHING IN MIND?</p>
<span class="contact-spark-text">GOOD THINGS START WITH A CONVERSATION.</span>
</div>
<a href="mailto:aryan.analytics@example.com" class="contact-title-big">
<span>Let's make customer retention <em>magic.</em></span>
<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" style="color: #D7C6FF;">
<path d="M7 7h10v10"></path>
<path d="M7 17 17 7"></path>
</svg>
</a>
<div style="margin: 20px 0;">
<svg class="manifesto-star-svg" viewBox="0 0 100 100" fill="currentColor" style="width: 44px; height: 44px;">
<path d="M50 0C56 32 68 44 100 50C68 56 56 68 50 100C44 68 32 56 0 50C32 44 44 32 50 0Z"></path>
</svg>
</div>
<div class="contact-bottom">
<div>
<a href="mailto:aryan.analytics@example.com" class="contact-email-link">aryan.analytics@example.com ↗</a>
</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #A59DBF;">
Unified Mentor Analytics Fellow &bull; European Central Bank Case Framework
</div>
</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #A59DBF; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 12px 0;">
✦ DESIGN, TECHNOLOGY &amp; A LITTLE MAGIC &bull; DEVELOPED BY ARYAN &bull; UNIFIED MENTOR BANKING STUDY
</div>
""", unsafe_allow_html=True)
