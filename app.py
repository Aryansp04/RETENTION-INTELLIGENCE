"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
Executive Retention Intelligence Platform
Inspired by the Design, Technology & Visual Language of Maryl Gonzalez (marylgonzalez.com)
Palette: Deep Velvet Plum Noir (#171323), Ethereal Lavender (#D7C6FF), Sage Mint (#B3E6DB), Electric Violet (#6658FF) & Coral Rose (#FB7185)
Author: Aryan (Internship Portfolio Project)
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
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
    page_title="Retention Intelligence · Design, Data & Strategy",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Maryl Gonzalez Inspired CSS Styling (Velvet Noir, Lavender & Sparkles)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,400;0,600;1,400;1,600&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Core Velvet Canvas */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #F8F6F2;
    }
    
    .stApp {
        background-color: #171323;
        background-image: 
            radial-gradient(at 0% 0%, rgba(102, 88, 255, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(215, 198, 255, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(179, 230, 219, 0.08) 0px, transparent 50%);
    }

    /* Maryl-Inspired Editorial Hero Header */
    .magic-hero {
        position: relative;
        background: linear-gradient(145deg, rgba(35, 23, 51, 0.75) 0%, rgba(23, 19, 35, 0.85) 100%);
        border: 1px solid rgba(215, 198, 255, 0.16);
        border-radius: 24px;
        padding: 38px 44px;
        margin-bottom: 26px;
        box-shadow: 
            0 24px 60px -15px rgba(10, 6, 18, 0.8),
            inset 0 1px 0 rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(24px);
        overflow: hidden;
    }

    .magic-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #D7C6FF;
        margin-bottom: 12px;
    }

    .magic-sparkle {
        display: inline-block;
        color: #F5D061;
        font-size: 14px;
        filter: drop-shadow(0 0 8px rgba(245, 208, 97, 0.6));
    }

    .magic-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        color: #FFFFFF;
        margin: 0;
    }

    .magic-title em {
        font-family: 'Playfair Display', Georgia, serif;
        font-style: italic;
        font-weight: 400;
        background: linear-gradient(135deg, #D7C6FF 0%, #EAC5ED 50%, #B3E6DB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .magic-tagline {
        color: #C9C2E0;
        font-size: 15px;
        line-height: 1.6;
        margin-top: 10px;
        max-width: 900px;
        font-weight: 400;
    }

    .magic-metrics-strip {
        display: flex;
        gap: 28px;
        margin-top: 22px;
        padding-top: 18px;
        border-top: 1px solid rgba(215, 198, 255, 0.12);
        flex-wrap: wrap;
    }

    .magic-strip-item {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #9D96B8;
    }
    
    .magic-strip-item strong {
        color: #B3E6DB;
        font-weight: 600;
    }

    /* Velvet Glass KPI Cards */
    .magic-card {
        background: linear-gradient(145deg, rgba(35, 23, 51, 0.65) 0%, rgba(23, 19, 35, 0.8) 100%);
        border: 1px solid rgba(215, 198, 255, 0.14);
        border-radius: 18px;
        padding: 24px 26px;
        box-shadow: 
            0 16px 36px -10px rgba(10, 6, 18, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(18px);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
    }

    .magic-card:hover {
        transform: translateY(-3px);
        border-color: rgba(215, 198, 255, 0.35);
        box-shadow: 
            0 24px 48px -12px rgba(10, 6, 18, 0.8),
            0 0 25px rgba(102, 88, 255, 0.15);
    }

    .magic-card-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #A59DBF;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .magic-card-val {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 34px;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }

    /* Maryl-Inspired Badges */
    .pill-lavender {
        display: inline-block;
        background: rgba(215, 198, 255, 0.12);
        border: 1px solid rgba(215, 198, 255, 0.3);
        color: #D7C6FF;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
    }

    .pill-mint {
        display: inline-block;
        background: rgba(179, 230, 219, 0.12);
        border: 1px solid rgba(179, 230, 219, 0.3);
        color: #B3E6DB;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
    }

    .pill-rose {
        display: inline-block;
        background: rgba(251, 113, 133, 0.12);
        border: 1px solid rgba(251, 113, 133, 0.35);
        color: #FB7185;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
    }

    .pill-gold {
        display: inline-block;
        background: rgba(245, 208, 97, 0.12);
        border: 1px solid rgba(245, 208, 97, 0.35);
        color: #F5D061;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Maryl-Style Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(23, 19, 35, 0.75);
        padding: 6px;
        border-radius: 16px;
        border: 1px solid rgba(215, 198, 255, 0.12);
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
        background: rgba(102, 88, 255, 0.25) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: 1px solid rgba(215, 198, 255, 0.35) !important;
        box-shadow: 0 4px 15px rgba(102, 88, 255, 0.2) !important;
    }

    /* Velvet Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #120E1C;
        border-right: 1px solid rgba(215, 198, 255, 0.08);
    }

    /* Maryl-Inspired Contact Card */
    .magic-contact-box {
        background: linear-gradient(135deg, rgba(35, 23, 51, 0.8) 0%, rgba(20, 16, 30, 0.9) 100%);
        border: 1px solid rgba(215, 198, 255, 0.2);
        border-radius: 20px;
        padding: 34px 40px;
        margin-top: 36px;
        box-shadow: 0 20px 50px rgba(10, 6, 18, 0.6);
        position: relative;
    }
    
    .magic-contact-heading {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 32px;
        color: #FFFFFF;
        margin: 6px 0 12px 0;
        line-height: 1.2;
    }

    .magic-contact-heading em {
        font-style: italic;
        color: #D7C6FF;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Velvet & Lavender Plotly Chart Helper
# ---------------------------------------------------------
def style_magic_chart(fig, height=390, title=None):
    """
    Styles Plotly charts with Maryl Gonzalez's velvet night, lavender and mint aesthetic.
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
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 16px 0; border-bottom: 1px solid rgba(215, 198, 255, 0.1); margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 20px; color: #F5D061;">✦</span>
            <div>
                <div style="font-size: 15px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.3px;">RETENTION LAB</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #D7C6FF; letter-spacing: 1px;">DATA &bull; DESIGN &bull; INTENTION</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🌍 Jurisdiction & Demographics")
    geo_list = sorted(raw_df['Geography'].unique().tolist())
    selected_geos = st.multiselect("Geography", geo_list, default=geo_list)

    gender_list = sorted(raw_df['Gender'].unique().tolist())
    selected_genders = st.multiselect("Gender", gender_list, default=gender_list)

    st.markdown("---")
    st.markdown("#### ⚡ Activity & Product Depth")
    activity_filter = st.radio("Activity Status", ["All Accounts", "Active Members Only", "Inactive Members Only"])

    all_prods = sorted(raw_df['NumOfProducts'].unique().tolist())
    selected_prods = st.multiselect("Products Held", all_prods, default=all_prods)

    st.markdown("---")
    st.markdown("#### 💶 Balance & Salary Bounds")
    bal_min, bal_max = int(raw_df['Balance'].min()), int(raw_df['Balance'].max())
    bal_range = st.slider("Ledger Balance (€)", bal_min, bal_max, (bal_min, bal_max), step=5000)

    sal_min, sal_max = int(raw_df['EstimatedSalary'].min()), int(raw_df['EstimatedSalary'].max())
    sal_range = st.slider("Estimated Salary (€)", sal_min, sal_max, (sal_min, sal_max), step=5000)

# Filter application
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

st.sidebar.markdown(f"""
<div style="background: rgba(35, 23, 51, 0.6); border: 1px solid rgba(215, 198, 255, 0.2); border-radius: 12px; padding: 12px; margin-top: 15px;">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #A59DBF;">ACTIVE ANALYTICAL COHORT</div>
    <div style="font-size: 20px; font-weight: 700; color: #B3E6DB; margin-top: 2px;">{len(filtered_df):,} <span style="font-size: 12px; color: #A59DBF;">/ {len(raw_df):,} folios</span></div>
</div>
""", unsafe_allow_html=True)

if len(filtered_df) == 0:
    st.error("No customer records match your filter criteria. Adjust the sidebar sliders.")
    st.stop()

# Compute KPIs
kpis = calculate_portfolio_kpis(filtered_df)

# ---------------------------------------------------------
# Maryl-Style Hero Header
# ---------------------------------------------------------
st.markdown(f"""
<div class="magic-hero">
    <div class="magic-eyebrow">
        <span class="magic-sparkle">✦</span> EQUAL PARTS INTENTION, DATA & A LITTLE <em>MAGIC.</em>
    </div>
    <h1 class="magic-title">
        Customer Engagement & Product Utilization <em>Analytics.</em>
    </h1>
    <p class="magic-tagline">
        Where quantitative banking telemetry meets thoughtful behavioral design: evaluating relationship depth, product retention moats, and silent capital flight across European commercial banking.
    </p>
    <div class="magic-metrics-strip">
        <div class="magic-strip-item">Monitored Liquidity: <strong>€{kpis['total_balance']/1e6:.1f}M</strong></div>
        <div class="magic-strip-item">Portfolio Baseline Attrition: <strong>{kpis['overall_churn_rate']*100:.2f}%</strong></div>
        <div class="magic-strip-item">Active Protection Multiplier: <strong>{kpis['err']:.2f}&times;</strong></div>
        <div class="magic-strip-item">Unengaged VIP Liquidity: <strong>€{kpis['at_risk_balance']/1e6:.1f}M</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Primary Navigation Tabs
# ---------------------------------------------------------
tab_command, tab_3d_hologram, tab_engagement, tab_products, tab_vip_radar, tab_simulation, tab_methodology = st.tabs([
    "✦ Executive Pulse",
    "🪐 3D Relationship Space",
    "⚡ Engagement Dynamics",
    "📦 The 2-Product Golden Zone",
    "🎯 VIP Account Radar",
    "📈 Capital Preservation Engine",
    "📑 Econometric & Policy Tests"
])

# =========================================================
# TAB 1: EXECUTIVE PULSE
# =========================================================
with tab_command:
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.markdown(f"""
        <div class="magic-card">
            <div class="magic-card-eyebrow">Portfolio Churn</div>
            <div class="magic-card-val">{kpis['overall_churn_rate']*100:.2f}%</div>
            <span class="pill-rose">{kpis['churned_customers']:,} exited / {kpis['total_customers']:,}</span>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="magic-card">
            <div class="magic-card-eyebrow">Engagement Ratio</div>
            <div class="magic-card-val">{kpis['err']:.2f}&times;</div>
            <span class="pill-mint">Active: {kpis['active_churn_rate']*100:.1f}% vs Inact: {kpis['inactive_churn_rate']*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="magic-card">
            <div class="magic-card-eyebrow">VIP Disengagement</div>
            <div class="magic-card-val">{kpis['hbdr']*100:.2f}%</div>
            <span class="pill-gold">{kpis['inactive_high_bal_count']:,} VIPs Dormant (>€100k)</span>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="magic-card">
            <div class="magic-card-eyebrow">Capital At Risk</div>
            <div class="magic-card-val">€{kpis['at_risk_balance']/1e6:.1f}M</div>
            <span class="pill-rose">{kpis['pct_balance_at_risk']:.1f}% of total balance</span>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        st.markdown(f"""
        <div class="magic-card">
            <div class="magic-card-eyebrow">Card Stickiness Delta</div>
            <div class="magic-card-val">{kpis['cc_stickiness_delta']:+.2f}%</div>
            <span class="pill-lavender">p = 0.492 (Zero direct lift)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dial & Sunburst Row
    col_g1, col_g2, col_g3 = st.columns([1, 1, 1.2])

    with col_g1:
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=kpis['overall_churn_rate'] * 100,
            title={'text': "<b>Portfolio Churn Risk</b><br><span style='font-size:12px;color:#A59DBF'>Target Benchmark < 12.0%</span>", 'font': {'family': 'Playfair Display, serif', 'color': '#FFFFFF'}},
            delta={'reference': 12.0, 'increasing': {'color': '#FB7185'}, 'decreasing': {'color': '#B3E6DB'}},
            gauge={
                'axis': {'range': [0, 40], 'tickcolor': '#A59DBF'},
                'bar': {'color': '#FB7185'},
                'bgcolor': 'rgba(35, 23, 51, 0.6)',
                'borderwidth': 1,
                'bordercolor': 'rgba(215, 198, 255, 0.2)',
                'steps': [
                    {'range': [0, 12], 'color': 'rgba(179, 230, 219, 0.25)'},
                    {'range': [12, 25], 'color': 'rgba(245, 208, 97, 0.25)'},
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
# TAB 2: 3D RELATIONSHIP SPACE (VELVET NOIR 3D)
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

    # 3D layout without duplicate keyword argument
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

    st.markdown("""
    <div style="background: rgba(35, 23, 51, 0.7); border: 1px solid rgba(215, 198, 255, 0.2); border-radius: 14px; padding: 16px 22px;">
        <span style="color: #D7C6FF; font-weight: 700;">✦ Spatial Behavioral Insight:</span> 
        Observe the distinct cluster of <span style="color: #FB7185; font-weight: 700;">rose churned points</span> concentrated at 
        <strong>Age 45–60 with balances above €100,000 and 1 or 3+ products</strong>. 
        This cluster accounts for over sixty percent of total retail capital flight.
    </div>
    """, unsafe_allow_html=True)

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
        # Bar Trace: Customer Volume
        fig_golden.add_trace(go.Bar(
            x=prod_kpis['NumOfProducts'],
            y=prod_kpis['Count'],
            name='Customer Accounts',
            yaxis='y',
            marker_color='rgba(215, 198, 255, 0.35)',
            marker_line=dict(color='#D7C6FF', width=1.5)
        ))
        # Line Trace: Churn Rate
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

        # Explicit Layout Settings without conflicting unpacked dictionaries
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

    # 3 Analytical Findings Cards
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        <div class="magic-card">
            <div class="magic-card-eyebrow">1 PRODUCT VULNERABILITY</div>
            <div style="font-family: 'Playfair Display', serif; font-size: 26px; font-weight: 600; color: #FB7185; margin: 4px 0;">27.71% Churn</div>
            <p style="font-size: 13px; color: #C9C2E0; margin: 0;">50.8% of customers hold no secondary anchors, resulting in near-zero switching resistance.</p>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <div class="magic-card" style="border-color: rgba(179, 230, 219, 0.4); box-shadow: 0 0 25px rgba(179, 230, 219, 0.12);">
            <div class="magic-card-eyebrow" style="color: #B3E6DB;">THE 2-PRODUCT GOLDEN ZONE</div>
            <div style="font-family: 'Playfair Display', serif; font-size: 26px; font-weight: 600; color: #B3E6DB; margin: 4px 0;">7.58% Churn (&minus;72.6%)</div>
            <p style="font-size: 13px; color: #C9C2E0; margin: 0;">The mathematical optimum: pairing checking with savings or mortgage maximizes loyalty.</p>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown("""
        <div class="magic-card" style="border-color: rgba(251, 113, 133, 0.35);">
            <div class="magic-card-eyebrow" style="color: #FB7185;">MULTI-PRODUCT HOCKEY STICK</div>
            <div style="font-family: 'Playfair Display', serif; font-size: 26px; font-weight: 600; color: #FB7185; margin: 4px 0;">82.7% to 100% Churn</div>
            <p style="font-size: 13px; color: #C9C2E0; margin: 0;">Forced bundling beyond 2 products triggers customer alienation, fee fatigue, and immediate flight.</p>
        </div>
        """, unsafe_allow_html=True)

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

    # Individual Radar Dossier
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
            st.markdown(f"""
            <div class="magic-card">
                <div class="magic-card-eyebrow">CUSTOMER PROFILE #{cust_record['CustomerId']}</div>
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
                    &bull; <strong>Activity Status:</strong> {'Active Member' if cust_record['IsActiveMember']==1 else 'Inactive / Dormant'}<br>
                    &bull; <strong>Credit Card Held:</strong> {'Yes' if cust_record['HasCrCard']==1 else 'No'}
                </div>
                <div style="margin-top: 14px; font-size: 12px; color: #FB7185; background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.3); padding: 10px; border-radius: 8px;">
                    ⚠️ <strong>Recommended Action:</strong> Proactively reach out with personalized wealth advisory. Offer 2-product fixed term or automated savings sweeps to avoid capital drain.
                </div>
            </div>
            """, unsafe_allow_html=True)

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
            st.markdown(f"""
            <div class="magic-card">
                <div class="magic-card-eyebrow">Projected Portfolio Churn</div>
                <div class="magic-card-val">{sim_res['simulated_churn_rate']*100:.2f}%</div>
                <span class="pill-mint">Down from {sim_res['baseline_churn_rate']*100:.2f}% (&minus;{sim_res['churn_rate_reduction_bps']:.0f} bps)</span>
            </div>
            """, unsafe_allow_html=True)
        with o2:
            st.markdown(f"""
            <div class="magic-card">
                <div class="magic-card-eyebrow">Customer Accounts Retained</div>
                <div class="magic-card-val">{sim_res['total_customers_saved']:,}</div>
                <span class="pill-lavender">Accounts Prevented from Exiting</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        o3, o4 = st.columns(2)
        with o3:
            st.markdown(f"""
            <div class="magic-card">
                <div class="magic-card-eyebrow">Deposit Capital Preserved</div>
                <div class="magic-card-val">€{sim_res['total_capital_preserved']/1e6:.2f}M</div>
                <span class="pill-mint">Protected from Liquidity Run-Off</span>
            </div>
            """, unsafe_allow_html=True)
        with o4:
            st.markdown(f"""
            <div class="magic-card">
                <div class="magic-card-eyebrow">Annual Net Interest Margin Protected</div>
                <div class="magic-card-val">€{sim_res['annual_revenue_protected']/1e6:.2f}M</div>
                <span class="pill-mint">At {sim_nim:.2f}% Annual Spread</span>
            </div>
            """, unsafe_allow_html=True)

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
# Maryl Gonzalez-Inspired Contact & Inquiry Card
# ---------------------------------------------------------
st.markdown("""
<div class="magic-contact-box" id="contact">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #D7C6FF; letter-spacing: 2px; text-transform: uppercase;">
        <span style="color: #F5D061;">✦</span> HAVE AN INQUIRY OR RETENTION MANDATE?
    </div>
    <div style="font-size: 13px; color: #A59DBF; margin-top: 4px; letter-spacing: 0.5px;">
        GOOD THINGS START WITH A CONVERSATION.
    </div>
    <div class="magic-contact-heading">
        Let’s make customer retention <em>magic.</em>
    </div>
    <div style="display: flex; gap: 20px; align-items: center; margin-top: 18px; flex-wrap: wrap;">
        <a href="mailto:aryan.analytics@example.com" style="color: #FFFFFF; font-size: 16px; font-weight: 600; text-decoration: none; border-bottom: 1px solid #D7C6FF; padding-bottom: 2px;">
            aryan.analytics@example.com ↗
        </a>
        <span style="color: #A59DBF; font-size: 13px;">&bull;</span>
        <span style="color: #D7C6FF; font-family: 'JetBrains Mono', monospace; font-size: 12px;">Unified Mentor Banking Analytics Fellow</span>
        <span style="color: #A59DBF; font-size: 13px;">&bull;</span>
        <span style="color: #B3E6DB; font-family: 'JetBrains Mono', monospace; font-size: 12px;">European Central Bank Case Framework</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #A59DBF; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 10px 0;">
    RETENTION INTELLIGENCE &bull; DESIGN, DATA & A LITTLE MAGIC &bull; DEVELOPED BY ARYAN
</div>
""", unsafe_allow_html=True)
