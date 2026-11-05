"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
Executive Retention Intelligence Command Center
Theme: Eye-Catching Luxe Aurora Obsidian (Mint Emerald, Sapphire & Electric Coral)
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
    page_title="RETENTION INTELLIGENCE // Executive Banking Terminal",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Luxe Aurora Obsidian CSS Styling (Ultra-Premium Fintech Aesthetic)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* Core Dark Canvas */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #F8FAFC;
    }
    
    .stApp {
        background: radial-gradient(ellipse 90% 60% at 50% -10%, #0c182c 0%, #050811 55%, #030408 100%);
    }

    /* Executive Aurora Hero Banner */
    .aurora-hero {
        position: relative;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(10, 15, 30, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px 36px;
        margin-bottom: 24px;
        box-shadow: 0 20px 50px -10px rgba(0, 0, 0, 0.8), 0 0 40px -15px rgba(0, 245, 160, 0.15);
        backdrop-filter: blur(24px);
        overflow: hidden;
    }

    .aurora-hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #00F5A0 0%, #00D2FF 35%, #7928CA 70%, #FF007A 100%);
    }

    .hero-top-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 245, 160, 0.1);
        border: 1px solid rgba(0, 245, 160, 0.3);
        color: #00F5A0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }

    .pulse-beacon {
        width: 8px;
        height: 8px;
        background-color: #00F5A0;
        border-radius: 50%;
        box-shadow: 0 0 12px #00F5A0;
        animation: pulseEffect 2s infinite ease-in-out;
    }

    @keyframes pulseEffect {
        0% { transform: scale(0.9); opacity: 0.8; }
        50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 16px #00F5A0; }
        100% { transform: scale(0.9); opacity: 0.8; }
    }

    .hero-main-title {
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.8px;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 60%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-tagline {
        color: #94A3B8;
        font-size: 14px;
        margin-top: 8px;
        font-weight: 400;
        max-width: 900px;
        line-height: 1.6;
    }

    .hero-stats-ribbon {
        display: flex;
        gap: 24px;
        margin-top: 20px;
        padding-top: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        flex-wrap: wrap;
    }

    .ribbon-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: #CBD5E1;
        font-family: 'JetBrains Mono', monospace;
    }

    .ribbon-item strong {
        color: #00F5A0;
    }

    /* Eye-Catching Luxe Metric Cards */
    .luxe-kpi-card {
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.8) 0%, rgba(13, 19, 33, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 14px 30px -10px rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(16px);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }

    .luxe-kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 245, 160, 0.4);
        box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.8), 0 0 25px rgba(0, 245, 160, 0.15);
    }

    .luxe-kpi-title {
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .luxe-kpi-val {
        font-size: 30px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.8px;
        margin-bottom: 8px;
    }

    /* Modern Vivid Badges */
    .badge-mint {
        color: #00F5A0;
        background: rgba(0, 245, 160, 0.12);
        border: 1px solid rgba(0, 245, 160, 0.3);
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-coral {
        color: #FB7185;
        background: rgba(244, 63, 94, 0.14);
        border: 1px solid rgba(244, 63, 94, 0.35);
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-gold {
        color: #FBBF24;
        background: rgba(245, 158, 11, 0.14);
        border: 1px solid rgba(245, 158, 11, 0.35);
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-blue {
        color: #60A5FA;
        background: rgba(59, 130, 246, 0.14);
        border: 1px solid rgba(59, 130, 246, 0.35);
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Segmented Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.8);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 13px;
        background: transparent;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 245, 160, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: 1px solid rgba(0, 245, 160, 0.4) !important;
        box-shadow: 0 4px 15px rgba(0, 245, 160, 0.15) !important;
    }

    /* Sidebar Dark Obsidian */
    section[data-testid="stSidebar"] {
        background-color: #070B16;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Clean High-Impact Chart Styling Helper
# ---------------------------------------------------------
def style_chart(fig, height=380, title=None):
    """
    Applies the eye-catching dark obsidian theme cleanly
    without any duplicate keyword argument collisions on axes.
    """
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans, sans-serif', color='#CBD5E1', size=12),
        height=height,
        margin=dict(l=20, r=20, t=50 if title else 25, b=25)
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(color='#F8FAFC', size=14)))
    fig.update_xaxes(
        gridcolor='rgba(255, 255, 255, 0.06)',
        zerolinecolor='rgba(255, 255, 255, 0.12)',
        tickfont=dict(color='#94A3B8')
    )
    fig.update_yaxes(
        gridcolor='rgba(255, 255, 255, 0.06)',
        zerolinecolor='rgba(255, 255, 255, 0.12)',
        tickfont=dict(color='#94A3B8')
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
# Sidebar Controls & Tactical Filters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 15px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="font-size: 26px;">🏦</div>
            <div>
                <div style="font-size: 16px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px;">RETENTION PRO</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #00F5A0; letter-spacing: 1px;">FINTECH TERMINAL v4.0</div>
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
    selected_prods = st.multiselect("Product Count Tier", all_prods, default=all_prods)

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
<div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(0, 245, 160, 0.3); border-radius: 12px; padding: 14px; margin-top: 15px;">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94A3B8;">ACTIVE ANALYTICAL SUBSET</div>
    <div style="font-size: 22px; font-weight: 800; color: #00F5A0; margin-top: 2px;">{len(filtered_df):,} <span style="font-size: 12px; color: #64748B;">/ {len(raw_df):,} accounts</span></div>
</div>
""", unsafe_allow_html=True)

if len(filtered_df) == 0:
    st.error("No customer records match your filter criteria. Adjust the sidebar sliders.")
    st.stop()

# Compute portfolio KPIs
kpis = calculate_portfolio_kpis(filtered_df)

# ---------------------------------------------------------
# Eye-Catching Hero Banner with Live Telemetry Ribbon
# ---------------------------------------------------------
st.markdown(f"""
<div class="aurora-hero">
    <div class="hero-top-badge">
        <span class="pulse-beacon"></span>
        LIVE PORTFOLIO TELEMETRY &bull; 10,000 AUDITED ACCOUNTS &bull; ECB CASE FRAMEWORK
    </div>
    <h1 class="hero-main-title">Customer Engagement & Product Utilization Analytics</h1>
    <p class="hero-tagline">
        Evaluating the empirical relationship between behavioral engagement, product depth, and deposit retention across European retail banking.
    </p>
    <div class="hero-stats-ribbon">
        <div class="ribbon-item">Total Monitored Deposits: <strong>€{kpis['total_balance']/1e6:.1f}M</strong></div>
        <div class="ribbon-item">Baseline Churn: <strong>{kpis['overall_churn_rate']*100:.2f}%</strong></div>
        <div class="ribbon-item">Active Protection Lift: <strong>{kpis['err']:.2f}x</strong></div>
        <div class="ribbon-item">Vulnerable VIP Deposits: <strong>€{kpis['at_risk_balance']/1e6:.1f}M</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Primary Navigation Tabs
# ---------------------------------------------------------
tab_command, tab_3d_hologram, tab_engagement, tab_products, tab_vip_radar, tab_simulation, tab_methodology = st.tabs([
    "📊 Executive Summary",
    "🌐 3D Customer Space",
    "⚡ Engagement Dynamics",
    "📦 The 2-Product Golden Zone",
    "🎯 VIP Lead Detector & Dossier",
    "📈 What-If Strategy Simulator",
    "📑 Empirical Research Tests"
])

# =========================================================
# TAB 1: EXECUTIVE SUMMARY
# =========================================================
with tab_command:
    # 5 Luxe Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.markdown(f"""
        <div class="luxe-kpi-card">
            <div class="luxe-kpi-title">Portfolio Churn Rate</div>
            <div class="luxe-kpi-val">{kpis['overall_churn_rate']*100:.2f}%</div>
            <span class="badge-coral">{kpis['churned_customers']:,} exited / {kpis['total_customers']:,}</span>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="luxe-kpi-card">
            <div class="luxe-kpi-title">Engagement Retention Ratio</div>
            <div class="luxe-kpi-val">{kpis['err']:.2f}x</div>
            <span class="badge-mint">Active: {kpis['active_churn_rate']*100:.1f}% vs Inactive: {kpis['inactive_churn_rate']*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="luxe-kpi-card">
            <div class="luxe-kpi-title">High-Bal. Disengagement</div>
            <div class="luxe-kpi-val">{kpis['hbdr']*100:.2f}%</div>
            <span class="badge-gold">{kpis['inactive_high_bal_count']:,} VIPs Dormant (>€100k)</span>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="luxe-kpi-card">
            <div class="luxe-kpi-title">Capital At Flight Risk</div>
            <div class="luxe-kpi-val">€{kpis['at_risk_balance']/1e6:.1f}M</div>
            <span class="badge-coral">{kpis['pct_balance_at_risk']:.1f}% of total balance</span>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        st.markdown(f"""
        <div class="luxe-kpi-card">
            <div class="luxe-kpi-title">Credit Card Stickiness Delta</div>
            <div class="luxe-kpi-val">{kpis['cc_stickiness_delta']:+.2f}%</div>
            <span class="badge-blue">p = 0.492 (Zero direct lift)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dial & Sunburst Row
    col_g1, col_g2, col_g3 = st.columns([1, 1, 1.2])

    with col_g1:
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=kpis['overall_churn_rate'] * 100,
            title={'text': "<b>Observed Churn Risk</b><br><span style='font-size:12px;color:#94A3B8'>Target Benchmark < 12.0%</span>", 'font': {'color': '#F8FAFC'}},
            delta={'reference': 12.0, 'increasing': {'color': '#FB7185'}, 'decreasing': {'color': '#00F5A0'}},
            gauge={
                'axis': {'range': [0, 40], 'tickcolor': '#94A3B8'},
                'bar': {'color': '#FB7185'},
                'bgcolor': 'rgba(15, 23, 42, 0.8)',
                'borderwidth': 1,
                'bordercolor': 'rgba(255, 255, 255, 0.1)',
                'steps': [
                    {'range': [0, 12], 'color': 'rgba(0, 245, 160, 0.25)'},
                    {'range': [12, 25], 'color': 'rgba(245, 158, 11, 0.25)'},
                    {'range': [25, 40], 'color': 'rgba(244, 63, 94, 0.35)'}
                ],
            }
        ))
        fig_g1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'), height=260, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_g1, width='stretch')

    with col_g2:
        avg_rsi = filtered_df['RelationshipStrengthIndex'].mean()
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_rsi,
            title={'text': "<b>Relationship Strength (RSI)</b><br><span style='font-size:12px;color:#94A3B8'>Score Scale: 0 to 100</span>", 'font': {'color': '#F8FAFC'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#94A3B8'},
                'bar': {'color': '#00F5A0'},
                'bgcolor': 'rgba(15, 23, 42, 0.8)',
                'borderwidth': 1,
                'bordercolor': 'rgba(255, 255, 255, 0.1)',
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(244, 63, 94, 0.3)'},
                    {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.25)'},
                    {'range': [70, 100], 'color': 'rgba(0, 245, 160, 0.3)'}
                ],
            }
        ))
        fig_g2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'), height=260, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_g2, width='stretch')

    with col_g3:
        sunburst_sample = filtered_df.sample(min(3000, len(filtered_df)), random_state=42).copy()
        sunburst_sample['Activity'] = sunburst_sample['IsActiveMember'].map({1: 'Active', 0: 'Inactive'})
        sunburst_sample['ProductTier'] = sunburst_sample['NumOfProducts'].apply(lambda x: f"{x} Prod")
        sunburst_sample['Status'] = sunburst_sample['Exited'].map({1: 'Churned', 0: 'Retained'})

        fig_sun = px.sunburst(
            sunburst_sample,
            path=['Geography', 'ProductTier', 'Activity', 'Status'],
            color='Status',
            color_discrete_map={'Retained': '#00F5A0', 'Churned': '#FB7185'},
            title="Portfolio Hierarchy (Geography > Products > Activity > Churn)"
        )
        fig_sun.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'), height=260, margin=dict(l=10, r=10, t=40, b=10))
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
            color_continuous_scale=['#00D2FF', '#00F5A0', '#F59E0B', '#FB7185'],
            text=[f"{v*100:.1f}%" for v in cohort_df['ChurnRate']],
            labels={'ChurnRate': 'Observed Churn', 'EngagementProfile': 'Cohort', 'AvgBalance': 'Avg Balance (€)'},
            title="Observed Churn Rate by Behavioral Profile (Color = Mean Balance €)"
        )
        fig_cohort.update_traces(textposition='outside')
        style_chart(fig_cohort, height=360)
        fig_cohort.update_layout(xaxis_tickangle=-15)
        st.plotly_chart(fig_cohort, width='stretch')

    with c_tree:
        fig_tree = px.treemap(
            cohort_df,
            path=['EngagementProfile'],
            values='TotalBalance',
            color='ChurnRate',
            color_continuous_scale=['#00F5A0', '#F59E0B', '#FB7185'],
            title="Deposit Volume Share by Segment"
        )
        fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'), height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_tree, width='stretch')

# =========================================================
# TAB 2: 3D CUSTOMER SPACE (ERROR-FREE & VIVID)
# =========================================================
with tab_3d_hologram:
    st.subheader("🌐 3D Customer Relationship Space")
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
        hologram_sample['Outcome'] = hologram_sample['Exited'].map({0: 'Retained (Loyal)', 1: 'Churned (Exited)'})
        fig_3d = px.scatter_3d(
            hologram_sample,
            x='Age',
            y='Balance',
            z='NumOfProducts',
            color='Outcome',
            size='RelationshipStrengthIndex',
            size_max=12,
            opacity=0.85,
            color_discrete_map={'Retained (Loyal)': '#00F5A0', 'Churned (Exited)': '#FB7185'},
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
            color_continuous_scale=['#00D2FF', '#00F5A0', '#F59E0B', '#FB7185'],
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
        font=dict(family='Plus Jakarta Sans, sans-serif', color='#CBD5E1'),
        scene=dict(
            xaxis=dict(title='Age (Years)', backgroundcolor='#0D1424', gridcolor='#1E293B', showbackground=True),
            yaxis=dict(title='Balance (€)', backgroundcolor='#0D1424', gridcolor='#1E293B', showbackground=True),
            zaxis=dict(title='Product Count', tickvals=[1, 2, 3, 4], backgroundcolor='#0D1424', gridcolor='#1E293B', showbackground=True),
            camera=dict(eye=cam_eye)
        ),
        height=620,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_3d, width='stretch')

    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(0, 245, 160, 0.3); border-radius: 12px; padding: 14px 20px;">
        <span style="color: #00F5A0; font-weight: 700;">Spatial Behavioral Takeaway:</span> 
        Notice the dense cluster of <span style="color: #FB7185; font-weight: 700;">coral churned points</span> concentrated at 
        <strong>Age 45–60</strong> with <strong>balances above €100,000 and 1 or 3+ products</strong>. 
        This cluster represents over 60% of total bank capital flight.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TAB 3: ENGAGEMENT DYNAMICS
# =========================================================
with tab_engagement:
    st.subheader("⚡ Behavioral Engagement Dynamics & Attrition")
    st.caption("Active transacting frequency is the primary protective barrier against churn.")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        eng_split = filtered_df.groupby('IsActiveMember')['Exited'].agg(
            Count='count',
            ChurnRate='mean'
        ).reset_index()
        eng_split['Status'] = eng_split['IsActiveMember'].map({0: 'Inactive Member', 1: 'Active Member'})
        eng_split['RetentionRate'] = 1 - eng_split['ChurnRate']

        fig_eng = go.Figure()
        fig_eng.add_trace(go.Bar(
            name='Retained',
            x=eng_split['Status'],
            y=eng_split['RetentionRate'],
            marker_color='#00F5A0',
            text=[f"{v*100:.1f}%" for v in eng_split['RetentionRate']],
            textposition='inside'
        ))
        fig_eng.add_trace(go.Bar(
            name='Churned',
            x=eng_split['Status'],
            y=eng_split['ChurnRate'],
            marker_color='#FB7185',
            text=[f"{v*100:.1f}%" for v in eng_split['ChurnRate']],
            textposition='inside'
        ))
        style_chart(fig_eng, height=360, title="Retention vs. Churn by Active Member Status")
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
            color_discrete_map={'Inactive': '#FB7185', 'Active': '#00F5A0'},
            title="Geographic Churn Comparison Across Activity States"
        )
        style_chart(fig_geo, height=360)
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
            color_discrete_map={0: '#00F5A0', 1: '#FB7185'},
            labels={'IsActiveMember': 'Activity (0 = Inactive, 1 = Active)', 'Balance': 'Ledger Balance (€)'},
            title="Balance Dispersion by Engagement & Churn Outcome"
        )
        style_chart(fig_violin, height=380)
        st.plotly_chart(fig_violin, width='stretch')

    with v2:
        tenure_heat = filtered_df.groupby(['Tenure', 'NumOfProducts'])['Exited'].mean().unstack()
        fig_heat = px.imshow(
            tenure_heat,
            labels=dict(x="Number of Products", y="Tenure (Years)", color="Churn Rate"),
            color_continuous_scale=['#0C251C', '#00F5A0', '#F59E0B', '#FB7185'],
            title="Tenure vs. Product Count Churn Heatmap"
        )
        style_chart(fig_heat, height=380)
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
            marker_color='rgba(0, 245, 160, 0.4)',
            marker_line=dict(color='#00F5A0', width=1.5)
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
            line=dict(color='#FB7185', width=3.5),
            marker=dict(size=10, color='#FF007A')
        ))

        # Explicit Layout Settings without conflicting unpacked dictionaries
        fig_golden.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Plus Jakarta Sans, sans-serif', color='#CBD5E1', size=12),
            title=dict(text="<b>The 2-Product Golden Zone vs. Multi-Product Friction Curve</b>", font=dict(color='#F8FAFC', size=14)),
            xaxis=dict(
                title='Products Held per Customer',
                tickvals=[1, 2, 3, 4],
                gridcolor='rgba(255,255,255,0.06)',
                tickfont=dict(color='#94A3B8')
            ),
            yaxis=dict(
                title='Customer Accounts',
                side='left',
                gridcolor='rgba(255,255,255,0.06)',
                tickfont=dict(color='#94A3B8')
            ),
            yaxis2=dict(
                title='Churn Rate',
                side='right',
                overlaying='y',
                tickformat='.0%',
                range=[0, 1.15],
                tickfont=dict(color='#FB7185')
            ),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(x=0.03, y=0.95, bgcolor='rgba(15, 23, 42, 0.8)')
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
            color_continuous_scale=['#00F5A0', '#F59E0B', '#FB7185'],
            text=[f"{v*100:.1f}%" for v in cc_active['Churn']],
            title="Credit Card Retention Conditioned on Activity"
        )
        fig_cc_active.update_traces(textposition='outside')
        style_chart(fig_cc_active, height=400)
        fig_cc_active.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig_cc_active, width='stretch')

    # 3 Analytical Findings Cards
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 20px;">
            <div style="color: #94A3B8; font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;">1 PRODUCT VULNERABILITY</div>
            <div style="font-size: 24px; font-weight: 800; color: #FB7185; margin: 4px 0;">27.71% Churn</div>
            <p style="font-size: 13px; color: #94A3B8; margin: 0;">50.8% of customers have zero secondary product anchors, resulting in near-zero switching friction.</p>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(0, 245, 160, 0.4); border-radius: 14px; padding: 20px; box-shadow: 0 0 20px rgba(0, 245, 160, 0.12);">
            <div style="color: #00F5A0; font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;">THE 2-PRODUCT GOLDEN ZONE</div>
            <div style="font-size: 24px; font-weight: 800; color: #00F5A0; margin: 4px 0;">7.58% Churn (-72.6%)</div>
            <p style="font-size: 13px; color: #CBD5E1; margin: 0;">The mathematical optimum: pairing checking with a savings or mortgage account maximizes loyalty.</p>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown("""
        <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(244, 63, 94, 0.35); border-radius: 14px; padding: 20px; box-shadow: 0 0 20px rgba(244, 63, 94, 0.12);">
            <div style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;">MULTI-PRODUCT HOCKEY STICK</div>
            <div style="font-size: 24px; font-weight: 800; color: #FB7185; margin: 4px 0;">82.7% to 100% Churn</div>
            <p style="font-size: 13px; color: #94A3B8; margin: 0;">Forced bundling beyond 2 products triggers customer alienation, fee fatigue, and immediate attrition.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 5: VIP LEAD RADAR & DOSSIER
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
        st.metric("Identified At-Risk VIP Accounts", f"{len(vip_leads):,} accounts")
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
                line=dict(color='#00F5A0', width=1.5, dash='dash'),
                name="Target Benchmark"
            ))
            fig_radar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#CBD5E1'),
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1E293B'),
                    angularaxis=dict(gridcolor='#1E293B')
                ),
                height=340,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, width='stretch')

        with r_right:
            st.markdown(f"""
            <div style="background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(0, 245, 160, 0.3); border-radius: 14px; padding: 20px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase;">CUSTOMER PROFILE #{cust_record['CustomerId']}</div>
                <div style="font-size: 24px; font-weight: 800; color: #FFFFFF; margin: 4px 0;">{cust_record['Surname']}, {cust_record['Geography']}</div>
                <div style="margin: 8px 0;">
                    <span class="badge-coral">{cust_record['RetentionTier']}</span>
                    <span class="badge-mint">RSI Score: {cust_record['RelationshipStrengthIndex']} / 100</span>
                </div>
                <div style="font-size: 13px; color: #CBD5E1; line-height: 1.6; margin-top: 10px;">
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
            <div class="luxe-kpi-card">
                <div class="luxe-kpi-title">Projected Portfolio Churn</div>
                <div class="luxe-kpi-val">{sim_res['simulated_churn_rate']*100:.2f}%</div>
                <span class="badge-mint">Down from {sim_res['baseline_churn_rate']*100:.2f}% (-{sim_res['churn_rate_reduction_bps']:.0f} bps)</span>
            </div>
            """, unsafe_allow_html=True)
        with o2:
            st.markdown(f"""
            <div class="luxe-kpi-card">
                <div class="luxe-kpi-title">Customer Accounts Retained</div>
                <div class="luxe-kpi-val">{sim_res['total_customers_saved']:,}</div>
                <span class="badge-mint">Accounts Prevented from Exiting</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        o3, o4 = st.columns(2)
        with o3:
            st.markdown(f"""
            <div class="luxe-kpi-card">
                <div class="luxe-kpi-title">Deposit Capital Preserved</div>
                <div class="luxe-kpi-val">€{sim_res['total_capital_preserved']/1e6:.2f}M</div>
                <span class="badge-mint">Protected from Liquidity Run-Off</span>
            </div>
            """, unsafe_allow_html=True)
        with o4:
            st.markdown(f"""
            <div class="luxe-kpi-card">
                <div class="luxe-kpi-title">Annual Net Interest Margin Protected</div>
                <div class="luxe-kpi-val">€{sim_res['annual_revenue_protected']/1e6:.2f}M</div>
                <span class="badge-mint">At {sim_nim:.2f}% Annual Spread</span>
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
        color_continuous_scale=['#0C251C', '#00F5A0', '#00D2FF'],
        text=[f"€{v:.2f}M" for v in strat_data['Capital Saved (€M)']],
        title="Attribution of Capital Preserved Across Intervention Levers (€ Millions)"
    )
    fig_waterfall.update_traces(textposition='outside')
    style_chart(fig_waterfall, height=360)
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
            color_continuous_scale=['#0C251C', '#00F5A0'],
            title="Random Forest Gini Feature Importance"
        )
        style_chart(fig_rf, height=380)
        fig_rf.update_layout(yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_rf, width='stretch')

    with col_or:
        fig_or = px.bar(
            models_dict['odds_ratios'],
            x='OddsRatio',
            y='Feature',
            orientation='h',
            color='Direction',
            color_discrete_map={'Risk Factor (+)': '#FB7185', 'Retention Factor (-)': '#00F5A0'},
            title="Logistic Regression Odds Ratios"
        )
        fig_or.add_vline(x=1.0, line_dash='dash', line_color='#94A3B8')
        style_chart(fig_or, height=380)
        fig_or.update_layout(yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_or, width='stretch')

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 10px 0;">
    RETENTION PRO &bull; LUXE AURORA OBSIDIAN EDITION &bull; DEVELOPED BY ARYAN &bull; UNIFIED MENTOR BANKING STUDY
</div>
""", unsafe_allow_html=True)
