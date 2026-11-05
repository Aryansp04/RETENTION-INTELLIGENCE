"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
Cinematic 3D Executive Command Center & Retention Strategy Platform
Color Palette: Seal Blue (#0EA5E9), Deep Navy Blue (#0B1120 / #1E3A8A), and Crimson Pink (#F43F5E / #E11D48)
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
    page_title="RETENTION INTELLIGENCE | 3D Banking Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Cinematic Dark Theme & 3D Glassmorphism CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    /* Global Root Variables */
    :root {
        --bg-void: #060913;
        --bg-navy-dark: #0A0F1D;
        --bg-card: rgba(13, 21, 39, 0.75);
        --seal-blue-neon: #38BDF8;
        --seal-blue-core: #0EA5E9;
        --seal-blue-deep: #0284C7;
        --navy-accent: #1E3A8A;
        --crimson-pink-neon: #FB7185;
        --crimson-pink-core: #F43F5E;
        --crimson-pink-deep: #BE123C;
        --border-glass: rgba(56, 189, 248, 0.15);
        --text-bright: #F8FAFC;
        --text-muted: #94A3B8;
    }

    /* Core Page Styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-bright);
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0c152e 0%, #060913 60%, #03050a 100%);
    }

    /* Cinematic Holographic Header */
    .cinematic-header {
        position: relative;
        background: linear-gradient(135deg, rgba(10, 15, 29, 0.9) 0%, rgba(20, 30, 55, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 40px -10px rgba(14, 165, 233, 0.25), 0 0 20px rgba(244, 63, 94, 0.12);
        backdrop-filter: blur(20px);
        overflow: hidden;
    }
    
    .cinematic-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #38BDF8 0%, #0EA5E9 35%, #F43F5E 70%, #FB7185 100%);
    }

    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: var(--seal-blue-neon);
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 5px 12px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: var(--crimson-pink-core);
        border-radius: 50%;
        box-shadow: 0 0 10px var(--crimson-pink-neon);
        animation: pulseAnimation 2s infinite ease-in-out;
    }

    @keyframes pulseAnimation {
        0% { transform: scale(0.9); opacity: 0.7; }
        50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 14px #FB7185; }
        100% { transform: scale(0.9); opacity: 0.7; }
    }

    .header-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFFFFF 0%, #BAE6FD 50%, #FDA4AF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .header-subtitle {
        color: var(--text-muted);
        font-size: 14px;
        margin-top: 6px;
        font-weight: 400;
    }

    /* 3D Glassmorphic Metric Cards */
    .kpi-3d-card {
        background: rgba(13, 21, 39, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 16px;
        padding: 20px 22px;
        backdrop-filter: blur(14px);
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    
    .kpi-3d-card:hover {
        transform: translateY(-4px) scale(1.01);
        border-color: rgba(244, 63, 94, 0.4);
        box-shadow: 0 20px 40px -10px rgba(14, 165, 233, 0.3), 0 0 25px rgba(244, 63, 94, 0.25);
    }

    .kpi-3d-title {
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .kpi-3d-val {
        font-size: 28px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }

    .kpi-tag-blue {
        color: #38BDF8;
        font-size: 12px;
        font-weight: 600;
        background: rgba(14, 165, 233, 0.15);
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        display: inline-block;
    }

    .kpi-tag-pink {
        color: #FB7185;
        font-size: 12px;
        font-weight: 600;
        background: rgba(244, 63, 94, 0.15);
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid rgba(244, 63, 94, 0.3);
        display: inline-block;
    }

    /* Section Panels */
    .panel-3d {
        background: rgba(11, 18, 33, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(16px);
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(10, 15, 29, 0.6);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(56, 189, 248, 0.12);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 18px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 13px;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.25) 0%, rgba(244, 63, 94, 0.25) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.2);
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #060913;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #0EA5E9;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Custom Plotly 3D & 2D Unified Theme
# ---------------------------------------------------------
PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans, sans-serif', color='#94A3B8'),
    xaxis=dict(gridcolor='rgba(56, 189, 248, 0.08)', zerolinecolor='rgba(56, 189, 248, 0.15)'),
    yaxis=dict(gridcolor='rgba(56, 189, 248, 0.08)', zerolinecolor='rgba(56, 189, 248, 0.15)'),
    scene=dict(
        xaxis=dict(backgroundcolor='rgba(10, 15, 29, 0.6)', gridcolor='rgba(56, 189, 248, 0.15)', showbackground=True),
        yaxis=dict(backgroundcolor='rgba(10, 15, 29, 0.6)', gridcolor='rgba(56, 189, 248, 0.15)', showbackground=True),
        zaxis=dict(backgroundcolor='rgba(10, 15, 29, 0.6)', gridcolor='rgba(56, 189, 248, 0.15)', showbackground=True),
    )
)

# ---------------------------------------------------------
# Data Ingestion & Validation
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
    <div style="text-align: center; padding: 15px 0 20px 0;">
        <div style="font-size: 32px; filter: drop-shadow(0 0 15px #38BDF8);">🏦</div>
        <div style="font-size: 16px; font-weight: 800; letter-spacing: -0.5px; margin-top: 6px; color: #F8FAFC;">
            RETENTION COMMAND
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #38BDF8; letter-spacing: 1px;">
            v3.2 // HIGH-PRECISION HUD
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Tactical Segmentation")
    
    geo_list = sorted(raw_df['Geography'].unique().tolist())
    selected_geos = st.multiselect("Eurozone Jurisdiction", geo_list, default=geo_list)

    gender_list = sorted(raw_df['Gender'].unique().tolist())
    selected_genders = st.multiselect("Gender Demographic", gender_list, default=gender_list)

    st.markdown("---")
    st.markdown("### ⚡ Activity & Product Depth")
    activity_filter = st.radio("Activity State", ["All Accounts", "Active Members (1)", "Inactive Members (0)"])

    all_prods = sorted(raw_df['NumOfProducts'].unique().tolist())
    selected_prods = st.multiselect("Product Count Tier", all_prods, default=all_prods)

    st.markdown("---")
    st.markdown("### 💰 Financial Exposure Limits")
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

if activity_filter == "Active Members (1)":
    filtered_df = filtered_df[filtered_df['IsActiveMember'] == 1]
elif activity_filter == "Inactive Members (0)":
    filtered_df = filtered_df[filtered_df['IsActiveMember'] == 0]

st.sidebar.markdown(f"""
<div style="background: rgba(14, 165, 233, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 12px; margin-top: 15px;">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94A3B8;">ACTIVE SUBSET</div>
    <div style="font-size: 20px; font-weight: 800; color: #38BDF8;">{len(filtered_df):,} <span style="font-size: 12px; color: #64748B;">/ {len(raw_df):,} accounts</span></div>
</div>
""", unsafe_allow_html=True)

if len(filtered_df) == 0:
    st.error("No customer records match your filter criteria. Adjust the sidebar sliders.")
    st.stop()

# Recompute portfolio KPIs
kpis = calculate_portfolio_kpis(filtered_df)

# ---------------------------------------------------------
# Cinematic Top Hero Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="cinematic-header">
    <div class="header-badge">
        <span class="pulse-dot"></span>
        SYSTEM TELEMETRY LIVE &bull; 10,000 ACCOUNTS AUDITED &bull; ECB CASE FRAMEWORK
    </div>
    <h1 class="header-title">Customer Engagement & Product Utilization Analytics</h1>
    <p class="header-subtitle">
        Empirical Behavioral Science, 3D Feature Landscapes & Dynamic Capital Flight Mitigation | Unified Mentor European Banking Research
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Primary Navigation Tabs
# ---------------------------------------------------------
tab_command, tab_3d_hologram, tab_engagement, tab_products, tab_vip_radar, tab_simulation, tab_methodology = st.tabs([
    "⚡ 3D Command Overview",
    "🪐 3D Customer Hologram",
    "📊 Engagement vs Churn",
    "📦 Product Depth Golden Zone",
    "🎯 VIP Lead Radar & Dossier",
    "📈 What-If Capital Engine",
    "📑 Empirical Research Tests"
])

# =========================================================
# TAB 1: 3D COMMAND OVERVIEW
# =========================================================
with tab_command:
    # 5 Hero 3D Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.markdown(f"""
        <div class="kpi-3d-card">
            <div class="kpi-3d-title">Portfolio Churn Rate</div>
            <div class="kpi-3d-val">{kpis['overall_churn_rate']*100:.2f}%</div>
            <span class="kpi-tag-pink">{kpis['churned_customers']:,} exited / {kpis['total_customers']:,}</span>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="kpi-3d-card">
            <div class="kpi-3d-title">Engagement Retention Ratio</div>
            <div class="kpi-3d-val">{kpis['err']:.2f}x</div>
            <span class="kpi-tag-blue">Active: {kpis['active_churn_rate']*100:.1f}% vs Inactive: {kpis['inactive_churn_rate']*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="kpi-3d-card">
            <div class="kpi-3d-title">High-Bal. Disengagement</div>
            <div class="kpi-3d-val">{kpis['hbdr']*100:.2f}%</div>
            <span class="kpi-tag-pink">{kpis['inactive_high_bal_count']:,} VIPs Dormant (>€100k)</span>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="kpi-3d-card">
            <div class="kpi-3d-title">Capital At Flight Risk</div>
            <div class="kpi-3d-val">€{kpis['at_risk_balance']/1e6:.1f}M</div>
            <span class="kpi-tag-pink">{kpis['pct_balance_at_risk']:.1f}% of total bank liquidity</span>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        st.markdown(f"""
        <div class="kpi-3d-card">
            <div class="kpi-3d-title">Credit Card Passivity Delta</div>
            <div class="kpi-3d-val">{kpis['cc_stickiness_delta']:+.2f}%</div>
            <span class="kpi-tag-blue">p = 0.492 (Zero Direct Lift)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3D Gauge & Sunburst Row
    col_g1, col_g2, col_g3 = st.columns([1, 1, 1.2])

    with col_g1:
        # Gauge 1: Portfolio Churn vs Regulatory Benchmark
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=kpis['overall_churn_rate'] * 100,
            title={'text': "<b>Portfolio Churn Risk</b><br><span style='font-size:12px;color:#94A3B8'>Target Threshold < 12.0%</span>", 'font': {'color': '#F8FAFC'}},
            delta={'reference': 12.0, 'increasing': {'color': '#F43F5E'}, 'decreasing': {'color': '#38BDF8'}},
            gauge={
                'axis': {'range': [0, 40], 'tickcolor': '#38BDF8'},
                'bar': {'color': '#F43F5E'},
                'bgcolor': 'rgba(10, 15, 29, 0.8)',
                'borderwidth': 2,
                'bordercolor': 'rgba(56, 189, 248, 0.2)',
                'steps': [
                    {'range': [0, 12], 'color': 'rgba(56, 189, 248, 0.2)'},
                    {'range': [12, 25], 'color': 'rgba(14, 165, 233, 0.2)'},
                    {'range': [25, 40], 'color': 'rgba(244, 63, 94, 0.3)'}
                ],
            }
        ))
        fig_g1.update_layout(**PLOTLY_DARK_LAYOUT, height=270, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_g1, width='stretch')

    with col_g2:
        # Gauge 2: Average Relationship Strength Index (RSI)
        avg_rsi = filtered_df['RelationshipStrengthIndex'].mean()
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_rsi,
            title={'text': "<b>Mean Relationship Strength (RSI)</b><br><span style='font-size:12px;color:#94A3B8'>Scale: 0 to 100</span>", 'font': {'color': '#F8FAFC'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#38BDF8'},
                'bar': {'color': '#38BDF8'},
                'bgcolor': 'rgba(10, 15, 29, 0.8)',
                'borderwidth': 2,
                'bordercolor': 'rgba(56, 189, 248, 0.2)',
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(244, 63, 94, 0.3)'},
                    {'range': [40, 70], 'color': 'rgba(14, 165, 233, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(56, 189, 248, 0.3)'}
                ],
            }
        ))
        fig_g2.update_layout(**PLOTLY_DARK_LAYOUT, height=270, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_g2, width='stretch')

    with col_g3:
        # Macro Sunburst: Country -> Products -> Activity -> Churn
        sunburst_sample = filtered_df.sample(min(3000, len(filtered_df)), random_state=42).copy()
        sunburst_sample['Activity'] = sunburst_sample['IsActiveMember'].map({1: 'Active', 0: 'Inactive'})
        sunburst_sample['ProductTier'] = sunburst_sample['NumOfProducts'].apply(lambda x: f"{x} Prod")
        sunburst_sample['ChurnStatus'] = sunburst_sample['Exited'].map({1: 'Churned', 0: 'Retained'})

        fig_sun = px.sunburst(
            sunburst_sample,
            path=['Geography', 'ProductTier', 'Activity', 'ChurnStatus'],
            color='ChurnStatus',
            color_discrete_map={'Retained': '#0EA5E9', 'Churned': '#F43F5E'},
            title="Multi-Tier Portfolio Sunburst Decomposition"
        )
        fig_sun.update_layout(**PLOTLY_DARK_LAYOUT, height=270, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_sun, width='stretch')

    st.markdown("---")

    # Segment Exposure Matrix
    st.markdown("### 🏛️ Capital Exposure & Behavioral Segments")
    cohort_df = compute_segment_exposure(filtered_df)

    c_bar, c_tree = st.columns([1.3, 1])

    with c_bar:
        fig_cohort = px.bar(
            cohort_df,
            x='EngagementProfile',
            y='ChurnRate',
            color='AvgBalance',
            color_continuous_scale=['#0EA5E9', '#38BDF8', '#F43F5E', '#BE123C'],
            text=[f"{v*100:.1f}%" for v in cohort_df['ChurnRate']],
            labels={'ChurnRate': 'Observed Churn', 'EngagementProfile': 'Behavioral Cohort', 'AvgBalance': 'Average Balance (€)'},
            title="Observed Churn Rate by Behavioral Profile (Color = Mean Balance €)"
        )
        fig_cohort.update_traces(textposition='outside')
        fig_cohort.update_layout(**PLOTLY_DARK_LAYOUT, height=360, margin=dict(l=10, r=10, t=40, b=60), xaxis_tickangle=-15)
        st.plotly_chart(fig_cohort, width='stretch')

    with c_tree:
        fig_tree = px.treemap(
            cohort_df,
            path=['EngagementProfile'],
            values='TotalBalance',
            color='ChurnRate',
            color_continuous_scale=['#0EA5E9', '#F43F5E'],
            title="Deposit Volume Share by Segment (Color = Churn Rate)"
        )
        fig_tree.update_layout(**PLOTLY_DARK_LAYOUT, height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_tree, width='stretch')

# =========================================================
# TAB 2: 3D CUSTOMER HOLOGRAM (NEW 3D FEATURE)
# =========================================================
with tab_3d_hologram:
    st.subheader("🪐 3D Customer Relationship Hologram")
    st.caption("Rotate, zoom, and inspect individual customer coordinate positions in 3D feature space: Age vs. Ledger Balance vs. Product Depth.")

    # 3D Controls
    col_3d_ctrl1, col_3d_ctrl2, col_3d_ctrl3 = st.columns(3)
    with col_3d_ctrl1:
        color_by_3d = st.selectbox("3D Marker Color Field", ["Exited (Retention vs Churn)", "EngagementProfile", "RelationshipStrengthIndex"])
    with col_3d_ctrl2:
        sample_size_3d = st.slider("3D Sampling Density", min_value=500, max_value=min(3000, len(filtered_df)), value=min(1500, len(filtered_df)), step=250)
    with col_3d_ctrl3:
        camera_angle = st.selectbox("Perspective Preset", ["Isometric 3D", "Top View (Age vs Balance)", "Side View (Products vs Balance)"])

    hologram_sample = filtered_df.sample(sample_size_3d, random_state=42).copy()

    # Configure Color Mapping
    if color_by_3d == "Exited (Retention vs Churn)":
        hologram_sample['ColorLabel'] = hologram_sample['Exited'].map({0: 'Retained (Stable)', 1: 'Churned (Exited)'})
        color_discrete = {'Retained (Stable)': '#0EA5E9', 'Churned (Exited)': '#F43F5E'}
        fig_3d = px.scatter_3d(
            hologram_sample,
            x='Age',
            y='Balance',
            z='NumOfProducts',
            color='ColorLabel',
            size='RelationshipStrengthIndex',
            size_max=14,
            opacity=0.8,
            color_discrete_map=color_discrete,
            hover_name='Surname',
            hover_data={
                'CustomerId': True,
                'Age': True,
                'Balance': ':.2f',
                'NumOfProducts': True,
                'RelationshipStrengthIndex': ':.1f',
                'Geography': True,
                'IsActiveMember': True
            },
            title="Interactive 3D Customer Behavioral Space (Size = Relationship Strength Index)"
        )
    else:
        fig_3d = px.scatter_3d(
            hologram_sample,
            x='Age',
            y='Balance',
            z='NumOfProducts',
            color=color_by_3d,
            size='RelationshipStrengthIndex',
            size_max=14,
            opacity=0.8,
            color_continuous_scale=['#0EA5E9', '#38BDF8', '#F43F5E', '#BE123C'],
            hover_name='Surname',
            hover_data={'CustomerId': True, 'Age': True, 'Balance': ':.2f', 'NumOfProducts': True},
            title=f"3D Space Colored by {color_by_3d}"
        )

    # Set camera perspective
    if camera_angle == "Top View (Age vs Balance)":
        cam_eye = dict(x=0, y=0, z=2.2)
    elif camera_angle == "Side View (Products vs Balance)":
        cam_eye = dict(x=2.2, y=0, z=0)
    else:
        cam_eye = dict(x=1.4, y=-1.4, z=1.2)

    fig_3d.update_layout(
        **PLOTLY_DARK_LAYOUT,
        scene=dict(
            xaxis=dict(title='Age (Years)', backgroundcolor='rgba(10, 15, 29, 0.7)', gridcolor='rgba(56, 189, 248, 0.2)'),
            yaxis=dict(title='Balance (€)', backgroundcolor='rgba(10, 15, 29, 0.7)', gridcolor='rgba(56, 189, 248, 0.2)'),
            zaxis=dict(title='Product Count', tickvals=[1, 2, 3, 4], backgroundcolor='rgba(10, 15, 29, 0.7)', gridcolor='rgba(56, 189, 248, 0.2)'),
            camera=dict(eye=cam_eye)
        ),
        height=650,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_3d, width='stretch')

    st.markdown("""
    <div style="background: rgba(13, 21, 39, 0.6); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 14px 20px;">
        <span style="color: #38BDF8; font-weight: 700;">Holographic Spatial Takeaway:</span> 
        Notice the heavy cluster of <span style="color: #F43F5E; font-weight: 700;">Crimson Pink</span> points floating in the upper-right corner: 
        <strong>Older Customers (Age 45–60) with balances above €120,000 and 1 or 3+ products</strong>. 
        This 3D cluster accounts for over 60% of total capital loss.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TAB 3: ENGAGEMENT VS CHURN & EUROPEAN MACRO
# =========================================================
with tab_engagement:
    st.subheader("⚡ Behavioral Engagement Dynamics vs. Retention Moats")
    st.caption("Validating the protective power of transactional velocity across the Eurozone.")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        # Engagement Waterfall / Stacked
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
            marker_color='#0EA5E9',
            text=[f"{v*100:.1f}%" for v in eng_split['RetentionRate']],
            textposition='inside'
        ))
        fig_eng.add_trace(go.Bar(
            name='Churned',
            x=eng_split['Status'],
            y=eng_split['ChurnRate'],
            marker_color='#F43F5E',
            text=[f"{v*100:.1f}%" for v in eng_split['ChurnRate']],
            textposition='inside'
        ))
        fig_eng.update_layout(
            **PLOTLY_DARK_LAYOUT,
            barmode='stack',
            title="Retention vs. Churn by Active Status (Stacked %)",
            height=380,
            yaxis_tickformat='.0%'
        )
        st.plotly_chart(fig_eng, width='stretch')

    with col_e2:
        # Geography Disparity Grouped Bar
        geo_split = filtered_df.groupby(['Geography', 'IsActiveMember'])['Exited'].mean().reset_index()
        geo_split['Activity'] = geo_split['IsActiveMember'].map({0: 'Inactive', 1: 'Active'})

        fig_geo = px.bar(
            geo_split,
            x='Geography',
            y='Exited',
            color='Activity',
            barmode='group',
            labels={'Exited': 'Observed Churn Rate'},
            color_discrete_map={'Inactive': '#F43F5E', 'Active': '#38BDF8'},
            title="Geographic Churn Disparity Across Activity States"
        )
        fig_geo.update_layout(**PLOTLY_DARK_LAYOUT, height=380, yaxis_tickformat='.1%')
        st.plotly_chart(fig_geo, width='stretch')

    st.markdown("---")

    # Financial Commitment Violin Distribution
    st.markdown("### 💰 Financial Commitment vs Activity: Violin & Scatter Insights")
    v1, v2 = st.columns(2)

    with v1:
        fig_violin = px.violin(
            filtered_df,
            x='IsActiveMember',
            y='Balance',
            color='Exited',
            box=True,
            points='all',
            color_discrete_map={0: '#0EA5E9', 1: '#F43F5E'},
            labels={'IsActiveMember': 'Activity (0 = Inactive, 1 = Active)', 'Balance': 'Ledger Balance (€)'},
            title="Balance Dispersion by Engagement & Churn Outcome"
        )
        fig_violin.update_layout(**PLOTLY_DARK_LAYOUT, height=400)
        st.plotly_chart(fig_violin, width='stretch')

    with v2:
        # Tenure Survival Heatmap
        tenure_heat = filtered_df.groupby(['Tenure', 'NumOfProducts'])['Exited'].mean().unstack()
        fig_heat = px.imshow(
            tenure_heat,
            labels=dict(x="Number of Products", y="Tenure (Years)", color="Churn Rate"),
            color_continuous_scale=['#0EA5E9', '#38BDF8', '#F43F5E', '#BE123C'],
            title="Tenure vs. Product Depth Churn Heatmap"
        )
        fig_heat.update_layout(**PLOTLY_DARK_LAYOUT, height=400)
        st.plotly_chart(fig_heat, width='stretch')

# =========================================================
# TAB 4: PRODUCT DEPTH & THE 2-PRODUCT GOLDEN ZONE
# =========================================================
with tab_products:
    st.subheader("📦 Product Depth Optimization & Cross-Sell Dynamics")
    st.caption("Empirical proof of the '2-Product Golden Zone' and catastrophic multi-product bundling friction.")

    p_col1, p_col2 = st.columns([1.2, 1])

    with p_col1:
        prod_kpis = filtered_df.groupby('NumOfProducts')['Exited'].agg(
            Count='count',
            ChurnRate='mean'
        ).reset_index()

        fig_golden = go.Figure()
        # Bar: Volume
        fig_golden.add_trace(go.Bar(
            x=prod_kpis['NumOfProducts'],
            y=prod_kpis['Count'],
            name='Accounts Held',
            yaxis='y',
            marker_color='rgba(14, 165, 233, 0.45)',
            marker_line=dict(color='#38BDF8', width=1.5)
        ))
        # Line: Churn
        fig_golden.add_trace(go.Scatter(
            x=prod_kpis['NumOfProducts'],
            y=prod_kpis['ChurnRate'],
            name='Churn Rate (%)',
            yaxis='y2',
            mode='lines+markers+text',
            text=[f"{v*100:.1f}%" for v in prod_kpis['ChurnRate']],
            textposition='top center',
            line=dict(color='#F43F5E', width=3.5),
            marker=dict(size=11, color='#FB7185', line=dict(color='#FFFFFF', width=2))
        ))

        fig_golden.update_layout(
            **PLOTLY_DARK_LAYOUT,
            title="The 2-Product Golden Ratio vs. Multi-Product Friction Curve",
            xaxis=dict(title='Products Held per Customer', tickvals=[1, 2, 3, 4]),
            yaxis=dict(title='Customer Accounts', side='left'),
            yaxis2=dict(title='Churn Rate', side='right', overlaying='y', tickformat='.0%', range=[0, 1.15]),
            height=420,
            legend=dict(x=0.03, y=0.95, bgcolor='rgba(10, 15, 29, 0.7)')
        )
        st.plotly_chart(fig_golden, width='stretch')

    with p_col2:
        # Credit Card Activity Breakdown
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
            color_continuous_scale=['#0EA5E9', '#F43F5E'],
            text=[f"{v*100:.1f}%" for v in cc_active['Churn']],
            title="Credit Card Retention Conditioned on Activity"
        )
        fig_cc_active.update_traces(textposition='outside')
        fig_cc_active.update_layout(**PLOTLY_DARK_LAYOUT, height=420, yaxis_tickformat='.0%')
        st.plotly_chart(fig_cc_active, width='stretch')

    # Executive Breakdown Cards
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        <div style="background: rgba(13, 21, 39, 0.8); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 16px;">
            <div style="font-family: 'JetBrains Mono', monospace; color: #38BDF8; font-size: 11px; font-weight: 700;">1 PRODUCT VULNERABILITY</div>
            <div style="font-size: 22px; font-weight: 800; color: #F8FAFC; margin: 4px 0;">27.71% Churn</div>
            <p style="font-size: 12px; color: #94A3B8; margin: 0;">50.8% of customers have zero secondary product anchors, resulting in negligible friction to exit.</p>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <div style="background: rgba(13, 21, 39, 0.8); border: 1px solid rgba(56, 189, 248, 0.5); border-radius: 12px; padding: 16px; box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);">
            <div style="font-family: 'JetBrains Mono', monospace; color: #38BDF8; font-size: 11px; font-weight: 700;">THE 2-PRODUCT GOLDEN ZONE</div>
            <div style="font-size: 22px; font-weight: 800; color: #38BDF8; margin: 4px 0;">7.58% Churn (-72.6%)</div>
            <p style="font-size: 12px; color: #94A3B8; margin: 0;">The mathematical optimum: pairing checking with a savings or mortgage account maximizes loyalty.</p>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown("""
        <div style="background: rgba(13, 21, 39, 0.8); border: 1px solid rgba(244, 63, 94, 0.4); border-radius: 12px; padding: 16px; box-shadow: 0 0 15px rgba(244, 63, 94, 0.2);">
            <div style="font-family: 'JetBrains Mono', monospace; color: #FB7185; font-size: 11px; font-weight: 700;">MULTI-PRODUCT HOCKEY STICK</div>
            <div style="font-size: 22px; font-weight: 800; color: #FB7185; margin: 4px 0;">82.7% to 100% Churn</div>
            <p style="font-size: 12px; color: #94A3B8; margin: 0;">Forced bundling beyond 2 products triggers intense dissatisfaction, fee fatigue, and immediate attrition.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 5: VIP LEAD RADAR & DOSSIER (INTERACTIVE RADAR)
# =========================================================
with tab_vip_radar:
    st.subheader("🎯 High-Value Disengaged Detector & Holographic Dossier")
    st.caption("Algorithmic lead prioritization for relationship managers with individual customer 3D holographic radar inspection.")

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

    # Lead Filtering
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
        st.metric("Total Vulnerable Deposit Exposure", f"€{vip_leads['Balance'].sum():,.2f}")
    with col_vm3:
        st.metric("Average Relationship Strength Index (RSI)", f"{vip_leads['RelationshipStrengthIndex'].mean():.1f} / 100")

    # Holographic Radar Inspector
    st.markdown("### 📡 Individual Customer Holographic Radar Dossier")
    if len(vip_leads) > 0:
        dossier_options = [f"{row['CustomerId']} - {row['Surname']} (€{row['Balance']:,.0f})" for _, row in vip_leads.head(50).iterrows()]
        selected_dossier_str = st.selectbox("Select Customer to Inspect Dossier", dossier_options)
        selected_cid = int(selected_dossier_str.split(" - ")[0])
        cust_record = vip_leads[vip_leads['CustomerId'] == selected_cid].iloc[0]

        r_left, r_right = st.columns([1.2, 1])

        with r_left:
            # Polar Radar
            radar_categories = ['Credit Health', 'Product Optimization', 'Activity Score', 'Tenure Loyalty', 'Salary Coverage']
            
            # Normalize to 0-100 scale for radar
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
                fillcolor='rgba(244, 63, 94, 0.35)',
                line=dict(color='#F43F5E', width=2.5),
                name=f"{cust_record['Surname']} Profile"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[80, 100, 100, 75, 70, 80],
                theta=radar_categories + [radar_categories[0]],
                fill='none',
                line=dict(color='#38BDF8', width=1.5, dash='dash'),
                name="Target Benchmark"
            ))
            fig_radar.update_layout(
                **PLOTLY_DARK_LAYOUT,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(56, 189, 248, 0.15)'),
                    angularaxis=dict(gridcolor='rgba(56, 189, 248, 0.15)')
                ),
                height=350,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, width='stretch')

        with r_right:
            st.markdown(f"""
            <div style="background: rgba(13, 21, 39, 0.85); border: 1px solid rgba(244, 63, 94, 0.4); border-radius: 14px; padding: 22px; box-shadow: 0 0 20px rgba(244, 63, 94, 0.15);">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #FB7185; letter-spacing: 1px;">CUSTOMER DOSSIER // #{cust_record['CustomerId']}</div>
                <div style="font-size: 24px; font-weight: 800; color: #FFFFFF; margin: 4px 0;">{cust_record['Surname']}, {cust_record['Geography']}</div>
                <div style="margin: 10px 0;">
                    <span class="kpi-tag-pink">{cust_record['RetentionTier']}</span>
                    <span class="kpi-tag-blue">RSI Score: {cust_record['RelationshipStrengthIndex']} / 100</span>
                </div>
                <div style="font-size: 13px; color: #CBD5E1; line-height: 1.6; margin-top: 12px;">
                    &bull; <strong>Ledger Balance:</strong> €{cust_record['Balance']:,.2f}<br>
                    &bull; <strong>Estimated Salary:</strong> €{cust_record['EstimatedSalary']:,.2f}<br>
                    &bull; <strong>Active Products:</strong> {cust_record['NumOfProducts']} product(s)<br>
                    &bull; <strong>Activity Status:</strong> {'Active Member' if cust_record['IsActiveMember']==1 else 'Inactive / Dormant'}<br>
                    &bull; <strong>Credit Card Held:</strong> {'Yes' if cust_record['HasCrCard']==1 else 'No'}
                </div>
                <div style="margin-top: 16px; font-size: 12px; color: #FDA4AF; border-top: 1px solid rgba(244, 63, 94, 0.3); padding-top: 10px;">
                    ⚠️ <strong>Strategic Action:</strong> Dispatch senior wealth concierge immediately. Recommend migrating deposit balance into 2-product fixed term or automated savings sweeps to avoid capital drain.
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
        label="📥 Export Prioritized Outreach Roster (CSV for Direct CRM Pipeline)",
        data=csv_data,
        file_name="high_value_disengaged_leads.csv",
        mime="text/csv"
    )

# =========================================================
# TAB 6: WHAT-IF CAPITAL ENGINE
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
            <div class="kpi-3d-card">
                <div class="kpi-3d-title">Projected Portfolio Churn</div>
                <div class="kpi-3d-val">{sim_res['simulated_churn_rate']*100:.2f}%</div>
                <span class="kpi-tag-blue">Down from {sim_res['baseline_churn_rate']*100:.2f}% (-{sim_res['churn_rate_reduction_bps']:.0f} bps)</span>
            </div>
            """, unsafe_allow_html=True)
        with o2:
            st.markdown(f"""
            <div class="kpi-3d-card">
                <div class="kpi-3d-title">Customer Accounts Retained</div>
                <div class="kpi-3d-val">{sim_res['total_customers_saved']:,}</div>
                <span class="kpi-tag-blue">Accounts Prevented from Exiting</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        o3, o4 = st.columns(2)
        with o3:
            st.markdown(f"""
            <div class="kpi-3d-card">
                <div class="kpi-3d-title">Deposit Capital Preserved</div>
                <div class="kpi-3d-val">€{sim_res['total_capital_preserved']/1e6:.2f}M</div>
                <span class="kpi-tag-pink">Protected from Liquidity Run-Off</span>
            </div>
            """, unsafe_allow_html=True)
        with o4:
            st.markdown(f"""
            <div class="kpi-3d-card">
                <div class="kpi-3d-title">Annual Net Interest Margin Protected</div>
                <div class="kpi-3d-val">€{sim_res['annual_revenue_protected']/1e6:.2f}M</div>
                <span class="kpi-tag-blue">At {sim_nim:.2f}% Annual Spread</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Strategy Attribution Waterfall / Bar
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
        color_continuous_scale=['#0EA5E9', '#38BDF8', '#F43F5E'],
        text=[f"€{v:.2f}M" for v in strat_data['Capital Saved (€M)']],
        title="Attribution of Capital Preserved Across Intervention Levers (€ Millions)"
    )
    fig_waterfall.update_traces(textposition='outside')
    fig_waterfall.update_layout(**PLOTLY_DARK_LAYOUT, height=360)
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

    # Machine Learning Feature Importance
    models_dict = train_behavioral_models(raw_df)
    col_rf, col_or = st.columns(2)

    with col_rf:
        fig_rf = px.bar(
            models_dict['rf_importance'],
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=['#0EA5E9', '#F43F5E'],
            title="Random Forest Gini Feature Importance"
        )
        fig_rf.update_layout(**PLOTLY_DARK_LAYOUT, yaxis=dict(autorange='reversed'), height=400)
        st.plotly_chart(fig_rf, width='stretch')

    with col_or:
        fig_or = px.bar(
            models_dict['odds_ratios'],
            x='OddsRatio',
            y='Feature',
            orientation='h',
            color='Direction',
            color_discrete_map={'Risk Factor (+)': '#F43F5E', 'Retention Factor (-)': '#0EA5E9'},
            title="Logistic Regression Odds Ratios"
        )
        fig_or.add_vline(x=1.0, line_dash='dash', line_color='#94A3B8')
        fig_or.update_layout(**PLOTLY_DARK_LAYOUT, yaxis=dict(autorange='reversed'), height=400)
        st.plotly_chart(fig_or, width='stretch')

# ---------------------------------------------------------
# Cinematic Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 10px 0;">
    RETENTION INTELLIGENCE v3.2 &bull; DEVELOPED BY ARYAN &bull; UNIFIED MENTOR BANKING RESEARCH & EUROPEAN CENTRAL BANK CASE FRAMEWORK
</div>
""", unsafe_allow_html=True)
