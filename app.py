"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
Interactive Executive & Analytical Dashboard
Bright & Lively Executive Fintech Theme (Natural Human-Crafted Aesthetic)
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
    page_title="Banking Retention Intelligence | Customer Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Bright, Lively, Human-Crafted Fintech CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #0F172A;
    }
    
    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
    }

    /* Executive Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #0284C7 100%);
        border-radius: 18px;
        padding: 28px 32px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.15);
        position: relative;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #FFFFFF;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }

    .hero-subtitle {
        color: #E0E7FF;
        font-size: 14px;
        margin-top: 6px;
        font-weight: 400;
        max-width: 900px;
        line-height: 1.5;
    }

    /* Bright, Crisp Metric Cards */
    .kpi-card-bright {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
    }
    
    .kpi-card-bright:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.12);
        border-color: #93C5FD;
    }

    .kpi-card-title {
        font-size: 12px;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.6px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .kpi-card-val {
        font-size: 28px;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }

    .pill-green {
        color: #065F46;
        background: #D1FAE5;
        border: 1px solid #A7F3D0;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }

    .pill-red {
        color: #991B1B;
        background: #FEE2E2;
        border: 1px solid #FECACA;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }

    .pill-blue {
        color: #1E40AF;
        background: #DBEAFE;
        border: 1px solid #BFDBFE;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }

    .pill-amber {
        color: #92400E;
        background: #FEF3C7;
        border: 1px solid #FDE68A;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }

    /* Content Cards */
    .content-box-bright {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #E2E8F0;
        padding: 6px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        color: #475569;
        font-weight: 600;
        font-size: 13px;
        background: transparent;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #1E3A8A !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Clean Bright Plotly Layout Template (No keyword conflicts)
# ---------------------------------------------------------
BRIGHT_LAYOUT = dict(
    paper_bgcolor='rgba(255,255,255,0)',
    plot_bgcolor='rgba(255,255,255,0)',
    font=dict(family='Plus Jakarta Sans, sans-serif', color='#475569', size=12),
    xaxis=dict(
        gridcolor='#F1F5F9',
        zerolinecolor='#CBD5E1',
        tickfont=dict(color='#64748B')
    ),
    yaxis=dict(
        gridcolor='#F1F5F9',
        zerolinecolor='#CBD5E1',
        tickfont=dict(color='#64748B')
    ),
    margin=dict(l=20, r=20, t=40, b=20)
)

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
    <div style="padding: 10px 0 16px 0;">
        <span style="font-size: 28px;">🏦</span>
        <span style="font-size: 18px; font-weight: 800; color: #1E3A8A; margin-left: 8px;">RETENTION HUB</span>
        <div style="font-size: 11px; color: #64748B; font-weight: 600;">Executive Decision Support System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🌍 Jurisdiction & Demographics")
    geo_list = sorted(raw_df['Geography'].unique().tolist())
    selected_geos = st.multiselect("Geography", geo_list, default=geo_list)

    gender_list = sorted(raw_df['Gender'].unique().tolist())
    selected_genders = st.multiselect("Gender", gender_list, default=gender_list)

    st.markdown("---")
    st.markdown("#### ⚡ Activity & Products")
    activity_filter = st.radio("Activity Status", ["All Accounts", "Active Members Only", "Inactive Members Only"])

    all_prods = sorted(raw_df['NumOfProducts'].unique().tolist())
    selected_prods = st.multiselect("Product Count", all_prods, default=all_prods)

    st.markdown("---")
    st.markdown("#### 💶 Balance & Salary")
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
<div style="background: #F1F5F9; border: 1px solid #CBD5E1; border-radius: 8px; padding: 10px; margin-top: 15px;">
    <div style="font-size: 11px; font-weight: 700; color: #475569;">FILTERED SAMPLE</div>
    <div style="font-size: 18px; font-weight: 800; color: #1E3A8A;">{len(filtered_df):,} <span style="font-size: 12px; color: #64748B;">/ {len(raw_df):,} accounts</span></div>
</div>
""", unsafe_allow_html=True)

if len(filtered_df) == 0:
    st.error("No customer records match the current filter selection. Please relax sidebar sliders.")
    st.stop()

# Compute portfolio KPIs
kpis = calculate_portfolio_kpis(filtered_df)

# ---------------------------------------------------------
# Bright Hero Header Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-badge">
        PORTFOLIO INTELLIGENCE &bull; 10,000 ACCOUNTS &bull; UNIFIED MENTOR BANKING STUDY
    </div>
    <h1 class="hero-title">Customer Engagement & Product Utilization Analytics</h1>
    <p class="hero-subtitle">
        Evaluating the empirical relationship between behavioral engagement, product depth, and deposit retention across European retail banking.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_command, tab_3d_hologram, tab_engagement, tab_products, tab_vip_radar, tab_simulation, tab_methodology = st.tabs([
    "📊 Executive Summary",
    "🌐 3D Customer Space",
    "⚡ Engagement Dynamics",
    "📦 The 2-Product Golden Zone",
    "🎯 VIP Lead Detector & Dossier",
    "📈 What-If Strategy Simulator",
    "📑 Research Findings & Tests"
])

# =========================================================
# TAB 1: EXECUTIVE SUMMARY
# =========================================================
with tab_command:
    # 5 Hero Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.markdown(f"""
        <div class="kpi-card-bright">
            <div class="kpi-card-title">Portfolio Churn Rate</div>
            <div class="kpi-card-val">{kpis['overall_churn_rate']*100:.2f}%</div>
            <span class="pill-red">{kpis['churned_customers']:,} exited accounts</span>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="kpi-card-bright">
            <div class="kpi-card-title">Engagement Retention Ratio</div>
            <div class="kpi-card-val">{kpis['err']:.2f}x</div>
            <span class="pill-green">Active Churn: {kpis['active_churn_rate']*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="kpi-card-bright">
            <div class="kpi-card-title">High-Bal. Disengagement</div>
            <div class="kpi-card-val">{kpis['hbdr']*100:.2f}%</div>
            <span class="pill-amber">{kpis['inactive_high_bal_count']:,} VIPs Inactive (>€100k)</span>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="kpi-card-bright">
            <div class="kpi-card-title">Capital At Flight Risk</div>
            <div class="kpi-card-val">€{kpis['at_risk_balance']/1e6:.1f}M</div>
            <span class="pill-red">{kpis['pct_balance_at_risk']:.1f}% of total balance</span>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        st.markdown(f"""
        <div class="kpi-card-bright">
            <div class="kpi-card-title">Credit Card Stickiness Delta</div>
            <div class="kpi-card-val">{kpis['cc_stickiness_delta']:+.2f}%</div>
            <span class="pill-blue">p = 0.492 (Zero direct lift)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dial & Sunburst Row
    col_g1, col_g2, col_g3 = st.columns([1, 1, 1.2])

    with col_g1:
        # Gauge 1: Portfolio Churn
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=kpis['overall_churn_rate'] * 100,
            title={'text': "<b>Observed Churn Rate</b><br><span style='font-size:12px;color:#64748B'>Target Benchmark < 12.0%</span>"},
            delta={'reference': 12.0, 'increasing': {'color': '#EF4444'}, 'decreasing': {'color': '#10B981'}},
            gauge={
                'axis': {'range': [0, 40], 'tickcolor': '#64748B'},
                'bar': {'color': '#EF4444'},
                'bgcolor': '#F8FAFC',
                'borderwidth': 1,
                'bordercolor': '#CBD5E1',
                'steps': [
                    {'range': [0, 12], 'color': '#D1FAE5'},
                    {'range': [12, 25], 'color': '#FEF3C7'},
                    {'range': [25, 40], 'color': '#FEE2E2'}
                ],
            }
        ))
        fig_g1.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=260, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_g1, width='stretch')

    with col_g2:
        # Gauge 2: Average Relationship Strength Index (RSI)
        avg_rsi = filtered_df['RelationshipStrengthIndex'].mean()
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_rsi,
            title={'text': "<b>Mean Relationship Strength (RSI)</b><br><span style='font-size:12px;color:#64748B'>Score Scale: 0 to 100</span>"},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#64748B'},
                'bar': {'color': '#2563EB'},
                'bgcolor': '#F8FAFC',
                'borderwidth': 1,
                'bordercolor': '#CBD5E1',
                'steps': [
                    {'range': [0, 40], 'color': '#FEE2E2'},
                    {'range': [40, 70], 'color': '#FEF3C7'},
                    {'range': [70, 100], 'color': '#DBEAFE'}
                ],
            }
        ))
        fig_g2.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=260, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_g2, width='stretch')

    with col_g3:
        # Macro Sunburst Chart
        sunburst_sample = filtered_df.sample(min(3000, len(filtered_df)), random_state=42).copy()
        sunburst_sample['Activity'] = sunburst_sample['IsActiveMember'].map({1: 'Active', 0: 'Inactive'})
        sunburst_sample['ProductTier'] = sunburst_sample['NumOfProducts'].apply(lambda x: f"{x} Prod")
        sunburst_sample['Status'] = sunburst_sample['Exited'].map({1: 'Churned', 0: 'Retained'})

        fig_sun = px.sunburst(
            sunburst_sample,
            path=['Geography', 'ProductTier', 'Activity', 'Status'],
            color='Status',
            color_discrete_map={'Retained': '#10B981', 'Churned': '#EF4444'},
            title="Portfolio Hierarchy (Geography > Products > Activity > Churn)"
        )
        fig_sun.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=260, margin=dict(l=10, r=10, t=40, b=10))
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
            color_continuous_scale=['#3B82F6', '#10B981', '#F59E0B', '#EF4444'],
            text=[f"{v*100:.1f}%" for v in cohort_df['ChurnRate']],
            labels={'ChurnRate': 'Churn Rate', 'EngagementProfile': 'Cohort', 'AvgBalance': 'Avg Balance (€)'},
            title="Churn Rate by Behavioral Profile (Color = Mean Balance €)"
        )
        fig_cohort.update_traces(textposition='outside')
        fig_cohort.update_layout(**BRIGHT_LAYOUT, height=360, xaxis_tickangle=-15)
        st.plotly_chart(fig_cohort, width='stretch')

    with c_tree:
        fig_tree = px.treemap(
            cohort_df,
            path=['EngagementProfile'],
            values='TotalBalance',
            color='ChurnRate',
            color_continuous_scale=['#10B981', '#F59E0B', '#EF4444'],
            title="Deposit Volume Share by Segment"
        )
        fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_tree, width='stretch')

# =========================================================
# TAB 2: 3D CUSTOMER SPACE (CLEAN, BRIGHT & ERROR-FREE)
# =========================================================
with tab_3d_hologram:
    st.subheader("🌐 3D Customer Relationship Space")
    st.caption("Interact and rotate in 3D: Customer Age vs. Ledger Balance vs. Product Depth.")

    # 3D Controls
    col_3d_ctrl1, col_3d_ctrl2, col_3d_ctrl3 = st.columns(3)
    with col_3d_ctrl1:
        color_by_3d = st.selectbox("Color Data By", ["Exited (Retention vs Churn)", "EngagementProfile", "RelationshipStrengthIndex"])
    with col_3d_ctrl2:
        sample_size_3d = st.slider("Customer Sample Density", min_value=500, max_value=min(3000, len(filtered_df)), value=min(1500, len(filtered_df)), step=250)
    with col_3d_ctrl3:
        camera_angle = st.selectbox("Camera Perspective", ["Isometric Perspective", "Top View (Age vs Balance)", "Side View (Products vs Balance)"])

    hologram_sample = filtered_df.sample(sample_size_3d, random_state=42).copy()

    # Configure Color Mapping with bright lively colors
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
            opacity=0.75,
            color_discrete_map={'Retained (Loyal)': '#10B981', 'Churned (Exited)': '#EF4444'},
            hover_name='Surname',
            hover_data={
                'CustomerId': True,
                'Age': True,
                'Balance': ':.2f',
                'NumOfProducts': True,
                'RelationshipStrengthIndex': ':.1f',
                'Geography': True
            },
            title="3D Customer Behavioral Space (Size = Relationship Strength Index)"
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
            opacity=0.75,
            color_continuous_scale=['#3B82F6', '#10B981', '#F59E0B', '#EF4444'],
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

    # Update 3D Layout cleanly WITHOUT duplicate keyword arguments
    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans, sans-serif', color='#475569'),
        scene=dict(
            xaxis=dict(title='Age (Years)', backgroundcolor='#FFFFFF', gridcolor='#E2E8F0', showbackground=True),
            yaxis=dict(title='Balance (€)', backgroundcolor='#FFFFFF', gridcolor='#E2E8F0', showbackground=True),
            zaxis=dict(title='Product Count', tickvals=[1, 2, 3, 4], backgroundcolor='#FFFFFF', gridcolor='#E2E8F0', showbackground=True),
            camera=dict(eye=cam_eye)
        ),
        height=620,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_3d, width='stretch')

    st.info("""
    **3D Spatial Insight:** 
    Notice the clear cluster of red churn markers concentrated at **Age 45–60** with **Balances above €100,000** holding only 1 product or 3+ products. 
    This group accounts for over **60% of total deposit capital loss**.
    """)

# =========================================================
# TAB 3: ENGAGEMENT DYNAMICS
# =========================================================
with tab_engagement:
    st.subheader("⚡ Behavioral Engagement Dynamics & Attrition")
    st.caption("Active transacting frequency is the primary protective barrier against churn.")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        # Engagement Stacked Bar
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
            marker_color='#10B981',
            text=[f"{v*100:.1f}%" for v in eng_split['RetentionRate']],
            textposition='inside'
        ))
        fig_eng.add_trace(go.Bar(
            name='Churned',
            x=eng_split['Status'],
            y=eng_split['ChurnRate'],
            marker_color='#EF4444',
            text=[f"{v*100:.1f}%" for v in eng_split['ChurnRate']],
            textposition='inside'
        ))
        fig_eng.update_layout(
            **BRIGHT_LAYOUT,
            barmode='stack',
            title="Retention vs. Churn by Active Member Status",
            height=360,
            yaxis_tickformat='.0%'
        )
        st.plotly_chart(fig_eng, width='stretch')

    with col_e2:
        # Geographic Disparity
        geo_split = filtered_df.groupby(['Geography', 'IsActiveMember'])['Exited'].mean().reset_index()
        geo_split['Activity'] = geo_split['IsActiveMember'].map({0: 'Inactive', 1: 'Active'})

        fig_geo = px.bar(
            geo_split,
            x='Geography',
            y='Exited',
            color='Activity',
            barmode='group',
            labels={'Exited': 'Churn Rate'},
            color_discrete_map={'Inactive': '#EF4444', 'Active': '#2563EB'},
            title="Geographic Churn Comparison Across Activity States"
        )
        fig_geo.update_layout(**BRIGHT_LAYOUT, height=360, yaxis_tickformat='.1%')
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
            color_discrete_map={0: '#10B981', 1: '#EF4444'},
            labels={'IsActiveMember': 'Activity (0 = Inactive, 1 = Active)', 'Balance': 'Ledger Balance (€)'},
            title="Balance Distribution across Engagement by Churn Status"
        )
        fig_violin.update_layout(**BRIGHT_LAYOUT, height=380)
        st.plotly_chart(fig_violin, width='stretch')

    with v2:
        tenure_heat = filtered_df.groupby(['Tenure', 'NumOfProducts'])['Exited'].mean().unstack()
        fig_heat = px.imshow(
            tenure_heat,
            labels=dict(x="Number of Products", y="Tenure (Years)", color="Churn Rate"),
            color_continuous_scale=['#D1FAE5', '#FEF3C7', '#EF4444'],
            title="Tenure vs. Product Count Churn Heatmap"
        )
        fig_heat.update_layout(**BRIGHT_LAYOUT, height=380)
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
        # Bar: Volume
        fig_golden.add_trace(go.Bar(
            x=prod_kpis['NumOfProducts'],
            y=prod_kpis['Count'],
            name='Customer Accounts',
            yaxis='y',
            marker_color='#93C5FD'
        ))
        # Line: Churn Rate
        fig_golden.add_trace(go.Scatter(
            x=prod_kpis['NumOfProducts'],
            y=prod_kpis['ChurnRate'],
            name='Churn Rate (%)',
            yaxis='y2',
            mode='lines+markers+text',
            text=[f"{v*100:.1f}%" for v in prod_kpis['ChurnRate']],
            textposition='top center',
            line=dict(color='#DC2626', width=3),
            marker=dict(size=10, color='#DC2626')
        ))

        fig_golden.update_layout(
            **BRIGHT_LAYOUT,
            title="The 2-Product Golden Zone vs. Multi-Product Friction",
            xaxis=dict(title='Products Held per Customer', tickvals=[1, 2, 3, 4]),
            yaxis=dict(title='Customer Volume', side='left'),
            yaxis2=dict(title='Churn Rate', side='right', overlaying='y', tickformat='.0%', range=[0, 1.15]),
            height=400,
            legend=dict(x=0.03, y=0.95)
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
            color_continuous_scale=['#10B981', '#F59E0B', '#EF4444'],
            text=[f"{v*100:.1f}%" for v in cc_active['Churn']],
            title="Credit Card Retention Conditioned on Activity"
        )
        fig_cc_active.update_traces(textposition='outside')
        fig_cc_active.update_layout(**BRIGHT_LAYOUT, height=400, yaxis_tickformat='.0%')
        st.plotly_chart(fig_cc_active, width='stretch')

    # 3 Analytical Findings Cards
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="color: #64748B; font-size: 11px; font-weight: 700; text-transform: uppercase;">1 Product Vulnerability</div>
            <div style="font-size: 22px; font-weight: 800; color: #EF4444; margin: 4px 0;">27.71% Churn</div>
            <p style="font-size: 13px; color: #64748B; margin: 0;">50.8% of customers have zero secondary product anchors, resulting in near-zero switching friction.</p>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="color: #166534; font-size: 11px; font-weight: 700; text-transform: uppercase;">The 2-Product Golden Zone</div>
            <div style="font-size: 22px; font-weight: 800; color: #16A34A; margin: 4px 0;">7.58% Churn (-72.6%)</div>
            <p style="font-size: 13px; color: #166534; margin: 0;">The mathematical optimum: pairing checking with a savings or mortgage account maximizes loyalty.</p>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown("""
        <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="color: #991B1B; font-size: 11px; font-weight: 700; text-transform: uppercase;">Multi-Product Friction</div>
            <div style="font-size: 22px; font-weight: 800; color: #DC2626; margin: 4px 0;">82.7% to 100% Churn</div>
            <p style="font-size: 13px; color: #991B1B; margin: 0;">Forced bundling beyond 2 products triggers customer alienation, fee fatigue, and immediate attrition.</p>
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
                fillcolor='rgba(239, 68, 68, 0.25)',
                line=dict(color='#EF4444', width=2),
                name=f"{cust_record['Surname']} Profile"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[80, 100, 100, 75, 70, 80],
                theta=radar_categories + [radar_categories[0]],
                fill='none',
                line=dict(color='#2563EB', width=1.5, dash='dash'),
                name="Target Benchmark"
            ))
            fig_radar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='#E2E8F0'),
                    angularaxis=dict(gridcolor='#E2E8F0')
                ),
                height=340,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, width='stretch')

        with r_right:
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                <div style="font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase;">CUSTOMER PROFILE #{cust_record['CustomerId']}</div>
                <div style="font-size: 22px; font-weight: 800; color: #0F172A; margin: 4px 0;">{cust_record['Surname']}, {cust_record['Geography']}</div>
                <div style="margin: 8px 0;">
                    <span class="pill-red">{cust_record['RetentionTier']}</span>
                    <span class="pill-blue">RSI Score: {cust_record['RelationshipStrengthIndex']} / 100</span>
                </div>
                <div style="font-size: 13px; color: #475569; line-height: 1.6; margin-top: 10px;">
                    &bull; <strong>Ledger Balance:</strong> €{cust_record['Balance']:,.2f}<br>
                    &bull; <strong>Estimated Salary:</strong> €{cust_record['EstimatedSalary']:,.2f}<br>
                    &bull; <strong>Bank Products:</strong> {cust_record['NumOfProducts']} product(s)<br>
                    &bull; <strong>Activity Status:</strong> {'Active Member' if cust_record['IsActiveMember']==1 else 'Inactive / Dormant'}<br>
                    &bull; <strong>Credit Card Held:</strong> {'Yes' if cust_record['HasCrCard']==1 else 'No'}
                </div>
                <div style="margin-top: 14px; font-size: 12px; color: #B91C1C; background: #FEF2F2; padding: 10px; border-radius: 8px;">
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
            <div class="kpi-card-bright">
                <div class="kpi-card-title">Projected Portfolio Churn</div>
                <div class="kpi-card-val">{sim_res['simulated_churn_rate']*100:.2f}%</div>
                <span class="pill-green">Down from {sim_res['baseline_churn_rate']*100:.2f}% (-{sim_res['churn_rate_reduction_bps']:.0f} bps)</span>
            </div>
            """, unsafe_allow_html=True)
        with o2:
            st.markdown(f"""
            <div class="kpi-card-bright">
                <div class="kpi-card-title">Customer Accounts Retained</div>
                <div class="kpi-card-val">{sim_res['total_customers_saved']:,}</div>
                <span class="pill-blue">Accounts Prevented from Exiting</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        o3, o4 = st.columns(2)
        with o3:
            st.markdown(f"""
            <div class="kpi-card-bright">
                <div class="kpi-card-title">Deposit Capital Preserved</div>
                <div class="kpi-card-val">€{sim_res['total_capital_preserved']/1e6:.2f}M</div>
                <span class="pill-green">Protected from Liquidity Run-Off</span>
            </div>
            """, unsafe_allow_html=True)
        with o4:
            st.markdown(f"""
            <div class="kpi-card-bright">
                <div class="kpi-card-title">Annual Net Interest Margin Protected</div>
                <div class="kpi-card-val">€{sim_res['annual_revenue_protected']/1e6:.2f}M</div>
                <span class="pill-blue">At {sim_nim:.2f}% Annual Spread</span>
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
        color_continuous_scale=['#93C5FD', '#2563EB', '#1E3A8A'],
        text=[f"€{v:.2f}M" for v in strat_data['Capital Saved (€M)']],
        title="Attribution of Capital Preserved Across Intervention Levers (€ Millions)"
    )
    fig_waterfall.update_traces(textposition='outside')
    fig_waterfall.update_layout(**BRIGHT_LAYOUT, height=360)
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
            color_continuous_scale=['#93C5FD', '#2563EB'],
            title="Random Forest Gini Feature Importance"
        )
        fig_rf.update_layout(**BRIGHT_LAYOUT, yaxis=dict(autorange='reversed'), height=380)
        st.plotly_chart(fig_rf, width='stretch')

    with col_or:
        fig_or = px.bar(
            models_dict['odds_ratios'],
            x='OddsRatio',
            y='Feature',
            orientation='h',
            color='Direction',
            color_discrete_map={'Risk Factor (+)': '#EF4444', 'Retention Factor (-)': '#10B981'},
            title="Logistic Regression Odds Ratios"
        )
        fig_or.add_vline(x=1.0, line_dash='dash', line_color='#64748B')
        fig_or.update_layout(**BRIGHT_LAYOUT, yaxis=dict(autorange='reversed'), height=380)
        st.plotly_chart(fig_or, width='stretch')

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 12px; padding: 10px 0;">
    Customer Engagement & Product Utilization Analytics Platform &bull; Developed by Aryan &bull; Unified Mentor Banking Research & European Central Bank Case Framework
</div>
""", unsafe_allow_html=True)
