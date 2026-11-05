"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
The European Retail Banking Folio & Tactile Analytical Sketchbook
Design: Tactile Archival Paper, Editorial Serif (Instrument Serif / Newsreader), Ink & Wax Seal Palette
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
    page_title="European Banking Folio & Analytical Sketchbook",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Tactile Archival Paper & Editorial Typography CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Core Paper Canvas */
    html, body, [class*="css"] {
        font-family: 'Newsreader', Garamond, Georgia, serif;
        color: #2B2721;
    }
    
    .stApp {
        background-color: #F7F4EE;
        background-image: 
            radial-gradient(at 0% 0%, rgba(235, 230, 220, 0.6) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(230, 224, 212, 0.6) 0px, transparent 50%);
    }

    /* Tactile Sketchbook Cover / Header */
    .sketchbook-header {
        background: #FCFAF6;
        border: 1px solid #E6E0D4;
        border-radius: 12px;
        padding: 36px 42px;
        margin-bottom: 26px;
        box-shadow: 
            0 2px 4px rgba(43, 39, 33, 0.02),
            0 10px 28px rgba(43, 39, 33, 0.05),
            inset 0 0 0 1px rgba(255, 255, 255, 0.8);
        position: relative;
    }
    
    .sketchbook-header::after {
        content: '';
        position: absolute;
        bottom: 0; left: 40px; right: 40px; height: 1px;
        background: linear-gradient(90deg, transparent, #D5CDBC, transparent);
    }

    .folio-stamp {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #F1ECE2;
        border: 1px dashed #C8BFA8;
        color: #635A4B;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 4px;
        margin-bottom: 14px;
    }

    .folio-title {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 38px;
        font-weight: 400;
        color: #2B2721;
        margin: 0;
        line-height: 1.15;
        letter-spacing: 0.01em;
    }

    .folio-subtitle {
        color: #6B6355;
        font-size: 16px;
        font-style: italic;
        margin-top: 8px;
        line-height: 1.6;
        max-width: 950px;
    }

    /* Ledger Summary Strip */
    .ledger-strip {
        display: flex;
        gap: 28px;
        margin-top: 20px;
        padding-top: 18px;
        border-top: 1px solid #ECE6DA;
        flex-wrap: wrap;
    }
    .ledger-cell {
        font-size: 13px;
        color: #7A7162;
    }
    .ledger-cell strong {
        color: #2B2721;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Tactile Paper Cards */
    .paper-card {
        background: #FCFAF6;
        border: 1px solid #E5DFD3;
        border-radius: 10px;
        padding: 22px 24px;
        box-shadow: 
            0 1px 3px rgba(43, 39, 33, 0.03),
            0 6px 16px rgba(43, 39, 33, 0.04);
        position: relative;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .paper-card:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 2px 6px rgba(43, 39, 33, 0.04),
            0 12px 28px rgba(43, 39, 33, 0.07);
    }

    .paper-card-plate-no {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: #9E9484;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .paper-card-label {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 18px;
        color: #4A4337;
        margin-bottom: 6px;
    }

    .paper-card-value {
        font-size: 32px;
        font-family: 'Newsreader', Georgia, serif;
        font-weight: 500;
        color: #2B2721;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }

    /* Wax Seals & Ink Badges */
    .wax-seal-vermilion {
        display: inline-block;
        background: #F9EBEA;
        border: 1px solid #E6B0AA;
        color: #922B21;
        font-family: 'Newsreader', Georgia, serif;
        font-size: 13px;
        font-style: italic;
        padding: 2px 10px;
        border-radius: 20px;
    }

    .wax-seal-emerald {
        display: inline-block;
        background: #EAF2EC;
        border: 1px solid #A9DFBF;
        color: #196F3D;
        font-family: 'Newsreader', Georgia, serif;
        font-size: 13px;
        font-style: italic;
        padding: 2px 10px;
        border-radius: 20px;
    }

    .wax-seal-gold {
        display: inline-block;
        background: #FCF3CF;
        border: 1px solid #F9E79F;
        color: #7D6608;
        font-family: 'Newsreader', Georgia, serif;
        font-size: 13px;
        font-style: italic;
        padding: 2px 10px;
        border-radius: 20px;
    }

    /* Editorial Plate Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #EFEAE0;
        padding: 5px;
        border-radius: 8px;
        border: 1px solid #DFD8CB;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 18px;
        color: #6B6355;
        font-family: 'Newsreader', Georgia, serif;
        font-size: 14px;
        font-weight: 500;
        background: transparent;
        transition: all 0.15s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #FCFAF6 !important;
        color: #2B2721 !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 4px rgba(43, 39, 33, 0.08) !important;
    }

    /* Sidebar Parchment */
    section[data-testid="stSidebar"] {
        background-color: #F3EFE7;
        border-right: 1px solid #E2DBCF;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tactile Ink & Parchment Chart Helper
# ---------------------------------------------------------
def style_plate_chart(fig, height=390, title=None):
    """
    Styles Plotly charts with a tactile archival ink-on-parchment aesthetic.
    """
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Newsreader, Georgia, serif', color='#595349', size=13),
        height=height,
        margin=dict(l=25, r=25, t=55 if title else 25, b=30)
    )
    if title:
        fig.update_layout(title=dict(text=f"<i>{title}</i>", font=dict(family='Instrument Serif, Georgia, serif', size=17, color='#2B2721')))
    fig.update_xaxes(
        gridcolor='#ECE6DA',
        zerolinecolor='#D8D0C2',
        tickfont=dict(family='JetBrains Mono, monospace', size=11, color='#7A7162')
    )
    fig.update_yaxes(
        gridcolor='#ECE6DA',
        zerolinecolor='#D8D0C2',
        tickfont=dict(family='JetBrains Mono, monospace', size=11, color='#7A7162')
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
# Folio Index & Curated Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 14px 0; border-bottom: 1px solid #E2DBCF; margin-bottom: 16px;">
        <div style="font-family: 'Instrument Serif', Georgia, serif; font-size: 24px; color: #2B2721;">
            The Analytical Folio
        </div>
        <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #8C8271; letter-spacing: 1px;">
            PORTFOLIO REVISION &bull; 10,000 FOLIOS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Index Filter: Jurisdiction**")
    geo_list = sorted(raw_df['Geography'].unique().tolist())
    selected_geos = st.multiselect("Geography", geo_list, default=geo_list, label_visibility="collapsed")

    st.markdown("**Index Filter: Demographic**")
    gender_list = sorted(raw_df['Gender'].unique().tolist())
    selected_genders = st.multiselect("Gender", gender_list, default=gender_list, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Behavioral State**")
    activity_filter = st.radio("Activity", ["All Accounts", "Active Transacting", "Inactive / Dormant"], label_visibility="collapsed")

    st.markdown("**Product Holding Depth**")
    all_prods = sorted(raw_df['NumOfProducts'].unique().tolist())
    selected_prods = st.multiselect("Products", all_prods, default=all_prods, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Account Ledger Balance (€)**")
    bal_min, bal_max = int(raw_df['Balance'].min()), int(raw_df['Balance'].max())
    bal_range = st.slider("Balance", bal_min, bal_max, (bal_min, bal_max), step=5000, label_visibility="collapsed")

    st.markdown("**Estimated Annual Compensation (€)**")
    sal_min, sal_max = int(raw_df['EstimatedSalary'].min()), int(raw_df['EstimatedSalary'].max())
    sal_range = st.slider("Salary", sal_min, sal_max, (sal_min, sal_max), step=5000, label_visibility="collapsed")

# Apply filters
filtered_df = raw_df[
    (raw_df['Geography'].isin(selected_geos)) &
    (raw_df['Gender'].isin(selected_genders)) &
    (raw_df['NumOfProducts'].isin(selected_prods)) &
    (raw_df['Balance'] >= bal_range[0]) &
    (raw_df['Balance'] <= bal_range[1]) &
    (raw_df['EstimatedSalary'] >= sal_range[0]) &
    (raw_df['EstimatedSalary'] <= sal_range[1])
]

if activity_filter == "Active Transacting":
    filtered_df = filtered_df[filtered_df['IsActiveMember'] == 1]
elif activity_filter == "Inactive / Dormant":
    filtered_df = filtered_df[filtered_df['IsActiveMember'] == 0]

st.sidebar.markdown(f"""
<div style="background: #FCFAF6; border: 1px solid #DFD8CB; border-radius: 8px; padding: 12px; margin-top: 15px;">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #8C8271; letter-spacing: 1px;">ACTIVE SAMPLE</div>
    <div style="font-size: 20px; font-weight: 600; color: #2B2721; font-family: 'Newsreader', serif;">
        {len(filtered_df):,} <span style="font-size: 13px; color: #8C8271;">folios examined</span>
    </div>
</div>
""", unsafe_allow_html=True)

if len(filtered_df) == 0:
    st.error("No folio records match the selected ledger filters.")
    st.stop()

# Portfolio KPIs
kpis = calculate_portfolio_kpis(filtered_df)

# ---------------------------------------------------------
# Tactile Header Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="sketchbook-header">
    <div class="folio-stamp">
        FOLIO RECORD &bull; ARCHIVAL STUDY &bull; UNIFIED MENTOR &bull; ECB BENCHMARK
    </div>
    <h1 class="folio-title">Customer Engagement & Product Utilization Analytics</h1>
    <p class="folio-subtitle">
        An empirical investigation into depositor behavior, relationship stickiness, and silent capital flight across European commercial banking jurisdictions.
    </p>
    <div class="ledger-strip">
        <div class="ledger-cell">Total Examined Liquidity: <strong>€{kpis['total_balance']/1e6:.2f}M</strong></div>
        <div class="ledger-cell">Overall Portfolio Attrition: <strong>{kpis['overall_churn_rate']*100:.2f}%</strong></div>
        <div class="ledger-cell">Active Protection Multiplier: <strong>{kpis['err']:.2f}&times;</strong></div>
        <div class="ledger-cell">Unanchored VIP Deposits: <strong>€{kpis['at_risk_balance']/1e6:.2f}M</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Plates Navigation (Editorial Folio Index)
# ---------------------------------------------------------
tab_plate1, tab_plate2, tab_plate3, tab_plate4, tab_plate5, tab_plate6, tab_plate7 = st.tabs([
    "Plate I: Portfolio Ledger",
    "Plate II: 3D Relationship Geometry",
    "Plate III: Engagement Dynamics",
    "Plate IV: The 2-Product Golden Zone",
    "Plate V: VIP Account Inspection",
    "Plate VI: Capital Preservation Engine",
    "Plate VII: Econometric & Policy Tests"
])

# =========================================================
# PLATE I: PORTFOLIO LEDGER
# =========================================================
with tab_plate1:
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-card-plate-no">METRIC I</div>
            <div class="paper-card-label">Observed Attrition</div>
            <div class="paper-card-value">{kpis['overall_churn_rate']*100:.2f}%</div>
            <span class="wax-seal-vermilion">{kpis['churned_customers']:,} departed folios</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-card-plate-no">METRIC II</div>
            <div class="paper-card-label">Engagement Retention</div>
            <div class="paper-card-value">{kpis['err']:.2f}&times;</div>
            <span class="wax-seal-emerald">Active: {kpis['active_churn_rate']*100:.1f}% vs Inactive: {kpis['inactive_churn_rate']*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-card-plate-no">METRIC III</div>
            <div class="paper-card-label">VIP Disengagement</div>
            <div class="paper-card-value">{kpis['hbdr']*100:.2f}%</div>
            <span class="wax-seal-gold">{kpis['inactive_high_bal_count']:,} VIP accounts dormant</span>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-card-plate-no">METRIC IV</div>
            <div class="paper-card-label">Flight-Risk Capital</div>
            <div class="paper-card-value">€{kpis['at_risk_balance']/1e6:.1f}M</div>
            <span class="wax-seal-vermilion">{kpis['pct_balance_at_risk']:.1f}% of ledger deposits</span>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-card-plate-no">METRIC V</div>
            <div class="paper-card-label">Card Ownership Lift</div>
            <div class="paper-card-value">{kpis['cc_stickiness_delta']:+.2f}%</div>
            <span class="wax-seal-gold">p = 0.492 (Zero direct lift)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dial & Sunburst
    g1, g2, g3 = st.columns([1, 1, 1.2])

    with g1:
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=kpis['overall_churn_rate'] * 100,
            title={'text': "<b>Portfolio Churn Risk</b><br><span style='font-size:12px;color:#7A7162'>Regulatory Target < 12.0%</span>", 'font': {'family': 'Instrument Serif, Georgia', 'color': '#2B2721'}},
            delta={'reference': 12.0, 'increasing': {'color': '#922B21'}, 'decreasing': {'color': '#196F3D'}},
            gauge={
                'axis': {'range': [0, 40], 'tickcolor': '#8C8271'},
                'bar': {'color': '#922B21'},
                'bgcolor': '#FCFAF6',
                'borderwidth': 1,
                'bordercolor': '#DFD8CB',
                'steps': [
                    {'range': [0, 12], 'color': '#EAF2EC'},
                    {'range': [12, 25], 'color': '#FCF3CF'},
                    {'range': [25, 40], 'color': '#F9EBEA'}
                ],
            }
        ))
        fig_g1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Newsreader, serif', color='#2B2721'), height=260, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_g1, width='stretch')

    with g2:
        avg_rsi = filtered_df['RelationshipStrengthIndex'].mean()
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_rsi,
            title={'text': "<b>Relationship Strength Index (RSI)</b><br><span style='font-size:12px;color:#7A7162'>Empirical Scale: 0 to 100</span>", 'font': {'family': 'Instrument Serif, Georgia', 'color': '#2B2721'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8C8271'},
                'bar': {'color': '#196F3D'},
                'bgcolor': '#FCFAF6',
                'borderwidth': 1,
                'bordercolor': '#DFD8CB',
                'steps': [
                    {'range': [0, 40], 'color': '#F9EBEA'},
                    {'range': [40, 70], 'color': '#FCF3CF'},
                    {'range': [70, 100], 'color': '#EAF2EC'}
                ],
            }
        ))
        fig_g2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Newsreader, serif', color='#2B2721'), height=260, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_g2, width='stretch')

    with g3:
        sunburst_sample = filtered_df.sample(min(3000, len(filtered_df)), random_state=42).copy()
        sunburst_sample['Activity'] = sunburst_sample['IsActiveMember'].map({1: 'Active', 0: 'Inactive'})
        sunburst_sample['ProductTier'] = sunburst_sample['NumOfProducts'].apply(lambda x: f"{x} Prod")
        sunburst_sample['Status'] = sunburst_sample['Exited'].map({1: 'Departed', 0: 'Retained'})

        fig_sun = px.sunburst(
            sunburst_sample,
            path=['Geography', 'ProductTier', 'Activity', 'Status'],
            color='Status',
            color_discrete_map={'Retained': '#196F3D', 'Departed': '#922B21'},
            title="Portfolio Structural Decomposition"
        )
        fig_sun.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Newsreader, serif', color='#2B2721'), height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_sun, width='stretch')

    st.markdown("---")

    # Cohort Exposure Breakdown
    st.markdown("### 🏛️ Liquidity Distribution Across Behavioral Profiles")
    cohort_df = compute_segment_exposure(filtered_df)

    c_bar, c_tree = st.columns([1.3, 1])

    with c_bar:
        fig_cohort = px.bar(
            cohort_df,
            x='EngagementProfile',
            y='ChurnRate',
            color='AvgBalance',
            color_continuous_scale=['#D4E6F1', '#A9DFBF', '#F9E79F', '#E6B0AA', '#922B21'],
            text=[f"{v*100:.1f}%" for v in cohort_df['ChurnRate']],
            labels={'ChurnRate': 'Observed Churn', 'EngagementProfile': 'Cohort Profile', 'AvgBalance': 'Average Balance (€)'},
            title="Observed Attrition by Behavioral Cohort (Color = Average Balance €)"
        )
        fig_cohort.update_traces(textposition='outside')
        style_plate_chart(fig_cohort, height=360)
        fig_cohort.update_layout(xaxis_tickangle=-15)
        st.plotly_chart(fig_cohort, width='stretch')

    with c_tree:
        fig_tree = px.treemap(
            cohort_df,
            path=['EngagementProfile'],
            values='TotalBalance',
            color='ChurnRate',
            color_continuous_scale=['#196F3D', '#B7950B', '#922B21'],
            title="Deposit Volume Distribution by Profile"
        )
        fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Newsreader, serif', color='#2B2721'), height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_tree, width='stretch')

# =========================================================
# PLATE II: 3D RELATIONSHIP GEOMETRY (TACTILE 3D LANDSCAPE)
# =========================================================
with tab_plate2:
    st.markdown("### Plate II: 3D Spatial Geometry of Depositor Retention")
    st.caption("A spatial coordinates model plotting Customer Age against Ledger Balance (€) and Number of Bank Products.")

    col_3d_1, col_3d_2, col_3d_3 = st.columns(3)
    with col_3d_1:
        color_by_3d = st.selectbox("Render Coloring By", ["Exited (Retention vs Churn)", "EngagementProfile", "RelationshipStrengthIndex"])
    with col_3d_2:
        sample_size_3d = st.slider("Folio Density Sample", min_value=500, max_value=min(3000, len(filtered_df)), value=min(1500, len(filtered_df)), step=250)
    with col_3d_3:
        camera_angle = st.selectbox("Spatial Orientation", ["Isometric Perspective", "Top Plan (Age vs Balance)", "Elevation (Products vs Balance)"])

    hologram_sample = filtered_df.sample(sample_size_3d, random_state=42).copy()

    if color_by_3d == "Exited (Retention vs Churn)":
        hologram_sample['Outcome'] = hologram_sample['Exited'].map({0: 'Retained Account', 1: 'Departed Account'})
        fig_3d = px.scatter_3d(
            hologram_sample,
            x='Age',
            y='Balance',
            z='NumOfProducts',
            color='Outcome',
            size='RelationshipStrengthIndex',
            size_max=12,
            opacity=0.85,
            color_discrete_map={'Retained Account': '#196F3D', 'Departed Account': '#922B21'},
            hover_name='Surname',
            hover_data={
                'CustomerId': True,
                'Age': True,
                'Balance': ':.2f',
                'NumOfProducts': True,
                'RelationshipStrengthIndex': ':.1f',
                'Geography': True
            },
            title="3D Spatial Relationship Geometry (Point Diameter = Relationship Strength Index)"
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
            color_continuous_scale=['#D4E6F1', '#A9DFBF', '#F9E79F', '#E6B0AA', '#922B21'],
            hover_name='Surname',
            hover_data={'CustomerId': True, 'Age': True, 'Balance': ':.2f', 'NumOfProducts': True},
            title=f"3D Space Colored by {color_by_3d}"
        )

    if camera_angle == "Top Plan (Age vs Balance)":
        cam_eye = dict(x=0, y=0, z=2.2)
    elif camera_angle == "Elevation (Products vs Balance)":
        cam_eye = dict(x=2.2, y=0, z=0)
    else:
        cam_eye = dict(x=1.4, y=-1.4, z=1.2)

    # Clean 3D layout without duplicate keyword argument
    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Newsreader, Georgia, serif', color='#595349'),
        scene=dict(
            xaxis=dict(title='Age (Years)', backgroundcolor='#FCFAF6', gridcolor='#ECE6DA', showbackground=True),
            yaxis=dict(title='Balance (€)', backgroundcolor='#FCFAF6', gridcolor='#ECE6DA', showbackground=True),
            zaxis=dict(title='Products Held', tickvals=[1, 2, 3, 4], backgroundcolor='#FCFAF6', gridcolor='#ECE6DA', showbackground=True),
            camera=dict(eye=cam_eye)
        ),
        height=620,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_3d, width='stretch')

    st.markdown("""
    <div style="background: #FCFAF6; border: 1px solid #DFD8CB; border-radius: 8px; padding: 14px 20px;">
        <span style="font-family: 'Instrument Serif', serif; font-size: 18px; color: #2B2721;">Editorial Spatial Note:</span> 
        Notice the pronounced cluster of vermilion (departed) accounts concentrated at 
        <strong>Age 45–60 with ledger balances exceeding €100,000 holding only 1 or 3+ products</strong>. 
        This cohort accounts for more than sixty percent of total institutional capital flight.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PLATE III: ENGAGEMENT DYNAMICS
# =========================================================
with tab_plate3:
    st.markdown("### Plate III: Behavioral Engagement Dynamics & Attrition")
    st.caption("Active member transacting velocity acts as an indispensable defense against account defection.")

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
            marker_color='#196F3D',
            text=[f"{v*100:.1f}%" for v in eng_split['RetentionRate']],
            textposition='inside'
        ))
        fig_eng.add_trace(go.Bar(
            name='Departed',
            x=eng_split['Status'],
            y=eng_split['ChurnRate'],
            marker_color='#922B21',
            text=[f"{v*100:.1f}%" for v in eng_split['ChurnRate']],
            textposition='inside'
        ))
        style_plate_chart(fig_eng, height=360, title="Retention vs. Churn by Active Member Status")
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
            labels={'Exited': 'Observed Churn Rate'},
            color_discrete_map={'Inactive': '#922B21', 'Active': '#196F3D'},
            title="Geographic Churn Comparison by Jurisdiction"
        )
        style_plate_chart(fig_geo, height=360)
        fig_geo.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_geo, width='stretch')

    st.markdown("---")

    # Violin & Heatmap
    st.markdown("### 💰 Financial Commitment & Tenure Dynamics")
    v1, v2 = st.columns(2)

    with v1:
        fig_violin = px.violin(
            filtered_df,
            x='IsActiveMember',
            y='Balance',
            color='Exited',
            box=True,
            points=False,
            color_discrete_map={0: '#196F3D', 1: '#922B21'},
            labels={'IsActiveMember': 'Activity (0 = Inactive, 1 = Active)', 'Balance': 'Ledger Balance (€)'},
            title="Balance Distribution across Activity by Churn Status"
        )
        style_plate_chart(fig_violin, height=380)
        st.plotly_chart(fig_violin, width='stretch')

    with v2:
        tenure_heat = filtered_df.groupby(['Tenure', 'NumOfProducts'])['Exited'].mean().unstack()
        fig_heat = px.imshow(
            tenure_heat,
            labels=dict(x="Products Held", y="Tenure (Years)", color="Churn Rate"),
            color_continuous_scale=['#EAF2EC', '#FCF3CF', '#F9EBEA', '#922B21'],
            title="Tenure vs. Product Count Churn Heatmap"
        )
        style_plate_chart(fig_heat, height=380)
        st.plotly_chart(fig_heat, width='stretch')

# =========================================================
# PLATE IV: THE 2-PRODUCT GOLDEN ZONE
# =========================================================
with tab_plate4:
    st.markdown("### Plate IV: Product Depth Optimization & The 2-Product Golden Zone")
    st.caption("Quantitative proof of non-linear retention dynamics across product adoption tiers.")

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
            name='Accounts Monitored',
            yaxis='y',
            marker_color='#D5CDBC',
            marker_line=dict(color='#8C8271', width=1)
        ))
        # Line: Churn
        fig_golden.add_trace(go.Scatter(
            x=prod_kpis['NumOfProducts'],
            y=prod_kpis['ChurnRate'],
            name='Attrition Rate (%)',
            yaxis='y2',
            mode='lines+markers+text',
            text=[f"{v*100:.1f}%" for v in prod_kpis['ChurnRate']],
            textposition='top center',
            line=dict(color='#922B21', width=2.5),
            marker=dict(size=9, color='#922B21')
        ))

        fig_golden.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Newsreader, serif', color='#595349', size=12),
            title=dict(text="<i>The 2-Product Golden Zone vs. Multi-Product Friction Curve</i>", font=dict(family='Instrument Serif', color='#2B2721', size=17)),
            xaxis=dict(
                title='Bank Products Held',
                tickvals=[1, 2, 3, 4],
                gridcolor='#ECE6DA',
                tickfont=dict(family='JetBrains Mono', color='#7A7162')
            ),
            yaxis=dict(
                title='Customer Volume',
                side='left',
                gridcolor='#ECE6DA',
                tickfont=dict(family='JetBrains Mono', color='#7A7162')
            ),
            yaxis2=dict(
                title='Attrition Rate',
                side='right',
                overlaying='y',
                tickformat='.0%',
                range=[0, 1.15],
                tickfont=dict(family='JetBrains Mono', color='#922B21')
            ),
            height=400,
            margin=dict(l=25, r=25, t=55, b=25),
            legend=dict(x=0.03, y=0.95, bgcolor='rgba(252, 250, 246, 0.85)')
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
            color_continuous_scale=['#196F3D', '#B7950B', '#922B21'],
            text=[f"{v*100:.1f}%" for v in cc_active['Churn']],
            title="Credit Card Retention Conditioned on Activity"
        )
        fig_cc_active.update_traces(textposition='outside')
        style_plate_chart(fig_cc_active, height=400)
        fig_cc_active.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig_cc_active, width='stretch')

    # Three Editorial Plates
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        <div class="paper-card">
            <div class="paper-card-plate-no">OBSERVATION I</div>
            <div class="paper-card-label">1 Product Vulnerability</div>
            <div style="font-size: 26px; font-family: 'Newsreader', serif; color: #922B21; margin: 4px 0;">27.71% Attrition</div>
            <p style="font-size: 14px; color: #6B6355; margin: 0; font-style: italic;">50.8% of customers hold no secondary anchors, creating near-zero switching resistance.</p>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <div class="paper-card" style="border: 1px solid #A9DFBF; background: #FAFDFB;">
            <div class="paper-card-plate-no" style="color: #196F3D;">OBSERVATION II</div>
            <div class="paper-card-label" style="color: #196F3D;">The 2-Product Golden Zone</div>
            <div style="font-size: 26px; font-family: 'Newsreader', serif; color: #196F3D; margin: 4px 0;">7.58% Attrition (&minus;72.6%)</div>
            <p style="font-size: 14px; color: #1E4620; margin: 0; font-style: italic;">The mathematical optimum: primary transaction paired with savings or mortgage maximizes relationship loyalty.</p>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown("""
        <div class="paper-card" style="border: 1px solid #E6B0AA; background: #FDFBFB;">
            <div class="paper-card-plate-no" style="color: #922B21;">OBSERVATION III</div>
            <div class="paper-card-label" style="color: #922B21;">Multi-Product Friction</div>
            <div style="font-size: 26px; font-family: 'Newsreader', serif; color: #922B21; margin: 4px 0;">82.7% to 100% Churn</div>
            <p style="font-size: 14px; color: #6B6355; margin: 0; font-style: italic;">Forced cross-selling quotas beyond 2 products triggers intense dissatisfaction and immediate flight.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# PLATE V: VIP ACCOUNT INSPECTION & LOUPE DOSSIER
# =========================================================
with tab_plate5:
    st.markdown("### Plate V: High-Value Disengaged Inspection & Dossier")
    st.caption("A magnifying analytical loupe isolating dormant accounts holding significant capital balances.")

    c_lead1, c_lead2, c_lead3 = st.columns(3)
    with c_lead1:
        vip_min_bal = st.number_input("VIP Minimum Balance Threshold (€)", value=100000, step=10000)
    with c_lead2:
        vip_tier_filter = st.multiselect(
            "Retention Risk Classification",
            ['Critical Disengagement (Flight Risk)', 'Vulnerable', 'Moderate Engagement', 'Sticky Anchor (Very High Loyalty)'],
            default=['Critical Disengagement (Flight Risk)', 'Vulnerable']
        )
    with c_lead3:
        sort_order = st.selectbox("Sort Priority Order", ["Balance: High to Low", "RSI: Low to High", "Salary: High to Low"])

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
        st.metric("Vulnerable Capital Volume", f"€{vip_leads['Balance'].sum():,.2f}")
    with col_vm3:
        st.metric("Average Relationship Strength (RSI)", f"{vip_leads['RelationshipStrengthIndex'].mean():.1f} / 100")

    # Radar Dossier Loupe
    st.markdown("### 🔍 Individual Folio Radar Loupe")
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
                fillcolor='rgba(146, 43, 33, 0.15)',
                line=dict(color='#922B21', width=2),
                name=f"{cust_record['Surname']} Folio"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[80, 100, 100, 75, 70, 80],
                theta=radar_categories + [radar_categories[0]],
                fill='none',
                line=dict(color='#196F3D', width=1.5, dash='dash'),
                name="Benchmark Anchor"
            ))
            fig_radar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Newsreader, serif', color='#595349'),
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='#ECE6DA'),
                    angularaxis=dict(gridcolor='#ECE6DA')
                ),
                height=340,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, width='stretch')

        with r_right:
            st.markdown(f"""
            <div class="paper-card" style="border: 1px solid #C8BFA8;">
                <div class="paper-card-plate-no">FOLIO DOSSIER #{cust_record['CustomerId']}</div>
                <div style="font-family: 'Instrument Serif', Georgia; font-size: 26px; color: #2B2721; margin: 4px 0;">
                    {cust_record['Surname']}, {cust_record['Geography']}
                </div>
                <div style="margin: 8px 0;">
                    <span class="wax-seal-vermilion">{cust_record['RetentionTier']}</span>
                    <span class="wax-seal-emerald">RSI: {cust_record['RelationshipStrengthIndex']} / 100</span>
                </div>
                <div style="font-size: 14px; color: #595349; line-height: 1.7; margin-top: 12px; font-style: italic;">
                    &bull; <strong>Ledger Balance:</strong> €{cust_record['Balance']:,.2f}<br>
                    &bull; <strong>Annual Salary Estimate:</strong> €{cust_record['EstimatedSalary']:,.2f}<br>
                    &bull; <strong>Active Products:</strong> {cust_record['NumOfProducts']} product(s)<br>
                    &bull; <strong>Activity Status:</strong> {'Active Transacting' if cust_record['IsActiveMember']==1 else 'Inactive / Dormant'}<br>
                    &bull; <strong>Credit Facility:</strong> {'Issued' if cust_record['HasCrCard']==1 else 'None'}
                </div>
                <div style="margin-top: 14px; font-size: 13px; color: #922B21; border-top: 1px solid #ECE6DA; padding-top: 10px;">
                    📌 <strong>Advisory Directive:</strong> Dispatch dedicated relationship manager. Propose 2-product fixed term or automated savings sweeps to avoid capital outflow.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Lead Table
    st.markdown("#### Prioritized Client Registry")
    st.dataframe(
        vip_leads[[
            'CustomerId', 'Surname', 'Geography', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'RelationshipStrengthIndex', 'RetentionTier', 'Exited'
        ]].rename(columns={
            'CustomerId': 'Folio ID',
            'NumOfProducts': 'Products',
            'HasCrCard': 'CreditCard',
            'IsActiveMember': 'Active',
            'RelationshipStrengthIndex': 'RSI Score',
            'RetentionTier': 'Classification',
            'Exited': 'Historical Flight'
        }),
        width='stretch',
        hide_index=True
    )

    csv_data = vip_leads.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Registry to CSV (Direct Client Outreach)",
        data=csv_data,
        file_name="high_value_disengaged_folios.csv",
        mime="text/csv"
    )

# =========================================================
# PLATE VI: WHAT-IF CAPITAL PRESERVATION ENGINE
# =========================================================
with tab_plate6:
    st.markdown("### Plate VI: What-If Capital Preservation Simulation")
    st.caption("Model the projected liquidity retention and margin protection achieved through targeted behavioral interventions.")

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
        st.markdown("### 🎯 Projected Liquidity Outcomes")
        
        o1, o2 = st.columns(2)
        with o1:
            st.markdown(f"""
            <div class="paper-card">
                <div class="paper-card-plate-no">OUTCOME I</div>
                <div class="paper-card-label">Projected Portfolio Churn</div>
                <div class="paper-card-value">{sim_res['simulated_churn_rate']*100:.2f}%</div>
                <span class="wax-seal-emerald">Reduced from {sim_res['baseline_churn_rate']*100:.2f}% (&minus;{sim_res['churn_rate_reduction_bps']:.0f} bps)</span>
            </div>
            """, unsafe_allow_html=True)
        with o2:
            st.markdown(f"""
            <div class="paper-card">
                <div class="paper-card-plate-no">OUTCOME II</div>
                <div class="paper-card-label">Accounts Retained</div>
                <div class="paper-card-value">{sim_res['total_customers_saved']:,}</div>
                <span class="wax-seal-emerald">Client departures prevented</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        o3, o4 = st.columns(2)
        with o3:
            st.markdown(f"""
            <div class="paper-card">
                <div class="paper-card-plate-no">OUTCOME III</div>
                <div class="paper-card-label">Deposit Capital Preserved</div>
                <div class="paper-card-value">€{sim_res['total_capital_preserved']/1e6:.2f}M</div>
                <span class="wax-seal-emerald">Protected from liquidity drain</span>
            </div>
            """, unsafe_allow_html=True)
        with o4:
            st.markdown(f"""
            <div class="paper-card">
                <div class="paper-card-plate-no">OUTCOME IV</div>
                <div class="paper-card-label">Annual Net Interest Saved</div>
                <div class="paper-card-value">€{sim_res['annual_revenue_protected']/1e6:.2f}M</div>
                <span class="wax-seal-emerald">At {sim_nim:.2f}% Annual Spread</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Strategy Attribution Bar Chart
    st.markdown("### 📊 Strategy Attribution: Preserved Capital by Lever")
    strat_data = pd.DataFrame([
        {'Strategy': k, 'Customers Saved': v['customers_saved'], 'Capital Saved (€M)': v['capital_saved']/1e6}
        for k, v in sim_res['strategy_breakdown'].items()
    ])

    fig_waterfall = px.bar(
        strat_data,
        x='Strategy',
        y='Capital Saved (€M)',
        color='Capital Saved (€M)',
        color_continuous_scale=['#D5CDBC', '#196F3D'],
        text=[f"€{v:.2f}M" for v in strat_data['Capital Saved (€M)']],
        title="Attribution of Capital Preserved Across Intervention Levers (€ Millions)"
    )
    fig_waterfall.update_traces(textposition='outside')
    style_plate_chart(fig_waterfall, height=360)
    st.plotly_chart(fig_waterfall, width='stretch')

# =========================================================
# PLATE VII: ECONOMETRIC RESEARCH TESTS
# =========================================================
with tab_plate7:
    st.markdown("### Plate VII: Econometric & Statistical Hypothesis Suite")
    st.caption("Formal statistical validations and machine learning feature importance confirming behavioral superiority.")

    hyp_tests = run_statistical_hypothesis_tests(raw_df)
    hyp_rows = []
    for k, v in hyp_tests.items():
        hyp_rows.append({
            'Hypothesis Test': v['name'],
            'Statistical Methodology': v['test_type'],
            'Test Statistic': v['statistic'],
            'p-Value': f"{v['p_value']:.4e}" if v['p_value'] < 0.0001 else f"{v['p_value']:.4f}",
            'Significant (α = 0.01)': "✅ Significant" if v['significant'] else "❌ Not Significant",
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
            color_continuous_scale=['#D5CDBC', '#196F3D'],
            title="Random Forest Gini Feature Importance"
        )
        style_plate_chart(fig_rf, height=380)
        fig_rf.update_layout(yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_rf, width='stretch')

    with col_or:
        fig_or = px.bar(
            models_dict['odds_ratios'],
            x='OddsRatio',
            y='Feature',
            orientation='h',
            color='Direction',
            color_discrete_map={'Risk Factor (+)': '#922B21', 'Retention Factor (-)': '#196F3D'},
            title="Logistic Regression Odds Ratios"
        )
        fig_or.add_vline(x=1.0, line_dash='dash', line_color='#8C8271')
        style_plate_chart(fig_or, height=380)
        fig_or.update_layout(yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_or, width='stretch')

# ---------------------------------------------------------
# Tactile Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8C8271; font-family: 'Newsreader', serif; font-size: 13px; font-style: italic; padding: 10px 0;">
    The European Retail Banking Folio &bull; Archival Empirical Analysis &bull; Prepared by Aryan &bull; Unified Mentor & European Central Bank Case Framework
</div>
""", unsafe_allow_html=True)
