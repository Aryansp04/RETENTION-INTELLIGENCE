"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
Interactive Executive & Analytical Dashboard
Author: Aryan (Internship Portfolio Project)
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_processing import load_and_preprocess_data, calculate_portfolio_kpis
from src.analytics import run_statistical_hypothesis_tests, train_behavioral_models, compute_segment_exposure
from src.simulator import simulate_retention_interventions

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Engagement & Product Analytics | Banking Retention Strategy",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: #F8FAFC;
        margin-bottom: 25px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #94A3B8;
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .kpi-delta-pos {
        font-size: 12px;
        font-weight: 500;
        color: #059669;
    }
    .kpi-delta-neg {
        font-size: 12px;
        font-weight: 500;
        color: #DC2626;
    }
    
    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-danger { background-color: #FEE2E2; color: #991B1B; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; }
    .badge-success { background-color: #D1FAE5; color: #065F46; }
    .badge-info { background-color: #DBEAFE; color: #1E40AF; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Ingestion & Caching
# ---------------------------------------------------------
@st.cache_data
def get_dataset():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Churn_Modelling.csv")
    if not os.path.exists(data_path):
        data_path = "Churn_Modelling.csv"
    return load_and_preprocess_data(data_path)

raw_df = get_dataset()

# ---------------------------------------------------------
# Sidebar Filter Controls
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
st.sidebar.title("Portfolio Filtering")
st.sidebar.caption("Filter customer segments for tactical analysis")

# 1. Geography Multi-select
geo_options = sorted(raw_df['Geography'].unique().tolist())
selected_geos = st.sidebar.multiselect("Geography", geo_options, default=geo_options)

# 2. Gender Filter
gender_options = sorted(raw_df['Gender'].unique().tolist())
selected_genders = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

# 3. Activity Filter
activity_choice = st.sidebar.radio("Engagement Status", ["All Customers", "Active Members Only", "Inactive Members Only"])

# 4. Product Count Filter
prod_counts = sorted(raw_df['NumOfProducts'].unique().tolist())
selected_prods = st.sidebar.multiselect("Number of Products", prod_counts, default=prod_counts)

# 5. Financial Sliders
min_bal, max_bal = int(raw_df['Balance'].min()), int(raw_df['Balance'].max())
balance_range = st.sidebar.slider("Account Balance Range (€)", min_bal, max_bal, (min_bal, max_bal), step=5000)

min_sal, max_sal = int(raw_df['EstimatedSalary'].min()), int(raw_df['EstimatedSalary'].max())
salary_range = st.sidebar.slider("Estimated Salary Range (€)", min_sal, max_sal, (min_sal, max_sal), step=5000)

# Filter dataframe application
filtered_df = raw_df[
    (raw_df['Geography'].isin(selected_geos)) &
    (raw_df['Gender'].isin(selected_genders)) &
    (raw_df['NumOfProducts'].isin(selected_prods)) &
    (raw_df['Balance'] >= balance_range[0]) &
    (raw_df['Balance'] <= balance_range[1]) &
    (raw_df['EstimatedSalary'] >= salary_range[0]) &
    (raw_df['EstimatedSalary'] <= salary_range[1])
]

if activity_choice == "Active Members Only":
    filtered_df = filtered_df[filtered_df['IsActiveMember'] == 1]
elif activity_choice == "Inactive Members Only":
    filtered_df = filtered_df[filtered_df['IsActiveMember'] == 0]

st.sidebar.divider()
st.sidebar.markdown(f"**Filtered Cohort:** `{len(filtered_df):,}` / `{len(raw_df):,}` accounts")
if len(filtered_df) == 0:
    st.error("No customers match the current filter selection. Please relax sidebar filters.")
    st.stop()

# Compute KPIs for the filtered dataset
kpis = calculate_portfolio_kpis(filtered_df)

# ---------------------------------------------------------
# Top Executive Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🏦 Customer Engagement & Product Utilization Analytics</h1>
    <p>Empirical Behavioral Analysis, Capital Flight Diagnostics & Strategic Retention Optimization | Unified Mentor European Banking Study</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Top Navigation Tabs
# ---------------------------------------------------------
tab_exec, tab_eng, tab_prod, tab_detector, tab_sim, tab_research = st.tabs([
    "📊 Executive KPI Overview",
    "⚡ Engagement vs. Churn",
    "📦 Product Depth & Utilization",
    "🎯 High-Value Disengaged Detector",
    "📈 Retention What-If Simulator",
    "📑 Research Findings & Methodology"
])

# =========================================================
# TAB 1: EXECUTIVE KPI OVERVIEW
# =========================================================
with tab_exec:
    st.subheader("Core Strategic Retention KPIs")
    st.caption("Standardized quantitative metrics aligning customer behavior, product adoption, and balance exposure.")

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Portfolio Churn Rate</div>
            <div class="kpi-value">{kpis['overall_churn_rate']*100:.2f}%</div>
            <div class="{"kpi-delta-neg" if kpis['overall_churn_rate'] > 0.20 else "kpi-delta-pos"}">
                {kpis['churned_customers']:,} exited / {kpis['total_customers']:,} total
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Engagement Retention Ratio (ERR)</div>
            <div class="kpi-value">{kpis['err']:.2f}x</div>
            <div class="kpi-delta-pos">
                Active Churn: {kpis['active_churn_rate']*100:.1f}% vs Inactive: {kpis['inactive_churn_rate']*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">High-Bal. Disengagement (HBDR)</div>
            <div class="kpi-value">{kpis['hbdr']*100:.2f}%</div>
            <div class="kpi-delta-neg">
                {kpis['inactive_high_bal_count']:,} disengaged VIPs (>€100k)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Disengaged Deposits At-Risk</div>
            <div class="kpi-value">€{kpis['at_risk_balance']/1e6:.1f}M</div>
            <div class="kpi-delta-neg">
                {kpis['pct_balance_at_risk']:.1f}% of total balance exposure
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Credit Card Stickiness Delta</div>
            <div class="kpi-value">{kpis['cc_stickiness_delta']:+.2f}%</div>
            <div class="{"kpi-delta-pos" if kpis['cc_stickiness_delta'] > 1.0 else "kpi-delta-neg"}">
                Holder Ret.: {kpis['cc_retention']*100:.1f}% vs Non: {kpis['non_cc_retention']*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section: Macro Behavioral Profiles
    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.markdown("### 🧩 Behavioral Engagement Cohort Matrix")
        st.caption("Distribution of portfolio accounts, churn rates, and capital exposure across behavioral segments.")
        
        cohort_summary = compute_segment_exposure(filtered_df)
        
        fig_cohort = px.bar(
            cohort_summary,
            x='EngagementProfile',
            y='ChurnRate',
            color='AvgBalance',
            color_continuous_scale='Viridis',
            text=[f"{v*100:.1f}%" for v in cohort_summary['ChurnRate']],
            labels={'ChurnRate': 'Observed Churn Rate', 'EngagementProfile': 'Behavioral Cohort', 'AvgBalance': 'Avg Balance (€)'},
            title="Churn Rate by Engagement Profile (Color = Average Account Balance)"
        )
        fig_cohort.update_traces(textposition='outside')
        fig_cohort.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=80), xaxis_tickangle=-15)
        st.plotly_chart(fig_cohort, use_container_width=True)

    with c_right:
        st.markdown("### 🏛️ Capital Exposure by Behavioral Profile")
        st.caption("Share of total deposit liquidity held by each customer engagement segment.")
        
        fig_pie = px.pie(
            cohort_summary,
            values='TotalBalance',
            names='EngagementProfile',
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Total Deposit Distribution Across Profiles (€)"
        )
        fig_pie.update_traces(textinfo='percent+label', textposition='inside')
        fig_pie.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Detailed Cohort Table
    st.markdown("#### Detailed Cohort Metrics Breakdown")
    formatted_cohort = cohort_summary.copy()
    formatted_cohort['ChurnRate'] = formatted_cohort['ChurnRate'].apply(lambda x: f"{x*100:.2f}%")
    formatted_cohort['DepositShare'] = formatted_cohort['DepositShare'].apply(lambda x: f"{x*100:.2f}%")
    formatted_cohort['TotalBalance'] = formatted_cohort['TotalBalance'].apply(lambda x: f"€{x:,.2f}")
    formatted_cohort['AvgBalance'] = formatted_cohort['AvgBalance'].apply(lambda x: f"€{x:,.2f}")
    formatted_cohort['CapitalAtRisk'] = formatted_cohort['CapitalAtRisk'].apply(lambda x: f"€{x:,.2f}")
    formatted_cohort['AvgRSI'] = formatted_cohort['AvgRSI'].apply(lambda x: f"{x:.1f} / 100")
    
    st.dataframe(
        formatted_cohort.rename(columns={
            'EngagementProfile': 'Engagement Cohort Profile',
            'TotalCustomers': 'Account Count',
            'ChurnedCustomers': 'Exited Accounts',
            'ChurnRate': 'Churn Rate',
            'TotalBalance': 'Total Balance (€)',
            'AvgBalance': 'Avg Balance (€)',
            'AvgRSI': 'Avg RSI Score',
            'DepositShare': 'Deposit Share',
            'CapitalAtRisk': 'Capital At Flight Risk (€)'
        }),
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# TAB 2: ENGAGEMENT VS CHURN OVERVIEW
# =========================================================
with tab_eng:
    st.subheader("Behavioral Engagement Dynamics vs. Retention Stability")
    st.caption("Investigating the empirical relationship between digital/transactional activity, geography, and churn.")

    col_a, col_b = st.columns(2)

    with col_a:
        # Active vs Inactive Churn comparison
        act_summary = filtered_df.groupby('IsActiveMember')['Exited'].agg(
            Count='count',
            ChurnRate='mean'
        ).reset_index()
        act_summary['Status'] = act_summary['IsActiveMember'].map({0: 'Inactive Member', 1: 'Active Member'})
        act_summary['RetentionRate'] = 1 - act_summary['ChurnRate']

        fig_act = px.bar(
            act_summary,
            x='Status',
            y=['RetentionRate', 'ChurnRate'],
            barmode='stack',
            labels={'value': 'Proportion', 'variable': 'Retention Outcome'},
            title="Retention vs. Churn by Active Member Status",
            color_discrete_map={'RetentionRate': '#059669', 'ChurnRate': '#DC2626'}
        )
        fig_act.update_layout(height=360, yaxis_tickformat='.0%')
        st.plotly_chart(fig_act, use_container_width=True)

    with col_b:
        # Geography vs Engagement Churn
        geo_act = filtered_df.groupby(['Geography', 'IsActiveMember'])['Exited'].mean().reset_index()
        geo_act['Status'] = geo_act['IsActiveMember'].map({0: 'Inactive', 1: 'Active'})
        
        fig_geo = px.bar(
            geo_act,
            x='Geography',
            y='Exited',
            color='Status',
            barmode='group',
            labels={'Exited': 'Churn Rate'},
            title="Churn Disparity Across European Geographies by Activity",
            color_discrete_map={'Inactive': '#EF4444', 'Active': '#3B82F6'}
        )
        fig_geo.update_layout(height=360, yaxis_tickformat='.1%')
        st.plotly_chart(fig_geo, use_container_width=True)

    st.markdown("---")
    
    # Financial Commitment vs Activity Cross-Analysis
    st.markdown("### 💰 Financial Commitment vs. Engagement Cross-Analysis")
    st.caption("Addressing the core banking paradox: Why do high-balance customers exhibit significantly higher attrition?")

    col_c, col_d = st.columns(2)

    with col_c:
        fig_box = px.box(
            filtered_df,
            x='EngagementProfile',
            y='Balance',
            color='Exited',
            labels={'Exited': 'Exited Bank', 'Balance': 'Account Balance (€)'},
            title="Balance Distribution across Engagement Cohorts by Churn Status",
            color_discrete_map={0: '#10B981', 1: '#EF4444'}
        )
        fig_box.update_layout(height=380, xaxis_tickangle=-15)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_d:
        # Salary to Balance Mismatch vs Churn
        sample_df = filtered_df.sample(min(1500, len(filtered_df)), random_state=42)
        fig_scatter = px.scatter(
            sample_df,
            x='EstimatedSalary',
            y='Balance',
            color=sample_df['Exited'].map({0: 'Retained', 1: 'Churned'}),
            symbol=sample_df['IsActiveMember'].map({1: 'Active', 0: 'Inactive'}),
            opacity=0.7,
            labels={'EstimatedSalary': 'Estimated Annual Salary (€)', 'Balance': 'Account Balance (€)'},
            title="Salary-Balance Relationship & Activity Clustering (Sample)",
            color_discrete_map={'Retained': '#3B82F6', 'Churned': '#EF4444'}
        )
        fig_scatter.update_layout(height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.info("""
    **Core Diagnostic Takeaway:** 
    1. Zero-balance accounts churn at only **13.8%**, whereas positive-balance accounts churn at **24.1%**.
    2. German customers maintain an average balance of **€119,730** (0% zero-balance accounts) and experience **32.4% churn**, compared to France (**16.2%**) and Spain (**16.7%**).
    3. Wealth is not a protective barrier against attrition; unengaged wealth represents a primary source of capital instability.
    """)

# =========================================================
# TAB 3: PRODUCT UTILIZATION IMPACT ANALYSIS
# =========================================================
with tab_prod:
    st.subheader("Product Depth, Bundling Dynamics & The 2-Product Golden Zone")
    st.caption("Quantitative proof of non-linear retention dynamics across product count.")

    col_p1, col_p2 = st.columns([1.1, 1])

    with col_p1:
        prod_churn = filtered_df.groupby('NumOfProducts')['Exited'].agg(
            Count='count',
            ChurnRate='mean'
        ).reset_index()
        prod_churn['RetentionRate'] = 1 - prod_churn['ChurnRate']

        fig_prod = go.Figure()
        fig_prod.add_trace(go.Bar(
            x=prod_churn['NumOfProducts'],
            y=prod_churn['Count'],
            name='Customer Count',
            yaxis='y',
            marker_color='#94A3B8',
            opacity=0.6
        ))
        fig_prod.add_trace(go.Scatter(
            x=prod_churn['NumOfProducts'],
            y=prod_churn['ChurnRate'],
            name='Churn Rate',
            yaxis='y2',
            mode='lines+markers+text',
            text=[f"{v*100:.1f}%" for v in prod_churn['ChurnRate']],
            textposition='top center',
            line=dict(color='#DC2626', width=3),
            marker=dict(size=10, color='#DC2626')
        ))

        fig_prod.update_layout(
            title="The Product Depth Paradox: Volume vs Churn Rate",
            xaxis=dict(title="Number of Products Held", tickvals=[1, 2, 3, 4]),
            yaxis=dict(title="Customer Count", side='left'),
            yaxis2=dict(title="Churn Rate", side='right', overlaying='y', tickformat='.0%', range=[0, 1.1]),
            legend=dict(x=0.05, y=0.95),
            height=400
        )
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_p2:
        # Cross-table of Products and Activity
        prod_act = filtered_df.groupby(['NumOfProducts', 'IsActiveMember'])['Exited'].mean().reset_index()
        prod_act['Status'] = prod_act['IsActiveMember'].map({0: 'Inactive', 1: 'Active'})

        fig_prod_act = px.bar(
            prod_act,
            x='NumOfProducts',
            y='Exited',
            color='Status',
            barmode='group',
            labels={'Exited': 'Churn Rate', 'NumOfProducts': 'Number of Products'},
            title="Product Count vs Churn by Engagement Status",
            color_discrete_map={'Inactive': '#F97316', 'Active': '#10B981'}
        )
        fig_prod_act.update_layout(height=400, yaxis_tickformat='.0%', xaxis=dict(tickvals=[1, 2, 3, 4]))
        st.plotly_chart(fig_prod_act, use_container_width=True)

    st.markdown("### 🔍 Product Utilization Analytical Findings")
    
    c_card1, c_card2, c_card3 = st.columns(3)
    with c_card1:
        st.markdown("""
        **1. Single-Product Vulnerability**
        - **50.8%** of the portfolio holds only 1 product.
        - Churn rate: **27.71%**.
        - Lack of secondary anchors makes switching friction virtually zero.
        """)
    with c_card2:
        st.markdown("""
        **2. The 2-Product "Golden Ratio"**
        - **45.9%** of the portfolio holds 2 products.
        - Churn rate plunges to **7.58%** (a **72.6% relative churn reduction**).
        - Primary transaction + secondary anchor (e.g. mortgage/savings) maximizes loyalty.
        """)
    with c_card3:
        st.markdown("""
        **3. Multi-Product Friction / Hockey-Stick**
        - 3 products: **82.71%** churn rate.
        - 4 products: **100.00%** churn rate.
        - Symptoms of aggressive predatory cross-selling, unrequested accounts, or distress consolidations.
        """)

    st.markdown("---")
    
    # Credit Card Ownership Analysis
    st.subheader("💳 Credit Card Stickiness vs Active Utilization")
    col_cc1, col_cc2 = st.columns(2)
    
    with col_cc1:
        cc_summary = filtered_df.groupby(['HasCrCard', 'IsActiveMember'])['Exited'].agg(
            Count='count',
            ChurnRate='mean'
        ).reset_index()
        cc_summary['Cardholder'] = cc_summary['HasCrCard'].map({0: 'No Credit Card', 1: 'Holds Credit Card'})
        cc_summary['Activity'] = cc_summary['IsActiveMember'].map({0: 'Inactive', 1: 'Active'})

        fig_cc = px.bar(
            cc_summary,
            x='Cardholder',
            y='ChurnRate',
            color='Activity',
            barmode='group',
            labels={'ChurnRate': 'Observed Churn Rate'},
            title="Credit Card Retention Impact Conditioned on Active Status",
            color_discrete_map={'Inactive': '#EF4444', 'Active': '#10B981'}
        )
        fig_cc.update_layout(height=340, yaxis_tickformat='.1%')
        st.plotly_chart(fig_cc, use_container_width=True)

    with col_cc2:
        st.markdown("""
        #### Critical Credit Card Policy Insight:
        - Standalone Credit Card Ownership yields **no meaningful retention benefit** (`20.18%` churn for cardholders vs `20.81%` for non-cardholders, p = 0.498, statistically insignificant).
        - However, **Credit Card + Active Member status** yields **14.2%** churn vs **27.1%** for inactive cardholders!
        - **Actionable Bank Strategy**: Stop rewarding sales teams for raw credit card issuances. Restructure incentives around card **activation, recurring subscriptions, and direct debit utility payments**.
        """)

# =========================================================
# TAB 4: HIGH-VALUE DISENGAGED DETECTOR
# =========================================================
with tab_detector:
    st.subheader("🎯 High-Value Disengaged Customer Lead Engine")
    st.caption("Algorithmic identification and prioritization of dormant, high-capital accounts vulnerable to silent churn.")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        vip_min_balance = st.number_input("VIP Account Balance Threshold (€)", value=100000, step=10000)
    with col_f2:
        vip_risk_filter = st.multiselect(
            "Retention Tier Filter",
            ['Critical Disengagement (Flight Risk)', 'Vulnerable', 'Moderate Engagement', 'Sticky Anchor (Very High Loyalty)'],
            default=['Critical Disengagement (Flight Risk)', 'Vulnerable']
        )
    with col_f3:
        sort_by = st.selectbox("Sort Priority By", ["Balance (Descending)", "RSI (Ascending)", "EstimatedSalary (Descending)"])

    # Query detector
    vip_leads = filtered_df[
        (filtered_df['Balance'] >= vip_min_balance) &
        (filtered_df['RetentionTier'].isin(vip_risk_filter))
    ].copy()

    if sort_by == "Balance (Descending)":
        vip_leads = vip_leads.sort_values(by='Balance', ascending=False)
    elif sort_by == "RSI (Ascending)":
        vip_leads = vip_leads.sort_values(by='RelationshipStrengthIndex', ascending=True)
    else:
        vip_leads = vip_leads.sort_values(by='EstimatedSalary', ascending=False)

    col_met1, col_met2, col_met3 = st.columns(3)
    with col_met1:
        st.metric("Identified At-Risk VIP Accounts", f"{len(vip_leads):,} accounts")
    with col_met2:
        st.metric("Total Vulnerable Deposit Volume", f"€{vip_leads['Balance'].sum():,.2f}")
    with col_met3:
        st.metric("Average Relationship Strength Index", f"{vip_leads['RelationshipStrengthIndex'].mean():.1f} / 100")

    st.markdown("#### Prioritized Outreach Roster")
    display_cols = [
        'CustomerId', 'Surname', 'Geography', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'RelationshipStrengthIndex', 'RetentionTier', 'Exited'
    ]

    st.dataframe(
        vip_leads[display_cols].rename(columns={
            'CustomerId': 'Customer ID',
            'NumOfProducts': 'Products',
            'HasCrCard': 'Credit Card',
            'IsActiveMember': 'Active',
            'RelationshipStrengthIndex': 'RSI Score',
            'RetentionTier': 'Risk Tier',
            'Exited': 'Historical Churn'
        }),
        use_container_width=True,
        hide_index=True
    )

    # Download button for relationship managers
    csv_data = vip_leads[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Prioritized Outreach Roster (CSV for CRM Direct Outreach)",
        data=csv_data,
        file_name="high_value_disengaged_leads.csv",
        mime="text/csv",
        type="primary"
    )

# =========================================================
# TAB 5: RETENTION WHAT-IF STRATEGY SIMULATOR
# =========================================================
with tab_sim:
    st.subheader("📈 Retention Strategy What-If Simulator & Capital Preservation Engine")
    st.caption("Model the projected capital protection and churn reduction from three targeted behavioral interventions.")

    col_sim_left, col_sim_right = st.columns([1, 1.2])

    with col_sim_left:
        st.markdown("### 🎛️ Intervention Levers")
        
        sim_reengage = st.slider(
            "1. Inactive Digital Re-engagement Rate (%)",
            min_value=0.0, max_value=50.0, value=20.0, step=5.0,
            help="% of inactive customers successfully converted into active transacting members via digital prompts, app features, and cash-back incentives."
        )

        sim_cross_sell = st.slider(
            "2. Single-to-Optimal-2 Cross-Sell Rate (%)",
            min_value=0.0, max_value=40.0, value=15.0, step=5.0,
            help="% of single-product holders successfully cross-sold to the 2-product golden ratio (e.g. automatic savings account or auto-invest)."
        )

        sim_concierge = st.slider(
            "3. High-Balance Concierge Churn Reduction (%)",
            min_value=0.0, max_value=50.0, value=25.0, step=5.0,
            help="Relative churn risk reduction achieved through dedicated wealth relationship managers contacting disengaged clients with >€100k balance."
        )

        sim_nim = st.number_input(
            "Bank Net Interest Margin / Deposit Yield (%)",
            min_value=0.5, max_value=6.0, value=2.5, step=0.25,
            help="Net annualized interest margin earned by the bank on retained customer deposits."
        )

    sim_res = simulate_retention_interventions(
        filtered_df,
        reengagement_conversion_pct=sim_reengage,
        single_to_multi_cross_sell_pct=sim_cross_sell,
        high_bal_concierge_retention_lift=sim_concierge,
        annual_nim_pct=sim_nim
    )

    with col_sim_right:
        st.markdown("### 🎯 Projected Simulation Outcomes")

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Projected Portfolio Churn</div>
                <div class="kpi-value">{sim_res['simulated_churn_rate']*100:.2f}%</div>
                <div class="kpi-delta-pos">
                    Down from {sim_res['baseline_churn_rate']*100:.2f}% (-{sim_res['churn_rate_reduction_bps']:.0f} bps)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with sc2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Customers Retained</div>
                <div class="kpi-value">{sim_res['total_customers_saved']:,}</div>
                <div class="kpi-delta-pos">
                    Accounts prevented from exiting
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        sc3, sc4 = st.columns(2)
        with sc3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Deposit Capital Preserved</div>
                <div class="kpi-value">€{sim_res['total_capital_preserved']/1e6:.2f}M</div>
                <div class="kpi-delta-pos">
                    Protected from liquidity run-off
                </div>
            </div>
            """, unsafe_allow_html=True)

        with sc4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Annual Net Interest Protected</div>
                <div class="kpi-value">€{sim_res['annual_revenue_protected']/1e6:.2f}M</div>
                <div class="kpi-delta-pos">
                    At {sim_nim:.2f}% annual margin
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Strategy attribution waterfall / bar
    st.markdown("### 📊 Strategy Attribution: Customers Saved by Initiative")
    strat_df = pd.DataFrame([
        {'Strategy': k, 'Customers Saved': v['customers_saved'], 'Capital Saved (€M)': v['capital_saved']/1e6}
        for k, v in sim_res['strategy_breakdown'].items()
    ])

    fig_strat = px.bar(
        strat_df,
        x='Strategy',
        y='Customers Saved',
        color='Capital Saved (€M)',
        color_continuous_scale='Tealgrn',
        text='Customers Saved',
        title="Attribution of Retained Accounts Across Levers (Color = Deposit Capital Preserved in €M)"
    )
    fig_strat.update_traces(textposition='outside')
    fig_strat.update_layout(height=360)
    st.plotly_chart(fig_strat, use_container_width=True)

# =========================================================
# TAB 6: RESEARCH FINDINGS & METHODOLOGY
# =========================================================
with tab_research:
    st.subheader("📑 Empirical Methodology, Statistical Hypothesis Tests & Models")
    st.caption("Formal statistical validations supporting all conclusions and predictive modeling benchmarks.")

    # 1. Hypothesis tests table
    st.markdown("### 1. Formal Hypothesis Testing Roster")
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
    
    st.dataframe(pd.DataFrame(hyp_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Machine Learning Behavioral Feature Importance & Odds Ratios
    st.markdown("### 2. Behavioral vs. Demographic Feature Ranking")
    st.caption("Validating that engagement metrics dominate traditional demographic predictors.")
    
    models_res = train_behavioral_models(raw_df)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_rf = px.bar(
            models_res['rf_importance'],
            x='Importance',
            y='Feature',
            orientation='h',
            color='Category',
            title="Random Forest Gini Feature Importance",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_rf.update_layout(yaxis=dict(autorange='reversed'), height=400)
        st.plotly_chart(fig_rf, use_container_width=True)

    with col_m2:
        fig_or = px.bar(
            models_res['odds_ratios'],
            x='OddsRatio',
            y='Feature',
            orientation='h',
            color='Direction',
            title="Logistic Regression Odds Ratios (Impact on Churn)",
            color_discrete_map={'Risk Factor (+)': '#EF4444', 'Retention Factor (-)': '#10B981'}
        )
        fig_or.add_vline(x=1.0, line_dash='dash', line_color='black')
        fig_or.update_layout(yaxis=dict(autorange='reversed'), height=400)
        st.plotly_chart(fig_or, use_container_width=True)

    st.markdown("""
    ### 3. Executive Research Summary
    - **Hypothesis 1 Confirmed**: Digital activity is an indispensable retention shield ($p = 8.79 \\times 10^{-55}$). Disengaged members have nearly **2x** the churn rate of active peers.
    - **Hypothesis 2 Confirmed**: The product depth curve is acutely non-linear ($p < 10^{-100}$). Two products represent the mathematical optimal relationship depth. Three or more products reflect forced bundling friction.
    - **Hypothesis 3 Disproved**: High balances do **not** ensure loyalty ($p = 2.45 \\times 10^{-32}$). Inactive high-balance customers carry **32.8%** attrition risk, holding over **€312 Million** (40.8% of portfolio deposits).
    """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 12px; padding: 10px 0;">
    Customer Engagement & Product Utilization Analytics Platform | Developed for Unified Mentor Banking Retention Research & European Central Bank Case Analysis | Aryan
</div>
""", unsafe_allow_html=True)
