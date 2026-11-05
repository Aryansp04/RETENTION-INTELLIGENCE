"""
Customer Engagement & Product Utilization Analytics
Data Processing and Feature Engineering Module
Author: Aryan (Internship Portfolio Project)
"""

import numpy as np
import pandas as pd


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads bank customer churn dataset, validates consistency,
    and performs domain-specific feature engineering.
    """
    df = pd.read_csv(file_path)

    # 1. Clean column names and check essential fields
    df.columns = [c.strip() for c in df.columns]

    # Ensure binary types
    binary_cols = ['HasCrCard', 'IsActiveMember', 'Exited']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # 2. Balance Categorization & Financial Commitment Ratios
    # Zero balance represents customers who use the bank for transacting/salary routing or have emptied accounts
    df['BalanceTier'] = pd.cut(
        df['Balance'],
        bins=[-np.inf, 0.01, 50000, 100000, np.inf],
        labels=['Zero Balance', 'Low (<€50k)', 'Medium (€50k-€100k)', 'High (>€100k)']
    )

    df['BalanceToSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'].replace(0, np.nan))
    df['BalanceToSalaryRatio'] = df['BalanceToSalaryRatio'].fillna(0)

    # 3. Product Depth Categorization
    def categorize_product_depth(num_prod):
        if num_prod == 1:
            return 'Single Product'
        elif num_prod == 2:
            return 'Two Products (Optimal)'
        elif num_prod == 3:
            return 'Three Products (High Friction)'
        else:
            return 'Four Products (Critical Flight)'

    df['ProductDepthCategory'] = df['NumOfProducts'].apply(categorize_product_depth)

    # 4. Engagement Classification (Methodology requirement)
    def classify_engagement(row):
        is_active = row['IsActiveMember'] == 1
        num_prod = row['NumOfProducts']
        balance = row['Balance']

        if not is_active and balance > 100000:
            return 'Inactive High-Balance (At-Risk Premium)'
        elif is_active and num_prod >= 2:
            return 'Active Highly-Engaged'
        elif is_active and num_prod == 1:
            return 'Active Low-Product'
        else:
            return 'Inactive Disengaged (Standard)'

    df['EngagementProfile'] = df.apply(classify_engagement, axis=1)

    # 5. Composite Relationship Strength Index (RSI: 0 - 100)
    # Formulated from empirical retention weights:
    # - Activity state: 35 points (strongest single behavioral stickiness driver)
    # - Product depth: 35 points for optimal 2 products, 15 for 1, 5 for 3, 0 for 4
    # - Tenure commitment: up to 15 points (proportional to years with bank)
    # - Active Credit Card interaction: 15 points (cardholder actively transacting)
    def calculate_rsi(row):
        score = 0.0

        # Activity Component (0 to 35)
        if row['IsActiveMember'] == 1:
            score += 35.0

        # Product Depth Component (0 to 35)
        if row['NumOfProducts'] == 2:
            score += 35.0
        elif row['NumOfProducts'] == 1:
            score += 15.0
        elif row['NumOfProducts'] == 3:
            score += 5.0
        elif row['NumOfProducts'] >= 4:
            score += 0.0

        # Tenure Loyalty Component (0 to 15)
        tenure_score = min(row['Tenure'] / 10.0, 1.0) * 15.0
        score += tenure_score

        # Active Credit Card Synergy Component (0 to 15)
        if row['HasCrCard'] == 1 and row['IsActiveMember'] == 1:
            score += 15.0
        elif row['HasCrCard'] == 1 and row['IsActiveMember'] == 0:
            score += 5.0  # Card held but dormant

        return round(score, 1)

    df['RelationshipStrengthIndex'] = df.apply(calculate_rsi, axis=1)

    # 6. Retention Stability / Risk Tiers
    def assign_retention_tier(rsi):
        if rsi >= 75:
            return 'Sticky Anchor (Very High Loyalty)'
        elif rsi >= 50:
            return 'Moderate Engagement'
        elif rsi >= 30:
            return 'Vulnerable'
        else:
            return 'Critical Disengagement (Flight Risk)'

    df['RetentionTier'] = df['RelationshipStrengthIndex'].apply(assign_retention_tier)

    return df


def calculate_portfolio_kpis(df: pd.DataFrame) -> dict:
    """
    Computes key performance indicators as specified in the project analytical mandate.
    """
    total_customers = len(df)
    churned_customers = int(df['Exited'].sum())
    retained_customers = total_customers - churned_customers
    overall_churn_rate = churned_customers / total_customers

    # 1. Engagement Retention Ratio (ERR)
    # Ratio of Active Member Retention Rate to Inactive Member Retention Rate
    active_df = df[df['IsActiveMember'] == 1]
    inactive_df = df[df['IsActiveMember'] == 0]

    active_churn = active_df['Exited'].mean()
    inactive_churn = inactive_df['Exited'].mean()

    active_retention = 1.0 - active_churn
    inactive_retention = 1.0 - inactive_churn

    err = active_retention / (inactive_retention if inactive_retention > 0 else 1.0)
    churn_odds_ratio = inactive_churn / (active_churn if active_churn > 0 else 1.0)

    # 2. Product Depth Index (PDI)
    # Retention efficacy across product lines
    pdi_by_prod = df.groupby('NumOfProducts')['Exited'].agg(
        total='count',
        churn_rate='mean',
        retention_rate=lambda x: 1.0 - x.mean()
    ).to_dict(orient='index')

    # 3. High-Balance Disengagement Rate (HBDR)
    high_bal_df = df[df['Balance'] > 100000]
    total_high_bal = len(high_bal_df)
    inactive_high_bal = len(high_bal_df[high_bal_df['IsActiveMember'] == 0])
    hbdr = (inactive_high_bal / total_high_bal) if total_high_bal > 0 else 0.0

    at_risk_balance = high_bal_df[high_bal_df['IsActiveMember'] == 0]['Balance'].sum()
    total_balance = df['Balance'].sum()
    pct_balance_at_risk = (at_risk_balance / total_balance * 100) if total_balance > 0 else 0.0

    # 4. Credit Card Stickiness Score (CCSS)
    cc_holders = df[df['HasCrCard'] == 1]
    non_cc_holders = df[df['HasCrCard'] == 0]

    cc_retention = 1.0 - cc_holders['Exited'].mean()
    non_cc_retention = 1.0 - non_cc_holders['Exited'].mean()
    cc_stickiness_delta = (cc_retention - non_cc_retention) * 100.0  # percentage point impact

    # 5. Relationship Strength Index (RSI) Summary
    avg_rsi_retained = df[df['Exited'] == 0]['RelationshipStrengthIndex'].mean()
    avg_rsi_churned = df[df['Exited'] == 1]['RelationshipStrengthIndex'].mean()
    avg_rsi_overall = df['RelationshipStrengthIndex'].mean()
    product_depth_index = df['NumOfProducts'].mean()

    return {
        'total_customers': total_customers,
        'churned_customers': churned_customers,
        'retained_customers': retained_customers,
        'overall_churn_rate': overall_churn_rate,
        'active_churn_rate': active_churn,
        'inactive_churn_rate': inactive_churn,
        'err': err,
        'engagement_retention_ratio': err,
        'churn_odds_ratio': churn_odds_ratio,
        'pdi_by_prod': pdi_by_prod,
        'product_depth_index': product_depth_index,
        'hbdr': hbdr,
        'high_balance_disengagement_rate': hbdr,
        'total_high_bal_count': total_high_bal,
        'inactive_high_bal_count': inactive_high_bal,
        'total_balance': total_balance,
        'at_risk_balance': at_risk_balance,
        'pct_balance_at_risk': pct_balance_at_risk,
        'cc_retention': cc_retention,
        'non_cc_retention': non_cc_retention,
        'cc_stickiness_delta': cc_stickiness_delta,
        'credit_card_stickiness_score': cc_stickiness_delta / 100.0,
        'avg_rsi_retained': avg_rsi_retained,
        'avg_rsi_churned': avg_rsi_churned,
        'avg_rsi_overall': avg_rsi_overall,
        'avg_relationship_strength': avg_rsi_overall,
    }

