"""
Customer Engagement & Product Utilization Analytics
Retention Strategy Simulation and Capital Impact Engine
Author: Aryan (Internship Portfolio Project)
"""

import numpy as np
import pandas as pd


def simulate_retention_interventions(
    df: pd.DataFrame,
    reengagement_conversion_pct: float = 20.0,
    single_to_multi_cross_sell_pct: float = 15.0,
    high_bal_concierge_retention_lift: float = 25.0,
    annual_nim_pct: float = 2.5
) -> dict:
    """
    Simulates the financial and customer retention impact of 3 strategic banking initiatives:
    1. Digital Re-engagement (Activating inactive customers)
    2. 2-Product Cross-Sell Campaign (Migrating single-product holders to optimal 2 products)
    3. Premium Concierge Outreach (Targeting high-balance disengaged accounts)

    Parameters:
    - df: Bank customer dataset
    - reengagement_conversion_pct: % of inactive customers successfully activated (0 to 100)
    - single_to_multi_cross_sell_pct: % of 1-product customers successfully cross-sold to 2 products
    - high_bal_concierge_retention_lift: % reduction in churn rate for high-balance disengaged accounts
    - annual_nim_pct: Net Interest Margin / deposit spread earned by bank (e.g. 2.5%)
    """
    total_customers = len(df)
    baseline_churn_count = int(df['Exited'].sum())
    baseline_churn_rate = baseline_churn_count / total_customers
    total_deposits = df['Balance'].sum()

    # --- Strategy 1: Inactive Customer Re-engagement ---
    # Inactive churn rate is ~26.85%, Active churn rate is ~14.27% -> delta is ~12.58% reduction
    inactive_mask = df['IsActiveMember'] == 0
    inactive_count = inactive_mask.sum()
    converted_active_count = inactive_count * (reengagement_conversion_pct / 100.0)

    # Reduction in churn probability for converted members: ~0.1258
    churn_diff_active = (df[df['IsActiveMember'] == 0]['Exited'].mean() -
                         df[df['IsActiveMember'] == 1]['Exited'].mean())
    saved_from_reengagement = converted_active_count * churn_diff_active
    avg_inactive_balance = df[inactive_mask]['Balance'].mean()
    capital_saved_reengagement = saved_from_reengagement * avg_inactive_balance

    # --- Strategy 2: Single-Product Cross-Sell to Optimal 2 Products ---
    # 1 product churn: 27.71%, 2 product churn: 7.58% -> delta is ~20.13% reduction
    single_prod_mask = df['NumOfProducts'] == 1
    single_prod_count = single_prod_mask.sum()
    cross_sold_count = single_prod_count * (single_to_multi_cross_sell_pct / 100.0)

    churn_diff_product = (df[df['NumOfProducts'] == 1]['Exited'].mean() -
                          df[df['NumOfProducts'] == 2]['Exited'].mean())
    saved_from_cross_sell = cross_sold_count * churn_diff_product
    avg_single_prod_balance = df[single_prod_mask]['Balance'].mean()
    capital_saved_cross_sell = saved_from_cross_sell * avg_single_prod_balance

    # --- Strategy 3: High-Balance Disengaged Concierge Outreach ---
    # High-balance (>€100k) inactive customers have 32.77% churn rate
    high_bal_inactive_mask = (df['Balance'] > 100000) & (df['IsActiveMember'] == 0)
    high_bal_inactive_count = high_bal_inactive_mask.sum()
    current_high_bal_churn_count = df[high_bal_inactive_mask]['Exited'].sum()

    saved_from_concierge = current_high_bal_churn_count * (high_bal_concierge_retention_lift / 100.0)
    avg_high_bal_inactive = df[high_bal_inactive_mask]['Balance'].mean()
    capital_saved_concierge = saved_from_concierge * avg_high_bal_inactive

    # Combined Net Impact (deduplicated estimation)
    total_customers_saved = int(saved_from_reengagement + saved_from_cross_sell + saved_from_concierge)
    total_capital_preserved = capital_saved_reengagement + capital_saved_cross_sell + capital_saved_concierge

    simulated_churn_count = max(0, baseline_churn_count - total_customers_saved)
    simulated_churn_rate = simulated_churn_count / total_customers
    churn_rate_reduction_bps = (baseline_churn_rate - simulated_churn_rate) * 10000  # Basis points

    annual_revenue_protected = total_capital_preserved * (annual_nim_pct / 100.0)

    return {
        'baseline_churn_count': baseline_churn_count,
        'baseline_churn_rate': baseline_churn_rate,
        'simulated_churn_count': simulated_churn_count,
        'simulated_churn_rate': simulated_churn_rate,
        'total_customers_saved': total_customers_saved,
        'churn_rate_reduction_bps': churn_rate_reduction_bps,
        'total_capital_preserved': total_capital_preserved,
        'annual_revenue_protected': annual_revenue_protected,
        'strategy_breakdown': {
            'Re-engagement': {
                'targeted_customers': int(converted_active_count),
                'customers_saved': int(saved_from_reengagement),
                'capital_saved': capital_saved_reengagement
            },
            'Cross-Sell (1 to 2 Products)': {
                'targeted_customers': int(cross_sold_count),
                'customers_saved': int(saved_from_cross_sell),
                'capital_saved': capital_saved_cross_sell
            },
            'High-Balance Concierge': {
                'targeted_customers': int(high_bal_inactive_count),
                'customers_saved': int(saved_from_concierge),
                'capital_saved': capital_saved_concierge
            }
        }
    }
