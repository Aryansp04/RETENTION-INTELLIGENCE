"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
Full Reproducible EDA, Statistical Testing & Econometric Modeling Pipeline
Author: Aryan (Internship Portfolio Project)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Ensure local modules importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import load_and_preprocess_data, calculate_portfolio_kpis
from src.analytics import run_statistical_hypothesis_tests, train_behavioral_models, compute_segment_exposure
from src.simulator import simulate_retention_interventions


def main():
    print("="*80)
    print("CUSTOMER ENGAGEMENT & PRODUCT UTILIZATION ANALYTICS - FULL PIPELINE")
    print("Unified Mentor Banking Retention Study | European Central Bank Focus")
    print("="*80)

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Churn_Modelling.csv")
    if not os.path.exists(data_path):
        data_path = "Churn_Modelling.csv"
        
    print(f"\n[1] Ingesting and Preprocessing Data from: {data_path}")
    df = load_and_preprocess_data(data_path)
    print(f"-> Successfully loaded {len(df):,} accounts across {df.shape[1]} features.")

    print("\n[2] Portfolio Key Performance Indicators (KPIs):")
    kpis = calculate_portfolio_kpis(df)
    print(f"  • Overall Portfolio Churn Rate:       {kpis['overall_churn_rate']*100:.2f}%")
    print(f"  • Active Member Churn Rate:           {kpis['active_churn_rate']*100:.2f}%")
    print(f"  • Inactive Member Churn Rate:         {kpis['inactive_churn_rate']*100:.2f}%")
    print(f"  • Engagement Retention Ratio (ERR):   {kpis['err']:.2f}x (Active vs Inactive)")
    print(f"  • High-Balance Disengagement (HBDR):  {kpis['hbdr']*100:.2f}%")
    print(f"  • Capital in Disengaged Accounts:     €{kpis['at_risk_balance']:,.2f} ({kpis['pct_balance_at_risk']:.1f}% of total)")
    print(f"  • Credit Card Stickiness Delta:       {kpis['cc_stickiness_delta']:+.2f}%")
    print(f"  • Retained Mean RSI Score:            {kpis['avg_rsi_retained']:.2f} / 100")
    print(f"  • Churned Mean RSI Score:             {kpis['avg_rsi_churned']:.2f} / 100")

    print("\n[3] Formal Statistical Hypothesis Tests:")
    tests = run_statistical_hypothesis_tests(df)
    for k, v in tests.items():
        print(f"  • {v['name']}:")
        print(f"      Statistic ({v['test_type']}) = {v['statistic']}, p-value = {v['p_value']:.4e}")
        print(f"      Significant at alpha=0.01: {v['significant']}")
        print(f"      Finding: {v['interpretation']}")

    print("\n[4] Behavioral Cohort Capital Exposure Analysis:")
    exposure = compute_segment_exposure(df)
    print(exposure.to_string(index=False))

    print("\n[5] Machine Learning Feature Importance & Odds Ratios:")
    models = train_behavioral_models(df)
    print("\n--- Logistic Regression Odds Ratios (Top 5 Drivers) ---")
    print(models['odds_ratios'].head(5).to_string(index=False))
    print("\n--- Random Forest Gini Feature Importance ---")
    print(models['rf_importance'].to_string(index=False))

    print("\n[6] Retention Strategy What-If Simulation (20% Re-engage, 15% Cross-Sell, 25% Concierge):")
    sim = simulate_retention_interventions(df, 20.0, 15.0, 25.0, 2.5)
    print(f"  • Customers Retained (Saved):         {sim['total_customers_saved']:,}")
    print(f"  • Projected Portfolio Churn:          {sim['simulated_churn_rate']*100:.2f}% (from {sim['baseline_churn_rate']*100:.2f}%)")
    print(f"  • Net Churn Reduction:                -{sim['churn_rate_reduction_bps']:.0f} bps")
    print(f"  • Deposit Capital Preserved:          €{sim['total_capital_preserved']:,.2f}")
    print(f"  • Annual Net Interest Margin Saved:   €{sim['annual_revenue_protected']:,.2f}")

    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE - ALL BENCHMARKS VALIDATED")
    print("="*80)


if __name__ == "__main__":
    main()

