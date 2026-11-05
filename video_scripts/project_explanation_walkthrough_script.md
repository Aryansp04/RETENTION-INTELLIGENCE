# Project Explanation & Technical Walkthrough Video Script

**Project Title:** Customer Engagement & Product Utilization Analytics for Retention Strategy  
**Presenter:** Aryan  
**Role:** Quantitative Data Analytics & Retention Strategy Intern  
**Target Duration:** 5 to 7 Minutes  
**Format:** Screen Recording + Camera-in-Corner (Loom / OBS / Google Meet style)  

---

## Technical Setup & Preparation Checklist Before Recording
- [ ] Launch the Streamlit dashboard in terminal: `streamlit run app.py`
- [ ] Open the dashboard in full screen at `http://localhost:8501`
- [ ] Ensure the browser zoom is set to $100\%$ or $110\%$ for crisp visualization readability.
- [ ] Have the GitHub repository and `RESEARCH_PAPER.md` open in adjacent browser tabs.
- [ ] Test microphone audio and lighting.

---

## Detailed Script with Visual Cues & Timestamps

### [0:00 - 0:45] Section 1: Hook, Introduction & Core Banking Paradox
**[Visual Cue: Camera Full Screen or Camera Top-Right over Dashboard Title Banner]**

**Spoken Words:**
> "Hello everyone! My name is Aryan, and today I am excited to walk you through my internship capstone project: **Customer Engagement & Product Utilization Analytics for Retention Strategy**.
> 
> In retail banking, there is a dangerous misconception that has quietly cost financial institutions billions of euros: the belief that customers with high balances and credit cards are naturally loyal. Banks routinely treat affluent account holders as safe assets. 
> 
> But when we look at actual customer behavior, that assumption falls apart. In this project, analyzing an empirical portfolio of 10,000 European bank accounts across France, Germany, and Spain, I proved that **behavioral engagement and product relationship depth—not demographic wealth—dictate customer retention**. 
> 
> Let's dive right into the live analytics dashboard I built to address this challenge."

---

### [0:45 - 2:00] Section 2: Executive KPI Overview & Behavioral Cohorts
**[Visual Cue: Screen Share on Dashboard — Tab 1: Executive KPI Overview]**
*(Cursor hovers smoothly across the 5 top KPI cards)*

**Spoken Words:**
> "Here on the **Executive KPI Overview**, we have distilled portfolio health into standardized quantitative metrics. 
> 
> Notice our baseline portfolio churn rate is **20.37%**. However, looking at the second card, our **Engagement Retention Ratio (ERR)** is **1.17x**, showing that active members churn at only **14.27%**, compared to **26.85%** for inactive members. Inactive customers are nearly twice as likely to walk out the door.
> 
> Even more alarming is the third metric: the **High-Balance Disengagement Rate (HBDR)**. Exactly **49.09%** of all customer accounts holding over €100,000 are completely disengaged. 
> 
> If we look at the capital exposure chart below, those disengaged VIP accounts hold over **€312.27 Million**—which accounts for **40.83% of the bank's entire deposit base**! Their observed churn rate is a staggering **32.77%**. That means over **€102 Million** in retail liquidity is at imminent flight risk because the bank failed to actively engage its most valuable depositors."

---

### [2:00 - 3:30] Section 3: Product Utilization & The "2-Product Golden Zone"
**[Visual Cue: Click on Tab 3: Product Depth & Utilization]**
*(Point to the combo bar-line chart showing Product Count vs. Churn Rate)*

**Spoken Words:**
> "Now let's examine the product utilization dynamics, which uncovered what I consider the most fascinating finding of this research.
> 
> Conventional banking dogma says: *'The more products you cross-sell a customer, the stickier they become.'* Our empirical analysis proves this is dangerously wrong beyond two products.
> 
> Take a look at this non-linear curve:
> - **1 Product:** Holds **50.8%** of customers, with a churn rate of **27.71%**. Having only one account means switching barriers are virtually non-existent.
> - **2 Products:** This is the **'Golden Zone'**. Churn plunges from 27.7% down to just **7.58%**! That is a **72.6% relative churn reduction**. Pairing a primary checking account with a secondary anchor—like a recurring savings account or mortgage—maximizes switching friction in a healthy way.
> - **3 & 4 Products:** But look what happens next: at 3 products, churn spikes to **82.71%**, and at 4 products, churn hits an unbelievable **100%**! 
> 
> In banking, when churn jumps from 7% to 100%, it indicates severe sales friction: customers were pushed into unwanted insurance or credit lines to hit branch sales quotas, or financially distressed borrowers were taking out multiple consolidation facilities right before defaulting. 
> 
> Furthermore, looking at our Credit Card analysis on the right, passive card ownership has a $p$-value of **0.498**—statistically zero retention benefit! A credit card only aids retention when paired with active, recurring transaction utility."

---

### [3:30 - 4:45] Section 4: Lead Detector & What-If Strategy Simulator
**[Visual Cue: Click on Tab 4: High-Value Disengaged Detector, then Tab 5: Retention What-If Simulator]**
*(Briefly show the filterable leads table and the CSV export button, then switch to Tab 5 and adjust the sliders)*

**Spoken Words:**
> "To make these insights actionable for branch managers, I built the **High-Value Disengaged Detector** in Tab 4. With one click, relationship managers can filter dormant VIP accounts holding over €100,000, review their Relationship Strength Index, and export a prioritized outreach roster directly into their CRM.
> 
> But for executive leadership and the Board of Directors, the ultimate question is: *What is the financial return of acting on these findings?*
> 
> That brings us to Tab 5: our **Retention What-If Strategy Simulator**.
> 
> Here, bank executives can model three realistic behavioral interventions:
> 1. A digital re-engagement push converting 20% of inactive users.
> 2. A cross-sell campaign transitioning 15% of single-product users into the 2-product golden zone.
> 3. A dedicated relationship concierge team targeting high-balance dormant accounts to achieve a 25% churn reduction.
> 
> Look at the real-time calculated impact:
> - We save **468 customer accounts** from defecting.
> - We slash overall portfolio churn from **20.37% down to 15.69%**—a massive **468 basis point improvement**.
> - Most importantly, we preserve **€50.12 Million in deposit capital** and protect **€1.25 Million in recurring Net Interest Margin** each year."

---

### [4:45 - 6:00] Section 5: Technical Architecture, Methodology & Conclusion
**[Visual Cue: Click on Tab 6: Research Findings & Methodology, or display GitHub Repo README]**

**Spoken Words:**
> "Under the hood, this entire platform is architected with modular, production-grade Python:
> - Robust feature engineering in `data_processing.py` formulating the composite **Relationship Strength Index (RSI)**.
> - Formal econometric and hypothesis testing in `analytics.py`, incorporating Pearson's Chi-Square tests, Mann-Whitney U distributions, and Random Forest Gini feature importances—which confirmed that behavioral features outweigh traditional demographics like salary by nearly 10 to 1.
> - Real-time scenario projection in `simulator.py`.
> - And a responsive, interactive UI built using **Streamlit** and **Plotly**.
> 
> In conclusion, this project reframes customer retention from demographic guesswork into precision behavioral science. It demonstrates how retail banks can protect deposit reserves, eliminate toxic cross-selling, and safeguard balance sheet stability.
> 
> The complete source code, research paper, and stakeholder executive summary are documented on my GitHub repository. Thank you so much for watching!"

---

## Post-Recording Tips
- Review the video for clear pacing and confident voice delivery.
- Ensure the recorded video resolution is at least 1080p.
- Upload to YouTube (Unlisted or Public) or Google Drive (with public view permissions) and copy the link for the internship submission.
