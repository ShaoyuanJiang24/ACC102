# ACC102
# Profitability Analysis of U.S. Retail Apparel Firms (2019–2024)

## Project Overview

This project analyzes the financial performance of U.S. listed companies in the *Retail - Apparel & Specialty* industry using Python and SimFin data from 2019 to 2024.

The analysis focuses on how cost structure (Cost of Goods Sold and SG&A expenses) affects firm profitability, measured by Net Margin and Return on Assets (ROA).

---

## Objectives

* Clean and preprocess firm-level financial data
* Construct key financial ratios
* Analyze cost structure and profitability trends
* Visualize firm and industry-level patterns
* Estimate regression models to study relationships between cost ratios and profitability

---

## Dataset

* **Source:** SimFin
* **Market:** U.S.
* **Frequency:** Annual
* **Time Period:** 2019–2024
* **Industry:** Retail - Apparel & Specialty

### Data Included

* Income statement data
* Balance sheet data
* Company and industry classification

---

##  Methods

### 1. Data Cleaning

* Removed missing and invalid observations
* Filtered firms with at least 3 years of data
* Excluded atypical firms (e.g., AMZN, EBAY, HD, RGR) to ensure comparability
* Winsorized financial ratios at 1% and 99% to reduce outlier influence

### 2. Variable Construction

* COGS Ratio = COGS / Revenue
* SG&A Ratio = SG&A / Revenue
* Operating Margin = Operating Income / Revenue
* Net Margin = Net Income / Revenue
* ROA = Net Income / Total Assets
* Log Assets = log(Total Assets)

### 3. Analysis

* Descriptive statistics
* Company-level summary
* Year-level trend analysis
* Data visualization (line charts, scatter plots, bar charts)

### 4. Regression Models

Two OLS models with robust (HC3) standard errors:

**Model 1: Net Margin**
Net Margin ~ COGS Ratio + SG&A Ratio + Log Assets + Year Fixed Effects

**Model 2: ROA**
ROA ~ COGS Ratio + SG&A Ratio + Log Assets + Year Fixed Effects

---

##  Repository Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   └── acc102.py
├── notebooks/
│   └── acc102.ipynb
├── outputs/
│   ├── descriptive_statistics.csv
│   ├── company_summary.csv
│   ├── year_summary.csv
│   ├── cost_ratios_over_time.png
│   ├── profitability_over_time.png
│   ├── cogs_vs_net_margin.png
│   ├── sga_vs_net_margin.png
│   ├── top15_net_margin.png
│   ├── regression_net_margin.txt
│   ├── regression_roa.txt
│   └── regression_coefficients_*.csv



---

##  How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set SimFin API Key

```bash
export SIMFIN_API_KEY=your_api_key_here
```

(Windows)

```bash
set SIMFIN_API_KEY=your_api_key_here
```

### 3. Run the analysis

```bash
python src/retail_apparel_analysis.py
```

---

##  Key Results

* Cost ratios (COGS and SG&A) vary significantly across firms and over time
* Profitability indicators (Net Margin and ROA) show noticeable fluctuations between 2019–2024
* Firms with lower cost ratios tend to exhibit higher profitability
* Regression results suggest a systematic relationship between cost structure and profitability

---

##  Limitations

* The analysis focuses on a single industry
* Some firms are manually excluded to improve comparability
* Results are based on accounting data and may contain reporting inconsistencies
* Regression results indicate correlation, not causation

---

##  Future Improvements

* Extend analysis to multiple industries
* Apply panel data models (fixed effects)
* Compare pre- and post-COVID performance more formally
* Incorporate additional financial indicators

---


