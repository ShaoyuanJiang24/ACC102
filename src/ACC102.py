import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import simfin as sf


# =========================
# 0. Basic settings
# =========================
warnings.filterwarnings(
    "ignore",
    message=".*date_parser.*deprecated.*",
    category=FutureWarning,
    module=r"simfin\.load"
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

API_KEY = "a9f00d54-604b-42f2-9757-4bdfb780d46b"
DATA_DIR = "simfin_data"
OUTPUT_DIR = Path("outputs_robustness")
OUTPUT_DIR.mkdir(exist_ok=True)

START_YEAR = 2019
END_YEAR = 2024
TARGET_INDUSTRY = "Retail - Apparel & Specialty"
MIN_YEARS_PER_FIRM = 3

# 手动剔除 atypical firms
EXCLUDED_TICKERS = ["AMZN", "EBAY", "HD", "RGR"]


# =========================
# 1. Utility functions
# =========================
def winsorize_series(s, lower=0.01, upper=0.99):
    q_low = s.quantile(lower)
    q_high = s.quantile(upper)
    return s.clip(lower=q_low, upper=q_high)


# =========================
# 2. Load SimFin data
# =========================
sf.set_api_key(API_KEY)
sf.set_data_dir(DATA_DIR)

print("Loading SimFin datasets...")
income = sf.load_income(variant="annual", market="us").reset_index()
balance = sf.load_balance(variant="annual", market="us").reset_index()
companies = sf.load_companies(market="us").reset_index()
industries = sf.load_industries().reset_index()
print("Data loaded.\n")


# =========================
# 3. Merge company + industry info
# =========================
company_industry = companies.merge(industries, on="IndustryId", how="left")

target_companies = company_industry.loc[
    company_industry["Industry"].eq(TARGET_INDUSTRY),
    ["Ticker", "SimFinId", "Company Name", "Sector", "Industry"]
].drop_duplicates()

print("Target industry:", TARGET_INDUSTRY)
print("Number of firms before exclusion:", target_companies["Ticker"].nunique())
print("\nExcluded tickers:", EXCLUDED_TICKERS)

target_companies = target_companies[
    ~target_companies["Ticker"].isin(EXCLUDED_TICKERS)
].copy()

print("Number of firms after exclusion:", target_companies["Ticker"].nunique())
print("\nSample firms after exclusion:")
print(target_companies.head(20))
print()


# =========================
# 4. Merge financial data
# =========================
balance_keep = balance[
    ["Ticker", "SimFinId", "Report Date", "Fiscal Year", "Total Assets"]
].copy()

df = income.merge(
    balance_keep,
    on=["Ticker", "SimFinId", "Report Date", "Fiscal Year"],
    how="left"
)

df = df.merge(
    target_companies,
    on=["Ticker", "SimFinId"],
    how="inner"
)

# Keep target years only
df = df[(df["Fiscal Year"] >= START_YEAR) & (df["Fiscal Year"] <= END_YEAR)].copy()

# Keep annual observations
if "Fiscal Period" in df.columns:
    df = df[df["Fiscal Period"].isin(["FY", "Q4", "Annual"]) | df["Fiscal Period"].isna()].copy()

print("Merged panel shape:", df.shape)
print()


# =========================
# 5. Keep required columns
# =========================
required_cols = [
    "Ticker",
    "Company Name",
    "Sector",
    "Industry",
    "Fiscal Year",
    "Report Date",
    "Revenue",
    "Cost of Revenue",
    "Selling, General & Administrative",
    "Operating Income (Loss)",
    "Net Income",
    "Total Assets"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise KeyError(f"Missing required columns: {missing_cols}")

df = df[required_cols].copy()

df = df.rename(columns={
    "Selling, General & Administrative": "SGA",
    "Operating Income (Loss)": "Operating_Income",
    "Net Income": "Net_Income",
    "Cost of Revenue": "COGS",
    "Fiscal Year": "Fiscal_Year"
})

# Basic cleaning
df = df.replace([np.inf, -np.inf], np.nan)
df = df[df["Revenue"].notna() & (df["Revenue"] > 0)].copy()
df = df.dropna(subset=["COGS", "SGA", "Operating_Income", "Net_Income", "Total Assets"]).copy()
df = df[df["Total Assets"] > 0].copy()


# =========================
# 6. Construct variables
# =========================
df["COGS_ratio"] = -df["COGS"] / df["Revenue"]
df["SGA_ratio"] = -df["SGA"] / df["Revenue"]
df["Operating_margin"] = df["Operating_Income"] / df["Revenue"]
df["Net_margin"] = df["Net_Income"] / df["Revenue"]
df["ROA"] = df["Net_Income"] / df["Total Assets"]
df["Log_Assets"] = np.log(df["Total Assets"])

# Remove unreasonable values
df = df[df["COGS_ratio"].between(0, 2)].copy()
df = df[df["SGA_ratio"].between(0, 2)].copy()
df = df[df["Operating_margin"].between(-2, 2)].copy()
df = df[df["Net_margin"].between(-2, 2)].copy()
df = df[df["ROA"].between(-2, 2)].copy()

# Keep firms with at least MIN_YEARS_PER_FIRM observations
firm_year_counts = df.groupby("Ticker")["Fiscal_Year"].nunique().reset_index(name="Years")
eligible_tickers = firm_year_counts.loc[
    firm_year_counts["Years"] >= MIN_YEARS_PER_FIRM, "Ticker"
]
df = df[df["Ticker"].isin(eligible_tickers)].copy()

# Winsorize key ratios
for col in ["COGS_ratio", "SGA_ratio", "Operating_margin", "Net_margin", "ROA"]:
    df[col] = winsorize_series(df[col], 0.01, 0.99)

print("Cleaned robustness dataset shape:", df.shape)
print("\nPreview:")
print(df.head())
print()


# =========================
# 7. Save cleaned data
# =========================
cleaned_path = OUTPUT_DIR / "retail_apparel_specialty_robustness_cleaned.csv"
df.to_csv(cleaned_path, index=False)
print(f"Saved cleaned dataset to: {cleaned_path}")


# =========================
# 8. Descriptive statistics
# =========================
desc = df[
    ["Revenue", "COGS", "SGA", "Operating_Income", "Net_Income",
     "COGS_ratio", "SGA_ratio", "Operating_margin", "Net_margin", "ROA"]
].describe()

desc_path = OUTPUT_DIR / "descriptive_statistics.csv"
desc.to_csv(desc_path)
print(f"Saved descriptive statistics to: {desc_path}")
print("\nDescriptive statistics:")
print(desc)
print()


# =========================
# 9. Company summary
# =========================
company_summary = (
    df.groupby(["Ticker", "Company Name"], as_index=False)
    .agg(
        Avg_Revenue=("Revenue", "mean"),
        Avg_COGS_ratio=("COGS_ratio", "mean"),
        Avg_SGA_ratio=("SGA_ratio", "mean"),
        Avg_Operating_Margin=("Operating_margin", "mean"),
        Avg_Net_Margin=("Net_margin", "mean"),
        Avg_ROA=("ROA", "mean"),
        Years=("Fiscal_Year", "nunique")
    )
    .sort_values("Avg_Net_Margin", ascending=False)
)

company_summary_path = OUTPUT_DIR / "company_summary.csv"
company_summary.to_csv(company_summary_path, index=False)
print(f"Saved company summary to: {company_summary_path}")
print("\nTop companies by average net margin:")
print(company_summary.head(15))
print()


# =========================
# 10. Year summary
# =========================
year_summary = (
    df.groupby("Fiscal_Year", as_index=False)
    .agg(
        Mean_COGS_ratio=("COGS_ratio", "mean"),
        Mean_SGA_ratio=("SGA_ratio", "mean"),
        Mean_Operating_Margin=("Operating_margin", "mean"),
        Mean_Net_Margin=("Net_margin", "mean"),
        Mean_ROA=("ROA", "mean")
    )
    .sort_values("Fiscal_Year")
)

year_summary_path = OUTPUT_DIR / "year_summary.csv"
year_summary.to_csv(year_summary_path, index=False)
print(f"Saved year summary to: {year_summary_path}")
print()


# =========================
# 11. Plots
# =========================
plt.figure(figsize=(9, 5))
plt.plot(year_summary["Fiscal_Year"], year_summary["Mean_COGS_ratio"], marker="o", label="Mean COGS Ratio")
plt.plot(year_summary["Fiscal_Year"], year_summary["Mean_SGA_ratio"], marker="o", label="Mean SGA Ratio")
plt.title("Robustness Check: Average Cost Ratios (2019-2024)")
plt.xlabel("Fiscal Year")
plt.ylabel("Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cost_ratios_over_time.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(year_summary["Fiscal_Year"], year_summary["Mean_Net_Margin"], marker="o", label="Mean Net Margin")
plt.plot(year_summary["Fiscal_Year"], year_summary["Mean_ROA"], marker="o", label="Mean ROA")
plt.title("Robustness Check: Profitability (2019-2024)")
plt.xlabel("Fiscal Year")
plt.ylabel("Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "profitability_over_time.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(df["COGS_ratio"], df["Net_margin"], alpha=0.6)
plt.title("COGS Ratio vs Net Margin")
plt.xlabel("COGS Ratio")
plt.ylabel("Net Margin")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cogs_vs_net_margin.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(df["SGA_ratio"], df["Net_margin"], alpha=0.6)
plt.title("SGA Ratio vs Net Margin")
plt.xlabel("SGA Ratio")
plt.ylabel("Net Margin")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sga_vs_net_margin.png", dpi=300)
plt.close()

top15 = company_summary.head(15).copy()
plt.figure(figsize=(12, 6))
plt.bar(top15["Ticker"], top15["Avg_Net_Margin"])
plt.title("Top 15 Firms by Average Net Margin")
plt.xlabel("Ticker")
plt.ylabel("Average Net Margin")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top15_net_margin.png", dpi=300)
plt.close()

print("Saved all figures to outputs_robustness/.\n")


# =========================
# 12. Regression 1: Net Margin
# =========================
reg_df1 = df.dropna(subset=["Net_margin", "COGS_ratio", "SGA_ratio", "Log_Assets"]).copy()

model_net = smf.ols(
    formula="Net_margin ~ COGS_ratio + SGA_ratio + Log_Assets + C(Fiscal_Year)",
    data=reg_df1
).fit(cov_type="HC3")

net_regression_txt = OUTPUT_DIR / "regression_net_margin.txt"
with open(net_regression_txt, "w", encoding="utf-8") as f:
    f.write(model_net.summary().as_text())

print("Net margin regression finished.")
print(model_net.summary())
print(f"\nSaved net margin regression summary to: {net_regression_txt}")


# =========================
# 13. Regression 2: ROA
# =========================
reg_df2 = df.dropna(subset=["ROA", "COGS_ratio", "SGA_ratio", "Log_Assets"]).copy()

model_roa = smf.ols(
    formula="ROA ~ COGS_ratio + SGA_ratio + Log_Assets + C(Fiscal_Year)",
    data=reg_df2
).fit(cov_type="HC3")

roa_regression_txt = OUTPUT_DIR / "regression_roa.txt"
with open(roa_regression_txt, "w", encoding="utf-8") as f:
    f.write(model_roa.summary().as_text())

print("\nROA regression finished.")
print(model_roa.summary())
print(f"\nSaved ROA regression summary to: {roa_regression_txt}")


# =========================
# 14. Save coefficient tables
# =========================
coef_net = pd.DataFrame({
    "Variable": model_net.params.index,
    "Coefficient": model_net.params.values,
    "P_value": model_net.pvalues.values
})
coef_net_path = OUTPUT_DIR / "regression_coefficients_net_margin.csv"
coef_net.to_csv(coef_net_path, index=False)

coef_roa = pd.DataFrame({
    "Variable": model_roa.params.index,
    "Coefficient": model_roa.params.values,
    "P_value": model_roa.pvalues.values
})
coef_roa_path = OUTPUT_DIR / "regression_coefficients_roa.csv"
coef_roa.to_csv(coef_roa_path, index=False)

print(f"Saved net margin coefficients to: {coef_net_path}")
print(f"Saved ROA coefficients to: {coef_roa_path}")


print("\nDone.")
print("Main outputs:")
print(f"- {cleaned_path}")
print(f"- {desc_path}")
print(f"- {company_summary_path}")
print(f"- {year_summary_path}")
print(f"- {net_regression_txt}")
print(f"- {roa_regression_txt}")
print(f"- {coef_net_path}")
print(f"- {coef_roa_path}")