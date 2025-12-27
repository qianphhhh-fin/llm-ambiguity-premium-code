import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import os

# ==========================================
# 0. 配置与工具函数
# ==========================================
FILE_DISPERSION = "step2_fomc_dispersion_results.csv"
FILE_VIX = r"data\vix\vix_daily.csv"
FILE_EPU = r"data\epu\US_EPU.csv"

# 输出文件
OUT_TEX_1 = r"tab\Tab1.1_step3_vix_and_dt.tex"
OUT_TEX_2 = r"tab\Tab1.2_step3_epu_and_dt.tex"

def load_data():
    """加载并清洗数据，计算差分(Shocks)"""
    print("Loading data...")
    
    # 1. D_t (Event)
    df_d = pd.read_csv(FILE_DISPERSION)
    df_d['date'] = pd.to_datetime(df_d['date']).dt.tz_localize(None).dt.normalize()
    df_d = df_d.sort_values('date')
    # 计算 D_t 的新息 (Shock)
    df_d['D_Level'] = df_d['Semantic_Dispersion']
    df_d['D_Shock'] = df_d['Semantic_Dispersion'].diff()
    
    # 2. VIX (Daily)
    df_vix = pd.read_csv(FILE_VIX)
    df_vix['Date'] = pd.to_datetime(df_vix['Date'])
    df_vix = df_vix.set_index('Date').sort_index()
    # 计算 VIX 的变化
    df_vix['VIX_Level'] = df_vix['VIX']
    df_vix['VIX_Change'] = df_vix['VIX'].diff()
    # 未来5天的变化 (用于预测回归)
    df_vix['VIX_Change_Fut'] = df_vix['VIX'].shift(-5) - df_vix['VIX']
    
    # 3. EPU (Monthly)
    df_epu = pd.read_csv(FILE_EPU)
    df_epu['date_key'] = pd.to_datetime(df_epu[['Year', 'Month']].assign(DAY=1))
    df_epu = df_epu.set_index('date_key').sort_index()
    df_epu = df_epu[['News_Based_Policy_Uncert_Index']].rename(columns={'News_Based_Policy_Uncert_Index': 'EPU_Level'})
    df_epu['EPU_Shock'] = df_epu['EPU_Level'].diff()
    
    # 4. Merge (Event-based)
    print("Merging...")
    df_merged = pd.merge_asof(df_d, df_vix, left_on='date', right_index=True, direction='backward', tolerance=pd.Timedelta('5d'))
    df_merged = pd.merge_asof(df_merged, df_epu, left_on='date', right_index=True, direction='backward', tolerance=pd.Timedelta('32d'))
    
    # 标准化自变量 (Standardize Regressors) 以便比较系数
    df = df_merged.dropna().copy()
    for col in ['D_Shock', 'EPU_Shock', 'D_Level', 'EPU_Level']:
        df[f'{col}_Z'] = (df[col] - df[col].mean()) / df[col].std()
        
    return df

def run_ols_nw(y, X):
    """运行 OLS 并返回 Newey-West (HAC) 结果"""
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    return model

def format_coef(val, t_stat):
    """格式化系数：0.123*** (2.54)"""
    stars = ""
    if abs(t_stat) > 2.58: stars = "***"
    elif abs(t_stat) > 1.96: stars = "**"
    elif abs(t_stat) > 1.65: stars = "*"
    
    return f"{val:.3f}{stars}", f"({t_stat:.2f})"

# ==========================================
# 1. 生成 Table 1.1: VIX 与 D_t 关系
# ==========================================
def generate_table_vix(df):
    print("Generating Table 1.1 (VIX)...")
    
    # --- Panel A: Regressions ---
    # Model 1: Contemporaneous (Delta VIX ~ Delta D)
    m1 = run_ols_nw(df['VIX_Change'], df['D_Shock_Z'])
    
    # Model 2: Contemporaneous with Control (Delta VIX ~ Delta D + Lagged VIX Level)
    # 控制 VIX 水平是因为波动率有均值回归特性
    m2 = run_ols_nw(df['VIX_Change'], df[['D_Shock_Z', 'VIX_Level']])
    
    # Model 3: Predictive (Future Delta VIX ~ Delta D)
    m3 = run_ols_nw(df['VIX_Change_Fut'], df['D_Shock_Z'])
    
    # --- Panel B: Granger Causality ---
    # D -> VIX
    gc_res_1 = grangercausalitytests(df[['VIX_Change', 'D_Shock']].values, maxlag=[1], verbose=False)
    f_stat_1 = gc_res_1[1][0]['ssr_ftest'][0]
    p_val_1 = gc_res_1[1][0]['ssr_ftest'][1]
    
    # VIX -> D
    gc_res_2 = grangercausalitytests(df[['D_Shock', 'VIX_Change']].values, maxlag=[1], verbose=False)
    f_stat_2 = gc_res_2[1][0]['ssr_ftest'][0]
    p_val_2 = gc_res_2[1][0]['ssr_ftest'][1]

    # --- Latex Construction ---
    latex = r"""
\begin{table}[h!]
\centering
\caption{Narrative Ambiguity and Market Fear (VIX)}
\label{tab:vix_mechanism}
\begin{threeparttable}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{Contemporaneous} & \multicolumn{1}{c}{Predictive} \\
 & \multicolumn{2}{c}{$\Delta \text{VIX}_{t}$} & \multicolumn{1}{c}{$\Delta \text{VIX}_{t \to t+5}$} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-4}
Variable & (1) & (2) & (3) \\
\midrule
\multicolumn{4}{l}{\textit{\textbf{Panel A: OLS Regressions with Newey-West Adjustments}}} \\
"""
    # Rows
    rows = [
        ("Narrative Shock ($\Delta D_t$)", 'D_Shock_Z'),
        ("Lagged VIX Level ($VIX_{t-1}$)", 'VIX_Level'),
        ("Constant", 'const')
    ]
    
    for label, var in rows:
        line1 = f"    \\hspace{{1em}}{label} & "
        line2 = "         & "
        
        for m in [m1, m2, m3]:
            if var in m.params:
                coef, t = format_coef(m.params[var], m.tvalues[var])
                line1 += f"{coef} & "
                line2 += f"{t} & "
            else:
                line1 += " & "
                line2 += " & "
        
        latex += line1[:-2] + "\\\\\n" + line2[:-2] + "\\\\\n"

    latex += r"""
\midrule
Observations & """ + f"{int(m1.nobs)} & {int(m2.nobs)} & {int(m3.nobs)} \\\\" + r"""
Adj. $R^2$ & """ + f"{m1.rsquared_adj:.3f} & {m2.rsquared_adj:.3f} & {m3.rsquared_adj:.3f} \\\\" + r"""
\midrule
\multicolumn{4}{l}{\textit{\textbf{Panel B: Granger Causality Tests (F-statistic)}}} \\
    \hspace{1em}Null: $D_t$ does not cause VIX & \multicolumn{3}{c}{""" + f"{f_stat_1:.2f} (p={p_val_1:.3f})" + r"""} \\
    \hspace{1em}Null: VIX does not cause $D_t$ & \multicolumn{3}{c}{""" + f"{f_stat_2:.2f} (p={p_val_2:.3f})" + r"""} \\
\bottomrule
\end{tabular}
\begin{tablenotes}[para,flushleft]
  \item Note: This table reports the relationship between FOMC narrative ambiguity shocks ($\Delta D_t$) and changes in the VIX index. $\Delta D_t$ is standardized. t-statistics (in parentheses) are computed using Newey-West HAC standard errors with 1 lag. ***, **, and * denote significance at the 1\%, 5\%, and 10\% levels, respectively.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    with open(OUT_TEX_1, "w", encoding='utf-8') as f:
        f.write(latex)
    print(f"Saved {OUT_TEX_1}")

# ==========================================
# 2. 生成 Table 1.2: D_t 与 EPU 赛马
# ==========================================
def generate_table_epu(df):
    print("Generating Table 1.2 (EPU Horse Race)...")
    
    # Model 1: VIX ~ D (Baseline)
    m1 = run_ols_nw(df['VIX_Change'], df['D_Shock_Z'])
    
    # Model 2: VIX ~ EPU (Competitor)
    m2 = run_ols_nw(df['VIX_Change'], df['EPU_Shock_Z'])
    
    # Model 3: VIX ~ D + EPU (Horse Race)
    m3 = run_ols_nw(df['VIX_Change'], df[['D_Shock_Z', 'EPU_Shock_Z']])
    
    # Correlation
    corr_val = df[['D_Shock_Z', 'EPU_Shock_Z']].corr().iloc[0,1]

    latex = r"""
\begin{table}[h!]
\centering
\caption{The Horse Race: Narrative Ambiguity vs. Economic Policy Uncertainty}
\label{tab:epu_horse_race}
\begin{threeparttable}
\begin{tabular}{lccc}
\toprule
 & \multicolumn{3}{c}{Dependent Variable: $\Delta \text{VIX}_{t}$} \\
\cmidrule(lr){2-4}
Variable & (1) & (2) & (3) \\
\midrule
\multicolumn{4}{l}{\textbf{Panel A: Independent Shocks}} \\
    \hspace{1em}Narrative Shock ($\Delta D_t$) & """ 
    
    # Row for D
    coef, t = format_coef(m1.params['D_Shock_Z'], m1.tvalues['D_Shock_Z'])
    latex += f"{coef} & & "
    coef, t = format_coef(m3.params['D_Shock_Z'], m3.tvalues['D_Shock_Z'])
    latex += f"{coef} \\\\\n"
    latex += f"         & ({t}) & & ({t}) \\\\\n" # Using t from m1 and m3
    
    # Row for EPU
    latex += r"    \hspace{1em}EPU Shock ($\Delta \text{EPU}_t$) & & "
    coef, t = format_coef(m2.params['EPU_Shock_Z'], m2.tvalues['EPU_Shock_Z'])
    latex += f"{coef} & "
    coef, t = format_coef(m3.params['EPU_Shock_Z'], m3.tvalues['EPU_Shock_Z'])
    latex += f"{coef} \\\\\n"
    # T-stats
    t2 = m2.tvalues['EPU_Shock_Z']
    t3 = m3.tvalues['EPU_Shock_Z']
    latex += f"         & & ({t2:.2f}) & ({t3:.2f}) \\\\\n"

    latex += r"""
\midrule
Constant & Yes & Yes & Yes \\
Observations & """ + f"{int(m1.nobs)} & {int(m2.nobs)} & {int(m3.nobs)} \\\\" + r"""
Adj. $R^2$ & """ + f"{m1.rsquared_adj:.3f} & {m2.rsquared_adj:.3f} & {m3.rsquared_adj:.3f} \\\\" + r"""
\midrule
\multicolumn{4}{l}{\textbf{Panel B: Correlation Analysis}} \\
    \hspace{1em}Corr($\Delta D_t$, $\Delta \text{EPU}_t$) & \multicolumn{3}{c}{""" + f"{corr_val:.3f}" + r"""} \\
\bottomrule
\end{tabular}
\begin{tablenotes}[para,flushleft]
  \item Note: This table compares the explanatory power of Narrative Ambiguity shocks ($\Delta D_t$) and Economic Policy Uncertainty shocks ($\Delta \text{EPU}_t$) on VIX changes. Both independent variables are standardized to facilitate comparison. EPU data is from Baker, Bloom, and Davis (2016). Newey-West standard errors are used.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    with open(OUT_TEX_2, "w", encoding='utf-8') as f:
        f.write(latex)
    print(f"Saved {OUT_TEX_2}")

def main():
    df = load_data()
    generate_table_vix(df)
    generate_table_epu(df)
    print("\nDone. Check the .tex files.")

if __name__ == "__main__":
    main()