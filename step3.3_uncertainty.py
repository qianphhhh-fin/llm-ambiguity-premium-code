import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# ==========================================
# 0. 配置路径
# ==========================================
FILE_DISPERSION = "step2_fomc_dispersion_results.csv"
FILE_VIX = r"data\vix\vix_daily.csv"
# 我们选择 Real Uncertainty 作为"硬数据"的代表
FILE_JLN = r"data\ludvigson Macro and Financial Uncertainty Indexes\RealUncertaintyToCirculate.xlsx"
OUT_TEX = r"tab\Tab1.4_step3.3_uncertainty.tex"

# 确保输出目录存在
os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)

def load_and_process_data():
    print("Loading data...")
    
    # 1. Load D_t (Narrative Ambiguity)
    df_d = pd.read_csv(FILE_DISPERSION)
    df_d['date'] = pd.to_datetime(df_d['date']).dt.tz_localize(None).dt.normalize()
    # 计算 D 的新息 (Shock)
    df_d['D_Shock'] = df_d['Semantic_Dispersion'].diff()
    df_d = df_d.sort_values('date')
    
    # 2. Load VIX (Market Fear)
    df_vix = pd.read_csv(FILE_VIX)
    df_vix['Date'] = pd.to_datetime(df_vix['Date'])
    df_vix = df_vix.set_index('Date').sort_index()
    df_vix['VIX_Change'] = df_vix['VIX'].diff()
    
    # 3. Load JLN Real Uncertainty
    print(f"Loading JLN Data from {FILE_JLN}...")
    try:
        df_jln = pd.read_excel(FILE_JLN)
        
        # 清洗日期: 格式为 "7/1960" (Month/Year)
        # 将其设置为该月的最后一天，以便 merge_asof backward 能够正确匹配到最近的一个月数据
        df_jln['Date'] = pd.to_datetime(df_jln['Date'], format='%m/%Y') + pd.offsets.MonthEnd(0)
        
        # 选取 h=1 (1-month ahead uncertainty)，最贴近 FOMC 的决策频率
        # 重命名为 Real_Uncertainty
        df_jln = df_jln[['Date', 'h=1']].rename(columns={'h=1': 'Real_Unc'})
        df_jln = df_jln.set_index('Date').sort_index()
        
        # 计算新息 (Shock): Delta JLN
        # 逻辑：市场对基本面不确定性的"变化"做出反应
        df_jln['Real_Unc_Shock'] = df_jln['Real_Unc'].diff()
        
    except Exception as e:
        print(f"Error loading JLN data: {e}")
        return None

    # 4. Merge Data (Event Study Alignment)
    print("Merging datasets...")
    
    # A. Merge VIX (Daily -> Event)
    df_merged = pd.merge_asof(
        df_d, 
        df_vix, 
        left_on='date', 
        right_index=True, 
        direction='backward', 
        tolerance=pd.Timedelta('5d')
    )
    
    # B. Merge JLN (Monthly -> Event)
    # FOMC会议通常在月中。direction='backward' 会找到上个月月末发布的 JLN 数据
    # 或者本月刚刚发布的。
    # 既然 JLN 是基于月度宏观数据计算的，我们假设会议当时能观测到的最新宏观状态
    df_merged = pd.merge_asof(
        df_merged,
        df_jln,
        left_on='date',
        right_index=True,
        direction='backward',
        tolerance=pd.Timedelta('40d') # 允许回溯一个月
    )
    
    # 清洗：去除空值
    df_final = df_merged.dropna(subset=['D_Shock', 'VIX_Change', 'Real_Unc_Shock']).copy()
    
    # 5. Z-Score 标准化 (方便比较回归系数大小)
    for col in ['D_Shock', 'Real_Unc_Shock']:
        df_final[f'{col}_Z'] = (df_final[col] - df_final[col].mean()) / df_final[col].std()
        
    print(f"Final Sample Size: {len(df_final)} events")
    return df_final

def run_regressions(df):
    results = []
    
    # Model 1: VIX ~ Narrative Shock Only
    X1 = sm.add_constant(df['D_Shock_Z'])
    y = df['VIX_Change']
    m1 = sm.OLS(y, X1).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    results.append(m1)
    
    # Model 2: VIX ~ Real Uncertainty Shock Only
    # 检验：基本面变得不可预测时，VIX 是否上升？(理论上应该显著正相关)
    X2 = sm.add_constant(df['Real_Unc_Shock_Z'])
    m2 = sm.OLS(y, X2).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    results.append(m2)
    
    # Model 3: Horse Race (VIX ~ Narrative + Real Uncertainty)
    X3 = sm.add_constant(df[['D_Shock_Z', 'Real_Unc_Shock_Z']])
    m3 = sm.OLS(y, X3).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    results.append(m3)
    
    # Check Correlation
    corr = df[['D_Shock_Z', 'Real_Unc_Shock_Z']].corr().iloc[0, 1]
    
    return results, corr, len(df)

def format_coef(model, var_name):
    if var_name not in model.params:
        return "", ""
    coef = model.params[var_name]
    t_stat = model.tvalues[var_name]
    
    stars = ""
    if abs(t_stat) > 2.58: stars = "***"
    elif abs(t_stat) > 1.96: stars = "**"
    elif abs(t_stat) > 1.65: stars = "*"
    
    return f"{coef:.3f}{stars}", f"({t_stat:.2f})"

def generate_latex(results, corr_val, n_obs):
    print(f"Generating LaTeX table: {OUT_TEX}")
    
    m1, m2, m3 = results
    
    # Format Coefficients
    d_c1, d_t1 = format_coef(m1, 'D_Shock_Z')
    d_c3, d_t3 = format_coef(m3, 'D_Shock_Z')
    
    jln_c2, jln_t2 = format_coef(m2, 'Real_Unc_Shock_Z')
    jln_c3, jln_t3 = format_coef(m3, 'Real_Unc_Shock_Z')
    
    latex_str = r"""
\begin{table}[h!]
\centering
\caption{Narrative Ambiguity vs. Real Economic Uncertainty (JLN)}
\label{tab:jln_horse_race}
\begin{threeparttable}
\begin{tabular}{lccc}
\toprule
 & \multicolumn{3}{c}{Dependent Variable: $\Delta \text{VIX}_{t}$} \\
\cmidrule(lr){2-4}
Variable & (1) & (2) & (3) \\
\midrule
\multicolumn{4}{l}{\textbf{Panel A: Regression Analysis}} \\
    \hspace{1em}Narrative Shock ($\Delta D_t$) & """ + f"{d_c1} & & {d_c3} \\\\" + r"""
         & """ + f"{d_t1} & & {d_t3} \\\\" + r"""
    \hspace{1em}Real Uncertainty Shock ($\Delta \text{JLN}_t$) & & """ + f"{jln_c2} & {jln_c3} \\\\" + r"""
         & & """ + f"{jln_t2} & {jln_t3} \\\\" + r"""
\midrule
Constant & Yes & Yes & Yes \\
Observations & """ + f"{n_obs} & {n_obs} & {n_obs} \\\\" + r"""
Adj. $R^2$ & """ + f"{m1.rsquared_adj:.3f} & {m2.rsquared_adj:.3f} & {m3.rsquared_adj:.3f} \\\\" + r"""
\midrule
\multicolumn{4}{l}{\textbf{Panel B: Orthogonality Check}} \\
    \hspace{1em}Corr($\Delta D_t$, $\Delta \text{JLN}_t$) & \multicolumn{3}{c}{""" + f"{corr_val:.3f}" + r"""} \\
\bottomrule
\end{tabular}
\begin{tablenotes}[para,flushleft]
  \item Note: This table investigates whether narrative ambiguity is distinct from fundamental economic uncertainty. Real Economic Uncertainty (JLN) is the 1-month ahead uncertainty index in real economic activity constructed by Jurado, Ludvigson, and Ng (2015). Both independent variables are standardized. Newey-West standard errors (1 lag) are used.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    
    with open(OUT_TEX, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    print("LaTeX table generated successfully.")

def main():
    df = load_and_process_data()
    if df is not None:
        results, corr, n_obs = run_regressions(df)
        generate_latex(results, corr, n_obs)
    else:
        print("Failed to run analysis due to data loading errors.")

if __name__ == "__main__":
    main()