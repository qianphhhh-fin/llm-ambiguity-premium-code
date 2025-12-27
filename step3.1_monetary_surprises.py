import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# ==========================================
# 0. 配置路径
# ==========================================
FILE_DISPERSION = "step2_fomc_dispersion_results.csv"
FILE_VIX = r"data\vix\vix_daily.csv"
FILE_MPS = r"data\monetary-policy-surprises\mps.csv"
OUT_TEX = r"tab\Tab1.3_step3.1_monetary_surprises.tex"

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
    
    # 3. Load MP Surprises (Acosta et al., 2025)
    # 注意：我们需要检查 CSV 的具体列名。
    # 假设该文件包含日期列和冲击列。
    try:
        df_mps = pd.read_csv(FILE_MPS)
        
        # 尝试智能识别日期列
        date_cols = [c for c in df_mps.columns if 'date' in c.lower() or 'time' in c.lower()]
        if not date_cols:
            raise ValueError("Cannot find Date column in mps.csv")
        date_col = date_cols[0]
        
        df_mps['date'] = pd.to_datetime(df_mps[date_col]).dt.tz_localize(None).dt.normalize()
        
        # 尝试识别 Statement 冲击列
        # 通常名为 'mps_stmt', 'stmt', 'factor1' 等
        # 我们优先找带 'stmt' 的列，如果没有，找第一列数值列
        stmt_cols = [c for c in df_mps.columns if 'stmt' in c.lower()]
        
        if stmt_cols:
            target_col = stmt_cols[0] # 使用第一个找到的 Statement 冲击
            print(f"Using Monetary Policy Shock column: {target_col}")
        else:
            # 兜底：找除了日期外的第一列数值
            numeric_cols = df_mps.select_dtypes(include=[np.number]).columns.tolist()
            target_col = numeric_cols[0]
            print(f"Warning: Specific 'STMT' column not found. Using generic column: {target_col}")
            
        df_mps = df_mps.set_index('date')[[target_col]].rename(columns={target_col: 'MP_Shock'})
        
    except Exception as e:
        print(f"Error loading MPS data: {e}")
        return None

    # 4. Merge Data (Event Study Alignment)
    print("Merging datasets...")
    # 以 D_t 的日期为基准 (Left Join)
    df_merged = pd.merge_asof(
        df_d, 
        df_vix, 
        left_on='date', 
        right_index=True, 
        direction='backward', 
        tolerance=pd.Timedelta('3d')
    )
    
    # 合并 MP Shock (Exact Match usually, but let's use merge to be safe)
    df_merged = pd.merge(df_merged, df_mps, on='date', how='left')
    
    # 清洗：去除空值
    df_final = df_merged.dropna(subset=['D_Shock', 'VIX_Change', 'MP_Shock']).copy()
    
    # 5. 变量构建
    # 我们不仅关心冲击的方向，更关心冲击的【幅度】(Magnitude)
    # 因为 VIX 是波动率，正向和负向的巨大政策意外都会推高 VIX
    df_final['Abs_MP_Shock'] = df_final['MP_Shock'].abs()
    
    # Z-Score 标准化 (方便比较回归系数)
    for col in ['D_Shock', 'MP_Shock', 'Abs_MP_Shock']:
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
    
    # Model 2: VIX ~ MP Shock Magnitude Only
    # 检验：是否政策意外越大，VIX 越高？
    X2 = sm.add_constant(df['Abs_MP_Shock_Z'])
    m2 = sm.OLS(y, X2).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    results.append(m2)
    
    # Model 3: Horse Race (VIX ~ Narrative + MP Shock Magnitude)
    X3 = sm.add_constant(df[['D_Shock_Z', 'Abs_MP_Shock_Z']])
    m3 = sm.OLS(y, X3).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    results.append(m3)
    
    # Check Correlation
    corr = df[['D_Shock_Z', 'Abs_MP_Shock_Z']].corr().iloc[0, 1]
    
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
    
    mp_c2, mp_t2 = format_coef(m2, 'Abs_MP_Shock_Z')
    mp_c3, mp_t3 = format_coef(m3, 'Abs_MP_Shock_Z')
    
    latex_str = r"""
\begin{table}[h!]
\centering
\caption{Narrative Ambiguity vs. Monetary Policy Surprises}
\label{tab:mp_shock_horse_race}
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
    \hspace{1em}MP Shock Magnitude ($|MPS_t|$) & & """ + f"{mp_c2} & {mp_c3} \\\\" + r"""
         & & """ + f"{mp_t2} & {mp_t3} \\\\" + r"""
\midrule
Constant & Yes & Yes & Yes \\
Observations & """ + f"{n_obs} & {n_obs} & {n_obs} \\\\" + r"""
Adj. $R^2$ & """ + f"{m1.rsquared_adj:.3f} & {m2.rsquared_adj:.3f} & {m3.rsquared_adj:.3f} \\\\" + r"""
\midrule
\multicolumn{4}{l}{\textbf{Panel B: Orthogonality Check}} \\
    \hspace{1em}Corr($\Delta D_t$, $|MPS_t|$) & \multicolumn{3}{c}{""" + f"{corr_val:.3f}" + r"""} \\
\bottomrule
\end{tabular}
\begin{tablenotes}[para,flushleft]
  \item Note: This table examines whether narrative ambiguity is simply a proxy for the magnitude of monetary policy surprises. MP Surprises are from Acosta et al. (2025), using the high-frequency shock around the FOMC Statement release. We use the absolute value of the shock ($|MPS_t|$) to capture the magnitude of the surprise regardless of direction. Newey-West standard errors (1 lag) are used.
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