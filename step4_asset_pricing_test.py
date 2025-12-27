import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg
import os

# ==========================================
# 0. 配置路径
# ==========================================
FILE_DISPERSION = "step2_fomc_dispersion_results.csv"
FILE_FF3 = r"data\ff3\F-F_Research_Data_Factors.csv"
FILE_IND49 = r"data\ff3\49_Industry_Portfolios_Value_Weighted_Monthly.csv"

OUT_TAB = r"tab\Tab1.6_step4_asset_pricing_results.tex"
OUT_FIG_BETA = r"fig\fig1.3_step4_beta_exposure.png"
OUT_FIG_FIT = r"fig\fig1.4_step4_model_fit.png"

# 确保目录存在
os.makedirs(os.path.dirname(OUT_TAB), exist_ok=True)
os.makedirs(os.path.dirname(OUT_FIG_BETA), exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据加载与因子构建 (复用之前经过验证的逻辑)
# ==========================================
def load_data():
    # 1. Ind 49
    try:
        df_ind = pd.read_csv(FILE_IND49)
        df_ind.columns = [c.strip() for c in df_ind.columns]
        df_ind = df_ind.dropna(subset=['Date'])
        df_ind.index = pd.to_datetime(df_ind['Date'].astype(int).astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
        df_ind = df_ind.drop(columns=['Date']).replace([-99.99, -999], np.nan).dropna().astype(float)/100.0
    except: return None, None, None

    # 2. FF3
    try:
        df_ff = pd.read_csv(FILE_FF3).dropna()
        df_ff.columns = [c.strip() for c in df_ff.columns]
        df_ff = df_ff.dropna(subset=['Date'])
        df_ff.index = pd.to_datetime(df_ff['Date'].astype(int).astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
        df_ff = df_ff.drop(columns=['Date']).astype(float)/100.0
    except: return None, None, None
    
    # 3. Dispersion
    df_disp = pd.read_csv(FILE_DISPERSION)
    df_disp['date'] = pd.to_datetime(df_disp['date']).dt.tz_localize(None)
    
    return df_ind, df_ff, df_disp

def construct_factor(df_disp, df_ff):
    # Event-Only Difference Logic
    df_disp['month'] = df_disp['date'] + pd.offsets.MonthEnd(0)
    df_event = df_disp.groupby('month')['Semantic_Dispersion'].mean()
    meeting_months = df_event.index
    
    full_idx = pd.date_range(start=df_event.index.min(), end=df_event.index.max(), freq='M')
    factor = df_event.reindex(full_idx).ffill().diff().dropna()
    
    # Filter Event Months
    factor = factor.loc[factor.index.intersection(meeting_months)]
    
    # Orthogonalize against FF3 (Strict Control)
    common = factor.index.intersection(df_ff.index)
    y = factor.loc[common]
    X = sm.add_constant(df_ff.loc[common, ['Mkt-RF', 'SMB', 'HML']])
    pure = sm.OLS(y, X).fit().resid
    
    # Standardize
    return (pure - pure.mean()) / pure.std()

# ==========================================
# 2. Fama-MacBeth 核心引擎
# ==========================================
def run_fmb_regression(df_ex_ret, factors_df, factor_names):
    """
    运行 Fama-MacBeth 两步回归
    factor_names: 参与回归的因子列表 ['Mkt-RF', 'Narrative', ...]
    """
    # Step 1: Time Series (Get Betas)
    X_ts = sm.add_constant(factors_df[factor_names])
    betas = []
    
    for port in df_ex_ret.columns:
        model = sm.OLS(df_ex_ret[port], X_ts).fit()
        # 提取除了const以外的系数
        b = model.params[factor_names]
        b.name = port
        betas.append(b)
        
    df_betas = pd.DataFrame(betas) # Index=Port, Cols=Factor Betas
    
    # Step 2: Cross Section (Get Lambdas)
    y = df_ex_ret.mean()
    X_cs = sm.add_constant(df_betas) # Add lambda_0
    
    # 使用 GLS 或 OLS (这里用 OLS 简单明了，符合通常展示)
    model_cs = sm.OLS(y, X_cs).fit()
    
    return model_cs, df_betas

# ==========================================
# 3. 制表与绘图逻辑
# ==========================================
def format_tex_coef(val, t_stat):
    stars = ""
    if abs(t_stat) > 2.58: stars = "***"
    elif abs(t_stat) > 1.96: stars = "**"
    elif abs(t_stat) > 1.65: stars = "*"
    return f"{val*100:.2f}{stars}", f"({t_stat:.2f})" # Return as percentage

def generate_table_1_6(res_capm, res_ff3, nobs):
    print(f"Generating LaTeX Table: {OUT_TAB}")
    
    # Helper to get row string
    def get_row(var_name, label):
        row_str = f"    {label} & "
        # Model 1
        if var_name in res_capm.params:
            c, t = format_tex_coef(res_capm.params[var_name], res_capm.tvalues[var_name])
            row_str += f"{c} & "
        else: row_str += " & "
        # Model 2
        if var_name in res_ff3.params:
            c, t = format_tex_coef(res_ff3.params[var_name], res_ff3.tvalues[var_name])
            row_str += f"{c} \\\\\n"
        else: row_str += " \\\\\n"
        
        # T-stats row
        row_str += "         & "
        if var_name in res_capm.params:
            c, t = format_tex_coef(res_capm.params[var_name], res_capm.tvalues[var_name])
            row_str += f"{t} & "
        else: row_str += " & "
        if var_name in res_ff3.params:
            c, t = format_tex_coef(res_ff3.params[var_name], res_ff3.tvalues[var_name])
            row_str += f"{t} \\\\\n"
        else: row_str += " \\\\\n"
        return row_str

    latex = r"""
\begin{table}[h!]
\centering
\caption{Cross-Sectional Pricing of Narrative Ambiguity (49 Industries)}
\label{tab:pricing_results}
\begin{threeparttable}
\begin{tabular}{lcc}
\toprule
 & \multicolumn{2}{c}{Dependent Variable: Average Excess Returns} \\
\cmidrule(lr){2-3}
Factor Risk Premia ($\lambda$) & (1) CAPM + Narrative & (2) FF3 + Narrative \\
\midrule
"""
    latex += get_row('Narrative', 'Narrative Ambiguity ($\lambda_{D}$)')
    latex += r"\addlinespace" + "\n"
    latex += get_row('Mkt-RF', 'Market ($\lambda_{MKT}$)')
    latex += get_row('SMB', 'Size ($\lambda_{SMB}$)')
    latex += get_row('HML', 'Value ($\lambda_{HML}$)')
    latex += get_row('const', 'Constant ($\lambda_{0}$)')
    
    latex += r"""
\midrule
Test Assets & 49 Industries & 49 Industries \\
Event Months & """ + str(nobs) + r""" & """ + str(nobs) + r""" \\
Adj. $R^2$ & """ + f"{res_capm.rsquared_adj:.2f}" + r""" & """ + f"{res_ff3.rsquared_adj:.2f}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}[para,flushleft]
  \item Note: This table reports Fama-MacBeth cross-sectional regression results using 49 Industry Portfolios. The sample is restricted to months with FOMC meetings (Event-Only). The Narrative Ambiguity factor is orthogonalized against FF3 factors. Coefficients are reported in percentage. t-statistics are in parentheses. *, **, *** indicate significance at 10\%, 5\%, and 1\% levels.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    with open(OUT_TAB, 'w', encoding='utf-8') as f:
        f.write(latex)

def plot_fig_1_3_beta(df_betas):
    """绘制 Beta 条形图 (Top 5 vs Bottom 5)"""
    print(f"Plotting Beta Exposure: {OUT_FIG_BETA}")
    
    # 排序
    sorted_betas = df_betas['Narrative'].sort_values()
    
    # 取两头
    top_5 = sorted_betas.tail(5)
    bot_5 = sorted_betas.head(5)
    plot_data = pd.concat([bot_5, top_5])
    
    # 绘图
    plt.figure(figsize=(10, 6))
    colors = ['#8B0000']*5 + ['#006400']*5 # Dark Red for Negative, Dark Green for Positive
    
    bars = plt.barh(plot_data.index, plot_data.values, color=colors, alpha=0.8)
    
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Industry Exposure to Narrative Ambiguity (Beta)', fontsize=14, fontweight='bold')
    
    # [FIX] 使用 r'' (raw string) 防止 \beta 被转义为退格符
    plt.xlabel(r'Narrative Beta ($\beta_{i,D}$)', fontsize=12)
    
    # 添加标注
    # 根据你的数据范围，这里的坐标可能需要微调
    # 既然是横向条形图，y轴是 0~9 的索引位置
    plt.text(plot_data.values.max()*0.1, 7, 'Hedge Assets\n(Gold, Utilities)', 
             color='#006400', fontweight='bold', ha='left')
    plt.text(plot_data.values.min()*0.1, 2, 'Exposed Assets\n(Durables, Tech)', 
             color='#8B0000', fontweight='bold', ha='right')
    
    plt.tight_layout()
    plt.savefig(OUT_FIG_BETA, dpi=300)

def plot_fig_1_4_fit(res_ff3, y_real):
    """绘制模型拟合图"""
    print(f"Plotting Model Fit: {OUT_FIG_FIT}")
    
    pred_ret = res_ff3.predict() # Cross-sectional fitted values
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pred_ret, y=y_real, s=100, color='#003366', alpha=0.7, edgecolor='w')
    
    # 45 degree line
    line_min = min(pred_ret.min(), y_real.min())
    line_max = max(pred_ret.max(), y_real.max())
    plt.plot([line_min, line_max], [line_min, line_max], 'r--', linewidth=2, label='Perfect Pricing')
    
    # 标注几个关键点
    # 计算残差绝对值
    resid = abs(pred_ret - y_real)
    # 标注残差最大的5个 + Gold + Chips + Autos
    important = ['Gold', 'Chips', 'Autos', 'Softw', 'Fin']
    top_resid = resid.nlargest(5).index.tolist()
    labels = list(set(important + top_resid))
    
    for label in labels:
        if label in y_real.index:
            x_pos = pred_ret[y_real.index.get_loc(label)]
            y_pos = y_real[label]
            plt.text(x_pos + 0.0002, y_pos, label, fontsize=9, fontweight='bold')

    plt.xlabel('Predicted Excess Returns (FF3 + Narrative)', fontsize=12)
    plt.ylabel('Realized Excess Returns', fontsize=12)
    plt.title(f'Model Fit: 49 Industry Portfolios (Adj $R^2$ = {res_ff3.rsquared_adj:.2f})', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG_FIT, dpi=300)

def main():
    # 1. 准备数据
    df_ind, df_ff, df_disp = load_data()
    if df_ind is None: 
        print("Data load failed.")
        return

    s_innov = construct_factor(df_disp, df_ff)
    
    # 对齐
    common = df_ind.index.intersection(df_ff.index).intersection(s_innov.index)
    common = common[common.year >= 2000]
    
    print(f"Sample: {len(common)} Event Months")
    
    df_ind = df_ind.loc[common]
    df_ff = df_ff.loc[common]
    s_innov = s_innov.loc[common]
    df_ex = df_ind.sub(df_ff['RF'], axis=0)
    
    # 组合因子DataFrame
    factors_all = df_ff.copy()
    factors_all['Narrative'] = s_innov
    
    # 2. 运行回归
    # Model 1: CAPM + D
    res_1, betas_1 = run_fmb_regression(df_ex, factors_all, ['Mkt-RF', 'Narrative'])
    # Model 2: FF3 + D
    res_2, betas_2 = run_fmb_regression(df_ex, factors_all, ['Mkt-RF', 'SMB', 'HML', 'Narrative'])
    
    print("\n--- Model 2 (FF3+D) Results ---")
    print(res_2.summary())
    
    # 3. 生成产出
    generate_table_1_6(res_1, res_2, len(common))
    plot_fig_1_3_beta(betas_2) # 使用 FF3 模型下的 Beta
    plot_fig_1_4_fit(res_2, df_ex.mean())
    
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()