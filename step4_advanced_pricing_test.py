import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import math
from getfactormodels import (
    FamaFrenchFactors, CarhartFactors, QFactors, 
    DHSFactors, ICRFactors, LiquidityFactors, 
    BarillasShankenFactors, HMLDevilFactors
)
import os
import warnings
import pickle

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 0. 配置与路径
# ==========================================
FILE_DISPERSION = "step2_fomc_dispersion_results.csv"
FILE_IND49 = r"data\ff3\49_Industry_Portfolios_Value_Weighted_Monthly.csv"

# 缓存与输出
DIR_CACHE = r"data\factor_cache"
FILE_CACHE_PKL = os.path.join(DIR_CACHE, "all_factors_cache_v2.pkl")

# 表格输出
OUT_TAB = r"tab\Tab1.6_step4_asset_pricing_results.tex"
OUT_EXCEL = r"tab\Asset_Pricing_Full_Results.xlsx"

# 图片输出 (主图)
OUT_FIG_FIT_MAIN = r"fig\fig1.4_step4_model_fit.png"
OUT_FIG_BETA_MAIN = r"fig\fig1.3_step4_beta_exposure.png"

# 图片输出 (附录图)
OUT_FIG_FIT_APP = r"fig\appendix_fig1.4_step4_model_fit.png"
OUT_FIG_BETA_APP = r"fig\appendix_fig1.3_step4_beta_exposure.png"

# 确保目录存在
for p in [OUT_TAB, OUT_FIG_FIT_MAIN]:
    os.makedirs(os.path.dirname(p), exist_ok=True)
os.makedirs(DIR_CACHE, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据获取引擎
# ==========================================
def process_downloaded_data(df):
    """统一处理下载数据的索引和单位"""
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp(freq='M') + pd.offsets.MonthEnd(0)
    else:
        df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
    
    # 自动识别百分比单位并转换
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty and df[numeric_cols].abs().mean().mean() > 0.2:
        df = df / 100.0
        
    return df.dropna()

def fetch_all_factors():
    """获取除SY4外的所有模型"""
    if os.path.exists(FILE_CACHE_PKL):
        print(f"Loading Factor Models from Cache: {FILE_CACHE_PKL}")
        try:
            with open(FILE_CACHE_PKL, 'rb') as f:
                return pickle.load(f)
        except: pass

    print("Fetching ALL Factor Models via API...")
    models_data = {}
    
    # 定义获取任务 (Name, Class, Kwargs)
    # 排除 SY4 (StambaughYuan) 因为数据只到2016
    tasks = [
        ('CAPM', FamaFrenchFactors, {'model': 3}), # 后续只取Mkt
        ('FF3', FamaFrenchFactors, {'model': 3}),
        ('Carhart4', CarhartFactors, {'frequency': 'm'}),
        ('FF5', FamaFrenchFactors, {'model': 5}),
        ('FF6', FamaFrenchFactors, {'model': 6}),
        ('HML_Devil', HMLDevilFactors, {'frequency': 'm'}),
        ('DHS', DHSFactors, {'frequency': 'm'}),
        ('ICR', ICRFactors, {'frequency': 'm'}),
        ('Liq', LiquidityFactors, {'frequency': 'm'}),
        ('HXZ5', QFactors, {'frequency': 'm'}),
        ('Barillas', BarillasShankenFactors, {'frequency': 'm'})
    ]
    
    for label, cls, kwargs in tasks:
        try:
            print(f"  Fetching {label}...")
            df_raw = cls(**kwargs).download()
            df_clean = process_downloaded_data(df_raw)
            
            # 特殊处理 CAPM
            if label == 'CAPM':
                df_clean = df_clean[['Mkt-RF']]
            
            models_data[label] = df_clean
        except Exception as e:
            print(f"  [Skip] Failed to fetch {label}: {e}")
            
    # 保存缓存
    with open(FILE_CACHE_PKL, 'wb') as f:
        pickle.dump(models_data, f)
        
    return models_data

def get_local_industry_49():
    """读取本地 49 Industry Portfolios"""
    print("Loading Local 49 Industry Portfolios...")
    try:
        df = pd.read_csv(FILE_IND49)
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(subset=['Date'])
        df.index = pd.to_datetime(df['Date'].astype(int).astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
        df = df.drop(columns=['Date'])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.replace([-99.99, -999], np.nan).dropna()
        return df.astype(float) / 100.0
    except Exception as e:
        print(f"Error loading Industry 49: {e}")
        return None

def construct_narrative_factor(df_disp, start_date='2000-01-01'):
    """构建叙事因子"""
    print("Constructing Narrative Factor...")
    df_disp['date'] = pd.to_datetime(df_disp['date']).dt.tz_localize(None)
    df_disp['month'] = df_disp['date'] + pd.offsets.MonthEnd(0)
    df_event = df_disp.groupby('month')['Semantic_Dispersion'].mean()
    meeting_months = df_event.index
    
    full_idx = pd.date_range(start=df_event.index.min(), end=df_event.index.max(), freq='ME')
    factor = df_event.reindex(full_idx).ffill().diff().dropna()
    factor = factor.loc[factor.index.intersection(meeting_months)]
    
    factor = (factor - factor.mean()) / factor.std()
    factor.name = 'Narrative'
    return factor[factor.index >= pd.Timestamp(start_date)]

# ==========================================
# 2. Fama-MacBeth 核心逻辑
# ==========================================
def run_fmb(returns, factors):
    """运行 Fama-MacBeth 回归"""
    common = returns.index.intersection(factors.index)
    if len(common) < 20: return None
    
    y = returns.loc[common]
    X = factors.loc[common]
    
    # Step 1: Time Series
    X_ts = sm.add_constant(X)
    betas = []
    for port in y.columns:
        y_col = y[port]
        if not y_col.index.equals(X_ts.index):
            y_col = y_col.reindex(X_ts.index)
        res = sm.OLS(y_col, X_ts).fit()
        b = res.params.drop('const')
        b.name = port
        betas.append(b)
    df_betas = pd.DataFrame(betas)
    
    # Step 2: Cross Section
    y_cs = y.mean()
    X_cs = sm.add_constant(df_betas)
    if not y_cs.index.equals(X_cs.index):
        y_cs = y_cs.reindex(X_cs.index)
    res_cs = sm.OLS(y_cs, X_cs).fit()
    
    return {
        'lambdas': res_cs.params,
        'tvalues': res_cs.tvalues,
        'adj_r2': res_cs.rsquared_adj,
        'n_obs': len(common),
        'pred_ret': res_cs.predict(X_cs),
        'real_ret': y_cs,
        'betas': df_betas
    }

# ==========================================
# 3. 复杂表格生成 (Landscape Table)
# ==========================================
def generate_comprehensive_table(results_dict, model_order):
    """生成包含所有系数的横向大表 (修复下划线报错)"""
    print(f"Generating Comprehensive LaTeX Table: {OUT_TAB}")
    
    # 收集所有出现过的因子名称
    all_factors = set()
    for m in model_order:
        if results_dict[m]['aug']:
            all_factors.update(results_dict[m]['aug']['lambdas'].index.tolist())
    
    # 排序因子：Narrative 第一，Const 最后，其他字母序
    if 'const' in all_factors: all_factors.remove('const')
    if 'Narrative' in all_factors: all_factors.remove('Narrative')
    sorted_factors = ['Narrative'] + sorted(list(all_factors)) + ['const']
    
    def fmt(val, t):
        s = "***" if abs(t)>2.58 else "**" if abs(t)>1.96 else "*" if abs(t)>1.65 else ""
        return f"{val*100:.2f}{s} ({t:.2f})"

    # LaTeX Header (Landscape)
    latex = r"""
\begin{landscape}
\begin{table}[p]
\begin{adjustbox}{width=\linewidth, center} 
\begin{threeparttable}
\caption{Comprehensive Asset Pricing Tests: All Factor Models}
\label{tab:full_asset_pricing}
\scriptsize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.1}

\begin{tabular}{l""" + "c" * len(model_order) + r"""}
\toprule
 & \multicolumn{""" + str(len(model_order)) + r"""}{c}{\textbf{Augmented Models (Factor + Narrative)}} \\
\cmidrule(lr){2-""" + str(len(model_order)+1) + r"""}
Risk Premia ($\lambda$) & """ + " & ".join(model_order).replace('_', '-') + r""" \\
\midrule
"""
    
    # Body: Coefficients
    for factor in sorted_factors:
        # [关键修改] 将下划线替换为连字符，防止 LaTeX 报错
        factor_label = factor.replace('_', '-')
        
        # 特殊美化
        factor_label = factor_label.replace('Narrative', r'\textbf{Narrative}')
        if factor == 'const': factor_label = 'Constant ($\lambda_0$)'
        
        row_str = f"{factor_label} & "
        
        for m in model_order:
            res = results_dict[m]['aug']
            if res and factor in res['lambdas']:
                val = res['lambdas'][factor]
                t = res['tvalues'][factor]
                row_str += f"{fmt(val, t)} & "
            else:
                row_str += " & "
        latex += row_str[:-2] + r" \\" + "\n"

    # Statistics
    latex += r"\midrule" + "\n" + r"Adj. $R^2_{CS}$ & "
    for m in model_order:
        if results_dict[m]['aug']:
            latex += f"{results_dict[m]['aug']['adj_r2']:.2f} & "
        else: latex += "- & "
    latex = latex[:-2] + r" \\" + "\n"
    
    latex += r"Event Months & "
    for m in model_order:
        if results_dict[m]['aug']:
            latex += f"{results_dict[m]['aug']['n_obs']} & "
        else: latex += "- & "
    latex = latex[:-2] + r" \\" + "\n"

    # Panel A R2 (Base Model Comparison)
    latex += r"\midrule" + "\n" + r"\textit{Base Model Adj. $R^2$} & "
    for m in model_order:
        if results_dict[m]['base']:
            latex += f"{results_dict[m]['base']['adj_r2']:.2f} & "
        else: latex += "- & "
    latex = latex[:-2] + r" \\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}[para,flushleft]
  \item Note: This table reports the risk premia ($\lambda$) estimated from Fama-MacBeth cross-sectional regressions on 49 Industry Portfolios. All models include the Narrative Ambiguity factor. Coefficients are in percentage. t-statistics are reported in parentheses. 'Base Model Adj. $R^2$' refers to the model fit without the Narrative factor.
\end{tablenotes}
\end{threeparttable}
\end{adjustbox}
\end{table}
\end{landscape}
"""
    with open(OUT_TAB, 'w', encoding='utf-8') as f:
        f.write(latex)

# ==========================================
# 4. 绘图逻辑 (Main vs Appendix)
# ==========================================
# def plot_grid_fits(results_dict, model_list, filename, title_prefix=""):
#     """通用的 Grid Fit Plot"""
#     print(f"Plotting Fits to {filename}...")
#     valid_models = [m for m in model_list if m in results_dict and results_dict[m]['aug']]
#     n = len(valid_models)
#     if n == 0: return

#     cols = 2
#     rows = math.ceil(n / cols)
    
#     fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
#     if n == 1: axes = [axes] # Handle single plot case
#     axes = np.array(axes).flatten()
    
#     for i, m in enumerate(valid_models):
#         ax = axes[i]
#         res = results_dict[m]['aug']
        
#         real = res['real_ret'] * 100
#         pred = res['pred_ret'] * 100
        
#         ax.scatter(pred, real, s=60, alpha=0.7, edgecolors='k', c='#4682B4')
        
#         # 45度线
#         lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
#         ax.plot(lims, lims, 'r--', alpha=0.5, zorder=0)
        
#         # 标注异常值
#         resid = (real - pred).abs()
#         top_resid = resid.nlargest(3).index.tolist()
#         important = ['Gold', 'Chips', 'Autos']
#         to_label = list(set(top_resid + important))
        
#         for txt in to_label:
#             if txt in real.index:
#                 ax.annotate(txt, (pred[txt], real[txt]), fontsize=8, xytext=(3,3), textcoords='offset points')
            
#         ax.set_title(f"{m} + Narrative (Adj $R^2$={res['adj_r2']:.2f})", fontweight='bold')
#         ax.set_xlabel("Predicted (%)")
#         ax.set_ylabel("Realized (%)")
        
#     # Hide empty
#     for j in range(i+1, len(axes)): fig.delaxes(axes[j])
    
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300)
#     plt.close()
def plot_grid_fits(results_dict, model_list, filename, title_prefix=""):
    """通用的 Grid Fit Plot (对比 Base vs Augmented)"""
    print(f"Plotting Fits to {filename}...")
    valid_models = [m for m in model_list if m in results_dict and results_dict[m]['aug']]
    n = len(valid_models)
    if n == 0: return

    cols = 2
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    if n == 1: axes = [axes] # Handle single plot case
    axes = np.array(axes).flatten()
    
    for i, m in enumerate(valid_models):
        ax = axes[i]
        res_aug = results_dict[m]['aug']
        res_base = results_dict[m]['base']
        
        # 准备数据 (转百分比)
        real = res_aug['real_ret'] * 100
        pred_aug = res_aug['pred_ret'] * 100
        
        # 确定坐标轴范围 (包含所有点)
        if res_base:
            pred_base = res_base['pred_ret'] * 100
            all_preds = pd.concat([pred_aug, pred_base])
        else:
            all_preds = pred_aug
            
        lim_min = min(real.min(), all_preds.min())
        lim_max = max(real.max(), all_preds.max())
        # 加一点边距
        padding = (lim_max - lim_min) * 0.1
        lim_min -= padding
        lim_max += padding
        
        # 1. 绘制 45度线 (基准)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label='Perfect Pricing', zorder=0)
        
        # 2. 绘制 Base Model (灰色)
        if res_base:
            pred_base = res_base['pred_ret'] * 100
            # 散点
            ax.scatter(pred_base, real, s=30, alpha=0.3, color='gray', marker='o', label='Base Model')
            # 拟合趋势线
            m_b, c_b = np.polyfit(pred_base, real, 1)
            x_line = np.array([lim_min, lim_max])
            ax.plot(x_line, m_b * x_line + c_b, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)

        # 3. 绘制 Augmented Model (蓝色)
        # 散点
        ax.scatter(pred_aug, real, s=60, alpha=0.8, edgecolors='k', c='#4682B4', label='+ Narrative', zorder=10)
        # 拟合趋势线
        m_a, c_a = np.polyfit(pred_aug, real, 1)
        x_line = np.array([lim_min, lim_max])
        ax.plot(x_line, m_a * x_line + c_a, color='#4682B4', linestyle='-', linewidth=2, zorder=9)
        
        # 4. 标注异常值 (只标注 Augmented 的，避免混乱)
        resid = (real - pred_aug).abs()
        top_resid = resid.nlargest(3).index.tolist()
        important = ['Gold', 'Chips', 'Autos']
        to_label = list(set(top_resid + important))
        
        for txt in to_label:
            if txt in real.index:
                # 稍微偏移一点
                ax.annotate(txt, (pred_aug[txt], real[txt]), fontsize=8, xytext=(3,3), textcoords='offset points')
        
        # 5. 标题显示 R2 提升
        r2_base_str = f"{res_base['adj_r2']:.2f}" if res_base else "N/A"
        r2_aug_str = f"{res_aug['adj_r2']:.2f}"
        ax.set_title(f"{m}: Adj $R^2$ ({r2_base_str} $\\rightarrow$ {r2_aug_str})", fontweight='bold')
        
        ax.set_xlabel("Predicted Returns (%)")
        ax.set_ylabel("Realized Returns (%)")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        
        # 图例只显示一次或精简
        ax.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.9)
        
    # Hide empty axes
    for j in range(i+1, len(axes)): fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_grid_betas(results_dict, model_list, filename):
    """通用的 Grid Beta Plot"""
    print(f"Plotting Betas to {filename}...")
    valid_models = [m for m in model_list if m in results_dict and results_dict[m]['aug']]
    n = len(valid_models)
    if n == 0: return

    cols = 2
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    if n == 1: axes = [axes]
    axes = np.array(axes).flatten()
    
    for i, m in enumerate(valid_models):
        ax = axes[i]
        betas = results_dict[m]['aug']['betas']['Narrative'].sort_values()
        
        # 只画 Top 5 和 Bottom 5
        plot_data = pd.concat([betas.head(5), betas.tail(5)])
        
        colors = ['#8B0000' if x < 0 else '#006400' for x in plot_data.values]
        ax.barh(plot_data.index, plot_data.values, color=colors, alpha=0.8)
        ax.axvline(0, color='k', linewidth=0.8)
        
        ax.set_title(f'{m}: Narrative Betas', fontsize=10, fontweight='bold')
        
    for j in range(i+1, len(axes)): fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ==========================================
# 5. 主程序
# ==========================================
def main():
    # 1. 准备数据
    df_ind = get_local_industry_49()
    df_disp = pd.read_csv(FILE_DISPERSION)
    s_narrative = construct_narrative_factor(df_disp)
    
    if df_ind is None: return

    # 2. 获取外部因子
    factor_models = fetch_all_factors()
    if not factor_models: return

    # 3. 循环回归
    results = {}
    print("\nStarting Cross-Sectional Regressions...")
    
    # 排序：Main Models first
    main_models = ['FF3', 'HXZ5', 'DHS', 'Barillas'] # Main Text
    # 剩余的放 Appendix
    all_keys = list(factor_models.keys())
    appendix_models = [m for m in all_keys if m not in main_models]
    
    # 全集顺序用于表格
    full_order = main_models + appendix_models
    
    for m_name in full_order:
        df_f = factor_models[m_name]
        print(f"  Model: {m_name}")
        
        # Base
        common_base = df_ind.index.intersection(df_f.index).intersection(s_narrative.index)
        if len(common_base) < 20:
            results[m_name] = {'base': None, 'aug': None}
            continue
            
        res_base = run_fmb(df_ind.loc[common_base], df_f.loc[common_base])
        
        # Aug
        df_aug = df_f.copy()
        df_aug['Narrative'] = s_narrative
        common_aug = df_ind.index.intersection(df_aug.dropna().index)
        res_aug = run_fmb(df_ind.loc[common_aug], df_aug.loc[common_aug])
        
        results[m_name] = {'base': res_base, 'aug': res_aug}
        
        if res_aug:
            lam = res_aug['lambdas']['Narrative']
            t = res_aug['tvalues']['Narrative']
            print(f"    -> Narrative Lambda: {lam:.4f} (t={t:.2f})")

    # 4. 输出
    if results:
        # A. Excel
        # save_results_to_excel(results) # 暂略，逻辑同上个版本
        
        # B. LaTeX Table (Landscape, All Models)
        generate_comprehensive_table(results, full_order)
        
        # C. Main Figures (4 Models)
        plot_grid_fits(results, main_models, OUT_FIG_FIT_MAIN)
        plot_grid_betas(results, main_models, OUT_FIG_BETA_MAIN)
        
        # D. Appendix Figures (The Rest)
        plot_grid_fits(results, appendix_models, OUT_FIG_FIT_APP)
        plot_grid_betas(results, appendix_models, OUT_FIG_BETA_APP)
        
        print("\nAnalysis Completed.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()