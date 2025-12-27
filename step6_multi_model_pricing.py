import pandas as pd
import numpy as np
import statsmodels.api as sm
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
# [关键] 指定要检验的 LLM 模型名称
# TARGET_LLMS = [
#     "qwen-flash", 
#     "deepseek-v3.2",
#     "qwen3-max",
#     "qwen3-30b-a3b-instruct-2507",
#     "qwen3-8b",
#     "qwen3-14b",
#     "qwen3-4b"
# ]
TARGET_LLMS = [
    "deepseek-chat_CMC", 
    "qwen-flash_CMC",
    "qwen3-8b_CMC",
    "qwen3-4b_CMC",
    "qwen3-1.7b_CMC",
    "qwen3-0.6b_CMC",
]

FILE_IND49 = r"data\ff3\49_Industry_Portfolios_Value_Weighted_Monthly.csv"
DATA_DIR = "data"

# 缓存路径
DIR_CACHE = r"data\factor_cache"
FILE_CACHE_PKL = os.path.join(DIR_CACHE, "all_factors_cache_v2.pkl")

# 输出目录
OUT_TAB_DIR = r"tab"
os.makedirs(OUT_TAB_DIR, exist_ok=True)
os.makedirs(DIR_CACHE, exist_ok=True)

# ==========================================
# 1. 基础数据加载
# ==========================================
def process_downloaded_data(df):
    """统一处理下载数据的索引和单位"""
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp(freq='M') + pd.offsets.MonthEnd(0)
    else:
        df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty and df[numeric_cols].abs().mean().mean() > 0.2:
        df = df / 100.0
    return df.dropna()

def fetch_all_factors():
    """获取/读取所有因子模型缓存"""
    if os.path.exists(FILE_CACHE_PKL):
        print(f"Loading Standard Factor Models from Cache: {FILE_CACHE_PKL}")
        try:
            with open(FILE_CACHE_PKL, 'rb') as f:
                return pickle.load(f)
        except: pass

    print("Cache not found. Fetching via API...")
    models_data = {}
    
    tasks = [
        ('CAPM', FamaFrenchFactors, {'model': 3}), 
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
            if label == 'CAPM':
                df_clean = df_clean[['Mkt-RF']]
            models_data[label] = df_clean
        except Exception as e:
            print(f"  [Skip] Failed to fetch {label}: {e}")
            
    with open(FILE_CACHE_PKL, 'wb') as f:
        pickle.dump(models_data, f)
    return models_data

def get_local_industry_49():
    """读取本地 49 行业数据"""
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

def load_specific_llm_dispersion(llm_name):
    """加载特定 LLM 的离散度数据"""
    filename = f"llm_fomc_dispersion_results_{llm_name}.csv"
    file_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return None
        
    try:
        df = pd.read_csv(file_path)
        if 'Semantic_Dispersion' not in df.columns:
             cols = [c for c in df.columns if 'dispersion' in c.lower()]
             if cols:
                 df.rename(columns={cols[0]: 'Semantic_Dispersion'}, inplace=True)
             else:
                 raise ValueError("Column 'Semantic_Dispersion' not found")
        return df
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def construct_narrative_factor(df_disp, start_date='2000-01-01'):
    """
    构建叙事因子: Event-Only 
    [注意] 严格使用向后滚动窗口 (Trailing Window)，无 Look-ahead Bias。
    """
    # 1. 基础处理
    df_disp['date'] = pd.to_datetime(df_disp['date']).dt.tz_localize(None)
    df_disp['month'] = df_disp['date'] + pd.offsets.MonthEnd(0)
    
    # 2. 月度聚合 (取均值)
    df_event = df_disp.groupby('month')['Semantic_Dispersion'].mean()
    meeting_months = df_event.index
    
    # 3. 构建连续时间序列 (以处理无会议的月份)
    # 使用 'ME' 替代 'M' 以兼容新版 Pandas
    full_idx = pd.date_range(start=df_event.index.min(), end=df_event.index.max(), freq='ME')
    
    # 前向填充 (ffill): 假设如果没有会议，本月的认知状态延续上个月
    # 这是构建投资者"当前信息集"的标准做法
    df_continuous = df_event.reindex(full_idx).ffill()
    
    # 4. [关键修改] MA(3) 平滑 (Trailing Moving Average)
    # rolling(3) 默认是取 [t-2, t-1, t] 的均值，完全依赖过去信息
    df_smooth = df_continuous.rolling(window=1).mean()
    
    # 5. 计算变化 (Shock) 并剔除空值
    factor = df_smooth.diff().dropna()
    
    # 6. 只保留【真实发生会议】的月份
    # 逻辑：虽然我们每天都在平滑信念，但只有开会这天，信念的更新才会被资产重新定价
    factor = factor.loc[factor.index.intersection(meeting_months)]
    
    # 7. 标准化
    factor = (factor - factor.mean()) / factor.std()
    factor.name = 'Narrative'
    
    return factor[factor.index >= pd.Timestamp(start_date)]

# ==========================================
# 2. 回归引擎
# ==========================================
def run_fmb(returns, factors):
    common = returns.index.intersection(factors.index)
    if len(common) < 20: return None
    
    y = returns.loc[common]
    X = factors.loc[common]
    
    # Step 1: Time Series
    X_ts = sm.add_constant(X)
    betas = []
    for port in y.columns:
        y_col = y[port]
        if not y_col.index.equals(X_ts.index): y_col = y_col.reindex(X_ts.index)
        res = sm.OLS(y_col, X_ts).fit()
        b = res.params.drop('const')
        b.name = port
        betas.append(b)
    df_betas = pd.DataFrame(betas)
    
    # Step 2: Cross Section
    y_cs = y.mean()
    X_cs = sm.add_constant(df_betas)
    if not y_cs.index.equals(X_cs.index): y_cs = y_cs.reindex(X_cs.index)
    res_cs = sm.OLS(y_cs, X_cs).fit()
    
    return {
        'lambdas': res_cs.params,
        'tvalues': res_cs.tvalues,
        'adj_r2': res_cs.rsquared_adj,
        'n_obs': len(common)
    }

# ==========================================
# 3. 表格生成
# ==========================================
def generate_latex_for_llm(llm_name, results_dict, model_order):
    """为特定 LLM 生成 LaTeX 大表"""
    out_file = os.path.join(OUT_TAB_DIR, f"appendix_step6_asset_pricing_{llm_name.split("_")[0]}.tex")
    print(f"  Generating Table: {out_file}")
    
    all_factors = set()
    for m in model_order:
        if results_dict[m]['aug']:
            all_factors.update(results_dict[m]['aug']['lambdas'].index.tolist())
    
    if 'const' in all_factors: all_factors.remove('const')
    if 'Narrative' in all_factors: all_factors.remove('Narrative')
    sorted_factors = ['Narrative'] + sorted(list(all_factors)) + ['const']
    
    def fmt(val, t):
        s = "***" if abs(t)>2.58 else "**" if abs(t)>1.96 else "*" if abs(t)>1.65 else ""
        return f"{val*100:.2f}{s} ({t:.2f})"

    # 清洗 LLM 名字用于标题
    clean_name = llm_name.split("_")[0].replace('_', '-').upper()

    latex = r"""
\begin{landscape}
\begin{table}[p]
\begin{adjustbox}{width=\linewidth, center} 
\begin{threeparttable}
\caption{Robustness Check: Asset Pricing Tests using """ + clean_name + r"""}
\label{tab:pricing_""" + llm_name.replace('-', '_') + r"""}
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
    
    # Body
    for factor in sorted_factors:
        factor_label = factor.replace('_', '-')
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

    # Stats
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

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}[para,flushleft]
  \item Note: This table reports Fama-MacBeth regression results using narrative ambiguity constructed by the \textbf{""" + clean_name + r"""} model. Coefficients are in percentage.
\end{tablenotes}
\end{threeparttable}
\end{adjustbox}
\end{table}
\end{landscape}
"""
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(latex)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    # 1. Load Universal Data
    df_ind = get_local_industry_49()
    factor_models = fetch_all_factors()
    
    if df_ind is None or not factor_models:
        print("Error loading base data.")
        return

    # Model Order for Table
    main_models = ['FF3', 'HXZ5', 'DHS', 'Barillas']
    all_keys = list(factor_models.keys())
    appendix_models = [m for m in all_keys if m not in main_models]
    full_order = main_models + appendix_models

    # 2. Iterate through each LLM
    print(f"\nStarting Robustness Checks for {len(TARGET_LLMS)} LLMs...")
    
    for llm_name in TARGET_LLMS:
        print(f"\n>>> Processing LLM: {llm_name}")
        
        # Load specific dispersion
        df_disp = load_specific_llm_dispersion(llm_name)
        if df_disp is None: continue
        
        # Construct specific narrative factor (WITH MA-3)
        s_narrative = construct_narrative_factor(df_disp)
        
        # Run regressions for all factor models
        results = {}
        for m_name in full_order:
            df_f = factor_models[m_name]
            
            # Aug Model Only (Baseline is static)
            df_aug = df_f.copy()
            df_aug['Narrative'] = s_narrative
            
            common_aug = df_ind.index.intersection(df_aug.dropna().index)
            res_aug = run_fmb(df_ind.loc[common_aug], df_aug.loc[common_aug])
            
            results[m_name] = {'aug': res_aug}
            
            if res_aug:
                lam = res_aug['lambdas']['Narrative']
                t = res_aug['tvalues']['Narrative']
                print(f"    [{m_name}] Narrative Lambda: {lam:.4f} (t={t:.2f})")
        
        # Generate Table
        generate_latex_for_llm(llm_name, results, full_order)

    print("\nAll Robustness Checks Completed.")

if __name__ == "__main__":
    main()