import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from getfactormodels import FamaFrenchFactors
import os
import glob
import re
import pickle

# ==========================================
# 0. 配置
# ==========================================
DATA_DIR = "data"
FILE_PATTERN = os.path.join(DATA_DIR, "llm_fomc_dispersion_results_*.csv")
FILE_IND49 = r"data\ff3\49_Industry_Portfolios_Value_Weighted_Monthly.csv"
FILE_CACHE_PKL = r"data\factor_cache\all_factors_cache_v2.pkl"

OUT_FIG_SCATTER = r"fig\fig1.7_persistence_vs_tstat.png"
OUT_FIG_RESCUE = r"fig\fig1.8_qwen3max_rescue.png"

os.makedirs(os.path.dirname(OUT_FIG_SCATTER), exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 基础工具函数
# ==========================================
def load_specific_llm(model_name):
    # 查找文件
    f = os.path.join(DATA_DIR, f"llm_fomc_dispersion_results_{model_name}.csv")
    if not os.path.exists(f): return None
    df = pd.read_csv(f)
    if 'Semantic_Dispersion' not in df.columns:
        # 兼容旧命名
        col = [c for c in df.columns if 'dispersion' in c.lower()][0]
        df.rename(columns={col: 'Semantic_Dispersion'}, inplace=True)
    
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
    # 构造月度因子
    df['month'] = df['date'] + pd.offsets.MonthEnd(0)
    df_event = df.groupby('month')['Semantic_Dispersion'].mean()
    
    # 填充并差分
    full_idx = pd.date_range(start=df_event.index.min(), end=df_event.index.max(), freq='ME')
    factor = df_event.reindex(full_idx).ffill().diff().dropna()
    # 只保留会议月
    factor = factor.loc[factor.index.intersection(df_event.index)]
    
    # 标准化
    return (factor - factor.mean()) / factor.std()

def get_ff3_and_ind49():
    # Load FF3
    if os.path.exists(FILE_CACHE_PKL):
        with open(FILE_CACHE_PKL, 'rb') as f:
            ff3 = pickle.load(f)['FF3']
    else:
        # Fallback
        ff3 = FamaFrenchFactors(model=3, frequency='m').download()
        if ff3.abs().mean().mean() > 0.2: ff3 /= 100.0
        ff3.index = ff3.index.to_timestamp(freq='M') + pd.offsets.MonthEnd(0)
    
    # Load Ind49
    df_ind = pd.read_csv(FILE_IND49)
    df_ind.columns = [c.strip() for c in df_ind.columns]
    df_ind = df_ind.dropna(subset=['Date'])
    df_ind.index = pd.to_datetime(df_ind['Date'].astype(int).astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
    df_ind = df_ind.drop(columns=['Date']).replace([-99.99, -999], np.nan).dropna().astype(float)/100.0
    
    return ff3, df_ind

def run_fmb_tstat(factor_series, ff3, ind49):
    """跑一次 FMB 回归，只返回 Narrative 因子的 t-stat"""
    # Align
    common = ind49.index.intersection(ff3.index).intersection(factor_series.index)
    if len(common) < 50: return 0
    
    y = ind49.loc[common]
    X = ff3.loc[common].copy()
    X['Narrative'] = factor_series.loc[common]
    X_ts = sm.add_constant(X)
    
    # TS
    betas = []
    for port in y.columns:
        res = sm.OLS(y[port], X_ts).fit()
        betas.append(res.params['Narrative'])
    
    # CS
    y_cs = y.mean()
    X_cs = sm.add_constant(pd.DataFrame({'Beta': betas}, index=y.columns))
    res_cs = sm.OLS(y_cs, X_cs).fit()
    
    return res_cs.tvalues['Beta']

# ==========================================
# 2. 实验一：元回归 (Scatter Plot)
# ==========================================
def experiment_1_meta_analysis(ff3, ind49):
    print("\nRunning Experiment 1: Persistence vs Pricing Power...")
    files = glob.glob(FILE_PATTERN)
    results = []
    
    for f in files:
        # Extract name
        fname = os.path.basename(f)
        match = re.search(r'results_(.+)\.csv', fname)
        model_name = match.group(1) if match else fname
        
        # Load Raw Series for AR(1) calculation
        df_raw = pd.read_csv(f)
        series = df_raw['Semantic_Dispersion'] if 'Semantic_Dispersion' in df_raw.columns else df_raw.iloc[:, -1]
        ar1 = series.autocorr(lag=1)
        
        # Load Factor for Pricing Test
        factor = load_specific_llm(model_name)
        if factor is None: continue
        
        t_stat = run_fmb_tstat(factor, ff3, ind49)
        
        results.append({
            'Model': model_name,
            'AR1': ar1,
            'Abs_T_Stat': abs(t_stat),
            'T_Stat_Raw': t_stat
        })
        
    df_res = pd.DataFrame(results)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df_res, x='AR1', y='Abs_T_Stat', scatter_kws={'s': 100}, line_kws={'color': 'red', 'alpha': 0.5})
    
    for i, row in df_res.iterrows():
        plt.text(row['AR1']+0.005, row['Abs_T_Stat'], row['Model'], fontsize=9)
        
    plt.xlabel('Persistence (AR(1))', fontsize=12)
    plt.ylabel('Pricing Power (|t-statistic|)', fontsize=12)
    plt.title('Mechanism Check: Persistence Drives Pricing', fontsize=14, fontweight='bold')
    
    # Calc correlation
    corr = df_res['AR1'].corr(df_res['Abs_T_Stat'])
    plt.text(0.05, 0.9, f"Correlation = {corr:.2f}", transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUT_FIG_SCATTER, dpi=300)
    print(f"Scatter plot saved. Correlation: {corr:.4f}")

# ==========================================
# 3. 实验二：拯救 Qwen3-Max (Signal Extraction)
# ==========================================
def experiment_2_rescue_qwen3max(ff3, ind49):
    print("\nRunning Experiment 2: Rescuing Qwen3-max via Smoothing...")
    
    target_model = "qwen3-max"
    
    # 1. Load Raw Data
    f = os.path.join(DATA_DIR, f"llm_fomc_dispersion_results_{target_model}.csv")
    if not os.path.exists(f):
        print(f"Error: {target_model} data not found.")
        return

    df = pd.read_csv(f)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
    df = df.set_index('date').sort_index()
    raw_series = df['Semantic_Dispersion'] if 'Semantic_Dispersion' in df.columns else df.iloc[:, -1]
    
    # 2. Apply Smoothing (Signal Extraction)
    # 使用向后窗口 (Rolling Mean) 模拟投资者提取趋势，避免前视偏差
    # 窗口设为 3 (季度) 或 6 (半年)
    windows = [1, 3, 6, 9] 
    t_stats = []
    ar1s = []
    
    for w in windows:
        # Construction Logic
        if w == 1:
            # Raw (No smoothing)
            smoothed = raw_series
        else:
            smoothed = raw_series.rolling(window=w).mean()
            
        # Standard Factor Construction (Agg -> Diff -> Filter Event)
        # Note: We need to handle the monthly aggregation carefully with smoothed data
        df_temp = pd.DataFrame({'val': smoothed})
        df_temp['month'] = df_temp.index + pd.offsets.MonthEnd(0)
        df_event = df_temp.groupby('month')['val'].last() # 取当月最后一个平滑值
        
        full_idx = pd.date_range(start=df_event.index.min(), end=df_event.index.max(), freq='ME')
        factor = df_event.reindex(full_idx).ffill().diff().dropna()
        # Filter Event Months
        valid_months = df_temp['month'].unique()
        factor = factor.loc[factor.index.intersection(valid_months)]
        
        # Test Pricing
        t = run_fmb_tstat(factor, ff3, ind49)
        ar1 = smoothed.dropna().autocorr(lag=1)
        
        t_stats.append(t)
        ar1s.append(ar1)
        print(f"  Window {w}: AR(1)={ar1:.2f}, t-stat={t:.2f}")

    # Plotting the Rescue
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(windows))
    ax1.bar(x, t_stats, color=['#A9A9A9', '#4682B4', '#003366', '#000080'], alpha=0.8, label='t-statistic')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Raw\n(MA-1)", "MA-3\n(Quarterly)", "MA-6\n(Semiannual)", "MA-9"])
    ax1.set_ylabel('Pricing Power (t-statistic)', fontsize=12, fontweight='bold', color='#003366')
    ax1.axhline(1.65, color='red', linestyle='--', label='10% Significance')
    ax1.axhline(0, color='black', linewidth=0.5)
    
    # Twin axis for AR(1)
    ax2 = ax1.twinx()
    ax2.plot(x, ar1s, color='darkorange', marker='o', linewidth=2, label='Persistence (AR1)')
    ax2.set_ylabel('Signal Persistence (AR1)', fontsize=12, fontweight='bold', color='darkorange')
    ax2.set_ylim(0, 1.0)
    
    plt.title(f'Rescuing {target_model}: Smoothing Induces Persistence & Pricing', fontsize=13, fontweight='bold')
    
    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUT_FIG_RESCUE, dpi=300)
    print("Rescue plot saved.")

def main():
    ff3, ind49 = get_ff3_and_ind49()
    
    # Experiment 1
    experiment_1_meta_analysis(ff3, ind49)
    
    # Experiment 2
    experiment_2_rescue_qwen3max(ff3, ind49)

if __name__ == "__main__":
    main()