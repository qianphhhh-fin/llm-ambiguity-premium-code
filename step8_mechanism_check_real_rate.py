import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy import stats
import os

# ==========================================
# 配置
# ==========================================
FILE_DISPERSION = "step2_fomc_dispersion_results.csv" # 你的 D_t 数据
START_DATE = '2003-01-01' # TIPS 数据起点 (数据源限制)
END_DATE = '2021-12-31'   # 截断点 (避开2022年后的通胀干扰)

# 输出图片路径 (与 LaTeX 对应)
OUT_FIG_PATH = 'fig/fig_1.7_step8_check_real_rate_mechanism.png'

def main():
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUT_FIG_PATH), exist_ok=True)

    print("1. Loading Semantic Dispersion (Dt)...")
    if not os.path.exists(FILE_DISPERSION):
        print(f"Error: {FILE_DISPERSION} not found.")
        return
        
    df_d = pd.read_csv(FILE_DISPERSION)
    df_d['date'] = pd.to_datetime(df_d['date']).dt.tz_localize(None).dt.normalize()
    df_d = df_d.sort_values('date')
    
    print(f"2. Fetching Real Rates (TIPS) from FRED ({START_DATE} to {END_DATE})...")
    # DFII10: 10-Year Treasury Inflation-Indexed Security, Constant Maturity
    try:
        # 为了保险起见，多抓一点数据，然后本地过滤
        fetch_start = pd.to_datetime(START_DATE) - pd.Timedelta(days=30)
        fetch_end = pd.to_datetime(END_DATE) + pd.Timedelta(days=30)
        
        df_real = web.DataReader('DFII10', 'fred', fetch_start, fetch_end)
        df_real.columns = ['Real_Rate']
        df_real.index = df_real.index.tz_localize(None).normalize()
    except Exception as e:
        print(f"FRED Download Failed: {e}")
        return

    # 3. Merge (AsOf)
    # 以 D_t 的会议日期为基准，匹配最近的 TIPS 利率
    df_merged = pd.merge_asof(
        df_d,
        df_real.reset_index().sort_values('DATE').rename(columns={'DATE': 'date'}),
        on='date',
        direction='nearest',
        tolerance=pd.Timedelta(days=5) # 只匹配前后5天内的数据
    ).dropna()

    # 4. [关键] 严格过滤时间窗口
    # 确保只分析 START_DATE 到 END_DATE 之间的数据
    mask = (df_merged['date'] >= pd.to_datetime(START_DATE)) & (df_merged['date'] <= pd.to_datetime(END_DATE))
    df_final = df_merged.loc[mask].copy()
    
    if len(df_final) < 10:
        print("Not enough data points after filtering.")
        return

    # 5. 统计检验
    corr, p_val = stats.pearsonr(df_final['Semantic_Dispersion'], df_final['Real_Rate'])
    
    print("\n" + "="*40)
    print(f"RESULTS ({START_DATE} to {END_DATE})")
    print("="*40)
    print(f"Observations: {len(df_final)}")
    print(f"Correlation:  {corr:.4f}")
    print(f"P-value:      {p_val:.4f}")
    if p_val < 0.01:
        print(">> Result is STATISTICALLY SIGNIFICANT at 1% level.")
    elif p_val < 0.05:
        print(">> Result is STATISTICALLY SIGNIFICANT at 5% level.")
    else:
        print(">> Result is NOT significant.")
    print("="*40 + "\n")
    
    # 6. 绘图 (双轴)
    print(f"Plotting to {OUT_FIG_PATH}...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左轴：Real Rate (实线)
    color_rate = '#8B0000' # Dark Red
    ax1.plot(df_final['date'], df_final['Real_Rate'], color=color_rate, linewidth=2, label='10Y Real Rate (TIPS)')
    ax1.set_ylabel('Real Interest Rate (%)', color=color_rate, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_rate)
    
    # 右轴：Dt (虚线)
    ax2 = ax1.twinx()
    color_dt = '#003366' # Navy Blue
    ax2.plot(df_final['date'], df_final['Semantic_Dispersion'], color=color_dt, linewidth=2, linestyle='--', label='Narrative Ambiguity ($D_t$)')
    ax2.set_ylabel('Semantic Dispersion ($D_t$)', color=color_dt, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_dt)

    # 标题
    plt.title(f'Mechanism Check: Narrative Ambiguity vs. Real Rates\n({START_DATE} to {END_DATE}, Corr = {corr:.2f})', fontsize=14, fontweight='bold', pad=15)
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', frameon=True, framealpha=0.9, fontsize=10)
    
    # 网格
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_FIG_PATH, dpi=300)
    print("Done.")

if __name__ == "__main__":
    main()