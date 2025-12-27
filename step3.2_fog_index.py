import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import textstat
import os

# ==========================================
# 0. 配置路径与绘图参数
# ==========================================
FILE_RAW_TEXT = "step1_fomc_statements_clean.csv" 
FILE_DISPERSION = "step2_fomc_dispersion_results.csv"
OUT_FIG_DIR = "fig"
OUT_FIG_PATH = os.path.join(OUT_FIG_DIR, "fig1.1_step3.2_fog_index.png")

# 确保输出目录存在
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# --- 顶级期刊绘图尺寸设置 ---
# 目标宽度: <= 14.99 cm
# Matplotlib 使用英寸 (inch). 1 inch = 2.54 cm
# 宽度设为 5.8 英寸 (约 14.73 cm)，留出微小余量防止边缘裁切
FIG_WIDTH_INCH = 14.75 / 2.54 
FIG_HEIGHT_INCH = 8 / 2.54   # 高度设为约 8 cm，保持长宽比协调 (接近 2:1)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9          # 基础字号调小以适应尺寸
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

def calculate_readability(text):
    """
    计算 Gunning Fog Index (复杂度)
    """
    try:
        if not isinstance(text, str) or len(text) < 10:
            return np.nan
        return textstat.gunning_fog(text)
    except:
        return np.nan

def load_and_process_data():
    print("Loading data...")
    
    # 1. Load Raw Text
    df_text = pd.read_csv(FILE_RAW_TEXT)
    df_text['date'] = pd.to_datetime(df_text['date']).dt.tz_localize(None).dt.normalize()
    
    print("Calculating Gunning Fog Index (Syntactic Complexity)...")
    # 这可能需要几秒钟
    df_text['Fog_Index'] = df_text['text'].apply(calculate_readability)
    
    # 2. Load D_t
    df_d = pd.read_csv(FILE_DISPERSION)
    df_d['date'] = pd.to_datetime(df_d['date']).dt.tz_localize(None).dt.normalize()
    
    # 3. Merge
    df_merged = pd.merge(df_d, df_text[['date', 'Fog_Index']], on='date', how='inner')
    df_merged = df_merged.sort_values('date')
    
    # 剔除无效值
    df_final = df_merged.dropna(subset=['Semantic_Dispersion', 'Fog_Index'])
    
    print(f"Final Sample: {len(df_final)} events")
    return df_final

def analyze_and_plot(df):
    print("Running Statistical Tests...")
    
    # 统计检验
    corr_p, p_val_p = stats.pearsonr(df['Semantic_Dispersion'], df['Fog_Index'])
    corr_s, p_val_s = stats.spearmanr(df['Semantic_Dispersion'], df['Fog_Index'])
    
    print(f"Pearson  Corr: {corr_p:.3f} (p={p_val_p:.3f})")
    
    # --- 绘图 ---
    print(f"Plotting to {OUT_FIG_PATH} (Width: {FIG_WIDTH_INCH:.2f} inches)...")
    
    fig, ax1 = plt.subplots(figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH))
    
    # 设置网格
    ax1.grid(visible=False, axis='y')
    ax1.grid(visible=False, axis='x')

    
    # --- 左轴: Fog Index (直方图) ---
    color_fog = "#474747" # 灰色
    # 柱子宽度设为 25 天，在时间轴上看起来比较合适
    ax1.bar(df['date'], df['Fog_Index'], color=color_fog, alpha=0.25, width=25, 
            label='Fog Index (Complexity)')
    
    # 轴标签
    ax1.set_ylabel('Fog Index (Years)', color=color_fog, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_fog)
    
    # 调整左轴范围：让柱子只占据下方 2/3，给上方的线留空间
    y_min, y_max = df['Fog_Index'].min(), df['Fog_Index'].max()
    ax1.set_ylim(bottom=y_min * 0.8, top=y_max *3) 
    
    # --- 右轴: D_t (折线图) ---
    ax2 = ax1.twinx()
    color_dt = '#003366' # 深海军蓝 (Academic Blue)
    
    ax2.plot(df['date'], df['Semantic_Dispersion'], color=color_dt, linewidth=0.5, 
             marker='o', markersize=1.5, label='Narrative Ambiguity ($D_t$)')
    
    ax2.set_ylabel('Semantic Dispersion ($D_t$)', color=color_dt, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_dt)

    ax2.grid(visible=True, which='major', axis='y', linestyle='--', alpha=0.3)
    ax2.grid(visible=False, axis='x')

    # --- 统计标注 (Inset Text) ---
    # 在图内左上角添加统计信息，节省空间
    stats_text = (
        f"$\\rho_{{Pearson}} \ \ \ = {corr_p:.2f}$ (p={p_val_p:.2f})\n"
        f"$\\rho_{{Spearman}} = {corr_s:.2f}$ (p={p_val_s:.2f})"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc', linewidth=0.5)
    ax2.text(0.5, 0.98, stats_text, transform=ax1.transAxes, fontsize=6,
             verticalalignment='top', horizontalalignment='center', bbox=props, fontname='Times New Roman')

    # --- 标题与图例 ---
    plt.title('Semantic Ambiguity vs. Syntactic Complexity (2000-2025)', fontsize=10, weight='bold', pad=10)
    
    # 合并图例
    # lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', 
    #            frameon=True, framealpha=0.9, edgecolor='white', fontsize=8)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存 (dpi=300 保证清晰度)
    plt.savefig(OUT_FIG_PATH, dpi=300, bbox_inches='tight')
    print("Done.")

if __name__ == "__main__":
    df = load_and_process_data()
    analyze_and_plot(df)