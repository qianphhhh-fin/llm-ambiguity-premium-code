import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 0. 配置路径与参数
# ==========================================
DATA_DIR = "data"

# [关键修改] 在这里手动指定要对比的模型名称列表
# 对应文件名: llm_fomc_dispersion_results_{NAME}.csv
TARGET_MODELS = [
    "deepseek-chat_CMC", 
    "qwen-flash_CMC",
    "qwen3-8b_CMC",
    "qwen3-4b_CMC",
]

OUT_FIG = r"fig\fig1.5_step5_cross_model_comparison.png"
OUT_TAB = r"tab\Tab1.7_step5_cross_model_stats.tex"

# 确保目录存在
os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
os.makedirs(os.path.dirname(OUT_TAB), exist_ok=True)

# 绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 绘图尺寸
FIG_WIDTH = 5.8 
FIG_HEIGHT = 4.0

def load_specified_models():
    """根据 TARGET_MODELS 列表加载数据"""
    print(f"Loading data for models: {TARGET_MODELS}...")
    
    df_combined = pd.DataFrame()
    
    for model_name in TARGET_MODELS:
        filename = f"llm_fomc_dispersion_results_{model_name}.csv"
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"  [Warning] File not found: {file_path}. Skipping.")
            continue
            
        print(f"  Loading {model_name}...")
        try:
            df = pd.read_csv(file_path)
            # 处理日期格式
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
            # 提取 D_t 并重命名列
            df = df.set_index('date')[['Semantic_Dispersion']]
            df.columns = [model_name]
            
            # 合并 (Outer Join)
            if df_combined.empty:
                df_combined = df
            else:
                df_combined = df_combined.join(df, how='outer')
        except Exception as e:
            print(f"  [Error] Failed to load {model_name}: {e}")
            
    # 按时间排序并去除包含空值的行 (取交集，确保对比公平)
    original_len = len(df_combined)
    df_combined = df_combined.sort_index().dropna()
    print(f"Data Merged: {len(df_combined)} overlapping events (dropped {original_len - len(df_combined)} non-overlapping).")
    
    return df_combined

def calculate_cronbach_alpha(df):
    """计算 Cronbach's Alpha"""
    itemvars = df.values
    itemvars_count = itemvars.shape[1]
    variance_sum = itemvars.var(axis=0, ddof=1).sum()
    total_var = itemvars.sum(axis=1).var(ddof=1)
    
    alpha = (itemvars_count / (itemvars_count - 1)) * (1 - variance_sum / total_var)
    return alpha

def plot_time_series(df):
    """绘制时序对比图"""
    print(f"Plotting to {OUT_FIG}...")
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    colors = sns.color_palette("husl", n_colors=len(df.columns))
    
    # 绘制各模型
    for i, col in enumerate(df.columns):
        ax.plot(df.index, df[col], label=col, color=colors[i], linewidth=1.2, alpha=0.7)
    
    # 绘制共识线 (Mean)
    df['Consensus'] = df.mean(axis=1)
    ax.plot(df.index, df['Consensus'], label='Consensus (Mean)', color='black', 
            linewidth=1.8, linestyle='--', alpha=1.0)
    
    ax.set_ylabel('Narrative Ambiguity ($D_t$)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    
    # 图例
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8, ncol=2)
    
    plt.title('Cross-Model Consistency of Narrative Ambiguity', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)

def generate_stats_table(df):
    """生成统计表"""
    print(f"Generating Table to {OUT_TAB}...")
    
    # 排除 Consensus 列
    if 'Consensus' in df.columns:
        data = df.drop(columns=['Consensus'])
    else:
        data = df
        
    # 1. 相关系数矩阵
    corr_matrix = data.corr()
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    
    # 2. PCA
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=1)
    pca.fit(data_scaled)
    explained_variance = pca.explained_variance_ratio_[0]
    
    # 3. Alpha
    alpha = calculate_cronbach_alpha(data)
    
    # 4. LaTeX 生成
    models = list(data.columns)
    n = len(models)
    
    latex = r"""
\begin{table}[h!]
\centering
\caption{Cross-Model Validation of Narrative Ambiguity}
\label{tab:cross_model_stats}
\begin{threeparttable}
\begin{tabular}{l""" + "c" * n + r"""}
\toprule
 & \multicolumn{""" + str(n) + r"""}{c}{\textbf{Pairwise Correlations}} \\
\cmidrule(lr){2-""" + str(n+1) + r"""}
Model & """ + " & ".join([f"({i+1})" for i in range(n)]) + r""" \\
\midrule
\multicolumn{""" + str(n+1) + r"""}{l}{\textit{Panel A: Correlation Matrix}} \\
"""
    
    for i in range(n):
        # 替换下划线防止Latex报错
        model_label = models[i].replace('_', '-')
        row_str = f"({i+1}) {model_label} & "
        for j in range(n):
            if j <= i:
                val = corr_matrix.iloc[i, j]
                if j == i: row_str += "1.00 & "
                else: row_str += f"{val:.3f} & "
            else:
                row_str += " & "
        latex += row_str[:-2] + r" \\" + "\n"
        
    latex += r"""
\midrule
\multicolumn{""" + str(n+1) + r"""}{l}{\textit{Panel B: Common Factor Analysis}} \\
\hspace{1em}Cronbach's Alpha (Reliability) & \multicolumn{""" + str(n) + r"""}{c}{""" + f"{alpha:.3f}" + r"""} \\
\hspace{1em}PCA First Component ($R^2$) & \multicolumn{""" + str(n) + r"""}{c}{""" + f"{explained_variance:.3f}" + r"""} \\
\hspace{1em}Average Pairwise Correlation & \multicolumn{""" + str(n) + r"""}{c}{""" + f"{avg_corr:.3f}" + r"""} \\
\bottomrule
\end{tabular}
\begin{tablenotes}[para,flushleft]
  \item Note: Panel A reports the pairwise correlation coefficients of the $D_t$ time series generated by different Large Language Models. Panel B reports statistics testing the existence of a single common factor. Cronbach's Alpha $> 0.7$ indicates high internal consistency reliability. PCA First Component ($R^2$) measures the proportion of total variance explained by the first principal component of the model outputs.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    
    with open(OUT_TAB, 'w', encoding='utf-8') as f:
        f.write(latex)

def main():
    df = load_specified_models()
    if df is not None and not df.empty:
        if df.shape[1] < 2:
            print("Need at least 2 valid models to compare! Check your list and file paths.")
            return
            
        plot_time_series(df)
        generate_stats_table(df)
        print("\nStep 5 Completed Successfully.")
    else:
        print("No data loaded.")

if __name__ == "__main__":
    main()