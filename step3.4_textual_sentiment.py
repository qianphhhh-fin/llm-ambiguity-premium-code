import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import re
import os

# ==========================================
# 0. 配置路径
# ==========================================
FILE_RAW_TEXT = "step1_fomc_statements_clean.csv"
FILE_DISPERSION = "step2_fomc_dispersion_results.csv"
FILE_VIX = r"data\vix\vix_daily.csv"
FILE_LM_DICT = r"data\Loughran-McDonald_MasterDictionary_1993-2024.csv"

OUT_FIG_PATH = r"fig\fig1.2_step3.4_textual_sentiment.png"
OUT_TAB_PATH = r"tab\Tab1.5_step3.4_textual_sentiment.tex"

# 确保目录存在
os.makedirs(os.path.dirname(OUT_FIG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUT_TAB_PATH), exist_ok=True)

# 绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心逻辑：LM 词典解析与计算
# ==========================================
def load_lm_dictionary():
    print("Loading Loughran-McDonald Dictionary...")
    try:
        df_dict = pd.read_csv(FILE_LM_DICT)
        # 转换为大写，建立集合以加速查找
        neg_words = set(df_dict[df_dict['Negative'] > 0]['Word'].str.upper())
        pos_words = set(df_dict[df_dict['Positive'] > 0]['Word'].str.upper())
        unc_words = set(df_dict[df_dict['Uncertainty'] > 0]['Word'].str.upper())
        return neg_words, pos_words, unc_words
    except Exception as e:
        print(f"Error loading dictionary: {e}")
        return None, None, None

def calculate_sentiment(text, neg_set, pos_set, unc_set):
    if not isinstance(text, str): return np.nan, np.nan, np.nan
    
    # 简单的正则分词 (全部转大写以匹配词典)
    tokens = re.findall(r'\b[A-Z]{2,}\b', text.upper())
    total = len(tokens)
    if total == 0: return 0, 0, 0
    
    neg_count = sum(1 for w in tokens if w in neg_set)
    pos_count = sum(1 for w in tokens if w in pos_set)
    unc_count = sum(1 for w in tokens if w in unc_set)
    
    # 构造指标
    # Tone: (Pos - Neg) / Total (Net Optimism)
    # 也有人用 Neg / Total，这里我们用 Net Tone，更全面
    tone = (pos_count - neg_count) / total
    
    # LM Uncertainty
    unc_freq = unc_count / total
    
    return tone, unc_freq, total

def process_data():
    # Load Dict
    neg_s, pos_s, unc_s = load_lm_dictionary()
    if neg_s is None: return None
    
    # Load Text
    df_text = pd.read_csv(FILE_RAW_TEXT)
    df_text['date'] = pd.to_datetime(df_text['date']).dt.tz_localize(None).dt.normalize()
    
    # Calculate Scores
    print("Calculating Sentiment Scores (this may take a moment)...")
    results = df_text['text'].apply(lambda x: calculate_sentiment(x, neg_s, pos_s, unc_s))
    df_text['LM_Tone'] = [x[0] for x in results]
    df_text['LM_Uncertainty'] = [x[1] for x in results]
    
    # Load D_t
    df_d = pd.read_csv(FILE_DISPERSION)
    df_d['date'] = pd.to_datetime(df_d['date']).dt.tz_localize(None).dt.normalize()
    
    # Load VIX (for regression check)
    df_vix = pd.read_csv(FILE_VIX)
    df_vix['Date'] = pd.to_datetime(df_vix['Date'])
    df_vix = df_vix.set_index('Date').sort_index()
    
    # Merge D and Text Stats
    df_merged = pd.merge(df_d, df_text[['date', 'LM_Tone', 'LM_Uncertainty']], on='date', how='inner')
    
    # Merge VIX (AsOf)
    df_merged = pd.merge_asof(df_merged.sort_values('date'), df_vix['VIX'], 
                              left_on='date', right_index=True, direction='backward')
    
    # Calculate Changes (Shocks)
    for col in ['Semantic_Dispersion', 'LM_Tone', 'LM_Uncertainty', 'VIX']:
        df_merged[f'd_{col}'] = df_merged[col].diff()
        
    return df_merged.dropna()

# ==========================================
# 2. 绘图 (严格限宽 < 14.99cm)
# ==========================================
def plot_comparison(df):
    print(f"Plotting to {OUT_FIG_PATH}...")
    
    # 14.99 cm = 5.9 inches
    fig, ax1 = plt.subplots(figsize=(5.8, 3.5)) 
    
    # --- 轴 1: LM Tone (背景) ---
    # Tone 通常在 0 附近波动。我们用面积图表示
    color_tone = "#8A8686" # DarkGray
    ax1.fill_between(df['date'], df['LM_Tone'], color=color_tone, alpha=0.3, label='LM Net Tone (Sentiment)')
    ax1.set_ylabel('Net Tone (Pos - Neg)', color='black', fontsize=9, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=8)

     # 设置网格
    ax1.grid(visible=False, axis='y')
    ax1.grid(visible=False, axis='x')

    # --- 轴 2: D_t (前景) ---
    ax2 = ax1.twinx()
    color_dt = '#003366' # Navy Blue
    ax2.plot(df['date'], df['Semantic_Dispersion'], color=color_dt, linewidth=0.5, 
             marker='o', markersize=1.5, label='Narrative Ambiguity ($D_t$)')
    ax2.set_ylabel('Semantic Dispersion ($D_t$)', color=color_dt, fontsize=9, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_dt, labelsize=8)

        # 调整左轴范围：让柱子只占据下方 2/3，给上方的线留空间
    y_min, y_max = df['Semantic_Dispersion'].min(),df['Semantic_Dispersion'].max()
    ax2.set_ylim(bottom=y_min * 0.8, top=y_max *1.2) 
    
    # 相关性标注
    corr = df[['Semantic_Dispersion', 'LM_Tone']].corr().iloc[0,1]
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray', linewidth=0.5)
    ax2.text(0.02, 0.95, f"Corr($D_t$, Tone) = {corr:.2f}", transform=ax1.transAxes, fontsize=8,
             verticalalignment='top', bbox=props)
    ax2.grid(visible=True, which='major', axis='y', linestyle='--', alpha=0.3)
    ax2.grid(visible=False, axis='x')
    plt.title('Semantic Ambiguity vs. LM Sentiment Tone', fontsize=10, weight='bold')
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels() # fill_between handles are tricky, often creates PolyCollection
    # 手动创建 Legend handle for fill
    import matplotlib.patches as mpatches
    patch = mpatches.Patch(color=color_tone, alpha=0.3, label='LM Net Tone')
    line2 = ax2.get_lines()[0]
    
    # ax1.legend([patch, line2], ['LM Net Tone', 'Narrative Ambiguity ($D_t$)'], 
    #            loc='upper right', frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUT_FIG_PATH, dpi=300)

# ==========================================
# 3. 统计检验与制表
# ==========================================
def run_tests_and_table(df):
    print("Running sophisticated tests...")
    
    # Standardize for regression comparison
    cols = ['d_Semantic_Dispersion', 'd_LM_Tone', 'd_LM_Uncertainty']
    for c in cols:
        df[c+'_Z'] = (df[c] - df[c].mean()) / df[c].std()
    
    # --- Test 1: Orthogonality (D ~ Tone + LM_Unc) ---
    # 检验 D_t 是否能被传统指标解释
    X_orth = sm.add_constant(df[['d_LM_Tone_Z', 'd_LM_Uncertainty_Z']])
    y_orth = df['d_Semantic_Dispersion_Z']
    res_orth = sm.OLS(y_orth, X_orth).fit(cov_type='HAC', cov_kwds={'maxlags':1})
    
    # --- Test 2: VIX Horse Race (VIX ~ D + Tone + LM_Unc) ---
    X_vix = sm.add_constant(df[['d_Semantic_Dispersion_Z', 'd_LM_Tone_Z', 'd_LM_Uncertainty_Z']])
    y_vix = df['d_VIX']
    res_vix = sm.OLS(y_vix, X_vix).fit(cov_type='HAC', cov_kwds={'maxlags':1})
    
    # Correlations
    corr_tone = df[['d_Semantic_Dispersion', 'd_LM_Tone']].corr().iloc[0,1]
    corr_unc = df[['d_Semantic_Dispersion', 'd_LM_Uncertainty']].corr().iloc[0,1]

    # --- Generate LaTeX ---
    def get_fmt(model, var):
        if var not in model.params: return "", ""
        c = model.params[var]
        t = model.tvalues[var]
        s = "***" if abs(t)>2.58 else "**" if abs(t)>1.96 else "*" if abs(t)>1.65 else ""
        return f"{c:.3f}{s}", f"({t:.2f})"

    latex = r"""
\begin{table}[h!]
\centering
\caption{Narrative Ambiguity vs. Traditional Textual Metrics}
\label{tab:textual_sentiment}
\begin{threeparttable}
\begin{tabular}{lcc}
\toprule
 & \multicolumn{1}{c}{Orthogonality Test} & \multicolumn{1}{c}{VIX Explanation} \\
 & Dep. Var: $\Delta D_t$ & Dep. Var: $\Delta \text{VIX}_t$ \\
\cmidrule(lr){2-2} \cmidrule(lr){3-3}
Variable & (1) & (2) \\
\midrule
\multicolumn{3}{l}{\textbf{Panel A: Regression Analysis}} \\
    \hspace{1em}Narrative Shock ($\Delta D_t$) & & """ + get_fmt(res_vix, 'd_Semantic_Dispersion_Z')[0] + r""" \\
         & & """ + get_fmt(res_vix, 'd_Semantic_Dispersion_Z')[1] + r""" \\
\addlinespace
    \hspace{1em}LM Net Tone ($\Delta \text{Tone}_t$) & """ + get_fmt(res_orth, 'd_LM_Tone_Z')[0] + r""" & """ + get_fmt(res_vix, 'd_LM_Tone_Z')[0] + r""" \\
         & """ + get_fmt(res_orth, 'd_LM_Tone_Z')[1] + r""" & """ + get_fmt(res_vix, 'd_LM_Tone_Z')[1] + r""" \\
\addlinespace
    \hspace{1em}LM Uncertainty ($\Delta \text{LM\_Unc}_t$) & """ + get_fmt(res_orth, 'd_LM_Uncertainty_Z')[0] + r""" & """ + get_fmt(res_vix, 'd_LM_Uncertainty_Z')[0] + r""" \\
         & """ + get_fmt(res_orth, 'd_LM_Uncertainty_Z')[1] + r""" & """ + get_fmt(res_vix, 'd_LM_Uncertainty_Z')[1] + r""" \\
\midrule
Adj. $R^2$ & """ + f"{res_orth.rsquared_adj:.3f}" + r""" & """ + f"{res_vix.rsquared_adj:.3f}" + r""" \\
Observations & """ + f"{int(res_vix.nobs)}" + r""" & """ + f"{int(res_vix.nobs)}" + r""" \\
\midrule
\multicolumn{3}{l}{\textbf{Panel B: Correlation with $\Delta D_t$}} \\
    \hspace{1em}Corr w/ $\Delta \text{Tone}_t$ & \multicolumn{2}{c}{""" + f"{corr_tone:.3f}" + r"""} \\
    \hspace{1em}Corr w/ $\Delta \text{LM\_Unc}_t$ & \multicolumn{2}{c}{""" + f"{corr_unc:.3f}" + r"""} \\
\bottomrule
\end{tabular}
\begin{tablenotes}[para,flushleft]
  \item Note: This table distinguishes the LLM-based Narrative Ambiguity ($D_t$) from traditional dictionary-based textual metrics constructed using the Loughran-McDonald (2011) Master Dictionary. Column (1) regresses $\Delta D_t$ on changes in LM Tone and LM Uncertainty to test for redundancy. Column (2) runs a horse race to explain VIX changes. All variables are standardized. Newey-West standard errors (1 lag) are used.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    with open(OUT_TAB_PATH, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"Table saved to {OUT_TAB_PATH}")

def main():
    df = process_data()
    if df is not None:
        plot_comparison(df)
        run_tests_and_table(df)

if __name__ == "__main__":
    main()