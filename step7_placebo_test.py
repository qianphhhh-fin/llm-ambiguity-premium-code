import pandas as pd
import numpy as np
import os
import json
import statsmodels.api as sm
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# 配置参数
# ==========================================
INPUT_TEXT_FILE = "step1_fomc_statements_clean.csv"
INPUT_MARKET_FILE = r"data/sp500/sp500_daily.csv"
OUTPUT_CSV = "step7_placebo_predictions.csv"
OUTPUT_TEX = r"tab\Tab1.9_step7_placebo_test.tex"

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_TEX), exist_ok=True)

# DeepSeek API 配置
API_KEY = "sk-b0bc5ee8b28e4d5bb9116a3dcfe88f3e" 
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

# 预测窗口 (交易日)
HORIZONS = {
    'Daily': 1,
    'Weekly': 5,
    'Monthly': 20
}

# ==========================================
# 1. 数据处理函数
# ==========================================
def load_and_align_data():
    print("Loading and aligning data...")
    
    # 加载文本
    if not os.path.exists(INPUT_TEXT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_TEXT_FILE}")
    df_text = pd.read_csv(INPUT_TEXT_FILE)
    df_text['date'] = pd.to_datetime(df_text['date']).dt.tz_localize(None).dt.normalize()
    
    # 加载市场数据
    if not os.path.exists(INPUT_MARKET_FILE):
        raise FileNotFoundError(f"Missing {INPUT_MARKET_FILE}")
    
    # [关键修正] 这里的 thousands=',' 参数可以自动处理 1,000.00 这种格式
    df_mkt = pd.read_csv(INPUT_MARKET_FILE, thousands=',') 
    
    date_col = 'Date' if 'Date' in df_mkt.columns else 'date'
    # 优先找 Close，如果没有找 Adj Close
    if 'Close' in df_mkt.columns:
        price_col = 'Close'
    elif 'Adj Close' in df_mkt.columns:
        price_col = 'Adj Close'
    else:
        # 兜底：找除了日期外的第一列
        cols = [c for c in df_mkt.columns if c != date_col]
        price_col = cols[0]
        print(f"Warning: 'Close' column not found, using '{price_col}'")

    # [关键修正] 强制转换为数值型，非数值转为 NaN
    df_mkt[price_col] = pd.to_numeric(df_mkt[price_col], errors='coerce')
    
    # 处理日期
    df_mkt[date_col] = pd.to_datetime(df_mkt[date_col]).dt.tz_localize(None).dt.normalize()
    df_mkt = df_mkt.dropna(subset=[price_col, date_col]).sort_values(date_col).set_index(date_col)
    
    # 计算真实收益率
    aligned_data = []
    
    # 过滤掉 text 为空的数据
    df_text = df_text.dropna(subset=['text'])

    for dt in df_text['date']:
        try:
            # 检查日期是否在市场数据范围内
            # 如果 dt 超过了市场数据的最后一天，直接跳过
            if dt > df_mkt.index[-1] or dt < df_mkt.index[0]:
                continue

            # 找到最近的交易日（FOMC会议日）
            # method='nearest' 可能会匹配到未来或者很久以前，最好加个限制
            # 这里我们用 get_indexer 找最近的，但检查一下时间差
            loc_idx = df_mkt.index.get_indexer([dt], method='nearest')[0]
            matched_date = df_mkt.index[loc_idx]
            
            # 如果匹配到的日期相差超过 5 天，说明该日期附近没有数据，跳过
            if abs((matched_date - dt).days) > 5:
                continue
                
            curr_price = df_mkt.iloc[loc_idx][price_col]
            
            row = {'date': dt, 'text': df_text.loc[df_text['date'] == dt, 'text'].values[0]}
            
            # 计算不同窗口的真实收益率
            has_valid_target = False
            for label, days in HORIZONS.items():
                if loc_idx + days < len(df_mkt):
                    fut_price = df_mkt.iloc[loc_idx + days][price_col]
                    # [关键修正] 确保是浮点数运算
                    ret_pct = ((float(fut_price) - float(curr_price)) / float(curr_price)) * 100
                    row[f'Real_{label}'] = ret_pct
                    has_valid_target = True
                else:
                    row[f'Real_{label}'] = np.nan
            
            # 至少要有一个有效的未来收益率才保留
            if has_valid_target:
                aligned_data.append(row)
            
        except Exception as e:
            print(f"Skipping {dt}: {e}")
            
    if not aligned_data:
        raise ValueError("No aligned data found. Check date formats in both CSV files.")

    return pd.DataFrame(aligned_data)

# ==========================================
# 2. LLM 预测函数
# ==========================================
def get_llm_forecast(client, text):
    """
    要求 LLM 输出三个维度的预测值
    """
    prompt = f"""
    You are a financial analyst. Read the following FOMC statement snippet carefully.
    
    STATEMENT:
    "{text[:3000]}..."
    
    TASK:
    Predict the S&P 500 index returns (in percentage %) for the following horizons starting from the statement release date:
    1. Daily (Next 1 trading day)
    2. Weekly (Next 5 trading days)
    3. Monthly (Next 20 trading days)
    
    Based SOLELY on the sentiment of the text (Hawkish/Dovish), provide your best numerical estimate. 
    Do NOT use any historical knowledge of the actual outcome.
    
    OUTPUT FORMAT:
    Return a valid JSON object with keys: "Daily", "Weekly", "Monthly". Values must be floats (representing percentage).
    Example: {{"Daily": 0.15, "Weekly": -0.5, "Monthly": 1.2}}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a financial forecasting assistant. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # 零温，确保确定性
            response_format={ "type": "json_object" }
        )
        content = response.choices[0].message.content
        # 简单的清洗，防止 ```json ``` 包裹
        content = content.replace("```json", "").replace("```", "").strip()
        res_json = json.loads(content)
        return res_json
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

# ==========================================
# 3. 回归与制表函数
# ==========================================
def run_regression_and_table(df):
    print("Running Placebo Regressions...")
    
    models = {}
    
    # 对每个期限运行回归：Real ~ Predicted
    for label in HORIZONS.keys():
        # 去除空值（因为最后几个样本可能没有 Monthly 数据）
        sub_df = df.dropna(subset=[f'Real_{label}', f'Pred_{label}'])
        
        if len(sub_df) < 10:
            print(f"Not enough data for {label} regression.")
            continue
            
        y = sub_df[f'Real_{label}']
        X = sm.add_constant(sub_df[f'Pred_{label}'])
        
        # Newey-West 调整 (HAC)
        try:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
            models[label] = model
        except Exception as e:
            print(f"Regression failed for {label}: {e}")

    # 生成 LaTeX 表格
    if models:
        generate_latex(models, len(df))
    else:
        print("No models were successfully fitted.")

def format_coef(model, var_name):
    if var_name not in model.params: return "", ""
    coef = model.params[var_name]
    t_stat = model.tvalues[var_name]
    
    # 显著性星号
    stars = ""
    if abs(t_stat) > 2.58: stars = "***"
    elif abs(t_stat) > 1.96: stars = "**"
    elif abs(t_stat) > 1.65: stars = "*"
    
    return f"{coef:.3f}{stars}", f"({t_stat:.2f})"

def generate_latex(models, n_obs):
    print(f"Generating LaTeX table to {OUTPUT_TEX}...")
    
    # 确保所有 key 都在，防止报错
    def get_res(label, var):
        if label in models:
            return format_coef(models[label], var)
        return "-", "-"
    
    def get_r2(label):
        if label in models:
            return f"{models[label].rsquared_adj:.3f}"
        return "-"

    daily_c, daily_t = get_res('Daily', 'Pred_Daily')
    weekly_c, weekly_t = get_res('Weekly', 'Pred_Weekly')
    monthly_c, monthly_t = get_res('Monthly', 'Pred_Monthly')
    
    const_d, const_dt = get_res('Daily', 'const')
    const_w, const_wt = get_res('Weekly', 'const')
    const_m, const_mt = get_res('Monthly', 'const')
    
    latex = r"""
\begin{table}[h!]
\centering
\caption{Placebo Test: Predictive Power of LLM on Future Returns}
\label{tab:placebo_test}
\begin{threeparttable}
\begin{tabular}{lccc}
\toprule
 & \multicolumn{3}{c}{Dependent Variable: Realized S\&P 500 Returns (\%)} \\
\cmidrule(lr){2-4}
 & (1) & (2) & (3) \\
Variable & Daily ($t+1$) & Weekly ($t+5$) & Monthly ($t+20$) \\
\midrule
LLM Predicted Return & """ + f"{daily_c} & {weekly_c} & {monthly_c} \\\\" + r"""
 & """ + f"{daily_t} & {weekly_t} & {monthly_t} \\\\" + r"""
\addlinespace
Constant & """ + f"{const_d} & {const_w} & {const_m} \\\\" + r"""
 & """ + f"{const_dt} & {const_wt} & {const_mt} \\\\" + r"""
\midrule
Observations & """ + f"{n_obs} & {n_obs} & {n_obs} \\\\" + r"""
Adj. $R^2$ & """ + f"{get_r2('Daily')} & {get_r2('Weekly')} & {get_r2('Monthly')} \\\\" + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[para,flushleft]
  \item Note: This table reports the results of a placebo test to check for look-ahead bias. We ask the LLM (DeepSeek-Chat) to explicitly predict the S\&P 500 returns (in percentage) for the next 1, 5, and 20 trading days based solely on the FOMC statement text. We then regress the realized returns on these LLM-predicted returns. A significant positive coefficient would indicate potential look-ahead bias (i.e., the model "knows" the future market reaction). t-statistics (in parentheses) are computed using Newey-West standard errors.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    with open(OUTPUT_TEX, 'w', encoding='utf-8') as f:
        f.write(latex)
    print("Table generated successfully.")

# ==========================================
# 主程序
# ==========================================
def main():
    # 1. 准备数据
    try:
        df = load_and_align_data()
        print(f"Data ready. N={len(df)}")
    except Exception as e:
        print(f"Critical Error: {e}")
        return
    
    # 2. 检查是否有缓存
    if os.path.exists(OUTPUT_CSV):
        print("Loading predictions from cache...")
        df_final = pd.read_csv(OUTPUT_CSV)
    else:
        # 3. 调用 API 进行预测
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        preds_daily = []
        preds_weekly = []
        preds_monthly = []
        
        print("Querying LLM for predictions...")
        for txt in tqdm(df['text']):
            res = get_llm_forecast(client, txt)
            if res:
                preds_daily.append(res.get('Daily', np.nan))
                preds_weekly.append(res.get('Weekly', np.nan))
                preds_monthly.append(res.get('Monthly', np.nan))
            else:
                preds_daily.append(np.nan)
                preds_weekly.append(np.nan)
                preds_monthly.append(np.nan)
        
        df['Pred_Daily'] = preds_daily
        df['Pred_Weekly'] = preds_weekly
        df['Pred_Monthly'] = preds_monthly
        
        df_final = df.dropna(subset=['Pred_Daily']) # 只要有预测就保存
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"Predictions saved to {OUTPUT_CSV}")

    # 4. 运行回归并生成 LaTeX
    run_regression_and_table(df_final)

if __name__ == "__main__":
    main()