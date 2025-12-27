import pandas as pd
import numpy as np
import os
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ==========================================
# 0. 顶级配置 (Configuration)
# ==========================================

# --- A. 嵌入模型配置 (固定使用 Qwen) ---
EMBEDDING_CONFIG = {
    "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", # 你的阿里 DashScope Key
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "text-embedding-v4"
}

# --- B. 聊天模型配置池 (在此处添加不同厂商) ---
LLM_PROVIDERS = {
    "qwen-flash": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", # 阿里 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-flash" # qwen-flash
    },
    "qwen-ds": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", # 阿里 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "deepseek-v3.2"
    },
    "qwen3-max": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", # 阿里 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-max"
    },
    "qwen3-30b": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", # 阿里 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-30b-a3b-instruct-2507"
    },
    "qwen3-14b": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", # 阿里 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-14b"
    },
    "qwen3-8b": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", # 阿里 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-8b"
    },
    "qwen3-4b": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", # 阿里 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-4b"
    },
    # "chatgpt": {
    #     "api_key": "sk-xxxxxxxxxxxx", # 填入 OpenAI Key
    #     "base_url": "https://api.openai.com/v1",
    #     "model": "gpt-4o-mini"
    # },
    # "gemini": {
    #     "api_key": "AIzaSyA0NxNiZq1aIhPWQ07LNtFt9nOUhKspDtk", # 填入 Google AI Studio Key
    #     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    #     "model": "gemini-3-pro-preview"
    # }
}
# --- C. 选择当前使用的模型 ---
CURRENT_PROVIDER = "qwen3-max"  # 可选: "qwen", "deepseek", "chatgpt", "gemini"
CURRENT_CONFIG = LLM_PROVIDERS[CURRENT_PROVIDER]

# --- D. 其他参数 ---
K_SAMPLES = 50    
TEMPERATURE = 1.2 
INPUT_FILE = "step1_fomc_statements_clean.csv"

# 自动生成输出文件名
OUTPUT_DIR = "data"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"llm_fomc_dispersion_results_{CURRENT_CONFIG['model']}.csv")

# ==========================================
# 1. 初始化客户端
# ==========================================

print(f"当前使用 Chat 模型: [{CURRENT_PROVIDER}] {CURRENT_CONFIG['model']}")
print(f"当前使用 Embedding 模型: [Aliyun] {EMBEDDING_CONFIG['model']}")

# 1. Chat 客户端
chat_client = OpenAI(
    api_key=CURRENT_CONFIG["api_key"],
    base_url=CURRENT_CONFIG["base_url"]
)

# 2. Embedding 客户端
embed_client = OpenAI(
    api_key=EMBEDDING_CONFIG["api_key"],
    base_url=EMBEDDING_CONFIG["base_url"]
)

# ==========================================
# 2. 核心逻辑函数
# ==========================================

def generate_narrative(text, index):
    """生成单个叙事解读"""
    try:
        prompt = f"""
        You are a sophisticated financial economist. 
        Read the following FOMC statement snippet carefully.
        
        Predict the tone of the NEXT monetary policy action (Hawkish/Dovish) and the economic outlook.
        Keep your response concise (under 50 words).
        
        FOMC Statement:
        {text[:2000]}...
        """
        
        response = chat_client.chat.completions.create(
            model=CURRENT_CONFIG["model"],
            messages=[
                {"role": "system", "content": "You are a financial expert."},
                {"role": "user", "content": prompt}
            ],
            extra_body={"enable_thinking": False},
            temperature=TEMPERATURE, 
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        # [修改点] 打印具体的错误信息！
        print(f"\n[Error] Index {index} | Model: {CURRENT_CONFIG['model']} | Reason: {e}")
        return None

def process_single_date(row_tuple):
    """处理单行数据：Chat生成 -> Embed向量化 -> 计算方差"""
    index, row = row_tuple
    text = row['text']
    date = row['date']
    
    # 1. 蒙特卡洛生成
    narratives = []
    # 这里设置内层并发
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(generate_narrative, text, index) for _ in range(K_SAMPLES)]
        for future in as_completed(futures):
            res = future.result()
            if res:
                narratives.append(res)
    
    if len(narratives) < K_SAMPLES * 0.8:
        print(f"[{date}] 生成失败，有效样本仅 {len(narratives)}/{K_SAMPLES}")
        return index, None, None

    # 2. 批量 Embedding
    embeddings_list = []
    BATCH_SIZE = 5 
    
    try:
        for i in range(0, len(narratives), BATCH_SIZE):
            batch_texts = narratives[i : i + BATCH_SIZE]
            retry_count = 0
            while retry_count < 3:
                try:
                    emb_resp = embed_client.embeddings.create(
                        input=batch_texts,
                        model=EMBEDDING_CONFIG["model"]
                    )
                    batch_embs = [d.embedding for d in emb_resp.data]
                    embeddings_list.extend(batch_embs)
                    break 
                except Exception as e:
                    retry_count += 1
                    time.sleep(1)
        
        if not embeddings_list:
            return index, None, None

        embeddings = np.array(embeddings_list)
        
        # 3. 计算离散度
        centroid = np.mean(embeddings, axis=0)
        squared_dists = np.sum((embeddings - centroid)**2, axis=1)
        dispersion = np.mean(squared_dists)
        
        return index, dispersion, len(narratives)
        
    except Exception as e:
        print(f"Calc Error at {date}: {e}")
        return index, None, None

# ==========================================
# [新增] 热身模块
# ==========================================
def warmup_worker(worker_id):
    """执行极简请求以建立连接池"""
    try:
        # 1. 简单的 Chat 请求
        chat_client.chat.completions.create(
            model=CURRENT_CONFIG["model"],
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1
        )
        # 2. 简单的 Embed 请求
        embed_client.embeddings.create(
            input="warmup",
            model=EMBEDDING_CONFIG["model"]
        )
        return True
    except Exception as e:
        print(f"Warmup worker {worker_id} warning: {e}")
        return False


# ==========================================
# 3. 主执行流
# ==========================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到 {INPUT_FILE}。请先运行 Step 1。")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"载入数据: {len(df)} 条 FOMC 声明。")
    
    # ----------------------------------------------------------------
    # [Modify] 断点续传逻辑
    # ----------------------------------------------------------------
    if os.path.exists(OUTPUT_FILE):
        print(f"检测到断点文件: {OUTPUT_FILE}，正在以追加模式运行...")
        try:
            df_done = pd.read_csv(OUTPUT_FILE)
            # 假设 CSV 包含 'date' 列且格式一致
            done_dates = set(df_done['date'].astype(str).tolist())
        except:
            done_dates = set()
            # 如果文件为空或损坏，重新初始化
            pd.DataFrame(columns=['date', 'text', 'Semantic_Dispersion']).to_csv(OUTPUT_FILE, index=False)
    else:
        done_dates = set()
        # 初始化文件头
        pd.DataFrame(columns=['date', 'text', 'Semantic_Dispersion']).to_csv(OUTPUT_FILE, index=False)

    # 过滤任务
    tasks = []
    for index, row in df.iterrows():
        d_str = str(row['date'])
        if d_str not in done_dates:
            tasks.append((index, row))
            
    print(f"总任务: {len(df)} | 已完成: {len(done_dates)} | 剩余: {len(tasks)}")
    
    if len(tasks) == 0:
        print("所有任务已完成！")
        return

    # ----------------------------------------------------------------
    # [Modify] 启动热身 (在正式跑之前)
    # ----------------------------------------------------------------
    MAX_WORKERS = 1

    # ----------------------------------------------------------------
    # 正式运行
    # ----------------------------------------------------------------
    print("开始正式计算...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {executor.submit(process_single_date, t): t[1] for t in tasks}
        
        for future in tqdm(as_completed(future_to_row), total=len(tasks), desc="Processing"):
            idx, disp, count = future.result()
            row = future_to_row[future]
            
            if disp is not None:
                # 实时追加写入 (Append Mode)
                res_df = pd.DataFrame([{
                    'date': row['date'],
                    'text': row['text'], 
                    'Semantic_Dispersion': disp
                }])
                res_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

    print(f"\n全部完成！结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
