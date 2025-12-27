import pandas as pd
import numpy as np
import os
import time
from openai import OpenAI
from sklearn.metrics.pairwise import euclidean_distances
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ==========================================
# 0. 顶级配置 (Configuration)
# ==========================================

# 替换为你的阿里 DashScope API Key
DASHSCOPE_API_KEY = "sk-18a9d2c5c0a84464a7caea5584fce8bf" 

# Qwen-Max 的 OpenAI 兼容端点
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模型配置
CHAT_MODEL = "deepseek-v3.2"           
EMBEDDING_MODEL = "text-embedding-v4" 

# 蒙特卡洛参数
K_SAMPLES = 50    # 保持20不变
TEMPERATURE = 1.2 

# 文件路径
INPUT_FILE = "step1_fomc_statements_clean.csv"
OUTPUT_FILE = "step2_fomc_dispersion_results_1.csv"

# 初始化客户端
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=BASE_URL
)

# ==========================================
# 1. 核心逻辑函数
# ==========================================

def generate_narrative(text, index):
    """生成单个叙事解读"""
    try:
        # Prompt 保持不变
        prompt = f"""
        You are a sophisticated financial economist. 
        Read the following FOMC statement snippet carefully.
        
        Predict the tone of the NEXT monetary policy action (Hawkish/Dovish) and the economic outlook.
        Keep your response concise (under 50 words).
        
        FOMC Statement:
        {text[:2000]}...
        """
        
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE, 
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return None

def process_single_date(row_tuple):
    """
    处理单行数据的完整流程：
    生成 K 个解读 -> 分批 Embedding -> 计算离散度
    """
    index, row = row_tuple
    text = row['text']
    date = row['date']
    
    # 1. 蒙特卡洛生成 (并发生成)
    narratives = []
    
    # 内层并发数建议不要太高，防止QPM瞬间爆炸
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(generate_narrative, text, index) for _ in range(K_SAMPLES)]
        for future in as_completed(futures):
            res = future.result()
            if res:
                narratives.append(res)
    
    # 过滤无效生成
    if len(narratives) < K_SAMPLES * 0.8:
        print(f"[{date}] 生成失败，有效样本仅 {len(narratives)}/{K_SAMPLES}")
        return index, None, None

    # 2. 批量 Embedding (关键修正点：分批处理)
    embeddings_list = []
    BATCH_SIZE = 5 # 阿里限制是10，我们设5更保险
    
    try:
        # 循环分批调用 Embedding
        for i in range(0, len(narratives), BATCH_SIZE):
            batch_texts = narratives[i : i + BATCH_SIZE]
            
            # 简单的重试机制
            retry_count = 0
            while retry_count < 3:
                try:
                    emb_resp = client.embeddings.create(
                        input=batch_texts,
                        model=EMBEDDING_MODEL
                    )
                    # 收集结果
                    batch_embs = [d.embedding for d in emb_resp.data]
                    embeddings_list.extend(batch_embs)
                    break # 成功则跳出重试循环
                except Exception as e:
                    retry_count += 1
                    time.sleep(1) # 歇一秒再试
                    if retry_count == 3:
                        print(f"[{date}] Embedding Batch Failed: {e}")
        
        if not embeddings_list:
            return index, None, None

        embeddings = np.array(embeddings_list)
        
        # 3. 计算语义离散度 (Semantic Dispersion)
        centroid = np.mean(embeddings, axis=0)
        squared_dists = np.sum((embeddings - centroid)**2, axis=1)
        dispersion = np.mean(squared_dists)
        
        return index, dispersion, len(narratives)
        
    except Exception as e:
        print(f"Calc Error at {date}: {e}")
        return index, None, None

# ==========================================
# 2. 主执行流
# ==========================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到 {INPUT_FILE}。请先运行 Step 1。")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"载入数据: {len(df)} 条 FOMC 声明。")
    print(f"模型: {CHAT_MODEL} | 采样数 K: {K_SAMPLES} | Embedding Batch Size: 5 (Fixed)")
    print("开始计算语义离散度... (请耐心等待)")

    results_map = {}
    
    # 外层并发数
    MAX_WORKERS = 3 
    
    rows_to_process = list(df.iterrows())
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_date = {executor.submit(process_single_date, row): row[1]['date'] for row in rows_to_process}
        
        for future in tqdm(as_completed(future_to_date), total=len(df), desc="Processing"):
            idx, disp, count = future.result()
            if disp is not None:
                results_map[idx] = disp

    df['Semantic_Dispersion'] = df.index.map(results_map)
    
    missing = df['Semantic_Dispersion'].isna().sum()
    print(f"\n计算完成。成功: {len(df)-missing}, 失败: {missing}")
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"最终结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()