import pandas as pd
import numpy as np
import os
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock

# ==========================================
# 0. 顶级配置
# ==========================================

EMBEDDING_CONFIG = {
    "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", 
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "text-embedding-v4"
}

LLM_PROVIDERS = {
    "qwen3-235b": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-235b-a22b-instruct-2507" 
    }
}

CURRENT_PROVIDER = "qwen3-235b"
CURRENT_CONFIG = LLM_PROVIDERS[CURRENT_PROVIDER]

# [核心修改 1] 提高温度到 1.5。
# 标准是 0.7-1.0。设为 1.5 可以强行拉平分布，解决 Top1=1.00 的问题。
# 我们需要的是模型"潜意识"里的犹豫，高温能让潜意识浮现。
TEMPERATURE_FOR_DISTRIBUTION = 1.0 
TOP_K = 5
INPUT_FILE = "step1_fomc_statements_clean.csv"

OUTPUT_DIR = "data"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"llm_fomc_ESD_Final_{CURRENT_CONFIG['model']}.csv")

# ==========================================
# 1. 客户端与缓存
# ==========================================

print(f"Chat Model: {CURRENT_CONFIG['model']} | Temp: {TEMPERATURE_FOR_DISTRIBUTION}")
chat_client = OpenAI(api_key=CURRENT_CONFIG["api_key"], base_url=CURRENT_CONFIG["base_url"])
embed_client = OpenAI(api_key=EMBEDDING_CONFIG["api_key"], base_url=EMBEDDING_CONFIG["base_url"])

token_cache = {}
cache_lock = Lock()

# [核心修改 2] Token 还原映射表
# Qwen 的 Tokenizer 经常把长词切断。为了保证 Embedding 的语义正确，
# 我们必须手动修复常见的首字母 Token。
TOKEN_MAPPING = {
    "H": "Hawkish",
    "D": "Dovish",
    "P": "Positive", # 有时也可能是 Pessimistic，视上下文，但在金融语境 Positive 更多
    "N": "Neutral",
    "S": "Stable",
    "U": "Uncertain",
    "Unc": "Uncertain",
    "Opt": "Optimistic",
    "Pes": "Pessimistic",
    "Rec": "Recessionary",
    "Inf": "Inflationary",
    "C": "Consistent" # 这是一个无意义词，如果出现说明 Prompt 没拦住，但在高熵计算中它会作为噪音处理
}

# ==========================================
# 2. 核心逻辑
# ==========================================

def get_token_embeddings(tokens):
    """获取向量，包含自动映射修复"""
    global token_cache
    
    # 1. 预处理：映射修复
    # 如果 token 在映射表中，替换为全词；否则保持原样
    cleaned_tokens = [TOKEN_MAPPING.get(t, t) for t in tokens]
    
    embeddings_map = {}
    missing_tokens = []
    missing_indices = []

    with cache_lock:
        for idx, token in enumerate(cleaned_tokens):
            if token in token_cache:
                embeddings_map[idx] = token_cache[token]
            else:
                missing_tokens.append(token)
                missing_indices.append(idx)
    
    if missing_tokens:
        try:
            # 这里的 missing_tokens 已经是修复过的全词（如 "Hawkish"）
            # clean inputs to avoid empty string errors
            valid_inputs = [t if t.strip() else "Neutral" for t in missing_tokens]
            
            resp = embed_client.embeddings.create(
                input=valid_inputs,
                model=EMBEDDING_CONFIG["model"]
            )
            new_embeddings = [d.embedding for d in resp.data]
            
            with cache_lock:
                for i, emb in enumerate(new_embeddings):
                    token_cache[missing_tokens[i]] = emb
                    embeddings_map[missing_indices[i]] = emb
        except Exception as e:
            print(f"Embed Error: {e}")
            return None

    result_matrix = []
    for i in range(len(tokens)):
        if i in embeddings_map:
            result_matrix.append(embeddings_map[i])
        else:
            return None
    return np.array(result_matrix)
# ==========================================
# 顶级配置修改
# ==========================================


# 2. Prompt 策略变更：使用完形填空（Cloze Test）范式
# 这种方式能捕捉到更细微的语气差异，而不仅是硬分类
def calculate_esd_single_step(row_tuple):
    index, row = row_tuple
    text = row['text']
    
    # [关键修改]：不要让它选词，让它"总结语气"。
    # 我们构建一个让模型必须输出一个形容词的语境。
    # 相比于 "Answer:", 使用 "The tone is" 可以诱导更自然的分布。
    prompt = f"""
    [Task]: Analyze the economic outlook described in the text below.
    [Text]: "{text[:3000]}"
    
    [Instruction]: Complete the following sentence with a SINGLE, precise adjective that best captures the implied market sentiment (e.g., Hawkish, Dovish, Uncertain, Optimistic, Cautious, Stable, etc.).
    
    The economic outlook implied by this text is strictly described as:"""
    
    try:
        # Step 1: 获取 Logprobs
        response = chat_client.chat.completions.create(
            model=CURRENT_CONFIG["model"],
            messages=[
                # system prompt 设为空或极简，避免干扰
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE_FOR_DISTRIBUTION, 
            max_tokens=1, 
            logprobs=True,      
            top_logprobs=TOP_K  # 扩大采样范围到10，捕捉长尾风险
        )
        
        if not response.choices or not response.choices[0].logprobs:
            return index, None

        top_data = response.choices[0].logprobs.content[0].top_logprobs
        
        # 清洗与过滤
        # 很多时候 Top1 是空格 " "，或者是无关词，需要极其小心的清洗
        valid_items = []
        raw_probs = []
        
        for item in top_data:
            token_str = item.token.strip()
            # 过滤掉非字母的 token，过滤掉过短的无意义 token
            if len(token_str) > 2 and token_str.isalpha():
                valid_items.append(token_str)
                raw_probs.append(np.exp(item.logprob))
        
        if not valid_items:
            return index, None
            
        # 归一化 (Renormalize)
        # 我们只关心这几个形容词之间的相对概率分布
        raw_probs = np.array(raw_probs)
        norm_probs = raw_probs / np.sum(raw_probs)
        
        # Step 2: 获取 Embeddings (你需要对 valid_items 获取 embedding)
        # 注意：这里直接拿 token_str 去 embed，不需要 token_mapping 了，
        # 因为我们允许模型输出任何形容词，这才是真正的"开放语义空间"。
        
        # 为了节省 API，这里应该有一个本地缓存逻辑 (略)
        try:
            emb_resp = embed_client.embeddings.create(
                input=valid_items,
                model=EMBEDDING_CONFIG["model"]
            )
            vectors = np.array([d.embedding for d in emb_resp.data])
        except:
            return index, None

        # Step 3: 计算语义离散度 (Eq 1 in Paper)
        # Mean Vector
        z_bar = np.average(vectors, axis=0, weights=norm_probs)
        
        # Dispersion: sum( p_i * ||z_i - z_bar||^2 )
        diffs = vectors - z_bar
        squared_dists = np.sum(diffs**2, axis=1)
        semantic_dispersion = np.average(squared_dists, weights=norm_probs)
        
        # 香农熵 (作为对比)
        shannon = -np.sum(norm_probs * np.log(norm_probs + 1e-9))
        
        return index, {
            'Semantic_Dispersion': semantic_dispersion, # 这才是你的 D_t
            'Shannon_Entropy': shannon,
            'Top1_Token': valid_items[0],
            'Top1_Prob': norm_probs[0],
            'Top_Tokens_Str': "|".join(valid_items[:5])
        }

    except Exception as e:
        print(f"Err {index}: {e}")
        return index, None

# ==========================================
# 3. 主程序
# ==========================================

def main():
    if not os.path.exists(INPUT_FILE):
        print("无输入文件")
        return

    df = pd.read_csv(INPUT_FILE)
    columns = ['date', 'text', 'Semantic_Dispersion', 'Shannon_Entropy', 'Top1_Token', 'Top1_Prob', 'Top5_Tokens', 'Top5_Probs']

    # 覆盖写模式 (建议重新跑，因为之前的数据是坏的)
    if os.path.exists(OUTPUT_FILE):
        print(f"警告: 文件 {OUTPUT_FILE} 已存在。建议删除后重跑。这里演示追加模式。")
        try:
            df_done = pd.read_csv(OUTPUT_FILE)
            done_dates = set(df_done['date'].astype(str).tolist())
        except:
            done_dates = set()
            pd.DataFrame(columns=columns).to_csv(OUTPUT_FILE, index=False)
    else:
        done_dates = set()
        pd.DataFrame(columns=columns).to_csv(OUTPUT_FILE, index=False)

    tasks = []
    for index, row in df.iterrows():
        if str(row['date']) not in done_dates:
            tasks.append((index, row))
    
    print(f"处理任务数: {len(tasks)}")
    
    MAX_WORKERS = 5
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {executor.submit(calculate_esd_single_step, t): t[1] for t in tasks}
        
        for future in tqdm(as_completed(future_to_row), total=len(tasks)):
            idx, res = future.result()
            row = future_to_row[future]
            
            if res:
                save_data = {'date': row['date'], 'text': row['text'], **res}
                pd.DataFrame([save_data], columns=columns).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

if __name__ == "__main__":
    main()