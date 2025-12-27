import pandas as pd
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- A. 嵌入模型配置 (本地路径) ---
LOCAL_EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-8B"



# --- B. 聊天模型配置池 (在此处添加不同厂商) ---
LLM_PROVIDERS = {
    "qwen-flash": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", # 阿里 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-flash" # qwen-flash
    },
    "deepseek-v3.2": {
        "api_key": "sk-b0bc5ee8b28e4d5bb9116a3dcfe88f3e", # 阿里 Key
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
    },
    "qwen-max": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", # 阿里 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max"
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
        "qwen3-0.6b": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", # 阿里 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-0.6b"
    },
        "qwen3-1.7b": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf", # 阿里 Key
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-1.7b"
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
CURRENT_PROVIDER = "qwen3-1.7b"  # 可选: "qwen", "deepseek", "chatgpt", "gemini"
CURRENT_CONFIG = LLM_PROVIDERS[CURRENT_PROVIDER]

# --- D. 其他参数 ---
K_SAMPLES = 50    
TEMPERATURE = 1.0
TOP_P = 0.95
INPUT_FILE = "step1_fomc_statements_clean.csv"

# ==========================================
# [新增] 本地 Embedding 包装类
# ==========================================
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class LocalQwenEmbedding:
    def __init__(self, model_id="Qwen/Qwen3-Embedding-8B"):
        print(f"[Local AI] 正在加载本地模型: {model_id} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # 针对 5060 Ti 16G 的配置
        # 如果遇到 OOM (显存不足)，请取消注释 load_in_8bit=True (需要 pip install bitsandbytes)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16, 
            # load_in_8bit=True, # <--- 如果爆显存，请打开这一行
        )
        self.model.eval()
        print(f"[Local AI] 模型加载完成，设备: {self.device}")

    def create(self, input, model=None):
        if isinstance(input, str):
            input = [input]
        
        # Qwen-Embedding 支持长文本，但为了安全设为 8192
        batch_dict = self.tokenizer(
            input, 
            max_length=8192, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        data = []
        cpu_embeddings = embeddings.cpu().numpy().tolist()
        for i, emb in enumerate(cpu_embeddings):
            data.append(SimpleNamespace(embedding=emb, index=i))
            
        return SimpleNamespace(data=data)

    def last_token_pool(self, last_hidden_states, attention_mask):
        # Qwen3-Embedding 推荐使用 Last Token Pooling 或者 Mean Pooling
        # 这里使用跟官方示例一致的 pooling 方式 (通常 transform 库会自动处理，这里手动写死逻辑以防万一)
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]



# 自动生成输出文件名
OUTPUT_DIR = "data"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"llm_fomc_dispersion_results_{CURRENT_CONFIG['model']}_CMC.csv")

# ==========================================
# 1. 初始化客户端
# ==========================================

print(f"当前使用 Chat 模型: [API] {CURRENT_CONFIG['model']}")
print(f"当前使用 Embedding 模型: [Local GPU] {LOCAL_EMBED_MODEL_ID}")

# 1. Chat 客户端 (保持远程 API)
chat_client = OpenAI(
    api_key=CURRENT_CONFIG["api_key"],
    base_url=CURRENT_CONFIG["base_url"]
)

# 2. Embedding 客户端 (替换为本地类)
embed_client = LocalQwenEmbedding(model_id=LOCAL_EMBED_MODEL_ID)

# ==========================================
# 2. 核心逻辑函数
# ==========================================
def generate_narrative(text, index):
    """
    生成单个叙事解读 - 受控续写模式
    """
    try:
        prompt = f"""
        Read the following FOMC statement snippet carefully:
        "{text[:2000]}..."

        Task: Based strictly on this text, complete the following sentence regarding the future policy path. Be specific and decisive.
        
        The Committee's language implies that the most likely next policy action is to
        """
        
        response = chat_client.chat.completions.create(
            model=CURRENT_CONFIG["model"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={"enable_thinking": False},
            temperature=TEMPERATURE, 
            top_p=TOP_P,      
            max_tokens=30    
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"\n[Error] Index {index} | Reason: {e}")
        return None

def process_single_date(row_tuple):
    """处理单行数据：Chat生成 -> Local Embed向量化 -> 计算方差"""
    index, row = row_tuple
    text = row['text']
    date = row['date']
    
    # 1. 蒙特卡洛生成 (IO 密集型，可以使用多线程并发请求 API)
    narratives = []
    # Chat生成依然可以用多线程，因为是请求远程 API
    with ThreadPoolExecutor(max_workers=1) as executor: 
        futures = [executor.submit(generate_narrative, text, index) for _ in range(K_SAMPLES)]
        for future in as_completed(futures):
            res = future.result()
            if res:
                narratives.append(res)
    
    if len(narratives) < K_SAMPLES * 0.8:
        print(f"[{date}] 生成失败，有效样本仅 {len(narratives)}/{K_SAMPLES}")
        return index, None, None

    # 2. 本地 Embedding (计算密集型)
    # 对于本地 GPU，直接把所有文本一次性送进去效率最高 (Batch Processing)
    try:
        # 直接调用本地模型的 create，传入整个列表
        # 50条短文本对于 5060Ti 来说是小菜一碟，不需要分 Batch
        emb_resp = embed_client.create(input=narratives)
        
        embeddings = np.array([d.embedding for d in emb_resp.data])
        
        # 3. 计算离散度
        centroid = np.mean(embeddings, axis=0)
        squared_dists = np.sum((embeddings - centroid)**2, axis=1)
        dispersion = np.mean(squared_dists)
        
        return index, dispersion, len(narratives)
        
    except Exception as e:
        print(f"Calc Error at {date}: {e}")
        import traceback
        traceback.print_exc()
        return index, None, None

# ==========================================
# 3. 主执行流
# ==========================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到 {INPUT_FILE}。")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"载入数据: {len(df)} 条 FOMC 声明。")
    
    # 断点续传逻辑
    if os.path.exists(OUTPUT_FILE):
        print(f"检测到断点文件: {OUTPUT_FILE}，追加模式...")
        try:
            df_done = pd.read_csv(OUTPUT_FILE)
            done_dates = set(df_done['date'].astype(str).tolist())
        except:
            done_dates = set()
            pd.DataFrame(columns=['date', 'text', 'Semantic_Dispersion']).to_csv(OUTPUT_FILE, index=False)
    else:
        done_dates = set()
        pd.DataFrame(columns=['date', 'text', 'Semantic_Dispersion']).to_csv(OUTPUT_FILE, index=False)

    tasks = []
    for index, row in df.iterrows():
        d_str = str(row['date'])
        if d_str not in done_dates:
            tasks.append((index, row))
            
    print(f"总任务: {len(df)} | 已完成: {len(done_dates)} | 剩余: {len(tasks)}")
    
    if len(tasks) == 0:
        print("所有任务已完成！")
        return

    print("开始正式计算...")
    
    # [关键修改] MAX_WORKERS 设为 1
    # 因为 process_single_date 内部会调用本地 GPU。
    # 如果并行处理多个日期，会导致 GPU 显存竞争和频繁的上下文切换，效率反而变低甚至 OOM。
    # 实际上，单个任务内部我们在生成文本时已经用了多线程(IO并发)，在Embedding时用了GPU批处理，这已经是最高效的了。
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_row = {executor.submit(process_single_date, t): t[1] for t in tasks}
        
        for future in tqdm(as_completed(future_to_row), total=len(tasks), desc="Processing"):
            idx, disp, count = future.result()
            row = future_to_row[future]
            
            if disp is not None:
                res_df = pd.DataFrame([{
                    'date': row['date'],
                    'text': row['text'], 
                    'Semantic_Dispersion': disp
                }])
                res_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

    print(f"\n全部完成！结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()