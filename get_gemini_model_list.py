import pandas as pd
from openai import OpenAI
import time

# ================= 配置区域 =================
# 请在此处填入你的 API Key
# 注意：Qwen使用的是阿里云DashScope的Key，Gemini使用的是Google AI Studio的Key
CONFIGS = {
    "OpenAI": {
        "api_key": "YOUR_OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1"
    },
    "DeepSeek": {
        "api_key": "sk-7f5b8b112a444fbdbf479685ec5cfd45",
        "base_url": "https://api.deepseek.com"
    },
    "Gemini": {
        "api_key": "AIzaSyA0NxNiZq1aIhPWQ07LNtFt9nOUhKspDtk", # 例如: AIzaSy...
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
    },
    "Qwen": {
        "api_key": "sk-18a9d2c5c0a84464a7caea5584fce8bf",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
}

OUTPUT_FILE = "llm_models_list.xlsx"
# ===========================================

def get_models_for_provider(provider_name, config):
    """
    连接指定厂商的API并获取模型列表
    """
    print(f"正在获取 {provider_name} 的模型列表...")
    
    # 检查是否有 Key，如果没有则跳过
    if "YOUR_" in config["api_key"] or not config["api_key"]:
        print(f"  [跳过] 未配置 {provider_name} 的 API Key")
        return pd.DataFrame({"Error": ["API Key not configured"]})

    try:
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        
        # 获取模型列表
        models_response = client.models.list()
        
        # 解析数据
        model_data = []
        for model in models_response:
            # 不同厂商返回的字段可能略有不同，这里取最通用的字段
            model_info = {
                "id": model.id,
                "created": getattr(model, "created", None),
                "owned_by": getattr(model, "owned_by", None),
                "object": getattr(model, "object", "model")
            }
            model_data.append(model_info)
            
        # 转换为 DataFrame 并按 ID 排序
        df = pd.DataFrame(model_data)
        if not df.empty and "id" in df.columns:
            df = df.sort_values(by="id")
            
        print(f"  [成功] 获取到 {len(df)} 个模型")
        return df

    except Exception as e:
        print(f"  [失败] 获取 {provider_name} 出错: {e}")
        return pd.DataFrame({"Error": [str(e)]})

def main():
    # 用于存储所有数据的字典
    all_sheets = {}

    # 遍历配置获取数据
    for provider, config in CONFIGS.items():
        df = get_models_for_provider(provider, config)
        all_sheets[provider] = df
        time.sleep(1) # 稍微暂停，防止请求过快

    # 写入 Excel
    print(f"\n正在写入文件: {OUTPUT_FILE} ...")
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            for sheet_name, df in all_sheets.items():
                # 如果 DataFrame 为空，创建一个空的以防止报错
                if df.empty:
                    df = pd.DataFrame({"Status": ["No Data Fetched"]})
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 尝试自动调整列宽 (简单的视觉优化)
                worksheet = writer.sheets[sheet_name]
                for column_cells in worksheet.columns:
                    length = max(len(str(cell.value)) for cell in column_cells)
                    if length > 50: length = 50 # 限制最大宽度
                    worksheet.column_dimensions[column_cells[0].column_letter].width = length + 2
                    
        print("完成！请查看生成的 Excel 文件。")
        
    except Exception as e:
        print(f"写入 Excel 时发生错误: {e}")
        print("请确保已安装 openpyxl: pip install openpyxl")

if __name__ == "__main__":
    main()