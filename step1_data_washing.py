import pandas as pd
import re
import os
import ftfy # 必须先 pip install ftfy

# ==========================================
# 配置路径
# ==========================================
INPUT_PATH = r'data/fed-statement-scraping/communications.csv'
OUTPUT_PATH = 'step1_fomc_statements_clean.csv'

def nuclear_cleaning(text):
    """
    终极清洗函数：
    1. 使用 ftfy 修复 Mojibake (编码错乱)
    2. 暴力正则替换所有非 ASCII 的连字符
    """
    if not isinstance(text, str):
        return ""
    
    # --- 第1层：使用 ftfy 自动修复 ---
    # ftfy 会自动识别 "â€‘" 并将其修回 "-" (U+2011) 或正确符号
    text = ftfy.fix_text(text)
    
    # --- 第2层：暴力正则 (Regex) ---
    # 你的顽疾是：数字中间夹着乱码，比如 3â€‘3/4
    # 下面的正则意思是：如果发现 数字 + (非ASCII字符) + 数字，就把中间那坨替换成 '-'
    # \d 表示数字，[^\x00-\x7F]+ 表示连续的非ASCII字符
    text = re.sub(r'(\d)[^\x00-\x7F]+(\d)', r'\1-\2', text)
    
    # --- 第3层：兜底替换 (针对特定的残留) ---
    # 把剩下的所有常见非 ASCII 连字符强制转为标准 ASCII 减号
    # 包括：Non-breaking hyphen, En dash, Em dash, Minus sign...
    text = text.replace('\u2011', '-') 
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '-')
    text = text.replace('\u2212', '-')
    
    # 再次清理可能残留的 Mojibake (以防 ftfy 漏掉)
    text = text.replace('â€‘', '-')
    text = text.replace('â€“', '-')
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€', '"')
    
    # 清理多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    print(f"Loading raw data from: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Path not found: {INPUT_PATH}")
        return

    try:
        # 使用 Latin-1 读取以保留原始字节流，防止 pandas 自动乱猜
        # 或者是 utf-8，视乎原始文件。如果 github 下载通常是 utf-8。
        # 这里我们先按标准 utf-8 读取，如果报错再换
        df = pd.read_csv(INPUT_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 failed, trying Latin-1...")
        df = pd.read_csv(INPUT_PATH, encoding='latin1')

    print(f"Raw shape: {df.shape}")
    df.columns = [c.capitalize() for c in df.columns]
    
    # 1. 过滤 Statement
    if 'Type' in df.columns:
        df = df[df['Type'].str.contains('Statement', case=False, na=False)].copy()
        print(f"Statements count: {len(df)}")
    
    # 2. 终极清洗
    if 'Text' in df.columns:
        print("Applying nuclear cleaning (ftfy + regex)...")
        df['Text'] = df['Text'].apply(nuclear_cleaning)
        
        # 检查你的顽疾是否还在
        # 找任何包含 'â' 的行
        bad_rows = df[df['Text'].str.contains('â', na=False)]
        if not bad_rows.empty:
            print(f"\n[Warning] 依然有 {len(bad_rows)} 行包含 'â'。请人工检查 step1_debug.csv")
            bad_rows.to_csv('step1_debug.csv', index=False)
        else:
            print("\n[Success] 'â' 字符已完全根除！")
            
        # 再次检查 "3-3/4" 这种模式
        sample_check = df[df['Text'].str.contains(r'3-3/4', na=False)]
        if not sample_check.empty:
            print(f"[Check] 成功发现修复后的 '3-3/4' 格式，共 {len(sample_check)} 处。")

        # 剔除短文本
        df = df[df['Text'].str.len() > 200]
    
    # 3. 日期处理
    date_col = 'Date' if 'Date' in df.columns else 'Release date'
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
        df = df.rename(columns={date_col: 'date', 'Text': 'text', 'Type': 'type'})
        df = df[['date', 'text']]
    
    # 4. 保存
    print("-" * 30)
    print(f"Final Cleaned Shape: {df.shape}")
    print("-" * 30)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()