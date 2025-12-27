import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取两个结果文件
df_ds = pd.read_csv("data/llm_fomc_dispersion_results_deepseek-chat_CMC.csv")
df_qw = pd.read_csv("data/llm_fomc_dispersion_results_qwen-flash_CMC.csv")

# 2. 合并数据
df_ds['date'] = pd.to_datetime(df_ds['date'])
df_qw['date'] = pd.to_datetime(df_qw['date'])

merged = pd.merge(df_ds[['date', 'Semantic_Dispersion']], 
                  df_qw[['date', 'Semantic_Dispersion']], 
                  on='date', suffixes=('_DS', '_QW'))

# 3. 绘图 1: 时间序列对比
plt.figure(figsize=(14, 6))
plt.plot(merged['date'], merged['Semantic_Dispersion_DS'], label='DeepSeek-V3', alpha=0.7)
plt.plot(merged['date'], merged['Semantic_Dispersion_QW'], label='Qwen-Flash', alpha=0.7)
plt.title("Time Series Divergence: DeepSeek vs Qwen")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 4. 绘图 2: 散点图 (检查负相关来源)
plt.figure(figsize=(8, 8))
sns.regplot(data=merged, x='Semantic_Dispersion_QW', y='Semantic_Dispersion_DS')
plt.title(f"Correlation: {merged['Semantic_Dispersion_DS'].corr(merged['Semantic_Dispersion_QW']):.3f}")
plt.xlabel("Qwen Dispersion")
plt.ylabel("DeepSeek Dispersion")
plt.show()

# 5. [关键] 检查极值点的原始文本
# 找出 DeepSeek 极低但 Qwen 极高的日子（可能是危机时刻 DeepSeek 收敛了）
merged['diff'] = merged['Semantic_Dispersion_QW'] - merged['Semantic_Dispersion_DS']
top_diff_dates = merged.nlargest(5, 'diff')['date']

print("检查这些日期，DeepSeek 是否输出了重复的安全回复？")
print(top_diff_dates)