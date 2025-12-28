
# Narrative Ambiguity, Cognitive Robustness, and Asset Pricing
# 叙事模糊性、认知稳健性与资产定价

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://github.com/qianphhhh-fin/llm-ambiguity-premium/blob/main/Narrative_Ambiguity_Paper.pdf)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

本仓库包含了论文 **"Narrative Ambiguity, Cognitive Robustness, and Asset Pricing"** 的官方实现代码与数据处理流程。

> **仓库地址**: [https://github.com/qianphhhh-fin/llm-ambiguity-premium](https://github.com/qianphhhh-fin/llm-ambiguity-premium)

## 📄 摘要 (Abstract)

即便经济基本面未发生实质性变化，金融市场常因央行沟通的模棱两可而剧烈波动。为了捕捉这种非结构化信息带来的认知摩擦，本研究引入**大语言模型（LLM）**作为代表性投资者的认知代理，利用 LLM 模拟投资者在面对模糊信息时构建多重未来情景的思维过程，并将情景间的语义分歧定义为**叙事模糊性（Narrative Ambiguity, $D_t$）**。

基于内生稳健控制理论，本文从理论上证明了叙事模糊性不仅仅是噪音，而是一种被定价的奈特不确定性（Knightian Uncertainty）。实证结果表明：
1. **负向风险溢价**：叙事模糊性因子在横截面上承载了显著的负向风险溢价。
2. **安全资产挤兑**：模糊性上升导致资金涌向无风险资产，压低实际无风险利率。
3. **资产分化**：传统避险资产（如黄金）因无法提供绝对确定性而遭受价值回撤，而长久期资产（如耐用消费品）则展现出对冲价值。

## 📂 项目结构 (Repository Structure)

本项目代码按照数据处理和实证分析的步骤进行组织：

### 1. 数据清洗与准备
- `step1_data_washing.py`: 对原始 FOMC 声明文本进行清洗（去除乱码、格式化）。
- `get_gemini_model_list.py`: 获取模型列表辅助工具。

### 2. 核心指标构建 ($D_t$)
- `step2_calculate_dispersion.py`: **[核心代码]** 调用 LLM (DeepSeek/Qwen) 进行蒙特卡洛文本续写，并计算语义离散度（Semantic Dispersion）。
- `step2_calculate_dispersion_logprob.py`: 基于 Logprobs 的替代计算方法。
- `step2_calculate_dispersion_controlled_MC.py`: 受控蒙特卡洛模拟版本。

### 3. 宏观机制检验
- `step3_vix_epu.py`: 检验 $D_t$ 与 VIX (市场恐慌) 及 EPU (经济政策不确定性) 的关系。
- `step3.1_monetary_surprises.py`: 控制货币政策意外冲击的影响。
- `step3.2_fog_index.py`: 对比文本可读性指标 (Fog Index)。
- `step3.3_uncertainty.py`: 对比实体经济不确定性 (JLN Index)。
- `step3.4_textual_sentiment.py`: 对比传统文本情绪指标 (Loughran-McDonald)。

### 4. 资产定价测试
- `step4_asset_pricing_test.py`: 49个行业组合的 Fama-MacBeth 回归测试。
- `step4_advanced_pricing_test.py`: 包含更多因子模型（FF5, HXZ5, Barillas-Shanken 等）的高级定价测试。

### 5. 稳健性与安慰剂检验
- `step6_multi_model_pricing.py`: 跨模型稳健性检验（测试不同 LLM 架构下的结果一致性）。
- `step7_placebo_test.py`: 安慰剂检验，排除前视偏差 (Look-ahead Bias)。
- `step8_mechanism_check_real_rate.py`: 检验叙事模糊性对实际利率（TIPS）的影响机制。
- `X_step*.py`: 其他辅助性或扩展性测试脚本。

## 🚀 快速开始 (Quick Start)

### 环境要求
推荐使用 Python 3.8+。请安装以下依赖：
```bash
pip install pandas numpy statsmodels matplotlib seaborn torch transformers openai textstat ftfy scipy pandas_datareader
```
*(注：部分脚本需要 `getfactormodels` 库或手动下载 Fama-French 因子数据)*

### API 配置
本项目依赖大模型 API 进行推理。请在 `step2_*.py` 等文件中配置您的 API Key：
```python
# 示例配置 (请在代码中替换为您自己的 Key)
CONFIGS = {
    "DeepSeek": {
        "api_key": "YOUR_DEEPSEEK_KEY",
        "base_url": "https://api.deepseek.com"
    },
    "Qwen": {
        "api_key": "YOUR_ALIYUN_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
}
```

### 复现步骤

1.  **数据清洗**:
    ```bash
    python step1_data_washing.py
    ```
2.  **计算叙事模糊性 ($D_t$)**:
    *这是最耗时的一步，需要调用 LLM API。*
    ```bash
    python step2_calculate_dispersion.py
    ```
3.  **运行资产定价回归**:
    ```bash
    python step4_advanced_pricing_test.py
    ```
4.  **生成图表与机制分析**:
    按需运行 `step3_*.py` 和 `step8_*.py` 系列脚本。

## 📊 主要结果 (Key Results)

实证分析表明，叙事模糊性因子 ($D_t$) 是一个独特的定价因子：

*   **与 VIX 脱钩**: $D_t$ 与 VIX 的相关性较低，捕捉了独立于波动率之外的认知不确定性。
*   **定价能力**: 在控制了 FF5、HXZ5 等主流因子后，$D_t$ 依然显著。
*   **大模型涌现能力**: 只有参数量达到一定规模且具备深层语义理解能力的模型（如 DeepSeek-V3, Qwen-Max），其生成的 $D_t$ 才能有效预测资产价格。

## 📝 引用 (Citation)

如果您在研究中使用了本代码或受到论文启发，请引用：

```bibtex
@article{2025narrative,
  title={Narrative Ambiguity, Cognitive Robustness, and Asset Pricing},
  author={Author Name},
  journal={Preprint submitted to Elsevier},
  year={2025},
  note={Available at GitHub: https://github.com/qianphhhh-fin/llm-ambiguity-premium}
}
```

## ⚠️ 免责声明 (Disclaimer)

*   本仓库提供的代码仅供学术研究使用。
*   部分数据（如 CRSP/Compustat, TIPS）需要相应的数据库权限，仓库中仅提供示例数据或处理逻辑。
*   请勿将 API Key 上传至公共仓库。

---
*Last Updated: December 2025*

