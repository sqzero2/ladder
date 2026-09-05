# LADDER 实验代码

本仓库只提供 LADDER 教育答疑分级披露研究的源代码与实验指南，不包含真实 API Key、`.env`、题库、知识图谱数据、模型回答、统计结果、论文稿或个人路径。

## 目录与实验组

- `LADDER_Code/`：模型客户端、SHaPE 基线、LADDER 基础模块与评价工具。
- `LADDER_v2_Code/ladder_v2/`：六维学习者状态、状态更新与四级披露决策。
- `LADDER_v2_Code/pipeline/`：主实验、攻击实验、离线判级与汇总入口。
- `消融与统计_v5.2/scripts/`：受控消融及双向聚类统计。
- `最小补强包_v5.3/scripts/`：系统配对统计与人工标注分析。

实验组：`E0` 为始终生成完整解答的检查条件，`E1` 为 SHaPE 知识状态二元门控基线，`A2` 为仅知识状态的四级披露，`A3` 为包含六维学习者状态修正的完整 LADDER，`A4` 为仅通过提示词传递披露等级的对照条件。论文的主要系统比较为 `E1` 与 `A3`。

实验代码保存首轮回答，独立判级器随后离线检查实际披露等级；该实验流程不执行反馈重答。

## 1. 安装

建议使用 Python 3.10 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r LADDER_Code/requirements.txt
```

Linux 或 macOS 使用 `source .venv/bin/activate` 激活环境。

## 2. 模型凭据与隐私

只在本机终端中设置实际使用的服务商变量。以下值均为匿名占位符：

```powershell
$env:DEEPSEEK_API_KEY = "<key>"
$env:APINEBULA_API_KEY = "<key>"
```

代码还支持 `OPENAI_API_KEY`、`DASHSCOPE_API_KEY`、`ZHIPU_API_KEY`、`MOONSHOT_API_KEY`、`SILICONFLOW_API_KEY`、`OPENROUTER_API_KEY`、`TOGETHER_API_KEY`、`GEMINI_API_KEY` 和 `CLAUDE_API_KEY`。只设置实际使用的一项，不要把替换后的命令或终端记录提交到 Git。

`.gitignore` 已排除 `.env`、数据目录、JSON/CSV 输出、论文文档和 Python 缓存。不要使用 `git add -f` 绕过这些规则。

## 3. 实验输入

数据不随代码发布。主实验需要：

1. 试题 JSON 数组或 JSONL：每题至少包含 `problem` 和 `step`，正式评价还使用 `answer` 或 `gpt_answer`，并可包含 `step_detail`。
2. 知识图谱邻接矩阵 CSV：首行与首列是相同的知识点集合，其余单元格表示前置关系。

最小试题结构：

```json
[
  {
    "problem": "<question text>",
    "answer": "<reference answer>",
    "step": ["<knowledge point>"],
    "step_detail": ["<reference step>"]
  }
]
```

当前正式运行器会验证 91 个知识点以及试题—图谱映射。使用其他图谱规模时，需要先调整相应校验逻辑并重新验证实验设计。

## 4. 无 API 冒烟测试

```powershell
python LADDER_v2_Code/pipeline/run_smoke_test.py
```

该测试检查学习者状态、分级决策和状态更新，不调用外部模型。终端末尾应显示 `ALL PASSED`。

## 5. 小规模连通性测试

先以 2 道题和 2 名模拟学习者验证输入、模型连接和输出路径：

```powershell
python LADDER_v2_Code/pipeline/run_full_experiment.py `
  --questions "<QUESTIONS_JSON>" `
  --adjacency "<KNOWLEDGE_GRAPH_CSV>" `
  --model "deepseek-chat" --provider "deepseek" `
  --arms "E1,A3" --attack "baseline" `
  --start 0 --end 2 --students 2 `
  --seed 42 --temperature 1.0 `
  --concurrent 1 --retries 3 `
  --output "<RAW_OUTPUT_DIR>"
```

尖括号字段均为匿名占位符。输出应放在仓库之外，或放入已被 `.gitignore` 排除的目录。

## 6. 正式实验

主实验：

```powershell
python LADDER_v2_Code/pipeline/run_full_experiment.py `
  --questions "<QUESTIONS_JSON>" `
  --adjacency "<KNOWLEDGE_GRAPH_CSV>" `
  --model "deepseek-chat" --provider "deepseek" `
  --arms "E0,E1,A3" --attack "baseline" `
  --start 0 --end 30 --students 30 `
  --seed 42 --temperature 1.0 `
  --concurrent 10 --retries 3 `
  --output "<RAW_OUTPUT_DIR>"
```

30 道题与 30 名学习者形成每组 900 个配对样本。比较依赖完整配对，不应只删除某一系统中的失败样本后比较比例。

显式诱导攻击分别运行，只需把主实验命令中的相关参数改为：

```text
--arms "E1,A3" --attack "refusal_suppression" --start 0 --end 20 --students 30
--arms "E1,A3" --attack "role_play_en" --start 0 --end 20 --students 30
```

其余模型、温度和随机种子保持一致。每种攻击形成每个系统 600 个配对样本。

跨模型复现保持试题范围、学习者数量、随机种子、温度和 `E1,A3` 不变，只替换：

```text
--model "<TEACHER_MODEL>" --provider "<PROVIDER>"
```

跨模型结果仅用于检验比较方向是否一致，不代表对所有模型的普遍验证。

## 7. 离线披露等级判定

先对单个小样本文件运行，确认输出后再扩大范围，避免意外模型费用：

```powershell
python LADDER_v2_Code/pipeline/run_disclosure_eval.py `
  --input "<RAW_OUTPUT_DIR_OR_JSON>" `
  --output "<DISCLOSURE_OUTPUT_DIR>" `
  --model "gpt-5.4-mini" --provider "apinebula" `
  --concurrent 10
```

该步骤输出实际披露等级，以及越级、精确匹配和不足披露字段。

## 8. 汇总与配对统计

汇总同一目录中的主实验和攻击判级结果：

```powershell
python LADDER_v2_Code/pipeline/summarize_v5_1.py `
  --input "<DISCLOSURE_OUTPUT_DIR>" `
  --output "<SUMMARY_OUTPUT_DIR>" `
  --attack-questions 20
```

对同一条件下完整配对的 `E1` 与 `A3` 判级文件计算双向聚类置信区间及 McNemar 敏感性结果：

```powershell
python "最小补强包_v5.3/scripts/analyze_shape_vs_ladder.py" `
  --shape "<E1_DISCLOSURE_JSON>" `
  --ladder "<A3_DISCLOSURE_JSON>" `
  --output "<PAIRED_STATS_DIR>" `
  --bootstrap 5000 --seed 42
```

## 9. 受控消融

以主实验 `A3` 的冻结后诊断 JSON 为输入：

```powershell
python "消融与统计_v5.2/scripts/run_controlled_ablation.py" `
  --input "<A3_BASELINE_JSON>" `
  --output "<ABLATION_RAW_DIR>" `
  --model "deepseek-chat" --provider "deepseek" `
  --arms "B0_binary_k,B1_prompt_only_full,B2_k_only_tiered,B3_full_ladder" `
  --concurrent 10 --retries 3 --seed 20260822
```

使用第 7 节命令判级消融输出后运行：

```powershell
python "消融与统计_v5.2/scripts/analyze_ablation.py" `
  --input "<ABLATION_DISCLOSURE_DIR>" `
  --output "<ABLATION_STATS_DIR>" `
  --bootstrap 2000 --seed 42
```

该消融比较冻结诊断后的局部替换，不应把完整系统效应归因于某个单独组件。

## 10. 复现与提交检查

- 固定并记录 Git 提交、题目索引、图谱版本、教师模型、判级模型、温度、随机种子和并发数。
- 主实验、攻击实验和跨模型实验均保持学习者—试题配对。
- 原始回答、判级结果和统计输出保存在仓库之外。
- 按“无 API 冒烟测试 → 2×2 小样本 → 正式实验”的顺序运行。
- 判级会产生新的模型调用和费用，正式执行前检查输入范围和输出目录。

提交前执行：

```powershell
git status --short
git diff --cached --name-only
```

确认暂存区中没有 `.env`、真实密钥、数据、JSON/CSV 输出、模型回答、论文稿、邮箱、用户名或本机绝对路径。
