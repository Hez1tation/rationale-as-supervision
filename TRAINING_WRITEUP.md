# Rationale-Critic 训练方案说明（Qwen2.5-7B）

> 对应脚本：`train_critic.py`；启动：`train_qwen.sh` / `job_qwen.sh`
> 数据：`data/{train,dev,test}.jsonl`；产物：`rationale_critic_checkpoint/`、`final_results.json`、`train.out`

本文件回答四个问题：**(1) 模型怎么搭的、(2) 数据怎么处理的、(3) 输入 prompt 和目标任务是什么、(4) 损失函数和参数为什么这么设**，并在最后列出**已知局限与可优化点**。

---

## 0. 一句话定位

我们**不是**从零实现网络结构，也**不是**把 Qwen 原样微调。我们以 **Qwen2.5-7B-Instruct 为主干（backbone）**，在其上：
- 加两个自定义输出头（分数回归头、span 分类头）；
- 复用 Qwen 原生 LM 头做 rationale（critique）生成；
- 用一个三路多任务损失联合训练；
- 配套自建的数据处理 pipeline（`prepare_spans.py` 把 `*_copy` 解析成真实 span）。

这对应 schedule.pdf 里 Stage 3 的核心贡献：**span 级失败定位 + 自然语言 critique 生成**，外加数据里本就有的多维数值分数监督。

---

## 1. 模型架构（一个主干，三路输出）

```
                       ┌─────────────────────────────┐
question + response ──▶│  Qwen2.5-7B-Instruct + LoRA │──▶ 最后一层 hidden states [B, T, H]
                       └─────────────────────────────┘
                                  │
        ┌─────────────────────────┼──────────────────────────┐
        ▼                         ▼                          ▼
   原生 LM 头              score_head (MLP)              span_head (Linear)
  (rationale 生成)     对 response token 做 mean-pool      每个 token 二分类
   逐 token 词表         → 5 个数值分数                   (是否问题片段)
   交叉熵                (overall/empathy/specificity/
                          factual/toxicity)
```

- **主干**：`AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")`。选 instruct 版是因为它已经会遵循 `### Question / ### Response` 这类结构化格式，省去额外对齐成本；7B + LoRA 可在 2×A100 上训练（schedule 原定 Llama-3.1-8B，我们换成同量级的 Qwen，便于中文/指令跟随）。
- **LoRA**：`r=32, alpha=64, dropout=0.05`，作用在 `q,k,v,o,gate,up,down` 全部投影层；`task_type=CAUSAL_LM`。全量微调 7B 代价过高，LoRA 是 schedule 指定的方案（rank 16–32）。
- **score_head**：`Linear(H,H) → ReLU → Linear(H,5)`，输入是 response 区域 token 的 hidden 均值池化。只池化 response token，是因为分数是针对"回答"打的，不该被 question 文本干扰。
- **span_head**：`Linear(H,4)`，对每个 token 做**带类型 4 分类**：none / medical / factual / toxicity（类型来自 `*_copy` 所属维度）。加 `--span_binary` 可退回 2 分类（问题/非问题）。
- **生成头**：直接用 Qwen 自带 LM 头，不新增参数，生成结构化 critique。

> 与 `train_critic_llama3_1.py` 的区别：旧版用 `AutoModel`（无 LM 头），**只能出分数和 span，不能生成 rationale**；本版换成 CausalLM 才能产出 `pred_rationale`。另外本版去掉了旧版的 medical 二分类头，把监督聚焦在 5 个数值维度上。

---

## 2. 数据处理 pipeline

### 2.1 两种 schema 的统一
- `train.jsonl`：`scores` 是嵌套 dict（`scores.overall.raw/normalized`），`rationales` 是 dict（`overall/medical/factual/toxicity`，部分为 null）。
- `dev/test.jsonl`：`scores` 是扁平 dict（`scores.overall = 1`），`rationales` 是 **list**（每个评审一条）。

`extract_gold_scores()` 用 `as_float()` 递归兼容 `int / str / {raw,normalized}` 三种形态；`extract_gold_rationale()` 用 `flatten_strings()` 兼容 dict 和 list，跳过 `survey_id`，去重后用换行拼接。**这一步已用真实数据三个 split 验证通过。**

### 2.2 提取的字段
- 5 个数值分数：`overall, empathy, specificity, factual, toxicity`（缺失填 3.0 中性值）。
- gold rationale：把该样本所有维度 / 所有评审的 rationale 文本拼成一段（这就是生成头的学习目标）。

### 2.3 真实 span 标注（来自 `*_copy`，由 `prepare_spans.py` 解析）
CounselBench 原始数据每个失败维度都有 `*_copy` 字段：标注者从 response 里**复制出来的问题原文**（medical / factual / toxicity）。清洗时它被保留在 `raw_refer.<dim>.raw_text`，但一直没被用上。`prepare_spans.py` 把它解析成真实 span：
1. 跨三种 schema（train/dev 的 `{dim:{raw_text}}`、test 的 per-survey list）统一抽取 copy 文本；
2. 在 response 里定位：精确匹配 → 长度保持的规范化（智能引号/破折号）→ difflib 模糊匹配；
3. 写出 `span_labels = [{start_char, end_char, failure_type, match, survey_id, text}]`（response 相对偏移）。

定位率（已实跑）：train 69.8% / dev 67.1% / test 66.3%；无法定位的多是标注者概括（如 "All of it"），按方案丢弃。train 集 1400 行中 401 行含真实 span。

> train 脚本读 `span_labels`，用 `offset_mapping` 把字符区间映射成 token 标签（当前**二分类**：问题/非问题，failure_type 已存好，后续可升级为带类型多分类）。若某行没有 `span_labels`（未跑预处理），自动回退到旧的词重叠伪标签并告警。

### 2.4 划分
按 **question** 划分 70/10/20（不是按标注行），避免同一问题的不同评审泄漏到不同 split。train 保留 5 个评审的独立样本（多样性即训练信号），dev/test 做多数表决聚合降噪。

---

## 3. 输入 Prompt 与目标任务

**输入（prompt，所有 token 不计 LM 损失）：**
```
### Question:
{question}

### Response:
{response}

### Task:
Identify the problems in the response, write a clinical rationale,
then give an overall score from 1 to 5.

```
**目标（target，只在这一段算 LM 损失）：**
```
### Rationale:
{gold_rationale}

### Overall: {overall}/5
```
- 训练时 `full = prompt + target` 一起 tokenize，`lm_labels` 把 prompt 段 mask 成 -100，**只监督 target 段**——这就是标准 SFT 的做法。
- 同一条序列里，response 区域的 token 同时被 score 头（池化）和 span 头（逐 token）使用，靠 `offset_mapping` 定位字符区间。
- 推理时只喂 prompt，`generate()` 贪心解码出 rationale，再用正则抽 `### Rationale:` 段作为 `pred_rationale`；分数用 score 头输出（不靠解析文本里的 `### Overall`，回归头更稳）。

---

## 4. 损失函数与超参数（含"为什么"）

### 4.1 损失
```
L_total = w_score · L_score      # 5 维分数的 MSE
        + w_span  · L_span       # span token 的交叉熵 (ignore_index=-100)
        + w_rat   · L_rationale  # rationale 目标段的 causal-LM 交叉熵
```
默认 `w_score = w_span = w_rat = 1.0`。
- **L_score 用 MSE**：分数是 1–5 的有序数值，回归 MSE 是最简单的基线。
- **L_span 用 token CE**：二分类，`ignore_index=-100` 跳过非 response / padding。
- **L_rationale 用 LM CE**：标准生成损失，`exp(loss)` 即困惑度。

### 4.2 关键超参（脚本默认值）
| 参数 | 值 | 为什么 |
|------|----|--------|
| max_length | 1024 | 覆盖 Q+R+rationale；rationale 中位数约 576 词 |
| max_new_tokens | 256 | 生成上限（可调大到 384–512 更完整，但更慢）|
| epochs | 5 | schedule 建议（比纯打分任务复杂，需更多轮）|
| batch_size × grad_accum | 1 × 8 = 8 | 7B 显存受限，用梯度累积凑有效 batch |
| lr | 1e-5 | LoRA + 7B 的保守稳定学习率 |
| warmup_ratio / wd / clip | 0.1 / 0.01 / 1.0 | 常规稳定项 |
| 精度 | bf16 | L40S/Ampere+ 原生支持；V100/P100 不支持 bf16，需改 `--fp16` |
| GPU 加载 | 单卡 `device_map={"":gpu}` | **不用 `device_map="auto"`**：7B/8B 在单张 L40S(48GB)放得下，auto 会把模型拆到两张卡走 PCIe 或 offload，反而慢数倍 |
| 注意力 | `--attn_impl flash_attention_2` | 装不上自动回退 `sdpa` |
| 梯度检查点 | 默认关 | 48GB 放得下；OOM 时再加 `--grad_checkpoint` |
| LoRA r/α/dropout | 32 / 64 / 0.05 | schedule 指定区间 |

### 4.3 评估指标
- 分数：MAE / Spearman / Kendall（每维）。
- rationale：token-F1 + ROUGE-L（手写、无外部依赖）。
- span：两套 token 级 P/R/F1 —— `detect_*`（任意失败 vs none，不看类型）和 `typed_*`（还要类型判对）。
- 选择标准：**dev 的 overall MAE 最低**存 best；最后 dev+test 全量评估导出 `final_results.json`（含每条样本 gold↔pred 对比）。

---

## 5. 已知局限与可优化点（重要——这部分决定结果可不可信）

1. **分数 MSE 在数值上压倒总损失**。实测 `loss≈25` 时 `score≈21.8 / span≈1.0 / rationale≈2.3`，三项不在一个量级，梯度被 score 主导，可能拖累生成与 span 学习。
   → 可选：把分数归一化到 [0,1]、或 `--score_loss_weight 0.1`、或换 SmoothL1 / 有序回归。

2. **span 现已用真实 `*_copy` 标注 + 带类型 4 分类**（none/medical/factual/toxicity，不再是伪标签）。剩余局限：只有约 67–70% 的 copy 片段能定位到字符位置，其余概括类（如 "All of it"）被丢弃；三类正样本偏少（medical/factual/toxicity 各约 250–340 条），`typed_f1` 在小数据下可能偏低。
   → 可选：对无法定位的片段做句子级兜底以提高召回；或先看 `detect_f1`（类型无关）是否健康，再看 `typed_f1`。

3. **目标格式省略了 span/failure-type 文本**。schedule 设计的是"先 span → 再 rationale → 最后 score"的 CoT 顺序，但数据没有 failure_type 标签，我们当前 target 只有 Rationale + Overall。
   → 可选：在 target 里加句子序号标记（"[sentence 3]: ..."）逼近原设计。

4. **gold rationale 质量参差**。它是多个评审、多个维度 rationale 的拼接，长、风格不一、可能含 "no issues"——生成目标本身偏噪声，会限制生成质量上限。
   → 可选：按维度分别训练，或先做清洗/结构化。

5. **rationale 评估是词面指标**（token-F1/ROUGE-L），只是临床正确性的弱代理。最终仍需 schedule 里的临床医生 faithfulness 评分（>80% 目标）。

6. **小数据 + 5 epoch 有过拟合风险**（train 仅 1400 行、80 个问题）。需盯住 dev 曲线，必要时早停。

7. **score 用 mean-pool**，忽略位置信息，可试 last-token 或 attention pooling。

---

