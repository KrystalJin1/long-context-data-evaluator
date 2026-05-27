长文训练数据入训适配性评估方案

## 0. 文档目标
本文档用于评估一批长文数据是否适合进入模型训练，并进一步判断它更适合支持哪一类长文本能力训练。

本文档重点回答三个问题：

1. **这批数据总体是否适合入训？**
关注数据是否干净、完整、低重复、低噪声、结构清晰、来源可靠。
2. **这批数据是否具备长文训练价值？**
关注它是否只是“文本很长”，还是确实包含长程依赖、多点信息、跨段结构、长输入/长输出等有效训练信号。
3. **这批数据更适合训练哪项长文能力？**
例如 MRCR、多文档摘要、长上下文检索、RAG/CITE、长对话、长文生成等。

本文档设计两类核心分数：

|**分数**|**英文**|**作用**|
|-|-|-|
|入训适配分|Entry Suitability Score, ESS|判断数据整体是否适合入训|
|任务适配分|Task Fit Score, TFS|判断数据是否适合某项特定任务训练|

需要注意：

> 数据侧评分只能说明数据具备某类训练潜力，不等价于模型能力一定提升。最终效果仍需要结合训练实验、benchmark 评测或小样本验证闭环。
---

## 1. 核心设计原则
### 1.1 不只看“长”，更看“有效长文信号”
长文本数据最容易出现误判：

> 文本很长，所以一定有长文训练价值。
但真实情况不是这样。一条 128K 文本如果只是重复模板、OCR 噪声、网页拼接、无结构 PDF 提取结果，它对长文能力提升可能很有限，甚至会伤害训练质量。

因此评估时要同时看：

|**维度**|**问题**|
|-|-|
|长度|是否覆盖 32K / 64K / 128K / 256K？|
|结构|文档边界、段落、对话轮次、标题是否清晰？|
|信息密度|是否大量重复、模板化、SEO、HTML 噪声？|
|长程信号|是否存在跨段引用、实体复现、多目标、多文档关系？|
|任务信号|是否适合 MRCR、摘要、检索、RAG、长生成等具体任务？|
|来源风险|是否过度集中在少数来源、语种或领域？|

---

### 1.2 保留旧报告口径，但从“已入训回看”改为“入训前判断”
旧抽样长文分析报告：[入训长文数据采样分析与数据侧缺口总结](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/Ep-tPYL1jN/206xJ1sZ9_8dr6?source=137?t=mention&mt=doc&dt=doc)

旧 1w 报告是对已入训长文数据池的回看，分析维度包括长度分布、输入/输出结构、来源分布、合成比例、质量风险、能力覆盖和数据缺口；该报告使用 EB5 SentencePiece tokenizer，样本来自全量已入训长文数据池随机采样 10,000 条。

新方案保留这些有用维度，但分析重点从：

> 这批已入训数据是什么样？
调整为：

> 新数据是否适合入训？是否补齐旧数据缺口？是否适合某项长文能力训练？
---

### 1.3 已有标签可接入，但不作为唯一依据
新数据原始字段默认不包含任务能力、数据质量、数据类型标签；若后续通过组内打标代码生成标签，则可作为辅助增强字段接入分析。入训适配评估仍以规则侧指标为主，标签结果用于交叉验证和补充解释。

|**标签类型**|**作用**|
|-|-|
|任务能力标签|判断数据可能训练哪些任务能力|
|数据质量标签|从 15 个维度评价文本质量|
|数据类型标签|判断文本属于新闻、百科、学术文献、代码、问答、对话文本等哪类|

其中数据质量标签包含 hygiene、hybrid、incentive 三组共 15 个维度，并给出 `Fatal / Low_Value / Baseline / Diamond / High_Quality / Gold` 等 overall category。

但本文档不完全依赖这些标签。原因是：

1. 标签来自模型打标，可能有误差；
2. 任务能力标签不一定等于真实训练价值；
3. 长文能力还需要结合长度、位置、结构、干扰项、证据分布等规则指标判断；
4. 所有标签都应与规则特征交叉验证。

---

## 2. 统一分析口径
|**项目**|**统一口径**|
|-|-|
|分析单位|一条训练样本|
|tokenizer|默认使用 EB5 SentencePiece 或组内指定 tokenizer|
|长文定义|默认 32K+ 为长文，128K+ 为超长文|
|长度分桶|0-4K、4-8K、8-16K、16-32K、32-64K、64-128K、128-256K、256K+|
|输入输出字段|`prompt_tokens`、`candidate_tokens`、`total_tokens`|
|candidate 为空|不直接判为低质；content-only 预训练数据中 candidate 为空是正常现象|
|能力标签|允许多标签，不强制单标签|
|数据类型标签|单标签或主标签，用于判断文本形态|
|质量风险标签|可多标签，例如高重复、模板化、HTML 噪声、乱码、低信息密度|
|对比口径|新数据与旧 1w baseline 使用同一 tokenizer 和同一长度分桶|

---

# 3. General 长文数据分析维度
这一部分适用于所有长文数据批次。无论数据最终用于 MRCR、多文档摘要还是 RAG，都应先完成 general 数据画像。

## 3.1 长度覆盖
### 目的
判断数据是否覆盖关键上下文窗口，尤其是 32K、64K、128K、256K。

### 指标
|**指标**|**计算方式**|**解释**|
|-|-|-|
|`total_tokens`|样本总 token 数|判断整体长度|
|`length_bucket`|按统一分桶划分|判断长度分布|
|`32k_plus_ratio`|32K+ 样本数 / 总样本数|长文覆盖|
|`64k_plus_ratio`|64K+ 样本数 / 总样本数|中高长文覆盖|
|`128k_plus_ratio`|128K+ 样本数 / 总样本数|超长窗口覆盖|
|`256k_plus_count`|256K+ 样本数|稀缺超长样本绝对量|

### 分析重点
旧 1w baseline 中，32K+ 占 29.69%，64K+ 占 11.49%，128K+ 占 4.68%，256K+ 仅 137 条。旧报告认为，如果目标是提升 128K/256K 窗口稳定能力，当前超长样本数量不足。

新数据应重点判断：

* 是否提升 128K+ / 256K+ 覆盖；
* 是否只是增加 32K-64K，而未补充真正超长窗口；
* 256K+ 样本是否质量足够高，是否值得保留或上采样。

---

## 3.2 输入/输出结构
### 目的
判断数据主要训练模型“读长文”，还是训练模型“写长文”。

### 指标
|**指标**|**计算方式**|**解释**|
|-|-|-|
|`prompt_tokens`|输入 token 数|长输入程度|
|`candidate_tokens`|输出 token 数|长输出程度|
|`candidate_empty_ratio`|candidate 为空比例|判断是否 content-only|
|`long_input_short_output_ratio`|长输入短输出样本占比|长文阅读、检索、问答|
|`short_input_long_output_ratio`|短输入长输出样本占比|长篇生成、续写、报告|
|`both_long_ratio`|输入输出都长样本占比|摘要、报告、多文档综合|



### 推荐分类
|**类型**|**规则示例**|**含义**|
|-|-|-|
|`content_only`|candidate 为空或极短|纯文本预训练材料|
|`long_input_short_output`|prompt ≥ 32K，candidate < 2K|长文理解 / 检索 / 问答|
|`short_input_long_output`|prompt < 8K，candidate ≥ 8K|长文生成 / 续写 / 长报告|
|`both_long`|prompt ≥ 32K，candidate ≥ 8K|长文摘要 / 多文档综合 / 长报告生成|
|`normal_or_medium`|其他|普通中短样本|

### 分析重点
旧 1w baseline 中，32K+ 样本里长输入短输出 2,966 条，占 99.9%；短输入长输出只有 2 条；candidate 为空 9,923 / 10,000 条。旧报告判断，这批已入训长文更像纯文本预训练材料，主要训练模型“读长文”，对长篇创作、长报告、长翻译、长代码生成等长输出稳定性帮助有限。

新数据要重点看：

* 是否补充 `short_input_long_output`；
* 是否出现 `both_long` 样本；
* 多文档摘要数据是否真的有有效 candidate；
* MRCR 数据 answer 是否可追溯到上下文。

---

## 3.3 数据类型覆盖
### 目的
判断数据“是什么类型的文本”。

### 可用标签
若已有数据类型标签，可直接使用。新模板中数据类型共 19 类，包括新闻、百科、翻译文本、学术文献、专利文本、非虚构图书、文学与创作、转录文本、OCR识别文档、试题、代码、教程、问答、论坛社区、专业文档、商业写作、用户评论、对话文本、其他。

### 推荐分析指标
|**指标**|**解释**|
|-|-|
|`data_type_distribution`|数据类型分布|
|`data_type_by_length_bucket`|不同长度段的数据类型|
|`data_type_by_language`|不同语种的数据类型|
|`data_type_by_quality`|不同数据类型的质量风险|
|`data_type_by_task_fit`|哪些类型更适合 MRCR / 摘要 / RAG|

### 为什么要拆出数据类型
数据类型回答的是：

> 这条数据是什么？
它不同于任务能力。比如：

|**数据类型**|**可能对应任务能力**|
|-|-|
|学术文献|知识问答、阅读理解、专业推理、摘要|
|对话文本|多轮记忆、聊天扮演、MRCR、长对话|
|问答|篇章问答、RAG、知识问答|
|专业文档|垂类理解、信息抽取、CITE|
|文学与创作|长文生成、续写、风格模仿|
|代码|代码解释、代码生成、仓库理解|

---

## 3.4 任务能力覆盖
### 目的
判断数据可能训练模型哪些能力。

### 可用标签
若已有任务能力标签：

|字段|含义|例子|
|-|-|-|
|`level1`|一级能力大类|文本创作、知识问答、推理计算、代码能力、信息处理|
|`level2`|二级能力方向|文稿创作、阅读理解、数学计算、代码解释|
|`level3`|更细的任务类型|报告、篇章问答、算术运算、概括总结|
|`score`|这条数据和该能力的匹配程度|1-5 分|
|`reason`|为什么打这个标签|简短理由|
|`primary_label`|最主要的任务能力标签|score 最高或最核心的标签|

任务能力标签最多输出 5 个标签，且 score ≥ 2 才输出。

### 推荐统计指标
|指标|解释|
|-|-|
|`primary_label_distribution`|主能力分布|
|`level1_distribution`|一级能力覆盖|
|`level2_distribution`|细分能力覆盖|
|`avg_label_score`|标签平均匹配分|
|`high_score_label_ratio`|score ≥ 4 的标签比例|
|`multi_label_count`|多能力样本数量|
|`task_label_entropy`|能力分布多样性|

### 分析重点
任务能力标签可以改进旧报告中较粗的 `ability_type` 分类。但它仍然是辅助信号。最终是否适合某个长文任务，还要看特定任务结构，例如 MRCR 的 turn、target_index、distractor、target_position。

---

## 3.5 来源多样性与集中度
### 目的
判断数据是否过度集中在少数来源，避免上采样时放大某一来源的格式偏差、领域偏差和噪声。

### 指标
|**指标**|**计算方式**|
|-|-|
|`source_count`|不同来源数量|
|`source_top1_share`|最大来源占比|
|`source_top3_share`|前 3 来源占比|
|`source_top5_share`|前 5 来源占比|
|`source_hhi`|来源占比平方和|
|`unknown_source_ratio`|来源未知比例|

### 分析重点
旧报告发现，越往超长区间，来源越容易集中在 finepdfs 等少数 PDF 来源；128K+ 样本中约 65% 来自书籍和学术论文，来源单一，存在超长窗口泛化风险。

新数据要重点看：

* 高长度段来源是否更丰富；
* 是否仍然集中在 PDF / 书籍 / 学术论文；
* MRCR、多文档摘要数据是否过度依赖单一 pipeline 或单一来源；
* 是否需要 source cap，限制单一来源上采样比例。

---

## 3.6 语种与领域覆盖
### 目的
判断数据是否覆盖目标语种和目标领域，尤其是中文长文和垂类专业长文。

### 指标
|指标|解释|
|-|-|
|`language_distribution`|全量语种分布|
|`zh_32k_ratio`|中文 32K+ 占比|
|`en_32k_ratio`|英文 32K+ 占比|
|`domain_distribution`|法律、金融、医疗、科研、教育、科技等领域|
|`domain_128k_ratio`|垂类 128K+ 覆盖|
|`domain_source_count`|每个领域来源数量|

### 分析重点
旧报告中，32K+ 中文样本仅 317 条，占 10.7%，且中文网页长文仅 5 条，说明中文长文不仅量少，而且来源类型单一。

如主要训练模型英文长文能力，主要拆分英文维度即可。

如需改进中文能力新数据要重点看：

* 中文长文是否增加；
* 中文数据是否仍主要来自书籍/论文；
* 是否新增中文网页、报告、问答、对话、专业文档；
* 英文多文档摘要候选是否符合当前 pipeline 需求。

---

## 3.7 合成数据占比与合成类型
### 目的
判断数据是自然长文、抓取清洗数据、合成数据、混合合成数据，还是未知来源；进一步判断合成数据是否有明确能力目标，以及是否存在模板单一、pipeline 失败、解析错误等风险。

### 合成类型定义
|**类型**|**定义**|**示例**|
|-|-|-|
|`natural_raw`|人类自然产生的原始文本，基本未经过任务化改造|书籍、论文、网页正文、新闻、论坛长文|
|`crawled_cleaned`|抓取后经过清洗、去噪、切分、格式化的自然文本|清洗后的网页、PDF 文档、书籍章节|
|`synthetic`|主要内容或任务由模型、模板、规则 pipeline 生成|模型生成问答、模型生成摘要、合成指令数据|
|`hybrid_synthetic`|原始材料是真实文档，但任务形式由合成 pipeline 构造|基于真实文档构造 MRCR、RAG QA、多文档摘要|
|`unknown`|缺少来源或加工信息，无法判断|source / pipeline / metadata 缺失|

建议不要只用二分类 `synthetic / non-synthetic`，因为长文数据中常见的是 **自然语料 + 合成任务结构**。例如多文档摘要可能使用真实网页/论文作为输入，但摘要或指令是模型生成的；MRCR 可能使用真实文本作为 needle，但多轮对话和第 N 个检索任务是 pipeline 构造的。



### 判定优先级
判断是否合成时，建议按以下优先级：

|优先级|判定依据|说明|
|-|-|-|
|P0|显式字段|如果已有 `is_synthetic`、`data_type`、`pipeline_type`、`synthetic_type`，优先使用|
|P1|来源 / pipeline 名称|source 或文件路径中包含 `synthetic`、`generated`、`mrcr`、`rag`、`summary`、`sft`、`pipeline` 等|
|P2|结构特征|是否存在模板化 instruction、query-answer、target_index、doc_id、evidence、summary 等任务结构|
|P3|内容特征|是否存在明显模型生成痕迹、模板复用、固定格式、多样性不足|
|P4|无法判断|没有明确字段或结构证据时，标为 `unknown`，不强行归类|



### 指标
|**指标**|**解释**|
|-|-|
|`natural_raw_ratio`|原始自然长文占比|
|`crawled_cleaned_ratio`|抓取清洗数据占比|
|`synthetic_ratio`|纯合成数据占比|
|`hybrid_synthetic_ratio`|自然文档 + 合成任务结构的数据占比|
|`unknown_type_ratio`|未知类型占比|
|`synthetic_pipeline_type`|MRCR、多文档摘要、RAG、检索拼接等 pipeline 类型|
|`template_diversity`|指令模板、query 模板数量|
|`pipeline_success_ratio`|pipeline 生成成功率|
|`parse_error_ratio`|解析失败比例|
|`answer_context_match_ratio`|answer 是否能在 context 中匹配|
|`source_traceable_ratio`|是否能追溯原始文档或 source|

### 分析重点
旧 1w baseline 中，32K+ 合成数据约占 29.9%，是长文训练的重要组成，但旧报告也提示，合成数据的多样性和能力覆盖需要进一步审视。

新数据要回答：

* 合成数据是为了哪个能力构造的；
* 模板是否多样；
* 目标答案是否可追溯；
* 是否存在大量同模板重复样本；
* 是否有明确 task fit 分数支持。

---

## 3.8 规则质量风险
### 目的
规则质量风险用于在不依赖大模型的情况下，初步识别可能影响训练质量的样本。这里的“高风险”不等于内容一定有害，而是表示样本存在较高的数据质量问题，入训前需要清洗、降权、抽样复核或过滤。

### 高风险定义
若样本触发一个或多个严重质量风险规则，则标记为 `high_risk = true`。常见风险包括高重复、模板化、HTML 噪声、乱码、低信息密度、字段缺失、结构破碎，以及特定任务关键字段缺失。

### 风险类型
|**风险标签**|**含义**|**可能处理方式**|
|-|-|-|
|`high_repetition`|重复 n-gram、重复行、重复段落过多|去重、降权、过滤|
|`template_like`|样本模板高度一致|降权、增加模板多样性|
|`html_noise`|HTML、URL、导航栏、广告、页脚混入正文|清洗后保留|
|`garbled_text`|乱码、OCR 错误、异常字符|严重则过滤|
|`low_information_density`|文本很长但有效信息少|降权或过滤|
|`code_heavy`|代码占比高但结构不清|单独进入代码质量检查|
|`metadata_missing`|source、language、doc_id 等缺失|补元数据或降置信度|
|`structure_broken`|段落、表格、代码、文档边界破碎|清洗/结构修复|
|`task_key_missing`|特定任务缺关键字段|不作为该任务主力数据|

### 初版触发规则
|**风险标签**|**触发条件示例**|
|-|-|
|`high_repetition`|`ngram_repetition_ratio > 0.25` 或 `duplicate_line_ratio > 0.20`|
|`template_like`|同模板样本比例过高，或 query/prompt 模板重复率 > 0.30|
|`html_noise`|`html_tag_ratio > 0.05`，或 URL / 导航 / 页脚明显过多|
|`garbled_text`|`weird_char_ratio > 0.02`|
|`low_information_density`|`info_density_ratio < 0.60`|
|`metadata_missing`|source / language / doc_id 等关键字段缺失|
|`structure_broken`|段落边界、doc boundary、turn boundary 大量缺失|
|`task_key_missing`|MRCR 缺 target_index；多文档摘要缺 doc boundary；RAG/CITE 缺 evidence|

### 处理原则
高风险样本不一定全部过滤，应结合任务稀缺性和内容价值判断：

* 严重乱码、结构完全破碎、不可读样本：过滤；
* 高重复、模板化样本：去重或降权；
* 内容价值高但格式差的样本：进入清洗池；
* 128K+ / 256K+ 稀缺超长样本：优先抽样复核，不直接删除；
* 特定任务关键字段缺失的样本：不作为该任务主力数据。



---

# 4. Information Density 信息密度
## 4.1 目的
判断长文是否是真正有效信息，还是靠重复、模板、HTML 噪声堆出来的长文本。

## 4.2 规则计算
```
Information Density =
1.0
- 0.8 × ngram_repetition_ratio
- 0.6 × duplicate_line_ratio
- 0.5 × html_noise_ratio
- 0.5 × weird_char_ratio
- 0.3 × template_ratio
```
最终截断到 `[0, 1]`：

```
info_density_ratio = max(0, min(1, Information Density))
```
其中：

* `ngram_repetition_ratio`：重复 n-gram 占比；、
* `duplicate_line_ratio`：重复行或重复段落占比；
* `html_noise_ratio`：HTML 标签、URL、导航栏、页脚等噪声占比；
* `weird_char_ratio`：乱码、异常字符、不可见字符占比；
* `template_ratio`：模板化片段占比。

## 4.3 解释
|**info_density_ratio**|**判断**|
|-|-|
|≥ 0.85|信息密度高，优先保留|
|0.70 - 0.85|正常可用|
|0.60 - 0.70|边界样本，建议复核|
|< 0.60|低信息密度，建议降权或过滤|

---

# 5. LCU：Long Context Utility 长文训练价值分
## 5.1 定义
LCU, Long Context Utility，用于衡量一条样本作为长文训练材料的综合价值。

它不是模型 benchmark，也不是学术标准指标，而是数据侧的规则近似评分，主要用于：

* 大规模样本筛选；
* 新旧数据对比；
* 入训优先级排序；
* 上采样 / 降权 / 过滤建议。

LCU 重点回答：

> 这条数据是否具备长上下文训练价值？
---

## 5.2 LCU-General 计算公式
所有子分数统一归一化到 0-100。

```
LCU-General =
0.25 × length_score
+ 0.20 × effective_context_score
+ 0.25 × information_density_score
+ 0.15 × structure_score
+ 0.15 × long_context_signal_score
- risk_penalty
```
最终结果截断到 0-100：

```
LCU-General = max(0, min(100, Raw Score - risk_penalty))
```
### 缺失字段处理
若某个子分数因为字段缺失无法计算，例如没有 target / evidence / doc boundary 时无法稳定计算 `long_context_signal_score`，则该项记为 `N/A`，不直接赋 0。

计算总分时，只在可计算项之间重新归一化权重。例如：

* 普通 content-only 长文：主要使用 length、effective_context、information_density、structure 和 risk_penalty；
* MRCR / RAG / 多文档摘要等结构化数据：额外计算 long_context_signal_score；
* 缺失字段较多时，应同时输出 `score_coverage`，表示本条样本有多少评分维度可计算。

---

## 5.3 分项口径
|**分项**|**含义**|**可量化来源**||
|-|-|-|-|
|`length_score`|样本是否足够长|`total_tokens`、长度分桶||
|`effective_context_score`|是否形成有效长上下文|`prompt_tokens`、`candidate_tokens`、`io_type`||
|`information_density_score`|长文是否有有效信息|重复率、HTML、乱码、模板化、行重复||
|`structure_score`|结构是否清晰|段落、标题、doc boundary、turn boundary||
|`long_context_signal_score`|是否有长程训练信号|实体复现、跨段引用、多文档、多目标、evidence 距离||
|`risk_penalty`|低质风险扣分|高重复、HTML、乱码、模板化、低密度等||

---

## 5.4 具体评分规则
### length_score
|**total_tokens**|**分数**|
|-|-|
|< 8K|0|
|8K-16K|20|
|16K-32K|40|
|32K-64K|60|
|64K-128K|75|
|128K-256K|90|
|256K+|100|

---

### effective_context_score
|**io_type**|**分数**|**说明**|
|-|-|-|
|`content_only`|60|纯预训练长文，有阅读价值，但任务信号较弱|
|`long_input_short_output`|90|适合长文理解、检索、问答|
|`both_long`|100|适合摘要、多文档综合、长报告|
|`short_input_long_output`|70|适合长文生成，但不一定训练长上下文阅读|
|`normal_or_medium`|40|普通中短样本，长文价值较弱|

备注：candidate 为空不等于低质，只说明它更像 content-only 预训练数据。

---

### information_density_score
```
information_density_score = info_density_ratio × 100
```
其中：

```
info_density_ratio =
1.0
- 0.8 × ngram_repetition_ratio
- 0.6 × duplicate_line_ratio
- min(html_tag_ratio / 20, 0.3)
- 0.5 × weird_char_ratio
- 0.3 × template_ratio
```
最终截断到 0-1：

```
info_density_ratio = max(0, min(1, info_density_ratio))
```
---

### structure_score
|**结构特征**|**加分**|
|-|-|
|段落边界清晰|+20|
|标题 / 章节 / 编号清晰|+20|
|文档边界清晰|+20|
|对话轮次清晰|+20|
|query / answer / evidence 字段清晰|+20|

最高 100 分。

备注：普通自然长文至少可计算段落和标题结构；多文档摘要重点看 doc boundary；MRCR / 长对话重点看 turn boundary。

---

### long_context_signal_score
|**长文信号**|**加分**|
|-|-|
|中部 / 尾部存在目标信息或 evidence|+25|
|同一实体跨远距离复现|+20|
|存在“前文 / 上述 / 该文档 / 第二篇”等跨段引用|+20|
|多文档或多段落之间主题一致|+15|
|存在多个候选目标或多证据点|+20|

最高 100 分。

备注：如果没有 target / evidence / doc boundary 等字段，该项可记为 N/A 或仅用弱 proxy，不应强行赋 0。

---

### risk_penalty
|**风险**|**Risk Penalty**|
|-|-|
|高重复|10|
|模板化|8|
|HTML / XML 噪声|8|
|乱码 / 异常字符|12|
|低信息密度|10|
|字段缺失严重|8|
|source 缺失且不可追溯|5|

`risk_penalty` 为正数扣分项，建议封顶 30 分。



最终计算时：

```
LCU-General = max(0, min(100, Raw Score - risk_penalty))
```
---

## 5.5 LCU 结果解释
|**LCU-General**|**判断**|**建议**|
|-|-|-|
|≥ 80|高价值长文|优先入训，可考虑上采样|
|65-80|正常优质长文|保留|
|50-65|边界长文|抽样复核或降权|
|35-50|风险较高|仅稀缺任务可保留|
|< 35|低价值|过滤或严格复核|

---

## 5.6 LCU-Enhanced：质量标签增强版，可选
如果数据后续通过打标代码生成了数据质量标签，可以计算增强版：

```
LCU-Enhanced =
0.80 × LCU-General
+ 0.20 × Quality Label Score
```
其中 Quality Label Score 可由已有质量标签计算：

```
Quality Label Score =
0.35 × Hygiene Score
+ 0.35 × Core Utility Score
+ 0.20 × Value Signal Score
+ 0.10 × Overall Category Score
```
该增强版只作为辅助，不替代规则侧 LCU-General。若新数据暂时没有质量标签，则只计算 LCU-General。

---

# 6. ESS：Entry Suitability Score 入训适配分
## 6.1 定义
ESS 判断一条样本整体是否适合入训。

它综合考虑：

* 基础数据质量；
* 长文训练价值；
* 质量标签辅助信息；
* 风险扣分。

---

## 6.2 计算公式
如果没有质量标签：

```
ESS =
0.55 × Rule Quality Score
+ 0.45 × LCU-General
```
如果已有质量标签：

```
ESS =
0.45 × Rule Quality Score
+ 0.40 × LCU-General
+ 0.15 × Quality Label Score
```
说明：
质量标签只占 15%，用于辅助判断，不让模型标签主导最终入训结论。

---

## 6.3 Rule Quality Score
```
Rule Quality Score =
0.25 × cleanliness_rule_score
+ 0.20 × dedup_repetition_score
+ 0.15 × structure_rule_score
+ 0.15 × completeness_rule_score
+ 0.15 × information_density_score
+ 0.10 × metadata_score
```
所有分数均转换为 0-100。

### 分项说明
|**分项**|**含义**|**可量化来源**|**可计算性**|
|-|-|-|-|
|`cleanliness_rule_score`|文本是否干净|HTML、URL、乱码、异常字符、广告噪声|基本可算|
|`dedup_repetition_score`|是否重复或模板化|n-gram 重复、重复行、重复段落、模板重复|可算|
|`structure_rule_score`|结构是否清晰|段落、标题、doc boundary、turn boundary|部分可算|
|`completeness_rule_score`|内容是否完整|字段缺失、截断、answer/evidence/target 缺失|部分可算|
|`information_density_score`|信息密度是否足够|重复率、HTML、乱码、模板化综合计算|可算|
|`metadata_score`|元数据是否完整|source、language、domain、doc_id、synthetic_type|取决于字段|

### 可计算性备注
`cleanliness_rule_score`、`dedup_repetition_score`、`information_density_score` 基本都能从原始文本中计算；`structure_rule_score` 和 `completeness_rule_score` 需要结合数据结构判断；`metadata_score` 取决于新数据是否提供 source、language、domain、doc_id 等字段。

若某些字段不存在，应标记为 `N/A` 或降低置信度，不建议直接当作严重低质。

---

## 6.4 ESS 解释
|**ESS**|**入训建议**|
|-|-|
|≥ 85|高质量入训样本|
|70-85|可正常入训|
|55-70|边界样本，建议复核或降权|
|40-55|风险样本，仅稀缺任务保留|
|< 40|不建议入训|

---

# 7. 特定任务适配分析
Task Fit Score, TFS 用于判断数据是否适合某项特定能力训练。

### 适用范围说明
Task Fit Score 不要求对所有样本都计算。它只对具备对应任务结构的样本计算。

例如：

* 没有多轮结构、target_index 或相似候选时，不计算 MRCR-Fit；
* 没有多文档边界和摘要输出时，不计算 MultiDocSummary-Fit；
* 没有 query、answer、evidence 时，不计算 RAG-CITE-Fit；
* 没有多轮对话结构时，不计算 LongDialogue-Fit。

如果样本缺少对应任务字段，应标记为 `N/A`，而不是直接给低分。低分表示“结构存在但质量差”，N/A 表示“该任务不适用”。



最终建议同时输出：

```
ESS
LCU-General / LCU-Enhanced
MRCR-Fit
MultiDocSummary-Fit
Retrieval-Fit
RAG-CITE-Fit
LongGeneration-Fit
LongDialogue-Fit
```
一条数据可以同时有多个 Task Fit 分数。

---

# 8. MRCR-like 数据适配评估
## 8.1 任务定位
MRCR-like 数据用于训练模型在长上下文、多轮对话或重复问答中完成：

* 多轮历史检索；
* 共指消解；
* 第 N 个目标定位；
* 多目标区分；
* 相似干扰项排除；
* 中部/尾部信息召回。

这是当前组内最应优先分析的任务方向。

---

## 8.2 MRCR-like 数据应具备的结构
|**维度**|**理想特征**|
|-|-|
|多轮结构|user/assistant 轮次清晰|
|多个 needle|同主题或相似主题目标 ≥ 2|
|target_index|明确要求找第 N 个目标|
|干扰项|filler 或相似回答足够多|
|共指表达|“刚才那个”“前面那篇”“第 N 个”等|
|位置分布|target 分布在 head/middle/tail|
|answer 可追溯|answer 能在历史 turn 或 needle 中找到|

---

## 8.3 MRCR-Fit Score
```
MRCR-Fit =
0.15 × context_length_score
+ 0.15 × turn_structure_score
+ 0.18 × target_candidate_score
+ 0.15 × distractor_score
+ 0.15 × target_position_score
+ 0.12 × coreference_marker_score
+ 0.10 × answer_extractability_score
```
总分 0-100。

### 权重说明
* `target_candidate_score` 权重最高，因为 MRCR 的核心是从多个相似候选目标中找到指定的第 N 个目标。
* `target_position_score` 和 `distractor_score` 对应中尾部召回和抗干扰能力；
* `coreference_marker_score` 用来区分 MRCR 和普通 multi-needle 检索。

---

## 8.4 MRCR 子指标口径
|**子指标**|**计算方式**|**评分参考**|
|-|-|-|
|`context_length_score`|total_tokens 分桶|32K=60，64K=75，128K=90，256K+=100|
|`turn_structure_score`|user/assistant 轮数|0 轮=0，2-4 轮=50，5-10 轮=75，10+ 轮=100|
|`target_candidate_score`|target_candidate_count + target_index 是否明确|1 个目标=30，2 个=60，4 个=80，8 个+=100|
|`distractor_score`|filler_count + 相似干扰数量|无干扰=20，主题不同=60，相似干扰=90+|
|`target_position_score`|target_start_token / total_tokens|头部=50，中部=100，尾部=85，均衡=100|
|`coreference_marker_score`|共指关键词命中|无=0，顺序表达=60，跨轮共指=80，多表达=100|
|`answer_extractability_score`|answer 是否能在上下文中匹配|不可匹配=0，部分匹配=50，精确匹配=90，可追溯 turn_id=100|

---

## 8.5 MRCR 共指表达词表
```
中文：
第N篇、第N个、刚才那个、前面那篇、上一个、之前写的、重新输出、原文给我、再发一遍

英文：
the N-th, the previous one, the earlier one, what you wrote before, reproduce, verbatim, send again
```
---

## 8.6 MRCR-Trainability
MRCR-Fit 只判断任务结构是否像 MRCR。最终是否适合入训，还要结合质量。

```
MRCR-Trainability =
0.70 × MRCR-Fit
+ 0.20 × ESS
+ 0.10 × LCU-General
```
如果已有质量标签，也可额外检查：

|**质量字段**|**重要性**|
|-|-|
|structure|turn 边界是否清晰|
|completeness|answer 是否缺失|
|topic_focus|是否乱拼|
|qa_match|query-answer 是否对应|
|info_density|是否水文/重复|

---

## 8.7 MRCR 难度分级
|**难度**|**数据特征**|**建议**|
|-|-|-|
|Easy|1-2 个目标，少量干扰，目标在开头/结尾|可少量保留，不宜过多|
|Medium|2-4 个目标，有相似干扰，需要找第 N 个|主力训练数据|
|Hard|8+ 个目标，目标在中部/尾部，候选相似|高价值，建议保留|
|Very Hard|128K+，多轮共指，多目标，多干扰，目标在中后部|稀缺高价值，上采样前复核|

---

## 8.8 MRCR 入训建议
|**MRCR-Trainability**|**建议**|
|-|-|
|≥ 85|高质量 MRCR 数据，可优先入训|
|70-85|可作为 MRCR 主力数据|
|55-70|可保留，但建议抽样复核|
|40-55|更像普通长文检索，不宜标为 MRCR 主力|
|< 40|不适合作为 MRCR 数据|

---

# 9. 多文档摘要数据适配评估
## 9.1 任务定位
多文档摘要数据用于训练模型：

* 跨文档阅读；
* 信息去重；
* 多来源整合；
* 摘要压缩；
* 冲突观点处理；
* 长文生成与综合表达。

---

## 9.2 MultiDocSummary-Fit Score
```
MultiDocSummary-Fit =
0.16 × doc_count_score
+ 0.14 × context_length_score
+ 0.16 × topic_coherence_score
+ 0.14 × redundancy_control_score
+ 0.16 × summary_coverage_score
+ 0.12 × structure_boundary_score
+ 0.12 × evidence_traceability_score
```
---

## 9.3 子指标口径
|**子指标**|**计算方式**|**解释**|
|-|-|-|
|`doc_count_score`|每条样本包含文档数|判断是否真正多文档|
|`context_length_score`|总 token 长度|判断是否覆盖长上下文|
|`topic_coherence_score`|关键词 / 标题 / 领域相似度|判断多篇文档是否围绕同一议题|
|`redundancy_control_score`|文档间 n-gram 重复率|避免只是重复拼接|
|`summary_coverage_score`|摘要覆盖文档数 / 文档总数|判断是否只摘要单篇文档|
|`structure_boundary_score`|doc_id、标题、分隔符是否明确|判断文档边界是否清晰|
|`evidence_traceability_score`|是否保留 doc_id / paragraph_id / span|支持后续 grounding / CITE 分析|

---

## 9.4 MultiDocSummary-Trainability
```
MultiDocSummary-Trainability =
0.70 × MultiDocSummary-Fit
+ 0.20 × ESS
+ 0.10 × LCU-General
```
若已有质量标签，可重点查看：

|**质量字段**|**用途**|
|-|-|
|topic_focus|判断是否乱拼|
|structure|判断 doc boundary 是否清晰|
|completeness|判断摘要是否截断|
|info_density|判断是否有有效信息|
|educational_value / professional_depth|判断知识价值和专业深度|

---

## 9.5 多文档摘要难度分级
|**难度**|**数据特征**|
|-|-|
|Easy|2 篇同主题文档，信息基本一致|
|Medium|3-5 篇文档，有重复信息，需要去重|
|Hard|5+ 文档，有冲突信息或互补证据|
|Very Hard|128K+，多来源、多观点、多证据，需要结构化摘要|

---

# 10. 通用长上下文检索数据适配评估
## 10.1 任务定位
用于训练：

* NIAH 式检索；
* 多 needle 检索；
* 中/尾部信息召回；
* 多文档定位；
* 精确抽取。

---

## 10.2 Retrieval-Fit Score
```
Retrieval-Fit =
0.18 × context_length_score
+ 0.18 × needle_count_score
+ 0.18 × position_strategy_score
+ 0.16 × distractor_quality_score
+ 0.15 × query_clarity_score
+ 0.15 × answer_exactness_score
```
---

## 10.3 核心指标
|**指标**|**判断**|
|-|-|
|`needle_count`|目标数量是否明确|
|`filler_count`|是否有足够无关文档填充|
|`target_position`|是否覆盖 head/middle/tail|
|`position_strategy`|uniform/front/back/cluster/random|
|`query_template_diversity`|query 模板是否多样|
|`answer_exactness`|answer 是否可从 needle 中精确抽取|
|`topic_gap`|needle 与 filler 是否不同主题|
|`target_topic_similarity`|needle 之间是否语义相近|

---

# 11. RAG / CITE / Grounding 数据适配评估
## 11.1 任务定位
用于训练：

* 基于上下文回答；
* 证据定位；
* 引用证据；
* 防幻觉；
* 多文档证据整合。

---

## 11.2 RAG-CITE-Fit Score
```
RAG-CITE-Fit =
0.18 × query_answer_alignment_score
+ 0.18 × evidence_availability_score
+ 0.16 × answer_in_context_score
+ 0.16 × hard_negative_score
+ 0.16 × evidence_position_score
+ 0.16 × citation_traceability_score
```
---

## 11.3 核心指标
|**指标**|**计算方式**|
|-|-|
|`query_available`|是否有 query 字段|
|`answer_available`|是否有 answer / candidate 字段|
|`answer_in_context`|answer 是否能在 context 中 substring / 关键词匹配|
|`evidence_span_available`|是否有 evidence span|
|`evidence_doc_count`|证据来自几个文档|
|`hard_negative_count`|相似但不支持答案的干扰文档数量|
|`evidence_position`|evidence 在上下文中的位置|
|`citation_count`|citation 或 doc_id 引用数量|

---

# 12. 长文生成数据适配评估
## 12.1 任务定位
用于训练：

* 长篇创作；
* 长报告；
* 长翻译；
* 长代码生成；
* 章节级稳定输出；
* 防重复、防跑题、防结构塌缩。

旧 1w baseline 中，短输入长输出样本仅 2 条，因此长生成能力是明确数据缺口。

---

## 12.2 LongGeneration-Fit Score
```
LongGeneration-Fit =
0.25 × output_length_score
+ 0.20 × structure_outline_score
+ 0.20 × repetition_control_score
+ 0.15 × constraint_clarity_score
+ 0.10 × topic_consistency_score
+ 0.10 × domain_style_diversity_score
```
---

## 12.3 核心指标
|**指标**|**计算方式**|
|-|-|
|`candidate_tokens`|输出长度|
|`output_length_bucket`|输出长度分桶|
|`outline_available`|是否有大纲/章节|
|`section_count`|章节数|
|`repetition_ngram_ratio`|输出重复率|
|`constraint_count`|prompt 中格式/风格/长度约束数量|
|`topic_consistency`|标题/大纲/正文关键词一致性|
|`style_type`|小说、报告、翻译、代码、总结等|

---

# 13. 长对话数据适配评估
## 13.1 任务定位
用于训练：

* 长程记忆；
* 用户偏好保持；
* 状态更新；
* 跨轮引用；
* 工具参数记忆；
* 多轮任务执行。

---

## 13.2 LongDialogue-Fit Score
```
LongDialogue-Fit =
0.20 × turn_count_score
+ 0.20 × cross_turn_reference_score
+ 0.20 × state_update_score
+ 0.15 × memory_distance_score
+ 0.15 × topic_shift_control_score
+ 0.10 × answer_consistency_score
```
---

## 13.3 核心指标
|**指标**|**计算方式**|
|-|-|
|`turn_count`|user/assistant 轮数|
|`cross_turn_reference_count`|“刚才、之前、上次、那个”等表达数量|
|`state_update_count`|偏好、参数、任务状态变化次数|
|`memory_distance`|被引用信息距当前轮的 token 距离|
|`topic_shift_count`|主题切换次数|
|`tool_parameter_reference`|是否引用历史工具参数|
|`role_consistency`|角色/任务是否前后一致|

---

# 14. 入训决策规则
## 14.1 双分数决策
最终不建议只看一个总分，而是同时看：

```
ESS：整体能不能入训
Task Fit / Trainability：适合训练什么任务
```
|**ESS**|**Task Trainability**|**建议动作**|
|-|-|-|
|高|高|优先入训，可考虑上采样|
|高|低|作为通用长文保留，不作为该任务主力|
|中|高|稀缺任务数据，抽样复核后保留|
|中|中|正常保留或轻度降权|
|低|高|高风险稀缺样本，必须复核|
|低|低|过滤或严格降权|

---

## 14.2 质量标签辅助决策
|**条件**|**建议**|
|-|-|
|`overall_category = Fatal`|直接过滤|
|`overall_category = Low_Value` 且 Task Fit < 70|降权或过滤|
|`overall_category = Diamond`|清洗池，不直接过滤|
|`overall_category = High_Quality / Gold`|优先保留，但仍需检查重复和来源集中|
|`cleanliness ≤ 2`|清洗后再判断|
|`structure ≤ 2`|结构修复或降权|
|`completeness ≤ 2`|抽样复核是否截断|
|`safety ≤ 2`|严格复核或过滤|

---

## 14.3 建议动作标签
```
Keep
Keep + Review
Clean + Keep
Upsample Candidate
Downsample
Filter
Strict Review
```
---

# 15. 与旧 1w baseline的关键对比模块
这一模块用于新数据出来后，判断它相比旧已入训长文数据是否补齐缺口。

---

## 15.1 旧 baseline 关键事实
|**维度**|**旧 baseline 现象**|
|-|-|
|长度覆盖|32K+ 29.69%，128K+ 4.68%，256K+ 137 条|
|输入输出结构|32K+ 主要是长输入短输出，短输入长输出仅 2 条|
|能力类型|学术论文、多语言、书籍占比较高，长生成、多跳推理、多轮交互不足|
|来源集中度|高长度段来源集中，超长样本主要来自书籍/学术论文/PDF|
|中文长文|中文 32K+ 仅 10.7%，中文网页长文很少|
|合成数据|32K+ 合成数据约 29.9%，但能力覆盖需拆解|
|质量风险|32K+ 高风险样本 18.2%|
|Code 风险|Code 类高风险显著偏高|
|长生成|短输入长输出几乎无覆盖|

---

## 15.2 新旧数据对比表
|**模块**|**指标**|**旧 baseline**|**新数据**|**变化**|**判断**|
|-|-|-|-|-|-|
|长度覆盖|`32k_plus_ratio`|29.69%|待填|Δ|是否保持长文覆盖|
|长度覆盖|`128k_plus_ratio`|4.68%|待填|Δ|是否补超长窗口|
|长度覆盖|`256k_plus_count`|137|待填|Δ|是否补稀缺超长|
|输入输出|`long_input_short_output_ratio`|99.9% of 32K+|待填|Δ|是否仍偏阅读/检索|
|输入输出|`short_input_long_output_count`|2|待填|Δ|是否补长生成|
|总体价值|`LCU-General mean`|待填|待填|Δ|长文训练价值|
|入训适配|`ESS mean`|待填|待填|Δ|整体入训质量|
|质量风险|`high_risk_ratio`|18.2% of 32K+|待填|Δ|是否更干净|
|来源|`source_top1_share_128k+`|高|待填|Δ|超长来源是否集中|
|中文|`zh_32k_ratio`|10.7%|待填|Δ|是否补中文长文|
|合成|`synthetic_ratio`|29.9% of 32K+|待填|Δ|合成占比是否合理|
|数据类型|`data_type_distribution`|待统计|待填|Δ|是否补对话/专业文档/问答|
|任务能力|`high_score_task_label_ratio`|待统计|待填|Δ|score≥4 能力覆盖|
|MRCR|`MRCR-Trainability mean`|未统计/低|待填|新增|是否补多轮共指|
|多文档摘要|`MultiDocSummary-Trainability mean`|未统计/低|待填|新增|是否补跨文档综合|
|RAG/CITE|`RAG-CITE-Fit mean`|待确认|待填|Δ|是否补 grounding|
|长生成|`LongGeneration-Fit mean`|极低|待填|Δ|是否补长输出|
|合成|`hybrid_synthetic_ratio`|待统计|待填|Δ||
|合成|`synthetic_pipeline_type_distribution`|待统计|待填|Δ||



---

# 16. 新数据报告建议结构
后续写新数据分析报告时，建议按下面结构：

```md
# 新批次长文数据入训适配性分析报告

## 1. 分析目标与数据口径
## 2. Executive Summary
## 3. General 数据画像
### 3.1 长度覆盖
### 3.2 输入/输出结构
### 3.3 数据类型覆盖
### 3.4 任务能力覆盖
### 3.5 来源多样性与集中度
### 3.6 语种与领域分布
### 3.7 合成数据占比与合成类型
### 3.8 规则质量风险
### 3.9 数据质量标签辅助分析
### 3.10 LCU-General / LCU-Enhanced
### 3.11 ESS 入训适配分

## 4. 特定任务适配分析
### 4.1 MRCR-like 数据适配
### 4.2 多文档摘要数据适配
### 4.3 通用长上下文检索数据适配
### 4.4 RAG / CITE 数据适配
### 4.5 长文生成数据适配
### 4.6 长对话数据适配

## 5. 与旧 1w 已入训 baseline 对比
## 6. 入训建议
### 6.1 优先入训
### 6.2 清洗后入训
### 6.3 降权入训
### 6.4 建议过滤

## 7. 后续验证方向
```
---

# 17. 结论模板
等每套新数据出来后，可按照如下模版出结论：

```md
## 结论

本批数据整体 ESS 为 XX，LCU-General 为 XX，说明其整体入训质量处于【高 / 中 / 低】水平，长文训练价值处于【高 / 中 / 低】水平。

从 general 数据画像看：
- 长度覆盖方面，32K+ 占比为 XX，128K+ 占比为 XX，256K+ 数量为 XX；
- 输入输出结构方面，主要类型为【long_input_short_output / both_long / short_input_long_output】；
- 来源方面，Top1 来源占比为 XX，Top3 来源占比为 XX；
- 质量风险方面，高风险比例为 XX，主要风险为【高重复 / 模板化 / HTML / 乱码 / 低信息密度】。

从任务适配看：
- MRCR-Trainability = XX，说明该批数据【适合 / 不适合】用于多轮共指与顺序定位训练；
- MultiDocSummary-Trainability = XX，说明该批数据【适合 / 不适合】用于多文档摘要训练；
- Retrieval-Fit = XX，说明该批数据【适合 / 不适合】用于长上下文检索训练；
- RAG-CITE-Fit = XX，说明该批数据【适合 / 不适合】用于 grounding / 引用问答训练；
- LongGeneration-Fit = XX，说明该批数据【适合 / 不适合】用于长文生成训练。

与旧 1w baseline 相比，本批数据主要补充了【xxx】方向，改善了【xxx】问题，但仍存在【xxx】风险。

建议：
1. 对 ESS ≥ 70 且 Task Trainability ≥ 70 的样本优先入训；
2. 对 Diamond / 高任务分但低卫生质量样本进入清洗池；
3. 对 Fatal / Low_Value / 高重复 / 严重乱码样本过滤或降权；
4. 对 128K+、256K+ 稀缺样本不直接过滤，应抽样复核后决定；
5. 对 MRCR、多文档摘要等当前重点任务，优先保留高 Task Fit 且 answer/evidence 可追溯的样本。
```
