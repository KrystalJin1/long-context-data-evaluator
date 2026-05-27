
# 附录：变量与字段依赖说明
## A. 整体字段需求总览
这一表先列出整个分析计划最需要的数据字段。后续所有指标基本都是从这些字段中推导出来的。

|**字段类别**|**字段名**|**含义**|**是否必须**|**来源**|**是否需要 LLM**|
|-|-|-|-|-|-|
|样本标识|`sample_id` / `data_id`|单条样本唯一 ID|建议必须|原始数据|否|
|主文本|`text`|content-only 预训练文本|必须至少有一种文本字段|原始数据|否|
|输入|`prompt` / `context`|模型输入内容|条件必须|原始数据|否|
|输出|`candidate` / `answer` / `response`|模型输出、答案或摘要|条件必须|原始数据|否|
|来源|`source` / `dataset_name`|数据来源|建议必须|原始数据 / metadata|否|
|语种|`language`|中文、英文、多语言等|可原始提供，也可检测|metadata / 规则检测|否|
|领域|`domain` / `topic`|金融、医疗、法律、科技等|可选|metadata / 规则 / 标签|可选|
|数据类型|`data_type`|新闻、百科、学术、问答、对话等|可选增强|标签 / metadata|可选|
|合成类型|`source_type` / `synthetic_type`|natural / crawled / synthetic / hybrid|建议|metadata / 规则判定|否|
|pipeline 类型|`pipeline_type`|MRCR、多文档摘要、RAG 等|条件必须|metadata / 规则判定|否|
|文档边界|`doc_id` / `<doc>` / `document_id`|多文档边界|多文档任务必须|原始字段 / 解析|否|
|对话边界|`turn_id` / `role` / user-assistant|多轮对话结构|MRCR / 长对话必须|原始字段 / 解析|否|
|目标字段|`target` / `needle`|需要召回的目标内容|MRCR / 检索任务必须|pipeline 字段|否|
|目标序号|`target_index`|第 N 个目标|MRCR 必须|pipeline 字段 / 规则抽取|否|
|证据字段|`evidence` / `evidence_span`|支持答案的证据|RAG/CITE 推荐|pipeline 字段|可选|
|引用字段|`citation` / `paragraph_id`|答案引用位置|CITE 推荐|pipeline 字段|可选|
|质量标签|`quality_labels`|15 维质量标签|可选增强|打标代码|是|
|任务能力标签|`task_labels`|level1/2/3 + score|可选增强|打标代码|是|

---

## B. 通用派生变量表
这些是所有数据批次都建议先计算的变量，主要用于 General 数据画像、LCU 和 ESS。

|**变量名**|**含义**|**计算方法**|**依赖字段**|**可得性**|**是否需要 LLM**|
|-|-|-|-|-|-|
|`total_tokens`|样本总长度|tokenizer(text/prompt+candidate)|`text` 或 `prompt/candidate`|必算|否|
|`prompt_tokens`|输入长度|tokenizer(prompt/context)|`prompt` / `context`|条件必算|否|
|`candidate_tokens`|输出长度|tokenizer(candidate/answer)|`candidate` / `answer`|条件必算|否|
|`length_bucket`|长度分桶|按 0-4K、4-8K、…、256K+ 分桶|`total_tokens`|必算|否|
|`candidate_empty_flag`|输出是否为空|candidate 为空或极短|`candidate`|条件可算|否|
|`io_type`|输入输出结构类型|根据 prompt/candidate 长度分类|`prompt_tokens`、`candidate_tokens`|条件可算|否|
|`source_type`|自然/清洗/合成/混合|metadata 优先，规则兜底|`source`、`pipeline_type`|条件可算|否|
|`language`|语种|原字段优先，否则 langdetect|`text` / `language`|可算|否|
|`domain`|领域|metadata / 关键词 / 标签|`domain`、`topic`、`text`|条件可算|可选|
|`data_type`|文本类型|原字段 / 标签 / 规则|`data_type`、`text`|条件可算|可选|
|`pipeline_type`|合成 pipeline 类型|metadata / 文件名 / 结构规则|`source`、`dataset_name`、结构字段|条件可算|否|
|`source_traceable_flag`|来源是否可追溯|是否有 source/doc_id/sample_id|`source`、`doc_id`、`sample_id`|可算|否|

---

## C. 数据集级统计变量表
这些不是单条样本分数，而是把单条样本聚合到整个数据集后得到的统计指标。

|**变量名**|**含义**|**计算方法**|**依赖变量**|**是否需要 LLM**|
|-|-|-|-|-|
|`sample_count`|样本总数|count(samples)|`sample_id`|否|
|`32k_plus_ratio`|32K+ 占比|32K+ 样本数 / 总样本数|`total_tokens`|否|
|`64k_plus_ratio`|64K+ 占比|64K+ 样本数 / 总样本数|`total_tokens`|否|
|`128k_plus_ratio`|128K+ 占比|128K+ 样本数 / 总样本数|`total_tokens`|否|
|`256k_plus_count`|256K+ 数量|count(total_tokens ≥ 256K)|`total_tokens`|否|
|`candidate_empty_ratio`|candidate 为空比例|空 candidate 样本数 / 总样本数|`candidate_empty_flag`|否|
|`long_input_short_output_ratio`|长输入短输出比例|对应 io_type 占比|`io_type`|否|
|`short_input_long_output_ratio`|短输入长输出比例|对应 io_type 占比|`io_type`|否|
|`both_long_ratio`|输入输出都长比例|对应 io_type 占比|`io_type`|否|
|`source_top1_share`|最大来源占比|top1 source count / total|`source`|否|
|`source_top3_share`|前 3 来源占比|top3 source count / total|`source`|否|
|`source_hhi`|来源集中度|sum(source_share²)|`source`|否|
|`language_distribution`|语种分布|按 language groupby|`language`|否|
|`domain_distribution`|领域分布|按 domain groupby|`domain`|否|
|`synthetic_ratio`|纯合成占比|synthetic 样本数 / 总数|`source_type`|否|
|`hybrid_synthetic_ratio`|混合合成占比|hybrid_synthetic 样本数 / 总数|`source_type`|否|
|`unknown_type_ratio`|未知类型占比|unknown 样本数 / 总数|`source_type`|否|
|`high_risk_ratio`|高风险样本占比|high_risk 样本数 / 总数|risk flags|否|
|`LCU_mean / P50 / P90`|LCU 分布|聚合样本 LCU|`LCU-General`|否|
|`ESS_mean / P50 / P90`|ESS 分布|聚合样本 ESS|`ESS`|否|
|`MRCR_like_ratio`|MRCR-like 样本占比|MRCR-Fit 可算且高于阈值|`MRCR-Fit`|否|
|`MultiDocSummary_like_ratio`|多文档摘要样本占比|MultiDocSummary-Fit 可算且高于阈值|`MultiDocSummary-Fit`|否|

---

## D. 质量风险变量表
这些主要用于 `Rule Quality Score`、`Information Density`、`risk_penalty` 和 high-risk flag。

|**变量名**|**含义**|**计算方法**|**依赖字段**|**是否需要 LLM**|
|-|-|-|-|-|
|`ngram_repetition_ratio`|n-gram 重复率|重复 n-gram 数 / 总 n-gram 数|`text`|否|
|`duplicate_line_ratio`|重复行/段落比例|重复行或段落数 / 总行段数|`text`|否|
|`html_tag_ratio`|HTML 标签比例|HTML 标签字符数 / 总字符数|`text`|否|
|`html_noise_ratio`|HTML/URL/导航噪声比例|HTML、URL、导航、页脚等占比|`text`|否|
|`url_ratio`|URL 比例|URL 字符数 / 总字符数|`text`|否|
|`weird_char_ratio`|异常字符比例|异常字符数 / 总字符数|`text`|否|
|`template_ratio`|模板化比例|重复模板片段 / 总样本或总文本|`prompt`、`text`|否|
|`info_density_ratio`|信息密度|1 - 重复/HTML/乱码/模板惩罚|上述风险变量|否|
|`high_repetition_flag`|高重复风险|repetition 超阈值|`ngram_repetition_ratio`|否|
|`template_like_flag`|模板化风险|模板占比超阈值|`template_ratio`|否|
|`html_noise_flag`|HTML 噪声风险|HTML/URL 超阈值|`html_noise_ratio`|否|
|`garbled_text_flag`|乱码风险|weird_char 超阈值|`weird_char_ratio`|否|
|`low_information_density_flag`|低信息密度风险|info_density < 阈值|`info_density_ratio`|否|
|`metadata_missing_flag`|元数据缺失|source/doc_id/language 缺失|metadata|否|
|`structure_broken_flag`|结构破碎|段落/边界异常|`text`、boundary 字段|否|
|`high_risk_flag`|综合高风险|任一严重风险触发|risk flags|否|

---

## E. LCU 变量依赖表
LCU 用于判断单条样本是否具备长文训练价值。你的文档里 LCU 主要由长度、有效上下文、信息密度、结构和长程信号组成。

|**LCU 子分数**|**需要的变量**|**子变量怎么计算**|**是否可得**|**是否需要 LLM**|
|-|-|-|-|-|
|`length_score`|`total_tokens`|tokenizer 后按长度分桶映射到 0-100|必算|否|
|`effective_context_score`|`prompt_tokens`、`candidate_tokens`、`io_type`|根据 content_only / long_input_short_output / both_long 等分类打分|条件可算|否|
|`information_density_score`|`info_density_ratio`|`info_density_ratio × 100`|必算|否|
|`structure_score`|段落、标题、doc boundary、turn boundary、QA/evidence 字段|每类结构清晰 +20，最高 100|部分可算|否|
|`long_context_signal_score`|target/evidence 位置、实体复现、跨段引用、多文档、多目标|每类信号加权，最高 100|条件可算|否为主|
|`risk_penalty`|风险 flags|高重复 10、模板化 8、HTML 8、乱码 12 等累加，封顶 30|必算|否|
|`score_coverage`|可计算子项数量|可计算权重和 / 总权重|必算|否|
|`LCU-General`|以上子分数|可计算项重新归一化后减 risk_penalty|条件可算|否|

建议在代码里实现成：

```
Raw Score = Σ(weight_i × score_i) / Σ(weight_i for available scores)LCU-General = max(0, min(100, Raw Score - risk_penalty))
```
如果没有 target / evidence / doc boundary，`long_context_signal_score` 可以标记为 N/A，不要强行赋 0。

---

## F. ESS 与 Rule Quality Score 变量依赖表
ESS 判断单条样本是否适合入训。它比 LCU 更偏“基础质量 + 入训安全”。

|**分数**|**需要的变量**|**计算方式**|**是否可得**|**是否需要 LLM**|
|-|-|-|-|-|
|`cleanliness_rule_score`|`html_noise_ratio`、`url_ratio`、`weird_char_ratio`|噪声越低分越高|基本可算|否|
|`dedup_repetition_score`|`ngram_repetition_ratio`、`duplicate_line_ratio`|重复越低分越高|可算|否|
|`structure_rule_score`|段落、标题、doc boundary、turn boundary|结构越清晰分越高|部分可算|否|
|`completeness_rule_score`|空字段、截断、answer/evidence/target 缺失|完整度越高分越高|部分可算|否|
|`information_density_score`|`info_density_ratio`|`info_density_ratio × 100`|可算|否|
|`metadata_score`|source、language、domain、doc_id、synthetic_type|元数据越完整分越高|取决于字段|否|
|`Rule Quality Score`|上述 6 项|加权平均|可算|否|
|`Quality Label Score`|15 维质量标签 + overall category|标签增强分|可选|是|
|`ESS`|Rule Quality + LCU + Quality Label|无标签时用规则版，有标签时加入 15% 标签分|可算|可选|

你的 ESS 公式里质量标签只占辅助权重，这个逻辑比较稳，因为你文档也强调新数据原始字段默认不包含这些标签，后续如果通过组内打标代码生成，再作为增强字段接入。

---

## G. MRCR-Fit 变量依赖表
MRCR-Fit 只对具备 MRCR 结构的数据计算，不对所有样本强行计算。

|**MRCR 变量**|**含义**|**怎么找 / 怎么算**|**依赖字段**|**是否需要 LLM**|
|-|-|-|-|-|
|`context_length_score`|上下文长度分|根据 `total_tokens` 分桶|`total_tokens`|否|
|`turn_count`|对话/问答轮数|统计 user/assistant 或 turn_id|`role`、`turn_id`、文本分隔符|否|
|`turn_structure_score`|多轮结构分|轮数越多、边界越清晰分越高|`turn_count`、`role`|否|
|`target_index`|目标是第几个|字段读取或从“第 N 个”抽取|`target_index` / 文本规则|否|
|`target_candidate_count`|可选目标数量|统计 needle / answer / 段落候选数|`needle`、`target`、doc/turn 结构|否|
|`target_candidate_score`|候选目标分|候选越多且 target_index 明确越高|上两项|否|
|`distractor_count`|干扰项数量|filler 或相似候选数量|`filler`、候选列表|否|
|`distractor_score`|抗干扰分|干扰项越相似、越多越高|`distractor_count`、相似度|可选|
|`target_start_token`|目标起始 token|answer/target/evidence 在 context 中的位置|`target`、`answer`、`context`|否|
|`target_position`|目标相对位置|`target_start_token / total_tokens`|上一项|否|
|`target_position_score`|位置分|中部/尾部/均衡分布更高|`target_position`|否|
|`coreference_marker_count`|共指表达数量|匹配“第 N 个 / 刚才那个 / 前面那篇”等|`prompt` / `query`|否|
|`coreference_marker_score`|共指信号分|共指表达越明确越高|上一项|否|
|`answer_extractability_score`|答案可追溯性|answer 是否能在 context / turn 中匹配|`answer`、`context`、`turn_id`|否|
|`MRCR-Fit`|MRCR 结构适配分|多维加权|上述变量|否|
|`MRCR-Trainability`|MRCR 入训适配分|`0.70 × MRCR-Fit + 0.20 × ESS + 0.10 × LCU`|MRCR-Fit、ESS、LCU|否|

MRCR 的核心是多轮历史检索、共指消解、第 N 个目标定位、多目标区分和相似干扰项排除，你在文档里已经把它作为当前组内最优先的专项方向。

---

## H. 多文档摘要 Fit 变量依赖表
|变量|含义|怎么找 / 怎么算|依赖字段|是否需要 LLM|
|-|-|-|-|-|
|`doc_count`|文档数量|统计 doc_id / `<doc>` / Document 分隔|`doc_id`、doc boundary|否|
|`doc_count_score`|文档数量分|2 篇起步，3-10 篇更高|`doc_count`|否|
|`total_context_length`|总上下文长度|tokenizer 统计 docs 总长度|docs/text|否|
|`context_length_score`|长度分|按长度分桶|`total_context_length`|否|
|`topic_coherence_score`|多文档主题一致性|标题/关键词/领域相似度|docs、title、domain|可选|
|`redundancy_ratio`|文档间重复程度|n-gram overlap / MinHash|docs|否|
|`redundancy_control_score`|去重质量分|重复越低分越高|`redundancy_ratio`|否|
|`summary`|摘要输出|candidate / summary 字段|`candidate` / `summary`|否|
|`summary_coverage_ratio`|摘要覆盖比例|covered_doc_count / doc_count|docs + summary|可选|
|`summary_coverage_score`|摘要覆盖分|覆盖越多文档越高|上一项|可选|
|`structure_boundary_score`|文档边界清晰度|doc_id、title、separator 是否明确|doc boundary|否|
|`evidence_traceability_score`|证据可追溯性|是否有 doc_id / paragraph_id / span|evidence metadata|否|
|`MultiDocSummary-Fit`|多文档摘要适配分|多维加权|上述变量|否为主|
|`MultiDocSummary-Trainability`|多文档摘要入训适配分|`0.70 × Fit + 0.20 × ESS + 0.10 × LCU`|Fit、ESS、LCU|否|

`summary_coverage_score` 如果不用 LLM，可以先做弱规则版：抽取每篇 doc 的关键词/实体，看 summary 是否覆盖多个 doc。更准确版本再用 LLM judge。

---

## I. Retrieval-Fit 变量依赖表
|**变量**|**含义**|**怎么找 / 怎么算**|**依赖字段**|**是否需要 LLM**|
|-|-|-|-|-|
|`needle_count`|目标 needle 数量|统计 target / needle 字段|needle/target|否|
|`needle_count_score`|needle 数量分|2/4/8 个目标分层|`needle_count`|否|
|`filler_count`|filler 数量|统计非目标填充文档|filler 字段 / doc 列表|否|
|`position_strategy`|needle 插入策略|uniform/front/back/random/cluster|pipeline metadata|否|
|`position_strategy_score`|位置策略分|覆盖中部/多位置更高|`target_position` / strategy|否|
|`distractor_quality_score`|干扰质量|filler 是否与 needle 构成干扰|filler + needle similarity|可选|
|`query_clarity_score`|query 清晰度|是否明确询问哪个目标|query/prompt|否|
|`answer_exactness_score`|answer 精确性|answer 是否可从 needle 中精确抽取|answer + needle/context|否|
|`Retrieval-Fit`|长上下文检索适配分|多维加权|上述变量|否为主|

---

## J. RAG-CITE-Fit 变量依赖表
|**变量**|**含义**|**怎么找 / 怎么算**|**依赖字段**|**是否需要 LLM**|
|-|-|-|-|-|
|`query_available`|是否有 query|query 字段是否存在|query/prompt|否|
|`answer_available`|是否有 answer|answer/candidate 是否存在|answer/candidate|否|
|`query_answer_alignment_score`|QA 对齐分|query 和 answer 是否字段齐全、关键词相关|query + answer|可选|
|`answer_in_context`|答案是否在上下文中|substring / fuzzy match|answer + context|否|
|`answer_in_context_score`|答案可定位分|完整匹配/部分匹配/模糊匹配打分|上一项|否|
|`evidence_available`|是否有 evidence|evidence/span/doc_id 是否存在|evidence metadata|否|
|`evidence_position`|证据位置|evidence_start_token / total_tokens|evidence + context|否|
|`evidence_doc_count`|证据文档数|支持答案的 doc 数量|evidence doc_id|否|
|`hard_negative_count`|hard negative 数量|相似但不支持答案的文档数量|negative docs|可选|
|`citation_count`|引用数量|citation/doc_id 引用数量|citation 字段|否|
|`citation_traceability_score`|引用可追溯性|citation 是否能定位到 doc/span|citation + doc_id|否|
|`RAG-CITE-Fit`|RAG/CITE 适配分|多维加权|上述变量|否为主|
|`answer_correctness`|答案正确性|有标准答案用 EM/F1；无标准答案需 judge|reference/context/answer|可能需要|

这里要注意：`answer_in_context` 不等于 `answer_correctness`。前者只看答案能否在上下文中找到，后者要判断是否真的回答了问题；没有标准答案时，通常需要 LLM judge 或人工抽样复核。

---

## K. LongGeneration-Fit 变量依赖表
|**变量**|**含义**|**怎么找 / 怎么算**|**依赖字段**|**是否需要 LLM**|
|-|-|-|-|-|
|`output_length_score`|输出长度分|根据 `candidate_tokens` 分桶|candidate/response|否|
|`output_length_bucket`|输出长度桶|0-2K、2-8K、8K+ 等|`candidate_tokens`|否|
|`outline_available`|是否有大纲|匹配 outline/章节/标题结构|prompt/candidate|否|
|`section_count`|章节数量|统计标题/编号/章节|candidate|否|
|`structure_outline_score`|结构大纲分|大纲和章节越清晰越高|outline/section|否|
|`repetition_ngram_ratio`|输出重复率|输出 n-gram 重复|candidate|否|
|`repetition_control_score`|重复控制分|重复越低越高|上一项|否|
|`constraint_count`|约束数量|prompt 中格式/风格/长度要求数量|prompt|否|
|`constraint_clarity_score`|约束清晰度|约束越明确越高|prompt|否|
|`topic_consistency_score`|主题一致性|标题/大纲/正文关键词一致性|prompt + candidate|可选|
|`style_type`|风格类型|小说、报告、翻译、代码等|metadata / 规则|可选|
|`LongGeneration-Fit`|长生成适配分|多维加权|上述变量|否为主|

---

## L. LongDialogue-Fit 变量依赖表
|**变量**|**含义**|**怎么找 / 怎么算**|**依赖字段**|**是否需要 LLM**|
|-|-|-|-|-|
|`turn_count`|对话轮数|user/assistant 成对统计|role/turn_id/text|否|
|`cross_turn_reference_count`|跨轮引用次数|匹配“刚才/之前/上次/那个”等|conversation text|否|
|`state_update_count`|状态更新次数|偏好/参数/任务状态变化|text / metadata|可选|
|`memory_distance`|被引用信息距离|引用目标到当前轮的 token 距离|target/turn_id|否|
|`topic_shift_count`|主题切换次数|话题关键词/metadata 变化|turns|可选|
|`tool_parameter_reference`|是否引用历史工具参数|参数名复现/工具调用字段|tool metadata|否|
|`role_consistency`|角色一致性|角色/persona 是否前后一致|dialogue text|可选|
|`LongDialogue-Fit`|长对话适配分|多维加权|上述变量|否为主|

---

## M. 可选质量标签变量表
如果后续用打标代码生成质量标签，可以接入以下变量；没有标签时不影响主流程。

|**标签变量**|**含义**|**用在哪里**|**是否需要 LLM**|
|-|-|-|-|
|`cleanliness`|文本干净程度|Quality Label Score / ESS|是|
|`structure`|结构清晰程度|Quality Label Score / ESS|是|
|`completeness`|内容完整程度|Quality Label Score / ESS|是|
|`fluency`|表达流畅程度|Quality Label Score|是|
|`safety`|安全合规|Quality Label Score|是|
|`topic_focus`|主题集中度|Quality Label Score / Summary / MRCR|是|
|`qa_match`|问答匹配|RAG / MRCR / ESS|是|
|`logic_reasoning`|逻辑推理价值|Quality Label Score|是|
|`info_density`|信息密度标签|Quality Label Score|是|
|`educational_value`|教育价值|Value Signal Score|是|
|`professional_depth`|专业深度|Value Signal Score|是|
|`artistic_value`|文学/审美价值|Value Signal Score|是|
|`persuasion`|说服力|Value Signal Score|是|
|`entertainment`|娱乐性|Value Signal Score|是|
|`emotional_intelligence`|情绪理解|Value Signal Score|是|
|`overall_category`|Fatal/Low/Gold 等等级|ESS 决策辅助|是|

---

## N. 开发优先级建议表
|**优先级**|**指标 / 变量**|**原因**|**是否需要 LLM**|
|-|-|-|-|
|P0|`total_tokens`、`length_bucket`|所有分析基础|否|
|P0|`prompt_tokens`、`candidate_tokens`、`io_type`|判断读长文还是写长文|否|
|P0|`source`、`language`、`source_type`|数据画像和新旧对比必需|否|
|P0|`ngram_repetition_ratio`、`html_noise_ratio`、`weird_char_ratio`|基础质量风险|否|
|P0|`info_density_ratio`|LCU / ESS 核心变量|否|
|P0|`LCU-General`|单样本长文训练价值分|否|
|P0|`Rule Quality Score`、`ESS`|入训适配判断|否|
|P0|`MRCR-Fit` 基础版|组内当前重点任务|否|
|P0|`MultiDocSummary-Fit` 基础版|组内当前重点任务|否|
|P1|`answer_in_context`、`target_position`|RAG / MRCR / 检索重要字段|否|
|P1|`summary_coverage` 弱规则版|多文档摘要质量判断|否|
|P1|`RAG-CITE-Fit`|grounding 相关|否为主|
|P1|`LongDialogue-Fit`|后续长对话方向|否为主|
|P2|`answer_correctness`|无标准答案时较难|可能需要|
|P2|`LLM judge summary coverage`|比规则更准但成本高|是|
|P2|15 维质量标签|可作为增强项|是|

---

## 