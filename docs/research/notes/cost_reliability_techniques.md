# Cost, Latency and Reliability Techniques for LLM Agent Workloads (state as of 2026-11-21)

Research date: 2026-11-21. Provider docs were fetched live on that date. Paper claims are from arXiv abstract pages fetched the same day unless marked "[not re-fetched]" (those come from well-known abstracts and should be checked before final citation). Model names in the Anthropic/OpenAI docs (Fable, Mythos, Opus 5.5, Sonnet 5.5, Haiku 5.5, GPT-5.6, GPT-6.x) are as shown in those live docs.

Project context: the target project already does multi-task batching per call, prompt caching, effort-level escalation, Batch API, Thompson-sampled batch sizing, circuit breakers, request hedging, AIMD rate limiting and a write-ahead journal. Each section notes the prior art for each of these and where the gaps are.

## 1. Model cascades and routing (FrugalGPT, RouteLLM, AutoMix, Hybrid LLM, 2025-2026 routers)

### Takeaway
Early papers reported large savings: FrugalGPT up to 98%, RouteLLM 35-85% against GPT-4-only, AutoMix more than 50%, Hybrid LLM up to 40% fewer large-model calls. But 2025-2026 benchmarks (RouterBench, RouterEval, LLMRouterBench) find that many routers barely beat a well-chosen single model under unified evaluation. Savings depend heavily on the task mix and the baseline. Escalating effort level within one model, as the project does, is a cheap alternative that sits close to a cascade.

### Cited Findings
- **FrugalGPT** (Lingjiao Chen, Matei Zaharia, James Zou; arXiv 2305.05176, submitted 2023-05-09). It names three cost levers: prompt adaptation, LLM approximation and LLM cascade. A learned cascade "can match the performance of the best individual LLM (e.g. GPT-4) with up to 98% cost reduction or improve the accuracy over GPT-4 by 4% with the same cost." — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176)
- **RouteLLM** (Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous, Stoica; arXiv 2406.18665, v1 2024-06-26, v4 2025-02-23). Routers are trained on human preference data plus data augmentation. The abstract says cost falls "by over 2 times in certain cases" without hurting quality, and routers transfer when the strong/weak models are swapped at test time. — [arXiv 2406.18665](https://arxiv.org/abs/2406.18665)
- **RouteLLM headline numbers** (LMSYS blog, 2024-07-01; GPT-4 Turbo as strong model, Mixtral 8x7B as weak). It reports cost reductions of more than 85% on MT Bench, 45% on MMLU and 35% on GSM8K against GPT-4 only, while reaching 95% of GPT-4 performance.
  - With Arena plus LLM-judge augmentation, the matrix-factorization router needed only 14% GPT-4 calls on MT Bench, about 75% cheaper than a random router.
  - On MMLU, routers trained only on Arena data were about as good as random. Adding about 1,500 gold-labeled samples (under 2% of the data) brought the router to 54% GPT-4 calls.
  - It reports the same quality as the commercial routers Martian and Unify at over 40% lower cost.
  - Without retraining, it generalized to a Claude 3 Opus / Llama 3 8B pair.
  - Source: [LMSYS RouteLLM blog](https://lmsys.org/blog/2024-07-01-routellm/)
- **AutoMix** (Aggarwal, Madaan et al.; arXiv 2310.12963, v1 2023-10-19, v5 2025-01-19; NeurIPS 2024 per the comments field). It uses few-shot self-verification to estimate output reliability, with a POMDP router that escalates to a larger model based on noisy confidence. The abstract says compute cost falls "by over 50% for comparable performance" across 5 LMs and 5 datasets. — [arXiv 2310.12963](https://arxiv.org/abs/2310.12963)
- **Hybrid LLM** (Ding, Mallick, Wang, Sim, Mukherjee, Ruhle, Lakshmanan, Awadallah; arXiv 2404.14618; ICLR 2024). A query-difficulty router allows "up to 40% fewer calls to the large model, with no drop in response quality." — [arXiv 2404.14618](https://arxiv.org/abs/2404.14618)
- **RouterBench** (arXiv 2403.12031, 2024) is a benchmark of more than 405k pre-generated inference outcomes for testing routers offline on dollar cost and quality. A later paper notes that the top 15 routers on RouterBench differ by only 0.23 percentage points in accuracy. This was seen in a search snippet and is not verified in full text. — [RouterBench](https://arxiv.org/html/2403.12031v2)
- **RouterEval** (arXiv 2503.10657, 2025) draws on more than 200M performance records across 12 evaluations and more than 8,500 models. It reports that most routing methods still have significant room for improvement. Search snippet only. — [RouterEval](https://arxiv.org/html/2503.10657v2)
- **LLMRouterBench** (Hao Li et al.; arXiv 2601.07206, 2026-01-12) covers more than 400K instances from 21 datasets and 33 models, with 10 baselines.
  - "Many routing methods exhibit similar performance under unified evaluation."
  - "Several recent approaches, including commercial routers, fail to reliably outperform a simple baseline."
  - "A substantial gap remains to the Oracle, driven primarily by persistent model-recall failures."
  - "Larger ensembles exhibit diminishing returns compared to careful model curation."
  - Source: [arXiv 2601.07206](https://arxiv.org/abs/2601.07206)
- **Router + batch-size co-optimization (RoBatch)** (Xu, Zhao, Xie; arXiv 2605.28268, 2026-05-27).
  - It states the "Route with Batching Problem": jointly choose a model and a batch size per query under a cost budget. The authors prove it NP-hard.
  - A batch-aware utility model separately estimates (a) utility without batching and (b) per-model degradation under batching. A greedy upgrade walks along the cost-utility Pareto frontier.
  - Evaluated on Qwen3 and Gemma3 across 6 benchmarks. It claims a better Pareto frontier than routing-only or batching-only baselines.
  - The abstract gives no numbers. A search snippet says accuracy stays stable for batches under about 8-16 and then drops steeply, worst on Qwen3-4B. This was not verified in full text.
  - Source: [arXiv 2605.28268](https://arxiv.org/abs/2605.28268)
- **LLM routing as contextual dueling bandits** using Feel-Good Thompson Sampling, evaluated on RouterBench and MixInstruct (arXiv 2510.00841, 2025). — [arXiv 2510.00841](https://www.arxiv.org/pdf/2510.00841)

### Inferences
- RoBatch (2605.28268) is the closest prior art to the project's combination of batching and per-batch configuration choice. The project's Thompson-sampled batch size and effort escalation are an online, bandit-style counterpart to RoBatch's offline greedy scheduling. This is worth citing and contrasting.
- The LLMRouterBench result (routers often fail to beat the best single model) supports a simpler single-model, effort-escalation design over multi-provider routing. It also means a fair evaluation must include a "best single config" baseline.
- AutoMix's lesson is that self-verification confidence is noisy and needs a router that models that noise (a POMDP). This maps to confidence-gated effort escalation; see section 4 on calibration.

### Gaps
- I did not fetch per-benchmark savings from RouterBench, RouterEval or LLMRouterBench full texts. Only abstracts and snippets were read.
- I found no measured study of "LLM-as-router" (a prompted cheap model choosing the target) against learned routers on the same benchmark. RouteLLM's causal-LLM router (a fine-tuned LLM classifier) is the nearest data point.
- I found no published study of escalating effort/thinking budget within a single model as a cascade, with measured savings.

## 2. Batch prompting: savings vs accuracy as batch size grows

### Takeaway
Packing several tasks into one call cuts tokens and calls roughly in inverse proportion to batch size: up to 5x at batch size 6 in the original paper, and 9-16% of the calls at batch size 32 in BatchPrompt. For standard models accuracy usually falls as batch size grows, and permutation/voting can recover it. A 2025-2026 study on reasoning models finds the opposite: batching cuts reasoning tokens by 76% while keeping accuracy, because it suppresses overthinking.

### Cited Findings
- **Batch Prompting** (Zhoujun Cheng, Jungo Kasai, Tao Yu; arXiv 2301.08721, v1 2023-01-19, v2 2023-10-24; EMNLP 2023 Industry Track). Token and time cost fall "up to 5x with six samples in batch" (Codex). In theory, inference cost falls almost inversely with batch size under few-shot ICL. Performance is "better or comparable" on 10 datasets, and the benefits also hold for GPT-3.5/GPT-4. Batch size and task complexity affect accuracy. — [arXiv 2301.08721](https://arxiv.org/abs/2301.08721)
- **BatchPrompt** with BPE (Batch Permutation and Ensembling) and SEAS (self-reflection-guided early stopping) (Lin, Diesendruck, Du, Abraham; arXiv 2309.00384, v1 2023-09-01, v3 2024-07-15). At batch size 32 it uses about 9-16% of SinglePrompt's LLM calls.
  - BoolQ: accuracy 90.6% to 90.9% using 27.4% of the tokens.
  - QQP: 87.2% to 88.4% using 18.6% of the tokens.
  - RTE: 91.5% to 91.1% using 30.8% of the tokens.
  - Source: [arXiv 2309.00384](https://arxiv.org/abs/2309.00384)
- **BatchPrompt full text:** "With the increase of batch size, the overall accuracy decreases," which the authors attribute partly to autoregressive conditioning on earlier answers. Accuracy becomes competitive with single prompting when there are more than 5 voting rounds. Seen in a search snippet of the paper PDF. — [BatchPrompt PDF](https://arxiv.org/pdf/2309.00384v2.pdf)
- **Auto-Demo Prompting** (arXiv 2410.01724, 2024) frames the contradiction that few-shot prompting usually helps but batch prompting "often results in performance degradation." It uses earlier outputs in the batch as demonstrations. With batch data selection, batch sizes 16/32 reportedly outperform single prompts on RTE/QQP. Search snippet. — [arXiv 2410.01724](https://arxiv.org/html/2410.01724v1)
- **BatchGEMBA** (MT evaluation; arXiv 2503.02756, 2025). All models lose quality at batch size above 1. GPT-4o-mini stays stable, while GPT-4o falls to "only 2% of its batch size 1 quality at batch size 16." Search snippet. — [arXiv 2503.02756](https://arxiv.org/pdf/2503.02756)
- **Batch prompting suppresses overthinking in reasoning models** (Srivastava et al.; arXiv 2511.04108, v1 2025-11-06, v4 2026-02-20; DeepSeek-R1 and OpenAI-o1 on 13 benchmarks).
  - Moving from batch size 1 to 15 cut reasoning tokens by 76% on average (2,950 to 710 tokens) "while preserving or improving accuracy."
  - The proposed mechanisms are shared-context amortization, pattern induction across items, and suppression of hedging ("wait", "let me double-check").
  - Explicit prompt limits such as "≤100 thinking tokens" fail: models ignore them or lose accuracy.
  - Source: [arXiv 2511.04108](https://arxiv.org/abs/2511.04108)
- **Routing plus batching:** RoBatch (see section 1) models per-model degradation under batching as a recalibration term. — [arXiv 2605.28268](https://arxiv.org/abs/2605.28268)

### Inferences
- The project's Thompson-sampled batch size is a reasonable online answer to the "optimal batch size depends on model and task" result. The evidence says the reward should combine per-item accuracy with per-item cost, because degradation is non-linear and model-specific: stable to about 8-16, then steep, with GPT-4o and GPT-4o-mini behaving very differently.
- For reasoning models with adaptive thinking (current Claude models have thinking always on), the 2511.04108 result suggests batching may save more than its prompt-amortization share alone. This is a testable hypothesis for the project.
- BPE-style permutation voting is a reliability add-on the project may not have: re-run a batch with items shuffled and vote. It costs about k times the tokens, so use it only on low-confidence items.
- Position effects inside a batch, where later answers are conditioned on earlier ones, argue for randomizing item order and checking per-position error rates.

### Gaps
- I found no 2026 study of batch prompting on Claude 5.x-era models specifically.
- I did not fetch the full-text accuracy-vs-batch-size curves for RoBatch or Auto-Demo.

## 3. Provider-side levers: prompt caching, Batch APIs, effort/thinking controls, structured outputs

### Takeaway
As of 2026-11-21, Anthropic charges cache writes at 1.25x (5-minute TTL) or 2x (1-hour TTL) the base input price. Cache reads are 0.1x on most models and 0.05x or 0.025x on the newest. Cached reads do not count toward input-token rate limits, which raises effective throughput. Both Anthropic and OpenAI give 50% off for batch with a 24-hour window, and the batch and caching discounts stack. OpenAI's cache pricing for GPT-5.6 and later now looks similar: a 1.25x write at a 30-minute default TTL, and reads at 0.1x or 0.05x. Anthropic's `effort` parameter (low/medium/high/xhigh/max) is now the main thinking-cost control. Changing top-level effort between requests can invalidate the cache, and per-message effort does not.

### Cited Findings
**Anthropic prompt caching (pricing page, fetched 2026-11-21)**
- Multipliers on base input price:
  - 5-minute cache write: 1.25x.
  - 1-hour cache write: 2x.
  - Cache read/refresh: 0.1x on most models.
  - 0.05x on Claude Opus 5.5 ($0.20/MTok) and Sonnet 5.5 ($0.10/MTok).
  - 0.025x on Fable/Mythos 5.1 and 5.2.
- Break-even is after 1 read for the 5-minute TTL and after 2 reads for the 1-hour TTL.
- "These multipliers stack with other pricing modifiers, including the Batch API discount and data residency."
- Source: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- Two cache modes: "automatic caching" (a single top-level `cache_control` field, with breakpoints managed as conversations grow) and explicit per-block breakpoints. — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- Base prices per MTok, input/output: Opus 5.5 $4/$20; Sonnet 5.5 $2/$10; Haiku 5.5 $0.10/$0.50 for prompts up to 100k tokens and $0.50/$2.50 above that; Fable 5.2 $8/$40; Opus 4.5–4.8 $5/$25; Sonnet 4.5/4.6 $3/$15; Haiku 4.5 $1/$5. — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- Other pricing modifiers:
  - Claude 4.6 and later (except Haiku 5.5) bill the 1M-token context at standard rates, with no long-context premium.
  - US-only `inference_geo` adds a 1.1x multiplier.
  - Fast mode gives "up to 2.5x higher output tokens per second" at premium prices (Opus 5.5 fast: $8/$40) and is not available with the Batch API.
  - Source: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- **Cache-aware rate limits:** "The API's default ITPM rate limits count only uncached input tokens." `cache_read_input_tokens` do not count toward ITPM, while `cache_creation_input_tokens` and `input_tokens` do. Example: with a 2M ITPM limit and an 80% cache hit rate, you can process 10M total input tokens per minute. — [Anthropic rate limits](https://platform.claude.com/docs/en/api/rate-limits)

**Anthropic Message Batches (fetched 2026-11-21)**
- Price: 50% discount on input and output tokens (for example, Opus 5.5 batch is $2/$10, Sonnet 5.5 $1/$5, Haiku 5.5 $0.05/$0.25).
- Limits: a batch holds up to 100,000 requests or 256 MB.
- Timing: "most batches completing within 1 hour." Batches expire if not finished within 24 hours, and expired requests are not billed.
- Results are kept for 29 days.
- Source: [Anthropic batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
- Caching inside batches is best-effort: "Users typically experience cache hit rates ranging from 30 percent to 98 percent." Anthropic recommends the 1-hour cache for batches with shared context, because batches can take longer than 5 minutes. Cache pre-warming (`max_tokens: 0`) is not supported inside a batch. — [Anthropic batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
- Batch rate limits by tier (Start/Build/Scale): 1,000/2,000/4,000 RPM; 200k/300k/500k requests in the processing queue; 100,000 requests per batch. The Batch API "isn't subject to ramp limits." — [Anthropic rate limits](https://platform.claude.com/docs/en/api/rate-limits)

**OpenAI (fetched 2026-11-21)**
- Prompt caching:
  - "Enabled by default for supported OpenAI models."
  - GPT-5.6 and later need a minimum cacheable prefix of 1,024 visible input tokens.
  - Cached input is "discounted up to 95%." Reads cost 0.1x, or 0.05x on GPT-6.2 Astra/Sol and GPT-6.1 Sol/Terra.
  - On GPT-5.6 and later, cache writes cost 1.25x for the default 30-minute TTL and 2.5x for a 12-hour TTL (`prompt_cache_options.ttl`, Responses API only).
  - Older models: `in_memory` lasts about 5-10 minutes of inactivity, up to 1 hour. `24h` retention is the default for non-ZDR organizations.
  - `prompt_cache_key` is a routing hint on older models and a per-customer accounting key on GPT-5.6 and later.
  - Source: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
  - Flag: older public material described OpenAI caching as a free automatic 50% discount with no write premium. The 2026 doc shows a write premium on new models, so do not cite the old numbers.
- Batch API:
  - 50% lower cost.
  - 24-hour completion window, the only option.
  - Up to 50,000 requests and 200 MB per input file.
  - Separate queued-token limits per model that do not count toward standard rate limits.
  - Up to 2,000 batches per hour.
  - Expired batches return completed results and bill only the completed requests. Output files expire after 30 days by default.
  - Source: [OpenAI batch](https://developers.openai.com/api/docs/guides/batch)

**Anthropic effort / thinking (fetched 2026-11-21)**
- Levels are `low`, `medium`, `high`, `xhigh`, `max`, set through `output_config.effort`.
- The default is `high` on most models and `medium` on Opus 5.5 and Haiku 5.5.
- Effort "affects all tokens in the response," including text, tool calls and thinking. "Lower effort also means fewer and terser tool calls."
- "Effort is a behavioral signal, not a strict token budget."
- Source: [Anthropic effort](https://platform.claude.com/docs/en/build-with-claude/effort)
- **Cache interaction:** "changing [top-level effort] between requests can invalidate cached prefixes from earlier turns." Per-message effort, set through a `system` message in `messages` on supported models, "preserves the prompt cache." — [Anthropic effort](https://platform.claude.com/docs/en/build-with-claude/effort)
- `xhigh` is aimed at long-running agentic tasks over 30 minutes "with token budgets in the millions." `max` "adds significant cost for relatively small quality gains" on most workloads and "can lead to overthinking" on structured-output tasks (Opus 4.7 guidance). Anthropic recommends running an effort sweep on your own evals. — [Anthropic effort](https://platform.claude.com/docs/en/build-with-claude/effort)
- Thinking modes:
  - Claude 4.7 and later reject `thinking: {type: "enabled"}` (budget_tokens) and use adaptive thinking plus effort instead.
  - On Opus 5.5 and the Fable/Mythos 5.x models, thinking cannot be disabled.
  - Sonnet 5.5 uses `thinking: {type: "between_tools"}` as its lowest setting.
  - Source: [Anthropic errors](https://platform.claude.com/docs/en/api/errors)
- **Structured outputs:** the newest models (Fable 5.2, Opus 5.5, Sonnet 5.5, ...) do not support forced `tool_choice` (`any`/`tool`). The docs direct users to strict tool use or structured outputs (`output_config.format`) for schema-valid output. Assistant prefill is unsupported on Claude 4.6 and later. — [Anthropic errors](https://platform.claude.com/docs/en/api/errors)

### Inferences
- For the project, the key interaction is that effort escalation through top-level `effort` can break cache hits. On supported models, use per-message effort to escalate while keeping the cached prefix. This is a concrete, cite-able design point.
- Because cache reads are free against ITPM, caching is a throughput lever as well as a cost lever. AIMD limits should be tuned on uncached tokens, not total tokens.
- Batch plus 1-hour cache plus multi-task prompt batching all stack. The theoretical floor on cached input for Opus 5.5 in batch is about $0.10/MTok (0.05x of $2), but the 30-98% batch cache hit rate makes the realized saving workload-dependent. That figure is my calculation from the documented stacking, not stated directly.
- Because prefill is gone and forced tool choice is unsupported on the newest models, batched-output parsing should rely on structured outputs rather than prefill tricks.

### Gaps
- Neither provider's docs publish measured token savings per effort level, such as "low uses X% of high."
- The 30-98% batch cache hit range is Anthropic's only published figure. I found no independent measurement.
- Structured-output latency/cost overhead (for example, grammar compilation on first use) was not checked.

## 4. Inference-time compute: self-consistency, adaptive stopping, confidence escalation, verifiers, calibration

### Takeaway
Self-consistency improves accuracy but multiplies cost. Adaptive and early-stopping variants recover most of that cost: Adaptive-Consistency cuts samples by up to 7.9x with less than 0.1% accuracy loss, and ESC cuts 34-84%. Verbalized LLM confidence is systematically overconfident and only moderately predictive of errors (AUROC about 0.52-0.61 in Xiong et al.). Agreement across samples is a better escalation signal than self-reported confidence.

### Cited Findings
- **Self-consistency** (Wang et al., 2022, arXiv 2203.11171): sample multiple CoT paths and take a majority vote. Reported gains: GSM8K +17.9%, SVAMP +11.0%, AQuA +12.2%, StrategyQA +6.4%, ARC-challenge +3.9%. [not re-fetched; check the abstract] — [arXiv 2203.11171](https://arxiv.org/abs/2203.11171)
- **Adaptive-Consistency** (Aggarwal, Madaan, Yang, Mausam; arXiv 2305.11860; EMNLP 2023) uses a lightweight stopping criterion on the agreement distribution. It "reduces sample budget by up to 7.9 times with an average accuracy drop of less than 0.1%" across 17 reasoning and code datasets with 3 LLMs. — [arXiv 2305.11860](https://arxiv.org/abs/2305.11860)
- **Early-Stopping Self-Consistency (ESC)** (Li et al.; arXiv 2401.10480; ICLR 2024) samples in windows and stops when a window is unanimous. Average samples fell by 33.8% on MATH, 80.1% on GSM8K, 76.8% on StrategyQA, 78.5% on CommonsenseQA, 84.2% on Coin Flip and 67.4% on Last Letters, with "comparable performances." — [arXiv 2401.10480](https://arxiv.org/abs/2401.10480)
- **Verbal confidence calibration** (Xiong, Hu, Lu, Li, Fu, He, Hooi; arXiv 2306.13063; ICLR 2024). LLMs verbalizing confidence "tend to be overconfident." Calibration improves with scale and with consistency across multiple responses. White-box methods beat black-box ones only narrowly ("0.522 to 0.605 in AUROC"). No single method wins, and all struggle on professional-knowledge tasks. — [arXiv 2306.13063](https://arxiv.org/abs/2306.13063)
- **Self-verification as a cascade signal:** AutoMix found few-shot self-verification noisy enough to need a POMDP router. — [arXiv 2310.12963](https://arxiv.org/abs/2310.12963)
- **Explicit token caps vs effort:** reasoning models ignore or are harmed by in-prompt thinking limits such as "≤100 tokens," per 2511.04108. This supports API-level effort controls over prompt-level caps. — [arXiv 2511.04108](https://arxiv.org/abs/2511.04108)

### Inferences
- For confidence-gated effort escalation, the better-supported trigger is disagreement across 2-3 cheap samples (ESC/Adaptive-Consistency style), not the model's self-reported confidence. An AUROC of about 0.6 means self-reported confidence misroutes a large share of items.
- ESC's windowed unanimity rule fits batch-level escalation well: run a batch at low effort twice, and escalate only the items whose answers disagree.

### Gaps
- I did not re-fetch 2025-2026 work on process reward models / LLM-as-judge verifier reliability (for example, judge position and self-preference bias). Its numbers are not included here.
- I found no calibration study for Claude 5.x-era adaptive-thinking models.
- Huang et al. 2023, "LLMs cannot self-correct reasoning yet" (arXiv 2310.01798), is relevant to verifier reliability but was not fetched.

## 5. Distributed-systems reliability applied to LLM APIs

### Takeaway
The classic toolkit is the same, but LLM APIs add specific failure semantics:
- Anthropic 429s come in several kinds: rate limit, ramp limit (`ramp_limit_exceeded`) and spend cap (`enforced_spend_limit_reached`, no `retry-after`, never retry).
- A 529 (overloaded) is a platform-wide error, not a per-tenant one.
- Rate limits use a token bucket that may be enforced over sub-minute windows.
- Ramp limits cap how fast usage can grow: roughly the recent peak plus one step per minute. This is effectively provider-enforced additive increase, which matches the "AI" half of AIMD.

Hedging (Dean & Barroso) cut BigTable p99.9 from 1,800 ms to 74 ms with 2% extra load. Google SRE suggests a retry budget of 3 attempts per request and 10% retries per client, plus client-side adaptive throttling with K=2.

### Cited Findings
**Anthropic error and limit semantics (fetched 2026-11-21)**
- 429 `rate_limit_error` covers rate limits, monthly spend caps and ramp limits.
- The spend-cap 429 "has no `retry-after` header and keeps failing until access resumes." It is identified by `error.details.error_code: enforced_spend_limit_reached`.
- 500 `api_error`: "Retry the request with exponential backoff."
- 504 `timeout_error`: use streaming for long requests.
- 529 `overloaded_error`: "can occur when the API experiences high traffic across all users."
- The SDK automatically retries connection errors, 429s and 5xx with exponential backoff, twice by default, honoring `retry-after`.
- Every response carries a `request-id`.
- Source: [Anthropic errors](https://platform.claude.com/docs/en/api/errors)
- Long requests: use streaming or Batches for anything over about 10 minutes. Idle connections can be dropped by networks. The SDK rejects non-streaming requests expected to exceed 10 minutes and sets TCP keep-alive. Streaming errors can arrive after an HTTP 200 as SSE error events. — [Anthropic errors](https://platform.claude.com/docs/en/api/errors)
- Rate limiting:
  - "The API uses the token bucket algorithm."
  - "a rate of 60 requests per minute (RPM) might be enforced as 1 request per second. Short bursts ... can exceed the limit."
  - Ramp limit: "your organization's highest per-minute usage of the model in the recent past, plus a step. Each minute you can go up to a step above that peak." Recommended handling: "Hold your traffic at a level where requests succeed, then increase it by at most one step per minute," and "Start scheduled or bulk jobs at a low rate and spread requests evenly."
  - Headers: `anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{limit,remaining,reset}`, plus `retry-after`.
  - Start tier for Opus 5.5 and Sonnet 5.5: 4,000 RPM, 2M ITPM, 400k OTPM.
  - Source: [Anthropic rate limits](https://platform.claude.com/docs/en/api/rate-limits)

**Hedging / tail latency**
- Dean & Barroso, "The Tail at Scale," CACM 56(2):74–80, 2013, DOI 10.1145/2408776.2408794.
  - BigTable benchmark: hedging after 10 ms (reading 1,000 keys across 100 servers) cut 99.9th-percentile latency from 1,800 ms to 74 ms while sending only 2% more requests.
  - Recommended hedge trigger: delay the backup request until the first has been outstanding longer than the 95th-percentile expected latency.
  - Sources: secondary summaries [ClickHouse tail latency](https://clickhouse.com/resources/engineering/tail-latency) and [the morning paper](https://blog.acolyer.org/2015/01/15/the-tail-at-scale/). The CACM page returned 403. One summary labels the figure p99 rather than p99.9, so verify against the paper.
- gRPC has native hedging policy support. — [gRPC request hedging](https://grpc.io/docs/guides/request-hedging)

**Google SRE "Handling Overload" (fetched 2026-11-21)**
- Client-side adaptive throttling: track `requests` and `accepts` over 2 minutes, and reject locally with a probability once `requests > K·accepts`. Google "generally prefer[s] the 2x multiplier."
- Per-request retry budget: 3 attempts.
- Per-client retry budget: retry only while retries are under 10% of requests.
- Criticality classes (CRITICAL_PLUS, CRITICAL, SHEDDABLE_PLUS, SHEDDABLE) propagate downstream, and batch jobs default to SHEDDABLE_PLUS.
- Source: [Google SRE book: Handling Overload](https://sre.google/sre-book/handling-overload/)

**Backoff with jitter**
- The AWS Architecture Blog "Exponential Backoff and Jitter" (Marc Brooker, 2015) is the standard reference for full, equal and decorrelated jitter. The page body did not render during fetch, so its quantitative results are not reproduced here. — [AWS blog](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)

**Durable execution for agents**
- Temporal: LLM calls go in Activities, which are retried by a configurable policy. The Workflow function must stay deterministic because LLM responses are not. — [dev.to Temporal for AI agents (2026, practitioner blog)](https://dev.to/jangwook_kim_e31e7291ad98/temporal-for-ai-agents-durable-execution-guide-2026-9hf)
- DBOS checkpoints workflow steps into Postgres as a library, with no separate orchestration server. Restate is journal-based: each step is persisted before proceeding, and any worker can resume. — [developersdigest durable execution](https://www.developersdigest.tech/blog/durable-execution-for-ai-agents); [greenabstracts: durable execution stack for agents](https://greenabstracts.substack.com/p/the-durable-execution-stack-for-agents)
- Pydantic AI officially supports Temporal, DBOS, Prefect and Restate as durable-execution backends. — [Pydantic AI durable execution docs](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/index.md)
- An anecdote: a pipeline without checkpoints restarted from the beginning and re-billed about $19 of already-completed LLM calls. This is a practitioner blog and illustrative only. — [developersdigest](https://www.developersdigest.tech/blog/durable-execution-for-ai-agents)
- **Idempotency:** the Anthropic error docs list `409 conflict_error`, but I found no documented idempotency-key header for Messages requests. Client-side dedup through a journal is therefore the user's responsibility. This is an inference from the absence of such a header in the errors and rate-limit docs fetched. — [Anthropic errors](https://platform.claude.com/docs/en/api/errors)

### Inferences
- The project's AIMD limiter has direct provider backing. Anthropic's ramp-limit guidance ("increase by at most one step per minute") is additive increase, and the token bucket plus sub-minute enforcement argue for smoothing requests rather than bursting.
- Circuit breakers should branch on `error.details.error_code`:
  - `enforced_spend_limit_reached`: open the breaker permanently and alert. Never retry.
  - `ramp_limit_exceeded`: hold the current rate, then additive increase.
  - Plain 429 with `retry-after`: honor it.
  - 529: platform-wide overload. A good trigger for failover or shedding SHEDDABLE work to the Batch API.
- LLM hedging differs from BigTable hedging. Duplicate LLM calls double token cost, and identical prompts may share a cache, which makes the second call cheaper on input but no faster on output. Hedge only latency-critical, short-output calls, after a p95 delay, and cancel the loser. SRE-style retry budgets (10%) bound the cost.
- The write-ahead journal is the same idea as Restate/Temporal/DBOS event histories. It is worth citing them as prior art, and also noting that those systems require deterministic orchestration code with nondeterministic LLM calls isolated as journaled steps.

### Gaps
- No provider publishes 529 frequency or SLOs, and I found no measured data on LLM API tail latency or hedging benefit.
- I found no primary Temporal/DBOS/Restate engineering blog with measured agent-recovery numbers. The sources found were practitioner write-ups. The reported "$300M Series D (Feb 2026)" for Temporal was seen only in a secondary source and is unverified.
- The AWS jitter blog's quantitative comparison was not retrieved.

## 6. Bandits / online learning for LLM configuration selection

### Takeaway
Thompson sampling and dueling bandits are established in 2025 work for selecting prompt strategies and routing queries between models. I found no paper that applies Thompson sampling specifically to choosing batch size for multi-task prompts. That is a possible novelty for the project, with RoBatch (offline greedy) as the closest neighbor.

### Cited Findings
- **OPTS** (arXiv 2503.01163, 2025) adds bandit-based selection of prompt-design strategies to EvoPrompt. "The Thompson sampling-based mechanism achieves the best overall results." — [arXiv 2503.01163](https://arxiv.org/pdf/2503.01163)
- **Prompt Duel Optimizer** (arXiv 2510.13907, 2025) treats label-free prompt optimization as a dueling bandit with LLM-judge pairwise feedback, using Double Thompson Sampling. — [arXiv 2510.13907](https://arxiv.org/pdf/2510.13907v1)
- **LLM routing as contextual dueling bandits** (arXiv 2510.00841, 2025) uses Feel-Good Thompson Sampling and is evaluated on RouterBench and MixInstruct. — [arXiv 2510.00841](https://www.arxiv.org/pdf/2510.00841)
- **Survey:** "Multi-Armed Bandits Meet Large Language Models" (arXiv 2505.13355, 2025) covers bandit methods for LLM prompt selection and fine-tuning. — [arXiv 2505.13355](https://arxiv.org/html/2505.13355v1)
- **RoBatch** (2026) jointly chooses model and batch size under a budget, but offline and greedily, not as a bandit. — [arXiv 2605.28268](https://arxiv.org/abs/2605.28268)

### Inferences
- The project's Thompson-sampled batch sizing can be positioned as "online, bandit-based Route-with-Batching" in contrast to RoBatch's offline Pareto greedy. The reward should be cost-normalized accuracy, since batching degradation is non-stationary across models and tasks (section 2).
- A contextual bandit, conditioned on task type or length, would follow the routing-bandit literature (2510.00841) more closely than a context-free Thompson sampler.

### Gaps
- I found no measured cost savings from bandit-based configuration selection in production LLM workloads.
- I found no work combining bandit batch sizing with effort level as a joint arm space.
