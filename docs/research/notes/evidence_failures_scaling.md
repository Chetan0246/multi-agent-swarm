# Empirical Evidence on LLM Multi-Agent Systems: When They Win, How They Fail, How They Scale (as of Nov 2026)

Research date: 2026-11-21. All figures were taken from the primary arXiv pages, arXiv HTML full text, or first-party blogs listed inline. "HTML full text" means the arXiv HTML rendering was read. Most numbers come from a summarizing fetch rather than a line-by-line read of each table, so check headline numbers against the PDF before quoting them in public material.

## Q1. When do multi-agent systems (MAS) outperform a single strong agent, and when not?

### Takeaway
MAS help mainly when (a) the task decomposes into independent parallel subtasks (research/breadth search, decomposable financial analysis), (b) the single-agent baseline is weak (below roughly 45% success in Google's study), and (c) the extra agents mostly add samples that a vote or verifier can aggregate. On sequential and tool-heavy tasks, and once the base agent is strong, MAS are neutral to strongly harmful (as low as -70%). Much of what debate gains comes from majority voting, not from the discussion itself.

### Cited Findings

**Google Research / MIT: "Towards a Science of Scaling Agent Systems" (Kim, Gu, Park et al.; arXiv 2512.08296, v1 9 Dec 2025, v3 8 Apr 2026)**
- Controlled study: 260 configurations (v3), 6 agentic benchmarks, 5 architectures (single-agent SAS plus Independent, Centralized, Decentralized, Hybrid MAS), 3 LLM families — [arXiv abstract](https://arxiv.org/abs/2512.08296). The Google blog (28 Jan 2026) describes an earlier version with 180 configurations, so cite the version used — [Google Research blog](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
- Models: GPT-5-nano/mini/5; Gemini 2.0 Flash, 2.5 Flash, 2.5 Pro; Claude Sonnet 3.7, 4, 4.5 (Intelligence Index 42–71). Agent count ranged from 1 to 4 — [arXiv HTML](https://arxiv.org/html/2512.08296)
- Performance change vs. single agent ranges from +80.8% (decomposable financial reasoning, Centralized) to -70.0% (sequential planning, Independent) — [arXiv abstract](https://arxiv.org/abs/2512.08296)
- Per benchmark (MAS vs. SAS) — [arXiv HTML](https://arxiv.org/html/2512.08296):
  - Finance-Agent: Centralized +80.8%, Decentralized +74.5%, Hybrid +73.1%
  - BrowseComp-Plus: Decentralized +9.2%, Centralized +0.2%, Independent about -35%
  - PlanCraft (sequential planning): every MAS degrades. Hybrid -39.1%, Decentralized -41.5%, Centralized -50.3%, Independent -70.0%
  - Workbench: Decentralized +5.6%; Centralized and Hybrid -1.2%
  - SWE-bench Verified: every MAS degrades, from -2.1% (Hybrid) to -14.9% (Independent)
  - Terminal-Bench: Independent +1.7%, Centralized -19.2%
- Mean MAS change across everything was about -0.3%, with large domain-dependent variance. More agents did not reliably help. Turn count grew super-linearly with agent count, and per-agent reasoning became "too thin" beyond 3–4 agents under fixed budgets — [arXiv HTML](https://arxiv.org/html/2512.08296)
- Capability saturation: MAS returns turn negative once single-agent baseline accuracy exceeds about 45% (decision boundary about 0.45) — [arXiv HTML](https://arxiv.org/html/2512.08296). The abstract names the saturation effect but gives no number — [arXiv abstract](https://arxiv.org/abs/2512.08296)
- Predictive model: cross-validated R² = 0.373, or 0.413 with a task-grounded capability metric. It picks the best architecture for 87% of held-out configurations — [arXiv abstract](https://arxiv.org/abs/2512.08296)
- Tool-coordination trade-off: efficiency × tool-count interaction β = -0.096 (95% CI -0.154 to -0.037, p = 0.002). With 4 or fewer tools the effect is negligible; MAS overhead hurts most on tool-heavy tasks — [arXiv HTML](https://arxiv.org/html/2512.08296)
- Recommendations: use a single agent when the baseline is strong, tools are few, or the task is sequential. Use centralized coordination for decomposable analysis where error control matters. Choose architecture by task structure, not by agent count — [arXiv HTML](https://arxiv.org/html/2512.08296); [Google Research blog](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)

**Sampling-and-voting: "More Agents Is All You Need" (Li, Zhang, Yu, Fu, Ye; arXiv 2402.05120, Feb 2024; TMLR)**
- Agent Forest: sample N answers and take a majority vote. Performance scales with N. The gain is orthogonal to other methods and correlates with task difficulty — [arXiv](https://arxiv.org/abs/2402.05120)
- Single query vs. 40-agent ensemble (Table 2) — [arXiv HTML](https://arxiv.org/html/2402.05120):
  - Llama2-13B on GSM8K: 0.35 → 0.59. This beats single Llama2-70B (0.54) and matches it at about 15 agents
  - GPT-3.5-Turbo: GSM8K 0.73 → 0.85, MATH 0.29 → 0.39, MMLU 0.59 → 0.70, HumanEval 0.67 → 0.73
  - The 40-agent GPT-3.5 never surpasses single GPT-4 (GSM8K 0.85 vs. 0.88, MATH 0.39 vs. 0.40, Chess 0.55 vs. 0.65, MMLU 0.70 vs. 0.77, HumanEval 0.73 vs. 0.88)
  - Implication: sampling and voting closes part of a capability gap but does not replace a stronger model
- Relative gains are larger on harder tasks. On MATH: +200% (13B), +120% (70B), +34% (GPT-3.5). On GSM8K: +69%, +37%, +16%. Gains shrink as the base model gets stronger. In synthetic tests, gains taper at extreme difficulty. Combining debate with Llama2 on HumanEval gave 0 accuracy — [arXiv HTML](https://arxiv.org/html/2402.05120)

**Multi-agent debate (MAD) and critiques**
- Du, Li, Torralba, Tenenbaum, Mordatch, "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (arXiv 2305.14325, May 2023). It reports that debate "significantly enhances" math and strategic reasoning and reduces hallucinations. The abstract gives no figures — [arXiv](https://arxiv.org/abs/2305.14325)
- Wang, Wang, Su, Tong, Song, "Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?" (arXiv 2402.18272, Feb 2024). A single LLM with strong prompts achieves "almost the same performance" as the best discussion method across reasoning tasks and backbones. Discussion wins only when the prompt has no demonstrations — [arXiv](https://arxiv.org/abs/2402.18272)
- Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet" (arXiv 2310.01798). Once voting is accounted for, debate's gains look like a self-consistency effect, not self-correction — [arXiv PDF](https://arxiv.org/pdf/2310.01798)
- Zhang et al., "Stop Overvaluing Multi-Agent Debate" (position paper, arXiv 2502.08788, Feb 2025, v3 Jun 2025) — [arXiv](https://arxiv.org/abs/2502.08788)
  - Evaluated 5 MAD methods × 9 benchmarks × 4 models
  - MAD "often fail[s] to outperform" CoT and self-consistency, even when it uses significantly more inference compute
  - Model heterogeneity (different models as debaters) is called a "universal antidote" that consistently improves MAD
- Choi, Zhu, Li, "Debate or Vote" (arXiv 2508.17536; NeurIPS 2025 Spotlight) — [arXiv](https://arxiv.org/abs/2508.17536)
  - Across 7 NLP benchmarks, majority voting alone accounts for most of MAD's gains
  - The authors prove debate induces a martingale over belief trajectories, so "debate alone does not improve expected correctness"
  - Biasing updates toward correction helps
- Bertalanič and Fortuna, "The Cost of Consensus" (arXiv 2605.00914, 29 Apr 2026) — [arXiv](https://arxiv.org/abs/2605.00914)
  - Setup: N=10 homogeneous 7–8B agents (Qwen2.5-7B, Llama-3.1-8B, Ministral-3-8B), 3 rounds, GSM-Hard and MMLU-Hard
  - Sycophantic conformity: modal adoption up to 85.5%
  - Peer rationales overturn previously correct answers up to 70.0% of the time
  - Plurality voting discards correct answers already in the pool, an oracle gap of up to 32.3 percentage points
  - Debate costs 2.1–3.4× the tokens of isolated self-correction (up to 28,631 tokens per problem) for equal or lower accuracy
  - Caveat: small models only, no roles
- Counterpoint: Wunderlich et al., "Multi-Agent Reasoning Improves Compute Efficiency" (arXiv 2605.01566, May 2026; ACL 2026 SRW) — [arXiv](https://arxiv.org/abs/2605.01566)
  - 34 configurations, MMLU-Pro and BBH
  - At matched compute, debate beats self-consistency by +1.3 points and mixture-of-agents by +2.7 points
  - Self-consistency saturates earlier
  - The best gain over CoT (+7.1 points on MMLU-Pro) needs about 20× CoT compute
  - Abstract-level only; no variance reported

**Anthropic Research system (Anthropic Engineering, 13 Jun 2025)**
- An orchestrator-worker MAS (Opus 4 lead, Sonnet 4 subagents) outperformed single-agent Opus 4 by 90.2% on an internal research eval — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- Caveat: the eval started with about 20 queries — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- Anthropic says MAS are a poor fit where all agents must share context or have many interdependencies, including "most coding tasks." LLM agents are not yet good at real-time coordination and delegation — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)

### Inferences
- Strong evidence supports MAS only for breadth-first, parallelizable work. For sequential, stateful work such as coding pipelines and planning, the best controlled evidence (Kim et al.) shows MAS degrading SWE-bench by 2–15% and planning by 39–70%.
- "More agents" gains are mostly ensemble and sampling effects. Voting, or better a deterministic verifier picking among samples, captures them more cheaply than debate.
- A portfolio design rule follows from this evidence (it is not a measured result): default to one agent, and add agents only for independent subtasks or as samplers feeding a verifier.

### Gaps
- Du et al.'s exact numbers (agents, rounds, accuracy deltas) were not retrieved, because the abstract page has none. See the paper body or project page.
- Exact per-benchmark numbers for Wang et al. (2402.18272) and Choi et al. (2508.17536) were not retrieved, only abstract-level claims.
- Whether the 45% threshold holds beyond 1–4 agents and the 9 models tested is unverified. Anthropic's 90.2% figure is not compute-matched and not externally replicated.

## Q2. How do multi-agent systems fail? (MAST and failure attribution)

### Takeaway
Across 7 open-source MAS frameworks and over 1,600 traces, failure rates ranged from 41% to 86.7%. About 44% of failures trace to system design and specification (step repetition, disobeying the task spec, not knowing when to stop), about 32% to inter-agent misalignment, and about 24% to verification and termination. Fixing failures after the fact is itself hard: the best automated attribution identifies the guilty agent only 53.5% of the time and the decisive step 14.2% of the time.

### Cited Findings
- Cemri, Pan, Yang et al. (UC Berkeley; Zaharia, Gonzalez, Stoica, Klein, Keutzer et al.), "Why Do Multi-Agent LLM Systems Fail?" (arXiv 2503.13657; v1 17 Mar 2025, v3 26 Oct 2025) — [arXiv](https://arxiv.org/abs/2503.13657)
  - 7 MAS frameworks, 1,600+ annotated traces (MAST-Data, 1,642 traces in the full text)
  - The taxonomy was built from 150 traces. Human inter-annotator agreement κ = 0.88
  - 14 failure modes in 3 categories
  - Models: GPT-4, Claude 3, Qwen2.5, CodeLlama
  - Dataset and annotator released
- Systems and benchmarks studied, and failure rate range — [arXiv HTML](https://arxiv.org/html/2503.13657):
  - ChatDev, MetaGPT, OpenManus on ProgramDev (GPT-4o)
  - HyperAgent on SWE-Bench Lite (Claude 3.7 Sonnet)
  - AppWorld on Test-C (GPT-4o)
  - AG2/MathChat on GSM-Plus, OlympiadBench, MMLU
  - Magentic-One on GAIA
  - Failure rates ranged from 41% to 86.7%. Per-system values are in Figure 5 and were not extracted
- Failure-mode frequencies (share of failures across 1,642 traces) — [arXiv HTML](https://arxiv.org/html/2503.13657):
  - **FC1, System design issues (sum about 44.2%):** FM-1.1 Disobey task specification 11.8%; FM-1.2 Disobey role specification 1.5%; FM-1.3 Step repetition 15.7%; FM-1.4 Loss of conversation history 2.8%; FM-1.5 Unaware of termination conditions 12.4%
  - **FC2, Inter-agent misalignment (sum about 32.4%):** FM-2.1 Conversation reset 2.2%; FM-2.2 Fail to ask for clarification 6.8%; FM-2.3 Task derailment 7.4%; FM-2.4 Information withholding 0.85%; FM-2.5 Ignored other agent's input 1.9%; FM-2.6 Reasoning-action mismatch 13.2%
  - **FC3, Task verification (sum about 23.5%):** FM-3.1 Premature termination 6.2%; FM-3.2 No or incomplete verification 8.2%; FM-3.3 Incorrect verification 9.1%
  - The category sums are my calculation from the per-mode figures. The paper's own category totals were not captured in the text, so check them against the paper's figure
- Interventions on ChatDev with GPT-4o — [arXiv HTML](https://arxiv.org/html/2503.13657):
  - Better role specification and workflow (CEO gets the final say): +9.4% task success. The fetched text attributes the +9.4% inconsistently between these two changes
  - Adding a high-level objective verification step: +15.6% on ProgramDev
  - The authors note completion rates stayed low after these fixes
- LLM-as-judge annotator (o1, few-shot) vs. humans: κ = 0.77, accuracy 94%. Zero-shot: κ = 0.58. On unseen MAS and benchmarks: κ = 0.79 — [arXiv HTML](https://arxiv.org/html/2503.13657)
- Recommendations — [arXiv HTML](https://arxiv.org/html/2503.13657):
  - Failures stem from system design, not just model limits
  - Use multi-level verification. Superficial checks such as "it compiles" are insufficient; check low-level correctness and high-level objectives
  - Write explicit role and termination specifications
  - Use standardized communication protocols, uncertainty quantification, and memory and state management
  - Prefer simple, modular agents
  - Prompt-level fixes have inconsistent effects, so structural fixes are preferred
- Zhang, Yin et al. (PSU, Duke, Google DeepMind and others), "Which Agent Causes Task Failures and When?" (arXiv 2505.00212; v1 30 Apr 2025, v3 2 Jun 2025; marked camera-ready, reportedly ICML 2025, venue unconfirmed on the page) — [arXiv](https://arxiv.org/abs/2505.00212)
  - Introduces the Who&When dataset of failure logs from 127 LLM MAS
  - Best method: 53.5% agent-level attribution accuracy, 14.2% step-level
  - Reasoning models such as o1 and R1 do not solve it
- Kim et al. measured error amplification at trace level vs. single agent: Independent 17.2×, Decentralized 7.8×, Hybrid 5.1×, Centralized 4.4× — [arXiv HTML](https://arxiv.org/html/2512.08296). The blog attributes the lower centralized figure to the orchestrator acting as a checkpoint — [Google Research blog](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
  - Caveat: the paper's regression found the amplification terms not statistically significant and attributes the performance gaps mainly to efficiency and overhead
- Production failure modes Anthropic observed — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system):
  - Spawning 50 subagents for simple queries
  - Endless searches for nonexistent sources
  - Excessive inter-agent updates distracting other agents
- Debate-specific failures — [arXiv 2605.00914](https://arxiv.org/abs/2605.00914):
  - Sycophantic conformity (up to 85.5%)
  - Correct answers overturned by peers (up to 70%)
  - Consensus discarding correct answers (oracle gap up to 32.3 points)

### Inferences
- About a quarter of MAST failures are verification and termination failures, and about 12% are "unaware of termination." Together with the +15.6% gain from adding objective verification, this supports deterministic verification gates and explicit stop conditions.
- Step-level attribution at 14.2% means post-hoc debugging of multi-agent chats is unreliable. Structured, logged, checkpointed steps make blame tractable.
- Independent agents with no checkpoint amplify errors about 4× more than centralized ones (17.2× vs. 4.4×). This argues that a single orchestrator plus a verifier is safer than peer swarms, though the statistical significance caveat applies.

### Gaps
- Per-system MAST failure rates (Figure 5) and the AG2 case-study numbers were not extracted.
- The Who&When log count and the full list of models tested were not retrieved.
- I found no 2026 replication of MAST frequencies on newer frontier models.

## Q3. Error compounding and long-horizon reliability

### Takeaway
Agent success falls roughly exponentially with task length. METR finds near-100% success on tasks humans finish in under 4 minutes and under 10% on tasks over 4 hours. Ord's constant-hazard model implies the 80%-reliability horizon is about a third of the 50% horizon, and the 99% horizon about 1/70. Horizons double about every 7 months (faster in 2024–25), but high reliability lags well behind 50% capability, so checkpoints and verification are what make long chains viable.

### Cited Findings
- METR (Kwa, West et al., 26 authors), "Measuring AI Ability to Complete Long Software Tasks" (arXiv 2503.14499, v1 18 Mar 2025, v4 10 Jul 2026) — [arXiv](https://arxiv.org/abs/2503.14499)
  - The 50% time horizon has doubled about every 7 months since 2019, possibly accelerating in 2024
  - Claude 3.7 Sonnet's 50% horizon is about 50 minutes
  - Task suites: RE-Bench, HCAST, and 66 novel shorter tasks
  - Improvement is driven mainly by greater reliability and the ability to adapt to mistakes
- METR blog (19 Mar 2025) — [METR](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)
  - A logistic fit of success vs. human task length gives near-100% success under about 4 minutes and under 10% above about 4 hours
  - Agents "struggle with stringing together longer sequences of actions"
  - SWE-Bench Verified shows a doubling time under 3 months
  - Newer "Time Horizon 1.1" results are on a separate page, not reviewed here
- Toby Ord, "Is there a half-life for the success rates of AI agents?" (arXiv 2505.05115, May 2025) — [arXiv](https://arxiv.org/abs/2505.05115v1); [Forethought](https://www.forethought.org/research/is-there-a-half-life-for-the-success-rates-of-ai-agents)
  - A constant per-step hazard rate gives an exponential survival curve, and the 50% horizon equals the agent's half-life
  - Predicted: T80 ≈ T50/3, T99 ≈ T50/70
  - Caveat: fit only on METR's suite
  - A secondhand report says a Weibull fit with a declining hazard (Gus Hamilton) may fit as well, and that Ord agreed. Not verified against a primary source — [EA Forum](https://forum.effectivealtruism.org/posts/qz3xyqCeriFHeTAJs)
- Checkpoint value, measured:
  - Centralized orchestration (an orchestrator acting as checkpoint) held error amplification to 4.4×, vs. 17.2× for independent agents — [Google Research blog](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
  - Adding a verification step gave +15.6% on ChatDev — [arXiv HTML 2503.13657](https://arxiv.org/html/2503.13657)

### Inferences
- Under constant hazard, a chain of n steps each succeeding with probability p succeeds with probability pⁿ. For example, 0.95²⁰ ≈ 0.36 (my arithmetic, illustrative). Adding agents adds handoffs, and so hazard; Kim et al. found turn count grows super-linearly with agent count. Fewer agents and shorter verified segments directly cut compounding.
- Because the 80% horizon is about 1/3 of the 50% horizon, a reliability-focused system should cut work into chunks well below the model's 50% horizon and verify each chunk deterministically.

### Gaps
- Exact METR 80% horizons, task counts, and the 2026 Time Horizon 1.1 numbers were not retrieved.
- I found no primary study that directly measures per-handoff error rates in MAS chains.

## Q4. Compute-matched comparisons: are MAS gains just more tokens?

### Takeaway
Largely yes. In Anthropic's BrowseComp analysis, token usage alone explains 80% of performance variance, and MAS use about 15× the tokens of chat. Google finds single agents about 5× more token-efficient than hybrid MAS (67.7 vs. 13.6 successes per 1K tokens). Several studies find debate does not beat self-consistency at equal compute. The one 2026 counterexample shows only +1.3 to +2.7 point gains.

### Cited Findings
- Anthropic (13 Jun 2025) — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system):
  - On BrowseComp, three factors explain 95% of performance variance. Token usage alone explains 80%; the others are tool-call count and model choice
  - Upgrading to Sonnet 4 gave a larger gain than doubling the token budget on Sonnet 3.7
  - Agents use about 4× the tokens of chat, and MAS about 15×
  - Parallel subagents (3–5) cut research time by up to 90%
  - Anthropic's effort-scaling guidance: 1 agent with 3–10 tool calls for simple facts; 2–4 subagents for comparisons; 10 or more only for complex research
- Kim et al., coordination overhead vs. SAS: Independent 58%, Decentralized 263%, Centralized 285%, Hybrid 515% — [arXiv HTML](https://arxiv.org/html/2512.08296)
  - Successes per 1K tokens: SAS 67.7, Independent 42.4, Decentralized 23.9, Centralized 21.5, Hybrid 13.6
  - Coordination efficiency: SAS 0.466 vs. Hybrid 0.074
- MAD vs. CoT and self-consistency at equal or greater compute:
  - MAD often fails to beat these baselines even with significantly more compute — [arXiv 2502.08788](https://arxiv.org/abs/2502.08788)
  - Voting explains most of MAD's gains — [arXiv 2508.17536](https://arxiv.org/abs/2508.17536)
  - Debate costs 2.1–3.4× the tokens of self-correction for equal or lower accuracy — [arXiv 2605.00914](https://arxiv.org/abs/2605.00914)
- Contrasting result — [arXiv 2605.01566](https://arxiv.org/abs/2605.01566):
  - At matched compute, debate gains +1.3 and mixture-of-agents +2.7 points over self-consistency on MMLU-Pro/BBH
  - The peak +7.1 points over CoT requires about 20× compute
- Sampling-and-voting with 40 GPT-3.5 samples still trails one GPT-4 call on all 5 benchmarks. Model capability can beat more samples — [arXiv HTML 2402.05120](https://arxiv.org/html/2402.05120)
- MacNet (Qian et al., arXiv 2406.07155, ICLR 2025) — [arXiv](https://arxiv.org/abs/2406.07155)
  - Scales DAG-organized collaboration to over 1,000 agents
  - Reports a "collaborative scaling law": performance grows logistically with agent count, and collaborative emergence comes earlier than neural emergence
  - Irregular topologies beat regular ones
  - Benchmark numbers and token costs were not retrieved. Logistic growth implies saturation, and the study is not compute-matched to a single agent

### Inferences
- When reporting MAS results, normalize by tokens or dollars. Without that, a MAS "win" is mostly a budget increase.
- For a cost-efficient portfolio design, the evidence favors:
  1. One capable agent by default
  2. Parallel sampling plus a deterministic verifier, rather than debate, when extra compute helps
  3. Escalating to a stronger model or a human on verifier failure (Anthropic found a model upgrade beat 2× tokens)
  4. Few agents, at most about 3–4, and only for independent subtasks (Kim et al.)

### Gaps
- No primary source compares "escalation to a stronger model" vs. debate at matched cost. This is inferred from Anthropic's model-upgrade finding and from Li et al. (GPT-4 single > 40× GPT-3.5).
- MacNet's quantitative curves and compute were not extracted.
- Anthropic's 80% variance figure is from internal analysis without a published sample size.
