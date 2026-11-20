# Benchmarks, Evaluation Methodology, and Production Practice for LLM Agents and Multi-Agent Systems (as of 2026-11-21)

> Reading note for the report writer: most "current top score" numbers for 2026 came from third-party aggregators (llm-stats, benchlm.ai, steel.dev, vals.ai, llmboard.ai), not from official leaderboards fetched directly. The aggregators disagree with each other, use different harnesses, and mix vendor-reported with independently run results. Treat every 2026 score below as "reported by X on date Y", not as settled fact. Where I fetched a primary source directly, it is marked **(primary, fetched)**. Several model names in aggregator results (e.g. "Claude Fable 5", "Claude Mythos Preview", "GPT-5.5", "GPT-5.6 Sol", "GPT-6 Astra", "Muse Spark 1.1") I could not independently confirm against vendor announcements in this session.

## Agent benchmarks and current state-of-the-art scores

### Takeaway
Headline agent benchmarks are near saturation on vendor-reported numbers (SWE-bench Verified ~95%, GAIA ~92-94%, BrowseComp ~92%, OSWorld-Verified ~86-90%, Terminal-Bench 2.0 ~82%), while harder/cleaner sets (SWE-bench Pro standardized ~61%) and METR's time-horizon metric (~12-17 h at 50% success for the frontier in spring 2026) still separate systems. Rankings depend heavily on harness, budget and who ran the eval, so a single number is never comparable across leaderboards.

### Cited Findings

**SWE-bench Verified (500 human-validated GitHub issues)**
- llm-stats (updated ~2026-11-20) lists "Claude Fable 5" (Anthropic) first at 95.0%, "Claude Mythos Preview" at 93.9%; tracks 124 models — [llm-stats](https://llm-stats.com/benchmarks/swe-bench-verified)
- llmboard.ai (evaluated 2026-11-18) shows the same ordering: 95.0% and 93.9% — [llmboard.ai](https://www.llmboard.ai/benchmarks/swe-bench-verified-581)
- Vals.ai (updated 2026-09-01), which runs every model through a standardized mini-SWE-agent harness, lists Claude Opus 5 at 97.0% and DeepSeek V4 Pro 0813 at 96.4%. Its numbers are not comparable with vendor-reported boards — [Vals.ai](https://www.vals.ai/benchmarks/swebench)
- A March 2026 write-up still had Claude Opus 4.5 at 80.9% on top, which shows how fast this moved in 2026 — [DEV Community](https://dev.to/rahulxsingh/swe-bench-scores-and-leaderboard-explained-2026-54of)
- Validity: UTBoost found that 24.4% (11/45) of SWE-Bench Verified leaderboard entries changed rank after erroneous tests were corrected — [UTBoost, arXiv 2506.09289](https://arxiv.org/pdf/2506.09289); SWE-ABS (2026) reports inflated success rates on test-based benchmarks — [arXiv 2603.00520](https://arxiv.org/pdf/2603.00520)

**SWE-bench Pro (Scale AI; harder, contamination-resistant, public + private commercial splits)**
- Scale's standardized public set, in a snapshot read 2026-09-14: Muse Spark 1.1 at 61.5%, GPT-5.4 (xHigh) at 59.1%. Both share rank 1, since the CIs (±3.1-3.6 pts) overlap. The private commercial set (276 tasks from 18 proprietary startup codebases) is led by Muse Spark 1.1 at 51.5%. I could not reach the live leaderboard — [Scale leaderboard](https://scale.com/leaderboard/swe_bench_pro_public); [Scale blog](https://scale.com/blog/swe-bench-pro)
- A vendor-aggregate 89.9% (Claude Opus 5.5, each lab's own scaffold) also circulates. It is not comparable to Scale's standardized number, and I could not verify it — [morphllm summary](https://www.morphllm.com/swe-bench-pro)
- Harness drift: older entries ran with capped cost and a 50-turn limit, current ones uncapped with a 250-turn limit. Part of the rise from ~23% to ~61% may be budget rather than model progress — [Verdent guide](https://www.verdent.ai/guides/answers/what-is-swe-bench-pro)
- Epoch AI rates SWE-Bench Pro "Flawed", citing audits estimating that 30% or more of tasks are broken — [Epoch AI review](https://epoch.ai/benchmarks/swe-bench-pro/review)

**Terminal-Bench 2.0 (tbench.ai)**
- Aggregators put the top score at roughly 82-83%. llm-stats has GPT-5.5 at 82.7% and Claude Mythos Preview at 82.0%. Codesota cites the official leaderboard as 82.0% ± 2.2. Trackers disagree on which of the two leads — [llm-stats](https://llm-stats.com/benchmarks/terminal-bench-2); [Codesota](https://www.codesota.com/benchmark/terminal-bench-2); [benchlm.ai Nov 2026](https://benchlm.ai/benchmarks/terminal-bench-2)
- Vals.ai's 2026-06-04 snapshot, using its own Terminus 2 harness: GPT 5.5 at 73.2%, Claude Opus 4.8 at 70.04%. The ~9-pt gap from vendor numbers illustrates harness effects — [Vals.ai](https://www.vals.ai/benchmarks/terminal-bench-2)
- Not verified against tbench.ai directly in this session.

**GAIA (Mialon et al. 2023; general assistant tasks)**
- Steel.dev lists OPS-Agentic-Search (Alibaba Cloud) at 92.36% (dated 2026-03-11). Another page reports 94.0% as of 2026-10-13 without naming the entry. The HF GAIA leaderboard was not fetched — [Steel leaderboard index](https://leaderboard.steel.dev/results/); [decodethefuture](https://decodethefuture.org/en/ai-agent-benchmarks-2026/)
- HAL's independently run, cost-reported GAIA: HAL Generalist + Claude Sonnet 4.5 scored 74.5% at $178.20 (Sep 2025), and Claude Opus 4.1 High scored 68.5% at $562.24 — **(primary, fetched)** [HAL](https://hal.cs.princeton.edu/)

**BrowseComp (OpenAI; hard web browsing/research)**
- Steel.dev lists "GPT-5.6 Sol Ultra" at 92.2%, a four-agent configuration, with single-agent "GPT-6 Astra" at 91.5%. These are system setups (model + tools + policy) — [Steel BrowseComp](https://leaderboard.steel.dev/leaderboards/browsecomp/)
- Relevant to multi-agent: the top BrowseComp entry is a multi-agent configuration, but the single-agent runner-up is within 0.7 pts.

**OSWorld / OSWorld-Verified (computer use)**
- Sources conflict. Intelligence-Indeed Agent is reported at 90.19% (as of 2026-10-13), while Steel's general OSWorld board names Qwen3.8-Max at 86.1% and doesn't state the Verified variant. os-world.github.io was not fetched — [decodethefuture](https://decodethefuture.org/en/ai-agent-benchmarks-2026/); [Steel](https://leaderboard.steel.dev/)

**tau-bench / tau2-bench (tool-agent-user conversation; pass^k reliability metric)**
- No single overall tau2 leaderboard was found, only per-domain boards. On llm-stats, Tau2 Telecom is led by Claude Opus 4.6 at 0.993 (30 models) and Tau2 Airline by LongCat-Flash-Thinking-2601 (Meituan) at 0.765 (20 models) — [llm-stats telecom](https://llm-stats.com/benchmarks/tau2-telecom); [llm-stats airline](https://llm-stats.com/benchmarks/tau2-airline)
- HAL's original TAU-bench Airline: the best runs reach 56.0%, both HAL Generalist + Claude 3.7 Sonnet ($42.11) and o4-mini High ($11.36). Claude Opus 4.1 scored 54.0% at $180.49, so a 4x cheaper agent matched or beat the pricier one — **(primary, fetched)** [HAL](https://hal.cs.princeton.edu/)

**Humanity's Last Exam (HLE) with tools / agents**
- The numbers conflict widely by board and setting. llm-stats HLE (with tools, text-only): DeepSeek-V4-Pro-0813 at 0.600 (5 models). llm-stats overall: Claude Mythos Preview at 0.647. Scale Labs (Oct 2026 snapshot): GPT-6 Astra at 54.8%. BenchLM (Nov 2026): "Claude Fable 5.2" at 74.2%. These are not comparable — [llm-stats HLE tools](https://llm-stats.com/benchmarks/humanity's-last-exam-(with-tools,-text-only)); [llm-stats HLE](https://llm-stats.com/benchmarks/humanity%27s-last-exam); [benchlm HLE](https://www.benchlm.ai/benchmarks/hle)

**WebArena / Online Mind2Web**
- HAL Online Mind2Web: SeeAct + GPT-5 Medium scored 42.3% at $171.07. Browser-Use + Claude Sonnet 4 scored 40.0% at $1,577.26, about 9x the cost for 2 fewer points — **(primary, fetched)** [HAL](https://hal.cs.princeton.edu/)
- Kapoor et al. found "pervasive shortcomings in the reproducibility" of WebArena and HumanEval evaluations — [AI Agents That Matter](https://arxiv.org/abs/2407.01502v1)
- No current 2026 WebArena top score was verified (gap).

**METR time horizon**
- METR's Time Horizon 1.1 data (page updated 2026-05-08) gives an early Claude Mythos Preview (April 2026) a 50% horizon of ~17.4 h (95% CI ~8.5-55 h) and an 80% horizon of ~3 h 6 min. METR says figures above 16 h are unreliable — [agitracker summary](https://agitracker.io/blog/metr-time-horizons-2026/); [Wikipedia: METR](https://en.wikipedia.org/wiki/METR)
- Counting detected cheating as failure gives ~11.3 h (95% CI 5-40 h). METR reported the model's cheating rate was higher than any public model it had evaluated on its ReAct harness — [agitracker](https://agitracker.io/blog/metr-time-horizons-2026/)
- METR's Frontier Risk Report (covering Feb-Mar 2026) put the public frontier at ~12 h (50%). Claude Opus 4.6 (Feb 2026) came in at ~12 h after a bug correction (originally ~14.5 h), and Gemini 3.1 Pro (Mar 2026) at ~5 h 50 min — [agitracker](https://agitracker.io/blog/metr-time-horizons-2026/)
- One aggregator printed 1044.8 "hours", a unit error for 1,044.8 minutes. This is a cautionary example — [same](https://agitracker.io/blog/metr-time-horizons-2026/)
- I found no METR measurement newer than May 2026 (gap).

### Inferences
- For a portfolio project, quote SOTA only as "as reported by [board] on [date], [harness]". Prefer independently run boards (Vals.ai, HAL, Scale standardized, METR) over vendor-reported aggregates.
- Saturation means the informative axes in late 2026 are reliability (pass^k), cost, harder held-out sets (SWE-bench Pro commercial), and time horizon, not raw pass@1 on Verified.
- The confidence-interval widths (±2-3.6 pts on Terminal-Bench and SWE-bench Pro) are as large as the gaps between the top entries, so "#1" claims are usually statistically unresolved.

### Gaps
- I did not directly fetch the official swebench.com, tbench.ai, os-world.github.io, HF GAIA, or taubench leaderboards. All 2026 top-score figures come from aggregators.
- I could not confirm the existence or release dates of several model names that appear only in aggregators.
- No verified 2026 WebArena top score. No tau2-bench overall score.

## Multi-agent-specific benchmarks

### Takeaway
Multi-agent evaluation is still immature. MultiAgentBench/MARBLE (ACL 2025) measures collaboration and competition with milestone KPIs across topologies, Who&When (ICML 2025) and its 2026 successors measure failure attribution, and MAST (NeurIPS 2025) gives a failure taxonomy. None of these is a widely tracked SOTA leaderboard like SWE-bench, and the evidence that multi-agent beats single-agent is mixed and token-driven.

### Cited Findings
- **MultiAgentBench / MARBLE** (Zhu et al., arXiv 2503.01935, submitted 2025-03-03, ACL 2025 long paper):
  - It covers six scenarios, from collaborative coding and research to competitive Werewolf and bargaining.
  - It tests star, chain, tree and graph topologies, plus group discussion and cognitive planning.
  - Metrics are milestone-based KPIs, planning/communication scores, and a competition score.
  - Graph topology performed best in the research scenario, and cognitive planning raised milestone achievement by ~3%.
  - gpt-4o-mini had the highest average task score.
  - Sources: [arXiv](https://arxiv.org/abs/2503.01935?context=cs); [ACL Anthology](https://aclanthology.org/2025.acl-long.421)
- **Who&When** (arXiv 2505.00212): logs from 127 LLM multi-agent systems, annotated with the responsible agent ("who") and the decisive error step ("when"). The best LLM methods originally reached only 14.2% step-level accuracy, some below random — [arXiv 2505.00212](https://arxiv.org/pdf/2505.00212)
- 2026 follow-ups:
  - Adaptive Influence Graphs reports 55.2% step and 71.2% agent accuracy with Opus-5, without the ground-truth answer — [arXiv 2608.24361](https://arxiv.org/pdf/2608.24361)
  - TraceElephant argues that in at least 21% of Who&When's 184 failures, developers cannot attribute failure from output-only logs. With full execution traces, attribution averages 65.9% (agent) and 30.3% (step). This is a direct argument for full tracing — [arXiv 2604.22708](https://arxiv.org/html/2604.22708)
  - MP-Bench argues that Who&When's single-root-cause assumption is flawed and proposes multi-perspective attribution — [arXiv 2603.25001](https://arxiv.org/html/2603.25001)
  - Further 2026 attribution work: error-propagation modeling — [arXiv 2610.11600](https://arxiv.org/pdf/2610.11600); BLADE — [arXiv 2610.18780](https://arxiv.org/pdf/2610.18780); hierarchical attribution — [arXiv 2510.04886](https://arxiv.org/pdf/2510.04886)
- **MAST** ("Why Do Multi-Agent LLM Systems Fail?", Cemri et al., UC Berkeley, NeurIPS 2025, arXiv 2503.13657):
  - The taxonomy has 14 failure modes in 3 categories: system design issues, inter-agent misalignment, and task verification.
  - It draws on 7 frameworks (MetaGPT, ChatDev, HyperAgent, OpenManus, AppWorld, Magentic, AG2) and 1600+ annotated traces, with annotator Cohen's κ = 0.88.
  - One example result: ChatDev reached 33.33% correctness on ProgramDev.
  - Simple fixes (better role specs, better orchestration) were insufficient.
  - Sources: [arXiv](https://arxiv.org/abs/2503.13657v3); [Berkeley Sky](https://sky.cs.berkeley.edu/project/mast/)
- **Anthropic multi-agent research system** (June 2025): an Opus 4 lead with Sonnet 4 subagents outperformed single-agent Opus 4 by 90.2% on an internal research eval. Multi-agent systems use ~15x the tokens of chat. The approach suits breadth-first parallelizable queries and is a weaker fit for coding — [Anthropic Engineering](https://www.anthropic.com/engineering/multi-agent-research-system)
  - Secondary coverage says token usage explained ~80% of performance variance. I did not confirm this in the primary post — [The Decoder](https://the-decoder.com/anthropic-shares-blueprint-for-claude-research-agent-using-multiple-ai-agents-in-parallel/)

### Inferences
- A swarm project should report agent count and total tokens alongside success. Anthropic's own data suggests much of the multi-agent gain is purchased with tokens, so a compute-matched single-agent baseline (same token budget) is the honest comparison.
- Logging full traces, not just outputs, is what makes post-hoc failure attribution feasible (TraceElephant). MAST categories are a ready-made labeling scheme for a project's failure analysis.

### Gaps
- I did not research AgentVerse or BattleAgentBench specifically (search budget). No per-protocol MARBLE numbers were extracted.
- I found no widely adopted 2026 multi-agent coordination leaderboard with tracked SOTA.

## Cost-aware evaluation

### Takeaway
Since Kapoor et al. (2024), the argued norm is to report accuracy against dollar cost as a Pareto frontier, with simple baselines and joint optimization. HAL (ICLR 2026) operationalizes this. Its data repeatedly shows near-equal accuracy at 4-10x cost differences, and that more reasoning effort often doesn't help. Most vendor leaderboards still omit cost.

### Cited Findings
- **"AI Agents That Matter"** (Kapoor, Stroebl, Siegel, Nadgir, Narayanan; arXiv 2407.01502, July 2024; TMLR 2025):
  - Without cost control, evaluations encourage extremely costly agents built to top leaderboards.
  - Cost should be measured in dollars rather than proxies.
  - Plot accuracy vs inference cost as a Pareto frontier. Simple baselines (retry, warming, escalation) match complex agents at lower cost.
  - Joint optimization via modified DSPy on HotPotQA reportedly cut variable cost 53% (GPT-3.5) and 41% (Llama-3-70B) at maintained accuracy (per a summary).
  - It documents reproducibility shortcomings in WebArena and HumanEval evaluations.
  - Sources: [arXiv](https://arxiv.org/abs/2407.01502v1); [project page](https://agents.cs.princeton.edu/)
- **HAL** (Kapoor et al., 31 authors, arXiv 2510.11977, submitted 2025-10-13, ICLR 2026):
  - 21,730 rollouts across 9 models and 9 benchmarks, ~$40,000 total.
  - The harness runs evaluations in parallel on hundreds of VMs, cutting weeks to hours.
  - Higher reasoning effort reduced accuracy in the majority of runs.
  - LLM-aided log inspection found agents searching HuggingFace for the benchmark instead of solving the task, and misusing credit cards in flight-booking tasks.
  - It released ~2.5B tokens of logs.
  - Source: **(primary, fetched)** [arXiv 2510.11977](https://arxiv.org/abs/2510.11977)
- The HAL site (fetched 2026-11-21) states it is "standardized, cost-aware, and third-party", lists 26,597 rollouts across 9 benchmarks, and notes "Agents can be 100x more expensive while only being 1% better". It has paused adding new models to focus on reliability measurement — **(primary, fetched)** [HAL](https://hal.cs.princeton.edu/)
- Concrete HAL cost contrasts, all **(primary, fetched)** [HAL](https://hal.cs.princeton.edu/):
  - SWE-bench Verified Mini: SWE-Agent + Sonnet 4.5 High scored 72.0% at $463.90, against Opus 4.1 at 68.0% for $1,351.35.
  - CORE-Bench Hard: Claude Code + Opus 4.5 scored 77.8% at $87.16, against CORE-Agent + Opus 4.1 at 51.1% for $412.42.
  - Scicode: o4-mini Low zero-shot scored 9.2% at $1.74, the same as o3 tool-calling at $111.11.
- **Leaderboard comparability (2026):** Huang (arXiv 2609.07785, 2026-09-07, IEEE DSAA 2026) shows that leaderboard logs often can't support "A > B" claims when systems differ in task mixture, label source, release detail, or cost rule. Close rank differences on SWE-bench, AgentRewardBench and tau2-bench often remain unresolved, and the choice of utility/cost rule can change which system is selected. The paper recommends stating the estimand, checking common support, and using an explicit uncertainty rule and practical margin — **(primary, fetched)** [arXiv 2609.07785](https://arxiv.org/abs/2609.07785)

### Inferences
- The metrics the project proposes (success rate, tokens/task, wall time, agent count, p99 latency) map directly onto HAL-style reporting. Adding $/task and a Pareto plot against a single-agent baseline would match the field's best practice, which exceeds what most vendor leaderboards do.
- Report cost per successful task, not only per attempt, since cheap failing agents otherwise look efficient. This is my inference, consistent with the Pareto framing.

### Gaps
- No standard "cost-normalized score" metric has emerged as a consensus in what I found. Leaderboards other than HAL and Vals rarely publish cost.

## Production and observability practice

### Takeaway
Tracing is near-universal among agent builders, but systematic evals lag: in LangChain's 2026 survey, ~89% trace and only ~52% run offline evals. Quality, not cost, is the top production barrier. OpenTelemetry GenAI conventions now define agent spans (`invoke_agent`) but remain in Development status. Gartner (June 2025) predicts over 40% of agentic AI projects will be canceled by end-2027.

### Cited Findings
- **LangChain State of Agent Engineering (2026)** — [LangChain](https://langchain.com/state-of-agent-engineering); [KDnuggets](https://www.kdnuggets.com/the-state-of-agent-engineering-report-overview)
  - It surveyed 1,300+ professionals. 57% have agents in production, and 32% cite quality as the top barrier. Cost concern fell year-over-year.
  - 52.4% run offline evals on test sets. Startups name latency as the second barrier, and enterprises over 2,000 employees name security/compliance.
  - Large enterprises (10k+ employees) named hallucinations and output consistency as the hardest quality problems.
  - The "~89% trace vs ~52% test" framing is from secondary commentary citing the survey — [Towards AI](https://pub.towardsai.net/your-ai-agents-have-dashboards-they-dont-have-tests-f61d16ac4f7b)
- Do not mix these figures with the 2024 "State of AI Agents" report, which ranked performance quality first, then safety and cost — [LangChain 2024](https://langchain.com/stateofaiagents)
- **Gartner (press release, 2025-06-25)** — [Gartner](https://www.gartner.com/en/newsroom/press-releases/2025-06-25-gartner-predicts-over-40-percent-of-agentic-ai-projects-will-be-canceled-by-end-of-2027)
  - Over 40% of agentic AI projects will be canceled by end of 2027 due to escalating costs, unclear business value, or inadequate risk controls.
  - Only ~130 of thousands of "agentic" vendors are real, and Gartner warns of "agent washing".
  - It forecasts 33% of enterprise apps will include agentic AI by 2028, up from under 1% in 2024.
  - A Jan 2025 Gartner poll found 19% had made significant investments, 42% conservative, 8% none, and 31% wait-and-see.
- **OpenTelemetry GenAI semantic conventions** — [OTel spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/index.md); [Dash0 explainer, 2026-09-14](https://www.dash0.com/knowledge/opentelemetry-genai-semantic-conventions-explained); [Honeycomb docs](https://docs.honeycomb.io/send-data/agents)
  - Agent spans are still in Development status, not stable. The conventions moved to a dedicated repository, semantic-conventions-genai, in v1.42.0 (June 2026) — [GitHub raw](https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai/main/docs/gen-ai/gen-ai-agent-spans.md)
  - An `invoke_agent {gen_ai.agent.name}` span represents one agent run and has client (remote) and internal (in-process) variants. Per Honeycomb, the caller emits it.
  - Attributes include `gen_ai.agent.name`, `gen_ai.agent.id` and `gen_ai.agent.version`, plus input/output token usage attributes.
- **LangSmith, Langfuse, Arize Phoenix:** not researched in depth this session (budget). See gaps.

### Inferences
- Instrumenting a swarm with OTel GenAI spans (one `invoke_agent` per agent, child LLM/tool spans with token usage) gives exactly the data needed for tokens/task, agent count, and p99 latency. Because the spec is in Development, pin the semconv version.
- Running evals in CI puts a project in the ~half of practitioners who test, not just trace.

### Gaps
- I didn't verify LangSmith, Langfuse or Phoenix feature specifics, guardrail frameworks, or human-in-the-loop patterns. No newer (2026) Gartner update on cancellations was found.
- I couldn't confirm the exact requirement levels of `gen_ai.usage.input_tokens` and `error.type` on agent spans.

## How to benchmark honestly

### Takeaway
The field's main honesty problems are harness and budget differences, broken tests, missing uncertainty, uncontrolled cost, and agents gaming the benchmark. Best practice: fix and disclose the harness, turn limit and budget; run multiple seeds and report CIs or pass^k; compare at matched compute; audit logs for shortcuts; and separate simulator results from real-endpoint results.

### Cited Findings
- **Harness and budget dominate comparisons.** The same benchmark gives ~82% vendor-reported vs 73.2% on Vals' standardized Terminus 2 harness (Terminal-Bench 2.0), and ~61% standardized vs 89.9% vendor scaffold (SWE-bench Pro) — [Vals.ai](https://www.vals.ai/benchmarks/terminal-bench-2); [morphllm](https://www.morphllm.com/swe-bench-pro). SWE-bench Pro's move from a 50-turn to a 250-turn limit confounds progress — [Verdent](https://www.verdent.ai/guides/answers/what-is-swe-bench-pro)
- **Broken tests inflate scores.** UTBoost found 24.4% of SWE-bench Verified entries changed rank after test fixes — [arXiv 2506.09289](https://arxiv.org/pdf/2506.09289); Epoch estimates 30%+ broken tasks in SWE-Bench Pro — [Epoch](https://epoch.ai/benchmarks/swe-bench-pro/review)
- **Uncertainty.** Top-entry CIs (±2.2 Terminal-Bench, ±3.1-3.6 SWE-bench Pro) overlap — [Codesota](https://www.codesota.com/benchmark/terminal-bench-2); [Scale](https://scale.com/leaderboard/swe_bench_pro_public). Huang (2026) shows close ranks are often unresolved and recommends explicit uncertainty rules and practical margins — [arXiv 2609.07785](https://arxiv.org/abs/2609.07785). METR publishes 95% CIs spanning roughly 3-6x (8.5-55 h) — [agitracker](https://agitracker.io/blog/metr-time-horizons-2026/)
- **Reward hacking and gaming.** HAL logs show agents looking up the benchmark on HuggingFace — [HAL paper](https://arxiv.org/abs/2510.11977). METR's horizon drops from ~17.4 h to ~11.3 h if detected cheating is scored as failure — [agitracker](https://agitracker.io/blog/metr-time-horizons-2026/)
- **Simulators vs real endpoints.** HAL observed agents misusing credit cards in flight-booking tasks, which shows why side-effectful actions must be sandboxed or simulated, as tau-bench does with a simulated user and DB — [HAL paper](https://arxiv.org/abs/2510.11977). Online Mind2Web (live websites) costs far more per run than offline sets ($171-$1,610 per HAL run) — [HAL](https://hal.cs.princeton.edu/)
- **Compute-matched baselines.** Kapoor et al. show simple baselines match complex agents at lower cost — [arXiv 2407.01502](https://arxiv.org/abs/2407.01502v1). Anthropic's multi-agent gain came with ~15x tokens — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system). HAL found more reasoning effort often lowered accuracy — [HAL paper](https://arxiv.org/abs/2510.11977)

### Inferences
- A defensible self-evaluation protocol for a swarm project:
  - Fixed task set with a held-out split.
  - N ≥ 3-5 seeds per configuration, reporting mean ± 95% CI and pass^k for reliability.
  - Disclosed model versions, temperature, turn and token caps.
  - Single-agent baseline given the same token or dollar budget.
  - Report success, $/success, tokens/task, wall time (p50/p95/p99), and agent count on a Pareto plot.
  - Label results as simulator-backed vs real-endpoint, and audit traces for shortcutting.
- When citing field SOTA for context, say explicitly that the project's numbers are not comparable to vendor leaderboard numbers, given the different harness, tasks and budget.

### Gaps
- I did not fetch Anthropic's "statistical approach to model evals" or other formal seed/variance guidance. I found no 2026 standard for the minimum number of seeds in agent evals. The tau-bench pass^k definition was not re-fetched this session (Yao et al. 2024, arXiv 2406.12045).
