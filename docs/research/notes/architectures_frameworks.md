# LLM Multi-Agent Orchestration: Architectures and Frameworks (state as of 2026-11-21)

Notation: claims tagged **[verified 2026-11-21]** were fetched or checked directly during this research session (web fetch or GitHub API). Claims tagged **[from prior knowledge, not re-fetched]** are well-established facts with canonical URLs, but I did not re-open the page in this session; the report writer should treat them as reliable but unverified here. GitHub star counts and latest releases come from the GitHub REST API, queried 2026-11-21.

## Architecture patterns: what each does well and badly, and which workloads practitioners recommend them for

### Takeaway
By late 2026 practitioners have settled on a few patterns. The first is orchestrator-worker: a lead agent fans work out to context-isolated workers. The second is its programmatic successor, code-driven "dynamic workflows" or map-reduce, where a script rather than the LLM coordinates many agents in phases. Both are good for breadth-first, parallelizable, read-heavy work. Writes stay single-threaded, and clean-context reviewers or verifiers are added. Free-form swarms and parallel writers are still considered fragile. Multi-agent setups cost roughly 3-15x more tokens, so they pay off only on high-value, parallelizable tasks.

### Cited Findings

**Canonical workflow and agent patterns (Anthropic, Dec 2024)**
- Anthropic's "Building effective agents" (Dec 19, 2024) distinguishes *workflows* (LLMs and tools orchestrated through predefined code paths) from *agents* (LLMs dynamically directing their own process). It names five workflow patterns: prompt chaining, routing, parallelization (sectioning and voting), orchestrator-workers, and evaluator-optimizer. It advises starting with the simplest solution and adding complexity only when needed. — [Anthropic, Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) **[from prior knowledge, not re-fetched]**
  - Prompt chaining corresponds to the pipeline/assembly-line pattern. Parallelization by sectioning corresponds to map-reduce fan-out. Voting is an ensemble pattern. Evaluator-optimizer is the debate/critique loop. Orchestrator-workers is distinguished from parallelization because the subtasks are not predefined: the orchestrator decides them dynamically. — same source **[from prior knowledge]**

**Orchestrator-worker (lead agent and subagents): Anthropic Research, June 13, 2025** **[verified]**
- Architecture: a LeadResearcher plans, spawns subagents, synthesizes their results, and decides whether more research is needed. It saves its plan to memory because context beyond about 200k tokens gets truncated. Subagents each have their own context window and tools and search iteratively with interleaved thinking. A separate CitationAgent attributes claims at the end. — [Anthropic, How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- A Claude Opus 4 lead with Sonnet 4 subagents outperformed single-agent Opus 4 by 90.2% on Anthropic's internal research eval. — same
- On BrowseComp, token usage alone explained about 80% of performance variance. Token usage, number of tool calls, and model choice together explained about 95%. — same
- Agents use about 4x the tokens of chat interactions, and multi-agent systems about 15x. — same
- Upgrading the model (to Sonnet 4) gave a larger gain than doubling the token budget on Sonnet 3.7. — same
- The lead spawns 3-5 subagents in parallel, and each subagent runs 3+ tool calls in parallel. This cut research time by up to 90% on complex queries. — same
- Effort-scaling rules are embedded in prompts: simple fact-finding uses 1 agent with 3-10 tool calls, direct comparisons use 2-4 subagents with 10-15 calls each, and complex research uses more than 10 subagents. — same
- Good fit: breadth-first, parallelizable work; information exceeding one context window; many complex tools; high-value tasks. Poor fit: tasks needing shared context or many interdependencies between agents, including "most coding tasks." — same
- Production lessons:
  - Agents are stateful, and errors compound, so resume from checkpoints rather than restarting.
  - Rely on full tracing.
  - Use "rainbow deployments" so in-flight agents aren't broken.
  - **Synchronous subagent execution is a bottleneck: the lead waits for each batch.** Asynchronous execution would add parallelism but brings coordination, state-consistency, and error-propagation challenges.
  - Subagents should write outputs to external storage and pass lightweight references back. — same

**Anthropic guidance, Jan 23, 2026: "when (and when not) to use multi-agent"** **[verified]**
- Three justified cases:
  - Context protection: a subtask generates more than about 1k tokens that are mostly irrelevant to the main task.
  - Parallelization: valued for thoroughness more than speed.
  - Specialization: signals include 20+ tools, domain confusion, or degradation when tools are added. — [Claude blog, Building multi-agent systems: when and how to use them](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)
- Multi-agent implementations "typically use 3-10x more tokens" than single-agent ones for equivalent tasks. — same
- Recommends **context-centric decomposition** (split by the context a subtask needs) over **problem-centric decomposition** (planner/implementer/tester/reviewer). The problem-centric split made subagents spend more tokens coordinating than working, and handoffs lose fidelity (the "telephone game"). — same
- Parallel systems "often take longer overall" than single-agent ones despite running in parallel. — same
- Recommends a verification subagent with explicit criteria, and warns of the "early victory problem," where a verifier runs a few tests and declares success prematurely. — same

**Programmatic orchestration ("dynamic workflows"), 2026** **[verified]**
- Claude Code dynamic workflows became generally available on May 28, 2026. Claude writes orchestration scripts that run "tens to hundreds of parallel subagents" in one session. Coordination happens outside the conversation, and progress is saved so interrupted jobs resume. Runs can last hours or days. Use cases: codebase-wide audits, migrations spanning thousands of files, and adversarial or independent verification. Example: the Bun Zig-to-Rust port produced about 750k lines of Rust with 99.8% of tests passing in 11 days, using hundreds of porting agents and two reviewers per file. Caveat: workflows use "substantially more tokens." — [Claude, Introducing dynamic workflows in Claude Code](https://claude.com/resources/articles/introducing-dynamic-workflows-in-claude-code)
- Claude Managed Agents docs describe the two modes:
  - **Subagents:** "the agent itself determines what happens next." Subagent threads persist and can receive follow-up messages.
  - **Dynamic workflows:** "Claude writes a program to orchestrate agents without Claude's direct involvement." Context and results pass programmatically, so the main thread stays free to talk to the user. The docs recommend workflows for "most work that needs more than one agent" (audits, migrations, deep research, cross-checking).
  - — [Claude Platform Docs, Multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration)

**Cognition's practitioner view of what works and what doesn't** **[verified]**
- Cognition's April 22, 2026 assessment:
  - What works: "writes stay single-threaded" while multiple agents contribute intelligence; read-only search subagents; clean-context reviewers (Devin Review catches about 2 bugs per PR on average, about 58% of them severe); "smart friend" escalation to a stronger model; and manager agents that spawn child Devins coordinated via an internal MCP. Cognition calls the last one "map-reduce-and-manage."
  - What fails: parallel-writer swarms, which make conflicting implicit decisions, and unstructured agent-negotiation networks, which are "mostly a distraction."
  - Manager failure modes: managers were overly prescriptive, assumed shared state that didn't exist, and didn't relay cross-agent discoveries.
  - — [Cognition, Multi-Agents: What's Actually Working](https://cognition.com/blog/multi-agents-working)

**Decentralized swarm / handoff**
- OpenAI Swarm (Oct 2024, experimental, educational) introduced lightweight "routines and handoffs," where an agent transfers control to another by returning it from a tool call. — [openai/swarm](https://github.com/openai/swarm) **[from prior knowledge for semantics; repo existence verified via GitHub API: ~22.2k stars, last push 2026-04-15]**
- LangGraph has a swarm library (`langgraph-swarm`, latest release 0.1.0, 2025-12-04, ~1.6k stars, still active) alongside the supervisor library (`langgraph-supervisor`, **now archived**, last release 0.0.31, 2025-11-19). — [langgraph-swarm-py](https://github.com/langchain-ai/langgraph-swarm-py), [langgraph-supervisor-py](https://github.com/langchain-ai/langgraph-supervisor-py) **[verified via GitHub API]**
- Microsoft Agent Framework 1.0 ships handoff, group chat, sequential, concurrent, and Magentic-One orchestrations as stable patterns. — [Microsoft Agent Framework 1.0](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/) **[verified]**

**Hierarchical / manager trees**
- Managed Agents subagent delegation is **one level deep**, with at most 25 child threads per session at a time (idle ones included). Each of `subagents` and `workflows` can list up to 20 predefined agents. All agents share one sandbox and filesystem but run in context-isolated threads. — [Claude Platform Docs, Multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration) **[verified]**
- Cognition Devin Bots (launched Nov 17, 2026) are persistent manager agents that coordinate teams of Devins on multi-day or multi-week projects. — [Fortune, Nov 17 2026](https://fortune.com/2026/11/17/exclusive-cognition-bets-on-devin-bots-coding-agents-in-challenge-to-anthropics-duet-and-openais-dots/) **[search snippet only; secondary source]**
- Google ADK models hierarchy as parent/sub-agents plus workflow agents (Sequential, Parallel, Loop). — [ADK docs](https://google.github.io/adk-docs/) **[from prior knowledge; ADK 2.x graph-workflow claim from secondary source, see below]**

**Blackboard / shared memory**
- Managed Agents provides a de facto blackboard: "All agents share the same sandbox, filesystem, and vault credentials," and each has its own conversation thread. — [Claude Platform Docs](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration) **[verified]**
- Anthropic recommends subagents write to external storage and pass references back (an artifact-store or blackboard pattern) to avoid the "game of telephone." — [Anthropic multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) **[verified]**
- MetaGPT uses a shared message pool with publish-subscribe, so agents read structured artifacts instead of chatting one-to-one. — [MetaGPT paper, arXiv 2308.00352](https://arxiv.org/abs/2308.00352) **[from prior knowledge]**

**Debate / critique**
- Evaluator-optimizer is Anthropic's name for a generate-critique loop. — [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) **[prior knowledge]**
- Clean-context reviewer agents are the most production-validated form of critique (Devin Review figures above). — [Cognition 2026](https://cognition.com/blog/multi-agents-working) **[verified]**
- Dynamic workflows market "adversarial agents that try to break the result." — [Claude dynamic workflows](https://claude.com/resources/articles/introducing-dynamic-workflows-in-claude-code) **[verified]**

### Inferences
- Pattern recommendations by workload:
  - Research, search, and audit (read-heavy, decomposable): orchestrator-worker or map-reduce fan-out.
  - Large mechanical transformations (migrations, ports): programmatic workflow with a fan-out of workers plus per-item reviewers.
  - Coding and tightly coupled writes: a single writer, optionally with read-only helpers and a clean-context reviewer.
  - Customer-service routing: handoff or routing.
  - Free-form swarms remain experimental.
- The industry moved from "the LLM is the scheduler" (lead agent decides each handoff) toward "code is the scheduler" (Claude writes a workflow program, Microsoft Agent Framework graph workflows, ADK graph workflows). A queue-plus-worker-pool design with an adaptive scheduler sits squarely in this direction. Its differentiator would be an *adaptive* scheduler (dynamic batch sizing, priority, and back-pressure based on observed cost or latency), in place of the static phases a lead agent writes.
- Anthropic's own named bottleneck (synchronous batches, where the lead waits for the slowest subagent) is exactly what a pull-based shared queue with continuous worker refill addresses. This makes a strong positioning point, but it trades off against the coordination and consistency problems Anthropic also flags.
- Token cost (3-15x) means a scheduler that is token-budget-aware, and not just concurrency-aware, would match the dominant concern in the literature.

### Gaps
- No primary benchmark found comparing pull-queue worker pools against lead-agent fan-out directly.
- Blackboard-specific LLM frameworks (as distinct papers) were not researched in this pass.
- Managed Agents workflow-run limits (for example, a reported 1,000-agent cap) appear only in secondary sources ([fourweekmba](https://fourweekmba.com/ai-claude-managed-agents-adds-workflows-of-up-to-1000-agents/)). The primary "Workflow runs" limits page ([docs](https://platform.claude.com/docs/en/managed-agents/workflow-runs)) was not fetched. A secondary claim of a 94.3% vs 20-38.6% single-agent bug benchmark is also unverified.

## Anthropic material, 2024-2026

### Takeaway
Anthropic's arc runs from five composable patterns (Dec 2024), to an orchestrator-worker research system (Jun 2025, +90.2% over a single agent at about 15x chat tokens), to explicit guidance on when *not* to use multi-agent (Jan 2026), to productized subagents and code-orchestrated "dynamic workflows" in both Claude Code (GA May 28, 2026) and Claude Managed Agents (beta).

### Cited Findings
- "Building effective agents," Dec 2024: five patterns plus autonomous agents; simplicity first. — [anthropic.com](https://www.anthropic.com/engineering/building-effective-agents) **[prior knowledge]**
- Multi-agent research system, June 13, 2025: see the figures in the first section. — [anthropic.com](https://www.anthropic.com/engineering/multi-agent-research-system) **[verified]**
- "Building multi-agent systems: when and how," Jan 23, 2026: 3-10x tokens; context-centric decomposition. — [claude.com](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them) **[verified]**
- Claude Code dynamic workflows GA, May 28, 2026: tens to hundreds of subagents, resumable, `ultracode` effort setting, on by default for Max, Ultra, Team, and Enterprise. — [claude.com](https://claude.com/resources/articles/introducing-dynamic-workflows-in-claude-code) **[verified]**
- Claude Managed Agents multiagent orchestration:
  - Beta, under header `managed-agents-2026-04-01`.
  - `multiagent.type = "multiagent_20261001"` enables subagents, dynamic workflows, and an optional "advisor" model consulted mid-turn. Agents on the older "coordinator" type must migrate.
  - Delegation is one level deep, with at most 25 concurrent child threads. Shared sandbox and filesystem.
  - Per-agent MCP servers; session-scoped vault credentials.
  - Session budgets cap spend.
  - — [Claude Platform Docs](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration) **[verified]**
- Secondary reporting dates Managed Agents dynamic workflows public beta to Oct 9, 2026. — [Latent Space](https://latent.space/p/claude-dynamic-workflows), [xenospectrum](https://xenospectrum.com/en/claude-managed-agents-dynamic-workflows/) **[search snippets; secondary]**
- Managed Agents launched in April 2026 (InfoQ, 2026/04). — [InfoQ](https://infoq.com/news/2026/04/anthropic-managed-agents), [Claude blog](https://claude.com/blog/claude-managed-agents) **[search result titles only]**
- Claude Agent SDK (Python): ~8.6k stars, latest release v0.2.186 (2026-11-19), actively developed. — [anthropics/claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python) **[verified via GitHub API]**

### Inferences
- Anthropic's own product line now offers both "LLM-as-orchestrator" (subagents) and "program-as-orchestrator" (workflows), and the docs recommend the latter for most multi-agent work. That validates a scheduler-driven design.

### Gaps
- Exact date and content of the Managed Agents multiagent launch post ([claude.com/blog/new-in-claude-managed-agents](https://claude.com/blog/new-in-claude-managed-agents)) were not fetched.
- Claude Agent SDK subagent semantics (for example, AgentDefinition and parallel Task tool calls) were not re-verified this session.

## OpenAI: Swarm to Agents SDK

### Takeaway
Swarm (Oct 2024) was an educational prototype of routines and handoffs. The Agents SDK (Mar 2025) productionized it with Agents, Handoffs, Guardrails, Sessions, and Tracing. The Apr 15, 2026 update added sandboxed execution and a "model-native harness" (Python first) and announced subagents and code mode as in progress. As of Nov 2026 the Python SDK is at v0.25.0 (still pre-1.0). Durable execution comes via the Temporal integration (Python GA Mar 23, 2026).

### Cited Findings
- Swarm repo: ~22.2k stars, last push 2026-04-15, no releases. It is described as experimental and educational and is superseded by the Agents SDK. — [openai/swarm](https://github.com/openai/swarm) **[stats verified; "superseded" status from prior knowledge of README]**
- The Agents SDK launched Mar 11, 2025 with primitives Agents, Handoffs, Guardrails, and Tracing. Supported multi-agent patterns are handoffs (control transfer) and "agents as tools" (manager-style). — [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) **[prior knowledge]**
- Apr 15, 2026, "The next evolution of the Agents SDK": sandbox execution in controlled computer environments and a model-native harness. Python first, TypeScript later. Subagents and code mode are "in the works." — [OpenAI](https://openai.com/index/the-next-evolution-of-the-agents-sdk/) **[search snippet; direct fetch returned 403]**
- openai-agents-python: ~30.8k stars, latest v0.25.0 (2026-11-18). — [GitHub](https://github.com/openai/openai-agents-python) **[verified via GitHub API]**
- Temporal + OpenAI Agents SDK: public preview Jul 30, 2025. Python GA Mar 23, 2026. Agent orchestration runs inside Temporal workflows, and model calls run as Activities. — [Temporal blog](https://temporal.io/blog/announcing-openai-agents-sdk-integration), [Temporal docs](https://docs.temporal.io/develop/python/openai-agents), [BusinessWire 2025-07-30](https://www.businesswire.com/news/home/20250730783559/en) **[search snippets]**

### Inferences
- OpenAI's model is decentralized handoff, the opposite of a central queue. A worker-pool design can position itself as "centralized scheduling, no agent-to-agent control transfer," which avoids handoff context loss.

### Gaps
- Exact feature list of v0.20-0.25 not reviewed; whether subagents have shipped by Nov 2026 is unconfirmed.

## LangGraph, AutoGen/AG2/Microsoft Agent Framework, CrewAI, MetaGPT, CAMEL, Google ADK/A2A, MCP

### Takeaway
LangGraph (1.x LTS), Microsoft Agent Framework (1.0 GA Apr 2026), CrewAI (1.x), Google ADK (2.x), and OpenAI Agents SDK (0.25) are the active production frameworks. AutoGen is **archived**, langgraph-supervisor is **archived**, and MetaGPT's open-source repo has been dormant since Jan 2026. A2A reached v1.0 (Mar 2026) under the Linux Foundation and later joined AAIF. MCP is the tool and context layer, and it is increasingly used as the inter-agent channel too (Cognition uses an internal MCP to coordinate child agents).

### Cited Findings

GitHub stats below are from the GitHub API, queried 2026-11-21. **[verified]**

| Repo | Stars | Latest release | Status |
|---|---|---|---|
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | ~45.2k | 1.2.24 (2026-11-20) | active |
| [microsoft/autogen](https://github.com/microsoft/autogen) | ~62.0k | python-v0.7.5 (2025-09-30) | **archived** |
| [ag2ai/ag2](https://github.com/ag2ai/ag2) | ~5.1k | v1.3.3 (2026-11-19) | active |
| [microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel) | ~28.8k | dotnet-1.82.0 (2026-11-07) | maintained |
| [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | ~61.1k | 1.15.44 (2026-11-19) | active |
| [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT) | ~71.4k | v0.8.1 (2024-04-22) | last push 2026-01-21 |
| [camel-ai/camel](https://github.com/camel-ai/camel) | ~18.1k | v0.2.90 (2026-03-22) | pushed 2026-11-16 |
| [camel-ai/owl](https://github.com/camel-ai/owl) | ~20.2k | none | pushed 2026-11-16 |
| [google/adk-python](https://github.com/google/adk-python) | ~22.3k | v2.15.0 (2026-11-19) | active |
| [a2aproject/A2A](https://github.com/a2aproject/A2A) | ~26.6k | v1.0.1 (2026-05-28) | active |
| [modelcontextprotocol/modelcontextprotocol](https://github.com/modelcontextprotocol/modelcontextprotocol) | ~9.7k | spec 2026-07-28 | active |

The `microsoft/agent-framework` repo returned "Not Found" from the API at query time, possibly because it was renamed or moved. The AutoGen README still links to it.

**LangGraph**
- LangGraph 1.0 and LangChain 1.0 shipped around Oct 22, 2025. The release is stability-focused (the core graph and durable-execution API is unchanged), `langgraph.prebuilt` is deprecated in favor of `langchain.agents`, and 1.0 is LTS until 2.0. — [LangChain blog](https://blog.langchain.com/langchain-langgraph-1dot0), [release policy](https://docs.langchain.com/oss/python/release-policy.md) **[search snippets]**
- Deep Agents (`deepagents`) is pre-1.0 and evolving quickly. — [LangChain release policy](https://docs.langchain.com/oss/python/release-policy.md) **[search snippet]**
- Core model: StateGraph state machines with checkpointers for persistence and human-in-the-loop. Multi-agent is done via supervisor (central router) or swarm (handoff) topologies. — [LangGraph docs](https://langchain-ai.github.io/langgraph/) **[prior knowledge]**

**AutoGen, AG2, and Microsoft Agent Framework**
- AutoGen paper (Wu et al., 2023): "conversable agents" and multi-agent conversation. — [arXiv 2308.08155](https://arxiv.org/abs/2308.08155) **[prior knowledge]**
- AutoGen 0.4 (Jan 2025) was an asynchronous, event-driven actor-model rewrite with Core, AgentChat, and Extensions layers. — [AutoGen docs](https://microsoft.github.io/autogen/) **[prior knowledge]**
- AG2 is the community fork by the original AutoGen creators (Nov 2024). It continues the 0.2-style API and is now at v1.3.3. — [ag2ai/ag2](https://github.com/ag2ai/ag2) **[fork history from prior knowledge; version verified]**
- **AutoGen is archived.** The README says Microsoft provides no bug fixes, security fixes, features, or support, and advises against production use. New users are pointed to Microsoft Agent Framework, with a migration guide. — [microsoft/autogen README](https://github.com/microsoft/autogen) **[verified]**
- Microsoft Agent Framework was introduced Oct 1, 2025 as the open-source successor combining Semantic Kernel and AutoGen. — [Microsoft Foundry blog](https://devblogs.microsoft.com/foundry/introducing-microsoft-agent-framework-the-open-source-engine-for-agentic-ai-apps/) **[search snippet]**
- Microsoft Agent Framework 1.0 GA, Apr 3, 2026, for .NET and Python, with stable APIs and LTS. — [Microsoft devblogs](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/) **[verified]**
  - A graph workflow engine with branching, fan-out, and convergence. Checkpointing and hydration let long-running processes survive interruptions.
  - Orchestrations: sequential, concurrent, handoff, group chat, and Magentic-One, all with streaming, checkpointing, human-in-the-loop, and pause/resume.
  - Connectors for Azure OpenAI, OpenAI, Anthropic, Bedrock, Gemini, and Ollama.
  - MCP is supported. "A2A 1.0 support" is "coming soon" as of the 1.0 post.

**CrewAI**
- Crews are role-based autonomous agent teams. Flows provide event-driven, stateful, deterministic orchestration around crews, and CrewAI docs recommend Flows for production. — [CrewAI docs](https://docs.crewai.com/) **[secondary summaries via search; consistent with prior knowledge]**
- CrewAI v1.0 was reached in 2025, and the project is now at 1.15.44. — [GitHub](https://github.com/crewAIInc/crewAI) **[version verified; 1.0 date not found]**

**MetaGPT**
- MetaGPT (Hong et al., 2023; ICLR 2024 oral) encodes human SOPs into prompts with software-company roles: product manager, architect, project manager, engineer, and QA. Agents produce structured artifacts (PRDs, designs) through a shared message pool with publish/subscribe. — [arXiv 2308.00352](https://arxiv.org/abs/2308.00352) **[prior knowledge]**
- MetaGPT's commercial product MGX was reportedly rebranded "Atoms" on Jan 13, 2026, and the OSS repo has had no push since late January 2026. The dormancy is consistent with the GitHub API (last push 2026-01-21; last release v0.8.1 in Apr 2024). — secondary source via search ([rework.com roundup](https://resources.rework.com/tools/ai-agents/best-multi-agent-frameworks-2026)); dormancy **[verified via GitHub API]**

**CAMEL**
- CAMEL (Li et al., NeurIPS 2023) introduced "role-playing" with inception prompting, in which an AI-user agent and an AI-assistant agent cooperate. — [arXiv 2303.17760](https://arxiv.org/abs/2303.17760) **[prior knowledge]**
- OWL (Optimized Workforce Learning) is CAMEL's multi-agent "workforce" system for real-world task automation, reported as top open-source on the GAIA benchmark in 2025. — [camel-ai/owl](https://github.com/camel-ai/owl) **[repo verified; GAIA claim from prior knowledge]**
- OASIS simulates up to 1M agents in social-media simulation. — [arXiv 2411.11581](https://arxiv.org/abs/2411.11581) **[prior knowledge]**

**Google ADK and A2A**
- Google announced ADK and A2A on Apr 9, 2025. — [Google Developers blog](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) **[prior knowledge]**
- ADK is now at v2.15.0. A secondary source says ADK 2.0 added graph-based workflows and code-expressed dynamic workflows. — [GitHub](https://github.com/google/adk-python) **[version verified]**; 2.0 features from search summary (unverified)
- A2A:
  - Google donated A2A to the Linux Foundation on Jun 23, 2025. — [LF press](https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year) **[search snippet]**
  - v1.0 shipped in March 2026: the first stable release, with breaking changes; multi-protocol bindings and version negotiation; multi-tenancy; and signed Agent Cards. — [Google Open Source blog, Apr 2026](https://opensource.googleblog.com/2026/04/a-year-of-open-collaboration-celebrating-the-anniversary-of-a2a.html), [Google dev forum](https://discuss.google.dev/t/what-s-new-in-a2a-v1-0-a-python-dx-glow-up-and-a-fresh-new-look/381896) **[search snippets]**
  - More than 150 supporting organizations at the one-year mark (LF press, Apr 9, 2026). — [LF](https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year) **[search snippet]**
  - A2A joined the Agentic AI Foundation (AAIF) as a hosted project (post dated Aug 17, 2026). — [AAIF](https://aaif.io/blog/a2a-joins-aaif) **[search snippet]**
  - The latest release is v1.0.1 (2026-05-28). — [GitHub](https://github.com/a2aproject/A2A) **[verified]**

**MCP**
- Anthropic introduced MCP in Nov 2024 as a protocol for connecting models to tools and context. It is complementary to A2A: MCP connects an agent to tools, and A2A connects agents to agents. — [modelcontextprotocol.io](https://modelcontextprotocol.io/) **[prior knowledge]**
- MCP was donated to the Linux Foundation's Agentic AI Foundation in Dec 2025. — **[prior knowledge, not re-verified; flag]**
- The latest spec release is tagged 2026-07-28. — [GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol) **[verified]**
- In multi-agent systems:
  - Managed Agents declares MCP servers per agent to scope each subagent's tool access. — [Claude docs](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration) **[verified]**
  - Cognition's manager Devin coordinates child Devins "through an internal MCP." — [Cognition](https://cognition.com/blog/multi-agents-working) **[verified]**
  - Microsoft Agent Framework 1.0 supports MCP. — [MS devblogs](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/) **[verified]**

### Inferences
- Every major framework now offers graph or workflow determinism plus agentic nodes: LangGraph StateGraph, Microsoft Agent Framework Workflows, CrewAI Flows, ADK workflow agents and graphs, and Claude dynamic workflows. "Agentic workflows" won over fully autonomous agent societies for production.
- Research-era role-play frameworks (MetaGPT, CAMEL, AutoGen 0.2) were influential but are archived, dormant, or research-oriented. Citing them as lineage is appropriate. Citing them as current competitors is not.
- None of these frameworks is built around a pull-based shared work queue with batch-executing workers and adaptive scheduling. Fan-out in all of them is push-based: the orchestrator or graph assigns work. This gives the portfolio design a clear differentiation, with the caveat that durable-execution engines such as Temporal already provide queue semantics (task queues and worker pools) underneath.

### Gaps
- Microsoft Agent Framework repo URL and stars could not be confirmed (API returned Not Found).
- CrewAI 1.0 release date and current Flows features were not verified from primary docs.
- ADK 2.0 release date and features come from a secondary source only.
- OWL's current status (beyond recent pushes) and GAIA ranking were not re-verified.
- The MCP-to-AAIF donation date is from prior knowledge.

## 2025-2026 trends: agent protocols, durable execution, workflows vs autonomy, and the "don't build multi-agents" debate

### Takeaway
The June 2025 "Don't Build Multi-Agents" vs "How we built our multi-agent research system" exchange (published one day apart) has largely converged. Both camps now endorse context-isolated, read-mostly subagents, clean-context reviewers, and centrally orchestrated map-reduce, with single-threaded writes. Both reject free-form swarms of parallel writers. Durable execution (checkpointing and resume) and interop protocols (MCP, A2A 1.0) became standard infrastructure.

### Cited Findings
- **Cognition, "Don't Build Multi-Agents," June 12, 2025 (Walden Yan):** **[verified]**
  - Two principles: "Share context" (full agent traces, not just messages) and "Actions carry implicit decisions."
  - Flappy Bird example: parallel subagents build a mismatched background and bird.
  - Names OpenAI Swarm and Microsoft AutoGen as promoting "the wrong way."
  - Recommends a single-threaded linear agent, plus context compression via a separate (fine-tuned) model for long tasks.
  - Approves of Claude Code-style subagents that answer narrow questions and don't write code.
  - Says it is "optimistic about the long-term possibilities."
  - — [Cognition](https://cognition.com/blog/dont-build-multi-agents)
- **Anthropic's multi-agent research post, June 13, 2025, the day after:** argues multi-agent wins for breadth-first research while conceding that coding and shared-context tasks fit poorly. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system) **[verified]**
- **Cognition follow-up, Apr 22, 2026, "Multi-Agents: What's Actually Working":** keeps writes single-threaded but endorses read-only subagents, reviewers, smart-friend escalation, and manager agents ("map-reduce-and-manage"). Parallel-writer swarms still lack adoption. — [Cognition](https://cognition.com/blog/multi-agents-working) **[verified]**
- Cognition launched Devin Bots, persistent manager agents, on Nov 17, 2026. This is product-level confirmation of the reversal toward hierarchical multi-agent. — [Fortune](https://fortune.com/2026/11/17/exclusive-cognition-bets-on-devin-bots-coding-agents-in-challenge-to-anthropics-duet-and-openais-dots/) **[search snippet]**
- Anthropic, Jan 2026: a "well-designed single agent" often suffices. Use multi-agent only for context protection, parallelization, or specialization. — [Claude blog](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them) **[verified]**
- **Durable execution:**
  - Temporal + OpenAI Agents SDK (Python GA Mar 23, 2026). — [Temporal](https://temporal.io/blog/announcing-openai-agents-sdk-integration) **[snippet]**
  - Microsoft Agent Framework workflow checkpointing and hydration. — [MS](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/) **[verified]**
  - LangGraph positions itself as a "durable agent framework" with checkpointers. — [LangChain](https://blog.langchain.com/langchain-langgraph-1dot0) **[snippet]**
  - Claude dynamic workflows save progress so interrupted jobs resume. — [Claude](https://claude.com/resources/articles/introducing-dynamic-workflows-in-claude-code) **[verified]**
  - Anthropic (2025) called for checkpoint-and-resume because errors compound. — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system) **[verified]**
- **Protocols:** A2A v1.0 (Mar 2026; LF; AAIF Aug 2026) and MCP (spec 2026-07-28) — sources above.

### Inferences
- How the debate settled: "multi-agent for reading, single agent for writing; code-orchestrated rather than chat-orchestrated; verify with fresh context." A portfolio project should state where it sits on these axes:
  - Are workers' outputs independent? Batch items are, ideally.
  - Who merges results?
  - Is there a verification stage?
- Durable execution is now table stakes. A shared-queue design gets natural retry and resume semantics (re-enqueue failed items), and that is worth highlighting.

### Gaps
- No quantitative industry survey of pattern adoption (for example, the percentage of production systems using each topology) was found.
- No primary figures found on A2A production deployments beyond the LF "150+ organizations" claim.
