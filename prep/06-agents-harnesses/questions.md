# Agents & Harnesses — questions

Entries follow the [Q&A schema](../../CONTRIBUTING.md#the-qa-entry-schema).

---

### Q: What is an "agent" in the 2026 LLM sense, and how does it differ from a chatbot?

**Category:** concept
**Difficulty:** intro
**Tags:** [agents, definition, autonomy]

**Short answer.** An agent is an LLM-driven system that takes actions over multiple steps to achieve a goal — it observes state, picks a tool/action, executes it, observes the result, and repeats. A chatbot generates a single response and stops. The defining trait of an agent is **looped tool use**: the model controls its own flow, deciding when to call which tool and when to stop. Practically: SWE agents, Operator-style web agents, Devin/SWE-bench agents, customer-support agents that read tickets and write follow-ups.

**Expansion / why this is the answer.**
- **Chatbot**: user → model → response (one shot).
- **Agent loop** (Wei et al. 2022, ReAct; Yao et al. 2022):
  1. Observe state (user goal + prior steps + environment).
  2. Reason about the next action.
  3. Choose a tool and arguments.
  4. Execute the tool; get the observation.
  5. Loop until the model decides to stop (or hits a step limit).
- Defining traits:
  - **Tool use**: the model can call external functions / APIs / code interpreters.
  - **Multi-step**: state evolves across steps.
  - **Autonomy**: the model decides what to do, not the developer.
- **Real-world agents in 2026**:
  - **Coding agents**: Claude Code, Devin, Cursor agent mode, OpenHands. SWE-bench scores are the benchmark of record.
  - **Browser/computer-use agents**: Anthropic Computer Use, OpenAI Operator.
  - **Research / "deep research" agents**: deep-search loops that retrieve, read, synthesize.
- **What's a "harness"?** The infrastructure around the agent — the loop, tool routing, error recovery, context management, evaluation. Sometimes called scaffolding. (See peer repo `harness-engineer-roadmap` for depth.)

**Common follow-ups.**
- "How is this different from RAG?" → RAG is single-shot retrieve-then-answer. Agents loop and can decide to retrieve more, run code, edit a file, etc.
- "When is an agent overkill?" → Single-turn lookup; a fixed pipeline you could hardcode does it.

**Common mistakes.**
- Calling any tool-using LLM an agent. The loop and autonomy are what define it.
- Treating agents as more capable than they are — long horizons still fail often.

**References.**
- [Yao et al. — "ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629) — ReAct.
- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — taxonomy of agent patterns.

---

### Q: Compare ReAct, Plan-and-Execute, Reflexion, and Tree-of-Thoughts.

**Category:** concept
**Difficulty:** mid
**Tags:** [react, planning, reflexion, tree-of-thoughts]

**Short answer.** **ReAct** (Yao et al. 2022): interleave reasoning (Thought) and acting (Action / Observation) one step at a time — the canonical agent pattern. **Plan-and-Execute** (Wang et al. 2023): two phases — first generate a high-level plan, then execute each step. **Reflexion** (Shinn et al. 2023): after a failure, the model writes a textual "lesson" into memory and retries; verbal RL. **Tree-of-Thoughts** (Yao et al. 2023): explore a *tree* of reasoning steps rather than a linear chain, with backtracking — useful for problems requiring search.

**Expansion / why this is the answer.**
- **ReAct**:
  - `Thought: ...` → `Action: tool(args)` → `Observation: ...` → next thought.
  - Strengths: clean coupling of reasoning and action; the modal agent pattern.
  - Weaknesses: small mistakes propagate; long horizons drift.
- **Plan-and-Execute** (a.k.a. "plan then act"):
  - Phase 1: generate a numbered plan.
  - Phase 2: execute step by step.
  - Strengths: stronger long-horizon coherence than pure ReAct.
  - Weaknesses: rigid; if the plan is wrong, hard to recover. Often replanned periodically.
- **Reflexion**:
  - On failure, the agent generates a "verbal reflection" ("the bug was X; next time check Y first") and stores it.
  - On retry, the reflection is prepended.
  - Strengths: improves performance over multiple attempts at the same task.
  - Weaknesses: needs a way to detect failure; reflections can be self-serving or wrong.
- **Tree-of-Thoughts**:
  - At each step, generate multiple candidate thoughts; expand best ones; can backtrack.
  - Strengths: search-style problems (puzzles, planning).
  - Weaknesses: expensive (many LLM calls); often overkill.
- **Modern practice** (2024–2026):
  - For most agentic tasks: **a tight ReAct loop with strong tools and clear stopping criteria**. The single-agent loop with good tools beats elaborate planning structures on most benchmarks.
  - Plan-and-Execute appears in long-horizon coding agents (Devin, OpenHands) — typically with replanning.
  - ToT and Reflexion are niche; useful for specific evaluation gains, not the default.

**Common follow-ups.**
- "Why has the field moved away from elaborate multi-agent / planning architectures?" → Anthropic's "Building effective agents" piece — empirically, a strong base model + ReAct + good tools outperforms most complex architectures.
- "When does Tree-of-Thoughts actually help?" → Game-24, crossword puzzles, problems with explicit search structure. Real interview answers should note ToT's high cost.

**Common mistakes.**
- Conflating ReAct (a pattern) with LangChain ReAct (one implementation).
- Citing ToT as a default — it's expensive and rarely beats ReAct in practice.

**References.**
- [Yao et al. — "ReAct"](https://arxiv.org/abs/2210.03629) — ReAct.
- [Wang et al. — "Plan-and-Solve Prompting"](https://arxiv.org/abs/2305.04091) — Plan-and-Execute style.
- [Shinn et al. — "Reflexion"](https://arxiv.org/abs/2303.11366) — Reflexion.
- [Yao et al. — "Tree of Thoughts"](https://arxiv.org/abs/2305.10601) — ToT.
- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — modern taxonomy.

---

### Q: How do you design tools / function schemas for an agent?

**Category:** concept
**Difficulty:** mid
**Tags:** [tool-use, function-calling, schema-design]

**Short answer.** Each tool needs: a **clear name** the model can map intent to, a **rich description** of when to use it (and when *not* to), a **typed argument schema** (JSON schema with constraints), an **idempotent / well-defined** behavior, and **clear error messages** the model can act on. Fewer, more powerful tools usually beat many narrow ones. Test the tools by asking the model to use them on borderline queries and seeing whether the routing decision matches your intent.

**Expansion / why this is the answer.**
- **Tool-naming and description**:
  - The model decides which tool to call largely from name + description.
  - Names should be *verb_noun* and unambiguous: `search_kb` vs. `lookup` vs. `get_data`.
  - Descriptions should include *when to use*, *when not to*, and example inputs.
- **Argument schema**:
  - JSON schema (or Anthropic / OpenAI tool format). Include types, descriptions per field, enums where applicable, required fields.
  - Models follow schemas more reliably when the schema is *simple* — flat objects, few enums, few optional fields.
- **Granularity**:
  - **Too many narrow tools** (`get_user_email`, `get_user_phone`, `get_user_address`) → routing errors.
  - **One overly-broad tool** (`do_anything`) → model has no signal.
  - Heuristic: a tool's description should fit in one sentence.
- **Error handling**:
  - Return structured errors (not stack traces) the model can read.
  - Distinguish recoverable errors ("invalid argument; try again with foo") from terminal ("the API is down; abort").
- **Idempotency where possible**:
  - Re-runnable safely; the model may retry on transient failures.
- **Anthropic Computer Use / browser tools** (2024): a small set of powerful primitives (click, type, screenshot) outperformed elaborate page-specific tools.
- **MCP (Model Context Protocol)** (Anthropic, late 2024): a standard for exposing tools to LLM agents; agnostic of model and harness.

**Common follow-ups.**
- "When do you parallelize tool calls?" → For independent queries (`get_user(id1)` and `get_user(id2)`); Claude and GPT both support parallel tool calls. Don't parallelize sequentially-dependent ones.
- "What's MCP?" → Model Context Protocol — a JSON-RPC-based way to expose tools / resources to a model client. Decouples tool definitions from the harness.

**Common mistakes.**
- Stack-trace error messages — opaque to the model.
- Tool descriptions that say what the tool *is* but not *when to use it*.
- Mandatory arguments the model can rarely provide.

**References.**
- [Anthropic — Tool use docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — primary docs.
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — primary docs.
- [Model Context Protocol spec](https://modelcontextprotocol.io/) — MCP.

---

### Q: Walk me through context-window management in a long-running agent.

**Category:** concept
**Difficulty:** senior
**Tags:** [context-management, compaction, memory, agents]

**Short answer.** As an agent runs, its context grows: each tool call, observation, and reasoning step adds tokens. Without management, you hit the context limit and either crash or silently truncate. Standard techniques: **summarize older turns** (auto-compaction), **retrieve from memory** (semantic search over prior steps), **prune low-signal tool output** (don't keep 100-line stack traces), and **use external memory** (files, KV stores) instead of in-context state. A well-engineered harness usually does all four.

**Expansion / why this is the answer.**
- **Why this matters**: coding agents on a 30-min task can accumulate millions of tokens of tool output (test runs, file listings, errors). At some point, even 1M-context models drop signal.
- **Techniques**:
  - **Automatic compaction**: when context exceeds a threshold (e.g. 70% of the limit), summarize older turns into a concise running summary, then discard the originals. Done well, the model doesn't notice.
  - **Tool-output pruning**: don't include full file contents in subsequent turns; reference by name. Truncate test logs to the relevant lines (the failing test, not the 5,000-line pass output).
  - **External memory**: write important facts to a "notes" file or KV store; have the model retrieve when needed.
  - **Retrieval over history**: embed prior turns; on each step retrieve top-k relevant.
  - **Sub-agent delegation**: spawn a fresh sub-agent for a sub-task; it returns a summary; main agent doesn't see its internals.
- **Pitfalls**:
  - Compaction can drop critical facts; the right summarization prompt matters.
  - Aggressive pruning causes the model to "rediscover" facts it already had.
  - State-machine bugs: the model doesn't know it summarized; references things it can no longer see.
- **Modern frame**:
  - Claude Code, OpenHands, Devin all implement variants.
  - Anthropic harness research distinguishes *short-term context* (what the model sees now) from *durable artifacts* (files, scratchpad, memory).

**Common follow-ups.**
- "What's auto-compaction in Claude Code?" → A built-in summarization step that fires when context fills; preserves the agent's progress while freeing tokens.
- "Do you summarize tool output or the model's reasoning?" → Both, with priority on tool output (it's the bulk).

**Common mistakes.**
- Letting tool output bloat unmanaged.
- Compacting too aggressively — agent loses the thread.
- Treating context as memory.

**References.**
- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — context-management discussion.
- [Park et al. — "Generative Agents: Interactive Simulacra of Human Behavior"](https://arxiv.org/abs/2304.03442) — early memory-management patterns.

---

### Q: Compare SWE-bench Verified, SWE-bench Live, TAU-bench, GAIA, AgentBench. What does each measure?

**Category:** concept
**Difficulty:** senior
**Tags:** [agent-evals, swe-bench, tau-bench, gaia]

**Short answer.** **SWE-bench**: real GitHub issues + repos; agent must produce a patch that passes hidden tests. **SWE-bench Verified**: a human-cleaned subset where the test specifications are unambiguous. **SWE-bench Live**: fresh issues to prevent contamination. **TAU-bench**: customer-support style multi-turn agent tasks (book a flight, modify a reservation) with tool use. **GAIA** (Mialon et al. 2023): general-assistant tasks requiring web browsing, code, multimodal reasoning. **AgentBench** (Liu et al. 2023): a multi-domain benchmark (OS commands, DB, web, knowledge graph, etc.). Each measures different agent capabilities; SWE-bench Verified is the most commonly-cited single number for "is my coding agent good?"

**Expansion / why this is the answer.**
- **SWE-bench** (Jimenez et al. 2023):
  - Real bugs from popular OSS Python repos with hidden test suites.
  - The agent gets the repo state and the issue; must produce a patch.
  - Pass = the patch makes the hidden tests go from fail to pass.
  - Hard: real-world issues are messy.
- **SWE-bench Verified** (OpenAI 2024): human review filtered out unclear / mis-specified tasks; the canonical benchmark cited in model-launch posts (Claude 4.x, GPT-5, Gemini 2.x).
- **SWE-bench Live**: continuously-updated set; resists training-time contamination.
- **TAU-bench** (Yao et al. 2024): two domains (airline, retail) with structured tools and customer dialogues; measures tool-use under realistic uncertainty.
- **GAIA** (Mialon et al. 2023): "general AI assistant" — 3 difficulty levels; tasks require web browsing, multi-step reasoning, file processing.
- **AgentBench** (Liu et al. 2023): 8 environments — OS, DB, web shopping, knowledge graph, lateral-thinking puzzles, code, web browsing, household.
- **WebArena / VisualWebArena** (Zhou et al. 2023): web-based agent tasks in realistic sandboxed sites.
- **OSWorld / WindowsAgentArena**: computer-use agents at the OS level.
- **What an interviewer wants you to know**:
  - The benchmarks measure different things; "my agent is good" needs a specific bench.
  - SWE-bench Verified is the universal coding-agent number.
  - All benchmarks have contamination risk; live variants exist for this reason.

**Common follow-ups.**
- "Why are agent benchmarks harder than LLM benchmarks?" → They involve sequential decisions, environment interaction, evaluation of trajectories (not just outputs).
- "What's the SOTA on SWE-bench Verified right now?" → It moves fast; cite the model + date when answering. Always verify before asserting.

**Common mistakes.**
- Citing "SWE-bench" without "Verified" — different sets; different scores.
- Treating one benchmark's ranking as the universal "best agent."

**References.**
- [Jimenez et al. — "SWE-bench"](https://arxiv.org/abs/2310.06770) — SWE-bench.
- [OpenAI — Introducing SWE-bench Verified blog](https://openai.com/index/introducing-swe-bench-verified/) — Verified.
- [Yao et al. — "τ-bench"](https://arxiv.org/abs/2406.12045) — TAU-bench.
- [Mialon et al. — "GAIA"](https://arxiv.org/abs/2311.12983) — GAIA.
- [Liu et al. — "AgentBench"](https://arxiv.org/abs/2308.03688) — AgentBench.

---

### Q: When does multi-agent beat single-agent? What's the empirical evidence?

**Category:** concept
**Difficulty:** senior
**Tags:** [multi-agent, single-agent, agent-architecture]

**Short answer.** Mostly single-agent wins. Multi-agent helps in narrow cases: (a) **parallelizable sub-tasks** with independent contexts; (b) **role specialization** where a critic / verifier improves correctness; (c) **debate / self-consistency** patterns at high cost. Empirically (Anthropic 2024, "Building effective agents"; OpenAI Practical Guide), the modal winning architecture in 2024–2026 is a *strong single agent with good tools and a tight loop*, not orchestrated multi-agent systems.

**Expansion / why this is the answer.**
- **The case for single-agent**:
  - One context, one decision-maker; no coordination overhead.
  - Modern frontier models are strong enough that the bottleneck is rarely "more cooks."
  - Multi-agent introduces failure modes: hand-off errors, role confusion, infinite-loop dynamics.
- **Where multi-agent is justified**:
  - **Parallel sub-research**: a "research lead" agent delegates `N` sub-queries to `N` parallel sub-agents; each returns a synthesis; main agent integrates. Anthropic's Claude Research / "swarm" patterns. Wins when sub-queries truly are independent.
  - **Verifier / critic**: one model generates, a separate model critiques. Real gains; the critic is often a smaller cheaper model.
  - **Debate / self-consistency**: sample multiple agents, vote; gains on hard reasoning but expensive.
- **Empirical pattern**:
  - Anthropic's "Building effective agents" (2024): warns against multi-agent unless task demands it; single-agent + tools is the default.
  - OpenAI's "Practical guide to building agents" (2024): similar.
  - SWE-bench leaderboard (as of 2026): top scores come from single-agent architectures (or single-agent + verifier), not orchestra-style multi-agent.
- **What an interviewer wants you to know**:
  - You're skeptical of multi-agent by default.
  - You can name the specific cases where it helps.
  - You can talk about coordination overhead.

**Common follow-ups.**
- "What's Anthropic's 'research swarm'?" → Lead agent + parallel sub-agents for independent research queries; ~90% gains on certain bench tasks but >10× the cost.
- "When would you not use a critic / verifier?" → If verification is harder than generation (some creative tasks), or if cost is binding.

**Common mistakes.**
- Treating multi-agent as universally better; the burden of proof is on the multi-agent side.
- Confusing multi-agent with parallel tool calls (different mechanism).

**References.**
- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — single-agent default.
- [Anthropic — "How we built our multi-agent research system"](https://www.anthropic.com/engineering/built-multi-agent-research-system) — when multi-agent does help.
- [OpenAI — "A practical guide to building agents" (PDF)](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) — practitioner guidance.

---

### Q: What is a verifier / critic pattern, and when is it cost-effective?

**Category:** concept
**Difficulty:** mid
**Tags:** [verifier, critic, agent-quality, self-consistency]

**Short answer.** A verifier (or critic) is a separate model — often smaller — that checks the main agent's output: did the patch pass the tests? Did the answer cite valid sources? Is the JSON valid? When verification is cheaper than generation (or has *better-than-generation accuracy* in checking), this is a free quality boost. The bar: verifier must be more reliable than the generator at saying "this is wrong"; otherwise it just adds noise.

**Expansion / why this is the answer.**
- The pattern:
  1. Main agent produces a candidate output.
  2. Verifier scores or judges it.
  3. If "wrong," either retry, ask the agent to revise, or escalate to a stronger model.
- **Programmatic verifiers** (best when available):
  - Code: run the unit tests.
  - Math: a calculator / SymPy / a known solution checker.
  - SQL: execute the query against a sandbox DB.
  - Schema: JSON schema validation.
- **LLM verifiers**:
  - LLM-as-judge: another model reads the output and the criteria, returns a verdict.
  - Same model with a different prompt ("now critique this answer"): self-verification, often weaker.
  - Smaller / cheaper model as verifier: usually fine if verification is structurally easier.
- **Asymmetric verification** (Yao et al. 2023, etc.):
  - For many tasks, verifying is easier than generating ("is this code's test passing?" vs. "write the code").
  - This is *why* verifier patterns work.
- **Where verifier doesn't help**:
  - Open-ended creative tasks (no truth signal).
  - Domains where the model is uncalibrated as a judge (e.g. domain expertise the model lacks).
- **GRPO**'s relation: GRPO RL needs a verifier as the reward function — programmatic verifiers power most reasoning-RL pipelines.

**Common follow-ups.**
- "Why are programmatic verifiers preferred over LLM judges?" → Cost, determinism, no propagated bias.
- "How do you train a verifier?" → For domains lacking programmatic checkers, fine-tune a model on `(output, correctness label)` pairs.

**Common mistakes.**
- Using the same model as both generator and verifier without isolation; it agrees with itself.
- Treating the verifier as ground truth; verifier errors compound.

**References.**
- [Lightman et al. — "Let's Verify Step by Step"](https://arxiv.org/abs/2305.20050) — process-reward / step-level verifier.
- [Cobbe et al. — "Training Verifiers to Solve Math Word Problems"](https://arxiv.org/abs/2110.14168) — math verifier.
- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — verifier patterns.

---

### Q: Walk me through agent failure modes and how to debug them.

**Category:** concept
**Difficulty:** senior
**Tags:** [debugging, failure-modes, agent-reliability]

**Short answer.** Common failure modes: (a) **wrong tool** picked (routing failure); (b) **bad arguments** passed (schema-failure); (c) **infinite loop / repetition**; (d) **drift** (agent forgets the goal); (e) **fabricated observations** (model makes up tool output instead of calling); (f) **giving up** (premature stop); (g) **non-recovery from errors**; (h) **lost progress on context overflow**. Debug by **trace inspection**: log every (state, thought, action, observation) tuple, replay locally, isolate the failing step.

**Expansion / why this is the answer.**
- **Wrong tool routing**: model picks `search_docs` when it should call `query_db`. Fix: tool descriptions emphasizing when-to-use; fewer overlapping tools; eval on routing decisions.
- **Bad arguments**: model passes string where int expected; missing required field; nonsense values. Fix: strict schema, structured output (constrained decoding), good error messages from the tool.
- **Infinite loop**: model repeats the same action — common when the observation is unchanged. Fix: max-step limit; loop detection (compare current state to recent states); "you've done this before, try something else" intervention.
- **Drift**: long task; model forgets the original goal. Fix: keep a goal-marker at the top of context; periodically remind.
- **Fabricated observations**: in trace logs you'll see `Action: tool(...)` followed by an `Observation:` the model wrote itself, with no actual tool call. This is bad. Fix: enforce strict tool-call format; verify every observation came from a real tool execution.
- **Premature stop**: agent says "done!" when not. Fix: explicit acceptance criteria the agent must show evidence for.
- **Non-recovery from errors**: tool returns error; agent doesn't adapt. Fix: agent training (RLHF) to handle errors; explicit "if you see an error, try X" prompting.
- **Context overflow**: see context-management question.
- **Debugging**:
  - **Trace logging**: full (thought, action, args, observation) per step. Required.
  - **Replay**: at each failure, can you reproduce by running the same trace forward from step N?
  - **Eval harness**: a small set of failing cases; run continuously to detect regressions.

**Common follow-ups.**
- "How do you detect a stuck agent in production?" → Watch step count, action-repetition rate, latency outliers; abort and surface to human.
- "How do you handle fabricated observations?" → Force tool-call mode (the API rejects model-written observations); validate every observation came from a registered tool.

**Common mistakes.**
- Trusting end-to-end output without tracing.
- Treating every retry as recovery; sometimes retries amplify the same bug.

**References.**
- [Yao et al. — "ReAct"](https://arxiv.org/abs/2210.03629) — trace structure.
- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — reliability patterns.

---

### Q: How do you control cost and latency in an agent loop?

**Category:** concept
**Difficulty:** mid
**Tags:** [agent-cost, agent-latency, optimization]

**Short answer.** Levers: (a) **step cap** (max iterations); (b) **smaller model on cheap turns** (cascade); (c) **parallel tool calls** where possible; (d) **prefix caching** of long system prompts; (e) **scope the tool-output** the model sees (truncate, summarize); (f) **early-exit conditions** ("if the answer is found, stop"); (g) **batch / dedupe** repeated subqueries. The dominant cost in most agents is *long contexts on later turns* — context management is the biggest lever.

**Expansion / why this is the answer.**
- **Cost decomposition**: cost ≈ Σ_steps (prefill tokens + decode tokens). Prefill grows with context; decode grows with model size and response length.
- **Levers in detail**:
  - **Step cap**: prevents runaway loops; combine with explicit "give up gracefully if blocked" instructions.
  - **Model cascade**: simple route → small model; complex reasoning → large. Big savings if the small model handles 70%+ of turns.
  - **Parallel tool calls**: where tools are independent (lookup user A and user B); reduces wall-clock latency.
  - **Prefix caching**: a 5k-token system prompt across 20 turns: 5k × 20 = 100k prefill if uncached, 5k × 1 + ~5k incremental if cached. Huge.
  - **Tool-output scoping**: don't include full file contents; reference by name; surface only the relevant lines.
  - **Compaction**: summarize older turns periodically.
  - **Caching deterministic tool results**: idempotent queries don't need to re-run.
- **Latency**:
  - First-token-latency per step is dominated by prefill.
  - Streaming output back to the user when terminal can hide late-stage latency.
- **In practice**, the modal optimizations are: prefix caching for system prompts, tool-output truncation, and compaction.

**Common follow-ups.**
- "What's a cascade?" → Cheap model decides whether to escalate to expensive model. Used in routing / triage agents.
- "How do you keep latency low on a 20-step task?" → Stream intermediate progress to the user; keep cumulative context small.

**Common mistakes.**
- Optimizing model size and not context length — context often dominates.
- No step cap; agents run forever on impossible tasks.

**References.**
- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — prefix caching for system prompts.
- [Patel et al. — "Splitwise"](https://arxiv.org/abs/2311.18677) — latency-aware serving.

---

### Q: What is MCP (Model Context Protocol), and why is it relevant?

**Category:** concept
**Difficulty:** mid
**Tags:** [mcp, tool-protocol, interoperability]

**Short answer.** MCP is an open standard (introduced by Anthropic, late 2024) for connecting LLMs to tools, resources, and prompts through a uniform protocol — JSON-RPC over various transports. It lets developers expose tools once and have any MCP-compliant client (Claude Desktop, Cursor, Continue, etc.) use them. Relevant because the alternative is per-vendor tool integration; MCP makes the tools-side ecosystem portable.

**Expansion / why this is the answer.**
- **The problem MCP solves**: every LLM vendor had its own tool API (OpenAI function-calling, Anthropic tool-use schema, Gemini tool config). Tool authors had to write N integrations.
- **MCP architecture**:
  - **Server**: exposes tools, resources, prompts. JSON-RPC over stdio or HTTP+SSE.
  - **Client**: an LLM-facing application (Claude Desktop, Cursor, etc.) that connects to one or more servers.
  - **Protocol primitives**: tools (functions), resources (data), prompts (parameterized prompts).
- **Adoption**:
  - Claude Desktop has first-class support.
  - Cursor, Continue, Codeium support MCP servers.
  - Hundreds of community-built servers (filesystem, git, GitHub, databases, browsers).
- **Why an interviewer cares**:
  - It's the answer to "how do I let my agent use my company's tools without rebuilding the integration for every model?"
  - It's an active standard with multi-vendor adoption.
- **Caveats**: still evolving (2024–2026 timeline); auth and discovery are works-in-progress.

**Common follow-ups.**
- "Is this just a function-call standard?" → It also defines resources (read-only data the model can browse) and prompts (parameterized templates) — broader than just tool use.
- "What's the alternative?" → Per-vendor SDK + custom integration code; or LangChain/LlamaIndex-style frameworks that abstract over vendors.

**Common mistakes.**
- Conflating MCP with LangChain (both expose tools, but MCP is a wire protocol; LangChain is a Python framework).
- Treating MCP as locked to Claude; it's open and growing across clients.

**References.**
- [Model Context Protocol spec](https://modelcontextprotocol.io/) — primary.
- [Anthropic — Introducing MCP](https://www.anthropic.com/news/model-context-protocol) — announcement post.

---

### Q: How does an agent know when to stop?

**Category:** concept
**Difficulty:** mid
**Tags:** [stopping-criteria, termination, agents]

**Short answer.** Combination of: (a) **explicit terminal action** the model emits when the goal is met (`finish(answer=...)`); (b) **acceptance criteria check** — programmatic or LLM-judge; (c) **step cap** as a hard upper bound; (d) **time budget** as a watchdog; (e) **stuck detection** (no new state across N steps). A well-designed harness uses all five — the model's self-declared "done" is necessary but not sufficient.

**Expansion / why this is the answer.**
- **Self-declared "done"**:
  - The simplest pattern: agent emits a terminal action like `finish(answer)` or `done()` and the harness returns control.
  - Failure mode: agent declares done too early (premature stop) or never declares done (runaway).
- **Programmatic acceptance**:
  - Coding tasks: do the tests pass? If yes, done.
  - Form-fill tasks: are all required fields populated? If yes, done.
  - The strongest signal when available.
- **LLM-judge acceptance**:
  - Use a separate model to check whether the goal is met.
  - Cost; can be wrong.
- **Step cap / time budget**:
  - Hard upper bound; surface to a human if hit.
- **Stuck detection**:
  - Compare current state to recent — if action+observation pairs repeat, intervene.
- **Anthropic Computer Use, OpenAI Operator** both implement variants — a terminal action plus a watchdog.

**Common follow-ups.**
- "What happens when the model declares done but the verifier says no?" → Replay with a prompt that explains what's missing; or escalate to a stronger model.
- "How do you handle 'I don't know' as a valid done state?" → Allow `give_up(reason)` as a terminal action; preferable to thrashing.

**Common mistakes.**
- Step cap as the only stop criterion — agent runs the full budget every time even on easy tasks.
- No giveup state — agent loops forever on impossible tasks.

**References.**
- [Anthropic — Computer Use docs](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) — terminal-action pattern.
- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — stopping criteria.

---

### Q: How do you build evals for an agent?

**Category:** concept
**Difficulty:** senior
**Tags:** [agent-eval, trajectories, eval-design]

**Short answer.** Agent evals are harder than LLM evals because they evaluate **trajectories**, not outputs. Build: (a) a **task set** of `(initial state, goal, success criterion)`; (b) a **deterministic environment** (sandboxed tools, fixed corpora); (c) **per-trajectory metrics** — success rate, steps to success, cost, latency, tool-call accuracy; (d) **trace logging** so failures are debuggable; (e) **regression tracking** — same eval, same env, every code change. SWE-bench and TAU-bench are the canonical templates.

**Expansion / why this is the answer.**
- **Why agent evals are hard**:
  - Non-determinism: stochastic models, environment side effects.
  - Path-dependence: many "right" trajectories.
  - Cost: each eval task is a full agent loop.
- **Design template**:
  - **Task spec**: clear goal, well-defined success.
  - **Environment**: sandboxed, reproducible. SWE-bench: a snapshot of the repo. TAU-bench: a typed customer-support state machine.
  - **Success criterion**: programmatic where possible (tests pass / form filled / DB state correct).
  - **Metrics**:
    - Success rate.
    - Steps to success / cost per task.
    - Tool-call accuracy (right tool, right args).
    - Trace length / context size at failure.
    - Specific failure-mode counts (hallucinated observation, infinite loop, etc.).
- **Trace logging**: every step's (state, thought, action, observation) recorded. Lets you re-run and slice.
- **Regression discipline**:
  - Pin the eval task set.
  - Pin the environment (e.g. fixed seed where possible).
  - Compare against the prior best by metric.
  - Add new task types as the agent matures.
- **Online evaluation**: shadow real user traffic; compare assistive vs. baseline.

**Common follow-ups.**
- "How do you handle stochastic environments?" → Run each task multiple times; report mean ± std.
- "What's a 'process reward' eval?" → Score *intermediate* steps, not just the final answer. Useful for credit assignment but expensive to label.

**Common mistakes.**
- Single-number eval (just "success rate") — misses where the agent fails.
- No environment determinism — can't reproduce, can't debug.

**References.**
- [Jimenez et al. — "SWE-bench"](https://arxiv.org/abs/2310.06770) — canonical agent eval design.
- [Yao et al. — "τ-bench"](https://arxiv.org/abs/2406.12045) — multi-turn structured eval.

---

### Q: What's the difference between a workflow and an agent? When do you pick which?

**Category:** concept
**Difficulty:** mid
**Tags:** [workflow, agent, architecture]

**Short answer.** **Workflow**: developer-defined sequence of LLM and non-LLM steps (deterministic flow). **Agent**: model-defined sequence — the model decides the next action at each step. Workflows are predictable, debuggable, and cheap; agents are flexible but expensive and harder to verify. Anthropic's taxonomy: start with a workflow; only introduce an agent when the task is open-ended enough that the path can't be hardcoded.

**Expansion / why this is the answer.**
- **Workflow patterns** (Anthropic's "Building effective agents"):
  - **Prompt chaining**: A → B → C, each step's output is the next step's input.
  - **Routing**: classify the request, then route to a specialized prompt/model.
  - **Parallelization**: split into independent sub-tasks; merge.
  - **Orchestrator–workers**: a higher-level model assigns work to specialized workers.
  - **Evaluator–optimizer**: a generator and a critic in a loop.
- **Agent**:
  - A loop where the model decides each next action.
  - Use cases: open-ended problem-solving, multi-step debugging, "fix this codebase issue."
- **When to pick which**:
  - **Workflow** if the path is known: data-pipeline-style tasks, structured-output transformations, RAG-style Q&A.
  - **Agent** if the path is open: SWE tasks, web exploration, multi-step debugging.
- **Cost**: workflow steps are predictable; agent loops can balloon (10× the cost on hard cases).
- **Debuggability**: workflow failures are easy to localize; agent failures need trace inspection.

**Common follow-ups.**
- "Can you combine them?" → Yes — agent at the top, workflows for known sub-tasks; or vice versa.
- "Is RAG a workflow or an agent?" → Vanilla RAG is a workflow. "Agentic RAG" (decide-when-to-search, multi-step retrieval) is an agent on top of retrieval.

**Common mistakes.**
- Defaulting to "agent" because it's more impressive. Workflows are cheaper and more reliable for fixed paths.
- Treating LangChain agents as the only way to do agents; the pattern is independent of the framework.

**References.**
- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — workflow vs agent taxonomy.
- [OpenAI — "A practical guide to building agents"](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) — practical comparison.

---

### Q: Walk through the design of a coding agent. What are the hard problems?

**Category:** system-design
**Difficulty:** senior
**Tags:** [coding-agent, system-design, swe-bench]

**Short answer.** A coding agent needs: (a) **strong code-aware base model** (RLHF for coding tasks); (b) **tools** for reading files, writing files, running tests, executing code, searching the codebase; (c) **a tight loop** with verification (run the tests, observe the result); (d) **context management** because codebase exploration generates lots of tokens; (e) **eval-driven iteration** against SWE-bench Verified or an internal benchmark. Hard problems: long-horizon planning, knowing what files to read, recovering from test failures, not making the bug worse.

**Expansion / why this is the answer.**
- **Core architecture**:
  - Base model: a strong code-trained LLM (Claude 4, GPT-5, DeepSeek-Coder, Llama).
  - Loop: ReAct or Plan-and-Execute-with-replanning.
  - Tools:
    - `read_file(path, line_range)`.
    - `write_file(path, content)` or `edit_file(path, search, replace)`.
    - `run_tests(test_filter)`.
    - `run_python(code)` for exploration.
    - `grep_codebase(pattern)`.
    - `list_files(directory)`.
- **Verification loop**: every change runs the tests; the agent reacts to test output.
- **Context management**:
  - Don't include full file contents on every turn; load on demand.
  - Summarize prior steps when context grows.
  - "Notebook" / scratchpad pattern: write progress to a file.
- **Hard problems**:
  - **Knowing what to read**: in a 100k-file codebase, the agent must triage. Codebase indexing (vector + grep + tree) helps.
  - **Long-horizon coherence**: on multi-hour tasks, the agent drifts. Periodic replan + summary.
  - **Recovering from regressions**: every change has the potential to break other tests; the agent must read the broader test output and not just the targeted test.
  - **Knowing when to stop**: tests pass; check the change is reasonable; don't accidentally "fix" by deleting the test.
- **Eval-driven**: SWE-bench Verified as the regression bench; internal hand-curated tasks for capability drift.
- **Production systems**: Claude Code, Devin, OpenHands, Cursor agent mode, GitHub Copilot Workspace, Aider. All converge on similar architectures.

**Common follow-ups.**
- "What's the most impactful improvement in SWE-bench scores from 2023 to 2026?" → A combination of stronger base models + better tools + tighter loops; less so elaborate multi-agent architectures.
- "How do you stop the agent from cheating (deleting the test)?" → Tests are read-only in many harnesses; or post-hoc check for test-file modifications.

**Common mistakes.**
- Trying to plan the whole solution upfront; coding tasks are too discovery-heavy.
- Not running tests after every change.

**References.**
- [Jimenez et al. — "SWE-bench"](https://arxiv.org/abs/2310.06770) — coding-agent eval.
- [Anthropic — Claude Code overview](https://docs.claude.com/en/docs/claude-code/overview) — coding agent design.
- [OpenAI — SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — verified subset.

---
