# D8 — Multi-turn enterprise LLM agent

A worked drill following the [6-step structure](../README.md#the-canonical-6-step-structure).

---

### Q: Design a multi-turn LLM agent that executes a business workflow (e.g. process an expense report end-to-end).

**Category:** system-design
**Difficulty:** senior
**Tags:** [agent, enterprise, tool-use, audit-trail]

**Short answer.** Single-agent loop with strict tool schemas, a strong base LLM, and clear stopping criteria. Provide structured tools for each business action (`fetch_receipt`, `categorize`, `check_policy`, `submit_for_approval`). Pair every state-mutating action with a verifier and an audit log. Require human approval gates for high-stakes actions (over-budget submissions). Heavy emphasis on safety, auditability, and reliability — enterprise agents fail differently than consumer ones because the failure mode is direct business risk.

**Expansion / why this is the answer.**

**1. Clarify requirements.**
Functional:
- User submits an expense report (or it's pulled from email/Slack).
- Agent: parse receipts, categorize, check policy, route for approval, submit to finance system.
- Multi-turn clarification when the receipt is ambiguous.

Non-functional:
- Latency: not real-time-tight; minutes are acceptable.
- Accuracy: errors are costly (mis-submitted expenses, audit findings).
- Auditability: every action logged with reasoning.
- Compliance: SOX, GDPR, internal policy.
- Reliability: must finish or fail-safe — no silent half-completion.

Clarifying Qs:
- "What's the existing finance system the agent integrates with?"
- "Is human approval required for every submission, or only above a threshold?"
- "What's the policy doc — is it parseable, or text?"

**2. Define metrics.**

Online:
- **Success rate**: % expense reports completed without human escalation.
- **Time to completion** vs. baseline manual.
- **Error rate**: post-submission corrections, audit findings.
- **User satisfaction**: NPS-style.

Offline:
- **Per-step success**: did each tool call succeed as intended?
- **End-to-end task success** on a synthetic / replay suite.
- **Tool-call accuracy**: right tool, right args.
- **Cost / latency per task**.

**3. Data and labels.**

Sources:
- Past expense reports (with PII scrubbed for training; not at all for closed-API inference).
- Policy documents.
- Approval logs.
- Audit findings (high-signal negatives).

Eval set:
- 100–500 representative task instances.
- Synthetic edge cases (mis-categorized, over-budget, missing receipts).
- Adversarial cases (potential fraud attempts).

**4. Modeling.**

**4a. Architecture**: single-agent loop (Anthropic's "Building effective agents" guidance — don't go multi-agent unless needed).

**4b. Base model**: a strong frontier model (Claude, GPT-class); cheap escalation possible for simple sub-tasks.

**4c. Tools** (each with clear schema + idempotency):
- `parse_receipt(image_or_pdf) → structured`
- `categorize_expense(parsed) → category`
- `check_policy(expense, policy_kb) → violation_or_none`
- `lookup_employee_budget(employee_id) → remaining`
- `submit_to_finance(expense) → submission_id` (state-mutating, requires verifier)
- `request_human_approval(expense, reason) → approval_or_denial` (escalation)

**4d. System prompt**:
- Role: expense-processing agent.
- Policy: when to escalate (over-budget, unusual category, missing receipt).
- Format: structured tool calls; cite policy section for any rejection.
- Refusals: never auto-submit if any ambiguity remains.

**4e. Memory / context management**:
- Each task is short-horizon (minutes); compaction less critical than in coding agents.
- Per-task working memory (the receipts, categorizations).
- Cross-task institutional memory (this employee tends to spend on X, recurring vendors).

**4f. Verifier / human gate**:
- For every state-mutating action: verifier reads the proposed change and confirms.
- Above-threshold expenses → mandatory human review.
- All actions logged with the agent's reasoning trace.

**5. Serving.**

Latency: per-step a few seconds; total task minutes.
- LLM API or self-hosted.
- Tool execution sandbox.
- Workflow orchestrator (Temporal, LangGraph, internal).

Reliability:
- Resumable tasks: if the agent fails mid-task, can resume from the last checkpoint.
- Timeout per step + escalate on timeout.

Security:
- Tools authenticated as the user (not over-privileged service accounts).
- Sandbox: agent can't break out of its scope.

Privacy:
- PII redacted in logs.
- No data egress for enterprise-on-prem.

**6. Monitoring + iteration.**

Per-step metrics:
- Tool-call success rate per tool.
- Escalation rate (target: low but not zero).
- Time per task.

Audit trail:
- Every action logged with the agent's reasoning.
- Periodic spot-audit by humans.

Drift:
- Policy updates → re-prompt or fine-tune.
- New vendor / category patterns.

Failure modes:
- **Hallucinated policy citations**: agent cites a policy that doesn't exist. Verifier checks.
- **Fraud attempts**: malicious users gaming the agent (split receipts, etc.). Anomaly detection.
- **Stuck states**: agent loops on an unparseable receipt. Escalate after N retries.

User feedback:
- Approvers can flag bad agent decisions.
- Feedback feeds into the eval set.

---

**Common follow-ups.**

- "Why single-agent over multi-agent?" → Anthropic guidance: single-agent + tools beats multi-agent unless task demands it. Expense processing fits in single-agent.
- "What if a tool call fails?" → Retry with backoff; if persistent, escalate to human.
- "How do you handle the policy doc?" → Either RAG against it (citable, easy to update) or distilled into the system prompt (lower latency, harder to update).

**Common mistakes.**

- No human-in-the-loop on state-mutating actions.
- No audit trail.
- Letting the agent take action it can't undo without escalation.
- Skipping verifier patterns.

**References.**

- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — single-agent + tools.
- [Yao et al. — "ReAct"](https://arxiv.org/abs/2210.03629) — reasoning + acting loop.
- [LangGraph docs](https://langchain-ai.github.io/langgraph/) — workflow orchestration.
- [Temporal — Workflow as Code](https://temporal.io/) — durable execution.
