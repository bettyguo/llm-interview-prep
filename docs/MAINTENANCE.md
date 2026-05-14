# Maintenance plan

The 2026 LLM landscape drifts fast — staleness is a known risk and visible maintenance is the mitigation. This file documents the cadence.

## Cadence

- **Weekly**: scheduled `linkcheck` workflow runs Mondays 09:00 UTC; failures open an issue.
- **Monthly**: review open `wrong-answer` and `new-question` issues; merge or close with rationale.
- **Quarterly content review**: re-read T2 (Transformers), T3 (Training), T4 (Inference), T6 (Agents) end to end for staleness. Update model-specific facts (model sizes, default hyperparameters, benchmark numbers). Bump `CHANGELOG.md` and the README "last updated" badge naturally updates from commit time.
- **Annually**: re-baseline the competition map (`PLANNING/00_think.md`). If a new entrant has surpassed this repo, document it and either match or differentiate.

## Triage rules

| Issue type | SLA | Notes |
|------------|-----|-------|
| `wrong-answer` | 7 days to acknowledge; fix or rebut within 30 days | Highest priority — wrong answers harm the user. |
| `broken-link` | 14 days | Fix the link or replace the reference. |
| `new-question` | 30 days to triage | Merge if accompanied by a verifying reference; otherwise leave open with a "needs source" label. |

## What constitutes staleness

- A cited paper has been superseded by a paper the field now treats as the standard reference (e.g. Kaplan → Chinchilla).
- A framework default has changed (e.g. PyTorch's default attention impl now SDPA-fused).
- A benchmark has been retired or known-contaminated.
- A best-practice has shifted (e.g. RLHF → DPO for many use cases).

## Hand-off

If the curator can no longer maintain the repo, the maintenance cadence above must be respected by the inheriting maintainer, or the maintenance-promise section of the README must be revised honestly. Do not let staleness sit while the badge still says "last updated."
