# Contributing to llm-interview-prep

Thanks for considering a contribution. This repo's value comes from being **correct**, **comprehensive**, and **maintained** — and you can help with any of those.

## What we welcome

- **New questions** — especially questions you were actually asked in a recent interview loop. Include the topic, the question verbatim, a short answer, and an authoritative reference.
- **Corrections to existing answers** — please cite an authoritative source (paper, primary docs, textbook) that supports your correction. See the `wrong-answer` issue template.
- **Resource additions** — links to high-quality further reading (original papers, official docs, widely-cited surveys).
- **Tooling and CI improvements.**
- **Study-plan refinements** — especially from people who have just been through a loop.

## What we do not want

- Bullet-only "answers" with no expansion.
- Marketing for a course, bootcamp, or product. Reference links go to primary sources; affiliate links and bootcamp funnels will be removed.
- Answers without verifying references for category ∈ {concept, derivation, system-design}. The validator will reject them, and the reviewer will close the PR.
- Mass copy-pastes from other interview repos (CC-BY-4.0 is permissive but copies still must be re-verified and attributed).
- LLM-generated answers that have not been fact-checked by the contributor. (LLM-assisted *drafting* is fine, but the contributor is responsible for correctness.)

## The Q&A entry schema

```markdown
### Q: <the question as it would be asked>

**Category:** concept | derivation | system-design | coding | behavioral
**Difficulty:** intro | mid | senior | staff
**Tags:** [comma, separated, tags]

**Short answer.** <1–3 sentences, the opener>

**Expansion / why this is the answer.**
<the walkthrough — math, contrasts, the *because*>

**Common follow-ups.**
- <follow-up>

**Common mistakes.**
- <mistake that signals shallow understanding>

**References.**
- [Title](https://url) — what it supports.
```

For `coding` entries replace `References` with `**Implementation.**` containing a fenced code block.
For `behavioral` entries replace `References` with `**Signal.**` describing what an interviewer is listening for.

## Local checks before opening a PR

```bash
pip install -r tools/requirements.txt
python tools/validate_entries.py --stats
python tools/build.py
python tools/linkcheck.py     # optional locally; CI runs it weekly
```

The validator is intentionally strict. If you think a rule is wrong, open an issue and discuss before bypassing.

## PR review bar

A reviewer must spot-check at least one new/changed answer against its cited reference before approving. This is the answer-correctness protocol; please don't sidestep it.

## Code of conduct

Be kind. Disagree on substance with sources, not tone. Reports to the curator (Betty Guo) via the issue tracker.

## License

By contributing you agree to license your contribution under the same terms as the repo: CC-BY-4.0 for content, MIT for tooling.
