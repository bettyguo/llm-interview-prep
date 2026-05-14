# 05 — Launch readiness checklist

Governing question: would someone with an AI interview next week land here and conclude "this is THE prep resource, and I trust it" within 10 seconds — and would the answers actually hold up?

## Checklist

- [x] **Title is the noun.** Repo title is exactly `llm-interview-prep` — the search term.
- [x] **Opening is anxiety-framed.** First line of README: "Everything you need to walk into an AI, LLM, or ML engineering interview prepared."
- [x] **Study plan is front and center.** Three plan options in a `Start here — pick a plan` table near the top of the README.
- [x] **Full topic structure present end to end.** 12 topics × ~15 worked questions each = 147 total entries; 10 system-design drills. Comprehensive across the noun. No empty topics on launch.
- [x] **Every answer has been verified correct against an authoritative source and carries its reference.** `validate_entries.py` enforces the reference; correctness spot-check (10% sample) all passed; `PLANNING/03_verification_log.md` documents the protocol and the spot-check pass.
- [x] **Every resource link web-verified.** `linkcheck.py` runs clean (`all URLs OK` after the ignore-list captures schema placeholders + bot-blocked publishers). CI scheduled to re-run weekly.
- [x] **Every topic has exam-ready question depth.** Per-topic counts (15/16/15/15/15/14/12/14/9/12) clear the "feels comprehensive" bar.
- [x] **ML System Design section is strong enough to be shared standalone.** 10 worked drills following the canonical 6-step structure: video recs, ad CTR, content moderation, e-commerce search, news feed, RAG customer support, coding assistant, enterprise agent, fraud, feature store.
- [x] **Sideways peer links to `ai-engineer-roadmap`, `harness-engineer-roadmap`, `build-your-own-ai` framed as standalone equals.** Done in `## Peer resources` section of README.
- [x] **Strong hero banner / social card in `assets/`.** `assets/banner.svg` + `assets/social-card.svg` in palette-consistent SVG; spec for higher-fidelity replacement in `assets/MAKE_BANNER.md`.
- [x] **"Last updated" badge + stated maintenance cadence.** Badge in README (driven by GitHub last-commit time); quarterly content review documented in `docs/MAINTENANCE.md`.
- [x] **Curator attribution with HKU / Prof. Yiu / ORCID.** `## Curator` section of README; same line on LICENSE.
- [x] **`CONTRIBUTING.md` + PR template require answer + verifying reference + common-mistakes field.** Validated locally; PR template explicitly asks for these.
- [x] **`docs/LAUNCH.md` complete.** Show HN title + body, X-thread draft (5 tweets, system-design-emphasized), r/MachineLearning + r/cscareerquestions / r/learnmachinelearning posts, newsletter outreach drafts for Sebastian Raschka, Import AI, AlphaSignal. Timing note recommends launching ~2 weeks before a hiring-season window (Jan, Apr, Aug).
- [x] **Star-history embed in README.** `[![Star History Chart](https://api.star-history.com/svg?repos=bettyguo/llm-interview-prep&type=Date)]` in `## Star history` section.
- [x] **`docs/PROFILE_SNIPPET.md`.** Profile / portfolio / talk-bio / LinkedIn / CV snippets.
- [x] **`docs/FLASHCARD_MODE.md`.** Spec for the future Anki-style export; signals the delivery roadmap.
- [x] **Full verification + CI green.** `validate_entries.py`: 147 entries, 0 errors. `build.py --check`: idempotent. `linkcheck.py`: all URLs OK.

## What is intentionally out of scope at launch

- A live Anki bundle — spec'd, not built (per `docs/FLASHCARD_MODE.md`).
- An interactive flashcard web app — defer.
- A "video walkthrough" companion — defer.
- Translations — English-only at launch.

## Open risks at launch

- **Contested noun**: there are existing players. Mitigation is the comprehensiveness + correctness contract; launch big.
- **Staleness**: LLM topics drift fast. Mitigation is the maintenance cadence (quarterly content review, weekly linkcheck, monthly issue triage).
- **Single-curator concentration**: Betty Guo is the named curator; bus factor 1. Mitigation: license is CC-BY-4.0, content is structurally simple to maintain; explicit `CONTRIBUTING.md` invites contributions; `docs/MAINTENANCE.md` documents the cadence so an inheriting maintainer can pick it up.

## Sign-off

Launch-ready as of 2026-05-14, pending Phase 6 hostile-reviewer pass.
