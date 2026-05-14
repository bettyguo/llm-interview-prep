# Flashcard mode — spec

`system-design-primer` ships an Anki deck. That delivery feature is half the reason it accumulated 300k+ stars. This document specs how to add the same to `llm-interview-prep` post-launch.

This is a **spec**, not yet built. Implementation deferred to a Phase 5+ follow-up release.

---

## Why

The audience for this repo splits roughly into two:

1. **Deep-readers** who want the full worked answer (the current product).
2. **Spaced-repetition learners** who want flashcards for retention.

The current schema was *designed* to allow flashcard generation: every entry has a `Short answer` (front-of-card) and an `Expansion` (back-of-card). The export is just a transformation.

## What the deck contains

For each `### Q:` entry across all topics:

- **Front**: the question text.
- **Back**: the `Short answer` (1–3 sentences). Optionally include the first 200 words of `Expansion` as supplementary.
- **Tags**: `Tags:` field from the entry, plus the topic name.
- **References**: an indicator that the full answer is in the source repo, with the topic + question slug.

**Optional advanced deck** (for senior candidates): include the `Common follow-ups` and `Common mistakes` as additional fields, so reviewing one flashcard tests the full conversation flow.

## What it does not contain

- Coding entries with multi-line code blocks (poor flashcard UX). Link to the snippet file instead.
- System-design drills (too long for flashcards). Link to the drill file instead.
- Behavioral entries (subjective; flashcards don't fit). Skip.

## Format

**Output**: `.apkg` (Anki package) file, plus a `.csv` fallback for non-Anki users (Quizlet, Mochi, RemNote import).

**Deck name**: `llm-interview-prep / <topic-name>` so users can subset by topic.

**Card style**: minimalist; monospace for code; rendering of LaTeX where used in answers (`$...$`).

## Tool

A `tools/build_anki.py` script:

```python
# Spec only — implementation deferred.
#
# 1. Walk prep/*/questions.md
# 2. For each entry, extract: question, short_answer, expansion, tags, category.
# 3. Skip if category in {coding, system-design, behavioral} (per above).
# 4. Generate Anki note via genanki or anki-cards.
# 5. Output: dist/llm-interview-prep.apkg
# 6. Output: dist/llm-interview-prep.csv
```

Dependencies:
- `genanki` (Apache-2.0) — the standard Anki-package generator in Python.

## Update cadence

Re-generate the deck on every CHANGELOG release. Bundle in GitHub Releases.

## License

The deck inherits CC-BY-4.0 (content license). Attribution required in any redistribution.

## Open questions

- **Image support**: do we include schematic images (e.g. attention diagram)? Adds complexity, may help retention. Probably defer.
- **Spaced-repetition schedule**: Anki's default works; we don't ship a custom schedule.
- **Localization**: deck is English-only initially. Translations welcome as separate decks.

## What this gets us

A second, complementary delivery channel that catches the spaced-repetition-learner audience. Historical precedent (`system-design-primer`'s Anki deck) suggests this is meaningful for compounding usage.

Until built, the user-facing pitch is: "the source schema is structured for Anki export — coming soon. PRs welcome."
