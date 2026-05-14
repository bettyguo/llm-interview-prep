# site/ — supplementary mock-interview UI

A static, single-page study app that sits on top of the question bank in [`prep/`](../prep/). Five modes, all driven by `questions.json` (auto-generated from the markdown):

| Mode | What it's for |
|------|---------------|
| **Browse** | Filter / search 209 questions; click to expand the full answer with follow-ups, mistakes, and references. |
| **Flashcards** | Front: question + tags. Back: short answer + expansion. Rate yourself; Leitner-style spaced repetition (1d → 3d → 7d → 14d → 30d) stored in `localStorage`. |
| **MCQ Quiz** | Auto-generated 4-option questions: the correct option is the first sentence of the short answer, the three distractors are pulled from the entry's documented **Common mistakes**. Tracks per-session and lifetime accuracy. |
| **Mock Interview** | Pick topic + difficulty + count + per-question timer. Sequential timed questions with a notes textarea, model-answer reveal, self-rating (1–5), and a per-topic scorecard at the end. |
| **Cloze** | Key terms in the short answer are blanked. Click to reveal — useful for cementing acronyms, formulas, and named concepts. |

Everything is local: no analytics, no backend, no account. Progress lives in your browser's `localStorage` under the key `lip_state_v1`.

## How it works

```
tools/build_site.py        # parses prep/*/questions.md → site/questions.json + questions.js
site/index.html            # SPA shell (5 mode tabs)
site/app.js                # logic for all five modes
site/styles.css            # styling, light + dark via prefers-color-scheme
site/questions.js          # auto-generated; loads window.QUESTIONS_DATA
site/questions.json        # auto-generated; same payload as JSON
```

`questions.js` exists so the site opens straight from `file://` without a local server. `questions.json` is the parallel artifact for any downstream tool that wants structured data.

## Running locally

Easiest — open the file directly:

```
# from the repo root
python tools/build_site.py
# then open site/index.html in any modern browser
```

Or via a local server (recommended on macOS/Linux where `file://` security policies vary):

```
python tools/build_site.py
cd site && python -m http.server 8000
# open http://localhost:8000
```

## Regenerating the question data

Re-run after any edit to `prep/**/questions.md`:

```
python tools/build_site.py            # writes site/questions.json + questions.js
python tools/build_site.py --check    # exits 1 if out-of-date (for CI)
```

The build is idempotent; CI can drop `--check` into the existing workflow.

## Deploying to GitHub Pages

The `.github/workflows/pages.yml` workflow builds `questions.{json,js}` on every push to `main` and publishes the `site/` directory to GitHub Pages. To enable:

1. Repository → **Settings** → **Pages** → Source: **GitHub Actions**.
2. Push to `main` once; the workflow does the rest.

The site is fully static — no secrets, no env vars.

## Design notes

- **Single source of truth.** All content comes from `prep/**/questions.md`. The site never duplicates content; it parses the canonical markdown.
- **Local-only progress.** Spaced repetition state and MCQ stats live in `localStorage`. Clearing browser storage resets all progress.
- **Markdown rendering.** Uses the [`marked`](https://github.com/markedjs/marked) library from a CDN (with a subresource-integrity hash). If the CDN is blocked or the user is offline, the site falls back to a minimal markdown-to-HTML conversion — content stays readable, just less prettily formatted.
- **MCQ correctness contract.** The MCQ option text is the entry's **first sentence of the short answer** versus three of its **Common mistakes** verbatim. Because the question bank's `Common mistakes` field is curated to describe *interview failure modes specifically attached to this question*, the distractors are interview-realistic — not synthetic noise.
- **Behavioral questions.** These have no canonical answer; the **Signal** block (what the interviewer is listening for) plays the role of model answer in Mock Interview mode.

## Keyboard shortcuts

- `/` — focus search (Browse mode)
- `space` / `enter` — reveal answer (Flashcards)
- `1`–`4` — rate Again / Hard / Good / Easy (Flashcards)
- `j` / `k` — skip to next card (Flashcards)
- `1`–`4` or `A`–`D` — pick option (MCQ Quiz)
- `enter` — next question after feedback (MCQ Quiz)

## Adding a new mode

1. Add a new `<section class="view hidden" id="view-foo">` to `index.html`.
2. Add a `<button class="mode-btn" data-mode="foo">` in the topbar.
3. In `app.js`, add `setupFoo()` and call it from `init()`. The mode-switcher already hides/shows views automatically.

## License

Same as the parent repo: CC-BY-4.0 for content, MIT for tooling.
