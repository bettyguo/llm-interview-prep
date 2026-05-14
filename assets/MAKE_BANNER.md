# Banner spec

`assets/banner.svg` is the hero image referenced from the README. A first-pass SVG is checked into the repo so the README renders correctly on GitHub; this file is the spec a designer (or you) can use to render a higher-fidelity replacement.

## Dimensions and constraints

- Aspect ratio: ~3:1. Suggested size 1280×420.
- Renders well on both light and dark GitHub themes — solid background or transparent + theme-aware text.
- Total file size < 200 KB if PNG; SVG preferred.

## Content

- Title: **`llm-interview-prep`** in a monospace or geometric sans-serif (e.g., JetBrains Mono, Inter, Space Grotesk).
- Subtitle (smaller): "Everything you need to walk into an AI engineering interview prepared."
- Three-line decorative motif (optional): the topic spine as a row of pills or a horizontal stack — "Transformers · Training · Inference · RAG · Agents · System Design · Behavioral".
- Curator microcredit at bottom-right: "Curated by Betty Guo · HKU · 2026".

## Color guidance

- Primary: a serious, calm palette. Suggested: deep blue/teal background (#0F1B2D or similar) with off-white text (#F4F4F2) and a single accent (e.g., #E0B450 for the topic pills).
- Avoid: gradients that look like marketing brochures; emoji; AI-generated photoreal imagery.

## Social card

Also produce `social-card.svg` (or `.png`) at 1200×630 for OpenGraph / Twitter. Same palette; the title can be larger relative to the subtitle. Used by GitHub's social-preview slot under repo settings.

## What is in the repo today

The current `banner.svg` is a clean SVG placeholder. Anyone is welcome to PR a higher-fidelity replacement that fits the spec above.
