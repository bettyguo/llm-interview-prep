# 03 — Verification log

The answer-correctness protocol (Phase 0) requires every non-trivial answer to carry an authoritative reference, verified at write time. This file records that verification: one row per answer.

## Schema

```
| topic | question-slug | reference(s) | verified-by | verified-on | spot-check |
```

- `verified-by`: the curator initials of who wrote / verified the entry.
- `verified-on`: ISO date.
- `spot-check`: blank at write time. Phase 5 fills this column with `pass`/`fail` for the 10% sample.

## Entries

| topic | question-slug | reference(s) | verified-by | verified-on | spot-check |
|-------|---------------|--------------|-------------|-------------|------------|
