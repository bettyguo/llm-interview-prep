"""Parse every prep/*/questions.md into a single site/questions.json.

Schema produced per entry:

    {
      "id":         "<topic_slug>-<index>",
      "topic":      "01-ml-dl-fundamentals",
      "topic_num":  1,
      "topic_title":"ML & DL Fundamentals",
      "question":   "<the Q line>",
      "category":   "concept|derivation|system-design|coding|behavioral",
      "difficulty": "intro|mid|senior|staff",
      "tags":       ["a", "b"],
      "short":      "<short answer text, markdown>",
      "short_first":"<first sentence of short answer>",
      "expansion":  "<markdown>",
      "follow_ups": ["...", "..."],
      "mistakes":   ["...", "..."],
      "tail_label": "References" | "Signal" | "Implementation" | null,
      "tail":       "<markdown of the trailing block>"
    }

Then writes site/questions.json. Idempotent.

Usage:
    python tools/build_site.py
    python tools/build_site.py --check    # exit 1 if questions.json out of date
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREP = ROOT / "prep"
OUT = ROOT / "site" / "questions.json"
OUT_JS = ROOT / "site" / "questions.js"

TOPIC_DIR_RE = re.compile(r"^(\d{2})-")
ENTRY_SPLIT_RE = re.compile(r"^---\s*$", re.MULTILINE)
Q_HEAD_RE = re.compile(r"^###\s+Q:\s*(.+?)\s*$", re.MULTILINE)

# Recognise a bolded field-label line like "**Category:** concept",
# "**Short answer.** Bias is error...", "**Common follow-ups.**".
# The label can contain spaces, slashes, hyphens, colons, periods.
FIELD_RE = re.compile(r"^\*\*(?P<key>[A-Za-z][A-Za-z /:.\-]*?)[:.]?\*\*\s*(?P<rest>.*?)\s*$")
INLINE_TAGS_RE = re.compile(r"\[(.+?)\]")


def topic_title(td: Path) -> str:
    readme = td / "README.md"
    if readme.exists():
        for line in readme.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("# "):
                return s[2:].strip()
    return td.name


def parse_tags(rest: str) -> list[str]:
    m = INLINE_TAGS_RE.search(rest)
    if not m:
        return []
    return [t.strip() for t in m.group(1).split(",") if t.strip()]


def parse_entry(raw: str) -> dict | None:
    """Parse one '### Q: ...' block into a dict, or return None if not a Q entry."""
    m = Q_HEAD_RE.search(raw)
    if not m:
        return None
    question = m.group(1).strip()

    # Body = everything after the Q heading line
    body = raw[m.end():].lstrip("\n")
    lines = body.splitlines()

    entry: dict = {
        "question": question,
        "category": None,
        "difficulty": None,
        "tags": [],
        "short": "",
        "expansion": "",
        "follow_ups": [],
        "mistakes": [],
        "tail_label": None,
        "tail": "",
    }

    # Walk line by line, splitting into named sections by bolded labels.
    # Sections in order: header fields (Category/Difficulty/Tags), Short answer,
    # Expansion / why this is the answer, Common follow-ups, Common mistakes,
    # then one of: References | Signal | Implementation.
    section = "header"
    buf: list[str] = []

    label_map = {
        "category":   "category",
        "difficulty": "difficulty",
        "tags":       "tags",
        "short answer": "short",
        "expansion / why this is the answer": "expansion",
        "expansion": "expansion",
        "common follow-ups": "follow_ups",
        "common mistakes": "mistakes",
        "references": "tail_refs",
        "signal":     "tail_signal",
        "implementation": "tail_impl",
    }

    def flush(current: str, lines: list[str]) -> None:
        text = "\n".join(lines).strip()
        if not text:
            return
        if current in ("short", "expansion"):
            existing = entry[current]
            entry[current] = (existing + "\n\n" + text).strip() if existing else text
        elif current in ("follow_ups", "mistakes"):
            entry[current] = extract_bullets(text)
        elif current == "tail_refs":
            entry["tail_label"] = "References"
            entry["tail"] = text
        elif current == "tail_signal":
            entry["tail_label"] = "Signal"
            entry["tail"] = text
        elif current == "tail_impl":
            entry["tail_label"] = "Implementation"
            entry["tail"] = text

    for line in lines:
        stripped = line.rstrip()
        fm = FIELD_RE.match(stripped)
        # FIELD_RE matches both "**Category:** concept" (rest on same line)
        # and standalone "**Common mistakes.**" (rest empty).
        if fm:
            key = fm.group("key").strip().lower()
            rest = fm.group("rest").strip()
            mapped = label_map.get(key)
            if mapped is None:
                # An unrelated bold tag in body (rare). Treat as content of current section.
                buf.append(line)
                continue
            # flush previous section
            flush(section, buf)
            buf = []
            if mapped == "category":
                entry["category"] = rest.lower() or None
                section = "header"
            elif mapped == "difficulty":
                entry["difficulty"] = rest.lower() or None
                section = "header"
            elif mapped == "tags":
                entry["tags"] = parse_tags(rest)
                section = "header"
            else:
                section = mapped
                if rest:
                    buf.append(rest)
            continue
        if section == "header":
            # Skip blank lines between header fields
            continue
        buf.append(line)

    flush(section, buf)

    # Derive short_first (first sentence) for MCQ option text.
    short = entry["short"].strip()
    entry["short_first"] = first_sentence(short)
    return entry


SENT_END_RE = re.compile(r"(?<=[\.\?\!])\s+")


def first_sentence(text: str) -> str:
    if not text:
        return ""
    # Strip leading inline bold like "Almost always — ..."
    parts = SENT_END_RE.split(text, maxsplit=1)
    return parts[0].strip()


BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")


def extract_bullets(text: str) -> list[str]:
    """Extract top-level bullets. Continuation lines fold in until the next bullet/blank."""
    out: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        m = BULLET_RE.match(line)
        if m:
            if current:
                out.append(" ".join(current).strip())
                current = []
            current.append(m.group(1).strip())
        elif line.strip() == "":
            if current:
                out.append(" ".join(current).strip())
                current = []
        else:
            if current:
                current.append(line.strip())
    if current:
        out.append(" ".join(current).strip())
    return [b for b in out if b]


def parse_questions_file(md: Path, topic_dir: Path) -> list[dict]:
    """Return entries from one markdown file, without IDs assigned yet (those are
    allocated by collect_all so they stay unique across files in the same topic)."""
    text = md.read_text(encoding="utf-8")
    chunks = ENTRY_SPLIT_RE.split(text)
    topic_slug = topic_dir.name
    tnum_match = TOPIC_DIR_RE.match(topic_slug)
    topic_num = int(tnum_match.group(1)) if tnum_match else 0
    ttitle = topic_title(topic_dir)

    out: list[dict] = []
    for chunk in chunks:
        if "### Q:" not in chunk:
            continue
        e = parse_entry(chunk)
        if e is None:
            continue
        e["topic"] = topic_slug
        e["topic_num"] = topic_num
        e["topic_title"] = ttitle
        e["_source"] = md.name
        out.append(e)
    return out


def collect_all() -> list[dict]:
    if not PREP.exists():
        return []
    topics = sorted([d for d in PREP.iterdir() if d.is_dir() and TOPIC_DIR_RE.match(d.name)])
    all_entries: list[dict] = []
    for td in topics:
        topic_entries: list[dict] = []
        qf = td / "questions.md"
        if qf.exists():
            topic_entries.extend(parse_questions_file(qf, td))
        drills = td / "drills"
        if drills.exists():
            for f in sorted(drills.glob("*.md")):
                if f.name.lower() == "readme.md":
                    continue
                topic_entries.extend(parse_questions_file(f, td))
        # Assign IDs that are unique per topic — index across all files in the topic.
        for idx, e in enumerate(topic_entries, 1):
            e["id"] = f"{td.name}-{idx:03d}"
            e.pop("_source", None)
            all_entries.append(e)
    return all_entries


def build_payload(entries: list[dict]) -> dict:
    # Topic summary for the UI sidebar.
    topics: dict[str, dict] = {}
    for e in entries:
        slug = e["topic"]
        if slug not in topics:
            topics[slug] = {
                "slug": slug,
                "num": e["topic_num"],
                "title": e["topic_title"],
                "count": 0,
            }
        topics[slug]["count"] += 1
    topic_list = sorted(topics.values(), key=lambda t: t["num"])
    return {
        "generated_by": "tools/build_site.py",
        "version": 1,
        "total": len(entries),
        "topics": topic_list,
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if questions.json would change")
    args = parser.parse_args(argv)

    entries = collect_all()
    payload = build_payload(entries)
    new_json = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    # Compact JSON inside a JS var so the site can open straight from file:// without a server.
    new_js = (
        "// Auto-generated by tools/build_site.py. Do not edit by hand.\n"
        "window.QUESTIONS_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing_json = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    existing_js = OUT_JS.read_text(encoding="utf-8") if OUT_JS.exists() else ""

    if args.check:
        if new_json != existing_json or new_js != existing_js:
            print("build_site --check: site/questions.{json,js} out of date; run `python tools/build_site.py`.")
            return 1
        print("build_site --check: site/questions.{json,js} up to date.")
        return 0

    OUT.write_text(new_json, encoding="utf-8")
    OUT_JS.write_text(new_js, encoding="utf-8")
    print(f"build_site: wrote {OUT.relative_to(ROOT)} and {OUT_JS.relative_to(ROOT)} ({payload['total']} entries, {len(payload['topics'])} topics).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
