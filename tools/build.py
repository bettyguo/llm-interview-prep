"""Aggregate per-topic stats and regenerate the auto-managed sections of
top-level README.md and per-topic README.md files.

The top-level README's hero / study-plan / curator sections are hand-curated.
This script regenerates only the sections delimited by:

    <!-- BUILD:STATS:START --> ... <!-- BUILD:STATS:END -->
    <!-- BUILD:TOPICS:START --> ... <!-- BUILD:TOPICS:END -->

Idempotent. Run on every commit via CI to ensure stats stay current.

Usage:
    python tools/build.py
    python tools/build.py --check    # exit 1 if the README would change
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

QUESTION_HEADING_RE = re.compile(r"^### Q:\s*(.+?)\s*$", re.MULTILINE)
TOPIC_DIR_RE = re.compile(r"^\d{2}-")


def count_entries(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        return 0
    return len(QUESTION_HEADING_RE.findall(text))


def topic_dirs(root: Path) -> list[Path]:
    prep = root / "prep"
    if not prep.exists():
        return []
    return sorted([d for d in prep.iterdir() if d.is_dir() and TOPIC_DIR_RE.match(d.name)])


def topic_title(td: Path) -> str:
    readme = td / "README.md"
    if readme.exists():
        for line in readme.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
    return td.name


def topic_entry_count(td: Path) -> int:
    n = 0
    qf = td / "questions.md"
    if qf.exists():
        n += count_entries(qf)
    drills = td / "drills"
    if drills.exists():
        for f in drills.glob("*.md"):
            if f.name.lower() == "readme.md":
                continue
            n += count_entries(f)
    return n


def render_stats_block(topics: list[Path], counts: dict[str, int]) -> str:
    total = sum(counts.values())
    lines = ["", "**Total questions:** " + str(total), ""]
    lines.append("| # | Topic | Questions |")
    lines.append("|---|-------|-----------|")
    for i, td in enumerate(topics, 1):
        lines.append(f"| {i} | {topic_title(td)} | {counts[td.name]} |")
    lines.append("")
    return "\n".join(lines)


def render_topics_block(topics: list[Path]) -> str:
    lines = [""]
    for td in topics:
        lines.append(f"- [{topic_title(td)}](prep/{td.name}/)")
    lines.append("")
    return "\n".join(lines)


def replace_block(text: str, marker: str, payload: str) -> str:
    start = f"<!-- BUILD:{marker}:START -->"
    end = f"<!-- BUILD:{marker}:END -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    replacement = f"{start}\n{payload}\n{end}"
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text  # markers missing — caller may warn separately


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if README would change")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    topics = topic_dirs(root)
    counts = {td.name: topic_entry_count(td) for td in topics}

    readme = root / "README.md"
    if not readme.exists():
        print("build: README.md not found at repo root.", file=sys.stderr)
        return 2
    original = readme.read_text(encoding="utf-8")
    updated = original
    updated = replace_block(updated, "STATS", render_stats_block(topics, counts))
    updated = replace_block(updated, "TOPICS", render_topics_block(topics))

    if args.check:
        if updated != original:
            print("build --check: README.md is out of date; run `python tools/build.py`.")
            return 1
        print("build --check: README.md up to date.")
        return 0

    if updated != original:
        readme.write_text(updated, encoding="utf-8")
        print(f"build: README.md rewritten ({sum(counts.values())} total questions).")
    else:
        print("build: README.md unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
