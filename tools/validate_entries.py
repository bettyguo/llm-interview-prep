"""Validate Q&A entries across prep/ topic directories.

Enforces the Phase 1 schema: every question must have all required structural
fields. The answer-correctness protocol (Phase 0) demands that every
concept / derivation / system-design entry carry at least one reference; the
validator refuses to pass without one. Coding entries must instead carry an
Implementation block; behavioral entries must carry a Signal block.

Usage:
    python tools/validate_entries.py              # validate everything
    python tools/validate_entries.py prep/04-*    # validate a subtree
    python tools/validate_entries.py --stats      # also print per-topic counts

Exit code 0 = pass, 1 = fail.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------

REQUIRED_HEADERS = [
    "**Category:**",
    "**Difficulty:**",
    "**Tags:**",
    "**Short answer.**",
    "**Expansion",
    "**Common follow-ups.**",
    "**Common mistakes.**",
]

VALID_CATEGORIES = {"concept", "derivation", "system-design", "coding", "behavioral"}
VALID_DIFFICULTIES = {"intro", "mid", "senior", "staff"}

REFERENCE_REQUIRED_CATEGORIES = {"concept", "derivation", "system-design"}
IMPLEMENTATION_REQUIRED_CATEGORIES = {"coding"}
SIGNAL_REQUIRED_CATEGORIES = {"behavioral"}

QUESTION_RE = re.compile(r"^### Q:\s*(.+?)\s*$", re.MULTILINE)
CATEGORY_RE = re.compile(r"^\*\*Category:\*\*\s*([a-z\-]+)\s*$", re.MULTILINE)
DIFFICULTY_RE = re.compile(r"^\*\*Difficulty:\*\*\s*([a-z]+)\s*$", re.MULTILINE)
URL_RE = re.compile(r"https?://\S+")


@dataclass
class EntryReport:
    path: Path
    question: str
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class FileReport:
    path: Path
    entries: list[EntryReport] = field(default_factory=list)
    file_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.file_errors and all(e.ok for e in self.entries)


# -----------------------------------------------------------------------------
# Parsing & validation
# -----------------------------------------------------------------------------

def split_entries(text: str) -> list[tuple[str, str]]:
    """Split a markdown file into (question_heading, entry_body) pairs."""
    parts = re.split(r"(?m)^### Q:\s*", text)
    if len(parts) <= 1:
        return []
    out = []
    for chunk in parts[1:]:
        # chunk starts at the question text
        lines = chunk.splitlines()
        if not lines:
            continue
        question = lines[0].strip()
        body = "\n".join(lines[1:])
        out.append((question, body))
    return out


def validate_entry(path: Path, question: str, body: str) -> EntryReport:
    rep = EntryReport(path=path, question=question)

    if not question:
        rep.errors.append("empty question text after `### Q:`")

    # Required structural headers
    for header in REQUIRED_HEADERS:
        if header not in body:
            rep.errors.append(f"missing required field: {header}")

    # Category
    cat_match = CATEGORY_RE.search(body)
    if not cat_match:
        rep.errors.append("missing or malformed `**Category:**` line")
        return rep
    category = cat_match.group(1).strip()
    if category not in VALID_CATEGORIES:
        rep.errors.append(
            f"invalid category `{category}`; must be one of {sorted(VALID_CATEGORIES)}"
        )

    # Difficulty
    diff_match = DIFFICULTY_RE.search(body)
    if not diff_match:
        rep.errors.append("missing or malformed `**Difficulty:**` line")
    else:
        difficulty = diff_match.group(1).strip()
        if difficulty not in VALID_DIFFICULTIES:
            rep.errors.append(
                f"invalid difficulty `{difficulty}`; must be one of {sorted(VALID_DIFFICULTIES)}"
            )

    # Per-category terminal block
    if category in REFERENCE_REQUIRED_CATEGORIES:
        if "**References.**" not in body:
            rep.errors.append(
                f"category `{category}` requires a `**References.**` block (answer-correctness protocol)"
            )
        else:
            refs_section = body.split("**References.**", 1)[1]
            urls = URL_RE.findall(refs_section)
            if not urls:
                rep.errors.append(
                    "`**References.**` block contains no http(s) URL — at least one authoritative source required"
                )
    elif category in IMPLEMENTATION_REQUIRED_CATEGORIES:
        if "**Implementation.**" not in body:
            rep.errors.append("category `coding` requires an `**Implementation.**` block")
        else:
            impl_section = body.split("**Implementation.**", 1)[1]
            if "```" not in impl_section:
                rep.errors.append(
                    "`**Implementation.**` block must contain a fenced code block"
                )
    elif category in SIGNAL_REQUIRED_CATEGORIES:
        if "**Signal.**" not in body:
            rep.errors.append("category `behavioral` requires a `**Signal.**` block")

    return rep


def validate_file(path: Path) -> FileReport:
    rep = FileReport(path=path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        rep.file_errors.append(f"could not decode as UTF-8: {e}")
        return rep

    entries = split_entries(text)
    # Duplicate-question check
    seen: dict[str, int] = {}
    for q, _ in entries:
        seen[q] = seen.get(q, 0) + 1
    for q, c in seen.items():
        if c > 1:
            rep.file_errors.append(f"duplicate question heading ({c}x): {q!r}")

    for q, body in entries:
        rep.entries.append(validate_entry(path, q, body))

    return rep


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------

QUESTIONS_FILE_PATTERNS = ["questions.md"]
SYSTEM_DESIGN_DRILL_DIR = "prep/08-ml-system-design/drills"


def discover_files(root: Path, args_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    search_roots: list[Path] = []
    if args_paths:
        for p in args_paths:
            search_roots.append(Path(p))
    else:
        search_roots.append(root / "prep")

    for sr in search_roots:
        if not sr.exists():
            continue
        for p in sr.rglob("questions.md"):
            files.append(p)
        # System-design drills: each drill file is its own entry container
        drill_dir = sr if sr.name == "drills" else (sr / "drills") if (sr / "drills").exists() else None
        if drill_dir is None:
            sd = root / SYSTEM_DESIGN_DRILL_DIR
            if sd.exists() and sd in sr.parents or sd == sr or root in sr.parents:
                drill_dir = sd
        if drill_dir and drill_dir.exists():
            for p in drill_dir.glob("*.md"):
                if p.name.lower() != "readme.md":
                    files.append(p)
    # Dedup
    seen: set[Path] = set()
    out: list[Path] = []
    for f in files:
        rp = f.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(f)
    return sorted(out)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="optional subset of paths to validate")
    parser.add_argument("--stats", action="store_true", help="print per-topic counts")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    files = discover_files(root, args.paths)

    if not files:
        print("validate_entries: no questions.md files found — nothing to validate.")
        return 0

    total_entries = 0
    total_errors = 0
    per_topic: dict[str, int] = {}
    fail_files: list[FileReport] = []

    for f in files:
        rep = validate_file(f)
        topic = f.relative_to(root).parts[1] if len(f.relative_to(root).parts) > 1 else "(root)"
        per_topic[topic] = per_topic.get(topic, 0) + len(rep.entries)
        total_entries += len(rep.entries)
        if not rep.ok:
            fail_files.append(rep)
            for fe in rep.file_errors:
                total_errors += 1
                print(f"FAIL {f}: {fe}")
            for er in rep.entries:
                if not er.ok:
                    for msg in er.errors:
                        total_errors += 1
                        print(f"FAIL {f} :: Q={er.question!r}: {msg}")

    if args.stats:
        print("\n--- per-topic entry counts ---")
        for k in sorted(per_topic):
            print(f"  {k}: {per_topic[k]}")
        print(f"  TOTAL: {total_entries}")

    print(f"\nvalidate_entries: {total_entries} entries across {len(files)} files; {total_errors} errors.")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
