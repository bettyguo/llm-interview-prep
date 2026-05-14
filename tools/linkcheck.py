"""Walk every markdown file in the repo, extract every external URL, and verify
that each URL resolves with an acceptable HTTP status code.

Acceptable status: 200, 301, 302, 303, 307, 308.
Acceptable on permissive mode: any 2xx/3xx.
Unacceptable: 4xx, 5xx, timeout, DNS failure, malformed URL.

URLs listed in `tools/linkcheck-ignore.txt` (one per line, # for comment) are skipped.

Exit code 0 if all reachable; 1 if any failure.

Usage:
    python tools/linkcheck.py
    python tools/linkcheck.py --permissive
    python tools/linkcheck.py --timeout 15
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import requests
except ImportError:
    print("linkcheck: please `pip install -r tools/requirements.txt` first.", file=sys.stderr)
    sys.exit(2)


URL_RE = re.compile(r"\[[^\]]+\]\((https?://[^\s)]+)\)")
ACCEPTABLE_STRICT = {200, 301, 302, 303, 307, 308}
HEADERS = {
    "User-Agent": "llm-interview-prep-linkcheck/1.0 (+https://github.com/bettyguo/llm-interview-prep)"
}


@dataclass
class Result:
    url: str
    status: int | None
    ok: bool
    note: str = ""
    sources: list[Path] | None = None


def load_ignores(root: Path) -> set[str]:
    path = root / "tools" / "linkcheck-ignore.txt"
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def discover_urls(root: Path) -> dict[str, list[Path]]:
    urls: dict[str, list[Path]] = {}
    md_files: list[Path] = []
    for sub in ("README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE"):
        f = root / sub
        if f.exists():
            md_files.append(f)
    for sub in ("prep", "study-plan", "docs", "PLANNING", ".github"):
        d = root / sub
        if d.exists():
            md_files.extend(d.rglob("*.md"))
    for f in md_files:
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for m in URL_RE.finditer(text):
            url = m.group(1).rstrip(".,;:")
            urls.setdefault(url, []).append(f)
    return urls


def check_one(url: str, timeout: float, permissive: bool) -> Result:
    try:
        # HEAD first; fall back to GET because many sites refuse HEAD.
        try:
            r = requests.head(url, allow_redirects=True, timeout=timeout, headers=HEADERS)
            if r.status_code >= 400:
                r = requests.get(url, allow_redirects=True, timeout=timeout, headers=HEADERS, stream=True)
                r.close()
        except requests.exceptions.RequestException:
            r = requests.get(url, allow_redirects=True, timeout=timeout, headers=HEADERS, stream=True)
            r.close()
        ok = r.status_code in ACCEPTABLE_STRICT or (permissive and 200 <= r.status_code < 400)
        return Result(url=url, status=r.status_code, ok=ok)
    except requests.exceptions.Timeout:
        return Result(url=url, status=None, ok=False, note="timeout")
    except requests.exceptions.ConnectionError as e:
        return Result(url=url, status=None, ok=False, note=f"conn-error: {e.__class__.__name__}")
    except requests.exceptions.RequestException as e:
        return Result(url=url, status=None, ok=False, note=f"req-error: {e.__class__.__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permissive", action="store_true", help="accept any 2xx/3xx")
    parser.add_argument("--timeout", type=float, default=10.0, help="per-URL timeout seconds")
    parser.add_argument("--workers", type=int, default=8, help="parallel workers")
    parser.add_argument("--report", default="tools/linkcheck-report.md", help="report output path")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    ignores = load_ignores(root)
    urls = discover_urls(root)
    to_check = [u for u in urls if u not in ignores]
    print(f"linkcheck: {len(urls)} unique URLs ({len(ignores)} ignored); checking {len(to_check)}.")

    results: list[Result] = []
    if to_check:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(check_one, u, args.timeout, args.permissive): u for u in to_check}
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                r.sources = urls.get(r.url, [])
                results.append(r)

    failures = [r for r in results if not r.ok]

    # Report
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Linkcheck report\n\n")
        f.write(f"- total unique URLs: {len(urls)}\n")
        f.write(f"- ignored: {len(ignores)}\n")
        f.write(f"- checked: {len(to_check)}\n")
        f.write(f"- failures: {len(failures)}\n\n")
        if failures:
            f.write("## Failures\n\n")
            for r in failures:
                f.write(f"- `{r.url}` — status={r.status} note={r.note}\n")
                for s in r.sources or []:
                    f.write(f"  - referenced in {s.relative_to(root)}\n")

    if failures:
        print(f"linkcheck: {len(failures)} failures (see {args.report}).")
        for r in failures[:25]:
            print(f"  FAIL {r.url} status={r.status} {r.note}")
        if len(failures) > 25:
            print(f"  ... and {len(failures) - 25} more.")
        return 1
    print("linkcheck: all URLs OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
