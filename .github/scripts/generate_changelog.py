#!/usr/bin/env python3
"""Generate/update CHANGELOG.md from git log, grouped by author.

No commit-message convention required (no Conventional Commits needed).
Author identity is normalized via .mailmap, which `git log` honors natively.

Two modes:
- Default (branch push): regenerates the "[Unreleased]" section from commits
  since the last tag and replaces it in place, leaving the rest of the file
  untouched. Safe to re-run on every push -- idempotent.
- `--freeze <version>` (tag push): renames the current "[Unreleased]" section
  to "[<version>] - <date>", freezing it as permanent history, and inserts a
  fresh empty "[Unreleased]" section above it. Must run *before* the branch-push
  mode would otherwise regenerate "[Unreleased]" past this tag, or the frozen
  content would never get a chance to be captured.
"""
import argparse
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date, timezone, datetime
from pathlib import Path

REPO_URL = "https://github.com/exoAtmospheres/ForMoSA"
CHANGELOG_PATH = Path("CHANGELOG.md")
HEADER = (
    "# Changelog\n\n"
    "Auto-generated from commit history, grouped by author. "
    "Not hand-maintained -- see `git log` for full detail.\n\n"
)

# Commit subjects to drop entirely -- noise, not user-facing changes.
NOISE_PATTERNS = [
    re.compile(r"^\(auto\) Paper PDF Draft$"),
]


def run(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def latest_tag():
    try:
        return run("git", "describe", "--tags", "--abbrev=0").strip()
    except subprocess.CalledProcessError:
        return None


def collect_commits(rev_range):
    args = ["git", "log", "--no-merges", "--format=%H%x1f%aN%x1f%s"]
    if rev_range:
        args.append(rev_range)
    out = run(*args)
    commits = []
    for line in out.strip("\n").split("\n"):
        if not line:
            continue
        sha, author, subject = line.split("\x1f")
        if any(p.match(subject) for p in NOISE_PATTERNS):
            continue
        commits.append((sha, author, subject))
    return commits


def render_section(title, commits):
    lines = [f"## {title}", ""]
    if not commits:
        return "\n".join(lines + ["_No changes._", ""])

    by_author = defaultdict(list)
    for sha, author, subject in commits:
        by_author[author].append((sha, subject))

    for author in sorted(by_author, key=lambda a: -len(by_author[a])):
        lines.append(f"### {author}")
        for sha, subject in by_author[author]:
            lines.append(f"- {subject} ([{sha[:7]}]({REPO_URL}/commit/{sha}))")
        lines.append("")
    return "\n".join(lines)


def update_changelog(new_section):
    existing = CHANGELOG_PATH.read_text() if CHANGELOG_PATH.exists() else HEADER

    match = re.search(r"^## ", existing, flags=re.MULTILINE)
    if match:
        header, rest = existing[: match.start()], existing[match.start() :]
    else:
        header, rest = existing.rstrip("\n") + "\n\n", ""

    # Drop a pre-existing "[Unreleased]" block (up to the next "## " heading),
    # so re-running this script doesn't pile up duplicate sections.
    rest = re.sub(
        r"^## \[Unreleased\].*?(?=^## |\Z)", "", rest, count=1, flags=re.MULTILINE | re.DOTALL
    )

    CHANGELOG_PATH.write_text(header + new_section + "\n" + rest.lstrip("\n"))


def freeze_unreleased(version):
    if not CHANGELOG_PATH.exists():
        print("No CHANGELOG.md to freeze; nothing to do.", file=sys.stderr)
        return

    existing = CHANGELOG_PATH.read_text()
    today = datetime.now(timezone.utc).date().isoformat()

    frozen, count = re.subn(
        r"^## \[Unreleased\]",
        f"## [{version}] - {today}",
        existing,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        print("No '[Unreleased]' section found; nothing to freeze.", file=sys.stderr)
        return

    # Insert a fresh empty Unreleased section right above the just-frozen one.
    fresh_unreleased = render_section("[Unreleased]", [])
    frozen = re.sub(
        rf"^## \[{re.escape(version)}\]",
        fresh_unreleased + f"## [{version}]",
        frozen,
        count=1,
        flags=re.MULTILINE,
    )
    CHANGELOG_PATH.write_text(frozen)
    print(f"Froze '[Unreleased]' into '[{version}] - {today}'", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", metavar="VERSION", help="Freeze [Unreleased] into a dated version section")
    args = parser.parse_args()

    if args.freeze:
        freeze_unreleased(args.freeze)
    else:
        tag = latest_tag()
        rev_range = f"{tag}..HEAD" if tag else None
        commits = collect_commits(rev_range)
        section = render_section("[Unreleased]", commits)
        update_changelog(section)
        print(f"Updated CHANGELOG.md with {len(commits)} commits since {tag or 'repo start'}", file=sys.stderr)
