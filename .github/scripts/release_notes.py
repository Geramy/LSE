#!/usr/bin/env python3
"""Build the release document a downloader reads.

Run by the Build & Release workflow, and standalone for a dry run:

  python3 .github/scripts/release_notes.py --tag v0.1.0 --out /tmp/notes.md
"""
import argparse
import os
import re
import subprocess
import sys

# Commit subjects are sentences, not conventional-commit prefixes, so grouping
# reads the subject rather than a tag nobody writes.
SECTIONS = [
    ("Fixes", re.compile(r"\b(fix|fixes|fixed|correct|corrects|repair|no longer|"
                         r"instead of a recycled|stop|stops)\b", re.I)),
    ("Performance", re.compile(r"\b(faster|speed|throughput|occupancy|tile|fuse|"
                               r"fusion|traffic|cost|price|prices|spill)\b", re.I)),
    ("Engine", re.compile(r"\b(engine|kernel|kernels|scheduler|graph|emit|emits|"
                          r"dialect|backend|quant|attention|decode|prefill)\b", re.I)),
]


def run(*args, default=""):
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return default


def previous_tag(tag):
    """The tag before `tag`, or the first commit when there is none."""
    prev = run("git", "describe", "--tags", "--abbrev=0", f"{tag}^",
               default="") if tag_exists(tag) else ""
    if not prev:
        tags = run("git", "tag", "--sort=-creatordate").splitlines()
        prev = next((t for t in tags if t != tag), "")
    return prev or run("git", "rev-list", "--max-parents=0", "HEAD").split("\n")[0]


def tag_exists(tag):
    return bool(run("git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"))


def changes(since):
    """Subjects since `since`, grouped. A subject matches at most one section."""
    if not since:
        return {}, 0
    log = run("git", "log", "--no-merges", "--pretty=%s", f"{since}..HEAD")
    subjects = [s for s in log.splitlines() if s.strip()]
    grouped, seen = {}, set()
    for name, pattern in SECTIONS:
        hits = [s for s in subjects if s not in seen and pattern.search(s)]
        seen.update(hits)
        if hits:
            grouped[name] = hits
    rest = [s for s in subjects if s not in seen]
    if rest:
        grouped["Other"] = rest
    return grouped, len(subjects)


def human_size(path):
    try:
        n = os.path.getsize(path)
    except OSError:
        return ""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--version", default="")
    p.add_argument("--families", default="")
    p.add_argument("--targets", default="")
    p.add_argument("--gcc", default="")
    p.add_argument("--cmake", default="")
    p.add_argument("--rocm", default="")
    p.add_argument("--tests", default="")
    p.add_argument("--artifact", default="")
    p.add_argument("--out", default="-")
    a = p.parse_args()

    sha = run("git", "rev-parse", "HEAD")
    short = run("git", "rev-parse", "--short", "HEAD")
    prev = previous_tag(a.tag)
    grouped, total = changes(prev)

    L = [f"# LSE {a.tag}", ""]
    L.append(f"Built from `{short}` on the "
             f"{'GPU families below' if a.families else 'targets below'}.")
    L.append("")

    L.append("## What this build runs on")
    L.append("")
    if a.families:
        L.append(f"- **Architectures:** {a.families}")
    if a.targets:
        L.append(f"- **GPU targets:** `{a.targets.replace(';', '`, `')}`")
    L.append("- **Platform:** Linux x86_64")
    L.append("")
    L.append("A GPU target that is not listed here has no ahead-of-time kernels "
             "in this build. The engine compiles its kernels at run time, so an "
             "unlisted device still runs, paying the compile once per kernel and "
             "caching it on disk.")
    L.append("")

    L.append("## Requirements")
    L.append("")
    L.append("The released binary links the ROCm and HRX runtimes and loads them "
             "at start-up:")
    L.append("")
    L.append("```bash")
    L.append("export LD_LIBRARY_PATH=/opt/rocm/lib:/path/to/hrx-install/lib:"
             "$LD_LIBRARY_PATH")
    L.append("```")
    L.append("")
    L.append("Without that path the HRX backend does not initialize and the "
             "engine falls back to the CPU backend, which runs the same models "
             "far slower. Building from source additionally needs g++ 16 for "
             "C++26 reflection; see the README.")
    L.append("")

    if grouped:
        L.append(f"## Changes since {prev}" if prev.startswith("v")
                 else "## Changes")
        L.append("")
        for name, items in grouped.items():
            L.append(f"### {name}")
            L.append("")
            L.extend(f"- {s}" for s in items)
            L.append("")

    L.append("## Verification")
    L.append("")
    if a.tests:
        L.append(f"- **Test suite:** {a.tests}")
    built = [x for x in ((f"g++ {a.gcc}" if a.gcc else ""),
                         (f"CMake {a.cmake}" if a.cmake else ""),
                         (f"ROCm {a.rocm}" if a.rocm else "")) if x]
    if built:
        L.append(f"- **Built with:** {', '.join(built)}")
    L.append(f"- **Commit:** `{sha}`")
    if a.artifact:
        size = human_size(a.artifact)
        L.append(f"- **Artifact:** `{os.path.basename(a.artifact)}`"
                 + (f" ({size})" if size else ""))
        L.append("- Verify the download with the accompanying `.sha256` file: "
                 "`sha256sum -c <file>.sha256`")
    L.append("")

    text = "\n".join(L)
    if a.out == "-":
        sys.stdout.write(text)
    else:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
