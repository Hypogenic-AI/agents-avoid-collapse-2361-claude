"""End-to-end finalisation: rebuild merged axis JSON, run analysis,
generate tables, splice the auto-generated sections into REPORT.md.

Idempotent — safe to re-run as more data lands.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(res.returncode)
    print(res.stdout.splitlines()[-1] if res.stdout else "(no stdout)")


def splice_report():
    report = Path("REPORT.md")
    sections_path = Path("results/_sections.md")
    if not sections_path.exists():
        print("no _sections.md — run build_report first")
        return
    sections = sections_path.read_text()
    text = report.read_text()
    placeholder = "[REPLACE_WITH_AUTO_SECTIONS]"
    if placeholder in text:
        text = text.replace(placeholder, sections)
    else:
        # Re-replace the auto-generated block (between sentinels)
        start = "<!-- AUTO_BEGIN -->"
        end = "<!-- AUTO_END -->"
        if start in text and end in text:
            text = re.sub(
                f"{re.escape(start)}.*?{re.escape(end)}",
                f"{start}\n{sections}\n{end}",
                text,
                flags=re.DOTALL,
            )
        else:
            print(
                "REPORT.md has neither placeholder nor sentinels — appending"
                " sections at the end."
            )
            text += "\n\n<!-- AUTO_BEGIN -->\n" + sections + "\n<!-- AUTO_END -->\n"
            report.write_text(text)
            return
    # Wrap in sentinels for next time
    if "<!-- AUTO_BEGIN -->" not in text:
        text = text.replace(
            sections,
            "<!-- AUTO_BEGIN -->\n" + sections + "\n<!-- AUTO_END -->",
            1,
        )
    report.write_text(text)
    print(f"spliced {len(sections)} bytes of auto-generated sections into REPORT.md")


def main():
    run(["python", "src/merge_axis.py"])
    run(["python", "src/analyze.py"])
    run(["python", "src/make_tables.py"])
    run(["python", "src/build_report.py"])
    run(["python", "src/stats.py"])
    splice_report()


if __name__ == "__main__":
    main()
