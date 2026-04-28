"""Merge the four E3 axis-comparison RAG runs into one file with a `mode`
column that the analyzer can pivot on."""

from __future__ import annotations

import json
from pathlib import Path

R = Path("results")

SOURCES = [
    ("single", R / "rag_single_n4.json"),
    ("data_segment", R / "rag_data_segment_n4.json"),
    ("persona", R / "rag_persona_n4.json"),
    ("model_family", R / "rag_model_family_n4.json"),
]


def main():
    merged = []
    for mode, path in SOURCES:
        if not path.exists():
            print(f"  skip {mode} (no {path})")
            continue
        with open(path) as f:
            runs = json.load(f)
        for run in runs:
            if run["cfg"].get("n") != 4:
                continue
            run = dict(run)
            run["cfg"] = dict(run["cfg"])
            run["cfg"]["mode"] = mode
            merged.append(run)
    out = R / "axis_compare.json"
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"wrote {len(merged)} runs to {out}")


if __name__ == "__main__":
    main()
