#!/usr/bin/env python3
"""Build laso_object_types.json from exported LASO metadata.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from grasp_dataset.laso_catalog import build

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=PROJECT / "data" / "laso_grasp_only_affordance")
    p.add_argument("--output", type=Path, default=PROJECT / "laso_object_types.json")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    catalog = build(args.data_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2) + "\n")
    print(
        f"Saved {sum(map(len, catalog.values()))} LASO objects in "
        f"{len(catalog)} categories to {args.output}"
    )


if __name__ == "__main__":
    main()
