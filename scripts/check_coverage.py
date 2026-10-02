"""Report all measured production code; retain the existing shared-core gate."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--core-min", type=float, default=70)
    args = parser.parse_args()
    files = json.loads(args.report.read_text(encoding="utf-8"))["files"]
    groups = {}
    for name, entry in files.items():
        directory = name.replace("\\", "/").split("/")[0]
        covered, statements = groups.get(directory, (0, 0))
        summary = entry["summary"]
        groups[directory] = (covered + summary["covered_lines"], statements + summary["num_statements"])
    if "iwac_common" not in groups:
        raise SystemExit("Coverage report did not measure iwac_common")
    for group, (covered, statements) in sorted(groups.items()):
        ratio = 100 * covered / statements if statements else 100
        print(f"{group}: {ratio:.1f}% ({covered}/{statements} lines)")
    covered, statements = groups["iwac_common"]
    if not statements or 100 * covered / statements < args.core_min:
        raise SystemExit(f"iwac_common coverage must be at least {args.core_min}%")


if __name__ == "__main__":
    main()
