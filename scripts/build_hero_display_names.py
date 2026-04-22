"""Generate configs/hero_display_names.json from Dota 2 localization files.

Usage:
    python scripts/build_hero_display_names.py \
        --game-path /path/to/dota2/game/dota \
        --out configs/hero_display_names.json \
        --locales en schinese

The localization files are KeyValues (.txt) files at:
    <game-path>/resource/localization/dota_<locale>.txt

Each hero entry follows the pattern:
    "npc_dota_hero_antimage"    "Anti-Mage"

This script parses all requested locales and emits a merged dict mapping
every display name to the GSI internal hero name (the npc_dota_hero_ prefix
stripped). Run this after each Dota 2 patch to keep the table current.
"""

import argparse
import json
import re
from pathlib import Path


_KEY_RE = re.compile(r'^"(npc_dota_hero_[^"]+)"\s+"([^"]+)"', re.MULTILINE)


def parse_localization(path: Path) -> dict[str, str]:
    """Parse a Valve KeyValues localization file, return {display_name: internal_name}."""
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {}
    for m in _KEY_RE.finditer(text):
        key, display = m.group(1), m.group(2)
        internal = key.removeprefix("npc_dota_hero_")
        result[display] = internal
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hero_display_names.json from game files")
    parser.add_argument("--game-path", required=True,
                        help="Path to the dota2/game/dota directory")
    parser.add_argument("--out", default="configs/hero_display_names.json",
                        help="Output JSON path (default: configs/hero_display_names.json)")
    parser.add_argument("--locales", nargs="+", default=["en", "schinese"],
                        help="Locales to include (default: en schinese)")
    args = parser.parse_args()

    game_path = Path(args.game_path)
    merged: dict[str, str] = {}

    for locale in args.locales:
        loc_file = game_path / "resource" / "localization" / f"dota_{locale}.txt"
        if not loc_file.exists():
            print(f"[warn] {loc_file} not found, skipping")
            continue
        entries = parse_localization(loc_file)
        print(f"[{locale}] {len(entries)} hero entries")
        merged.update(entries)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(merged)} entries to {out_path}")


if __name__ == "__main__":
    main()
