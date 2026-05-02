"""Refresh data/dota_knowledge/ from vendor/dotaconstants.

Reads dotaconstants build files (heroes, items, abilities, hero_abilities,
aghs_desc, patch, patchnotes) and renders flat per-entity JSON files for
the upcoming /tips endpoint's prefix builder.

Run after `git -C vendor/dotaconstants pull` on each Dota patch day:

    python scripts/refresh_dota_knowledge.py

Output directory is controlled by TIPS_KNOWLEDGE_DIR (default:
data/dota_knowledge under the repo root).
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DOTACONSTANTS_BUILD = REPO_ROOT / "vendor" / "dotaconstants" / "build"
DEFAULT_KNOWLEDGE_DIR = REPO_ROOT / "data" / "dota_knowledge"
KNOWLEDGE_ROOT = Path(os.getenv("TIPS_KNOWLEDGE_DIR", str(DEFAULT_KNOWLEDGE_DIR)))


def strip_hero_prefix(name: str) -> str:
    return name[len("npc_dota_hero_"):] if name.startswith("npc_dota_hero_") else name


# Talent dnames sometimes contain `{s:bonus_AbilityCastRange}` style template
# placeholders that pull values from related ability attribs. dotaconstants
# does not provide the substitution context, so for v1 we replace the
# placeholder with `?` — readable to the LLM as "talent that boosts X by an
# unknown amount", which is still useful guidance.
_TEMPLATE_RE = re.compile(r"\{[a-z]?:[^}]+\}")


def normalize_template(text: str) -> str:
    if not text:
        return text or ""
    return _TEMPLATE_RE.sub("?", text)


def parse_talents(talent_list: list, abilities_json: dict) -> list:
    """dotaconstants gives 8 talents as flat entries with level 1-4
    (mapping to in-game levels 10/15/20/25). Pair them into left/right.
    """
    by_level = {1: [], 2: [], 3: [], 4: []}
    for t in talent_list or []:
        level = t.get("level")
        name = t.get("name", "")
        ability = abilities_json.get(name, {})
        text = normalize_template(ability.get("dname", name))
        if level in by_level:
            by_level[level].append(text)

    in_game_level = {1: 10, 2: 15, 3: 20, 4: 25}
    out = []
    for level in (4, 3, 2, 1):  # render highest first
        items = by_level[level]
        out.append({
            "level": in_game_level[level],
            "left":  items[0] if len(items) >= 1 else "",
            "right": items[1] if len(items) >= 2 else "",
        })
    return out


def build_hero(hero_data: dict, hero_abilities_data: dict, abilities: dict, aghs: dict) -> dict:
    npc_name = hero_data["name"]
    short = strip_hero_prefix(npc_name)

    # hero_abilities.abilities may contain nested lists for transformation
    # heroes (e.g. Monkey King's untransform/transfigure forms). Flatten one
    # level so we capture every ability.
    raw_ab = hero_abilities_data.get("abilities", []) or []
    flat_ab = []
    for entry in raw_ab:
        if isinstance(entry, list):
            flat_ab.extend(entry)
        else:
            flat_ab.append(entry)

    seen = set()
    out_abilities = []
    for ab_name in flat_ab:
        if not ab_name or ab_name == "generic_hidden" or ab_name in seen:
            continue
        seen.add(ab_name)
        ab = abilities.get(ab_name)
        if not ab:
            continue
        out_abilities.append({
            "key":       ab_name,
            "name":      ab.get("dname", ab_name),
            "behavior":  ab.get("behavior", ""),
            "dmg_type":  ab.get("dmg_type", ""),
            "bkbpierce": ab.get("bkbpierce", ""),
            "desc":      ab.get("desc", ""),
            "cd":        ab.get("cd"),
            "mc":        ab.get("mc"),
            "attrib":    ab.get("attrib", []),
        })

    return {
        "short_name":   short,
        "name":         hero_data.get("localized_name", short),
        "primary_attr": hero_data.get("primary_attr"),
        "attack_type":  hero_data.get("attack_type"),
        "roles":        hero_data.get("roles", []),
        "base_stats": {
            "hp":            hero_data.get("base_health"),
            "hp_regen":      hero_data.get("base_health_regen"),
            "mana":          hero_data.get("base_mana"),
            "mana_regen":    hero_data.get("base_mana_regen"),
            "armor":         hero_data.get("base_armor"),
            "magic_resist":  hero_data.get("base_mr"),
            "attack_min":    hero_data.get("base_attack_min"),
            "attack_max":    hero_data.get("base_attack_max"),
            "str":           hero_data.get("base_str"),
            "agi":           hero_data.get("base_agi"),
            "int":           hero_data.get("base_int"),
            "str_gain":      hero_data.get("str_gain"),
            "agi_gain":      hero_data.get("agi_gain"),
            "int_gain":      hero_data.get("int_gain"),
            "attack_range":  hero_data.get("attack_range"),
            "attack_rate":   hero_data.get("attack_rate"),
            "move_speed":    hero_data.get("move_speed"),
            "day_vision":    hero_data.get("day_vision"),
            "night_vision":  hero_data.get("night_vision"),
        },
        "abilities": out_abilities,
        "talents":   parse_talents(hero_abilities_data.get("talents", []), abilities),
        "aghs_scepter": {
            "has":       aghs.get("has_scepter", False),
            "ability":   aghs.get("scepter_skill_name", ""),
            "new_skill": aghs.get("scepter_new_skill", False),
            "desc":      aghs.get("scepter_desc", ""),
        },
        "aghs_shard": {
            "has":       aghs.get("has_shard", False),
            "ability":   aghs.get("shard_skill_name", ""),
            "new_skill": aghs.get("shard_new_skill", False),
            "desc":      aghs.get("shard_desc", ""),
        },
    }


def build_item(short_name: str, item: dict) -> dict:
    return {
        "short_name": short_name,
        "name":       item.get("dname", short_name),
        "id":         item.get("id"),
        "cost":       item.get("cost"),
        "qual":       item.get("qual"),
        "behavior":   item.get("behavior"),
        "cd":         item.get("cd"),
        "mc":         item.get("mc"),
        "components": item.get("components"),
        "created":    item.get("created"),
        "charges":    item.get("charges"),
        "abilities":  item.get("abilities", []),
        "attrib":     item.get("attrib", []),
        "notes":      item.get("notes", ""),
    }


def build_patch(patch_list: list, patchnotes: dict) -> dict:
    latest = patch_list[-1]
    name = latest["name"]
    pn_key = name.replace(".", "_")
    pn = patchnotes.get(pn_key, {}) if isinstance(patchnotes, dict) else {}

    raw_general = pn.get("general", []) if isinstance(pn, dict) else []
    cleaned = [g for g in raw_general if g and g != "<br>"]

    return {
        "version":          name,
        "date":             latest.get("date"),
        "id":               latest.get("id"),
        "general_changes":  cleaned[:50],
        "fetched_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def collect_existing(directory: Path) -> set:
    if not directory.exists():
        return set()
    return {p.stem for p in directory.glob("*.json")}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> int:
    if not DOTACONSTANTS_BUILD.exists():
        print(f"ERROR: dotaconstants build dir not found at {DOTACONSTANTS_BUILD}", file=sys.stderr)
        print("Run `git submodule update --init --recursive` first.", file=sys.stderr)
        return 1

    def load(name: str):
        with (DOTACONSTANTS_BUILD / name).open() as f:
            return json.load(f)

    heroes         = load("heroes.json")
    hero_abilities = load("hero_abilities.json")
    abilities      = load("abilities.json")
    items          = load("items.json")
    aghs_list      = load("aghs_desc.json")
    patch_list     = load("patch.json")
    patchnotes     = load("patchnotes.json")

    aghs_by_npc = {a["hero_name"]: a for a in aghs_list}

    heroes_dir = KNOWLEDGE_ROOT / "heroes"
    items_dir  = KNOWLEDGE_ROOT / "items"

    prev_heroes = collect_existing(heroes_dir)
    prev_items  = collect_existing(items_dir)

    new_heroes = set()
    for hd in heroes.values():
        npc = hd["name"]
        ha = hero_abilities.get(npc, {})
        ag = aghs_by_npc.get(npc, {})
        flat = build_hero(hd, ha, abilities, ag)
        write_json(heroes_dir / f"{flat['short_name']}.json", flat)
        new_heroes.add(flat["short_name"])

    new_items = set()
    for short, idata in items.items():
        flat = build_item(short, idata)
        write_json(items_dir / f"{short}.json", flat)
        new_items.add(short)

    patch_obj = build_patch(patch_list, patchnotes)
    write_json(KNOWLEDGE_ROOT / "patch.json", patch_obj)

    # Remove stale files for entities that no longer exist upstream
    for stale in prev_heroes - new_heroes:
        (heroes_dir / f"{stale}.json").unlink(missing_ok=True)
    for stale in prev_items - new_items:
        (items_dir / f"{stale}.json").unlink(missing_ok=True)

    added_heroes   = sorted(new_heroes - prev_heroes)
    removed_heroes = sorted(prev_heroes - new_heroes)
    added_items    = sorted(new_items - prev_items)
    removed_items  = sorted(prev_items - new_items)

    print(f"Patch:  {patch_obj['version']}  (released {patch_obj['date']})")
    print(f"Heroes: {len(new_heroes)} total  ({len(added_heroes)} added, {len(removed_heroes)} removed)")
    if added_heroes:
        print(f"  + {added_heroes}")
    if removed_heroes:
        print(f"  - {removed_heroes}")
    print(f"Items:  {len(new_items)} total  ({len(added_items)} added, {len(removed_items)} removed)")
    if added_items:
        print(f"  + {added_items}")
    if removed_items:
        print(f"  - {removed_items}")
    print(f"Output: {KNOWLEDGE_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
