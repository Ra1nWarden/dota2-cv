"""Load data/dota_knowledge/ and render the per-match Markdown prefix
that the /tips endpoint feeds the LLM as a cached system prompt.

Two public entry points:

    load_knowledge(root) -> Knowledge
        Read every JSON under root into memory at startup.

    build_prefix(kb, fused_snapshot) -> str
        Per-match. Render patch summary + the 10 drafted heroes
        + the full item catalog + coaching rules. Output is byte-stable
        across polls of the same match because items render in sorted
        order and only the input snapshot's hero list affects the body.

    build_user_block(fused_snapshot, recent_tips) -> str
        Per-poll. Render the current snapshot as a short user message.

See spec/202604302055-tips-endpoint-llm.md (Revision 3) for the design
rationale on rendering all 501 items at full detail rather than a
curated subset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Knowledge:
    """In-memory mirror of data/dota_knowledge/."""
    patch: dict
    heroes: dict[str, dict]   # short_name -> hero entry
    items: dict[str, dict]    # short_name -> item entry


def load_knowledge(root: Path) -> Knowledge:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"knowledge root not found: {root}")

    with (root / "patch.json").open() as f:
        patch = json.load(f)

    heroes: dict[str, dict] = {}
    for p in sorted((root / "heroes").glob("*.json")):
        with p.open() as f:
            entry = json.load(f)
        heroes[entry["short_name"]] = entry

    items: dict[str, dict] = {}
    for p in sorted((root / "items").glob("*.json")):
        with p.open() as f:
            entry = json.load(f)
        items[entry["short_name"]] = entry

    return Knowledge(patch=patch, heroes=heroes, items=items)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_value(value: Any) -> str:
    """Render an attrib value: list → '25/30/35/40', scalar → '50'."""
    if isinstance(value, list):
        return " / ".join(str(v) for v in value)
    return str(value)


def _clean_name(name: str) -> str:
    """Normalize whitespace in dotaconstants display names. Some entries
    contain literal '\\n' sequences (two characters: backslash + n,
    used by Valve as line breaks in the in-game UI), and some contain
    real newlines. Both should collapse to single spaces."""
    if not name:
        return ""
    return " ".join(name.replace("\\n", " ").split())


def _render_behavior(behavior: Any) -> str:
    """Behavior may be a string or a list (e.g. ['Unit Target', 'AOE'])."""
    if isinstance(behavior, list):
        return ", ".join(behavior)
    return str(behavior or "")


def _render_header(header: str) -> str:
    """'MANA BURNED PER HIT:' -> 'Mana Burned Per Hit'."""
    if not header:
        return ""
    h = header.rstrip(":").strip()
    return h.title() if h else ""


def _is_useful_attrib(a: dict) -> bool:
    """Skip attribs that dotaconstants synthesizes as zeros — they're
    placeholders for talents/scepter values that don't apply at base."""
    if not a.get("generated"):
        return True
    val = a.get("value")
    if isinstance(val, list):
        return any(str(v) not in ("0", "0.0", "") for v in val)
    return str(val) not in ("0", "0.0", "", "None")


def _render_ability(ab: dict) -> list[str]:
    """Return a list of rendered lines for a single hero ability."""
    name = ab.get("name", ab.get("key", "?"))
    behavior = _render_behavior(ab.get("behavior"))
    dmg = ab.get("dmg_type") or ""
    bkb = ab.get("bkbpierce")

    head_bits = [b for b in (behavior, dmg) if b]
    if bkb and bkb not in ("", "None"):
        head_bits.append(f"BKB-pierces {bkb}")
    head = ", ".join(head_bits)

    lines = [f"  - **{name}**" + (f" ({head})" if head else "")]

    desc = (ab.get("desc") or "").strip()
    if desc:
        lines.append(f"    {desc}")

    cd = ab.get("cd")
    mc = ab.get("mc")
    cd_part = f"CD: {_render_value(cd)}" if cd not in (None, False, "", 0) else ""
    mc_part = f"Mana: {_render_value(mc)}" if mc not in (None, False, "", 0) else ""
    if cd_part or mc_part:
        lines.append("    " + " | ".join(p for p in (cd_part, mc_part) if p))

    for a in ab.get("attrib") or []:
        if not _is_useful_attrib(a):
            continue
        header = _render_header(a.get("header") or a.get("key") or "")
        value = _render_value(a.get("value"))
        if header:
            lines.append(f"    {header}: {value}")
        else:
            lines.append(f"    {value}")

    return lines


def _render_talents(talents: list[dict]) -> list[str]:
    if not talents:
        return []
    lines = ["Talents:"]
    for t in talents:
        level = t.get("level")
        left = t.get("left", "") or ""
        right = t.get("right", "") or ""
        lines.append(f"  L{level}: {left}  |  {right}")
    return lines


def _render_aghs(label: str, block: dict) -> list[str]:
    if not block or not block.get("has"):
        return []
    ability = block.get("ability") or ""
    desc = (block.get("desc") or "").strip()
    new = " — NEW SKILL" if block.get("new_skill") else ""
    suffix = f" ({ability})" if ability else ""
    return [f"{label}{suffix}{new}: {desc}" if desc else f"{label}{suffix}{new}"]


def _render_hero(hero: dict) -> str:
    """Render one hero's full detail block."""
    name = _clean_name(hero.get("name") or hero.get("short_name") or "?")
    roles = " / ".join(hero.get("roles") or []) or "—"
    bs = hero.get("base_stats") or {}

    primary_attr_label = {"agi": "Agility", "str": "Strength", "int": "Intelligence",
                         "all": "Universal"}.get(hero.get("primary_attr") or "", hero.get("primary_attr") or "")
    attack_type = hero.get("attack_type") or ""

    lines = [f"## {name} — {roles}"]

    head_bits = []
    if primary_attr_label:
        head_bits.append(f"Primary: {primary_attr_label}")
    if attack_type:
        head_bits.append(attack_type)
    if bs.get("move_speed") is not None:
        head_bits.append(f"{bs['move_speed']} move speed")
    if bs.get("attack_range") is not None:
        head_bits.append(f"{bs['attack_range']} attack range")
    if head_bits:
        lines.append(" | ".join(head_bits))

    base_bits = []
    if bs.get("hp") is not None:
        base_bits.append(f"HP {bs['hp']}")
    if bs.get("mana") is not None:
        base_bits.append(f"mana {bs['mana']}")
    if bs.get("armor") is not None:
        base_bits.append(f"armor {bs['armor']}")
    if bs.get("magic_resist") is not None:
        base_bits.append(f"MR {bs['magic_resist']}%")
    if bs.get("attack_min") is not None and bs.get("attack_max") is not None:
        base_bits.append(f"attack {bs['attack_min']}-{bs['attack_max']}")
    if base_bits:
        lines.append("Base: " + ", ".join(base_bits))

    gain_bits = []
    for k, label in (("str", "STR"), ("agi", "AGI"), ("int", "INT")):
        gain = bs.get(f"{k}_gain")
        base = bs.get(k)
        if base is not None and gain is not None:
            gain_bits.append(f"{label} {base}+{gain}")
    if gain_bits:
        lines.append("Stats: " + ", ".join(gain_bits))

    abilities = hero.get("abilities") or []
    if abilities:
        lines.append("Abilities:")
        for ab in abilities:
            lines.extend(_render_ability(ab))

    lines.extend(_render_talents(hero.get("talents") or []))

    lines.extend(_render_aghs("Aghanim's Scepter", hero.get("aghs_scepter") or {}))
    lines.extend(_render_aghs("Aghanim's Shard", hero.get("aghs_shard") or {}))

    return "\n".join(lines)


def _render_item_attrib(attrib: list[dict]) -> str:
    """Render item attribs that have a `display` template — i.e. the
    user-visible stats Valve labels with text like '+50 Damage'. We
    skip raw mechanic attribs (like `cleave_starting_width: 150`)
    because they duplicate detail already present in the item's
    ability descriptions and bloat the prefix without adding value
    for tip generation."""
    parts: list[str] = []
    for a in attrib or []:
        display = a.get("display") or ""
        if not (display and "{value}" in display):
            continue
        value = _render_value(a.get("value"))
        parts.append(display.replace("{value}", value).strip())
    return ", ".join(p for p in parts if p)


def _render_components(components: Any, items_by_short: dict[str, dict]) -> str:
    if not components:
        return ""
    names = []
    for c in components:
        if not c:
            continue
        name = _clean_name(items_by_short.get(c, {}).get("name") or c)
        if name:
            names.append(name)
    return " + ".join(names)


def _render_item(item: dict, items_by_short: dict[str, dict]) -> str:
    """Render one item. Adaptive: trivial items stay one-line; complex
    items expand to abilities."""
    name = _clean_name(item.get("name") or item.get("short_name") or "?")
    cost = item.get("cost")
    qual = item.get("qual") or ""
    behavior = _render_behavior(item.get("behavior"))
    cd = item.get("cd")
    mc = item.get("mc")

    head_bits = [name]
    cost_bits = []
    if cost not in (None, False):
        cost_bits.append(f"{cost}g")
    if qual:
        cost_bits.append(qual)
    head = f"**{name}**"
    if cost_bits:
        head += f" ({', '.join(cost_bits)})"

    meta_bits = []
    if behavior:
        meta_bits.append(behavior)
    if cd not in (None, False, 0, ""):
        meta_bits.append(f"CD {_render_value(cd)}s")
    if mc not in (None, False, 0, ""):
        meta_bits.append(f"Mana {_render_value(mc)}")
    head_line = head + ((" | " + " | ".join(meta_bits)) if meta_bits else "")

    components = _render_components(item.get("components"), items_by_short)
    stats = _render_item_attrib(item.get("attrib") or [])
    abilities = item.get("abilities") or []
    # Notes are dropped: they're typically long edge-case clarifications
    # (e.g. "Cleave damage goes through spell immunity") that the LLM
    # rarely needs to reason about for one-sentence tips, and they add
    # ~10-30 tokens per item across 501 items.

    if not components and not stats and not abilities:
        return f"- {head_line}"

    lines = [f"- {head_line}"]
    if components:
        lines.append(f"  Components: {components}")
    if stats:
        lines.append(f"  Stats: {stats}")
    for a in abilities:
        kind = a.get("type", "")
        title = a.get("title", "")
        desc = (a.get("description") or "").strip()
        lines.append(f"  {kind.capitalize()} \"{title}\": {desc}")

    return "\n".join(lines)


def _render_patch(patch: dict) -> str:
    version = patch.get("version") or "?"
    date = (patch.get("date") or "")[:10]
    lines = [f"# Patch {version}" + (f" — {date}" if date else "")]
    changes = patch.get("general_changes") or []
    if changes:
        lines.append("Key general changes:")
        for c in changes:
            text = (c or "").strip()
            if text:
                lines.append(f"- {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Match start: full prefix
# ---------------------------------------------------------------------------


COACHING_RULES = """\
# Coaching ground rules
- Output exactly ONE sentence per tip, under 25 words.
- Be specific and actionable: "Push mid tower while AM has Aegis" beats
  "Try to push lanes."
- If a number you want to cite is not in the data above, say you are
  uncertain — DO NOT invent values from memory.
- Trust the patch and values above over your training knowledge; the
  game updates frequently and your training data is stale.
- Avoid hedging language ("you might want to consider"); give one clear
  recommendation.
"""


def _heroes_in_match(fused: dict) -> list[str]:
    """Extract the 10 hero short names from a fused snapshot."""
    out: list[str] = []
    for side in ("radiant", "dire"):
        for slot in fused.get(side) or []:
            hero = (slot.get("hero") or "").strip()
            if hero:
                out.append(hero)
    return out


def build_prefix(kb: Knowledge, fused: dict) -> str:
    """Per-match prefix, byte-stable for the duration of the match."""
    parts: list[str] = []

    parts.append(_render_patch(kb.patch))

    heroes = _heroes_in_match(fused)
    if heroes:
        parts.append("\n# Heroes in this match\n")
        for short in heroes:
            entry = kb.heroes.get(short)
            if entry is None:
                parts.append(f"## {short}\n(no knowledge entry available)")
            else:
                parts.append(_render_hero(entry))

    parts.append("\n# Item reference (full catalog)\n")
    for short in sorted(kb.items.keys()):
        parts.append(_render_item(kb.items[short], kb.items))

    parts.append("\n" + COACHING_RULES)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Per-poll: user block
# ---------------------------------------------------------------------------


def _format_clock(seconds: int | None) -> str:
    if seconds is None:
        return "?"
    sign = "-" if seconds < 0 else ""
    s = abs(int(seconds))
    return f"{sign}{s // 60}:{s % 60:02d}"


def _render_slot(slot: dict, item_lookup: dict[str, dict]) -> str:
    hero = slot.get("hero") or "unknown"
    name = item_lookup.get(hero, {}).get("name") or hero
    is_player = slot.get("is_player")
    items = slot.get("items") or []
    item_names = [item_lookup.get(i, {}).get("name") or i for i in items]
    items_str = ", ".join(item_names) if item_names else "items unknown"
    marker = " (you)" if is_player else ""
    return f"  - {name}{marker}: {items_str}"


def build_user_block(
    fused: dict,
    recent_tips: list[str] | None = None,
    *,
    items_index: dict[str, dict] | None = None,
) -> str:
    """Per-poll user message. Compact: ~300-500 tokens."""
    items_index = items_index or {}
    recent_tips = recent_tips or []

    lines = [f"Game time: {_format_clock(fused.get('game_time_s'))}"]

    radiant = fused.get("radiant") or []
    dire = fused.get("dire") or []
    if radiant:
        lines.append("Radiant:")
        for s in radiant:
            lines.append(_render_slot(s, items_index))
    if dire:
        lines.append("Dire:")
        for s in dire:
            lines.append(_render_slot(s, items_index))

    if recent_tips:
        lines.append("")
        lines.append("Recent tips (most recent first, do not repeat):")
        for t in recent_tips[:5]:
            lines.append(f"  - {t}")

    return "\n".join(lines)
