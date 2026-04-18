#!/usr/bin/env python3
"""Diagnostic evaluator for the Dota 2 icon classifier.

Two subcommands:

    init-labels   Write a stub labels.json from a screenshots directory.
    run           Run evaluation, emit a 5-section report + report.json.

The `run` subcommand reuses the production preprocessing pipeline from
inference_service.py so the accuracy numbers reflect what the deployed
service actually does.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# Make the project root importable so we can pull preprocessing from the
# same source the inference service uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference_service import (  # noqa: E402
    compute_item_boxes,
    load_anchor_assets,
    preprocess_crop,
)


HERO_PREFIXES = ("radiant_hero", "dire_hero")
IMG_EXTS = (".png", ".jpg", ".jpeg")
HIST_LABELS = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
               "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]


def is_hero_slot(name: str) -> bool:
    return name.startswith(HERO_PREFIXES)


# ---------- init-labels ----------

def cmd_init_labels(args):
    crop_config = json.loads(Path(args.crop_config).read_text())
    slot_names = list(crop_config["regions"].keys())

    screenshots = sorted(
        p.name for p in Path(args.screenshots).iterdir()
        if p.suffix.lower() in IMG_EXTS
    )
    if not screenshots:
        print(f"No screenshots found in {args.screenshots}", file=sys.stderr)
        sys.exit(1)

    stub = {name: {slot: "" for slot in slot_names} for name in screenshots}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stub, indent=2))
    print(f"Wrote stub for {len(screenshots)} screenshots × "
          f"{len(slot_names)} slots → {out}")
    print("Fill in each slot value with a class name "
          "(e.g. \"antimage\", \"blink\", \"empty\"). "
          "Leave \"\" to skip a slot.")


# ---------- run ----------

def load_class_list(path: str) -> list[str]:
    raw = json.loads(Path(path).read_text())
    return [raw[str(i)] for i in range(len(raw))]


def crop_screenshot(image: Image.Image, crop_config: dict,
                    anchor_cfg: dict | None = None,
                    anchor_template=None):
    """Yield (slot_name, crop, anchor_meta) for each region.

    anchor_meta is the same dict for every yielded tuple (one match per
    image). When anchor_cfg/template are None or the match falls below
    threshold, item crops use the fixed coords from crop_config.
    """
    img_w, img_h = image.size
    ref_w, ref_h = crop_config["reference_resolution"]
    sx = img_w / ref_w
    sy = img_h / ref_h
    img_np = np.array(image)
    item_boxes, anchor_meta = compute_item_boxes(
        img_np, anchor_cfg, anchor_template, sx, sy
    )
    for name, c in crop_config["regions"].items():
        if name in item_boxes:
            x, y, w, h = item_boxes[name]
        else:
            x = int(c["x"] * sx)
            y = int(c["y"] * sy)
            w = int(c["w"] * sx)
            h = int(c["h"] * sy)
        yield name, image.crop((x, y, x + w, y + h)), anchor_meta


def classify_batch(session, crops, class_names):
    """Return [(top_class_name, top_conf), ...] — no threshold applied."""
    if not crops:
        return []
    batch = np.stack(crops).astype(np.float32)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: batch})[0]
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    out = []
    for prob in probs:
        idx = int(prob.argmax())
        out.append((class_names[idx], float(prob[idx])))
    return out


def validate_labels(labels: dict, region_names: list[str],
                    hero_classes: list[str], item_classes: list[str]) -> int:
    """Print warnings for unknown slot or class names. Returns warning count."""
    valid_slots = set(region_names)
    hero_set = set(hero_classes)
    item_set = set(item_classes)
    warnings = 0
    for fname, slots in labels.items():
        for slot, cls in slots.items():
            if slot not in valid_slots:
                print(f"WARN {fname}: unknown slot key '{slot}'", file=sys.stderr)
                warnings += 1
                continue
            if cls == "":
                continue
            target = hero_set if is_hero_slot(slot) else item_set
            kind = "hero" if is_hero_slot(slot) else "item"
            if cls not in target:
                print(f"WARN {fname}.{slot}: '{cls}' not in {kind} class list",
                      file=sys.stderr)
                warnings += 1
    return warnings


def cmd_run(args):
    crop_config = json.loads(Path(args.crop_config).read_text())
    region_names = list(crop_config["regions"].keys())
    hero_class_names = load_class_list(args.hero_classes)
    item_class_names = load_class_list(args.item_classes)
    labels = json.loads(Path(args.ground_truth).read_text())

    if not labels:
        print("labels.json is empty — nothing to evaluate.", file=sys.stderr)
        sys.exit(1)

    n_warn = validate_labels(labels, region_names, hero_class_names, item_class_names)
    if n_warn:
        print(f"({n_warn} validation warning(s) — proceeding anyway)\n",
              file=sys.stderr)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    hero_session = ort.InferenceSession(args.hero_model, providers=providers)
    item_session = ort.InferenceSession(args.item_model, providers=providers)

    # Optional anchor: workspace = directory containing configs/
    workspace = Path(args.crop_config).resolve().parent.parent
    anchor_cfg, anchor_template = load_anchor_assets(workspace)
    if anchor_template is not None:
        print(f"Anchor enabled: {anchor_cfg['anchor']!r}, "
              f"threshold={anchor_cfg['match_threshold']}")
    else:
        print("Anchor not configured; using fixed item crops")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = output_dir / "failures"
    if args.save_crops:
        failures_dir.mkdir(exist_ok=True)

    screenshot_dir = Path(args.screenshots)
    records = []  # one per (file, slot)
    anchor_scores = []  # per-screenshot match scores (or None)

    for fname, slot_labels in labels.items():
        path = screenshot_dir / fname
        if not path.exists():
            print(f"WARN: missing screenshot {path}", file=sys.stderr)
            continue

        image = Image.open(path).convert("RGB")
        hero_crops, hero_slots = [], []
        item_crops, item_slots = [], []
        crops_by_slot = {}
        per_image_anchor: dict | None = None

        for slot, crop_img, anchor_meta in crop_screenshot(
                image, crop_config, anchor_cfg, anchor_template):
            per_image_anchor = anchor_meta
            crops_by_slot[slot] = crop_img
            processed = preprocess_crop(crop_img)
            if is_hero_slot(slot):
                hero_crops.append(processed)
                hero_slots.append(slot)
            else:
                item_crops.append(processed)
                item_slots.append(slot)

        results = []
        results.extend(zip(hero_slots, classify_batch(hero_session, hero_crops, hero_class_names)))
        results.extend(zip(item_slots, classify_batch(item_session, item_crops, item_class_names)))

        anchor_scores.append({
            "file": fname,
            "score": per_image_anchor["score"] if per_image_anchor else None,
            "used": per_image_anchor["used"] if per_image_anchor else False,
        })

        for slot, (pred, conf) in results:
            true = slot_labels.get(slot, "")
            kind = "hero" if is_hero_slot(slot) else "item"
            records.append({
                "file": fname, "slot": slot, "kind": kind,
                "true": true, "pred": pred, "conf": conf,
            })

            # Dump wrong crops (above threshold and pred != true)
            if (args.save_crops and true and true != pred
                    and conf >= args.confidence_threshold):
                stem = Path(fname).stem
                safe = lambda s: s.replace("/", "_").replace(" ", "_")
                out_path = (failures_dir
                            / f"{stem}__{slot}__GT-{safe(true)}"
                              f"__PRED-{safe(pred)}__conf-{conf:.2f}.png")
                crops_by_slot[slot].save(out_path)

    metrics = compute_metrics(records, args.confidence_threshold)
    metrics["anchor_scores"] = anchor_scores
    render_report(metrics)

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nReport saved to {report_path}")
    if args.save_crops:
        n = len(list(failures_dir.glob("*.png")))
        print(f"{n} failure crops saved to {failures_dir}")


# ---------- metrics ----------

def outcome_for(rec, threshold):
    if rec["true"] == "":
        return "unlabeled"
    if rec["conf"] < threshold:
        return "unknown"
    return "correct" if rec["true"] == rec["pred"] else "wrong"


def compute_metrics(records, threshold):
    n_total = len(records)
    n_labeled = sum(1 for r in records if r["true"] != "")
    n_unlabeled = n_total - n_labeled

    counts = Counter(outcome_for(r, threshold) for r in records)
    n_correct = counts.get("correct", 0)
    n_wrong = counts.get("wrong", 0)
    n_unknown = counts.get("unknown", 0)
    n_decided = n_correct + n_wrong

    by_kind = defaultdict(Counter)
    for r in records:
        by_kind[r["kind"]][outcome_for(r, threshold)] += 1

    # Per-slot
    per_slot = defaultdict(lambda: {"correct": 0, "wrong": 0, "unknown": 0,
                                    "conf_correct": [], "conf_wrong": []})
    for r in records:
        o = outcome_for(r, threshold)
        if o == "unlabeled":
            continue
        per_slot[r["slot"]][o] += 1
        if o == "correct":
            per_slot[r["slot"]]["conf_correct"].append(r["conf"])
        elif o == "wrong":
            per_slot[r["slot"]]["conf_wrong"].append(r["conf"])

    slot_rows = []
    for slot, d in per_slot.items():
        decided = d["correct"] + d["wrong"]
        slot_rows.append({
            "slot": slot,
            "n_correct": d["correct"],
            "n_wrong": d["wrong"],
            "n_unknown": d["unknown"],
            "accuracy": (d["correct"] / decided) if decided else None,
            "avg_conf_correct": (float(np.mean(d["conf_correct"]))
                                 if d["conf_correct"] else None),
            "avg_conf_wrong": (float(np.mean(d["conf_wrong"]))
                               if d["conf_wrong"] else None),
        })
    slot_rows.sort(key=lambda r: (r["accuracy"] if r["accuracy"] is not None
                                  else 1.0, r["slot"]))

    # Per-class
    per_class = {
        "hero": defaultdict(lambda: {"n": 0, "correct": 0,
                                     "wrongs": Counter(),
                                     "wrong_confs": defaultdict(list)}),
        "item": defaultdict(lambda: {"n": 0, "correct": 0,
                                     "wrongs": Counter(),
                                     "wrong_confs": defaultdict(list)}),
    }
    for r in records:
        o = outcome_for(r, threshold)
        if o == "unlabeled":
            continue
        d = per_class[r["kind"]][r["true"]]
        d["n"] += 1
        if o == "correct":
            d["correct"] += 1
        elif o == "wrong":
            d["wrongs"][r["pred"]] += 1
            d["wrong_confs"][r["pred"]].append(r["conf"])
        else:  # unknown
            d["wrongs"]["<unknown>"] += 1

    class_rows = {}
    for kind, classes in per_class.items():
        rows = []
        for cls, d in classes.items():
            most_wrong = None
            if d["wrongs"]:
                pred, count = d["wrongs"].most_common(1)[0]
                confs = d["wrong_confs"][pred]
                most_wrong = {
                    "pred": pred, "count": count,
                    "avg_conf": float(np.mean(confs)) if confs else None,
                }
            rows.append({
                "true_class": cls,
                "n_seen": d["n"],
                "n_correct": d["correct"],
                "accuracy": (d["correct"] / d["n"]) if d["n"] else None,
                "most_frequent_wrong": most_wrong,
            })
        rows.sort(key=lambda r: (r["accuracy"] if r["accuracy"] is not None
                                 else 1.0, -r["n_seen"]))
        class_rows[kind] = rows

    confusion = Counter()
    for r in records:
        if outcome_for(r, threshold) == "wrong":
            confusion[(r["true"], r["pred"])] += 1
    confusion_pairs = [{"true": t, "pred": p, "count": c}
                       for (t, p), c in confusion.most_common(20)]

    hist = {"correct": [0] * 10, "wrong": [0] * 10, "unknown": [0] * 10}
    for r in records:
        o = outcome_for(r, threshold)
        if o == "unlabeled":
            continue
        bucket = min(int(r["conf"] * 10), 9)
        hist[o][bucket] += 1

    return {
        "threshold": threshold,
        "n_screenshots": len({r["file"] for r in records}),
        "n_predictions_total": n_total,
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_unknown": n_unknown,
        "overall_accuracy": (n_correct / n_decided) if n_decided else None,
        "by_kind": {
            kind: {
                "n_correct": c.get("correct", 0),
                "n_wrong": c.get("wrong", 0),
                "n_unknown": c.get("unknown", 0),
                "accuracy": (c.get("correct", 0)
                             / (c.get("correct", 0) + c.get("wrong", 0)))
                            if (c.get("correct", 0) + c.get("wrong", 0)) else None,
            }
            for kind, c in by_kind.items()
        },
        "per_slot": slot_rows,
        "per_class": class_rows,
        "confusion_pairs": confusion_pairs,
        "confidence_histogram": {
            "buckets": HIST_LABELS,
            "correct": hist["correct"],
            "wrong": hist["wrong"],
            "unknown": hist["unknown"],
        },
    }


# ---------- rendering ----------

def fmt_pct(x):
    return f"{x*100:5.1f}%" if x is not None else "  -  "


def fmt_conf(x):
    return f"{x:.2f}" if x is not None else "  -  "


def render_report(m):
    sep = "=" * 78

    print()
    print(sep)
    print("HEADLINE")
    print(sep)
    print(f"Screenshots evaluated: {m['n_screenshots']}")
    print(f"Slot predictions:      {m['n_predictions_total']} "
          f"({m['n_labeled']} labeled, {m['n_unlabeled']} unlabeled — skipped)")
    decided = m['n_correct'] + m['n_wrong']
    print(f"Overall accuracy:      {fmt_pct(m['overall_accuracy'])}  "
          f"({m['n_correct']}/{decided})")
    for kind in ("hero", "item"):
        if kind not in m["by_kind"]:
            continue
        d = m["by_kind"][kind]
        n = d["n_correct"] + d["n_wrong"]
        label = "Heroes:" if kind == "hero" else "Items:"
        print(f"  {label:<8}             {fmt_pct(d['accuracy'])}  "
              f"({d['n_correct']}/{n})")
    if m["n_labeled"]:
        rate = m["n_unknown"] / m["n_labeled"]
        print(f"\"unknown\" rate:        {fmt_pct(rate)}  "
              f"({m['n_unknown']}/{m['n_labeled']}) — model conf < {m['threshold']}")

    print()
    print(sep)
    print("PER-SLOT ACCURACY  (worst first)")
    print(sep)
    print(f"{'SLOT':<18}{'ACC':<10}{'CORRECT/DECIDED':<22}"
          f"{'AVG_CONF_OK':<14}{'AVG_CONF_WRONG'}")
    for row in m["per_slot"]:
        decided = row["n_correct"] + row["n_wrong"]
        cnt = f"{row['n_correct']}/{decided}"
        if row["n_unknown"]:
            cnt += f" (+{row['n_unknown']} unk)"
        print(f"{row['slot']:<18}{fmt_pct(row['accuracy']):<10}{cnt:<22}"
              f"{fmt_conf(row['avg_conf_correct']):<14}"
              f"{fmt_conf(row['avg_conf_wrong'])}")

    for kind in ("hero", "item"):
        rows = m["per_class"].get(kind, [])
        if not rows:
            continue
        print()
        print(sep)
        print(f"PER-CLASS ACCURACY — {kind.upper()}S  "
              "(worst first; classes that appeared in labels)")
        print(sep)
        print(f"{'TRUE_CLASS':<26}{'N_SEEN':<8}{'ACC':<10}"
              f"MOST_FREQUENT_WRONG_PREDICTION")
        for row in rows:
            mw = row["most_frequent_wrong"]
            if mw:
                conf_part = (f", avg conf {mw['avg_conf']:.2f}"
                             if mw["avg_conf"] is not None else "")
                wrong_str = f'"{mw["pred"]}" ({mw["count"]}x{conf_part})'
            else:
                wrong_str = "-"
            print(f"{row['true_class']:<26}{row['n_seen']:<8}"
                  f"{fmt_pct(row['accuracy']):<10}{wrong_str}")

    print()
    print(sep)
    print("TOP CONFUSION PAIRS  (top 20)")
    print(sep)
    if not m["confusion_pairs"]:
        print("(none)")
    else:
        print(f"{'TRUE -> PREDICTED':<50}COUNT")
        for pair in m["confusion_pairs"]:
            arrow = f"{pair['true']} -> {pair['pred']}"
            print(f"{arrow:<50}{pair['count']}")

    if m.get("anchor_scores"):
        print()
        print(sep)
        print("ANCHOR MATCH SCORES  (per screenshot)")
        print(sep)
        scores = [a["score"] for a in m["anchor_scores"] if a["score"] is not None]
        if not scores:
            print("(anchor not configured)")
        else:
            n_used = sum(1 for a in m["anchor_scores"] if a["used"])
            n_total = len(m["anchor_scores"])
            print(f"Used: {n_used}/{n_total}   "
                  f"min={min(scores):.3f}  median={float(np.median(scores)):.3f}  "
                  f"max={max(scores):.3f}")
            for a in sorted(m["anchor_scores"],
                            key=lambda x: (x["score"] is None, x["score"] or 0)):
                tag = "OK " if a["used"] else "FB "  # FB = fell back to fixed
                s = f"{a['score']:.3f}" if a["score"] is not None else "  -  "
                print(f"  {tag} {s}  {a['file']}")

    print()
    print(sep)
    print(f"CONFIDENCE HISTOGRAM  (raw model confidence; "
          f"<{m['threshold']} = \"unknown\")")
    print(sep)
    h = m["confidence_histogram"]
    print(f"{'BUCKET':<14}{'CORRECT':<12}{'WRONG':<12}{'UNKNOWN':<12}")
    for i, b in enumerate(h["buckets"]):
        print(f"{b:<14}{h['correct'][i]:<12}{h['wrong'][i]:<12}{h['unknown'][i]:<12}")


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init-labels", help="Write a stub labels.json")
    init.add_argument("--screenshots", required=True,
                      help="Directory containing screenshot images")
    init.add_argument("--output", required=True, help="Path to write labels.json")
    init.add_argument("--crop-config", default="configs/crop_config.json")
    init.set_defaults(func=cmd_init_labels)

    run = sub.add_parser("run", help="Evaluate against ground-truth labels")
    run.add_argument("--screenshots", required=True)
    run.add_argument("--ground-truth", required=True, help="labels.json path")
    run.add_argument("--crop-config", default="configs/crop_config.json")
    run.add_argument("--hero-model", default="models/hero_classifier.onnx")
    run.add_argument("--item-model", default="models/item_classifier.onnx")
    run.add_argument("--hero-classes", default="configs/heroes_classes.json")
    run.add_argument("--item-classes", default="configs/items_classes.json")
    run.add_argument("--confidence-threshold", type=float, default=0.5)
    run.add_argument("--output-dir", required=True,
                     help="Directory for report.json and (optional) failure crops")
    run.add_argument("--save-crops", action="store_true",
                     help="Dump every wrong crop as PNG into output-dir/failures/")
    run.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
