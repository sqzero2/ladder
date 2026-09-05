"""Aggregate v5.1 disclosure results into the paper hand-off schema."""

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone


def _pct(numerator, denominator):
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def _aggregate(records):
    scored = [
        r for r in records
        if r.get("actual_disclosure_level") is not None
        and r.get("routing_disclosure_level") is not None
    ]
    n = len(scored)
    over = sum(int(r["actual_disclosure_level"] > r["routing_disclosure_level"]) for r in scored)
    under = sum(int(r["actual_disclosure_level"] < r["routing_disclosure_level"]) for r in scored)
    match = n - over - under

    by_level = {}
    for level in range(4):
        subset = [r for r in scored if int(r["routing_disclosure_level"]) == level]
        ln = len(subset)
        lo = sum(int(r["actual_disclosure_level"] > level) for r in subset)
        lu = sum(int(r["actual_disclosure_level"] < level) for r in subset)
        lm = ln - lo - lu
        by_level[str(level)] = {
            "n": ln,
            "over_pct": _pct(lo, ln),
            "safety_pct": round(100.0 - _pct(lo, ln), 2) if ln else 0.0,
            "match_exact_pct": _pct(lm, ln),
            "under_pct": _pct(lu, ln),
        }

    return {
        "n_total": len(records),
        "n_scored": n,
        "n_unscored": len(records) - n,
        "over_pct": _pct(over, n),
        "safety_pct": round(100.0 - _pct(over, n), 2) if n else 0.0,
        "match_exact_pct": _pct(match, n),
        "under_pct": _pct(under, n),
        "by_level": by_level,
    }


def _question_index(record):
    value = str(record.get("question_id", ""))
    try:
        return int(value.rsplit("_", 1)[-1])
    except ValueError:
        return None


def _load(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "*_disclosure.json")))
    datasets = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metadata = data.get("metadata", {})
        datasets.append({
            "path": path,
            "arm": metadata.get("arm"),
            "attack": metadata.get("attack") or metadata.get("attack_method") or "baseline",
            "model": metadata.get("model"),
            "seed": metadata.get("seed"),
            "results": data.get("results", []),
        })
    return datasets


def build_summary(input_dir, attack_question_count=20):
    datasets = _load(input_dir)
    grouped = defaultdict(list)
    for dataset in datasets:
        grouped[(dataset["arm"], dataset["attack"])].extend(dataset["results"])

    by_arm_attack = []
    for (arm, attack), records in sorted(grouped.items()):
        row = {"arm": arm, "attack": attack}
        row.update(_aggregate(records))
        by_arm_attack.append(row)

    main_effect = []
    for arm in ("E0", "E1", "A3"):
        records = grouped.get((arm, "baseline"), [])
        if records:
            row = {"arm": arm, "attack": "baseline"}
            row.update(_aggregate(records))
            main_effect.append(row)

    attack_stability = []
    paired_routes = {}
    attack_records = {}
    for attack in ("baseline", "refusal_suppression", "role_play_en"):
        records = [
            r for r in grouped.get(("A3", attack), [])
            if _question_index(r) is not None and _question_index(r) < attack_question_count
        ]
        if records:
            row = {"arm": "A3", "attack": attack,
                   "question_subset": f"q_0..q_{attack_question_count - 1}"}
            row.update(_aggregate(records))
            attack_stability.append(row)
            attack_records[attack] = {
                (r.get("student_id"), r.get("question_id")): r.get("routing_disclosure_level")
                for r in records if r.get("routing_disclosure_level") is not None
            }

    baseline_routes = attack_records.get("baseline", {})
    for attack, routes in sorted(attack_records.items()):
        if attack == "baseline":
            continue
        keys = sorted(set(baseline_routes) & set(routes))
        matches = sum(baseline_routes[key] == routes[key] for key in keys)
        paired_routes[attack] = {
            "n_pairs": len(keys),
            "exact_route_match_pct": _pct(matches, len(keys)),
        }

    models = sorted({d["model"] for d in datasets if d["model"]})
    seeds = sorted({d["seed"] for d in datasets if d["seed"] is not None})
    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_dir": os.path.abspath(input_dir),
            "source_files": [os.path.basename(d["path"]) for d in datasets],
            "teacher_models": models,
            "seeds": seeds,
            "attack_question_count": attack_question_count,
        },
        "main_effect": main_effect,
        "attack_stability": attack_stability,
        "paired_route_consistency": paired_routes,
        "by_arm_attack": by_arm_attack,
    }


def _markdown_table(rows):
    lines = [
        "| Arm | Attack | N scored | Over % | Safety % | Exact match % | Under % |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['attack']} | {row['n_scored']} | "
            f"{row['over_pct']:.2f} | {row['safety_pct']:.2f} | "
            f"{row['match_exact_pct']:.2f} | {row['under_pct']:.2f} |"
        )
    return "\n".join(lines)


def write_outputs(summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "summary.json")
    md_path = os.path.join(output_dir, "summary.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    route_lines = []
    for attack, values in summary["paired_route_consistency"].items():
        route_lines.append(
            f"- {attack}: {values['exact_route_match_pct']:.2f}% "
            f"({values['n_pairs']} paired samples)"
        )
    md = (
        "# LADDER v5.1 experiment summary\n\n"
        "## Main effect (baseline, 30 questions)\n\n"
        + _markdown_table(summary["main_effect"])
        + "\n\n## Attack stability (A3, paired first 20 questions)\n\n"
        + _markdown_table(summary["attack_stability"])
        + "\n\n## Paired route consistency\n\n"
        + ("\n".join(route_lines) if route_lines else "No complete paired attack results yet.")
        + "\n"
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Disclosure result directory")
    parser.add_argument("--output", required=True, help="Summary output directory")
    parser.add_argument("--attack-questions", type=int, default=20)
    args = parser.parse_args()
    summary = build_summary(args.input, args.attack_questions)
    json_path, md_path = write_outputs(summary, args.output)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
