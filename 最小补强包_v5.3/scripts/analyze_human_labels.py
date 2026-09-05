"""Analyze double-coded D* and Judge-calibration annotation sheets.

Disagreements are excluded unless an adjudication CSV supplies a final label.
This script intentionally refuses to fabricate human labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


LEVELS = (0, 1, 2, 3)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-a-ann1", required=True)
    p.add_argument("--task-a-ann2", required=True)
    p.add_argument("--task-a-map", required=True)
    p.add_argument("--task-b-ann1", required=True)
    p.add_argument("--task-b-ann2", required=True)
    p.add_argument("--task-b-map", required=True)
    p.add_argument("--shape-disclosure", required=True)
    p.add_argument("--k-only-disclosure", required=True)
    p.add_argument("--ladder-disclosure", required=True)
    p.add_argument("--adjudication", default="")
    p.add_argument("--output", required=True)
    return p.parse_args()


def read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_level(value) -> int | None:
    if value is None:
        return None
    match = re.fullmatch(r"\s*[Ll]?([0-3])\s*", str(value))
    return int(match.group(1)) if match else None


def keyed(rows: list[dict], field="task_id") -> dict[str, dict]:
    result = {}
    for row in rows:
        item = str(row.get(field, "")).strip()
        if not item or item in result:
            raise ValueError(f"Missing or duplicate {field}: {item!r}")
        result[item] = row
    return result


def weighted_kappa(a: list[int], b: list[int]) -> float | None:
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = [[0] * 4 for _ in LEVELS]
    for x, y in zip(a, b): observed[x][y] += 1
    ra = [sum(row) for row in observed]
    cb = [sum(observed[i][j] for i in LEVELS) for j in LEVELS]
    denom = (len(LEVELS) - 1) ** 2
    obs_disagree = sum(((i-j)**2 / denom) * observed[i][j] for i in LEVELS for j in LEVELS) / n
    exp_disagree = sum(((i-j)**2 / denom) * ra[i] * cb[j] for i in LEVELS for j in LEVELS) / (n*n)
    return 1 - obs_disagree / exp_disagree if exp_disagree else 1.0


def confusion(truth: list[int], pred: list[int]) -> list[list[int]]:
    matrix = [[0] * 4 for _ in LEVELS]
    for t, p in zip(truth, pred): matrix[t][p] += 1
    return matrix


def macro_f1(truth: list[int], pred: list[int]) -> float:
    matrix = confusion(truth, pred)
    scores = []
    for level in LEVELS:
        tp = matrix[level][level]
        fp = sum(matrix[t][level] for t in LEVELS if t != level)
        fn = sum(matrix[level][p] for p in LEVELS if p != level)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / 4


def resolve_labels(
    ann1: dict[str, dict], ann2: dict[str, dict], label_field: str,
    adjudicated: dict[tuple[str, str], int], task_type: str,
) -> tuple[dict[str, int], dict]:
    if set(ann1) != set(ann2):
        raise ValueError(f"{task_type} annotator task IDs differ")
    a_values, b_values, final = [], [], {}
    missing, disagreements = 0, 0
    for task_id in sorted(ann1):
        a = parse_level(ann1[task_id].get(label_field))
        b = parse_level(ann2[task_id].get(label_field))
        if a is None or b is None:
            missing += 1
            continue
        a_values.append(a); b_values.append(b)
        if a == b:
            final[task_id] = a
        else:
            disagreements += 1
            resolved = adjudicated.get((task_type, task_id))
            if resolved is not None:
                final[task_id] = resolved
    return final, {
        "double_coded": len(a_values), "missing": missing,
        "disagreements": disagreements, "resolved_labels": len(final),
        "exact_agreement": sum(x == y for x, y in zip(a_values, b_values)) / len(a_values) if a_values else None,
        "quadratic_weighted_kappa": weighted_kappa(a_values, b_values),
    }


def load_disclosure(path: str) -> dict[tuple[str, str], dict]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)["results"]
    return {(str(r["student_id"]), str(r["question_id"])): r for r in rows}


def routing_metrics(ideal: list[int], route: list[int]) -> dict:
    n = len(ideal)
    return {
        "n": n,
        "exact": sum(a == b for a, b in zip(ideal, route)) / n,
        "within_one": sum(abs(a-b) <= 1 for a, b in zip(ideal, route)) / n,
        "mae": sum(abs(a-b) for a, b in zip(ideal, route)) / n,
        "over_route": sum(b > a for a, b in zip(ideal, route)) / n,
        "under_route": sum(b < a for a, b in zip(ideal, route)) / n,
        "weighted_kappa": weighted_kappa(ideal, route),
        "confusion_truth_rows_pred_cols": confusion(ideal, route),
    }


def main():
    args = parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    adjudicated = {}
    if args.adjudication:
        for row in read_csv(args.adjudication):
            label = parse_level(row.get("adjudicated_label"))
            task_type = str(row.get("task_type_A_or_B", "")).strip().upper()
            task_id = str(row.get("task_id", "")).strip()
            if label is not None and task_type in {"A", "B"} and task_id:
                adjudicated[(task_type, task_id)] = label

    a1, a2 = keyed(read_csv(args.task_a_ann1)), keyed(read_csv(args.task_a_ann2))
    b1, b2 = keyed(read_csv(args.task_b_ann1)), keyed(read_csv(args.task_b_ann2))
    a_final, a_agreement = resolve_labels(a1, a2, "ideal_level_L0_L3", adjudicated, "A")
    b_final, b_agreement = resolve_labels(b1, b2, "actual_level_L0_L3", adjudicated, "B")
    a_map, b_map = keyed(read_csv(args.task_a_map)), keyed(read_csv(args.task_b_map))

    shape = load_disclosure(args.shape_disclosure)
    k_only = load_disclosure(args.k_only_disclosure)
    ladder = load_disclosure(args.ladder_disclosure)
    systems = {"SHaPE": shape, "K_only_graded": k_only, "LADDER": ladder}
    routing = {}
    for name, table in systems.items():
        ideal, predicted = [], []
        for task_id, truth in a_final.items():
            row = a_map[task_id]
            item_key = (str(row["student_id"]), str(row["question_id"]))
            if item_key not in table: continue
            ideal.append(truth)
            predicted.append(int(table[item_key]["routing_disclosure_level"]))
        if ideal:
            routing[name] = routing_metrics(ideal, predicted)

    human_actual, judge_actual = [], []
    for task_id, truth in b_final.items():
        judge = parse_level(b_map[task_id].get("judge_actual_level"))
        if judge is not None:
            human_actual.append(truth); judge_actual.append(judge)
    calibration = {
        "n": len(human_actual),
        "exact_accuracy": sum(a == b for a, b in zip(human_actual, judge_actual)) / len(human_actual) if human_actual else None,
        "macro_f1": macro_f1(human_actual, judge_actual) if human_actual else None,
        "weighted_kappa": weighted_kappa(human_actual, judge_actual),
        "confusion_human_rows_judge_cols": confusion(human_actual, judge_actual) if human_actual else None,
    }
    payload = {
        "task_a_annotator_agreement": a_agreement,
        "task_b_annotator_agreement": b_agreement,
        "routing_vs_expert_D_star": routing,
        "judge_vs_human": calibration,
        "exclusion_rule": "Unresolved annotator disagreements and missing labels are excluded.",
    }
    (output / "human_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# LADDER 人工验证结果",
        "",
        f"任务 A 双标={a_agreement['double_coded']}，分歧={a_agreement['disagreements']}，最终可用={a_agreement['resolved_labels']}。",
        f"任务 B 双标={b_agreement['double_coded']}，分歧={b_agreement['disagreements']}，最终可用={b_agreement['resolved_labels']}。",
        "",
        "## 路由相对专家 D*",
        "",
        "| 系统 | n | exact | ±1 | MAE | 过度路由 | 不足路由 | 加权κ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in routing.items():
        lines.append(f"| {name} | {m['n']} | {m['exact']:.3f} | {m['within_one']:.3f} | {m['mae']:.3f} | {m['over_route']:.3f} | {m['under_route']:.3f} | {m['weighted_kappa']:.3f} |")
    lines += ["", "## Judge 校准", ""]
    if human_actual:
        lines.append(f"n={calibration['n']}，accuracy={calibration['exact_accuracy']:.3f}，macro-F1={calibration['macro_f1']:.3f}，加权κ={calibration['weighted_kappa']:.3f}。")
    else:
        lines.append("尚无可用人工标签。")
    (output / "人工验证统计.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
