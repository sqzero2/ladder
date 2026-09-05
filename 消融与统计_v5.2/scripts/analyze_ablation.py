"""Paired statistics for the frozen post-diagnosis LADDER ablation.

Uses a two-way cluster bootstrap over students and questions and exact paired
McNemar tests. This respects the crossed 30-student x 30-question design and
avoids treating 900 observations as independent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ARMS = ("B0_binary_k", "B1_prompt_only_full", "B2_k_only_tiered", "B3_full_ladder")
CONTRASTS = (
    ("graded_vs_binary_K", "B0_binary_k", "B2_k_only_tiered"),
    ("tier_prompt_vs_prompt_only", "B1_prompt_only_full", "B3_full_ladder"),
    ("six_dim_gate_vs_K_only", "B2_k_only_tiered", "B3_full_ladder"),
)
METRICS = {
    "over_disclosure": lambda r: int(r["over_disclosure"]),
    "exact_match": lambda r: int(r["disclosure_match"]),
    "under_disclosure": lambda r: int(r["under_disclosure"]),
}


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Disclosure output directory")
    parser.add_argument("--output", required=True, help="Statistics output directory")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_arms(directory: Path) -> dict[str, list[dict]]:
    output = {}
    for arm in ARMS:
        files = sorted(directory.glob(f"{arm}-*_disclosure.json"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one disclosure file for {arm}, found {files}")
        with files[0].open("r", encoding="utf-8") as handle:
            output[arm] = json.load(handle)["results"]
    return output


def record_key(record: dict) -> tuple[str, str]:
    return str(record["student_id"]), str(record["question_id"])


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def exact_mcnemar(control: list[int], treatment: list[int]) -> dict:
    b = sum(c == 1 and t == 0 for c, t in zip(control, treatment))
    c = sum(c0 == 0 and t == 1 for c0, t in zip(control, treatment))
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return {
        "discordant_control1_treatment0": b,
        "discordant_control0_treatment1": c,
        "mcnemar_exact_p": p,
        "matched_odds_ratio_haldane": (c + 0.5) / (b + 0.5),
    }


def holm_adjust(rows: list[dict]) -> None:
    for metric in METRICS:
        subset = [row for row in rows if row["metric"] == metric]
        ordered = sorted(subset, key=lambda row: row["mcnemar_exact_p"])
        running = 0.0
        m = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, (m - rank) * row["mcnemar_exact_p"])
            running = max(running, adjusted)
            row["holm_p"] = running


def bootstrap(
    indexed: dict[str, dict[tuple[str, str], dict]],
    students: list[str],
    questions: list[str],
    iterations: int,
    seed: int,
) -> tuple[dict, dict]:
    rng = random.Random(seed)
    arm_draws = {arm: {metric: [] for metric in METRICS} for arm in ARMS}
    diff_draws = {
        name: {metric: [] for metric in METRICS}
        for name, _, _ in CONTRASTS
    }
    for _ in range(iterations):
        sw = Counter(rng.choices(students, k=len(students)))
        qw = Counter(rng.choices(questions, k=len(questions)))
        denominator = sum(sw.values()) * sum(qw.values())
        for arm in ARMS:
            table = indexed[arm]
            for metric, getter in METRICS.items():
                total = sum(
                    getter(table[(s, q)]) * ns * nq
                    for s, ns in sw.items() for q, nq in qw.items()
                )
                arm_draws[arm][metric].append(total / denominator)
        for name, control, treatment in CONTRASTS:
            for metric in METRICS:
                t_draw = arm_draws[treatment][metric][-1]
                c_draw = arm_draws[control][metric][-1]
                diff_draws[name][metric].append(t_draw - c_draw)
    return arm_draws, diff_draws


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main():
    args = args_parser()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_arms(Path(args.input))
    indexed = {arm: {record_key(r): r for r in rows} for arm, rows in data.items()}
    keys = set(indexed[ARMS[0]])
    for arm in ARMS[1:]:
        if set(indexed[arm]) != keys:
            raise RuntimeError(f"Paired panel mismatch in {arm}")
    students = sorted({s for s, _ in keys})
    questions = sorted({q for _, q in keys})
    if len(keys) != len(students) * len(questions):
        raise RuntimeError("Panel is not a complete student x question crossing")

    arm_draws, diff_draws = bootstrap(indexed, students, questions, args.bootstrap, args.seed)
    arm_rows = []
    for arm in ARMS:
        rows = data[arm]
        n = len(rows)
        for metric, getter in METRICS.items():
            estimate = sum(getter(r) for r in rows) / n
            draws = arm_draws[arm][metric]
            arm_rows.append({
                "arm": arm,
                "metric": metric,
                "n": n,
                "count": sum(getter(r) for r in rows),
                "rate": estimate,
                "rate_pct": 100 * estimate,
                "cluster_bootstrap_ci_low": percentile(draws, 0.025),
                "cluster_bootstrap_ci_high": percentile(draws, 0.975),
                "students": len(students),
                "questions": len(questions),
            })

    pair_rows = []
    for name, control, treatment in CONTRASTS:
        for metric, getter in METRICS.items():
            control_values = [getter(indexed[control][key]) for key in sorted(keys)]
            treatment_values = [getter(indexed[treatment][key]) for key in sorted(keys)]
            c_rate = sum(control_values) / len(control_values)
            t_rate = sum(treatment_values) / len(treatment_values)
            test = exact_mcnemar(control_values, treatment_values)
            draws = diff_draws[name][metric]
            pair_rows.append({
                "contrast": name,
                "control": control,
                "treatment": treatment,
                "metric": metric,
                "control_rate": c_rate,
                "treatment_rate": t_rate,
                "risk_difference": t_rate - c_rate,
                "risk_difference_pp": 100 * (t_rate - c_rate),
                "relative_change": (t_rate - c_rate) / c_rate if c_rate else "",
                "relative_reduction": (c_rate - t_rate) / c_rate if c_rate else "",
                "cluster_bootstrap_diff_ci_low": percentile(draws, 0.025),
                "cluster_bootstrap_diff_ci_high": percentile(draws, 0.975),
                **test,
                "holm_p": "",
            })
    holm_adjust(pair_rows)

    route_rows = []
    for arm in ARMS:
        counts = Counter(int(r["routing_disclosure_level"]) for r in data[arm])
        for level in range(4):
            route_rows.append({
                "arm": arm, "level": level, "count": counts[level],
                "pct": 100 * counts[level] / len(data[arm]),
            })
    transitions = Counter(
        (indexed["B2_k_only_tiered"][key]["routing_disclosure_level"],
         indexed["B3_full_ladder"][key]["routing_disclosure_level"])
        for key in keys
    )
    transition_rows = [
        {"k_only_level": k, "six_dim_level": f, "count": transitions[(k, f)],
         "pct_all": 100 * transitions[(k, f)] / len(keys)}
        for k in range(4) for f in range(4)
    ]

    write_csv(output / "arm_metrics.csv", arm_rows)
    write_csv(output / "pairwise_effects.csv", pair_rows)
    write_csv(output / "route_distribution.csv", route_rows)
    write_csv(output / "route_transition_k_to_full.csv", transition_rows)

    summary = {
        "design": "frozen_post_diagnosis_paired",
        "panel": {"observations": len(keys), "students": len(students), "questions": len(questions)},
        "bootstrap": {"method": "two-way cluster bootstrap", "iterations": args.bootstrap, "seed": args.seed},
        "tests": "two-sided exact McNemar; Holm correction within each outcome across 3 planned contrasts",
        "arm_metrics": arm_rows,
        "pairwise_effects": pair_rows,
        "route_distribution": route_rows,
        "route_transitions": transition_rows,
    }
    with (output / "statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    metric_lookup = {(r["arm"], r["metric"]): r for r in arm_rows}
    lines = [
        "# LADDER v5.2 冻结诊断配对消融：统计结果",
        "",
        f"分析单元为 {len(keys)} 个学生—题目配对（{len(students)} 名学生 × {len(questions)} 道题）。",
        "上游清洗、知识点诊断和学生状态全部冻结；四个实验组只重新生成 Agent B 回答。",
        "95% CI 使用学生与题目双向聚类 bootstrap；显著性使用精确 McNemar 检验，",
        "并在每个指标的 3 个预注册式比较内做 Holm 校正。",
        "",
        "## 各组结果",
        "",
        "| 组别 | 过度披露 | 精确匹配 | 不足披露 |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        vals = []
        for metric in METRICS:
            row = metric_lookup[(arm, metric)]
            vals.append(
                f"{fmt_pct(row['rate'])} "
                f"[{fmt_pct(row['cluster_bootstrap_ci_low'])}, {fmt_pct(row['cluster_bootstrap_ci_high'])}]"
            )
        lines.append(f"| {arm} | {' | '.join(vals)} |")
    lines += ["", "## 计划比较", "",
              "| 比较（处理−对照） | 指标 | 差值(pp) | 95% CI(pp) | Holm p |",
              "|---|---|---:|---:|---:|"]
    for row in pair_rows:
        low = 100 * row["cluster_bootstrap_diff_ci_low"]
        high = 100 * row["cluster_bootstrap_diff_ci_high"]
        lines.append(
            f"| {row['contrast']} | {row['metric']} | {row['risk_difference_pp']:.2f} | "
            f"[{low:.2f}, {high:.2f}] | {row['holm_p']:.4g} |"
        )
    changed = sum(count for (k, f), count in transitions.items() if k != f)
    lines += [
        "",
        "## 路由变化",
        "",
        f"六维门控相对 K-only 门控改变了 {changed}/{len(keys)} 个样本的路由（{100*changed/len(keys):.2f}%）。",
        "完整 4×4 转移表见 `route_transition_k_to_full.csv`。",
        "",
        "## 解释边界",
        "",
        "该实验识别的是给定同一诊断快照后，门控/回答策略对披露行为的因果差异。",
        "它不覆盖上游知识点提取误差，也不能替代真人学习增益实验。LLM Judge 的标签仍需用",
        "`人工标注任务包`中的任务 B 做人工校准后，才能作为顶刊版本的最终结论。",
    ]
    (output / "统计分析.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"arm_metrics": arm_rows, "pairwise_effects": pair_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
