"""Crossed-panel paired statistics for the formal SHaPE vs LADDER comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path


METRICS = {
    "over_disclosure": lambda r: int(r["over_disclosure"]),
    "exact_match": lambda r: int(r["disclosure_match"]),
    "under_disclosure": lambda r: int(r["under_disclosure"]),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True)
    p.add_argument("--ladder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["results"]


def key(r: dict) -> tuple[str, str]:
    return str(r["student_id"]), str(r["question_id"])


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
        tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {
        "shape1_ladder0": b,
        "shape0_ladder1": c,
        "mcnemar_exact_p": p,
        "matched_odds_ratio_haldane": (c + 0.5) / (b + 0.5),
    }


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shape = {key(r): r for r in load(args.shape)}
    ladder = {key(r): r for r in load(args.ladder)}
    if set(shape) != set(ladder):
        raise RuntimeError("SHaPE and LADDER paired keys differ")
    keys = sorted(shape)
    students = sorted({s for s, _ in keys})
    questions = sorted({q for _, q in keys})
    if len(keys) != len(students) * len(questions):
        raise RuntimeError("Expected a complete crossed student x question panel")

    rng = random.Random(args.seed)
    draws = {metric: {"shape": [], "ladder": [], "difference": []} for metric in METRICS}
    for _ in range(args.bootstrap):
        sw = Counter(rng.choices(students, k=len(students)))
        qw = Counter(rng.choices(questions, k=len(questions)))
        denom = sum(sw.values()) * sum(qw.values())
        for metric, getter in METRICS.items():
            s_rate = sum(getter(shape[(s, q)]) * ns * nq for s, ns in sw.items() for q, nq in qw.items()) / denom
            l_rate = sum(getter(ladder[(s, q)]) * ns * nq for s, ns in sw.items() for q, nq in qw.items()) / denom
            draws[metric]["shape"].append(s_rate)
            draws[metric]["ladder"].append(l_rate)
            draws[metric]["difference"].append(l_rate - s_rate)

    rows = []
    for metric, getter in METRICS.items():
        s_values = [getter(shape[k]) for k in keys]
        l_values = [getter(ladder[k]) for k in keys]
        s_rate = sum(s_values) / len(keys)
        l_rate = sum(l_values) / len(keys)
        diff = l_rate - s_rate
        test = exact_mcnemar(s_values, l_values)
        rows.append({
            "metric": metric,
            "n_pairs": len(keys),
            "shape_rate": s_rate,
            "shape_ci_low": percentile(draws[metric]["shape"], 0.025),
            "shape_ci_high": percentile(draws[metric]["shape"], 0.975),
            "ladder_rate": l_rate,
            "ladder_ci_low": percentile(draws[metric]["ladder"], 0.025),
            "ladder_ci_high": percentile(draws[metric]["ladder"], 0.975),
            "risk_difference": diff,
            "risk_difference_pp": 100 * diff,
            "difference_ci_low": percentile(draws[metric]["difference"], 0.025),
            "difference_ci_high": percentile(draws[metric]["difference"], 0.975),
            "relative_change": diff / s_rate if s_rate else "",
            **test,
        })
    write_csv(output / "shape_vs_ladder_paired.csv", rows)
    payload = {
        "design": "formal_v5.1_complete_crossed_panel",
        "panel": {"pairs": len(keys), "students": len(students), "questions": len(questions)},
        "bootstrap": {"method": "two-way cluster bootstrap", "iterations": args.bootstrap, "seed": args.seed},
        "results": rows,
        "interpretation": "System-level comparison; does not isolate a single LADDER component.",
    }
    (output / "shape_vs_ladder_paired.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def pct(x): return f"{100*x:.2f}%"
    lines = [
        "# SHaPE vs LADDER：正式配对统计",
        "",
        f"样本为 {len(keys)} 个完整配对（{len(students)} 名学生 × {len(questions)} 道题）。",
        f"置信区间采用学生与题目双向聚类 bootstrap（{args.bootstrap} 次）。",
        "McNemar p 值作为逐样本配对敏感性分析；主推断优先看双向聚类 CI。",
        "",
        "| 指标 | SHaPE | LADDER | 差值（L−S） | 95% CI | 精确 McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['metric']} | {pct(r['shape_rate'])} | {pct(r['ladder_rate'])} | "
            f"{r['risk_difference_pp']:.2f}pp | "
            f"[{100*r['difference_ci_low']:.2f}, {100*r['difference_ci_high']:.2f}]pp | "
            f"{r['mcnemar_exact_p']:.4g} |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "该比较衡量整套 LADDER 相对 SHaPE 的端到端差异。由于两者同时改变提取隔离、",
        "门控位置、披露粒度和状态维度，结果不能全部归因于某一个单独组件。",
        "E0 永远路由到 L3，其相对自身目标的 0% 过度披露是定义结果，不纳入主比较。",
    ]
    (output / "SHaPE_vs_LADDER_配对统计.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
