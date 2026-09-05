"""Create blinded, deterministic annotation sheets from the formal v5.1 run."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--disclosure-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--task-a-per-level", type=int, default=150)
    parser.add_argument("--task-b-per-arm", type=int, default=200)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_one(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if len(files) != 1:
        raise RuntimeError(f"Expected one file for {pattern}, found {len(files)}: {files}")
    return files[0]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def balanced_sample(records: list[dict], total: int, rng: random.Random) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[int(record.get("routing_disclosure_level", 3))].append(record)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    levels = sorted(groups)
    quota = total // len(levels)
    for level in levels:
        selected.extend(groups[level][: min(quota, len(groups[level]))])
    selected_ids = {id(item) for item in selected}
    remainder = [item for level in levels for item in groups[level] if id(item) not in selected_ids]
    rng.shuffle(remainder)
    selected.extend(remainder[: total - len(selected)])
    if len(selected) != total:
        raise RuntimeError(f"Could select only {len(selected)} of requested {total}")
    return selected


def task_a(a3_records: list[dict], per_level: int, rng: random.Random):
    selected = []
    for level in range(4):
        candidates = [r for r in a3_records if int(r["routing_disclosure_level"]) == level]
        if len(candidates) < per_level:
            raise RuntimeError(f"L{level} has {len(candidates)} records, need {per_level}")
        selected.extend(rng.sample(candidates, per_level))
    rng.shuffle(selected)
    public, mapping = [], []
    for idx, record in enumerate(selected, 1):
        task_id = f"IDEAL-{idx:04d}"
        state = record.get("student_state") or {}
        required = record.get("diagnosed_required_kps") or []
        mastery = {kp: state.get("kp_mastery", {}).get(kp) for kp in required}
        profile = {
            key: state.get(key)
            for key in (
                "bloom_level", "dominant_error_type", "learning_speed", "help_seeking",
                "preference", "anxiety", "self_efficacy", "motivation", "trajectory",
                "behavioral_state",
            )
        }
        public.append({
            "task_id": task_id,
            "problem": record.get("problem", ""),
            "reference_steps_json": json.dumps(record.get("reference_steps") or [], ensure_ascii=False),
            "required_kps_json": json.dumps(required, ensure_ascii=False),
            "required_kp_mastery_json": json.dumps(mastery, ensure_ascii=False),
            "student_profile_json": json.dumps(profile, ensure_ascii=False),
            "ideal_level_L0_L3": "",
            "min_acceptable_level_L0_L3": "",
            "max_acceptable_level_L0_L3": "",
            "primary_reason_code": "",
            "confidence_1_5": "",
            "annotator_id": "",
            "annotation_seconds": "",
            "notes": "",
        })
        mapping.append({
            "task_id": task_id,
            "source_arm": record.get("arm"),
            "student_id": record.get("student_id"),
            "question_id": record.get("question_id"),
            "system_route": record.get("routing_disclosure_level"),
            "judge_actual_level": record.get("actual_disclosure_level", ""),
            "judge_match": record.get("disclosure_match", ""),
        })
    return public, mapping


def task_b(records_by_arm: dict[str, list[dict]], per_arm: int, rng: random.Random):
    selected = []
    for arm, records in records_by_arm.items():
        selected.extend((arm, r) for r in balanced_sample(records, per_arm, rng))
    rng.shuffle(selected)
    public, mapping = [], []
    for idx, (arm, record) in enumerate(selected, 1):
        task_id = f"ANSWER-{idx:04d}"
        public.append({
            "task_id": task_id,
            "problem": record.get("problem", ""),
            "expected_answer": record.get("expected_answer", ""),
            "reference_steps_json": json.dumps(record.get("reference_steps") or [], ensure_ascii=False),
            "model_response": record.get("model_answer", ""),
            "actual_level_L0_L3": "",
            "pedagogical_quality_1_5": "",
            "contains_final_answer_yes_no": "",
            "primary_reason_code": "",
            "confidence_1_5": "",
            "annotator_id": "",
            "annotation_seconds": "",
            "notes": "",
        })
        mapping.append({
            "task_id": task_id,
            "source_arm": arm,
            "student_id": record.get("student_id"),
            "question_id": record.get("question_id"),
            "system_route": record.get("routing_disclosure_level"),
            "judge_actual_level": record.get("actual_disclosure_level", ""),
            "judge_delta": record.get("disclosure_delta", ""),
            "judge_raw_output": record.get("judge_raw_output", ""),
        })
    return public, mapping


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    disclosure_dir = Path(args.disclosure_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    a3_file = find_one(disclosure_dir, "A3-*-baseline-30s-30q-*_disclosure.json")
    a3_records = load_json(a3_file)["results"]
    a_public, a_mapping = task_a(a3_records, args.task_a_per_level, rng)

    records_by_arm = {}
    source_files = {}
    for arm in ("E0", "E1", "A3"):
        path = find_one(disclosure_dir, f"{arm}-*-baseline-30s-30q-*_disclosure.json")
        records_by_arm[arm] = load_json(path)["results"]
        source_files[arm] = str(path.resolve())
    b_public, b_mapping = task_b(records_by_arm, args.task_b_per_arm, rng)

    write_csv(output / "任务A_专家理想披露等级_双人独立标注.csv", a_public, list(a_public[0]))
    write_csv(output / "任务A_抽样映射_勿给标注员.csv", a_mapping, list(a_mapping[0]))
    write_csv(output / "任务B_回答实际披露等级_双人独立标注.csv", b_public, list(b_public[0]))
    write_csv(output / "任务B_抽样映射_勿给标注员.csv", b_mapping, list(b_mapping[0]))
    adjudication = [{
        "task_type_A_or_B": "", "task_id": "", "annotator_1_label": "",
        "annotator_2_label": "", "adjudicated_label": "", "adjudicator_id": "",
        "decision_reason": "", "adjudication_seconds": "",
    }]
    write_csv(output / "裁决记录模板.csv", adjudication, list(adjudication[0]))
    manifest = {
        "seed": args.seed,
        "task_a": {"n": len(a_public), "per_system_route": args.task_a_per_level, "source": str(a3_file.resolve())},
        "task_b": {"n": len(b_public), "per_arm": args.task_b_per_arm, "sources": source_files},
        "blinding": "Public sheets contain neither model arm, system route, nor LLM judge label.",
    }
    with (output / "抽样清单.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
