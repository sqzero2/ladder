"""Run the frozen post-diagnosis LADDER ablation.

The source A3 run supplies a fixed student-question panel, extracted question,
diagnosed knowledge gaps, student state, and the K-only/full gate decisions.
This script regenerates only Agent B's answer, which makes every contrast
paired and prevents upstream diagnosis noise from contaminating the ablation.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
CODE_DIR = PACKAGE_DIR.parent
V2_DIR = CODE_DIR / "LADDER_v2_Code"
V1_DIR = CODE_DIR / "LADDER_Code"
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(V1_DIR))

from clients.openai_compatible_client import OpenAICompatibleClient
from context_based_tutor.prompts import DIRECT_ANSWER_PROMPT, build_tutoring_prompt
from ladder.prompts import build_ablation_prompt, build_hint_prompt, build_partial_prompt
from ladder_v2.interact_prompts import build_affective_agent_prompt
from ladder_v2.student_model import StudentState


ARM_DESCRIPTIONS = {
    "B0_binary_k": "K-only binary gate (all gaps -> L0; no gaps -> L3) + tier prompt",
    "B1_prompt_only_full": "six-dimension gate + one combined prompt-only policy",
    "B2_k_only_tiered": "K-only four-level gate + tier-specific prompt",
    "B3_full_ladder": "six-dimension four-level gate + tier-specific prompt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Formal A3 baseline JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--arms", default=",".join(ARM_DESCRIPTIONS))
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def k_level(record: dict) -> int:
    """Return the classic K-only graded level frozen in A3 gate metadata."""
    gate = record.get("gate_info") or {}
    if "base_level" not in gate:
        raise KeyError("gate_info.base_level is required for the controlled ablation")
    return int(gate["base_level"])


def arm_level(record: dict, arm: str) -> int:
    base = k_level(record)
    full = int(record["routing_disclosure_level"])
    if arm == "B0_binary_k":
        return 3 if base == 3 else 0
    if arm == "B2_k_only_tiered":
        return base
    if arm in {"B1_prompt_only_full", "B3_full_ladder"}:
        return full
    raise ValueError(f"Unknown arm: {arm}")


def tier_prompt(level: int, missing_text: str) -> str:
    if level == 0:
        return build_tutoring_prompt(missing_text)
    if level == 1:
        return build_hint_prompt(missing_text)
    if level == 2:
        return build_partial_prompt(missing_text)
    if level == 3:
        return DIRECT_ANSWER_PROMPT
    raise ValueError(f"Invalid level: {level}")


def build_messages(record: dict, arm: str) -> tuple[list[dict], int]:
    level = arm_level(record, arm)
    missing = list(record.get("diagnosed_missing_kps") or record.get("input_missing_kps") or [])
    missing_text = ", ".join(missing) if missing else "None (all required concepts mastered)"
    state = StudentState(**(record.get("student_state") or {}))
    if arm == "B1_prompt_only_full":
        base_prompt = build_ablation_prompt(level, missing_text)
    else:
        base_prompt = tier_prompt(level, missing_text)
    system_prompt = build_affective_agent_prompt(base_prompt, state, missing)

    steps = record.get("reference_steps") or []
    step_context = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))
    user_prompt = (
        "Answer the following clean student problem under the system policy.\n\n"
        f"Problem:\n{record.get('clean_question') or record.get('problem', '')}\n\n"
        "Reference solution context (use only to formulate the authorized response; "
        "do not exceed the assigned level):\n"
        f"{step_context}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ], level


async def generate_one(
    client: OpenAICompatibleClient,
    semaphore: asyncio.Semaphore,
    record: dict,
    arm: str,
    retries: int,
) -> dict:
    messages, intended = build_messages(record, arm)
    started = time.perf_counter()
    last_error = None
    answer = ""
    attempts = 0
    async with semaphore:
        for attempts in range(1, retries + 1):
            try:
                answer = await asyncio.to_thread(client.chat_messages, messages)
                if not answer.strip():
                    raise ValueError("empty model response")
                last_error = None
                break
            except Exception as exc:  # API failures are retained in the raw record
                last_error = str(exc)
                if attempts < retries:
                    await asyncio.sleep(min(2 ** (attempts - 1), 4))

    out = copy.deepcopy(record)
    out.update(
        {
            "arm": arm,
            "source_arm": record.get("arm"),
            "model_answer": answer,
            "success": last_error is None,
            "routing_disclosure_level": intended,
            "intended_disclosure_level": intended,
            "multi_dim_intended_level": intended,
            "source_k_only_level": k_level(record),
            "source_full_level": int(record["routing_disclosure_level"]),
            "ablation_design": "frozen_post_diagnosis_paired",
            "prompt_strategy": "combined_prompt_only" if arm == "B1_prompt_only_full" else "tier_specific",
            "gate_strategy": (
                "k_only_binary" if arm == "B0_binary_k" else
                "k_only_graded" if arm == "B2_k_only_tiered" else
                "six_dimension_graded"
            ),
            "attempts": attempts,
            "processing_time": round(time.perf_counter() - started, 3),
            "generation_error": last_error,
        }
    )
    for key in (
        "actual_disclosure_level", "judge_raw_output", "judge_error",
        "judge_attempts", "disclosure_delta", "over_disclosure",
        "under_disclosure", "disclosure_match",
    ):
        out.pop(key, None)
    return out


async def run_arm(
    client: OpenAICompatibleClient,
    records: list[dict],
    arm: str,
    args: argparse.Namespace,
) -> list[dict]:
    semaphore = asyncio.Semaphore(args.concurrent)
    tasks = [generate_one(client, semaphore, rec, arm, args.retries) for rec in records]
    output = []
    for idx, future in enumerate(asyncio.as_completed(tasks), 1):
        output.append(await future)
        if idx % 50 == 0 or idx == len(tasks):
            print(f"[{arm}] {idx}/{len(tasks)}", flush=True)
    output.sort(key=lambda r: (str(r.get("student_id")), str(r.get("question_id"))))
    return output


async def main_async(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    with open(args.input, "r", encoding="utf-8") as handle:
        source = json.load(handle)
    records = [r for r in source.get("results", []) if r.get("success", True)]
    records.sort(key=lambda r: (str(r.get("student_id")), str(r.get("question_id"))))
    if args.limit > 0:
        records = records[: args.limit]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = sorted(set(arms) - set(ARM_DESCRIPTIONS))
    if unknown:
        raise ValueError(f"Unknown arm(s): {unknown}")

    client = OpenAICompatibleClient(
        provider=args.provider,
        model=args.model,
        temperature=0.0,
        max_completion_tokens=1800,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        print(f"\nStarting {arm}: {ARM_DESCRIPTIONS[arm]}", flush=True)
        generated = await run_arm(client, records, arm, args)
        payload = {
            "metadata": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "design": "frozen_post_diagnosis_paired",
                "source_file": str(Path(args.input).resolve()),
                "source_n": len(records),
                "arm": arm,
                "arm_description": ARM_DESCRIPTIONS[arm],
                "model": args.model,
                "provider": args.provider,
                "temperature": 0.0,
                "seed": args.seed,
                "reference_context": "dataset reference_steps, held identical across arms",
            },
            "results": generated,
        }
        output_path = output_dir / f"{arm}-{args.model}-{len(records)}cases.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        failures = sum(not r["success"] for r in generated)
        print(f"Saved {output_path} (failures={failures})", flush=True)


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
