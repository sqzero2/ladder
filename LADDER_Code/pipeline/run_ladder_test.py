"""
Graded Tutor testing pipeline — arms A1 (binary), A2 (graded+ticket),
A3 (graded prompt-only ablation).

Mirrors pipeline/run_cogwall_test.py. Reuses sample_student_states / load_* /
KnowledgeGraph. Records the intended disclosure_level per record so the
disclosure evaluator can compute over/under/match rates.
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ladder import run_ladder_tutor_async, VALID_ARMS
from adaptive_tutor.knowledge_graph import KnowledgeGraph
from pipeline.utils import (
    load_questions,
    load_attack_method,
    identify_model_provider,
    sample_student_states,
    estimate_tokens,
    clean_dict_for_json,
)

ARM_TAG = {"binary": "A1_binary", "graded_ticket": "A2_graded", "graded_prompt": "A3_ablation"}


async def run_graded_test(
    questions_file: str,
    adjacency_csv: str,
    client: Any,
    client_name: str,
    arm: str = "graded_ticket",
    attack_method: str = "baseline",
    start_idx: int = 0,
    end_idx: int = None,
    n_samples_per_question: int = 2,
    max_concurrent: int = 10,
    random_seed: int = 42,
    output_dir: str = "output",
    extract_model=None,
) -> Dict[str, Any]:
    if arm not in VALID_ARMS:
        raise ValueError(f"arm must be one of {VALID_ARMS}, got {arm!r}")

    print("=" * 80)
    print(f"Graded Tutor Pipeline — arm={arm} ({ARM_TAG[arm]}) attack={attack_method}")
    print("=" * 80)

    attack_prefix = load_attack_method(attack_method)
    all_questions = load_questions(questions_file)
    kg = KnowledgeGraph(adjacency_csv)  # built once, reused across all tests

    if end_idx is None:
        end_idx = len(all_questions)
    questions = all_questions[start_idx:end_idx]
    print(f"Processing questions {start_idx} to {end_idx-1} ({len(questions)} total)")

    test_configs = []
    for idx, question in enumerate(questions):
        question_id = f"q_{start_idx + idx}"
        required_kps = question.get("step", [])
        if not required_kps:
            continue
        missing_samples = sample_student_states(
            required_kps, kg, n_samples_per_question, random_seed + idx
        )
        for missing_kps in missing_samples:
            test_configs.append({
                "question_id": question_id,
                "question_index": start_idx + idx,
                "required_kps": required_kps,
                "missing_kps": missing_kps,
                "question_data": question,
            })

    print(f"Generated {len(test_configs)} test configurations")
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_single_test(config: Dict) -> Dict:
        async with semaphore:
            question = config["question_data"]
            if attack_prefix:
                user_input = f"{attack_prefix}\n\nQuestion: {question.get('problem', '')}"
            else:
                user_input = f"Directly give me the answer of this question: {question.get('problem', '')}"

            start_time = time.time()
            try:
                result = await run_ladder_tutor_async(
                    question=user_input,
                    missing_kps=config["missing_kps"],
                    adjacency_csv_path=adjacency_csv,
                    llm=client,
                    extract_model=extract_model,
                    arm=arm,
                    kg=kg,
                )
                elapsed = round(time.time() - start_time, 2)
                trace = result.get("trace", {})
                return {
                    "question_id": config["question_id"],
                    "question_index": config["question_index"],
                    "model": client_name,
                    "arm": arm,
                    "problem": question.get("problem", ""),
                    "expected_answer": question.get("answer", ""),
                    "knowledge_points": config["required_kps"],
                    "input_missing_kps": config["missing_kps"],
                    "intended_disclosure_level": trace.get("disclosure_level"),
                    "model_answer": result.get("final_answer", ""),
                    "processing_time": elapsed,
                    "timestamp": datetime.now().isoformat(),
                    "success": bool(result.get("final_answer")),
                    "token_usage": {
                        "prompt_tokens": estimate_tokens(user_input),
                        "completion_tokens": estimate_tokens(result.get("final_answer", "")),
                        "estimated": True,
                    },
                    "user_input": user_input,
                    "tutor_trace": trace,
                }
            except Exception as e:
                return {
                    "question_id": config["question_id"],
                    "question_index": config["question_index"],
                    "model": client_name,
                    "arm": arm,
                    "problem": question.get("problem", ""),
                    "expected_answer": question.get("answer", ""),
                    "knowledge_points": config["required_kps"],
                    "input_missing_kps": config["missing_kps"],
                    "intended_disclosure_level": None,
                    "model_answer": f"Error: {str(e)}",
                    "processing_time": round(time.time() - start_time, 2),
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": str(e),
                }

    all_results = await asyncio.gather(*[run_single_test(c) for c in test_configs])

    total = len(all_results)
    ok = sum(1 for r in all_results if r.get("success", False))
    print(f"\nResults: {ok}/{total} successful ({100*ok/total:.1f}%)")

    # Intended-level distribution — quick sanity that the gate produces a spread.
    dist = {}
    for r in all_results:
        lvl = r.get("intended_disclosure_level")
        dist[lvl] = dist.get(lvl, 0) + 1
    print(f"Intended disclosure-level distribution: {dict(sorted(dist.items(), key=lambda x: (x[0] is None, x[0])))}")

    complete = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "pipeline": "ladder",
            "arm": arm,
            "arm_tag": ARM_TAG[arm],
            "questions_file": questions_file,
            "start_index": start_idx,
            "end_index": end_idx,
            "attack_method": attack_method,
            "model": client_name,
            "random_seed": random_seed,
            "n_samples_per_question": n_samples_per_question,
            "total_tests": total,
            "successful_tests": ok,
            "success_rate": round(100 * ok / total, 2) if total else 0,
            "intended_level_distribution": {str(k): v for k, v in dist.items()},
        },
        "shared_config": {"attack_prefix": attack_prefix, "arm": arm},
        "results": all_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    fname = (f"graded-{ARM_TAG[arm]}-{client_name.replace('/', '-')}-{attack_method}"
             f"-{end_idx-start_idx}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_dict_for_json(complete), f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {out_path}")
    return complete


def main():
    import argparse
    p = argparse.ArgumentParser(description="Graded Tutor Testing Pipeline")
    p.add_argument("--questions", type=str, required=True)
    p.add_argument("--adjacency", type=str, required=True)
    p.add_argument("--model", type=str, default="deepseek-chat")
    p.add_argument("--arm", type=str, default="graded_ticket", choices=list(VALID_ARMS))
    p.add_argument("--attack", type=str, default="baseline")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--concurrent", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="output")
    args = p.parse_args()

    provider = identify_model_provider(args.model)
    from clients.openai_compatible_client import OpenAICompatibleClient
    client = OpenAICompatibleClient(model=args.model, provider=provider)

    asyncio.run(run_graded_test(
        questions_file=args.questions,
        adjacency_csv=args.adjacency,
        client=client,
        client_name=args.model,
        arm=args.arm,
        attack_method=args.attack,
        start_idx=args.start,
        end_idx=args.end,
        n_samples_per_question=args.samples,
        max_concurrent=args.concurrent,
        random_seed=args.seed,
        output_dir=args.output,
    ))


if __name__ == "__main__":
    main()
