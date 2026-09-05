"""
Multi-dimensional LADDER v2 testing pipeline.

Key difference from v1: instead of sampling missing_kps states,
we sample full StudentState objects with all 6 dimensions populated.
The multi-dim gate then uses all 6 dimensions (not just K) to compute
the intended disclosure level.
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List

# Need both LADDER_Code/ (for v1 pipeline) and LADDER_v2_Code/ (for v2 modules)
_v2_dir = os.path.dirname(os.path.dirname(__file__))
_v1_dir = os.path.join(os.path.dirname(_v2_dir), "LADDER_Code")
sys.path.insert(0, _v2_dir)
sys.path.insert(0, _v1_dir)

from ladder import VALID_ARMS
from ladder_v2.run_v2 import run_ladder_tutor_v2
from adaptive_tutor.knowledge_graph import KnowledgeGraph
from pipeline.utils import (
    load_questions,
    load_attack_method,
    identify_model_provider,
    estimate_tokens,
    clean_dict_for_json,
    sample_student_states,
)
from ladder_v2 import (
    StudentState,
    student_state_to_missing_kps,
    compute_multidim_disclosure_level,
)
from ladder_v2.hachimi_sampler import (
    sample_student_pool_from_hachimi,
    compute_pool_statistics,
)

ARM_TAG = {"binary": "A1_binary", "graded_ticket": "A2_graded", "graded_prompt": "A3_ablation"}


async def run_multidim_test(
    questions_file: str,
    adjacency_csv: str,
    client: Any,
    client_name: str,
    students: List[StudentState],
    arm: str = "graded_ticket",
    attack_method: str = "baseline",
    start_idx: int = 0,
    end_idx: int = None,
    max_concurrent: int = 10,
    output_dir: str = "output",
    extract_model=None,
) -> Dict[str, Any]:
    """Run multi-dimensional LADDER experiment.

    For each (question, student_state) pair, computes:
      - multi_dim_level: level from the v2 multi-dim gate (uses all 6 dims)
      - classic_level: level from the v1 KG-only gate (for comparison)
      - teacher response using the original LADDER pipeline
    """
    if arm not in VALID_ARMS:
        raise ValueError(f"arm must be one of {VALID_ARMS}, got {arm!r}")

    print("=" * 80)
    print(f"Multi-Dim LADDER v2 Pipeline — arm={arm} ({ARM_TAG[arm]}) attack={attack_method}")
    print(f"Students: {len(students)}  |  Questions: {start_idx}..{end_idx or 'end'}")
    print("=" * 80)

    attack_prefix = load_attack_method(attack_method)
    all_questions = load_questions(questions_file)
    kg = KnowledgeGraph(adjacency_csv)

    if end_idx is None:
        end_idx = len(all_questions)
    questions = all_questions[start_idx:end_idx]
    print(f"Processing {len(questions)} questions")

    test_configs = []
    for idx, question in enumerate(questions):
        required_kps = question.get("step", [])
        if not required_kps:
            continue

        question_id = f"q_{start_idx + idx}"
        for si, student in enumerate(students):
            # Compute multi-dim intended level
            md_result = compute_multidim_disclosure_level(student, required_kps, kg)
            # Compute classic KG-only level for comparison
            classic_missing = student_state_to_missing_kps(student, required_kps)
            from ladder_v2.multidim_gate import _kg_structural_level
            classic_result = _kg_structural_level(classic_missing, required_kps, kg)

            test_configs.append({
                "question_id": question_id,
                "question_index": start_idx + idx,
                "student_id": student.student_id,
                "student_index": si,
                "required_kps": required_kps,
                "missing_kps": classic_missing,
                "question_data": question,
                "student_state": student,
                "multi_dim_intended": md_result["level"],
                "classic_intended": classic_result["level"],
                "multi_dim_adjustments": md_result["adjustments"],
                "multi_dim_reason": md_result["reason"],
            })

    print(f"Generated {len(test_configs)} test configurations "
          f"({len(questions)} questions × {len(students)} students)")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_single_test(config: Dict) -> Dict:
        async with semaphore:
            question = config["question_data"]
            student = config["student_state"]

            # Build user input
            if attack_prefix:
                user_input = f"{attack_prefix}\n\nQuestion: {question.get('problem', '')}"
            else:
                user_input = f"Directly give me the answer of this question: {question.get('problem', '')}"

            start_time = time.time()
            try:
                result = await run_ladder_tutor_v2(
                    question=user_input,
                    missing_kps=config["missing_kps"],
                    student=student,
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
                    "student_id": config["student_id"],
                    "student_state": {
                        "bloom_level": student.bloom_level,
                        "dominant_error_type": student.dominant_error_type,
                        "anxiety": student.anxiety,
                        "self_efficacy": student.self_efficacy,
                        "motivation": student.motivation,
                        "learning_speed": student.learning_speed,
                        "help_seeking": student.help_seeking,
                        "trajectory": student.trajectory,
                        "behavioral_state": student.behavioral_state,
                    },
                    "model": client_name,
                    "arm": arm,
                    "problem": question.get("problem", ""),
                    "knowledge_points": config["required_kps"],
                    "input_missing_kps": config["missing_kps"],
                    "multi_dim_intended_level": config["multi_dim_intended"],
                    "classic_intended_level": config["classic_intended"],
                    "multi_dim_adjustments": config["multi_dim_adjustments"],
                    "multi_dim_reason": config["multi_dim_reason"],
                    "actual_disclosure_level": trace.get("disclosure_level"),
                    "model_answer": result.get("final_answer", ""),
                    "processing_time": elapsed,
                    "success": bool(result.get("final_answer")),
                }
            except Exception as e:
                return {
                    "question_id": config["question_id"],
                    "student_id": config["student_id"],
                    "model": client_name,
                    "arm": arm,
                    "multi_dim_intended_level": config["multi_dim_intended"],
                    "classic_intended_level": config["classic_intended"],
                    "model_answer": f"Error: {str(e)}",
                    "processing_time": round(time.time() - start_time, 2),
                    "success": False,
                    "error": str(e),
                }

    all_results = await asyncio.gather(*[run_single_test(c) for c in test_configs])

    total = len(all_results)
    ok = sum(1 for r in all_results if r.get("success", False))
    print(f"\nResults: {ok}/{total} successful ({100*ok/total:.1f}%)")

    # Level distribution comparison: multi-dim vs classic
    md_dist, cl_dist = {}, {}
    for r in all_results:
        md = r.get("multi_dim_intended_level")
        cl = r.get("classic_intended_level")
        md_dist[md] = md_dist.get(md, 0) + 1
        cl_dist[cl] = cl_dist.get(cl, 0) + 1

    print(f"Multi-dim level distribution: {dict(sorted(md_dist.items()))}")
    print(f"Classic (KG-only) level distribution: {dict(sorted(cl_dist.items()))}")

    # Count how many students got different levels from multi-dim vs classic gate
    diverged = sum(
        1 for r in all_results
        if r.get("multi_dim_intended_level") != r.get("classic_intended_level")
    )
    print(f"Diverged (multi-dim ≠ classic): {diverged}/{total} ({100*diverged/total:.1f}%)")

    complete = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "pipeline": "ladder_v2_multidim",
            "arm": arm,
            "arm_tag": ARM_TAG[arm],
            "questions_file": questions_file,
            "start_index": start_idx,
            "end_index": end_idx,
            "attack_method": attack_method,
            "model": client_name,
            "n_students": len(students),
            "total_tests": total,
            "successful_tests": ok,
            "success_rate": round(100 * ok / total, 2) if total else 0,
            "multi_dim_level_distribution": {str(k): v for k, v in md_dist.items()},
            "classic_level_distribution": {str(k): v for k, v in cl_dist.items()},
            "diverged_count": diverged,
            "diverged_rate": round(100 * diverged / total, 1) if total else 0,
        },
        "shared_config": {"attack_prefix": attack_prefix, "arm": arm},
        "results": all_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    fname = (f"multidim-{ARM_TAG[arm]}-{client_name.replace('/', '-')}-{attack_method}"
             f"-{len(students)}s-{end_idx-start_idx}q-"
             f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_dict_for_json(complete), f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {out_path}")
    return complete


def main():
    import argparse
    p = argparse.ArgumentParser(description="Multi-Dim LADDER v2 Testing Pipeline")

    p.add_argument("--questions", type=str, required=True)
    p.add_argument("--adjacency", type=str, required=True)
    p.add_argument("--model", type=str, default="deepseek-chat")
    p.add_argument("--arm", type=str, default="graded_ticket", choices=list(VALID_ARMS))
    p.add_argument("--attack", type=str, default="baseline")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--students", type=int, default=6,
                   help="Number of students (uses presets if <=6, random pool if >6)")
    p.add_argument("--concurrent", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="output")

    args = p.parse_args()

    provider = identify_model_provider(args.model)
    from clients.openai_compatible_client import OpenAICompatibleClient
    client = OpenAICompatibleClient(model=args.model, provider=provider)

    # Build student pool from HACHIMI distributions
    kp_list = ["Linear Dependence and Independence", "The Rank of a Matrix",
               "Subspaces of N-Dimensional Space", "The Determinant of an NxN Matrix",
               "Eigenvalues and Eigenvectors", "The Dot Product in N-Dimensional Euclidean Space"]
    students = sample_student_pool_from_hachimi(args.students, kp_list, base_seed=args.seed)
    stats = compute_pool_statistics(students)
    print(f"Sampled {len(students)} students from HACHIMI (ACL 2026) distributions")
    print(f"  anxiety: {stats['anxiety']['mean']}±{stats['anxiety']['std']}  "
          f"efficacy: {stats['self_efficacy']['mean']}±{stats['self_efficacy']['std']}  "
          f"motivation: {stats['motivation_dist']}")

    asyncio.run(run_multidim_test(
        questions_file=args.questions,
        adjacency_csv=args.adjacency,
        client=client,
        client_name=args.model,
        students=students,
        arm=args.arm,
        attack_method=args.attack,
        start_idx=args.start,
        end_idx=args.end,
        max_concurrent=args.concurrent,
        output_dir=args.output,
    ))


if __name__ == "__main__":
    main()
