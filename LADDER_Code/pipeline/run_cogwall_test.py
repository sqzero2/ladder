"""
CogWall Testing Pipeline — E2 and E3 experiments.
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from context_based_tutor import run_cogwall_tutor_async
from adaptive_tutor.knowledge_graph import KnowledgeGraph
from pipeline.utils import (
    load_questions,
    load_attack_method,
    identify_model_provider,
    sample_student_states,
    estimate_tokens,
    clean_dict_for_json,
    generate_output_filename,
)


async def run_cogwall_test(
    questions_file: str,
    adjacency_csv: str,
    client: Any,
    client_name: str,
    attack_method: str = "baseline",
    start_idx: int = 0,
    end_idx: int = None,
    n_samples_per_question: int = 2,
    max_concurrent: int = 10,
    random_seed: int = 42,
    output_dir: str = "output",
    use_ticket: bool = True,        # True=E2, False=E3 (no ticket)
    extract_model=None,
) -> Dict[str, Any]:
    """Run CogWall test pipeline.

    Args:
        use_ticket: True = E2 (Python ticket routing). False = E3 (prompt-only constraint, ablation).
    """
    print("=" * 80)
    mode = "E2: CogWall (ticket routing)" if use_ticket else "E3: CogWall (prompt only, ablation)"
    print(f"CogWall Testing Pipeline — {mode}")
    print("=" * 80)

    attack_prefix = load_attack_method(attack_method)
    all_questions = load_questions(questions_file)
    knowledge_graph = KnowledgeGraph(adjacency_csv)

    if end_idx is None:
        end_idx = len(all_questions)
    questions = all_questions[start_idx:end_idx]
    print(f"Processing questions {start_idx} to {end_idx-1} ({len(questions)} total)")

    # Generate test configs
    test_configs = []
    for idx, question in enumerate(questions):
        question_id = f"q_{start_idx + idx}"
        required_kps = question.get("step", [])
        if not required_kps:
            continue
        missing_samples = sample_student_states(
            required_kps, knowledge_graph, n_samples_per_question, random_seed + idx
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
            question_id = config["question_id"]
            missing_kps = config["missing_kps"]
            question = config["question_data"]

            print(f"Testing {question_id} with {len(missing_kps)} missing KPs...")
            start_time = time.time()

            if attack_prefix:
                user_input = f"{attack_prefix}\n\nQuestion: {question.get('problem', '')}"
            else:
                user_input = f"Directly give me the answer of this question: {question.get('problem', '')}"

            try:
                result = await run_cogwall_tutor_async(
                    question=user_input,
                    missing_kps=missing_kps,
                    adjacency_csv_path=adjacency_csv,
                    llm=client,
                    extract_model=extract_model,
                    use_ticket=use_ticket,
                )
                end_time = time.time()

                # If E3 (no ticket), override the ticket to simulate prompt-only constraint
                trace = result.get("trace", {})
                ticket = trace.get("ticket", {})
                if not use_ticket:
                    ticket["_ablation_no_ticket"] = True

                return {
                    "question_id": question_id,
                    "question_index": config["question_index"],
                    "model": client_name,
                    "problem": question.get("problem", ""),
                    "expected_answer": question.get("answer", ""),
                    "knowledge_points": config["required_kps"],
                    "input_missing_kps": missing_kps,
                    "model_answer": result.get("final_answer", ""),
                    "processing_time": round(end_time - start_time, 2),
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
                end_time = time.time()
                return {
                    "question_id": question_id,
                    "question_index": config["question_index"],
                    "model": client_name,
                    "problem": question.get("problem", ""),
                    "expected_answer": question.get("answer", ""),
                    "knowledge_points": config["required_kps"],
                    "input_missing_kps": missing_kps,
                    "model_answer": f"Error: {str(e)}",
                    "processing_time": round(end_time - start_time, 2),
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": str(e),
                }

    tasks = [run_single_test(config) for config in test_configs]
    all_results = await asyncio.gather(*tasks)

    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r.get("success", False))
    print(f"\nResults: {successful_tests}/{total_tests} successful ({100*successful_tests/total_tests:.1f}%)")

    complete_results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "pipeline": "cogwall",
            "mode": "E2_ticket" if use_ticket else "E3_ablation",
            "questions_file": questions_file,
            "start_index": start_idx,
            "end_index": end_idx,
            "total_questions": end_idx - start_idx,
            "attack_method": attack_method,
            "model": client_name,
            "random_seed": random_seed,
            "n_samples_per_question": n_samples_per_question,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": round(100*successful_tests/total_tests, 2) if total_tests > 0 else 0,
        },
        "shared_config": {"attack_prefix": attack_prefix, "use_ticket": use_ticket},
        "results": all_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    mode_tag = "ticket" if use_ticket else "ablation"
    filename = f"cogwall-{client_name.replace('/', '-')}-{attack_method}-{mode_tag}-{end_idx-start_idx}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, filename)

    cleaned_results = clean_dict_for_json(complete_results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")
    return complete_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CogWall Testing Pipeline")
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--adjacency", type=str, required=True)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--attack", type=str, default="baseline")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--ablation", action="store_true",
                        help="E3 mode: disable ticket routing (prompt-only constraint)")
    args = parser.parse_args()

    provider = identify_model_provider(args.model)
    from clients.openai_compatible_client import OpenAICompatibleClient
    client = OpenAICompatibleClient(model=args.model, provider=provider)

    asyncio.run(run_cogwall_test(
        questions_file=args.questions,
        adjacency_csv=args.adjacency,
        client=client,
        client_name=args.model,
        attack_method=args.attack,
        start_idx=args.start,
        end_idx=args.end,
        n_samples_per_question=args.samples,
        max_concurrent=args.concurrent,
        random_seed=args.seed,
        output_dir=args.output,
        use_ticket=not args.ablation,
    ))


if __name__ == "__main__":
    main()
