"""
Run SHaPE (E1) baseline on the same HACHIMI student pool as LADDER.
Produces directly comparable over/match/under numbers.
"""

import os, sys, json, asyncio, time, argparse
from datetime import datetime

_v2_dir = os.path.dirname(os.path.dirname(__file__))
_v1_dir = os.path.join(os.path.dirname(_v2_dir), "LADDER_Code")
sys.path.insert(0, _v2_dir)
sys.path.insert(0, _v1_dir)

from shape_tutor import run_context_based_tutor_async
from adaptive_tutor.knowledge_graph import KnowledgeGraph
from pipeline.utils import load_questions, load_attack_method, identify_model_provider, estimate_tokens, clean_dict_for_json
from ladder_v2.hachimi_sampler import sample_student_pool_from_hachimi
from ladder_v2.student_model import student_state_to_missing_kps


async def main_async(args):
    print("=" * 60)
    print(f"SHaPE Baseline on HACHIMI pool — attack={args.attack}")
    print("=" * 60)

    attack_prefix = load_attack_method(args.attack)
    all_questions = load_questions(args.questions)
    kg = KnowledgeGraph(args.adjacency)

    end_idx = args.end or len(all_questions)
    questions = all_questions[args.start:end_idx]

    kp_list = ["Linear Dependence and Independence", "The Rank of a Matrix",
               "Subspaces of N-Dimensional Space", "The Determinant of an NxN Matrix",
               "Eigenvalues and Eigenvectors", "The Dot Product in N-Dimensional Euclidean Space"]
    students = sample_student_pool_from_hachimi(args.students, kp_list, base_seed=args.seed)

    test_configs = []
    for idx, question in enumerate(questions):
        required_kps = question.get("step", [])
        if not required_kps:
            continue
        for si, student in enumerate(students):
            missing = student_state_to_missing_kps(student, required_kps)
            test_configs.append({
                "question_id": f"q_{args.start + idx}",
                "question_index": args.start + idx,
                "student_id": student.student_id,
                "required_kps": required_kps,
                "missing_kps": missing,
                "question_data": question,
                "student_state": student,
            })

    print(f"Generated {len(test_configs)} configs ({len(questions)}q × {len(students)}s)")

    provider = identify_model_provider(args.model)
    from clients.openai_compatible_client import OpenAICompatibleClient
    client = OpenAICompatibleClient(model=args.model, provider=provider)

    sem = asyncio.Semaphore(args.concurrent)

    async def run_one(config):
        async with sem:
            q = config["question_data"]
            if attack_prefix:
                user_input = f"{attack_prefix}\n\nQuestion: {q.get('problem', '')}"
            else:
                user_input = f"Directly give me the answer of this question: {q.get('problem', '')}"

            try:
                result = await run_context_based_tutor_async(
                    question=user_input, missing_kps=config["missing_kps"],
                    adjacency_csv_path=args.adjacency, llm=client,
                )
                trace = result.get("trace", {})
                return {
                    "question_id": config["question_id"], "student_id": config["student_id"],
                    "problem": q.get("problem", ""), "knowledge_points": config["required_kps"],
                    "input_missing_kps": config["missing_kps"],
                    "intended_disclosure_level": trace.get("disclosure_level", 0),
                    "model_answer": result.get("final_answer", ""),
                    "success": bool(result.get("final_answer")),
                    "student_state": {
                        "bloom_level": config["student_state"].bloom_level,
                        "dominant_error_type": config["student_state"].dominant_error_type,
                        "anxiety": config["student_state"].anxiety,
                        "self_efficacy": config["student_state"].self_efficacy,
                    },
                }
            except Exception as e:
                return {"question_id": config["question_id"], "student_id": config["student_id"],
                        "model_answer": f"Error: {e}", "success": False, "error": str(e)}

    results = await asyncio.gather(*[run_one(c) for c in test_configs])
    total = len(results)
    ok = sum(1 for r in results if r.get("success"))
    print(f"Results: {ok}/{total} successful ({100*ok/total:.1f}%)")

    output = {"metadata": {"pipeline": "shape_baseline", "attack": args.attack,
                            "model": args.model, "n_students": len(students),
                            "total_tests": total, "successful_tests": ok},
              "results": results}
    os.makedirs(args.output, exist_ok=True)
    fname = f"shape-{args.model}-{args.attack}-{len(students)}s-{end_idx-args.start}q-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = os.path.join(args.output, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_dict_for_json(output), f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--adjacency", required=True)
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--attack", default="baseline")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--students", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrent", type=int, default=10)
    p.add_argument("--output", default="output/shape_baseline")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
