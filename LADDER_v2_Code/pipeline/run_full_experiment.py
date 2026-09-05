"""
Unified experiment runner for LADDER paper — all 5 arms.
E0 Vanilla / E1 SHaPE / A2 LADDER-K / A3 LADDER-full / A4 LADDER-prompt
"""

import os, sys, csv, json, asyncio, time, argparse, re
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_v2_dir = os.path.dirname(os.path.dirname(__file__))
_v1_dir = os.path.join(os.path.dirname(_v2_dir), "LADDER_Code")
sys.path.insert(0, _v1_dir)  # LADDER_Code first (pipeline.utils lives here)
sys.path.insert(0, _v2_dir)  # LADDER_v2_Code second (ladder_v2 lives here)

from adaptive_tutor.knowledge_graph import KnowledgeGraph
from pipeline.utils import load_questions, load_attack_method, identify_model_provider, clean_dict_for_json
from ladder_v2.hachimi_sampler import sample_student_pool_from_hachimi
from ladder_v2.student_model import student_state_to_missing_kps


def load_kp_vocab_from_adjacency(csv_path: str):
    """Load and validate the complete SHaPE KP vocabulary from its KG matrix."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"Invalid adjacency matrix: {csv_path}")

    header_kps = [cell.strip() for cell in rows[0][1:] if cell.strip()]
    row_kps = [row[0].strip() for row in rows[1:] if row and row[0].strip()]
    if not header_kps or len(header_kps) != len(set(header_kps)):
        raise ValueError("Adjacency matrix KP columns are empty or duplicated")
    if set(header_kps) != set(row_kps):
        raise ValueError("Adjacency matrix row and column KP sets differ")
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError("Adjacency matrix is not rectangular")
    return header_kps


async def run_arm(args, arm: str, questions, students, kg, client, client_name):
    """Run one experiment arm."""
    test_configs = []
    for idx, question in enumerate(questions):
        required_kps = question.get("step", [])
        if not required_kps: continue
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
                "arm": arm,
            })

    print(f"  {arm}: {len(test_configs)} configs — running...", flush=True)

    sem = asyncio.Semaphore(args.concurrent)
    attack_prefix = load_attack_method(args.attack)
    total_cfgs = len(test_configs)
    progress = {"done": 0}

    def _tick():
        progress["done"] += 1
        n = progress["done"]
        if n % 50 == 0 or n == total_cfgs:
            print(f"    [{arm}] {n}/{total_cfgs} done ({100*n/total_cfgs:.1f}%)", flush=True)

    async def run_one(config):
        async with sem:
            q = config["question_data"]
            # All attacks, including baseline, use the same extraction contract.
            parts = [attack_prefix.strip(), f"Question: {q.get('problem', '').strip()}"]
            user_input = "\n\n".join(part for part in parts if part)

            last_error = None
            t0 = time.time()
            for attempt in range(1, args.retries + 1):
                try:
                    if arm == "E0":
                        result = await _run_vanilla(user_input, q, client, config)
                    elif arm == "E1":
                        result = await _run_shape(user_input, q, config, args.adjacency, client)
                    elif arm == "A4":
                        result = await _run_ladder(user_input, config, args.adjacency, client, "graded_prompt", kg)
                    else:
                        result = await _run_ladder(user_input, config, args.adjacency, client, "graded_ticket", kg)
                    result["attack"] = args.attack
                    result["attempts"] = attempt
                    result["processing_time"] = round(time.time() - t0, 2)
                    _tick()
                    return result
                except Exception as e:
                    last_error = e
                    if attempt < args.retries:
                        await asyncio.sleep(min(2 ** (attempt - 1), 8))

            _tick()
            return {"question_id": config["question_id"], "student_id": config["student_id"],
                    "arm": arm, "attack": args.attack, "raw_question": user_input,
                    "problem": q.get("problem", ""), "success": False,
                    "attempts": args.retries, "processing_time": round(time.time() - t0, 2),
                    "error": str(last_error)}

    results = await asyncio.gather(*[run_one(c) for c in test_configs])
    total = len(results)
    ok = sum(1 for r in results if r.get("success"))
    print(f"  {arm}: {ok}/{total} ({100*ok/total:.1f}%)", flush=True)

    return results


async def _run_vanilla(user_input, question, client, config):
    """E0: No gate, always give full solution."""
    from context_based_tutor.adapters import adapt_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = adapt_llm(client)
    from context_based_tutor.prompts import DIRECT_ANSWER_PROMPT
    resp = await llm.ainvoke([
        SystemMessage(content=DIRECT_ANSWER_PROMPT),
        HumanMessage(content=user_input),
    ])
    ans = (resp.content or "").strip()
    return {
        "question_id": config["question_id"], "student_id": config["student_id"],
        "arm": "E0", "problem": question.get("problem", ""),
        "expected_answer": question.get("answer") or question.get("gpt_answer", ""),
        "reference_steps": question.get("step_detail", []),
        "dataset_required_kps": question.get("step", []),
        "raw_question": user_input, "clean_question": None,
        "extract_failed": None, "parse_failed": None,
        "model_answer": ans, "success": bool(ans),
        "routing_disclosure_level": 3,
        "student_state": _student_dict(config["student_state"]),
    }


async def _run_shape(user_input, question, config, adj_path, client):
    """E1: SHaPE binary gate."""
    from shape_tutor import run_context_based_tutor_async
    result = await run_context_based_tutor_async(
        question=user_input, missing_kps=config["missing_kps"],
        adjacency_csv_path=adj_path, llm=client,
    )
    trace = result.get("trace", {})
    return {
        "question_id": config["question_id"], "student_id": config["student_id"],
        "arm": "E1", "problem": question.get("problem", ""),
        "expected_answer": question.get("answer") or question.get("gpt_answer", ""),
        "reference_steps": question.get("step_detail", []),
        "dataset_required_kps": question.get("step", []),
        "raw_question": user_input, "clean_question": None,
        "extract_failed": None, "parse_failed": trace.get("parse_failed"),
        "model_answer": result.get("final_answer", ""),
        "success": bool(result.get("final_answer")),
        "routing_disclosure_level": trace.get("disclosure_level", 0),
        "diagnosed_required_kps": trace.get("required_kps", []),
        "diagnosed_missing_kps": trace.get("missing_kps", []),
        "input_missing_kps": config["missing_kps"],
        "student_state": _student_dict(config["student_state"]),
    }


async def _run_ladder(user_input, config, adj_path, client, arm, kg):
    """A2/A3/A4: LADDER arms."""
    if arm == "graded_ticket":
        from ladder_v2.run_v2 import run_ladder_tutor_v2
        # A3 routes via the six-dimension gate; A2 via the K-only gate.
        gate_mode = "multidim" if config.get("arm") == "A3" else "classic"
        result = await run_ladder_tutor_v2(
            question=user_input, missing_kps=config["missing_kps"],
            student=config["student_state"],
            adjacency_csv_path=adj_path, llm=client, arm=arm, kg=kg,
            gate_mode=gate_mode,
        )
    else:
        from ladder import run_ladder_tutor_async
        result = await run_ladder_tutor_async(
            question=user_input, missing_kps=config["missing_kps"],
            adjacency_csv_path=adj_path, llm=client, arm=arm, kg=kg,
        )
    trace = result.get("trace", {})
    ticket = trace.get("ticket", {}) or {}
    q = config["question_data"]
    return {
        "question_id": config["question_id"], "student_id": config["student_id"],
        "arm": config["arm"], "problem": q.get("problem", ""),
        "expected_answer": q.get("answer") or q.get("gpt_answer", ""),
        "reference_steps": q.get("step_detail", []),
        "dataset_required_kps": q.get("step", []),
        "raw_question": user_input,
        "clean_question": trace.get("clean_question"),
        "extract_failed": bool(trace.get("extract_failed", False)),
        "parse_failed": bool(trace.get("parse_failed", False)),
        "model_answer": result.get("final_answer", ""),
        "success": bool(result.get("final_answer")),
        "routing_disclosure_level": trace.get("disclosure_level"),
        "gate_info": trace.get("gate_info") or ticket.get("gate_info", {}),
        "diagnosed_required_kps": trace.get("required_kps", []),
        "diagnosed_missing_kps": trace.get("missing_kps", []),
        "input_missing_kps": config["missing_kps"],
        "student_state": _student_dict(config["student_state"]),
    }


def _student_dict(s):
    return {"kp_mastery": dict(s.kp_mastery), "misconceptions": list(s.misconceptions),
            "bloom_level": s.bloom_level, "dominant_error_type": s.dominant_error_type,
            "anxiety": s.anxiety, "self_efficacy": s.self_efficacy,
            "motivation": s.motivation, "learning_speed": s.learning_speed,
            "help_seeking": s.help_seeking, "preference": s.preference,
            "trajectory": s.trajectory,
            "behavioral_state": s.behavioral_state}


async def main_async(args):
    questions = load_questions(args.questions)
    kg = KnowledgeGraph(args.adjacency)
    end = args.end or len(questions)
    questions = questions[args.start:end]
    kp_vocab = load_kp_vocab_from_adjacency(args.adjacency)
    if len(kp_vocab) != 91:
        raise ValueError(f"Expected the SHaPE 91-KP graph, found {len(kp_vocab)} KPs")
    unknown_question_kps = sorted({
        kp for q in questions for kp in (q.get("step", []) or []) if kp not in set(kp_vocab)
    })
    if unknown_question_kps:
        raise ValueError("Question data references KPs outside the graph: " + ", ".join(unknown_question_kps))
    students = sample_student_pool_from_hachimi(args.students, kp_vocab, base_seed=args.seed)
    if any(len(s.kp_mastery) != len(kp_vocab) for s in students):
        raise ValueError("Student pool does not contain complete 91-KP mastery profiles")

    provider = args.provider or identify_model_provider(args.model)
    if provider == "ollama":
        from clients.ollama_native_client import OllamaNativeClient
        client = OllamaNativeClient(
            model=args.model,
            base_url=args.ollama_url,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        )
    else:
        from clients.openai_compatible_client import OpenAICompatibleClient
        client = OpenAICompatibleClient(
            model=args.model,
            provider=provider,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        )
    client_name = args.model

    arms = args.arms.split(",")
    all_results = {}

    for arm in arms:
        arm = arm.strip()
        print(f"\n{'='*50}\n  Running {arm}\n{'='*50}")
        results = await run_arm(args, arm, questions, students, kg, client, client_name)
        all_results[arm] = results

        # Save per-arm
        out = {"metadata": {"arm": arm, "attack": args.attack, "attack_method": args.attack,
                             "model": client_name, "provider": provider,
                             "temperature": args.temperature, "seed": args.seed,
                             "kp_count": len(kp_vocab), "kp_vocab": kp_vocab,
                             "n_students": len(students), "n_questions": len(questions),
                             "total": len(results)},
               "results": results}
        os.makedirs(args.output, exist_ok=True)
        safe_client_name = re.sub(r'[<>:"/\\|?*]+', "_", client_name).strip(" .") or "model"
        fname = f"{arm}-{safe_client_name}-{args.attack}-{len(students)}s-{len(questions)}q-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(args.output, fname), "w", encoding="utf-8") as f:
            json.dump(clean_dict_for_json(out), f, ensure_ascii=False, indent=2)

    print(f"\nAll done. Output: {args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--adjacency", required=True)
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--provider", default=None, help="override provider; use 'ollama' for local native API")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-completion-tokens", type=int, default=8000)
    p.add_argument("--arms", default="E0,E1,A2,A3,A4", help="comma-separated")
    p.add_argument("--attack", default="baseline")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--students", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrent", type=int, default=10)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--output", default="output/final_experiment")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
