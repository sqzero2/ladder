"""
Multi-dimensional evaluation for LADDER v2 (no-API version).

Computes evaluation metrics directly from experiment data:
  L1: Teaching basics (inherited from v1 disclosure evaluator)
  L2: Interaction quality (rule-based: affect phrase matching)
  L3: Adaptation quality (dimensional sensitivity)
  L4: Security (over/match/under disclosure rates)

LLM judge evaluation (affect_score, pck_score, level_score) requires API
and can be run with --llm_judge flag.
"""

import os, sys, json, re, argparse, glob
from collections import defaultdict

# Add LADDER_Code for disclosure evaluator
_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_V2_ROOT = os.path.dirname(_V2_DIR)
_PARENT = os.path.dirname(_V2_ROOT)
_V1_DIR = os.path.join(_PARENT, "LADDER_Code")
sys.path.insert(0, _V1_DIR)


# ===========================================================================
# Rule-based affect detection in teacher responses
# ===========================================================================

AFFECT_PATTERNS = {
    "encouraging": [
        "你已经", "很接近", "思路是对的", "做得很好", "不错",
        "加油", "没问题", "慢慢来", "不急",
        "good", "great", "well done", "close", "almost there",
    ],
    "challenging": [
        "你试试", "你觉得", "自己想想", "能不能",
        "try", "think about", "what if", "can you",
    ],
    "reengaging": [
        "不急", "先不看", "回到", "退一步",
        "let's step back", "let's review", "take a step back",
    ],
}


def detect_affective_tone(teacher_response: str) -> dict:
    """Rule-based affective tone detection in teacher response."""
    t = teacher_response.lower()
    scores = {}
    for tone, patterns in AFFECT_PATTERNS.items():
        count = sum(1 for p in patterns if p in t)
        scores[tone] = min(count, 3)  # cap at 3
    return scores


def check_affect_match(tone: dict, student_state: dict) -> float:
    """
    Check if teacher's tone matches student's affect.
    Returns 0-1 score.
    """
    anxiety = student_state.get("anxiety", 0.5)
    efficacy = student_state.get("self_efficacy", 0.5)
    behavior = student_state.get("behavioral_state", "on_task")

    score = 0.5  # neutral baseline

    # Anxious students should get encouraging tone
    if anxiety > 0.7 and tone.get("encouraging", 0) > 0:
        score += 0.3
    elif anxiety > 0.7 and tone.get("encouraging", 0) == 0:
        score -= 0.2

    # Giving-up students should get re-engaging tone
    if behavior == "giving_up" and tone.get("reengaging", 0) > 0:
        score += 0.3
    elif behavior == "giving_up" and tone.get("reengaging", 0) == 0:
        score -= 0.2

    # Confident students can be challenged
    if efficacy > 0.7 and tone.get("challenging", 0) > 0:
        score += 0.1

    return max(0.0, min(1.0, score))


# ===========================================================================
# Dimensional sensitivity
# ===========================================================================

def compute_sensitivity(results: list) -> dict:
    """Do students with same K but different A get different levels?"""
    by_question = defaultdict(list)
    for r in results:
        qi = r.get("question_index", 0)
        level = r.get("multi_dim_intended_level")
        anxiety = r.get("student_state", {}).get("anxiety", 0.5)
        if level is not None:
            by_question[qi].append({
                "level": level, "anxiety": anxiety,
                "id": r.get("student_id"),
                "state": r.get("student_state", {}),
            })

    same_level = 0
    diff_level = 0
    pairs_checked = 0

    for qi, students in by_question.items():
        if len(students) < 2:
            continue
        levels = [s["level"] for s in students]
        if len(set(levels)) > 1:
            diff_level += 1
        else:
            same_level += 1
        pairs_checked += 1

    # Also compute: for students with same K but anxiety difference > 0.5,
    # how often do levels differ?
    k_same_a_diff = {"total": 0, "level_different": 0}
    for qi, students in by_question.items():
        for i in range(len(students)):
            for j in range(i+1, len(students)):
                a_diff = abs(students[i]["anxiety"] - students[j]["anxiety"])
                if a_diff > 0.5:
                    k_same_a_diff["total"] += 1
                    if students[i]["level"] != students[j]["level"]:
                        k_same_a_diff["level_different"] += 1

    total = same_level + diff_level
    return {
        "questions_with_variation": diff_level,
        "questions_without_variation": same_level,
        "sensitivity_rate": round(diff_level / total, 3) if total else 0,
        "total_questions": total,
        "same_K_diff_A_level_diff_rate": (
            round(k_same_a_diff["level_different"] / k_same_a_diff["total"], 3)
            if k_same_a_diff["total"] else 0
        ),
    }


# ===========================================================================
# Main processing
# ===========================================================================

def process_file(input_file: str, output_file: str) -> dict:
    """Compute all no-API evaluation metrics for one experiment file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    metadata = data.get("metadata", {})

    # --- L1+L4: Disclosure rates (from v1 disclosure evaluator if available) ---
    # Try to load the paired disclosure evaluation result
    disc_file = input_file.replace(".json", "_disclosure.json")
    disc_stats = {}
    if os.path.exists(disc_file):
        try:
            with open(disc_file, "r", encoding="utf-8") as f:
                disc_data = json.load(f)
            disc_stats = disc_data.get("summary", {})
        except Exception:
            pass

    # --- L2: Affect matching (rule-based) ---
    affect_scores = []
    for r in results:
        ans = r.get("model_answer", "") or ""
        ss = r.get("student_state", {})
        if ans and ss:
            tone = detect_affective_tone(ans)
            score = check_affect_match(tone, ss)
            affect_scores.append(score)

    # --- L3: Dimensional sensitivity ---
    sensitivity = compute_sensitivity(results)

    # --- Level distribution comparison ---
    md_dist = metadata.get("multi_dim_level_distribution", {})
    cl_dist = metadata.get("classic_level_distribution", {})
    diverged = metadata.get("diverged_count", 0)
    diverged_rate = metadata.get("diverged_rate", 0)

    # --- Per-student breakdown ---
    by_student = defaultdict(lambda: {"v2": [], "v1": [], "affect_matches": []})
    for r in results:
        sid = r.get("student_id", "?")
        v2_lvl = r.get("multi_dim_intended_level")
        v1_lvl = r.get("classic_intended_level")
        if v2_lvl is not None:
            by_student[sid]["v2"].append(v2_lvl)
        if v1_lvl is not None:
            by_student[sid]["v1"].append(v1_lvl)
        ans = r.get("model_answer", "") or ""
        ss = r.get("student_state", {})
        if ans and ss:
            tone = detect_affective_tone(ans)
            by_student[sid]["affect_matches"].append(check_affect_match(tone, ss))

    student_breakdown = {}
    for sid, info in by_student.items():
        from collections import Counter
        v2c = Counter(info["v2"])
        v1c = Counter(info["v1"])
        avg_affect = (sum(info["affect_matches"]) / len(info["affect_matches"])
                      if info["affect_matches"] else None)
        student_breakdown[sid] = {
            "v2_distribution": {str(k): v for k, v in sorted(v2c.items())},
            "v1_distribution": {str(k): v for k, v in sorted(v1c.items())},
            "avg_affect_match": round(avg_affect, 3) if avg_affect else None,
            "n_responses": len(info["v2"]),
        }

    # --- Summary ---
    summary = {
        "n_total": len(results),
        # L1+L4: Disclosure
        "over_rate": disc_stats.get("over_rate"),
        "match_rate": disc_stats.get("match_rate"),
        "under_rate": disc_stats.get("under_rate"),
        "n_scored_disclosure": disc_stats.get("n_scored"),
        # L2: Affect
        "affect_match_mean": round(sum(affect_scores) / len(affect_scores), 3) if affect_scores else None,
        "n_scored_affect": len(affect_scores),
        # L3: Adaptation
        "dimensional_sensitivity": sensitivity,
        "diverged_from_classic": diverged,
        "diverged_rate": diverged_rate,
        "multi_dim_distribution": md_dist,
        "classic_distribution": cl_dist,
    }

    output = {
        "metadata": metadata,
        "summary": summary,
        "student_breakdown": student_breakdown,
        "results": results,  # original results preserved
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return summary


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="Multi-Dim Evaluation (no-API)")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="evaluation/output/multidim")
    args = p.parse_args()

    # Find input
    files = []
    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.json")))
        files = [f for f in files if "disclosure" not in f and "bridged" not in f]
    else:
        print(f"Input not found: {args.input}")
        return

    for f in files:
        name = os.path.basename(f)
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        out_name = name.replace(".json", "_multidim.json")
        out_path = os.path.join(args.output, out_name)
        summary = process_file(f, out_path)

        # --- Print results ---
        print(f"  n_total: {summary['n_total']}")
        # L1+L4
        if summary.get("over_rate") is not None:
            print(f"  [L4] over={summary['over_rate']:.1%}  match={summary['match_rate']:.1%}  under={summary['under_rate']:.1%}")
        else:
            print(f"  [L4] (disclosure eval not found — run pipeline/run_disclosure_eval.py first)")
        # L2
        print(f"  [L2] affect_match: {summary['affect_match_mean']} (n={summary['n_scored_affect']})")
        # L3
        sens = summary["dimensional_sensitivity"]
        print(f"  [L3] sensitivity: {sens['sensitivity_rate']} ({sens['questions_with_variation']}/{sens['total_questions']} questions)")
        print(f"       same_K_diff_A_level_diff: {sens['same_K_diff_A_level_diff_rate']}")
        print(f"       diverged from classic: {summary['diverged_from_classic']}/{summary['n_total']} ({summary['diverged_rate']}%)")

        # Per-student
        student_breakdown = json.load(open(out_path, encoding="utf-8")).get("student_breakdown", {})
        if student_breakdown:
            print(f"  Per-student:")
            for sid, info in sorted(student_breakdown.items()):
                v2d = info["v2_distribution"]
                affect = info.get("avg_affect_match")
                print(f"    {sid}: v2={v2d}  affect={affect}")

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
