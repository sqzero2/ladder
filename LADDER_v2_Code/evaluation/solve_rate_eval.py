"""
Post-dialog solve rate evaluation.
Adopted from PedagogicalRL (EMNLP 2025) and MHPO (ACL 2026).

LLM judge evaluates: given the teacher's response, would the student be
able to solve this problem independently? This is a proxy for actual
post-dialog solve rate.
"""

import os, sys, json, asyncio, argparse, glob, re
from collections import defaultdict

_V2_DIR = os.path.dirname(os.path.dirname(__file__))
_V1_DIR = os.path.join(os.path.dirname(_V2_DIR), "LADDER_Code")
sys.path.insert(0, _V1_DIR)

from evaluation.pedagogical_evaluation import PedagogicalEvaluator

_JUDGE_SYSTEM = """You are an educational assessment expert. Evaluate the pedagogical quality of an AI teacher's response on a 1-5 scale:

5: Excellent — provides the right amount of scaffolding: concept explanation + hint specific to the problem, student can take the next step independently.
4: Good — explains concepts with a relevant example, but doesn't quite connect to the specific problem.
3: Adequate — gives a general concept explanation and invites the student to try, without specific guidance.
2: Poor — only asks generic Socratic questions, gives no structural support.
1: Very poor — off-topic, incorrect, or directly gives the full answer without any teaching.

Return ONLY a JSON: {"pedagogy": <int 1-5>, "reason": "<one sentence>"}"""


class SolveRateEvaluator:
    def __init__(self, model="gpt-5.4-mini", provider="apinebula"):
        base = PedagogicalEvaluator(evaluator_model=model, evaluator_provider=provider)
        self.client = base.evaluator_client
        self.provider = base.evaluator_provider
        self.model = model

    async def judge(self, problem: str, answer: str, teacher_response: str,
                    disclosure_type: int) -> dict:
        if not self.client or not teacher_response:
            return None

        user = (
            f"Problem: {problem[:500]}\n"
            f"Correct answer: {answer[:200]}\n"
            f"Teacher's response (disclosure type L{disclosure_type}):\n{teacher_response[:1500]}\n"
        )

        loop = asyncio.get_event_loop()
        try:
            if self.provider in ("openai", "apinebula"):
                resp = await loop.run_in_executor(
                    None,
                    lambda: self.client.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": _JUDGE_SYSTEM},
                                  {"role": "user", "content": user}],
                        temperature=0.7, max_completion_tokens=300,
                    ),
                )
                text = resp.choices[0].message.content if resp and resp.choices else ""
            else:
                return None
        except Exception as e:
            print(f"  Judge error: {e}")
            return None

        text = (text or "").strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'"can_solve"\s*:\s*(\d)', text)
            if m:
                return {"can_solve": int(m.group(1)), "reason": "extracted"}
            return None


async def process_file(input_file: str, evaluator: SolveRateEvaluator,
                       output_file: str, max_concurrent: int = 5) -> dict:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    sem = asyncio.Semaphore(max_concurrent)

    async def rate_one(r):
        async with sem:
            ans = r.get("model_answer", "") or ""
            problem = r.get("problem", "")
            expected = r.get("expected_answer", "")
            intended = r.get("multi_dim_intended_level") or r.get("intended_disclosure_level", 0)

            result = await evaluator.judge(problem, expected, ans, intended)
            r = dict(r)
            if result:
                r["pedagogy_score"] = result.get("pedagogy", 0)
                r["pedagogy_reason"] = result.get("reason", "")
            return r

    rated = await asyncio.gather(*[rate_one(r) for r in results])

    # Compute pedagogy score by disclosure type
    by_type = defaultdict(list)
    for r in rated:
        t = r.get("multi_dim_intended_level") or r.get("intended_disclosure_level", 0)
        ps = r.get("pedagogy_score")
        if ps is not None and ps > 0:
            by_type[t].append(ps)

    ped_scores = {}
    for t in sorted(by_type.keys()):
        vals = by_type[t]
        ped_scores[f"L{t}"] = {
            "mean_pedagogy": round(sum(vals) / len(vals), 2) if vals else 0,
            "n": len(vals),
        }

    all_scores = [ps for v in by_type.values() for ps in v]

    summary = {
        "n_total": len(rated),
        "n_scored": len(all_scores),
        "overall_pedagogy": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0,
        "pedagogy_by_type": ped_scores,
    }

    output = {"metadata": data.get("metadata", {}), "summary": summary, "results": rated}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return summary


async def main_async(args):
    files = sorted(glob.glob(os.path.join(args.input, "*.json")))
    files = [f for f in files if "disclosure" not in f and "bridged" not in f and "solve" not in f]
    evaluator = SolveRateEvaluator(model=args.model, provider=args.provider)

    for f in files:
        name = os.path.basename(f)
        print(f"Evaluating: {name}")
        out_name = name.replace(".json", "_solve.json")
        summary = await process_file(f, evaluator, os.path.join(args.output, out_name), args.concurrent)

        print(f"  Overall pedagogy: {summary['overall_pedagogy']:.1f} (n={summary['n_scored']})")
        for t, v in summary.get("pedagogy_by_type", {}).items():
            print(f"  {t}: {v['mean_pedagogy']:.1f} (n={v['n']})")

    print(f"\nDone. Output: {args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="evaluation/output/solve_rate")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--provider", default="apinebula")
    p.add_argument("--concurrent", type=int, default=5)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
