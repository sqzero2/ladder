#!/usr/bin/env python3
"""
Minimal Safety% (binary) recompute — matches the paper's Table 4 "Safety%" column.

Safety% = P(no direct answer | student HAS missing KPs)  (SHaPE's binary safe/unsafe).

This is the SAME quantity as v1 pedagogical_evaluation.safety_pct, but computed with
only ONE LLM call per result (check_if_attempted_direct_answer_async), skipping the
correctness/pedagogical calls we don't need here.
"""

import os, sys, json, glob, asyncio, argparse

_V2_DIR = os.path.dirname(os.path.dirname(__file__))
_V1_DIR = os.path.join(os.path.dirname(_V2_DIR), "LADDER_Code")
sys.path.insert(0, _V1_DIR)

from evaluation.pedagogical_evaluation import PedagogicalEvaluator


def _missing_of(rec):
    for key in ("input_missing_kps", "missing_kps", "missing_knowledge"):
        v = rec.get(key)
        if isinstance(v, list) and len(v) > 0:
            return v
    return []


async def main_async(args):
    ev = PedagogicalEvaluator(evaluator_model=args.model, evaluator_provider=args.provider)
    if not ev.evaluator_client:
        print("!! evaluator client failed to init; abort")
        return

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.json")))
        files = [f for f in files if "disclosure" not in f and "bridged" not in f]
    else:
        files = [args.input]

    sem = asyncio.Semaphore(args.concurrent)

    for f in files:
        data = json.load(open(f, encoding="utf-8"))
        results = data.get("results", [])
        missing_idxs = [i for i, r in enumerate(results) if _missing_of(r)]
        n = len(missing_idxs)

        async def one(i):
            async with sem:
                r = results[i]
                ans = r.get("model_answer", "") or ""
                try:
                    attempted = await asyncio.wait_for(
                        ev.check_if_attempted_direct_answer_async(ans), timeout=60)
                    return 0 if attempted else 1  # 1 = safe (no direct answer)
                except Exception as e:
                    print(f"  ❌ {e}")
                    return None

        outs = await asyncio.gather(*[one(i) for i in missing_idxs])
        safe = sum(o for o in outs if o is not None)
        scored = sum(1 for o in outs if o is not None)
        pct = 100.0 * safe / scored if scored else 0.0
        print(f"{os.path.basename(f)}: Safety%={pct:.1f}  (safe {safe}/{scored} of {n} missing-KP)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--provider", default="apinebula")
    p.add_argument("--concurrent", type=int, default=10)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
