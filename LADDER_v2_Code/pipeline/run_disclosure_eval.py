"""
Bridge: run v1 disclosure evaluator on v2 experiment output.

Maps v2's multi_dim_intended_level → v1's intended_disclosure_level,
then calls the original disclosure_evaluation.py pipeline.
"""

import os, sys, json, glob, asyncio, argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_V2_DIR = os.path.dirname(os.path.dirname(__file__))
_V1_DIR = os.path.join(os.path.dirname(_V2_DIR), "LADDER_Code")
sys.path.insert(0, _V1_DIR)

from evaluation.disclosure_evaluation import DisclosureEvaluator, process_file_async


def bridge_v2_to_v1(input_file: str, output_dir: str):
    """Preprocess v2 output → v1-compatible format, then evaluate."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    for r in results:
        # Map v2 fields to v1 fields
        if "multi_dim_intended_level" in r and "intended_disclosure_level" not in r:
            r["intended_disclosure_level"] = r["multi_dim_intended_level"]
        # Also map input_missing_kps for the evaluator's context
        if "input_missing_kps" not in r:
            r["input_missing_kps"] = r.get("missing_kps", [])

    # Write bridged file
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(input_file).replace(".json", "_bridged.json")
    bridged_path = os.path.join(output_dir, base)
    with open(bridged_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return bridged_path


async def main_async(args):
    # Find input files
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.json")))
        files = [f for f in files if "bridged" not in f and "disclosure" not in f]
    else:
        files = [args.input]

    print(f"Processing {len(files)} files")
    evaluator = DisclosureEvaluator(model=args.model, provider=args.provider)

    for f in files:
        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(f)}")
        out_name = os.path.basename(f).replace(".json", "_disclosure.json")
        out_path = os.path.join(args.output, out_name)
        # v5.1 raw records already use the runtime routing schema; evaluate them
        # directly so there is no second, ambiguous "intended" level.
        stats = await process_file_async(f, evaluator, out_path, args.concurrent)

        n = stats.get("scored_results", 0)
        over = stats.get("over_disclosure_pct", 0)
        match = stats.get("match_exact_pct", 0)
        under = stats.get("under_disclosure_pct", 0)
        print(f"  n={n}  over={over:.1f}%  match={match:.1f}%  under={under:.1f}%")

    print(f"\nDone. Output: {args.output}")


def main():
    p = argparse.ArgumentParser(description="Run v1 disclosure evaluator on v2 output")
    p.add_argument("--input", required=True, help="v2 experiment output dir or file")
    p.add_argument("--output", default="output/disclosure", help="output dir")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--provider", default="apinebula")
    p.add_argument("--concurrent", type=int, default=10)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
