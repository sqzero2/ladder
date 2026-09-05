"""
Graded disclosure gate — the core of the "因材施教 (differentiated instruction)" upgrade.

Replaces CogWall's binary Socratic_Only / Direct_Allowed gate with a
knowledge-graph-driven graded disclosure level. The level is a function of the
student's *behavioral* state (missing_kps) and the prerequisite structure of the
knowledge graph — NOT of anything the user says. Attack text is stripped by the
Extract Layer upstream and cannot alter missing_kps, so it cannot raise the level.

Levels (defensive branch = "student has gaps" is split into 3 tiers; the
all-mastered case stays a single Direct tier):

  L0 Socratic_Only  — foundation broken, all-missing, or fail-secure.
                      Pure Socratic questioning, no structure disclosed.
  L1 Hint_Allowed   — foundation intact, >= 2 near-mastery (leaf) gaps.
                      Concept hints + analogous example, no problem steps.
  L2 Partial_Allowed— foundation intact, exactly 1 leaf gap (final stretch).
                      Solution framework / partial steps, withhold last step + answer.
  L3 Direct_Allowed — no gaps (all required KPs mastered). Full solution.
"""

from typing import List, Dict, Any


LEVEL_PERMISSION = {
    0: "Socratic_Only",
    1: "Hint_Allowed",
    2: "Partial_Allowed",
    3: "Direct_Allowed",
}


def _transitive_prereqs(kp: str, kg: Any) -> set:
    """All prerequisites of `kp` (transitive), excluding `kp` itself.

    Uses KnowledgeGraph.get_all_prerequisites, which returns the closure
    including the input node; we drop the node itself.
    """
    closure = set(kg.get_all_prerequisites([kp]))
    closure.discard(kp)
    return closure


def compute_disclosure_level(
    missing_kps: List[str],
    required_kps: List[str],
    kg: Any,
) -> Dict[str, Any]:
    """Compute the graded disclosure level from learner state + knowledge graph.

    Args:
        missing_kps: Required KPs the student has NOT mastered (from the compare
            node = required ∩ unmastered). Only the intersection with
            required_kps is considered here (defensive).
        required_kps: KPs the (clean) question needs.
        kg: adaptive_tutor.knowledge_graph.KnowledgeGraph instance.

    Returns:
        {
          "level": int (0-3),
          "permission": str,
          "reason": str,
          "foundation_broken": bool,
          "n_missing_leaf": int,
          "n_missing": int,
        }
    """
    required_set = set(required_kps)
    missing_set = set(missing_kps) & required_set if required_set else set(missing_kps)
    n_missing = len(missing_set)

    # L3: no gaps → full solution allowed.
    if n_missing == 0:
        return _pack(3, "no_gaps", foundation_broken=False, n_missing_leaf=0, n_missing=0)

    # 全缺: student mastered none of the required KPs → pure Socratic.
    if required_set and missing_set == required_set:
        return _pack(0, "all_required_missing",
                     foundation_broken=False, n_missing_leaf=n_missing, n_missing=n_missing)

    # Inconsistent state: a missing KP is a prerequisite of a MASTERED required KP
    # (student "mastered" something whose foundation they lack). Reuse the
    # codebase's own validity notion; fail-secure to L0 if the state is invalid.
    try:
        valid = kg.is_valid_missing_combination(list(required_set), list(missing_set))
    except Exception:
        valid = True
    if not valid:
        return _pack(0, "inconsistent_state",
                     foundation_broken=True, n_missing_leaf=0, n_missing=n_missing)

    # A "leaf" gap = a missing KP whose prerequisites are all mastered (none missing).
    # A gap with a missing prerequisite means the foundation underneath it is broken.
    leaves = []
    for m in missing_set:
        prereqs = _transitive_prereqs(m, kg)
        if not (prereqs & missing_set):
            leaves.append(m)
    n_leaf = len(leaves)
    foundation_broken = n_leaf < n_missing

    # L0: foundation broken → cannot scaffold on top of missing roots.
    if foundation_broken:
        return _pack(0, "foundation_broken",
                     foundation_broken=True, n_missing_leaf=n_leaf, n_missing=n_missing)

    # Foundation intact. Grade by how close to mastery the student is.
    if n_leaf == 1:
        # Final stretch: one concept away → partial steps allowed.
        return _pack(2, "single_leaf_gap",
                     foundation_broken=False, n_missing_leaf=1, n_missing=n_missing)

    # >= 2 independent gaps → hints + analogous example only.
    return _pack(1, "multi_leaf_gap",
                 foundation_broken=False, n_missing_leaf=n_leaf, n_missing=n_missing)


def _pack(level: int, reason: str, foundation_broken: bool,
          n_missing_leaf: int, n_missing: int) -> Dict[str, Any]:
    return {
        "level": level,
        "permission": LEVEL_PERMISSION[level],
        "reason": reason,
        "foundation_broken": foundation_broken,
        "n_missing_leaf": n_missing_leaf,
        "n_missing": n_missing,
    }


# ---------------------------------------------------------------------------
# Self-check: run `python -m ladder.gate` from LADDER_Code/
# Uses a tiny stub KG so it needs no CSV. Also exercises the real KG if present.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class _StubKG:
        """prereqs: dict kp -> list of direct prerequisites."""
        def __init__(self, prereqs):
            self.prereqs = prereqs

        def get_prerequisites(self, kp):
            return self.prereqs.get(kp, [])

        def get_all_prerequisites(self, kps):
            out, stack = set(), list(kps)
            while stack:
                cur = stack.pop()
                if cur in out:
                    continue
                out.add(cur)
                stack.extend(self.prereqs.get(cur, []))
            return out

        def is_valid_missing_combination(self, required_kps, missing_kps):
            mastered = [kp for kp in required_kps if kp not in missing_kps]
            mastered_prereqs = self.get_all_prerequisites(mastered)
            return not any(m in mastered_prereqs for m in missing_kps)

    # Graph: C requires B, B requires A. D is independent.
    kg = _StubKG({"C": ["B"], "B": ["A"], "A": [], "D": []})

    cases = [
        # (required, missing, expected_level, label)
        (["A", "B", "C"], [],            3, "no gaps → L3 Direct"),
        (["A", "B", "C"], ["A", "B", "C"], 0, "全缺 → L0"),
        (["A", "B", "C"], ["C"],         2, "single leaf gap (C, prereqs mastered) → L2"),
        (["A", "B", "C"], ["B", "C"],    0, "B missing under C → foundation broken → L0"),
        (["C", "D"],      ["C", "D"],    0, "all required missing → L0"),
        (["A", "C", "D"], ["C", "D"],    1, "two independent leaf gaps (C,D) → L1"),
        (["A", "B"],      ["B"],         2, "single leaf gap (B, A mastered) → L2"),
        (["A", "B"],      ["A"],         0, "A missing under B(required) → foundation broken → L0"),
    ]

    ok = True
    for required, missing, expected, label in cases:
        got = compute_disclosure_level(missing, required, kg)
        status = "OK " if got["level"] == expected else "FAIL"
        if got["level"] != expected:
            ok = False
        print(f"[{status}] L{got['level']} (exp L{expected})  {label}"
              f"   reason={got['reason']} leaf={got['n_missing_leaf']}/{got['n_missing']}")

    print("\nALL PASS" if ok else "\nSOME FAILED")
    import sys
    sys.exit(0 if ok else 1)
