"""
CogWall entry points for running the tutoring workflow.
"""

import os
from typing import Dict, List, Set, Any

from .utils import load_knowledge_graph
from .graph import build_graph
from .adapters import adapt_llm


DEFAULT_ADJ_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "adjacency_matrix_knowledge_graph.csv"
)


def _get_all_ancestors(
    missing_kps: List[str], kp_vocab: List[str], adj_matrix: List[List[int]]
) -> Set[str]:
    """Recursively get all ancestor (prerequisite) knowledge points."""
    ancestors = set(missing_kps)
    to_check = set(missing_kps)
    while to_check:
        current = to_check.pop()
        if current in kp_vocab:
            idx = kp_vocab.index(current)
            for j, dependency in enumerate(adj_matrix[idx]):
                if dependency == 1:
                    ancestor = kp_vocab[j]
                    if ancestor not in ancestors:
                        ancestors.add(ancestor)
                        to_check.add(ancestor)
    return ancestors


async def run_cogwall_tutor_async(
    question: str,
    missing_kps: List[str],
    adjacency_csv_path: str = None,
    llm: Any = None,
    extract_model=None,
    debug: bool = False,
    use_ticket: bool = True,
) -> Dict:
    """Run the CogWall tutor (async).

    Args:
        question: The student's question text (may contain attacks)
        missing_kps: Knowledge points the student is missing
        adjacency_csv_path: Path to knowledge graph CSV.
        llm: LLM for Agent A and Agent B.
        extract_model: Optional local model for Extract Layer.
        use_ticket: True=E2 (ticket routing), False=E3 ablation (prompt-only).
        debug: Print debug info.

    Returns:
        {final_answer, trace: {steps, required_kps, missing_kps, ticket, ...}}
    """
    if adjacency_csv_path is None:
        adjacency_csv_path = DEFAULT_ADJ_PATH

    kp_vocab, adj_matrix = load_knowledge_graph(adjacency_csv_path)

    missing_with_ancestors = _get_all_ancestors(
        missing_kps, kp_vocab, adj_matrix.tolist()
    )
    mastered_kps = [kp for kp in kp_vocab if kp not in missing_with_ancestors]

    init_state = {
        "question": question.strip(),
        "kp_vocab": kp_vocab,
        "mastered_kps": mastered_kps,
        "extra": {
            "missing_kps": list(missing_with_ancestors),
            "specified_missing_kps": missing_kps,
        },
    }

    llm = adapt_llm(llm)
    app = build_graph(llm=llm, extract_model=extract_model, use_ticket=use_ticket)
    out = await app.ainvoke(init_state)

    final_answer = out.get("direct_answer") or out.get("tutoring_answer") or ""

    if debug:
        import json
        print("== ticket ==")
        print(json.dumps(out.get("routing_ticket", {}), ensure_ascii=False, indent=2))
        print("== clean_question ==")
        print(out.get("clean_question"))
        print("== final_answer ==")
        print(final_answer[:200])

    return {
        "final_answer": final_answer,
        "trace": {
            "steps": out.get("steps", []),
            "required_kps": out.get("required_kps", []),
            "missing_kps": out.get("missing_kps", []),
            "specified_missing_kps": missing_kps,
            "ticket": out.get("routing_ticket", {}),
            "extract_failed": out.get("extract_failed", False),
            "parse_failed": out.get("parse_failed", False),
            "clean_question": out.get("clean_question", ""),
        },
    }
