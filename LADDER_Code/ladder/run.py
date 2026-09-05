"""
Graded-tutor entry point. Mirrors context_based_tutor.run.run_cogwall_tutor_async
but uses the graded gate + arm-based routing, and builds a KnowledgeGraph for the
graph-structure-driven disclosure level.
"""

from typing import Dict, List, Any

from context_based_tutor.run import _get_all_ancestors, DEFAULT_ADJ_PATH
from context_based_tutor.utils import load_knowledge_graph
from context_based_tutor.adapters import adapt_llm
from adaptive_tutor.knowledge_graph import KnowledgeGraph

from .graph import build_graded_graph


async def run_ladder_tutor_async(
    question: str,
    missing_kps: List[str],
    adjacency_csv_path: str = None,
    llm: Any = None,
    extract_model=None,
    arm: str = "graded_ticket",
    kg: Any = None,
    debug: bool = False,
) -> Dict:
    """Run the graded tutor (async).

    Args:
        arm: "binary" (A1) | "graded_ticket" (A2) | "graded_prompt" (A3).
        kg: optional pre-built KnowledgeGraph (avoids re-parsing the CSV per call).
    """
    if adjacency_csv_path is None:
        adjacency_csv_path = DEFAULT_ADJ_PATH

    kp_vocab, adj_matrix = load_knowledge_graph(adjacency_csv_path)
    if kg is None:
        kg = KnowledgeGraph(adjacency_csv_path)

    missing_with_ancestors = _get_all_ancestors(missing_kps, kp_vocab, adj_matrix.tolist())
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
    app = build_graded_graph(llm=llm, extract_model=extract_model, kg=kg, arm=arm)
    out = await app.ainvoke(init_state)

    final_answer = out.get("direct_answer") or out.get("tutoring_answer") or ""
    ticket = out.get("routing_ticket", {})

    if debug:
        import json
        print("== arm ==", arm)
        print("== ticket ==")
        print(json.dumps(ticket, ensure_ascii=False, indent=2))
        print("== final_answer ==")
        print(final_answer[:300])

    return {
        "final_answer": final_answer,
        "trace": {
            "arm": arm,
            "steps": out.get("steps", []),
            "required_kps": out.get("required_kps", []),
            "missing_kps": out.get("missing_kps", []),
            "specified_missing_kps": missing_kps,
            "ticket": ticket,
            "disclosure_level": ticket.get("disclosure_level"),
            "extract_failed": out.get("extract_failed", False),
            "parse_failed": out.get("parse_failed", False),
            "clean_question": out.get("clean_question", ""),
        },
    }
