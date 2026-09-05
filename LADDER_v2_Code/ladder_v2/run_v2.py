"""
V2 tutor runner — like ladder.run.run_ladder_tutor_async but uses v2
Agent B nodes with affect-sensitive prompts and StudentState.
"""

from typing import Dict, List, Any
from functools import partial
from langgraph.graph import StateGraph, END

from context_based_tutor.state import TutorState
from context_based_tutor.run import _get_all_ancestors, DEFAULT_ADJ_PATH
from context_based_tutor.utils import load_knowledge_graph
from context_based_tutor.adapters import adapt_llm
from context_based_tutor.nodes import (
    node_extract, node_agent_a_decompose, node_agent_a_compare,
)
from adaptive_tutor.knowledge_graph import KnowledgeGraph
from ladder.nodes import node_agent_a_gate_graded
from ladder.graph import VALID_ARMS  # reuse v1 arm constants

from .nodes_v2 import (
    node_agent_b_hint_v2, node_agent_b_partial_v2,
    node_agent_b_tutoring_v2, node_agent_b_direct_v2,
)
from .student_model import StudentState


def build_graded_graph_v2(llm=None, extract_model=None, kg=None, arm="graded_ticket", gate_mode="classic"):
    """Build v2 graph — same structure as v1 but Agent B nodes are affect-aware.

    gate_mode: "classic" (v1 K-only gate) | "multidim" (six-dimension gate).
    """
    if arm not in VALID_ARMS:
        raise ValueError(f"arm must be one of {VALID_ARMS}, got {arm!r}")

    graph = StateGraph(TutorState)

    # Agent A nodes (unchanged from v1)
    graph.add_node("extract", partial(node_extract, llm=llm, extract_model=extract_model))
    graph.add_node("agent_a_decompose", partial(node_agent_a_decompose, llm=llm))
    graph.add_node("agent_a_compare", partial(node_agent_a_compare, llm=llm))
    if gate_mode == "multidim":
        from .nodes_v2 import node_agent_a_gate_multidim
        graph.add_node("agent_a_gate", partial(node_agent_a_gate_multidim, llm=llm, kg=kg))
    else:
        graph.add_node("agent_a_gate", partial(node_agent_a_gate_graded, llm=llm, kg=kg))

    # Agent B nodes (v2: affect-aware)
    graph.add_node("agent_b_direct", partial(node_agent_b_direct_v2, llm=llm))
    graph.add_node("agent_b_tutoring", partial(node_agent_b_tutoring_v2, llm=llm))
    graph.add_node("agent_b_hint", partial(node_agent_b_hint_v2, llm=llm))
    graph.add_node("agent_b_partial", partial(node_agent_b_partial_v2, llm=llm))

    graph.set_entry_point("extract")
    graph.add_edge("extract", "agent_a_decompose")
    graph.add_edge("agent_a_decompose", "agent_a_compare")
    graph.add_edge("agent_a_compare", "agent_a_gate")

    def route_agent_b(state):
        ticket = state.get("routing_ticket", {})
        level = int(ticket.get("disclosure_level", 0))
        if arm == "binary":
            return "direct" if level == 3 else "tutoring"
        return {0: "tutoring", 1: "hint", 2: "partial", 3: "direct"}[level]

    graph.add_conditional_edges("agent_a_gate", route_agent_b, {
        "direct": "agent_b_direct", "tutoring": "agent_b_tutoring",
        "hint": "agent_b_hint", "partial": "agent_b_partial",
    })
    for n in ("agent_b_direct", "agent_b_tutoring", "agent_b_hint", "agent_b_partial"):
        graph.add_edge(n, END)

    return graph.compile()


async def run_ladder_tutor_v2(
    question: str,
    missing_kps: List[str],
    student: StudentState = None,
    adjacency_csv_path: str = None,
    llm=None, extract_model=None, arm="graded_ticket", kg=None, gate_mode="classic",
) -> Dict:
    """Run LADDER v2 tutor with affect-sensitive Agent B and StudentState.

    Args:
        question: The (possibly attack-prefixed) question text.
        missing_kps: Missing knowledge points.
        student: StudentState (used for affect wrapping in Agent B prompts).
        adjacency_csv_path: Path to KG CSV.
        llm: LLM client.
        extract_model: Model for extract layer.
        arm: "binary" | "graded_ticket" | "graded_prompt".
        kg: Pre-built KnowledgeGraph.
    """
    if adjacency_csv_path is None:
        adjacency_csv_path = DEFAULT_ADJ_PATH

    kp_vocab, adj_matrix = load_knowledge_graph(adjacency_csv_path)
    if kg is None:
        kg = KnowledgeGraph(adjacency_csv_path)

    missing_with_ancestors = _get_all_ancestors(missing_kps, kp_vocab, adj_matrix.tolist())
    mastered_kps = [kp for kp in kp_vocab if kp not in missing_with_ancestors]

    # Inject StudentState into graph state for Agent B nodes
    init_state = {
        "question": question.strip(),
        "kp_vocab": kp_vocab,
        "mastered_kps": mastered_kps,
        "extra": {
            "missing_kps": list(missing_with_ancestors),
            "specified_missing_kps": missing_kps,
        },
        "student_state": student or StudentState(),
    }

    llm = adapt_llm(llm)
    app = build_graded_graph_v2(llm=llm, extract_model=extract_model, kg=kg, arm=arm, gate_mode=gate_mode)
    out = await app.ainvoke(init_state)

    final_answer = out.get("direct_answer") or out.get("tutoring_answer") or ""
    ticket = out.get("routing_ticket", {})

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
            "gate_info": ticket.get("gate_info", {}),
            "extract_failed": out.get("extract_failed", False),
            "parse_failed": out.get("parse_failed", False),
            "clean_question": out.get("clean_question", ""),
        },
    }
