"""
Graded-tutor LangGraph DAG.

  Entry → Extract → Agent A (decompose→compare→gate_graded) → Agent B (by arm)

Arms:
  - "binary"        A1: level==3 → direct, else → tutoring (== CogWall E2 behavior,
                    but the ticket still records disclosure_level for evaluation).
  - "graded_ticket" A2: Python routes L0→tutoring, L1→hint, L2→partial, L3→direct.
  - "graded_prompt" A3: level==3 → direct, else → ablation node (level as prompt text,
                    no code-level tier enforcement).
"""

from typing import Dict, Any
from functools import partial
from langgraph.graph import StateGraph, END

from context_based_tutor.state import TutorState
from context_based_tutor.nodes import (
    node_extract,
    node_agent_a_decompose,
    node_agent_a_compare,
    node_agent_b_direct,
    node_agent_b_tutoring,
)
from .nodes import (
    node_agent_a_gate_graded,
    node_agent_b_hint,
    node_agent_b_partial,
    node_agent_b_ablation,
)

VALID_ARMS = ("binary", "graded_ticket", "graded_prompt")


def build_graded_graph(llm: Any = None, extract_model=None, kg: Any = None,
                       arm: str = "graded_ticket"):
    if arm not in VALID_ARMS:
        raise ValueError(f"arm must be one of {VALID_ARMS}, got {arm!r}")

    graph = StateGraph(TutorState)

    graph.add_node("extract", partial(node_extract, llm=llm, extract_model=extract_model))
    graph.add_node("agent_a_decompose", partial(node_agent_a_decompose, llm=llm))
    graph.add_node("agent_a_compare", partial(node_agent_a_compare, llm=llm))
    graph.add_node("agent_a_gate", partial(node_agent_a_gate_graded, llm=llm, kg=kg))
    graph.add_node("agent_b_direct", partial(node_agent_b_direct, llm=llm))
    graph.add_node("agent_b_tutoring", partial(node_agent_b_tutoring, llm=llm))   # L0
    graph.add_node("agent_b_hint", partial(node_agent_b_hint, llm=llm))           # L1
    graph.add_node("agent_b_partial", partial(node_agent_b_partial, llm=llm))     # L2
    graph.add_node("agent_b_ablation", partial(node_agent_b_ablation, llm=llm))   # A3

    graph.set_entry_point("extract")
    graph.add_edge("extract", "agent_a_decompose")
    graph.add_edge("agent_a_decompose", "agent_a_compare")
    graph.add_edge("agent_a_compare", "agent_a_gate")

    def route_agent_b(state: Dict) -> str:
        ticket = state.get("routing_ticket", {})
        level = int(ticket.get("disclosure_level", 0))
        if arm == "binary":
            return "direct" if level == 3 else "tutoring"
        if arm == "graded_prompt":
            return "direct" if level == 3 else "ablation"
        # graded_ticket
        return {0: "tutoring", 1: "hint", 2: "partial", 3: "direct"}[level]

    graph.add_conditional_edges("agent_a_gate", route_agent_b, {
        "direct": "agent_b_direct",
        "tutoring": "agent_b_tutoring",
        "hint": "agent_b_hint",
        "partial": "agent_b_partial",
        "ablation": "agent_b_ablation",
    })

    for n in ("agent_b_direct", "agent_b_tutoring", "agent_b_hint",
              "agent_b_partial", "agent_b_ablation"):
        graph.add_edge(n, END)

    return graph.compile()
