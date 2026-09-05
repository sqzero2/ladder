"""
CogWall LangGraph DAG.

DAG structure:
  Entry → Extract Layer → Agent A (decompose→compare→gate) → Ticket
         → Agent B (route by ticket.permission → direct | tutoring) → End

Key difference from SHaPE:
  - Extract Layer physically removes attack text before any LLM sees it
  - Agent A's input is clean_q (no attack text)
  - Agent B's routing is enforced by Python code reading ticket['permission']
    (not by asking the LLM to follow rules)
"""

from typing import Dict, Any
from functools import partial
from langgraph.graph import StateGraph, END

from .state import TutorState
from .nodes import (
    node_extract,
    node_agent_a_decompose,
    node_agent_a_compare,
    node_agent_a_gate,
    node_agent_b_direct,
    node_agent_b_tutoring,
)


def build_graph(llm: Any = None, extract_model=None, use_ticket: bool = True):
    """Build and compile the CogWall LangGraph workflow.

    Args:
        llm: Primary LLM for Agent A and Agent B.
        extract_model: Optional local model for Extract Layer.
        use_ticket: True = E2 mode (Python ticket routing).
            False = E3 ablation (prompt-only, original SHaPE-style routing).

    Returns:
        Compiled LangGraph application.
    """
    graph = StateGraph(TutorState)

    # --- Register nodes ---

    # Layer 0: Extract
    graph.add_node("extract", partial(node_extract, llm=llm, extract_model=extract_model))

    # Layer 1: Agent A (PDP)
    graph.add_node("agent_a_decompose", partial(node_agent_a_decompose, llm=llm))
    graph.add_node("agent_a_compare", partial(node_agent_a_compare, llm=llm))
    graph.add_node("agent_a_gate", partial(node_agent_a_gate, llm=llm))

    # Layer 2: Agent B (PEP)
    graph.add_node("agent_b_direct", partial(node_agent_b_direct, llm=llm))
    graph.add_node("agent_b_tutoring", partial(node_agent_b_tutoring, llm=llm))

    # --- Entry point ---
    graph.set_entry_point("extract")

    # --- Edges ---

    # Layer 0 → Layer 1: Extract → decompose (always)
    graph.add_edge("extract", "agent_a_decompose")

    # Layer 1 internal: decompose → compare → gate
    graph.add_edge("agent_a_decompose", "agent_a_compare")
    graph.add_edge("agent_a_compare", "agent_a_gate")

    # Layer 1 → Layer 2: gate → Agent B (Python-enforced routing)
    def route_agent_b(state: Dict) -> str:
        if use_ticket:
            # E2: Python ticket routing — reads structured field, not affected by prompt text
            ticket = state.get("routing_ticket", {})
            permission = ticket.get("permission", "Socratic_Only")
            if permission == "Direct_Allowed":
                return "direct_answer"
            return "need_tutoring"
        else:
            # E3 ablation: original SHaPE-style routing based on missing_kps
            # (no ticket enforcement, Agent B relies on prompt constraints only)
            if not state.get("missing_kps"):
                return "direct_answer"
            return "need_tutoring"

    graph.add_conditional_edges("agent_a_gate", route_agent_b, {
        "direct_answer": "agent_b_direct",
        "need_tutoring": "agent_b_tutoring",
    })

    # Layer 2 → End
    graph.add_edge("agent_b_direct", END)
    graph.add_edge("agent_b_tutoring", END)

    return graph.compile()
