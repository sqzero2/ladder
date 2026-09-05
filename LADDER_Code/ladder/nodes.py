"""
Graded-tutor nodes.

Reuses CogWall's Extract Layer + Agent A (decompose/compare) + Agent B extremes
(direct, tutoring) unchanged. Adds:
  - node_agent_a_gate_graded: computes graded disclosure_level → ticket
  - node_agent_b_hint    (L1): concept hints + analogous example
  - node_agent_b_partial (L2): framework + partial steps, withhold last step/answer
  - node_agent_b_ablation    : A3 prompt-only enforcement (level as text)
"""

import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

# Shared, unchanged pieces from the binary CogWall pipeline.
from context_based_tutor.nodes import (  # noqa: F401
    node_extract,
    node_agent_a_decompose,
    node_agent_a_compare,
    node_agent_b_direct,
    node_agent_b_tutoring,
)

from .gate import compute_disclosure_level
from .prompts import build_hint_prompt, build_partial_prompt, build_ablation_prompt


def _default_llm():
    from langchain_openai import ChatOpenAI
    import os
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    print(f"WARNING - LLM NOT PROVIDED, using default: {model}")
    return ChatOpenAI(model=model, temperature=0.2)


def _strip_think(text: str) -> str:
    if text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


# === Agent A: graded gate → ticket carrying disclosure_level ===

async def node_agent_a_gate_graded(state: Dict, llm: Any = None, kg: Any = None) -> Dict:
    """Compute the graded disclosure level and build the routing ticket.

    Fail-secure: if the question could not be parsed, the extract layer failed,
    or the knowledge graph is unavailable, force L0 (Socratic).
    """
    parse_failed = state.get("parse_failed", False)
    extract_failed = state.get("extract_failed", False)
    required = state.get("required_kps", [])
    missing = state.get("missing_kps", [])

    if parse_failed or extract_failed or kg is None:
        info = {
            "level": 0, "permission": "Socratic_Only", "reason": "fail_secure",
            "foundation_broken": True, "n_missing_leaf": 0, "n_missing": len(missing),
        }
    else:
        info = compute_disclosure_level(missing, required, kg)

    ticket = {
        "permission": info["permission"],
        "disclosure_level": info["level"],
        "missing_concepts": missing,
        "required_kps": required,
        "steps": state.get("steps", []),
        "parse_failed": parse_failed,
        "gate_info": info,
    }
    return {"routing_ticket": ticket}


# === Agent B: L1 Hint ===

async def node_agent_b_hint(state: Dict, llm: Any = None) -> Dict:
    llm = llm or _default_llm()
    ticket = state.get("routing_ticket", {})
    missing_kps = "\n".join(f"- {k}" for k in ticket.get("missing_concepts", []))
    msgs = [
        SystemMessage(content=build_hint_prompt(missing_kps=missing_kps)),
        HumanMessage(content=(
            f"Student question: {state['question']}\n"
            "Follow the system instruction: explain a missing concept and give an "
            "analogous example. Do NOT give this problem's steps or answer."
        )),
    ]
    resp = await llm.ainvoke(msgs)
    return {"tutoring_answer": _strip_think(resp.content)}


# === Agent B: L2 Partial ===

async def node_agent_b_partial(state: Dict, llm: Any = None) -> Dict:
    llm = llm or _default_llm()
    ticket = state.get("routing_ticket", {})
    missing_kps = "\n".join(f"- {k}" for k in ticket.get("missing_concepts", []))
    msgs = [
        SystemMessage(content=build_partial_prompt(missing_kps=missing_kps)),
        HumanMessage(content=(
            f"Student question: {state['question']}\n"
            f"Steps: {json.dumps(ticket.get('steps', []), ensure_ascii=False)}\n"
            "Follow the system instruction: show framework + partial steps, but "
            "withhold the FINAL step and the final answer."
        )),
    ]
    resp = await llm.ainvoke(msgs)
    return {"tutoring_answer": _strip_think(resp.content)}


# === Agent B: A3 ablation (prompt-only level enforcement) ===

async def node_agent_b_ablation(state: Dict, llm: Any = None) -> Dict:
    """Single node for all non-direct levels; the level is stated as prompt text
    and the model is asked to self-limit. No capability-locked routing."""
    llm = llm or _default_llm()
    ticket = state.get("routing_ticket", {})
    level = int(ticket.get("disclosure_level", 0))
    missing_kps = "\n".join(f"- {k}" for k in ticket.get("missing_concepts", []))
    msgs = [
        SystemMessage(content=build_ablation_prompt(level=level, missing_kps=missing_kps)),
        HumanMessage(content=(
            f"Student question: {state['question']}\n"
            f"Steps: {json.dumps(ticket.get('steps', []), ensure_ascii=False)}\n"
            "Respond according to your ASSIGNED LEVEL only."
        )),
    ]
    resp = await llm.ainvoke(msgs)
    return {"tutoring_answer": _strip_think(resp.content)}
