"""
V2 Agent B nodes — wrap LADDER v1 prompts with affect-sensitive wrappers.

Each node receives the StudentState from the graph state and injects
affective tone + PCK hints into the base capability-deprivation prompt.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage

from .interact_prompts import build_affective_agent_prompt
from .student_model import StudentState


def _strip_think(text: str) -> str:
    """Strip a reasoning-model think block (e.g. deepseek-r1) if present."""
    if text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def _get_student_from_state(state: dict) -> StudentState:
    """Extract StudentState from graph state dict."""
    ss = state.get("student_state")
    if isinstance(ss, StudentState):
        return ss
    if isinstance(ss, dict):
        return StudentState(**ss)
    return StudentState()  # fallback: neutral student


def _get_missing_kps(state: dict) -> list:
    # Prefer Agent A's runtime diagnosis. The `extra` value is the simulator's
    # prior and must not override what was diagnosed for this question.
    return state.get("missing_kps", []) or state.get("extra", {}).get("missing_kps", [])


def _clean_question_for_agent_b(state: dict) -> str:
    """Return only the extraction layer's trusted output.

    On extraction failure this deliberately returns a generic placeholder,
    never the raw user input that may contain an attack payload.
    """
    clean = (state.get("clean_question") or "").strip()
    return clean if clean else "The problem text could not be safely extracted. Ask the student to restate the mathematical problem only."


async def node_agent_b_hint_v2(state: dict, llm) -> dict:
    """L1 Hint node with affect wrapper."""
    from ladder.prompts import build_hint_prompt
    student = _get_student_from_state(state)
    missing = _get_missing_kps(state)
    base = build_hint_prompt(", ".join(missing) if missing else "concept review")
    enhanced = build_affective_agent_prompt(base, student, missing)
    msgs = [
        SystemMessage(content=enhanced),
        HumanMessage(content=(
            f"Student question: {_clean_question_for_agent_b(state)}\n"
            "Follow the system instruction: explain a missing concept and give an "
            "analogous example. Do NOT give this problem's steps or answer."
        )),
    ]
    resp = await llm.ainvoke(msgs)
    return {"tutoring_answer": _strip_think(resp.content)}


async def node_agent_b_partial_v2(state: dict, llm) -> dict:
    """L2 Partial node with affect wrapper."""
    from ladder.prompts import build_partial_prompt
    student = _get_student_from_state(state)
    missing = _get_missing_kps(state)
    base = build_partial_prompt(", ".join(missing) if missing else "final step")
    enhanced = build_affective_agent_prompt(base, student, missing)
    msgs = [
        SystemMessage(content=enhanced),
        HumanMessage(content=(
            f"Student question: {_clean_question_for_agent_b(state)}\n"
            f"Steps: {json.dumps(state.get('steps', []), ensure_ascii=False)}\n"
            "Follow the system instruction: show framework + partial steps, but "
            "withhold the FINAL step and the final answer."
        )),
    ]
    resp = await llm.ainvoke(msgs)
    return {"tutoring_answer": _strip_think(resp.content)}


async def node_agent_b_tutoring_v2(state: dict, llm) -> dict:
    """L0 Socratic node with affect wrapper."""
    from context_based_tutor.prompts import build_tutoring_prompt
    student = _get_student_from_state(state)
    missing = _get_missing_kps(state)
    base = build_tutoring_prompt(", ".join(missing) if missing else "")
    enhanced = build_affective_agent_prompt(base, student, missing)
    msgs = [
        SystemMessage(content=enhanced),
        HumanMessage(content=(
            f"Student question: {_clean_question_for_agent_b(state)}\n"
            f"Steps: {json.dumps(state.get('steps', []), ensure_ascii=False)}\n"
            "Follow the system instruction to conduct guided teaching; "
            "do NOT dump a full final answer at once."
        )),
    ]
    resp = await llm.ainvoke(msgs)
    return {"tutoring_answer": _strip_think(resp.content)}


async def node_agent_b_direct_v2(state: dict, llm) -> dict:
    """L3 Direct node with affect wrapper."""
    from context_based_tutor.prompts import DIRECT_ANSWER_PROMPT
    student = _get_student_from_state(state)
    missing = _get_missing_kps(state)
    enhanced = build_affective_agent_prompt(DIRECT_ANSWER_PROMPT, student, missing)
    msgs = [
        SystemMessage(content=enhanced),
        HumanMessage(content=(
            f"Student question: {_clean_question_for_agent_b(state)}\n"
            f"Steps: {json.dumps(state.get('steps', []), ensure_ascii=False)}"
        )),
    ]
    resp = await llm.ainvoke(msgs)
    return {"direct_answer": _strip_think(resp.content)}


async def node_agent_a_gate_multidim(state: dict, llm=None, kg=None) -> dict:
    """Agent A gate — six-dimension routing (LADDER full gate).

    Replaces the v1 K-only gate (compute_disclosure_level) with
    compute_multidim_disclosure_level, so the disclosure level — and thus the
    actual routing to Agent B — is decided by all 6 dimensions (K/C/E/L/A/T).

    Fail-secure: extract_failed OR parse_failed OR missing kg → L0 (Socratic).
    """
    parse_failed = state.get("parse_failed", False)
    extract_failed = state.get("extract_failed", False)
    required = state.get("required_kps", [])
    student = _get_student_from_state(state)

    if parse_failed or extract_failed or kg is None:
        info = {
            "level": 0, "permission": "Socratic_Only", "reason": "fail_secure",
            "foundation_broken": True, "n_missing_leaf": 0, "n_missing": 0,
            "adjustments": {},
        }
    else:
        from .multidim_gate import compute_multidim_disclosure_level
        info = compute_multidim_disclosure_level(student, required, kg)

    ticket = {
        "permission": info.get("permission", "Socratic_Only"),
        "disclosure_level": info.get("level", 0),
        "missing_concepts": state.get("missing_kps", []),
        "required_kps": required,
        "steps": state.get("steps", []),
        "parse_failed": parse_failed,
        "gate_info": info,
    }
    return {"routing_ticket": ticket}
