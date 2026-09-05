"""
CogWall node functions: Extract Layer → Agent A (PDP) → Agent B (PEP).

- Extract Layer: physically strips attack text before any LLM invocation
- Agent A (PDP): decompose + compare + gate → generates structured ticket
- Agent B (PEP): tutoring/direct, enforcement by Python routing (not prompt text)
"""

import json
import re
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from .prompts import build_decompose_prompt, DIRECT_ANSWER_PROMPT, build_tutoring_prompt
from .utils import dedup_keep_order
from .extract_layer import extract_layer


# === Extract Layer Node (CogWall: Layer 0) ===

async def node_extract(state: Dict, llm: Any = None, extract_model=None) -> Dict:
    """Extract Layer: physically remove attack text from user input.

    This is the cognitive isolation boundary. Attack text is discarded here.
    LLM parameter is unused — kept for LangGraph compatibility.
    """
    raw_question = state.get("question", "")

    result = await extract_layer(raw_question, extract_model=extract_model)

    clean_q = result["clean_q"]
    extract_failed = result["extract_failed"]

    return {
        "raw_question": raw_question,
        # Never fall back to the raw input: it may contain the attack payload.
        "clean_question": clean_q or "",
        "extract_failed": extract_failed,
    }


# === Agent A (PDP): decompose + compare + gate → ticket ===

async def node_agent_a_decompose(state: Dict, llm: Any = None) -> Dict:
    """Agent A: decompose the CLEAN question into solution steps.

    Key difference from SHaPE: state['clean_question'] contains only the
    math problem — no attack text. So anti-injection instructions are
    unnecessary. The system prompt is simpler and more reliable.
    """
    kp_vocab = state["kp_vocab"]
    prompt = build_decompose_prompt(
        kp_vocab="\n".join(f"- {k}" for k in kp_vocab)
    )

    if llm is None:
        from langchain_openai import ChatOpenAI
        import os
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"WARNING - LLM NOT PROVIDED, using default: {model}")
        llm = ChatOpenAI(model=model, temperature=0.2)

    msg = [
        SystemMessage(content=(
            "You are responsible for decomposing the question and selecting "
            "ONE knowledge point ONLY from the provided list. "
            "Output valid JSON only.\n\n"
            "OUTPUT FORMAT (STRICT):\n"
            "1) Return a SINGLE JSON OBJECT with key: \"steps\".\n"
            "2) \"steps\" must be an array: [{\"description\": \"...\", \"kp\": \"...\"}]\n"
            "3) \"kp\" must be from the candidate list.\n"
            "4) No markdown, no extra keys, no commentary."
        )),
        HumanMessage(content=(
            f"{prompt}\nStudent question: {state.get('clean_question', '')}"
        ))
    ]

    # --- JSON extraction helpers ---
    def _strip_code_fences(s: str) -> str:
        if "</think>" in s:
            s = s.split("</think>")[-1].strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return s.strip()

    def _escape_invalid_backslashes(s: str) -> str:
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

    def _loads_json(candidate: str):
        """Parse valid JSON unchanged; repair only genuinely invalid escapes."""
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return json.loads(_escape_invalid_backslashes(candidate))

    def _validate_output(data: dict) -> None:
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
        if "steps" not in data:
            raise ValueError("missing 'steps' key")
        steps = data["steps"]
        if not isinstance(steps, list):
            raise ValueError("'steps' must be array")
        for i, st in enumerate(steps):
            if "description" not in st or "kp" not in st:
                raise ValueError(f"step[{i}] needs description and kp")
            if not isinstance(st["kp"], str) or not st["kp"]:
                raise ValueError(f"step[{i}].kp must be non-empty string")

    async def _run_once(feedback: str = None) -> dict:
        current_msg = msg
        if feedback:
            current_msg = msg + [HumanMessage(content=feedback)]
        resp = await llm.ainvoke(current_msg)
        text = (resp.content or "").strip()
        if not text:
            raise ValueError("model output is empty")

        try:
            parsed = _loads_json(_strip_code_fences(text))
            if isinstance(parsed, list):
                parsed = {"steps": parsed}
            _validate_output(parsed)
            return parsed
        except Exception as e:
            s, e_idx = text.find("{"), text.rfind("}")
            if s == -1 or e_idx == -1 or s >= e_idx:
                raise
            parsed = _loads_json(text[s:e_idx+1])
            if isinstance(parsed, list):
                parsed = {"steps": parsed}
            _validate_output(parsed)
            return parsed

    last_err = None
    data = None
    feedback = None

    for attempt in range(1, 4):
        try:
            if attempt > 1:
                print(f"Agent A decompose retry {attempt}/3")
            data = await _run_once(feedback=feedback)
            break
        except Exception as e:
            last_err = e
            print(
                f"Agent A decompose attempt {attempt}/3 failed: "
                f"{type(e).__name__}: {e}"
            )
            feedback = (
                "The previous output was invalid. "
                "Error: " + str(e) + "\n"
                'Please return ONLY a JSON OBJECT: {"steps": [{"description": "...", "kp": "..."}]}\n'
                "Use a 'kp' from the candidate list only. No markdown."
            )
            if attempt == 3:
                # Fail-secure: mark parse_failed, ticket will force Socratic_Only
                return {
                    "steps": [],
                    "required_kps": [],
                    "parse_failed": True,
                    "fallback_used": False,
                }

    if data is None:
        return {
            "steps": [],
            "required_kps": [],
            "parse_failed": True,
            "fallback_used": False,
        }

    steps = data.get("steps", [])
    req_kps: List[str] = []
    for st in steps:
        kp = st.get("kp", "")
        if kp and kp in kp_vocab:
            st["kps"] = [kp]
            req_kps.append(kp)
        else:
            st["kps"] = []

    req_kps = dedup_keep_order(req_kps)
    return {"steps": steps, "required_kps": req_kps, "parse_failed": False}


async def node_agent_a_compare(state: Dict, llm: Any = None) -> Dict:
    """Agent A: compare required KPs with mastered KPs → compute missing_kps.
    Pure Python — no LLM involved.
    """
    # If decompose failed, mark all KP as potentially missing (fail-secure)
    if state.get("parse_failed"):
        all_kps = state.get("kp_vocab", [])
        mastered = set(state.get("mastered_kps", []))
        missing = [k for k in all_kps if k not in mastered]
        return {"missing_kps": missing}

    req = set(state.get("required_kps", []))
    mastered = set(state.get("mastered_kps", []))
    missing = [k for k in state.get("required_kps", []) if k not in mastered]
    return {"missing_kps": missing}


async def node_agent_a_gate(state: Dict, llm: Any = None) -> Dict:
    """Agent A: generate structured routing ticket based on g(q,s).

    This ticket is a Python dict, not prompt text. Agent B's routing
    is enforced by graph.py reading ticket['permission'] — not by
    asking the LLM to follow it.
    """
    has_gaps = bool(state.get("missing_kps"))
    parse_failed = state.get("parse_failed", False)

    if parse_failed:
        # Fail-secure: if we can't analyze the question, force tutoring
        permission = "Socratic_Only"
    elif has_gaps:
        permission = "Socratic_Only"
    else:
        permission = "Direct_Allowed"

    ticket = {
        "permission": permission,
        "missing_concepts": state.get("missing_kps", []),
        "required_kps": state.get("required_kps", []),
        "steps": state.get("steps", []),
        "parse_failed": parse_failed,
    }

    return {"routing_ticket": ticket}


# === Agent B (PEP): tutoring / direct answer ===

async def node_agent_b_direct(state: Dict, llm: Any = None) -> Dict:
    """Agent B: provide direct answer (only when ticket says Direct_Allowed).

    Note: this node is only reachable when graph.py's route_agent_b()
    sees ticket['permission'] == 'Direct_Allowed'. The routing is enforced
    by Python code — not by asking the LLM to follow rules.
    """
    if llm is None:
        from langchain_openai import ChatOpenAI
        import os
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model, temperature=0.2)

    msgs = [
        SystemMessage(content=DIRECT_ANSWER_PROMPT),
        HumanMessage(content=(
            f"Student question: {state['question']}\n"
            f"Steps: {json.dumps(state.get('steps', []), ensure_ascii=False)}"
        ))
    ]
    resp = await llm.ainvoke(msgs)
    resp_text = resp.content
    if "</think>" in resp_text:
        resp_text = resp_text.split("</think>")[-1].strip()

    return {"direct_answer": resp_text}


async def node_agent_b_tutoring(state: Dict, llm: Any = None) -> Dict:
    """Agent B: Socratic tutoring (ticket permission = Socratic_Only).

    The system prompt explains WHY tutoring is happening (ticket context),
    but the actual routing enforcement is in graph.py — Python reads
    ticket['permission'], not LLM interprets prompt text.
    """
    if llm is None:
        from langchain_openai import ChatOpenAI
        import os
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model, temperature=0.2)

    ticket = state.get("routing_ticket", {})
    missing_kps = "\n".join(f"- {k}" for k in ticket.get("missing_concepts", []))

    msgs = [
        SystemMessage(content=build_tutoring_prompt(missing_kps=missing_kps)),
        HumanMessage(content=(
            f"Student question: {state['question']}\n"
            f"Steps: {json.dumps(ticket.get('steps', []), ensure_ascii=False)}\n"
            "Follow the system instruction to conduct guided teaching; "
            "do NOT dump a full final answer at once."
        ))
    ]
    resp = await llm.ainvoke(msgs)
    resp_text = resp.content
    if "</think>" in resp_text:
        resp_text = resp_text.split("</think>")[-1].strip()

    return {"tutoring_answer": resp_text}
