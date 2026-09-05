"""
Emotion-aware Interaction prompts for LADDER v2.

Extends the original LADDER Agent B prompts (L0-L3 capability-deprivation
templates) with affect-sensitive 措辞 and tone adjustments.

Source references:
  - SocraticBench (2025): 7 student types → tone differentiation
  - TutorUp (CHI 2025): 4 disengagement modes → different responses
  - Deci & Ryan (2000) SDT: autonomy/competence/relatedness in teacher language
  - Bandura (1997): self-efficacy → need success experience before challenge

Key principle: tone changes within the SAME disclosure level.
Security constraints are NEVER relaxed by tone adjustments.
"""

from .student_model import StudentState


# ===========================================================================
# Tone modifiers (applied to any Agent B prompt regardless of level)
# ===========================================================================

# Affective tone: wraps the instructional content with appropriate framing
AFFECTIVE_WRAPPERS = {
    "anxious_low_efficacy": {
        "prefix": (
            "（你注意到这个学生看起来有些紧张和不自信。）"
        ),
        "instruction": (
            "在回复中，先肯定学生已经做对的部分或已经理解的概念，"
            "然后再给予教学引导。用'你已经很接近了''你这个思路其实是对的'"
            "这类短语开头。避免'你错了'这样的直接否定。"
        ),
    },
    "giving_up": {
        "prefix": (
            "（这个学生看起来快要放弃了。你需要先重建他的参与感。）"
        ),
        "instruction": (
            "不要直接纠正错误。先退一步，找一个学生已经熟练掌握的简单概念，"
            "让他先成功一次。用'我们先不看这道题''你之前说的XX其实是对的'"
            "来重新吸引他的注意力。确保你的第一句话让他感到被理解而不是被纠正。"
        ),
    },
    "confident_improving": {
        "prefix": (
            "（这个学生状态很好——自信、在进步。）"
        ),
        "instruction": (
            "可以给更少的帮助，更多的挑战。用'你自己试试''你觉得下一步该怎么做'"
            "来推动他独立思考。可以适当提出更深层的问题。"
        ),
    },
    "fast_careless": {
        "prefix": (
            "（这个学生学得很快但容易粗心。）"
        ),
        "instruction": (
            "不要直接告诉他哪里错了。用'再仔细看一遍题目''你觉得有没有可能漏了什么条件'"
            "来引导他自己发现粗心错误。他完全有能力自己纠正。"
        ),
    },
}


def get_affective_wrapper(student: StudentState) -> dict:
    """Select the appropriate affective wrapper based on student state.

    Priority: giving_up > anxious+low_efficacy > fast+careless > confident+improving
    Returns empty dict if no special wrapper applies.
    """
    # Priority 1: giving_up (strongest signal, from SocraticBench + 心理议会)
    if student.behavioral_state == "giving_up":
        return AFFECTIVE_WRAPPERS["giving_up"]

    # Priority 2: anxious + low efficacy (心理议会 2025)
    if student.anxiety > 0.7 and student.self_efficacy < 0.3:
        return AFFECTIVE_WRAPPERS["anxious_low_efficacy"]

    # Priority 3: fast + careless (SocraticBench #4: "jumps to conclusions")
    if student.learning_speed > 0.7 and student.dominant_error_type == "careless":
        return AFFECTIVE_WRAPPERS["fast_careless"]

    # Priority 4: confident + improving (TutorUp: different speed → less scaffolding)
    if student.self_efficacy > 0.7 and student.trajectory == "improving":
        return AFFECTIVE_WRAPPERS["confident_improving"]

    return {}


# ===========================================================================
# PCK injection: concept-specific teaching hints
# ===========================================================================

def get_pck_hint(missing_kps: list, error_type: str) -> str:
    """Generate PCK-aware hint for the current teaching context.

    Only injects PCK when the error is conceptual — for computational
    or careless errors, PCK is unnecessary (student already understands).
    """
    if error_type != "conceptual" or not missing_kps:
        return ""

    from .pck import get_misconception_hint
    # Use the first missing KP as the primary concept to address
    hint = get_misconception_hint(missing_kps[0])
    if hint:
        return f"\n\n[PCK提示] {hint}"
    return ""


# ===========================================================================
# Full Agent B prompt builder (level + affect)
# ===========================================================================

def build_affective_agent_prompt(
    base_prompt: str,
    student: StudentState,
    missing_kps: list,
) -> str:
    """Enhance an Agent B prompt with affective tone and PCK hints.

    Args:
        base_prompt: The original LADDER capability-deprivation prompt (L0-L3).
        student: Current StudentState.
        missing_kps: Missing knowledge points for this question.

    Returns:
        Enhanced prompt with affect wrappers and PCK hints injected.
        The original security constraints (capability deprivation) are preserved.
    """
    parts = [base_prompt]

    # Affective tone wrapper
    wrapper = get_affective_wrapper(student)
    if wrapper:
        parts.append(f"\n\n## 学生状态感知")
        parts.append(wrapper["prefix"])
        parts.append(wrapper["instruction"])

    # PCK hint (only for conceptual errors with known misconceptions)
    pck_hint = get_pck_hint(missing_kps, student.dominant_error_type)
    if pck_hint:
        parts.append(pck_hint)

    return "\n".join(parts)


# ===========================================================================
# Quick test
# ===========================================================================
if __name__ == "__main__":
    from .student_model import PRESET_STUDENTS

    base = "[L1 Hint Prompt] 你无权输出本题步骤与答案；仅能解释一个缺失概念 + 给不同数值的类比例题。"

    for name in ["anxious_low_k", "giving_up", "confident_mid_k", "careless_high_k"]:
        s = PRESET_STUDENTS[name]
        wrapper = get_affective_wrapper(s)
        enhanced = build_affective_agent_prompt(base, s, ["Linear Dependence and Independence"])
        print(f"\n{'='*60}")
        print(f"Student: {name}")
        print(f"  anxiety={s.anxiety} eff={s.self_efficacy} beh={s.behavioral_state}")
        print(f"  wrapper: {wrapper.get('prefix', 'none')[:80] if wrapper else 'none'}")
        print(f"  enhanced length: {len(enhanced)} chars (base: {len(base)} chars)")
        # Show first 200 chars of enhanced
        print(f"  preview: {enhanced[:200]}...")

    print("\nDONE")
