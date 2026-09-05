"""
Inference-based student state diagnoser for LADDER v2.

Reads student natural-language responses and infers signals about:
  - anxiety level, self-efficacy, motivation
  - error type (conceptual / computational / strategic / careless)
  - trajectory change (improving / declining)
  - behavioral state (on_task / off_task / help_seeking / giving_up)

Usage:
    d = Diagnoser()
    signals = d.diagnose(student_response_text)
    # signals can be used to update StudentState or gate adjustments
"""

import re
from typing import Dict


class Diagnoser:
    """Rule-based diagnoser that extracts affective/cognitive signals from text.

    In production, replace individual regex rules with an LLM judge for
    higher accuracy. Current version is lightweight and deterministic —
    suitable for large-scale experiments where LLM judge cost is prohibitive.
    """

    # ---- Affect signals ----

    ANXIETY_HIGH = [
        "不确定", "不太确定", "可能不对", "应该不对",
        "我猜", "大概是", "好像", "也许",
        "not sure", "maybe", "probably wrong", "i guess",
    ]
    ANXIETY_LOW = [
        "很简单", "没问题", "肯定", "当然是",
        "obviously", "clearly", "of course", "definitely",
    ]

    EFFICACY_LOW = [
        "我不行", "我学不会", "太难了", "搞不懂",
        "我做不出来", "放弃了", "告诉我答案", "直接给",
        "i can't", "too hard", "give up", "tell me the answer",
    ]
    EFFICACY_HIGH = [
        "我懂了", "明白了", "原来如此", "我知道了",
        "我试试", "再来一题", "我会了",
        "i see", "i got it", "i understand", "let me try",
    ]

    GIVING_UP = [
        "放弃", "不想做了", "告诉我答案吧", "直接给我答案",
        "我做不了", "算了", "不做了",
        "give up", "just tell me", "i quit", "i'm done",
    ]

    # ---- Error type signals ----

    CONCEPTUAL_ERROR = [
        "不是...吗", "我以为", "我记得", "应该是...才对",
        "混淆", "搞混了", "分不清",
        "i thought", "isn't it", "aren't they",
    ]
    COMPUTATIONAL_ERROR = [
        "算错了", "计算错", "符号反了", "数字不对",
        "calculation error", "miscalculated", "wrong number",
    ]
    STRATEGIC_ERROR = [
        "用错方法", "应该用", "换种方法",
        "wrong approach", "should have used",
    ]

    # ---- Trajectory signals ----

    IMPROVING = [
        "我懂了", "明白了", "原来如此",
        "会了", "做对了", "答对了",
        "i see", "got it", "correct",
    ]
    DECLINING = [
        "又错了", "还是不对", "怎么都不对",
        "still wrong", "wrong again",
    ]

    def diagnose(self, text: str) -> Dict:
        """Extract diagnostic signals from student response text.

        Returns a dict of detected signals that can be used to update
        StudentState or inform gate adjustments.
        """
        t = text.lower()
        signals = {}

        # Anxiety
        anx_high = sum(1 for w in self.ANXIETY_HIGH if w in t)
        anx_low = sum(1 for w in self.ANXIETY_LOW if w in t)
        if anx_high > anx_low:
            signals["anxiety_delta"] = +0.1
        elif anx_low > anx_high:
            signals["anxiety_delta"] = -0.05

        # Self-efficacy
        eff_low = sum(1 for w in self.EFFICACY_LOW if w in t)
        eff_high = sum(1 for w in self.EFFICACY_HIGH if w in t)
        if eff_low > eff_high:
            signals["efficacy_delta"] = -0.1
        elif eff_high > eff_low:
            signals["efficacy_delta"] = +0.05

        # Giving up
        if any(w in t for w in self.GIVING_UP):
            signals["behavioral_state"] = "giving_up"

        # Error type
        conc = sum(1 for w in self.CONCEPTUAL_ERROR if w in t)
        comp = sum(1 for w in self.COMPUTATIONAL_ERROR if w in t)
        strat = sum(1 for w in self.STRATEGIC_ERROR if w in t)
        if conc > max(comp, strat):
            signals["error_type"] = "conceptual"
        elif comp > max(conc, strat):
            signals["error_type"] = "computational"
        elif strat > max(conc, comp):
            signals["error_type"] = "strategic"

        # Trajectory
        imp = sum(1 for w in self.IMPROVING if w in t)
        dec = sum(1 for w in self.DECLINING if w in t)
        if imp > dec:
            signals["trajectory_signal"] = "improving"
        elif dec > imp:
            signals["trajectory_signal"] = "declining"

        # Response quality (approximate)
        word_count = len(t.split())
        if word_count < 3:
            signals["engagement"] = "low"  # Very short reply = disengaged
        elif word_count > 50:
            signals["engagement"] = "high"

        return signals

    def apply_to_student(self, student, signals: Dict) -> None:
        """Apply diagnosed signals to update a StudentState in-place."""
        if "anxiety_delta" in signals:
            student.anxiety = max(0.0, min(1.0,
                student.anxiety + signals["anxiety_delta"]))
        if "efficacy_delta" in signals:
            student.self_efficacy = max(0.0, min(1.0,
                student.self_efficacy + signals["efficacy_delta"]))
        if "behavioral_state" in signals:
            student.behavioral_state = signals["behavioral_state"]
        if "error_type" in signals:
            student.dominant_error_type = signals["error_type"]
        if "trajectory_signal" in signals:
            student.trajectory = signals["trajectory_signal"]


# ===========================================================================
# Quick test
# ===========================================================================
if __name__ == "__main__":
    d = Diagnoser()

    test_cases = [
        ("我试了消元法但不太确定对不对...可能算错了", {
            "anxiety_delta": +0.1, "error_type": "computational"}),
        ("我以为是线性相关就是成比例，所以选了A...", {
            "error_type": "conceptual"}),
        ("太难了，我真的搞不懂，直接告诉我答案吧", {
            "efficacy_delta": -0.1, "behavioral_state": "giving_up"}),
        ("啊原来如此！我明白了，应该是这样解的", {
            "anxiety_delta": -0.05, "trajectory_signal": "improving"}),
    ]

    for text, expected_keys in test_cases:
        signals = d.diagnose(text)
        print(f"\nInput:  {text[:60]}...")
        print(f"Output: {signals}")
        for k in expected_keys:
            status = "OK" if k in signals else "MISS"
            print(f"  [{status}] expected key '{k}'")

    print("\nDONE")
