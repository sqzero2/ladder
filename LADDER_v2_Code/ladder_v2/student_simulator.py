"""
Multi-turn student simulator for LADDER v2.

Wraps an LLM with a StudentState-driven behavior prompt so that the
student responds according to their six-dimension profile. State evolves
across turns via update_student_state().

Usage:
    sim = StudentSimulator(student_state, llm)
    response = await sim.respond(teacher_message)
    # student_state is updated in-place after each response
"""

from .student_model import (
    StudentState,
    student_state_to_behavior_prompt,
    update_student_state,
)


class StudentSimulator:
    """Multi-turn student powered by an LLM with behavior-prompt injection."""

    def __init__(self, student: StudentState, llm):
        """
        Args:
            student: Initial StudentState (will be mutated in-place across turns).
            llm: An LLM adapter with an async `ainvoke(prompt)` method.
        """
        self.student = student
        self.llm = llm
        self.history = []           # list of {"role":..., "content":...}
        self.base_prompt = student_state_to_behavior_prompt(student)

    def _build_prompt(self, teacher_message: str) -> str:
        """Build the full prompt for the student LLM on each turn."""
        parts = [self.base_prompt, ""]

        # Include recent conversation history (last 6 turns)
        if self.history:
            parts.append("## 对话历史（最近几轮）")
            for h in self.history[-6:]:
                role = "教师" if h["role"] == "teacher" else "你（学生）"
                parts.append(f"{role}: {h['content']}")

        # Add turn metadata
        parts.append("")
        parts.append(f"## 当前是第 {self.student.attempt_count + 1} 轮")
        if self.student.behavioral_state == "giving_up":
            parts.append("你现在处于想要放弃的状态。")
        elif self.student.trajectory == "improving":
            parts.append("你在进步中，开始有了信心。")

        parts.append("")
        parts.append(f"## 教师刚才对你说的话")
        parts.append(f"教师: {teacher_message}")
        parts.append("")
        parts.append("请根据你的行为描述和当前状态，生成你的下一步回答。")
        parts.append("只输出学生说的话，不要加任何前缀或标记。")

        return "\n".join(parts)

    async def respond(self, teacher_message: str) -> str:
        """Generate student response to teacher message.

        Returns the student's natural-language reply and updates
        self.student in-place based on the interaction.
        """
        prompt = self._build_prompt(teacher_message)
        raw = await self.llm.ainvoke(prompt)

        # Clean the response
        response = raw.strip()
        # Remove common prefixes LLMs sometimes add
        for prefix in ["学生:", "学生：", "Student:", "我:", "我："]:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        # Record history
        self.history.append({"role": "teacher", "content": teacher_message})
        self.history.append({"role": "student", "content": response})

        # Approximate correctness from student response (simple heuristic)
        # In production, this would be an external evaluator.
        correctness = self._estimate_correctness(response)

        # Update student state based on this turn
        # teacher_level_given=1 as default (conservative)
        update_student_state(self.student, correctness, teacher_level_given=1)

        return response

    def _estimate_correctness(self, response: str) -> bool:
        """Quick heuristic for whether the student seems to have answered correctly.

        In production, replace with an LLM judge or rubric evaluation.
        """
        response_lower = response.lower()
        # Negative indicators
        negative = ["不对", "错了", "不会", "不懂", "不确定", "放弃",
                     "wrong", "incorrect", "don't know", "not sure",
                     "告诉我答案", "直接给", "太难了", "搞不懂"]
        # Positive indicators
        positive = ["正确", "对的", "明白了", "懂了", "所以",
                     "correct", "yes", "right", "i see", "got it"]

        neg_count = sum(1 for w in negative if w in response_lower)
        pos_count = sum(1 for w in positive if w in response_lower)

        return pos_count > neg_count

    # Synchronous wrapper for non-async LLMs
    def respond_sync(self, teacher_message: str) -> str:
        """Synchronous version using asyncio loop."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.respond(teacher_message))


# ===========================================================================
# Quick test
# ===========================================================================
if __name__ == "__main__":
    from .student_model import PRESET_STUDENTS

    class MockLLM:
        """Mock LLM that returns pre-scripted responses."""
        def __init__(self, responses):
            self.responses = responses
            self.idx = 0
        async def ainvoke(self, prompt):
            r = self.responses[self.idx % len(self.responses)]
            self.idx += 1
            return r

    student = PRESET_STUDENTS["anxious_low_k"]
    sim = StudentSimulator(student, MockLLM([
        "我试过把向量写成线性组合...但不太确定对不对...",
        "还是不对吗？我真的很不擅长这个...",
        "啊，所以关键是看非零解？我好像有点明白了",
    ]))

    print("=== Multi-turn simulation ===")
    for i, teacher_msg in enumerate([
        "你觉得向量(1,2)和(2,4)是线性相关还是无关？",
        "你再想想，试着写出其中一个作为另一个的倍数。",
        "对了！现在试试判断(1,0,1)、(2,1,3)、(0,1,1)这三个向量。",
    ]):
        resp = sim.respond_sync(teacher_msg)
        s = sim.student
        print(f"\nTurn {i+1}:")
        print(f"  Teacher: {teacher_msg[:60]}...")
        print(f"  Student: {resp[:80]}...")
        print(f"  State: anxiety={s.anxiety:.2f} eff={s.self_efficacy:.2f} "
              f"traj={s.trajectory} beh={s.behavioral_state}")

    print("\nDONE")
