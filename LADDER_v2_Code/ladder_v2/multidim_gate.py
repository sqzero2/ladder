"""
Multi-dimensional disclosure gate for LADDER v2.

Every coefficient in this module is backed by an empirical finding or
theoretical framework from the literature. See inline citations.

Architecture (4 layers):
  Layer 1: KG structural base (LADDER v1 logic, self-contained)
  Layer 2: K refinement
  Layer 3: C / E / L / A / T independent corrections
  Layer 4: Cross-dimension interaction corrections
"""

from typing import Dict, List, Any
from .student_model import StudentState


# ===========================================================================
# Layer 1: KG structural base (inlined from LADDER v1 gate.py)
# ===========================================================================

LEVEL_PERMISSION = {0: "Socratic_Only", 1: "Hint_Allowed",
                    2: "Partial_Allowed", 3: "Direct_Allowed"}


def _transitive_prereqs(kp: str, kg: Any) -> set:
    closure = set(kg.get_all_prerequisites([kp]))
    closure.discard(kp)
    return closure


def _kg_structural_level(
    missing_kps: List[str],
    required_kps: List[str],
    kg: Any,
) -> Dict[str, Any]:
    """Layer 1: KG structural base.

    Rationale: UCO (2025) shows scaffolding should match ZPD proximity.
    foundation_broken = student doesn't have the conceptual prerequisites
    to benefit from intermediate help → must stay at L0 (outside ZPD).
    n_missing_leaf = "how close to mastery" → L2 when 1 step away,
    L1 when multiple independent gaps remain.
    """
    required_set = set(required_kps)
    missing_set = set(missing_kps) & required_set if required_set else set(missing_kps)
    n_missing = len(missing_set)

    if n_missing == 0:
        return {"level": 3, "permission": "Direct_Allowed", "reason": "no_gaps",
                "foundation_broken": False, "n_missing_leaf": 0, "n_missing": 0}
    if required_set and missing_set == required_set:
        return {"level": 0, "permission": "Socratic_Only", "reason": "all_required_missing",
                "foundation_broken": False, "n_missing_leaf": n_missing, "n_missing": n_missing}
    try:
        valid = kg.is_valid_missing_combination(list(required_set), list(missing_set))
    except Exception:
        # A consistency check that cannot run is not evidence of consistency.
        valid = False
    if not valid:
        return {"level": 0, "permission": "Socratic_Only", "reason": "inconsistent_state",
                "foundation_broken": True, "n_missing_leaf": 0, "n_missing": n_missing}

    leaves = []
    for m in missing_set:
        prereqs = _transitive_prereqs(m, kg)
        if not (prereqs & missing_set):
            leaves.append(m)
    n_leaf = len(leaves)
    foundation_broken = n_leaf < n_missing

    if foundation_broken:
        return {"level": 0, "permission": "Socratic_Only", "reason": "foundation_broken",
                "foundation_broken": True, "n_missing_leaf": n_leaf, "n_missing": n_missing}
    if n_leaf == 1:
        return {"level": 2, "permission": "Partial_Allowed", "reason": "single_leaf_gap",
                "foundation_broken": False, "n_missing_leaf": 1, "n_missing": n_missing}
    return {"level": 1, "permission": "Hint_Allowed", "reason": "multi_leaf_gap",
            "foundation_broken": False, "n_missing_leaf": n_leaf, "n_missing": n_missing}


# ===========================================================================
# Layer 2: K refinement
# ===========================================================================

def _k_refinement(base_level: int, student: StudentState,
                  required_kps: List[str]) -> int:
    """Refine K beyond binary missing/not-missing.

    Misconception → force L0:
      VanLehn (1990): misconceptions require "repair dialogue" — the
      student must first recognize their concept is wrong before new
      instruction can take hold. Giving any procedural help (L1/L2)
      before resolving the misconception is wasted.
      Wang et al. (ACL 2025): 认知原型中概念性错误需要先打破后重建。

    Mastery depth → adjust:
      PEARL (2026) uses continuous mastery (0-1) not binary. Their
      controllable student simulator shows that 掌握度 < 0.3 的学生
      对 >L1 的脚手架无法有效利用。
    """
    level = base_level

    if student.misconceptions:
        if set(student.misconceptions) & set(required_kps):
            return 0  # VanLehn 1990: must resolve misconception first

    if required_kps:
        missing_mastery = [kp for kp in required_kps if kp not in student.kp_mastery]
        if missing_mastery:
            raise ValueError(
                "Student mastery profile does not cover required KPs: "
                + ", ".join(missing_mastery)
            )
        avg_mastery = sum(student.kp_mastery[kp] for kp in required_kps) / len(required_kps)
        # PEARL (2026): mastery < 0.3 → can't use L2+ effectively
        if avg_mastery < 0.3 and level >= 2:
            level -= 1
        # 掌握度 > 0.8 → near-mastery, upgrade if gate was too conservative
        elif avg_mastery > 0.8 and level < 2:
            level += 1

    return level


# ===========================================================================
# Layer 3: Independent dimension corrections
# ===========================================================================

# --- C: Cognitive level (Bloom) ---
#
# Wang et al. (ACL 2025) — Student_100 dataset, the ONLY paper that
# explicitly models Bloom levels in LLM student simulation.
# Their finding: 记忆层学生只会"模板式套用单步"，应用层学生可以
# "在有提示的情况下完成多步推理"。Wrong et al.'s cognitive prototype
# beam search shows that different Bloom levels produce qualitatively
# different errors, requiring different scaffolding levels.
#
# Direction: memory needs MORE structure (+1), analysis+ needs LESS (-1/-2).
# Magnitude: ±1 per EDF/Copa's gradual fading principle (avoid >1 jumps).
C_ADJUST = {
    "memory":        +1,
    "comprehension":  0,
    "application":   -1,
    "analysis":      -1,
    "evaluation":    -1,
    "creation":      -2,
}


def _apply_C_correction(bloom_level: str) -> int:
    return C_ADJUST.get(bloom_level, 0)


# --- E: Error type ---
#
# ParaStudent (UC Berkeley, 2025): 基于CS 61A真实学生代码提交的
# 错误分类 — logic / runtime / compile 三类错误需要不同的教师反馈。
# 映射到我们的四类：
#   conceptual: VanLehn (1990) — 需要"repair dialogue"，必须重新教学 → -1
#   computational: Black & Wiliam (1998) — 形成性评估，"指出哪步算错即可" → 0
#   strategic: Chi (2000) — 自我解释，"给框架让学生自己选方法" → 0
#   careless: 只需自查，不需要新教学 → +1（让他自己找）
#
# Khan et al. (BEA 2026): LLM教师对"半对半错"诊断差。
# → 粗心错误不应降级（教师容易误判）
E_ADJUST = {
    "conceptual":    -1,
    "computational":  0,
    "strategic":      0,
    "careless":      +1,
}


def _apply_E_correction(error_type: str) -> int:
    return E_ADJUST.get(error_type, 0)


# --- L: Learning characteristics ---
#
# TutorUp (CHI 2025): 86+102名真实教师调查识别的四种脱离模式，
# 每种需要不同的教学响应。"缺乏自信"和"学习速度差异"是两个
# 独立维度，教师给不同处理。
#
# BEAGLE (arXiv 2026): 半马尔可夫行为模型 — passive help-seeking
# 学生在任务中期才达到assistance-seeking峰值；avoidant学生几乎
# 全程on-task但最终放弃。这意味着passive和avoidant需要不同的
# 对待方式：passive → +1（多给结构），avoidant → +1（更需要主动介入）。
#
# help-seeking correction:
#   active: 会主动求助 → 0（保持，他自己会要帮助）
#   passive: 卡住会自己试但不敢问 → +1（需要主动给结构）
#   avoidant: 完全不求助，猜了就走 → +1（更需要主动介入）
#
# speed correction:
#   slow (<0.3): +1 — Cronbach & Snow (1977) ATI: 慢学生需要更多
#     教学支持，但允许更长的思考时间
#   fast (>0.8): -1 — 快学生容易被过多脚手架"过度支持"，应减少干预
L_ADJUST = {
    "help_seeking": {"active": 0, "passive": +1, "avoidant": +1},
    "speed_slow": +1,
    "speed_fast": -1,
}


def _apply_L_correction(student: StudentState) -> int:
    adj = 0
    adj += L_ADJUST["help_seeking"].get(student.help_seeking, 0)
    if student.learning_speed < 0.3:
        adj += L_ADJUST["speed_slow"]
    elif student.learning_speed > 0.8:
        adj += L_ADJUST["speed_fast"]
    return adj


# --- A: Affect / motivation ---
#
# 心理议会 (Hu et al., arXiv 2025): 唯一显式建模"情感→行为"
# 因果链的系统。关键发现：焦虑和自我效能是乘法交互——
# "高焦虑+低效能"的组合效应远大于单独效应之和。
# 同一学生在代数焦虑但在几何自信（领域特异性）。
#
# TutorUp (CHI 2025): 缺乏自信 → 需要成功体验，而非更多内容讲解。
# → self_efficacy < 0.3 → +1（先降低认知负荷保证成功）
#
# Bandura (1997) 自我效能理论: 低效能学生遇到困难后不调用深层策略。
# → anxiety > 0.7 → +1（降低认知负荷补偿工作记忆下降; Eysenck & Calvo 1992）
#
# 心理议会发现"威胁回避子智能体"在多次失败后压制"目标追求子智能体"
# → giving_up → +1（需要主动介入重建参与，但不能直接给答案）
A_ADJUST = {
    "anxiety_high": +1,        # >0.7: Eysenck & Calvo 1992
    "efficacy_low": +1,        # <0.3: Bandura 1997, TutorUp 2025
    "giving_up":    +1,        # 心理议会 2025: 威胁回避压制目标追求
}


def _apply_A_correction(student: StudentState) -> int:
    """Affect correction with overlap prevention.

    giving_up state already implies high anxiety + low efficacy +
    repeated failure (心理议会: threat-avoidance suppresses goal-pursuit).
    When giving_up is active, skip the individual anxiety correction
    to avoid double-counting the same underlying phenomenon.
    Self-efficacy correction still applies (Bandura effect is independent).
    """
    adj = 0
    giving_up = student.behavioral_state == "giving_up"

    # Anxiety: skip when giving_up (already captured by giving_up correction)
    if student.anxiety > 0.7 and not giving_up:
        adj += A_ADJUST["anxiety_high"]

    # Self-efficacy: independent of giving_up (Bandura 1997)
    if student.self_efficacy < 0.3:
        adj += A_ADJUST["efficacy_low"]

    # giving_up as behavioral state (心理议会 2025)
    if giving_up:
        adj += A_ADJUST["giving_up"]

    return adj


# --- T: Temporal development ---
#
# TASA (AAAI 2026 Workshop): DKT + Ebbinghaus遗忘曲线 —
# 核心发现：应根据遗忘程度决定"复习 vs 推进"。
# trajectory == "declining" → 退步中 → -1（降级，强制复习/检索练习）
# trajectory == "improving" → 进步中 → +1（升级，利用学习动量）
#
# MHPO (ACL 2026): 轨迹级奖励R_long证明"效率优先"策略（最少轮次）
# 优于"结果优先"策略。→ improving学生可以少给帮助（+1），
# 让"效率"自然提升；declining学生不能推（-1），先稳定。
#
# EDF/Copa (AAAI 2025/2026): 脚手架应随掌握度淡出。
# → plateau → 0（不调整，维持当前档位观察）
T_ADJUST = {
    "improving": +1,
    "plateau":    0,
    "declining": -1,
    "new":        0,
}


def _apply_T_correction(student: StudentState) -> int:
    return T_ADJUST.get(student.trajectory, 0)


# ===========================================================================
# Layer 4: Cross-dimension interaction corrections
# ===========================================================================

def _apply_cross_corrections(student: StudentState) -> int:
    """Cross-dimension interactions grounded in literature.

    心理议会 (Hu et al., 2025): anxiety × self_efficacy 乘法交互。
    "高焦虑+低效能"组合效应远超单独效应之和 → +1额外补偿。
    同一学生在代数焦虑、在几何自信 → 情感修正应有领域特异性。

    EDF/Copa (AAAI 2025/2026): 脚手架淡出应渐进，避免 ±2 跳跃。
    → cross corrections 上限 ±1，层3总和已在最终clamp中受控。

    TASA + 心理议会结合: trajectory declining + low efficacy →
    学生正在遗忘且不相信自己能学会 → 不能降级（T=-1会被
    效能低触发+A=+1抵消），但也不能升级。cross correction
    给 +1 稳定当前档位。

    MHPO (ACL 2026): R_short + R_long 联合优化 → 当declining
    且anxiety高时，R_short（教学原则）和R_long（效率）冲突：
    此时应优先R_short（先稳住学生）。
    """
    adj = 0

    # 心理议会 2025: 高焦虑 + 低效能 = 乘法效应 → +1
    if student.anxiety > 0.7 and student.self_efficacy < 0.3:
        adj += 1

    # 心理议会 + TASA: declining + low efficacy → +1 稳定
    if student.trajectory == "declining" and student.self_efficacy < 0.3:
        adj += 1

    # MHPO 2026: anxiety high + conceptual error → 优先R_short
    if student.anxiety > 0.7 and student.dominant_error_type == "conceptual":
        adj += 1

    # Wang et al. + 心理议会: memory level + high anxiety → +1
    if student.bloom_level == "memory" and student.anxiety > 0.7:
        adj += 1

    # EDF/Copa: limit total cross correction to ±2 (allow up to 2 independent
    # interaction effects, but prevent runaway accumulation)
    return max(-2, min(2, adj))


# ===========================================================================
# Main entry point
# ===========================================================================

def compute_multidim_disclosure_level(
    student: StudentState,
    required_kps: List[str],
    kg: Any,
) -> Dict[str, Any]:
    """Compute disclosure level from full six-dimension student state.

    Args:
        student: StudentState with all 6 dimensions.
        required_kps: KPs required by the current question.
        kg: KnowledgeGraph.

    Returns:
        {level, permission, base_level, adjustments, reason, ...}
    """
    from .student_model import student_state_to_missing_kps
    missing_kps = student_state_to_missing_kps(student, required_kps)

    # Layer 1: KG structural base
    base = _kg_structural_level(missing_kps, required_kps, kg)
    base_level = base["level"]

    # Layer 2: K refinement
    k_level = _k_refinement(base_level, student, required_kps)
    level = k_level

    # Layer 3: Independent dimension corrections
    c_adj = _apply_C_correction(student.bloom_level)
    e_adj = _apply_E_correction(student.dominant_error_type)
    l_adj = _apply_L_correction(student)
    a_adj = _apply_A_correction(student)
    t_adj = _apply_T_correction(student)

    level += c_adj + e_adj + l_adj + a_adj + t_adj

    # Layer 4: Cross-dimension interaction
    cross_adj = _apply_cross_corrections(student)
    level += cross_adj

    # Final constraints
    level = max(0, min(3, level))

    # L3 requires "all mastered" per VanLehn — misconception blocks L3
    if base_level < 3 and student.misconceptions:
        level = min(level, 2)

    # 心理议会: giving_up → at least L1 to re-engage
    if student.behavioral_state == "giving_up":
        level = max(level, 1)

    # EDF/Copa: prevent >2-level jumps from base
    if abs(level - base_level) > 2:
        level = base_level + (2 if level > base_level else -2)

    return {
        "level": level,
        "permission": LEVEL_PERMISSION[level],
        "base_level": base_level,
        "base_reason": base["reason"],
        "foundation_broken": base.get("foundation_broken", False),
        "n_missing_leaf": base.get("n_missing_leaf", 0),
        "n_missing": base.get("n_missing", 0),
        "adjustments": {
            "K_refinement": k_level - base_level,
            "C_bloom": c_adj,
            "E_error": e_adj,
            "L_learning": l_adj,
            "A_affect": a_adj,
            "T_temporal": t_adj,
            "cross_interaction": cross_adj,
            "total_adjustment": level - base_level,
        },
        "reason": (
            f"base={base_level}({base['reason']}), "
            f"C={student.bloom_level}({c_adj:+d}) "
            f"E={student.dominant_error_type}({e_adj:+d}) "
            f"L=speed={student.learning_speed:.1f},help={student.help_seeking}({l_adj:+d}) "
            f"A=anx={student.anxiety:.1f},eff={student.self_efficacy:.1f}({a_adj:+d}) "
            f"T={student.trajectory}({t_adj:+d}) "
            f"cross={cross_adj:+d} → L{level}"
        ),
    }


# ===========================================================================
# Self-check: `python -m ladder_v2.multidim_gate`
# ===========================================================================
if __name__ == "__main__":
    from .student_model import PRESET_STUDENTS

    class _StubKG:
        def __init__(self, prereqs):
            self._prereqs = prereqs
        def get_prerequisites(self, kp):
            return self._prereqs.get(kp, [])
        def get_all_prerequisites(self, kps):
            out, stack = set(), list(kps)
            while stack:
                cur = stack.pop()
                if cur in out: continue
                out.add(cur)
                stack.extend(self._prereqs.get(cur, []))
            return out
        def is_valid_missing_combination(self, required_kps, missing_kps):
            mastered = [kp for kp in required_kps if kp not in missing_kps]
            mastered_prereqs = self.get_all_prerequisites(mastered)
            return not any(m in mastered_prereqs for m in missing_kps)

    kg = _StubKG({"rank": ["linear_independence"],
                   "linear_independence": ["vector_space"],
                   "vector_space": [], "determinant": []})
    required = ["vector_space", "linear_independence", "rank"]

    print("=" * 70)
    print("MULTI-DIM GATE: same K, different dimensions → different levels")
    print("=" * 70)
    for name in ["anxious_low_k", "confident_mid_k", "careless_high_k",
                 "slow_strategic_err", "giving_up"]:
        s = PRESET_STUDENTS[name]
        r = compute_multidim_disclosure_level(s, required, kg)
        print(f"\n{name}: L{r['level']} (base L{r['base_level']})  {r['permission']}")
        print(f"  {r['reason']}")

    print("\n" + "=" * 70)
    print("AFFECT SENSITIVITY: same K, different A → should differ")
    print("=" * 70)
    same_k = {"vector_space": 0.6, "linear_independence": 0.6, "rank": 0.3}
    for label, anx, eff in [("anxious+low_eff", 0.9, 0.2),
                              ("calm+high_eff", 0.1, 0.9)]:
        s = StudentState(kp_mastery=same_k, bloom_level="application",
                         dominant_error_type="computational",
                         anxiety=anx, self_efficacy=eff)
        r = compute_multidim_disclosure_level(s, required, kg)
        print(f"  {label}: L{r['level']} (base L{r['base_level']}) A_adj={r['adjustments']['A_affect']:+d}")

    print("\nDONE")
