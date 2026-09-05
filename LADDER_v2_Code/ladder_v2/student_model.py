"""
Multi-dimensional student model for LADDER v2.

Dimension sources (see docs/LADDER/LADDER_学生设置方案参照表.md):

  K - Knowledge state
      → PEARL (2026): per-KP continuous mastery 0-1, decoupled from E
      → BEAGLE (2026): BKT per-KP probability + bug injection (misconceptions)

  C - Cognitive level (Bloom)
      → Wang et al. (ACL 2025): 唯一显式Bloom六层建模的CCF-A工作
        认知原型+beam search, Student_100 dataset

  E - Error patterns
      → ParaStudent (UC Berkeley, 2025): logic/runtime/compile 三类显式标注
      → VanLehn (1990): conceptual/computational/strategic/careless 四类经典分类

  L - Learning characteristics
      → BEAGLE (2026): SRL+半马尔可夫, speed/help-seeking/attention 行为状态机
      → TutorUp (CHI 2025): 86+102教师调查→四种脱离模式
      → HACHIMI (ACL 2026): study_habits + attention 静态画像

  A - Affect / motivation
      → HACHIMI (ACL 2026): self_efficacy + motivation + anxiety, 100万画像,
        对标CEPS和PISA 2022. 唯一可复现的标准化情感画像.
      → 心理议会 (Hu et al., 2025): 多智能体辩论, 唯一L3情感-认知交互

  T - Temporal development
      → BEAGLE (2026): 行为状态随任务进度变化(off-task晚期增加)
      → TASA (AAAI 2026): DKT+Ebbinghaus, trajectory驱动复习vs推进决策
      → ParaStudent (2025): 提交时间戳序列
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random


# ===========================================================================
# StudentState — 六维学生状态
# ===========================================================================

@dataclass
class StudentState:
    """Six-dimension student state.

    All dimensions are independently configurable (decoupled design,
    following PEARL 2026: "controllable student simulator with
    decoupled cognitive state and diverse errors").
    """

    # === K: Knowledge state ===
    # PEARL (2026) + BEAGLE (2026): per-KP continuous mastery 0-1
    # 0 = no knowledge, 1 = fully mastered
    kp_mastery: Dict[str, float] = field(default_factory=dict)

    # BEAGLE (2026): bug injection — pre-defined misconceptions on specific KPs
    # VanLehn (1990): misconceptions trigger "repair dialogue", not procedural help
    misconceptions: List[str] = field(default_factory=list)

    # === C: Cognitive level ===
    # Wang et al. (ACL 2025): 知识图谱认知原型, Student_100 dataset
    # Bloom taxonomy (Anderson & Krathwohl 2001 revision)
    bloom_level: str = "application"
    # Options: memory | comprehension | application | analysis | evaluation | creation

    # === E: Error patterns ===
    # VanLehn (1990) + ParaStudent (2025): 显式错误类型标注
    dominant_error_type: str = "computational"
    # Options: conceptual | computational | strategic | careless

    # === L: Learning characteristics ===
    # BEAGLE (2026): SRL theory — learning speed, help-seeking, attention
    learning_speed: float = 0.5       # 0=very slow, 1=very fast

    # TutorUp (CHI 2025): 四种脱离模式区分 active/passive/avoidant
    help_seeking: str = "active"      # active | passive | avoidant

    # EMNLP Genetic+RAG (2025): 6-dim learning styles, we collapse to 3
    preference: str = "verbal"        # verbal | visual | symbolic

    # === A: Affect / motivation ===
    # HACHIMI (ACL 2026): self_efficacy, motivation_level, anxiety_level
    #   validated against CEPS (中国教育追踪调查) and PISA 2022
    anxiety: float = 0.5              # 0=calm, 1=extremely anxious
    self_efficacy: float = 0.5        # 0="I can't learn this", 1="I can definitely learn this"
    motivation: str = "extrinsic"     # intrinsic | extrinsic | none

    # === T: Temporal development ===
    # BEAGLE (2026): behavioral state evolves with task progress
    # TASA (AAAI 2026): forgetting curve → trajectory label
    attempt_count: int = 0
    previous_errors: List[str] = field(default_factory=list)  # last N: "correct"/"wrong"
    trajectory: str = "new"           # new | improving | plateau | declining

    # === Behavioral state (derived from L+A+T, per BEAGLE semi-Markov) ===
    behavioral_state: str = "on_task"
    # Options: on_task | off_task | help_seeking | giving_up

    # Metadata
    student_id: str = ""

    # HACHIMI compatibility: fields preserved from HACHIMI import
    grade_level: Optional[int] = None         # 1-12
    study_habits: Optional[str] = None        # regular | irregular | procrastinating
    attention: Optional[str] = None           # sustained | easily_distracted
    family_ses: Optional[str] = None          # low | mid | high


# ===========================================================================
# Dimension behavior descriptions (paper-grounded)
# ===========================================================================

# Wang et al. (ACL 2025): cognitive prototype → behavior differences at each Bloom level
BLOOM_BEHAVIORS = {
    "memory": (
        "你只能复述定义和公式，但不太理解它们为什么有效。"
        "做题时喜欢套公式，换一种问法就不会了。"
        "[Wang et al. ACL 2025: 记忆层学生仅做模板式单步应用]"
    ),
    "comprehension": (
        "你能用自己的话解释概念，也能举出简单例子。"
        "但遇到需要多步推理的题会卡住。"
    ),
    "application": (
        "你能把学过的解题步骤用到新题目上。"
        "但遇到需要自己选择方法（而不是套公式）的题会犹豫。"
    ),
    "analysis": (
        "你能拆解复杂问题，判断该用哪个方法。"
        "做错时能意识到不对，但不一定能自己找到错误根源。"
    ),
    "evaluation": (
        "你能检查自己的答案，发现错误并比较哪个解法更好。"
    ),
    "creation": (
        "你能把多个概念综合起来设计新的解法。"
    ),
}

# ParaStudent (2025): 三类显式错误 → 扩展到 VanLehn (1990) 四类
# Black & Wiliam (1998): 形成性评估 — 不同错误类型需要不同反馈
ERROR_BEHAVIORS = {
    "conceptual": (
        "你犯错的原因通常是概念理解有误。即使老师指出你某一步算错了，"
        "你还是会用同一个错误概念换一种方式重新做。"
        "你不会自己意识到概念错误——需要被明确纠正。"
        "[VanLehn 1990: 概念错误需要 repair dialogue]"
    ),
    "computational": (
        "你概念是理解的，但经常在计算步骤中出错——符号反了、数算错了、"
        "步骤跳了。老师提醒'检查一下计算'时你能自己找到并修正错误。"
        "[ParaStudent 2025: compile/runtime errors → fixable with hint]"
    ),
    "strategic": (
        "你概念和计算都没问题，但经常选不对方法——比如该用消元法时用了代入法。"
        "给你一个提示'换个方法试试'你就能自己完成。"
    ),
    "careless": (
        "你其实会做，但经常粗心——抄错数字、漏看条件、忘记最后一步。"
        "当老师提醒'再仔细看一遍题目'时你能立刻发现错误。"
    ),
}


# ===========================================================================
# Behavior prompt generator
# ===========================================================================

def student_state_to_behavior_prompt(s: StudentState) -> str:
    """Convert StudentState → behavior instruction prompt for student-simulator LLM.

    Each dimension is translated into concrete behavioral descriptions,
    grounded in the source papers' empirical findings.
    """
    lines = [
        "你是一个学生，正在向AI家教请教一道线性代数题。",
        "请严格按照以下行为描述来扮演你的角色。这些描述来自教育心理学和教育数据挖掘的实证研究。",
        "",
    ]

    # K: PEARL/BEAGLE — per-KP mastery
    weak_kps = [kp for kp, m in s.kp_mastery.items() if m < 0.5]
    if weak_kps:
        lines.append("## 知识状态 [PEARL 2026: per-KP continuous mastery]")
        lines.append(f"你对这些概念不太有把握（掌握度 < 0.5）：{', '.join(weak_kps)}。")
        lines.append("涉及这些概念的问题，你的回答会表现出犹豫、不确定或出错。")

    if s.misconceptions:
        lines.append(f"你对以下概念存在误解：{', '.join(s.misconceptions)}。")
        lines.append("当问题涉及这些概念时，你会稳定地用错误方法解答，不会自己意识到错了。")
        lines.append("[BEAGLE 2026: bug injection; VanLehn 1990: misconception → repair dialogue]")
        lines.append("")

    # C: Wang et al. (ACL 2025)
    lines.append(f"## 认知水平 [{s.bloom_level}] — Wang et al. ACL 2025")
    lines.append(BLOOM_BEHAVIORS.get(s.bloom_level, BLOOM_BEHAVIORS["application"]))
    lines.append("")

    # E: ParaStudent / VanLehn
    lines.append(f"## 错误模式 [{s.dominant_error_type}] — VanLehn 1990; ParaStudent 2025")
    lines.append(ERROR_BEHAVIORS.get(s.dominant_error_type, ERROR_BEHAVIORS["computational"]))
    lines.append("")

    # L: BEAGLE + TutorUp
    lines.append("## 学习特征 [BEAGLE 2026: SRL+半马尔可夫; TutorUp CHI 2025]")
    if s.learning_speed < 0.3:
        lines.append("- 你学得比较慢，需要更多时间思考和消化。进步是渐进的。")
    elif s.learning_speed > 0.7:
        lines.append("- 你学得很快，但有时会因为太快而忽略细节。")

    if s.help_seeking == "passive":
        lines.append("- 你不太主动求助。卡住时自己默默尝试，几次无效后倾向于沉默或放弃。")
        lines.append("  [BEAGLE: passive型 → assistance-seeking峰值在任务中期]")
    elif s.help_seeking == "avoidant":
        lines.append("- 你几乎从不主动求助。完全不会时宁愿猜答案或说'我不会'，而不是请求帮助。")
        lines.append("  [BEAGLE: avoidant型 → 全程on-task但最终放弃]")

    if s.preference == "visual":
        lines.append("- 你更容易理解图形化解释。老师用公式推导时跟不上，画图时你就懂了。")
    elif s.preference == "symbolic":
        lines.append("- 你更喜欢公式和符号化推演。用文字或例子解释时你觉得不够精确。")
    lines.append("")

    # A: HACHIMI + 心理议会
    lines.append("## 情感状态 [HACHIMI ACL 2026: CEPS/PISA验证; 心理议会 2025]")
    if s.anxiety > 0.7:
        lines.append("- 你很容易焦虑。焦虑让你更容易算错（工作记忆下降）。")
        lines.append("  [Eysenck & Calvo 1992: 高焦虑→工作记忆容量下降]")
        lines.append("- 你会频繁使用'我不确定'、'可能不对'这类短语。")
    elif s.anxiety < 0.3:
        lines.append("- 你比较放松，不害怕犯错，愿意尝试。")

    if s.self_efficacy < 0.3:
        lines.append("- 你不太相信自己能学会。遇到困难时第一反应是'我果然不行'。")
        lines.append("  [Bandura 1997: 低效能→不调用深层策略; TutorUp: 缺乏自信→需要成功体验]")
        if s.anxiety > 0.7:
            lines.append("- 高焦虑+低效能叠加：你很容易直接说'我不会，告诉我答案吧'。")
            lines.append("  [心理议会 2025: 焦虑×效能乘法交互]")
    elif s.self_efficacy > 0.7:
        lines.append("- 你相信自己能学会。遇到困难时想'我再试一次'而非'我不行'。")

    if s.motivation == "intrinsic":
        lines.append("- 你学习是因为真的想理解。比起'正确答案'，你更关心'为什么'。")
    elif s.motivation == "extrinsic":
        lines.append("- 你学习主要是因为考试或作业。比起'为什么'，你更关心'怎么做才能拿分'。")
    lines.append("")

    # T: TASA + BEAGLE
    lines.append("## 当前状态 [TASA AAAI 2026: trajectory; BEAGLE 2026: temporal behavior]")
    t_descriptions = {
        "improving": "- 你最近在进步！连续几次尝试都比之前更好了，有了一点信心。",
        "declining": "- 你最近正确率在下降，开始有些沮丧。",
        "plateau": "- 你卡在一个平台上——不差，但也没进步。你有点困惑。",
        "new": "- 这是你第一次尝试这道题。你还不确定自己能不能做出来。",
    }
    lines.append(t_descriptions.get(s.trajectory, t_descriptions["new"]))

    if s.attempt_count >= 3:
        lines.append(f"- 你已经尝试了 {s.attempt_count} 次。")
        if s.behavioral_state == "giving_up":
            lines.append("- 多次失败后，你已经接近放弃状态。")
            lines.append("  [心理议会: 威胁回避子智能体压制目标追求子智能体]")

    if s.behavioral_state == "giving_up":
        lines.append("- 你现在的行为状态是'想要放弃'——你可能会说'直接给我答案吧'或'我放弃了'。")
        lines.append("  [BEAGLE: off-task状态在任务晚期达到峰值]")

    lines.append("")
    lines.append("请根据以上所有行为描述来生成你的下一步回答。")
    lines.append("你的回答应该真实地反映这些特征，但不要让特征显得刻意或做作。")

    return "\n".join(lines)


# ===========================================================================
# State updater (multi-turn) — BEAGLE semi-Markov + 心理议会 dynamics
# ===========================================================================

def update_student_state(
    s: StudentState,
    student_response_correctness: bool,
    teacher_level_given: int,
) -> StudentState:
    """Update student state after one turn of teacher-student interaction.

    Grounded in:
      - BEAGLE (2026): semi-Markov behavioral state transitions
      - 心理议会 (2025): anxiety-efficacy interaction dynamics
      - TASA (2026): trajectory update from correctness history
    """
    # T: temporal progression
    s.attempt_count += 1
    s.previous_errors.append("correct" if student_response_correctness else "wrong")
    if len(s.previous_errors) > 5:
        s.previous_errors = s.previous_errors[-5:]

    # Track last 3 for trajectory
    recent = s.previous_errors[-3:] if len(s.previous_errors) >= 3 else s.previous_errors
    n_correct, n_wrong = recent.count("correct"), recent.count("wrong")
    if n_correct >= 2:
        s.trajectory = "improving"
    elif n_wrong >= 3:
        s.trajectory = "declining"
    elif n_correct == n_wrong and n_correct > 0:
        s.trajectory = "plateau"

    # A: affect dynamics (心理议会 2025: anxiety-efficacy co-evolution)
    if student_response_correctness:
        s.anxiety = max(0.0, s.anxiety - 0.1)
        s.self_efficacy = min(1.0, s.self_efficacy + 0.1)
    else:
        # 心理议会: low-efficacy students spiral faster on failure
        if s.self_efficacy < 0.3:
            s.anxiety = min(1.0, s.anxiety + 0.2)
        else:
            s.anxiety = min(1.0, s.anxiety + 0.05)
        s.self_efficacy = max(0.0, s.self_efficacy - 0.05)

    # Teacher support bonus: appropriate-level help reduces anxiety
    avg_mastery = sum(s.kp_mastery.values()) / max(len(s.kp_mastery), 1)
    help_appropriate = (avg_mastery > 0.6 and teacher_level_given >= 2) or \
                       (avg_mastery < 0.4 and teacher_level_given <= 1)
    if help_appropriate:
        s.anxiety = max(0.0, s.anxiety - 0.05)

    # Behavioral state transition (BEAGLE 2026: semi-Markov model)
    if s.trajectory == "declining" and s.anxiety > 0.7:
        s.behavioral_state = "giving_up"
    elif s.attempt_count > 3 and s.self_efficacy < 0.3:
        s.behavioral_state = "off_task"
    elif s.help_seeking == "active" and s.trajectory == "plateau":
        s.behavioral_state = "help_seeking"
    else:
        s.behavioral_state = "on_task"

    return s


# ===========================================================================
# Student pool builder
# ===========================================================================

# Real KP IDs from SHaPE benchmark (linear algebra, 91 KPs).
# Students use these exact keys so kp_mastery matches question required_kps.
_SHAPE_KPS = [
    "Linear Dependence and Independence",
    "Linear Combinations of Vectors in N-Dimensional Euclidean Space",
    "The Rank of a Matrix",
    "Subspaces of N-Dimensional Space",
    "The Column Space of a Matrix",
    "The Null Space of a Matrix",
    "The Dot Product in N-Dimensional Euclidean Space",
    "Orthogonal Vectors in Euclidean Spaces",
    "The Determinant of an NxN Matrix",
    "Finding Determinants Using Laplace Expansions",
]

# Five preset students covering distinct regions of the 6D space.
# KP keys match real SHaPE benchmark vocabulary for correct base-level computation.
PRESET_STUDENTS = {
    # Test: high anxiety + low efficacy → gate should upgrade
    # Source: HACHIMI anxiety=high, self_efficacy=low; Wang memory level
    "anxious_low_k": StudentState(
        student_id="anxious_low_k",
        kp_mastery={
            "Linear Dependence and Independence": 0.3,
            "Linear Combinations of Vectors in N-Dimensional Euclidean Space": 0.2,
            "The Rank of a Matrix": 0.15,
        },
        misconceptions=["Linear Dependence and Independence"],
        bloom_level="memory",
        dominant_error_type="conceptual",
        learning_speed=0.3, help_seeking="passive",
        anxiety=0.9, self_efficacy=0.15, motivation="extrinsic",
        trajectory="declining",
    ),

    # Test: calm + confident → gate should not upgrade
    # Source: HACHIMI anxiety=low, self_efficacy=high; intrinsic motivation
    "confident_mid_k": StudentState(
        student_id="confident_mid_k",
        kp_mastery={
            "Linear Dependence and Independence": 0.7,
            "Linear Combinations of Vectors in N-Dimensional Euclidean Space": 0.6,
            "The Rank of a Matrix": 0.4,
        },
        bloom_level="comprehension",
        dominant_error_type="computational",
        learning_speed=0.6, help_seeking="active",
        anxiety=0.2, self_efficacy=0.75, motivation="intrinsic",
        trajectory="improving",
    ),

    # Test: high K + careless → minimal intervention
    # Source: ParaStudent careless errors
    "careless_high_k": StudentState(
        student_id="careless_high_k",
        kp_mastery={
            "Linear Dependence and Independence": 0.9,
            "The Rank of a Matrix": 0.85,
            "Subspaces of N-Dimensional Space": 0.8,
        },
        bloom_level="application",
        dominant_error_type="careless",
        learning_speed=0.8, help_seeking="active",
        anxiety=0.3, self_efficacy=0.8, motivation="intrinsic",
        trajectory="plateau",
    ),

    # Test: slow learner + passive → gate should upgrade
    # Source: BEAGLE passive + slow; TutorUp "different learning speed"
    "slow_passive": StudentState(
        student_id="slow_passive",
        kp_mastery={
            "Linear Dependence and Independence": 0.6,
            "The Rank of a Matrix": 0.55,
            "Subspaces of N-Dimensional Space": 0.5,
        },
        bloom_level="application",
        dominant_error_type="strategic",
        learning_speed=0.25, help_seeking="passive",
        anxiety=0.5, self_efficacy=0.4, motivation="extrinsic",
        trajectory="new",
    ),

    # Test: giving_up → gate should never give L0
    # Source: 心理议会 threat-avoidance dominant; BEAGLE off-task peak
    "giving_up": StudentState(
        student_id="giving_up",
        kp_mastery={
            "Linear Dependence and Independence": 0.4,
            "The Rank of a Matrix": 0.35,
            "Subspaces of N-Dimensional Space": 0.3,
        },
        bloom_level="memory",
        dominant_error_type="conceptual",
        learning_speed=0.4, help_seeking="avoidant",
        anxiety=0.95, self_efficacy=0.05, motivation="none",
        trajectory="declining", attempt_count=4,
        previous_errors=["wrong", "wrong", "wrong", "wrong"],
        behavioral_state="giving_up",
    ),
}

# Dimension value pools for random student generation.
# Ranges validated against HACHIMI (CEPS/PISA) and BEAGLE empirical data.
DIM_POOLS = {
    "bloom_level": ["memory", "comprehension", "application", "analysis"],
    "dominant_error_type": ["conceptual", "computational", "strategic", "careless"],
    "learning_speed": [0.2, 0.5, 0.8],
    "help_seeking": ["active", "passive", "avoidant"],
    "preference": ["verbal", "visual", "symbolic"],
    "anxiety": [0.8, 0.5, 0.2],
    "self_efficacy": [0.2, 0.5, 0.8],
    "motivation": ["intrinsic", "extrinsic", "none"],
    "trajectory": ["new", "improving", "plateau", "declining"],
}


def generate_student_pool(
    n_students: int,
    kp_list: List[str],
    mastery_levels: List[float] = (0.2, 0.5, 0.8),
    seed: int = 42,
) -> List[StudentState]:
    """Generate diverse student pool with random 6D combinations.

    Mastery levels follow PEARL (2026) continuous 0-1 range.
    Dimension pools validated against HACHIMI and BEAGLE distributions.
    """
    rng = random.Random(seed)
    students = []

    for i in range(n_students):
        kp_mastery = {kp: rng.choice(mastery_levels) for kp in kp_list}

        # ~25% chance of misconception, per VanLehn (1990) frequency estimates
        misconceptions = []
        if rng.random() < 0.25 and len(kp_list) >= 2:
            misconceptions = [rng.choice(kp_list)]

        s = StudentState(
            student_id=f"student_{i:04d}",
            kp_mastery=kp_mastery,
            misconceptions=misconceptions,
            bloom_level=rng.choice(DIM_POOLS["bloom_level"]),
            dominant_error_type=rng.choice(DIM_POOLS["dominant_error_type"]),
            learning_speed=rng.choice(DIM_POOLS["learning_speed"]),
            help_seeking=rng.choice(DIM_POOLS["help_seeking"]),
            preference=rng.choice(DIM_POOLS["preference"]),
            anxiety=rng.choice(DIM_POOLS["anxiety"]),
            self_efficacy=rng.choice(DIM_POOLS["self_efficacy"]),
            motivation=rng.choice(DIM_POOLS["motivation"]),
            trajectory=rng.choice(DIM_POOLS["trajectory"]),
        )
        students.append(s)

    return students


def student_state_to_missing_kps(s: StudentState, required_kps: List[str]) -> List[str]:
    """Convert StudentState → LADDER v1 missing_kps format.

    A KP is 'missing' if mastery < 0.5 (PEARL threshold) OR in misconceptions.
    This enables backward compatibility with the v1 pipeline.
    """
    missing = []
    uncovered = [kp for kp in required_kps if kp not in s.kp_mastery]
    if uncovered:
        raise ValueError(
            "Student mastery profile does not cover required KPs: "
            + ", ".join(uncovered)
        )
    for kp in required_kps:
        mastery = s.kp_mastery[kp]
        if mastery < 0.5 or kp in s.misconceptions:
            missing.append(kp)
    return missing


# ===========================================================================
# HACHIMI import: map 13-dim profile → StudentState
# ===========================================================================

def from_hachimi_profile(profile: dict, student_id: str = None) -> StudentState:
    """Convert a HACHIMI (ACL 2026) 13-dim profile to StudentState.

    HACHIMI fields → our dimensions:
      academic_ability → kp_mastery (approximate, as single-value proxy)
      learning_strategy (surface/deep) → bloom_level (surface≈memory, deep≈comprehension)
      self_efficacy → self_efficacy
      motivation_level → motivation
      anxiety_level → anxiety
      study_habits → study_habits (preserved)
      attention → attention (preserved)
      family_SES → family_ses (preserved)

    Note: HACHIMI has no E, T, or per-KP K data → these use defaults.
    """
    s = StudentState(student_id=student_id or profile.get("student_id", ""))

    # K: HACHIMI uses a single academic_ability score, not per-KP
    # Map to a generic "math" KP as placeholder
    ability = profile.get("academic_ability", 0.5)
    s.kp_mastery = {"math_general": float(ability)}

    # C: surface learning ≈ memory, deep ≈ comprehension+
    strategy = profile.get("learning_strategy", "surface")
    s.bloom_level = "memory" if strategy == "surface" else "comprehension"

    # A: direct 1:1 mapping (HACHIMI validated against CEPS/PISA)
    # HACHIMI values: low=0.2, mid=0.5, high=0.8
    level_map = {"low": 0.2, "mid": 0.5, "high": 0.8}
    s.self_efficacy = level_map.get(profile.get("self_efficacy", "mid"), 0.5)
    s.anxiety = level_map.get(profile.get("anxiety_level", "mid"), 0.5)
    s.motivation = profile.get("motivation_level", "extrinsic")

    # L: HACHIMI study_habits + attention
    s.study_habits = profile.get("study_habits")
    s.attention = profile.get("attention")
    if s.study_habits == "irregular" or s.study_habits == "procrastinating":
        s.learning_speed = 0.3  # irregular habits → slower effective speed

    # HACHIMI extras
    s.grade_level = profile.get("grade_level")
    s.family_ses = profile.get("family_SES")

    # E, T: not in HACHIMI → use defaults
    return s


# ===========================================================================
# Self-check
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("STUDENT MODEL: preset behavior prompts")
    for name in ["anxious_low_k", "confident_mid_k", "giving_up"]:
        s = PRESET_STUDENTS[name]
        p = student_state_to_behavior_prompt(s)
        print(f"\n{'='*60}")
        print(f"PRESET: {name}")
        print(f"  dims: C={s.bloom_level} E={s.dominant_error_type} "
              f"A=(anx={s.anxiety},eff={s.self_efficacy}) "
              f"T={s.trajectory} behavior={s.behavioral_state}")
        print(f"  prompt length: {len(p)} chars")

    print(f"\n{'='*60}")
    print("HACHIMI IMPORT")
    hachi = {"academic_ability": 0.65, "learning_strategy": "surface",
             "self_efficacy": "low", "anxiety_level": "high",
             "motivation_level": "extrinsic", "study_habits": "irregular",
             "attention": "easily_distracted", "grade_level": 7,
             "family_SES": "low"}
    s = from_hachimi_profile(hachi, "hachi_001")
    print(f"  imported: C={s.bloom_level} A=(anx={s.anxiety},eff={s.self_efficacy}) "
          f"motivation={s.motivation} grade={s.grade_level}")

    print(f"\n{'='*60}")
    print("STATE UPDATE: multi-turn evolution")
    s = PRESET_STUDENTS["anxious_low_k"]
    for turn in range(3):
        s = update_student_state(s, False, teacher_level_given=0)
        print(f"  turn {turn+1}: anxiety={s.anxiety:.2f} efficacy={s.self_efficacy:.2f} "
              f"traj={s.trajectory} behavior={s.behavioral_state}")
    for turn in range(2):
        s = update_student_state(s, True, teacher_level_given=1)
        print(f"  turn {turn+4}: anxiety={s.anxiety:.2f} efficacy={s.self_efficacy:.2f} "
              f"traj={s.trajectory} behavior={s.behavioral_state}")

    print(f"\n{'='*60}")
    print("STUDENT POOL: 5 random students")
    pool = generate_student_pool(5, ["kp_a", "kp_b", "kp_c"], seed=99)
    for s in pool:
        print(f"  {s.student_id}: C={s.bloom_level} E={s.dominant_error_type} "
              f"A=({s.anxiety:.1f},{s.self_efficacy:.1f}) "
              f"L=(spd={s.learning_speed:.1f},help={s.help_seeking}) T={s.trajectory}")

    print("\nDONE")
