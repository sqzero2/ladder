"""
HACHIMI (ACL 2026) distribution-based student sampler for LADDER.

HACHIMI generated 1M K12 student profiles validated against CEPS and PISA 2022.
This module samples LADDER StudentState parameters from HACHIMI's reported
distributions, replacing hand-picked preset students with data-driven ones.

Distributions referenced:
  - Academic level: 高25% / 中25% / 低25% / 差25% (HACHIMI baseline_single_shot.py)
  - Anxiety/efficacy: CEPS 2014-2015 national norms (杨钋 et al.)
  - Motivation: PISA 2022 China report
"""

import random
from typing import List, Dict
from .student_model import StudentState

# ===========================================================================
# HACHIMI academic level → K mastery range
# ===========================================================================

ACADEMIC_LEVEL_DIST = {
    "高": {"weight": 0.25, "mastery_range": (0.7, 1.0)},
    "中": {"weight": 0.25, "mastery_range": (0.4, 0.7)},
    "低": {"weight": 0.25, "mastery_range": (0.2, 0.5)},
    "差": {"weight": 0.25, "mastery_range": (0.0, 0.3)},
}

# ===========================================================================
# CEPS-based anxiety distribution (中国教育追踪调查)
# ===========================================================================

ANXIETY_DIST = [
    # (label, weight, value_range)
    ("high",   0.20, (0.7, 1.0)),
    ("medium", 0.50, (0.3, 0.7)),
    ("low",    0.30, (0.0, 0.3)),
]

# ===========================================================================
# CEPS-based self-efficacy distribution
# ===========================================================================

EFFICACY_DIST = [
    ("low",    0.25, (0.0, 0.3)),
    ("medium", 0.50, (0.3, 0.7)),
    ("high",   0.25, (0.7, 1.0)),
]

# ===========================================================================
# PISA 2022-based motivation distribution
# ===========================================================================

MOTIVATION_DIST = [
    ("intrinsic",  0.35),
    ("extrinsic",  0.50),
    ("none",       0.15),
]

# ===========================================================================
# Dimension pools (for random sampling)
# ===========================================================================

BLOOM_POOL = ["memory", "comprehension", "application", "analysis"]
BLOOM_WEIGHTS = [0.15, 0.30, 0.40, 0.15]  # Most students at comprehension/application

ERROR_POOL = ["conceptual", "computational", "strategic", "careless"]
ERROR_WEIGHTS = [0.25, 0.35, 0.15, 0.25]

SPEED_POOL = [0.2, 0.5, 0.8]
SPEED_WEIGHTS = [0.25, 0.50, 0.25]

HELP_POOL = ["active", "passive", "avoidant"]
HELP_WEIGHTS = [0.40, 0.40, 0.20]

TRAJECTORY_POOL = ["new", "improving", "plateau", "declining"]
TRAJECTORY_WEIGHTS = [0.30, 0.30, 0.25, 0.15]


def _weighted_choice(items: list, weights: list, rng: random.Random):
    return rng.choices(items, weights=weights, k=1)[0]


def _sample_range(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 2)


def sample_student_from_hachimi(
    kp_list: List[str],
    seed: int = None,
) -> StudentState:
    """Sample one StudentState from HACHIMI-validated distributions.

    Args:
        kp_list: List of knowledge point IDs (from SHaPE benchmark).
        seed: Random seed for reproducibility.

    Returns:
        StudentState with parameters sampled from HACHIMI/CEPS/PISA distributions.
    """
    rng = random.Random(seed)

    # --- Academic level → K mastery ---
    levels = list(ACADEMIC_LEVEL_DIST.keys())
    level_weights = [ACADEMIC_LEVEL_DIST[l]["weight"] for l in levels]
    academic_level = _weighted_choice(levels, level_weights, rng)
    mastery_lo, mastery_hi = ACADEMIC_LEVEL_DIST[academic_level]["mastery_range"]
    kp_mastery = {kp: _sample_range(rng, mastery_lo, mastery_hi) for kp in kp_list}

    # --- Misconceptions: ~20% chance for academic_level in {低, 差} ---
    misconceptions = []
    if academic_level in ("低", "差") and rng.random() < 0.20 and len(kp_list) >= 2:
        misconceptions = [rng.choice(kp_list)]

    # --- C: Bloom level ---
    bloom_level = _weighted_choice(BLOOM_POOL, BLOOM_WEIGHTS, rng)

    # --- E: Error type ---
    dominant_error_type = _weighted_choice(ERROR_POOL, ERROR_WEIGHTS, rng)

    # --- L: Learning characteristics ---
    learning_speed = _weighted_choice(SPEED_POOL, SPEED_WEIGHTS, rng)
    help_seeking = _weighted_choice(HELP_POOL, HELP_WEIGHTS, rng)
    preference = _weighted_choice(["verbal", "visual", "symbolic"], [0.5, 0.3, 0.2], rng)

    # --- A: Affect (CEPS norms) ---
    anxiety_labels = [a[0] for a in ANXIETY_DIST]
    anxiety_weights = [a[1] for a in ANXIETY_DIST]
    anxiety_label = _weighted_choice(anxiety_labels, anxiety_weights, rng)
    anxiety_lo, anxiety_hi = next(a[2] for a in ANXIETY_DIST if a[0] == anxiety_label)
    anxiety = _sample_range(rng, anxiety_lo, anxiety_hi)

    eff_labels = [e[0] for e in EFFICACY_DIST]
    eff_weights = [e[1] for e in EFFICACY_DIST]
    eff_label = _weighted_choice(eff_labels, eff_weights, rng)
    eff_lo, eff_hi = next(e[2] for e in EFFICACY_DIST if e[0] == eff_label)
    self_efficacy = _sample_range(rng, eff_lo, eff_hi)

    motivation = _weighted_choice(
        [m[0] for m in MOTIVATION_DIST],
        [m[1] for m in MOTIVATION_DIST],
        rng,
    )

    # --- T: Trajectory ---
    trajectory = _weighted_choice(TRAJECTORY_POOL, TRAJECTORY_WEIGHTS, rng)

    # --- Behavioral state (derived) ---
    if anxiety > 0.7 and self_efficacy < 0.3:
        behavioral_state = "giving_up"
    elif anxiety > 0.5 and trajectory == "declining":
        behavioral_state = "off_task"
    elif help_seeking == "active" and trajectory == "plateau":
        behavioral_state = "help_seeking"
    else:
        behavioral_state = "on_task"

    return StudentState(
        student_id=f"hachimi_{seed or rng.randint(0, 999999):06d}",
        kp_mastery=kp_mastery,
        misconceptions=misconceptions,
        bloom_level=bloom_level,
        dominant_error_type=dominant_error_type,
        learning_speed=learning_speed,
        help_seeking=help_seeking,
        preference=preference,
        anxiety=anxiety,
        self_efficacy=self_efficacy,
        motivation=motivation,
        trajectory=trajectory,
        behavioral_state=behavioral_state,
    )


def sample_student_pool_from_hachimi(
    n: int,
    kp_list: List[str],
    base_seed: int = 42,
) -> List[StudentState]:
    """Sample n students from HACHIMI distributions.

    Args:
        n: Number of students.
        kp_list: KP IDs.
        base_seed: Base random seed.

    Returns:
        List of StudentState.
    """
    return [sample_student_from_hachimi(kp_list, base_seed + i) for i in range(n)]


def compute_pool_statistics(students: List[StudentState]) -> Dict:
    """Compute distribution statistics for a student pool.

    Returns a dict suitable for the paper's "student pool vs HACHIMI" table.
    """
    n = len(students)
    anxieties = [s.anxiety for s in students]
    efficacies = [s.self_efficacy for s in students]
    motivations = [s.motivation for s in students]
    blooms = [s.bloom_level for s in students]
    errors = [s.dominant_error_type for s in students]

    def _mean_std(vals):
        mu = sum(vals) / len(vals)
        std = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
        return round(mu, 2), round(std, 2)

    def _pct(vals):
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        return {k: round(v / n * 100) for k, v in counts.items()}

    return {
        "n": n,
        "anxiety": {"mean": _mean_std(anxieties)[0], "std": _mean_std(anxieties)[1]},
        "self_efficacy": {"mean": _mean_std(efficacies)[0], "std": _mean_std(efficacies)[1]},
        "motivation_dist": _pct(motivations),
        "bloom_dist": _pct(blooms),
        "error_dist": _pct(errors),
        "anxiety_high_pct": _pct(["high" if a > 0.7 else "medium" if a > 0.3 else "low" for a in anxieties]),
        "efficacy_low_pct": _pct(["low" if e < 0.3 else "medium" if e < 0.7 else "high" for e in efficacies]),
    }


# ===========================================================================
if __name__ == "__main__":
    kps = ["Linear Dependence and Independence", "The Rank of a Matrix",
           "Subspaces of N-Dimensional Space", "The Determinant of an NxN Matrix"]

    pool = sample_student_pool_from_hachimi(20, kps, base_seed=42)
    stats = compute_pool_statistics(pool)

    print("=== HACHIMI-sampled student pool (n=20) ===")
    print(f"anxiety:      mean={stats['anxiety']['mean']}±{stats['anxiety']['std']}")
    print(f"self_efficacy: mean={stats['self_efficacy']['mean']}±{stats['self_efficacy']['std']}")
    print(f"motivation:   {stats['motivation_dist']}")
    print(f"bloom:        {stats['bloom_dist']}")
    print(f"error:        {stats['error_dist']}")
    print(f"anxiety_high: {stats['anxiety_high_pct']}")
    print(f"efficacy_low: {stats['efficacy_low_pct']}")

    print("\n=== Sample students ===")
    for s in pool[:3]:
        print(f"  {s.student_id}: K_avg={sum(s.kp_mastery.values())/len(s.kp_mastery):.2f} "
              f"C={s.bloom_level} E={s.dominant_error_type} "
              f"A=(anx={s.anxiety:.2f},eff={s.self_efficacy:.2f}) "
              f"motiv={s.motivation} beh={s.behavioral_state}")

    print("\nDONE")
