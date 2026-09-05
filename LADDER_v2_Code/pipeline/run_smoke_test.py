"""
Quick local smoke test for LADDER v2 — no API calls needed.

Runs:
  1. Student model: preset students → behavior prompt generation
  2. Multi-dim gate: same K, different A → different levels
  3. State update: multi-turn evolution of student parameters
  4. Student pool: random generation + divergence stats
"""

import sys
import os

# Add parent to path so we can import ladder_v2
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ladder_v2.student_model import (
    StudentState, PRESET_STUDENTS, DIM_POOLS,
    student_state_to_behavior_prompt,
    update_student_state,
    generate_student_pool,
    student_state_to_missing_kps,
)
from ladder_v2.multidim_gate import compute_multidim_disclosure_level

# Stub KG
class StubKG:
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

kg = StubKG({
    "The Rank of a Matrix": ["Linear Dependence and Independence"],
    "Linear Dependence and Independence": [],
})
REQUIRED = ["Linear Dependence and Independence", "The Rank of a Matrix"]

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} — {detail}")


# ============================================================
# TEST 1: Student model — behavior prompt generation
# ============================================================
print("=" * 60)
print("TEST 1: Behavior prompt generation")
for name in ["anxious_low_k", "confident_mid_k", "giving_up"]:
    s = PRESET_STUDENTS[name]
    p = student_state_to_behavior_prompt(s)
    check(f"  {name}: prompt generated", len(p) > 200, f"length={len(p)}")
    # Should contain dimension-specific keywords
    if name == "anxious_low_k":
        check(f"  {name}: mentions anxiety", "焦虑" in p)
        check(f"  {name}: mentions low efficacy", "不相信" in p or "不行" in p)
    if name == "giving_up":
        check(f"  {name}: mentions giving up", "放弃" in p)

# ============================================================
# TEST 2: Multi-dim gate — same K, different A
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Multi-dim gate — affect sensitivity")

same_k = {"Linear Dependence and Independence": 0.6, "The Rank of a Matrix": 0.3}
anxious = StudentState(student_id="test_anx", kp_mastery=same_k,
    bloom_level="application", dominant_error_type="computational",
    anxiety=0.9, self_efficacy=0.2)
calm = StudentState(student_id="test_calm", kp_mastery=same_k,
    bloom_level="application", dominant_error_type="computational",
    anxiety=0.1, self_efficacy=0.9)

r_anx = compute_multidim_disclosure_level(anxious, REQUIRED, kg)
r_calm = compute_multidim_disclosure_level(calm, REQUIRED, kg)

print(f"  Anxious student: level={r_anx['level']} base={r_anx['base_level']} "
      f"A_adj={r_anx['adjustments']['A_affect']}")
print(f"  Calm student:    level={r_calm['level']} base={r_calm['base_level']} "
      f"A_adj={r_calm['adjustments']['A_affect']}")

check("Same K, anxious > calm level",
      r_anx["level"] > r_calm["level"],
      f"anxious={r_anx['level']}, calm={r_calm['level']}")
check("Anxious gets positive A adjustment",
      r_anx["adjustments"]["A_affect"] > 0)
check("Calm gets zero or negative A adjustment",
      r_calm["adjustments"]["A_affect"] <= 0)

# ============================================================
# TEST 3: Multi-dim gate — error type sensitivity
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Multi-dim gate — error type sensitivity")

conceptual = StudentState(student_id="test_conc", kp_mastery=same_k,
    bloom_level="application", dominant_error_type="conceptual",
    anxiety=0.5, self_efficacy=0.5)
careless = StudentState(student_id="test_care", kp_mastery=same_k,
    bloom_level="application", dominant_error_type="careless",
    anxiety=0.5, self_efficacy=0.5)

r_conc = compute_multidim_disclosure_level(conceptual, REQUIRED, kg)
r_care = compute_multidim_disclosure_level(careless, REQUIRED, kg)

print(f"  Conceptual error: level={r_conc['level']} E_adj={r_conc['adjustments']['E_error']}")
print(f"  Careless error:   level={r_care['level']} E_adj={r_care['adjustments']['E_error']}")

check("Conceptual < careless level (conceptual needs re-teaching)",
      r_conc["level"] < r_care["level"],
      f"conceptual={r_conc['level']}, careless={r_care['level']}")

# ============================================================
# TEST 4: Multi-dim gate — cognitive level sensitivity
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Multi-dim gate — cognitive level sensitivity")

memory_s = StudentState(student_id="test_mem", kp_mastery=same_k,
    bloom_level="memory", dominant_error_type="computational",
    anxiety=0.5, self_efficacy=0.5)
analysis_s = StudentState(student_id="test_ana", kp_mastery=same_k,
    bloom_level="analysis", dominant_error_type="computational",
    anxiety=0.5, self_efficacy=0.5)

r_mem = compute_multidim_disclosure_level(memory_s, REQUIRED, kg)
r_ana = compute_multidim_disclosure_level(analysis_s, REQUIRED, kg)

print(f"  Memory level:   level={r_mem['level']} C_adj={r_mem['adjustments']['C_bloom']}")
print(f"  Analysis level: level={r_ana['level']} C_adj={r_ana['adjustments']['C_bloom']}")

check("Memory > analysis level (memory needs more scaffolding)",
      r_mem["level"] > r_ana["level"],
      f"memory={r_mem['level']}, analysis={r_ana['level']}")

# ============================================================
# TEST 5: All preset students produce valid levels
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: All preset students → valid levels [0-3]")
for name, s in PRESET_STUDENTS.items():
    r = compute_multidim_disclosure_level(s, REQUIRED, kg)
    check(f"  {name}: level={r['level']} in [0,3]",
          0 <= r['level'] <= 3,
          f"got {r['level']}")
    check(f"  {name}: has permission",
          r['permission'] in {"Socratic_Only", "Hint_Allowed", "Partial_Allowed", "Direct_Allowed"})

# ============================================================
# TEST 6: "Giving up" student never gets L0
# ============================================================
print("\n" + "=" * 60)
print("TEST 6: Giving-up student → level >= 1")
gu = PRESET_STUDENTS["giving_up"]
r_gu = compute_multidim_disclosure_level(gu, REQUIRED, kg)
check("Giving-up student >= L1 (never pure Socratic)",
      r_gu["level"] >= 1,
      f"got L{r_gu['level']}")

# ============================================================
# TEST 7: State update — multi-turn evolution
# ============================================================
print("\n" + "=" * 60)
print("TEST 7: State update — multi-turn evolution")
s = PRESET_STUDENTS["anxious_low_k"]
a0, e0 = s.anxiety, s.self_efficacy
# Simulate 3 wrong answers
for i in range(3):
    s = update_student_state(s, student_response_correctness=False, teacher_level_given=0)
check("Anxiety increases after repeated failure", s.anxiety > a0,
      f"{a0:.2f} → {s.anxiety:.2f}")
check("Self-efficacy drops after repeated failure", s.self_efficacy < e0,
      f"{e0:.2f} → {s.self_efficacy:.2f}")
check("Trajectory becomes declining", s.trajectory == "declining",
      f"got {s.trajectory}")

# Then 2 correct answers
for i in range(2):
    s = update_student_state(s, student_response_correctness=True, teacher_level_given=1)
check("Anxiety decreases after success", s.anxiety < 1.0,
      f"anxiety={s.anxiety:.2f}")
check("Trajectory improves", s.trajectory == "improving",
      f"got {s.trajectory}")

# ============================================================
# TEST 8: Student pool generation
# ============================================================
print("\n" + "=" * 60)
print("TEST 8: Student pool generation")
pool = generate_student_pool(20, ["kp1", "kp2", "kp3"], seed=42)
check("Correct pool size", len(pool) == 20)
check("All have unique IDs", len(set(s.student_id for s in pool)) == 20)
check("All dimensions populated",
      all(s.bloom_level in DIM_POOLS["bloom_level"] for s in pool))
check("Mastery values in [0,1]",
      all(0 <= m <= 1 for s in pool for m in s.kp_mastery.values()))

# Verify dimensionality coverage
bloom_set = set(s.bloom_level for s in pool)
check("Pool covers multiple Bloom levels", len(bloom_set) >= 2,
      f"only {len(bloom_set)}: {bloom_set}")

# ============================================================
# TEST 9: Cross-dimension interaction
# ============================================================
print("\n" + "=" * 60)
print("TEST 9: Cross-dimension interaction effects")

# High anxiety + conceptual error → should get compensation
hi_anx_conc = StudentState(student_id="cross1", kp_mastery=same_k,
    bloom_level="application", dominant_error_type="conceptual",
    anxiety=0.9, self_efficacy=0.5)
r_cross = compute_multidim_disclosure_level(hi_anx_conc, REQUIRED, kg)
check("Cross-dimension adjustment applied",
      r_cross["adjustments"]["cross_interaction"] != 0,
      f"cross_adj={r_cross['adjustments']['cross_interaction']}")

# ============================================================
print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed}")
print(f"{'ALL PASSED' if failed == 0 else 'SOME FAILED — see above'}")
print(f"{'='*60}")
