"""
LADDER v2: Multi-dimensional student model + gate + simulator + diagnosis + PCK.

Seven-domain alignment:
  KNOW-PCK  → pck.py           (5 linear algebra concepts)
  DIAGNOSE  → diagnoser.py     (infer affect/error from dialogue)
  ADAPT     → multidim_gate.py (6D disclosure decisions)
  INTERACT  → interact_prompts.py (emotion-aware tone + PCK injection)
  META      → (future)
  REFLECT   → (future)
  ETHICS    → (implicit in capability-deprivation prompts)

Infrastructure:
  student_model.py      — 6D StudentState + behavior prompts + state updater
  student_simulator.py  — Multi-turn LLM student driven by behavior prompt
"""

from .student_model import (
    StudentState,
    student_state_to_behavior_prompt,
    update_student_state,
    generate_student_pool,
    student_state_to_missing_kps,
    from_hachimi_profile,
    PRESET_STUDENTS,
    DIM_POOLS,
)
from .multidim_gate import compute_multidim_disclosure_level
from .student_simulator import StudentSimulator
from .diagnoser import Diagnoser
from .pck import get_pck, get_misconception_hint, get_entry_point, PCK_BASE
from .interact_prompts import (
    get_affective_wrapper,
    get_pck_hint,
    build_affective_agent_prompt,
)

__all__ = [
    # Student model
    "StudentState",
    "student_state_to_behavior_prompt",
    "update_student_state",
    "generate_student_pool",
    "student_state_to_missing_kps",
    "from_hachimi_profile",
    "PRESET_STUDENTS",
    "DIM_POOLS",
    # Gate
    "compute_multidim_disclosure_level",
    # Simulator
    "StudentSimulator",
    # Diagnoser
    "Diagnoser",
    # PCK
    "get_pck",
    "get_misconception_hint",
    "get_entry_point",
    "PCK_BASE",
    # Interaction
    "get_affective_wrapper",
    "get_pck_hint",
    "build_affective_agent_prompt",
]
