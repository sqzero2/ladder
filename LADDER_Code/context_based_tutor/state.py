"""
CogWall state definitions.
Extends SHaPE's TutorState with Extract Layer and Ticket fields.
"""

from typing import List, TypedDict, Optional, Dict, Any


class StepDict(TypedDict):
    description: str
    kps: List[str]


class RoutingTicket(TypedDict, total=False):
    """Structured ticket from Agent A to Agent B."""
    permission: str          # "Direct_Allowed" | "Socratic_Only"
    missing_concepts: List[str]
    required_kps: List[str]
    steps: List[StepDict]
    parse_failed: bool


class TutorState(TypedDict, total=False):
    # --- Original SHaPE fields ---
    question: str
    kp_vocab: List[str]
    steps: List[StepDict]
    required_kps: List[str]
    mastered_kps: List[str]
    missing_kps: List[str]
    direct_answer: Optional[str]
    tutoring_answer: Optional[str]
    student_csv_path: Optional[str]
    student_state: Any             # Six-dimension StudentState (LADDER v2)
    extra: Dict[str, Any]
    fallback_used: bool

    # --- CogWall: Extract Layer ---
    raw_question: str              # Original user input (may contain attacks)
    clean_question: str            # After Extract Layer (clean math problem)
    extract_failed: bool           # Extract Layer failed → ticket=Socratic_Only

    # --- CogWall: Agent A (PDP) output ---
    routing_ticket: Dict[str, Any] # Structured ticket passed to Agent B

    # --- CogWall: Agent B (PEP) enforcement ---
    enforced_permission: str       # The permission Agent B MUST follow (set by code)
