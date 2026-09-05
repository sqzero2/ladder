"""
Adaptive Tutor Module

A system-prompt-based intelligent tutoring approach that adapts responses based on student's knowledge state.
Given a student's knowledge mastery state and a question, the system:
1. Creates an adaptive system prompt containing the student's known and missing knowledge points
2. Sends the question with the adaptive prompt to get personalized tutoring response

This approach differs from Context-Based Tutor by embedding knowledge state directly in the system prompt
rather than using a multi-step LangGraph workflow.
"""

from .run import run_adaptive_tutor, run_adaptive_tutor_async

__all__ = ['run_adaptive_tutor', 'run_adaptive_tutor_async']

