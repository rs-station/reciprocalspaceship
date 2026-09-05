"""errors and warnings shared by phase-alignment algorithms"""

from __future__ import annotations


class PhaseAlignmentInputError(ValueError):
    """Raised when phase-alignment inputs are invalid or insufficient."""
