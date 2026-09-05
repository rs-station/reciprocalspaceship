"""errors and warnings shared by phase-alignment algorithms"""

from __future__ import annotations


class PhaseAlignmentInputError(ValueError):
    """Raised when phase-alignment inputs are invalid or insufficient."""


class PhaseAlignmentOptimizationError(RuntimeError):
    """Raised when continuous phase alignment does not converge."""


class NoClearSolutionError(RuntimeError):
    """Raised when correlation scores do not identify a reliable solution."""


class LowCorrelationWarning(UserWarning):
    """Warned when the selected solution has low correlation."""
