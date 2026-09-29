"""Numeric parameter-convergence comparator (engine-neutral).

A static calculation's accuracy depends on numerical parameters (VASP ENCUT and
k-points, QE ecutwfc/ecutrho, a QC basis-set ladder). The convention is to run
the calculation at increasing settings and take the cheapest setting past which
the observable stops changing within a tolerance. This module holds the pure
decision — "given an observable measured along a ladder of settings, where does
it plateau?" — with no engine, LLM, or I/O. The engine skills declare the ladder,
the observable, and the tolerance; the sweep driver runs the ladder and feeds the
results here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


@dataclass
class ConvergenceResult:
    """Outcome of assessing one parameter ladder.

    Attributes:
        converged: True when a demonstrated plateau exists — the observable at
            some setting and every higher setting agree with the most-accurate
            (highest) setting within ``tolerance``, and at least two settings
            take part (a single point cannot demonstrate a plateau).
        setting: The cheapest setting that is converged (the one to adopt), or
            None when not converged.
        value: The observable at the most-accurate (highest) setting — the best
            available estimate — or None when there is nothing to report.
        deltas: ``(setting, |value - best_value|)`` for each setting, in input
            order; the evidence behind the verdict.
        reason: Short human-readable explanation.
    """

    converged: bool
    setting: Any
    value: Optional[float]
    deltas: List[Tuple[Any, float]]
    reason: str


def converged_setting(
    observations: Sequence[Tuple[Any, Optional[float]]],
    tolerance: float,
) -> ConvergenceResult:
    """Find where an observable plateaus along a ladder of settings.

    Args:
        observations: ``(setting, value)`` pairs ordered from least to most
            accurate/expensive (e.g. ascending ENCUT). ``value`` is the
            convergence observable (energy/atom, lattice constant, gap, …).
            A ``None`` value marks a setting whose run failed or produced no
            reading; such settings cannot anchor a plateau.
        tolerance: Maximum absolute difference (in the observable's units) that
            counts as "unchanged". Must be non-negative.

    Returns:
        A :class:`ConvergenceResult`. The adopted setting is the *cheapest* one
        from which the observable no longer moves beyond ``tolerance`` relative
        to the most-accurate setting, so downstream work runs at the cheapest
        trustworthy setting while the reported value is the most accurate.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    finite = [(s, v) for s, v in observations if v is not None]
    if not finite:
        return ConvergenceResult(False, None, None, [], "no readable observations")
    if len(finite) == 1:
        # One point cannot demonstrate a plateau, but it is the best estimate.
        s, v = finite[0]
        return ConvergenceResult(
            False, None, v, [(s, 0.0)],
            "only one setting produced a value — cannot demonstrate a plateau")

    best_value = finite[-1][1]  # highest (most accurate) setting
    deltas = [(s, abs(v - best_value)) for s, v in finite]

    # The cheapest setting from which every higher setting (inclusive) is within
    # tolerance of the best value. Walk from cheapest up; the first setting whose
    # own delta AND all deltas above it are within tolerance is the plateau start.
    converged_idx: Optional[int] = None
    for i in range(len(finite)):
        if all(d <= tolerance for _, d in deltas[i:]):
            converged_idx = i
            break

    if converged_idx is None or converged_idx >= len(finite) - 1:
        # Either nothing is within tolerance of the top, or only the top setting
        # agrees with itself — the ladder has not shown a plateau. Report the
        # best value so the caller can widen the ladder, but do not claim
        # convergence.
        return ConvergenceResult(
            False, None, best_value, deltas,
            "observable still changing at the top of the ladder — extend it")

    setting = finite[converged_idx][0]
    return ConvergenceResult(
        True, setting, best_value, deltas,
        f"converged at {setting}: observable within {tolerance} of the "
        f"most-accurate setting from here up")
