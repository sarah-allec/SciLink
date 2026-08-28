"""Adaptive statistical convergence for a noisy simulation observable.

Some observables — Green-Kubo transport coefficients above all — are only
trustworthy once averaged over enough independent replicas that the estimate of
the *mean* has settled. This module runs replicas until SciLink judges the
running mean converged (the inter-replica spread of the mean falls below a
target) or a cap is hit, so the AMOUNT of sampling is decided by the data, not
fixed up front.

Engine-neutral and dependency-injected — the replica runner and the per-replica
measurement are passed in — so the decision logic is unit-testable without any
MD. A caller on an HPC system provides a ``run_replica_fn`` that launches one
independent trajectory (a fresh seed) and a ``measure_fn`` that reduces it to a
scalar estimate (e.g. the ``viscosity_greenkubo`` skill).
"""

from __future__ import annotations

import statistics
from typing import Any, Callable, Dict, List, Optional


def relative_sem(values: List[float]) -> Optional[float]:
    """Relative standard error of the mean: ``std/sqrt(n) / |mean|``.

    Returns ``None`` when there are fewer than two values or the mean is ~0
    (a relative spread is undefined) — callers treat that as not-yet-converged.
    """
    n = len(values)
    if n < 2:
        return None
    mean = statistics.fmean(values)
    if abs(mean) < 1e-12:
        return None
    return (statistics.stdev(values) / (n ** 0.5)) / abs(mean)


def default_convergence_judge(estimates: List[Dict[str, Any]], *,
                              target_rel_sem: float = 0.1,
                              min_replicas: int = 3) -> Dict[str, Any]:
    """Judge a set of per-replica estimates converged.

    Converged when there are at least ``min_replicas`` estimates, EVERY replica's
    running integral plateaued (``plateau_reached``, default True when unstated),
    and the relative standard error of the mean is at most ``target_rel_sem``.
    Returns ``{"converged", "mean", "rel_sem", "n", "all_plateaued"}``.
    """
    values = [e["value"] for e in estimates if e.get("value") is not None]
    plateaus = [bool(e.get("plateau_reached", True)) for e in estimates]
    n = len(values)
    mean = statistics.fmean(values) if values else None
    rsem = relative_sem(values)
    all_plateaued = all(plateaus) if plateaus else False
    converged = bool(n >= min_replicas and all_plateaued
                     and rsem is not None and rsem <= target_rel_sem)
    return {"converged": converged, "mean": mean, "rel_sem": rsem, "n": n,
            "all_plateaued": all_plateaued}


def run_convergence_loop(
    *,
    run_replica_fn: Callable[[int], Any],
    measure_fn: Callable[[Any], Dict[str, Any]],
    judge_fn: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
    min_replicas: int = 3,
    max_replicas: int = 12,
    seed0: int = 0,
) -> Dict[str, Any]:
    """Accumulate independent replicas until the observable's mean converges.

    Args:
        run_replica_fn: ``seed -> replica output`` (launch one independent
            trajectory; the return is whatever ``measure_fn`` consumes).
        measure_fn: ``replica output -> {"value": float, "plateau_reached": bool,
            ...}`` — reduce one replica to a scalar estimate.
        judge_fn: ``estimates -> {"converged": bool, "mean", "rel_sem", ...}``.
            Defaults to :func:`default_convergence_judge` bound to ``min_replicas``.
        min_replicas, max_replicas: run at least / at most this many.
        seed0: first seed; replica ``i`` uses ``seed0 + i``.

    Returns ``{"converged", "mean", "uncertainty" (relative SEM), "n_replicas",
    "estimates", "reason"}``. Stops as soon as the judge reports converged; on
    reaching the cap unconverged it returns the best estimate with
    ``converged=False`` (the caller surfaces it as not-yet-verified).
    """
    if min_replicas < 2:
        raise ValueError("min_replicas must be >= 2 (a spread needs >=2 samples)")
    if max_replicas < min_replicas:
        raise ValueError("max_replicas must be >= min_replicas")
    judge = judge_fn or (lambda est: default_convergence_judge(
        est, min_replicas=min_replicas))

    estimates: List[Dict[str, Any]] = []
    verdict: Dict[str, Any] = {"converged": False, "mean": None, "rel_sem": None}
    for i in range(max_replicas):
        estimates.append(measure_fn(run_replica_fn(seed0 + i)))
        if len(estimates) < min_replicas:
            continue
        verdict = judge(estimates)
        if verdict.get("converged"):
            break
    return {
        "converged": bool(verdict.get("converged")),
        "mean": verdict.get("mean"),
        "uncertainty": verdict.get("rel_sem"),
        "n_replicas": len(estimates),
        "estimates": estimates,
        "reason": ("converged" if verdict.get("converged")
                   else f"hit max_replicas={max_replicas} without converging"),
    }


def run_convergence_loop_batched(
    *,
    run_batch_fn: Callable[[List[int]], List[Any]],
    measure_fn: Callable[[Any], Dict[str, Any]],
    judge_fn: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
    initial_batch: int = 4,
    increment: int = 4,
    min_replicas: int = 3,
    max_replicas: int = 12,
    seed0: int = 0,
) -> Dict[str, Any]:
    """Converge an observable by submitting replicas in PARALLEL batches.

    The controller model for a scheduler: ``run_batch_fn(seeds)`` launches every
    seed at once (e.g. a SLURM array) and returns their outputs once done; the
    loop judges convergence after each batch and submits another ``increment``
    replicas until converged or the ``max_replicas`` cap. Outputs that fail to
    come back are simply not measured, so a partial batch still makes progress.
    Same return shape as :func:`run_convergence_loop`, plus ``n_batches``.

    Injected ``run_batch_fn``/``measure_fn`` keep the decision logic MD-free and
    unit-testable; the caller supplies the array submit-and-wait.
    """
    if min_replicas < 2:
        raise ValueError("min_replicas must be >= 2 (a spread needs >=2 samples)")
    if max_replicas < min_replicas:
        raise ValueError("max_replicas must be >= min_replicas")
    if initial_batch < 1 or increment < 1:
        raise ValueError("initial_batch and increment must be >= 1")
    judge = judge_fn or (lambda est: default_convergence_judge(
        est, min_replicas=min_replicas))

    estimates: List[Dict[str, Any]] = []
    verdict: Dict[str, Any] = {"converged": False, "mean": None, "rel_sem": None}
    next_seed = seed0
    n_batches = 0
    batch = initial_batch
    while len(estimates) < max_replicas:
        k = min(batch, max_replicas - len(estimates))
        seeds = list(range(next_seed, next_seed + k))
        next_seed += k
        outputs = run_batch_fn(seeds) or []
        n_batches += 1
        estimates.extend(measure_fn(o) for o in outputs)
        if len(estimates) >= min_replicas:
            verdict = judge(estimates)
            if verdict.get("converged"):
                break
        batch = increment
    return {
        "converged": bool(verdict.get("converged")),
        "mean": verdict.get("mean"),
        "uncertainty": verdict.get("rel_sem"),
        "n_replicas": len(estimates),
        "n_batches": n_batches,
        "estimates": estimates,
        "reason": ("converged" if verdict.get("converged")
                   else f"hit max_replicas={max_replicas} without converging"),
    }
