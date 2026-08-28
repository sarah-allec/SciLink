"""Adaptive convergence loop: SciLink decides how many replicas to run.

The replica runner and per-replica measurement are injected, so the decision
logic is exercised with no MD: a replica's "value" is a function of its seed.
"""

import pytest

from scilink.agents.sim_agents.convergence import (
    default_convergence_judge, relative_sem, run_convergence_loop,
    run_convergence_loop_batched)


def _loop(values_by_seed, plateau_by_seed=None, **kw):
    """Run the loop with seed -> value (and optional seed -> plateau)."""
    def run(seed):
        return seed
    def measure(seed):
        e = {"value": values_by_seed(seed)}
        if plateau_by_seed is not None:
            e["plateau_reached"] = plateau_by_seed(seed)
        return e
    return run_convergence_loop(run_replica_fn=run, measure_fn=measure, **kw)


def test_relative_sem_basic():
    assert relative_sem([2.0]) is None                 # need >= 2
    assert relative_sem([2.0, 2.0, 2.0]) == 0.0        # no spread
    r = relative_sem([2.0, 2.2, 1.8])
    assert r is not None and r > 0


def test_converges_when_spread_is_small():
    # Tight cluster around 2.7 -> relative SEM tiny -> converges at min_replicas.
    vals = {0: 2.70, 1: 2.72, 2: 2.68}
    r = _loop(lambda s: vals[s], min_replicas=3, max_replicas=8)
    assert r["converged"] is True
    assert r["n_replicas"] == 3
    assert r["mean"] == pytest.approx(2.70, abs=0.02)
    assert r["uncertainty"] < 0.1


def test_noisy_never_converges_hits_cap():
    # Wildly alternating values -> relative SEM stays large -> runs to the cap.
    r = _loop(lambda s: 1.0 if s % 2 == 0 else 9.0,
              min_replicas=3, max_replicas=6)
    assert r["converged"] is False
    assert r["n_replicas"] == 6
    assert "hit max_replicas=6" in r["reason"]


def test_unplateaued_replica_blocks_convergence():
    # Values are tight, but one replica's running integral never plateaued.
    vals = {0: 2.70, 1: 2.71, 2: 2.69}
    r = _loop(lambda s: vals[s], plateau_by_seed=lambda s: s != 1,
              min_replicas=3, max_replicas=3)
    assert r["converged"] is False
    assert r["n_replicas"] == 3


def test_min_replicas_enforced_before_declaring_converged():
    # Even identical values do not converge before min_replicas is reached.
    r = _loop(lambda s: 2.7, min_replicas=4, max_replicas=10)
    assert r["converged"] is True
    assert r["n_replicas"] == 4                         # not 2 or 3


def test_default_judge_reports_fields():
    est = [{"value": 2.7}, {"value": 2.72}, {"value": 2.68}]
    v = default_convergence_judge(est, target_rel_sem=0.1, min_replicas=3)
    assert v["converged"] is True and v["n"] == 3
    assert v["all_plateaued"] is True and v["mean"] == pytest.approx(2.70, abs=0.02)


def _batched(values_by_seed, plateau_by_seed=None, **kw):
    def run_batch(seeds):
        return list(seeds)                                 # parallel array -> outputs
    def measure(seed):
        e = {"value": values_by_seed(seed)}
        if plateau_by_seed is not None:
            e["plateau_reached"] = plateau_by_seed(seed)
        return e
    return run_convergence_loop_batched(
        run_batch_fn=run_batch, measure_fn=measure, **kw)


def test_batched_converges_after_first_batch():
    vals = {0: 2.70, 1: 2.72, 2: 2.68, 3: 2.71}
    r = _batched(lambda s: vals[s], initial_batch=4, increment=4,
                 min_replicas=3, max_replicas=12)
    assert r["converged"] is True
    assert r["n_replicas"] == 4 and r["n_batches"] == 1


def test_batched_adds_increments_until_cap():
    # Alternating wide values never converge -> submits batches up to the cap.
    r = _batched(lambda s: 1.0 if s % 2 == 0 else 9.0,
                 initial_batch=4, increment=4, min_replicas=3, max_replicas=8)
    assert r["converged"] is False
    assert r["n_replicas"] == 8 and r["n_batches"] == 2
    assert "hit max_replicas=8" in r["reason"]


def test_batched_partial_batch_does_not_exceed_cap():
    r = _batched(lambda s: 1.0 if s % 2 == 0 else 9.0,
                 initial_batch=6, increment=6, min_replicas=3, max_replicas=8)
    assert r["n_replicas"] == 8                            # 6 then a clamped 2


def test_guards_on_bad_bounds():
    with pytest.raises(ValueError):
        run_convergence_loop(run_replica_fn=lambda s: s,
                             measure_fn=lambda s: {"value": 1.0}, min_replicas=1)
    with pytest.raises(ValueError):
        run_convergence_loop(run_replica_fn=lambda s: s,
                             measure_fn=lambda s: {"value": 1.0},
                             min_replicas=5, max_replicas=3)
