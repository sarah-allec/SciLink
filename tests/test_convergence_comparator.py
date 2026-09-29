"""Unit tests for the engine-neutral convergence comparator."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.convergence import (  # noqa: E402
    converged_setting, ConvergenceResult, run_convergence_sweep, SweepResult,
)


def test_clear_plateau_adopts_cheapest_trustworthy_setting():
    # Energy/atom (eV) vs ENCUT (eV): flat from 400 up, within 1 meV.
    obs = [(300, -5.20), (400, -5.401), (500, -5.4015), (600, -5.4012)]
    r = converged_setting(obs, tolerance=0.001)
    assert r.converged is True
    assert r.setting == 400            # cheapest within tol of the top
    assert r.value == -5.4012          # value from the most-accurate setting


def test_not_converged_when_still_drifting_at_top():
    obs = [(300, -5.0), (400, -5.2), (500, -5.35), (600, -5.47)]
    r = converged_setting(obs, tolerance=0.001)
    assert r.converged is False
    assert r.setting is None
    assert r.value == -5.47            # best estimate still reported
    assert "extend" in r.reason


def test_single_setting_cannot_demonstrate_plateau():
    r = converged_setting([(500, -5.4)], tolerance=0.01)
    assert r.converged is False
    assert r.setting is None
    assert r.value == -5.4


def test_empty_observations():
    r = converged_setting([], tolerance=0.01)
    assert r.converged is False and r.value is None
    assert "no readable" in r.reason


def test_none_values_are_skipped():
    # The 500 run failed; the plateau is still demonstrable from 400 vs 600.
    obs = [(300, -5.2), (400, -5.401), (500, None), (600, -5.4012)]
    r = converged_setting(obs, tolerance=0.001)
    assert r.converged is True
    assert r.setting == 400


def test_only_top_agrees_is_not_a_plateau():
    # Every cheaper setting is outside tol of the top; only the top agrees with
    # itself — not a demonstrated plateau.
    obs = [(300, -5.0), (400, -5.30), (500, -5.42)]
    r = converged_setting(obs, tolerance=0.001)
    assert r.converged is False
    assert r.value == -5.42


def test_tolerance_boundary_is_inclusive():
    obs = [(400, -5.400), (500, -5.401)]   # delta exactly 0.001
    assert converged_setting(obs, tolerance=0.001).converged is True
    assert converged_setting(obs, tolerance=0.0009).converged is False


def test_lattice_constant_ladder_over_kpoints():
    # k-mesh density ladder; lattice constant (Å) plateaus at 6x6x6.
    obs = [("2x2x2", 3.68), ("4x4x4", 3.615), ("6x6x6", 3.611), ("8x8x8", 3.612)]
    r = converged_setting(obs, tolerance=0.005)
    assert r.converged is True
    assert r.setting == "4x4x4"        # within 0.005 Å of the top from here up
    assert r.value == 3.612


def test_negative_tolerance_rejected():
    with pytest.raises(ValueError):
        converged_setting([(1, 1.0), (2, 1.0)], tolerance=-0.1)


def test_result_is_dataclass_with_deltas():
    r = converged_setting([(400, -5.401), (500, -5.4012)], tolerance=0.001)
    assert isinstance(r, ConvergenceResult)
    assert [s for s, _ in r.deltas] == [400, 500]


# ---------------------------------------------------------------------------
# run_convergence_sweep — engine-neutral driver, tested with fakes
# ---------------------------------------------------------------------------

def _fake_energies(mapping):
    """read_observable that maps a run_dir string to a canned energy."""
    return lambda run_dir: mapping.get(run_dir)


def test_sweep_converges_and_reports_cheapest_setting():
    ladder = [300, 400, 500, 600]
    built = {}

    def build_member(setting):
        m = {"INCAR": f"ENCUT = {setting}", "POSCAR": "..."}
        built[setting] = m
        return m

    def run_ladder(members):
        # Verify each rung got its own param-set deck, then "run" it.
        assert set(members) == set(ladder)
        return {s: f"/run/{s}" for s in members}

    energies = {"/run/300": -5.20, "/run/400": -5.401,
                "/run/500": -5.4015, "/run/600": -5.4012}

    res = run_convergence_sweep(
        ladder=ladder, build_member=build_member, run_ladder=run_ladder,
        read_observable=_fake_energies(energies), tolerance=0.001,
        param_name="ENCUT",
    )
    assert isinstance(res, SweepResult)
    assert res.param_name == "ENCUT"
    assert res.convergence.converged is True
    assert res.convergence.setting == 400
    assert res.convergence.value == -5.4012
    assert built[600]["INCAR"] == "ENCUT = 600"          # param-setter ran per rung
    assert res.observations[0] == (300, -5.20)


def test_sweep_marks_failed_rung_as_none():
    ladder = [300, 400, 500]

    def run_ladder(members):
        # The 400 rung failed to produce a run dir.
        return {300: "/run/300", 500: "/run/500"}

    res = run_convergence_sweep(
        ladder=ladder, build_member=lambda s: {"INCAR": str(s)},
        run_ladder=run_ladder,
        read_observable=_fake_energies({"/run/300": -5.0, "/run/500": -5.4}),
        tolerance=0.001,
    )
    assert res.observations == [(300, -5.0), (400, None), (500, -5.4)]
    assert res.convergence.converged is False   # only two points, still drifting


def test_sweep_unreadable_observable_is_none():
    ladder = [400, 500]

    res = run_convergence_sweep(
        ladder=ladder, build_member=lambda s: {"INCAR": str(s)},
        run_ladder=lambda m: {s: f"/run/{s}" for s in m},
        read_observable=lambda run_dir: None,   # parser found nothing
        tolerance=0.001,
    )
    assert [v for _, v in res.observations] == [None, None]
    assert res.convergence.converged is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
