"""Unit tests for the LAMMPS `default_run_command` env overrides.

`SCILINK_LAMMPS_BIN` picks an explicit binary (e.g. a cluster MPI build not on
PATH), and `SCILINK_MPI_LAUNCHER` prefixes a parallel launcher — so a site can
run LAMMPS across its allocated cores without any launch detail in the skill.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.skills.molecular_dynamics.lammps import lammps as L  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("SCILINK_LAMMPS_BIN", raising=False)
    monkeypatch.delenv("SCILINK_MPI_LAUNCHER", raising=False)


def test_explicit_binary_overrides_path(monkeypatch):
    # An existing file is used verbatim, PATH resolution untouched.
    binary = sys.executable  # any real, executable path
    monkeypatch.setenv("SCILINK_LAMMPS_BIN", binary)
    assert L.default_run_command("run.lammps") == f"{binary} -in run.lammps"


def test_launcher_prefixes_command(monkeypatch):
    binary = sys.executable
    monkeypatch.setenv("SCILINK_LAMMPS_BIN", binary)
    monkeypatch.setenv("SCILINK_MPI_LAUNCHER", "srun")
    assert L.default_run_command("run.lammps") == f"srun {binary} -in run.lammps"


def test_launcher_with_args(monkeypatch):
    binary = sys.executable
    monkeypatch.setenv("SCILINK_LAMMPS_BIN", binary)
    monkeypatch.setenv("SCILINK_MPI_LAUNCHER", "mpirun -np 8")
    assert L.default_run_command() == f"mpirun -np 8 {binary} -in {{script}}"


def test_missing_binary_falls_back_to_path(monkeypatch):
    # A bogus override is ignored; resolution falls through to check_lammps.
    monkeypatch.setenv("SCILINK_LAMMPS_BIN", "/no/such/lmp_binary_xyz")
    monkeypatch.setattr(L, "check_lammps",
                        lambda: {"available": True, "path": "/usr/bin/lmp"})
    assert L.default_run_command("in.lmp") == "/usr/bin/lmp -in in.lmp"


def test_no_binary_anywhere_returns_none(monkeypatch):
    monkeypatch.setattr(L, "check_lammps",
                        lambda: {"available": False, "path": None})
    assert L.default_run_command() is None


def test_launcher_ignored_when_no_binary(monkeypatch):
    # No binary resolvable -> None, even with a launcher set (nothing to launch).
    monkeypatch.setenv("SCILINK_MPI_LAUNCHER", "srun")
    monkeypatch.setattr(L, "check_lammps",
                        lambda: {"available": False, "path": None})
    assert L.default_run_command() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
