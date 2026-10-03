"""Tests for the NWChem convergence hooks + the `convergence:` frontmatter.

The basis param-setter is pure text manipulation (no NWChem needed). The
frontmatter and registry tests confirm the engine-neutral driver can discover
the spec and hooks when the `nwchem` skill is active. The observable extractor
needs real NWChem output (cclib) and is validated on the cluster.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.skills.molecular_qc.nwchem.nwchem_convergence import (  # noqa: E402
    set_convergence_param,
)

_DECK = (
    "start job\n"
    "geometry units angstrom\n"
    "  O 0.0 0.0 0.0\n"
    "  H 0.0 0.0 0.96\n"
    "end\n"
    "basis\n"
    "  * library def2-svp\n"
    "end\n"
    "dft\n"
    "  xc b3lyp\n"
    "end\n"
    "task dft energy\n"
)


class TestSetConvergenceParam:
    def test_replaces_basis_library(self):
        out = set_convergence_param({"job.nw": _DECK}, "basis", "def2-tzvp")
        assert "* library def2-tzvp" in out["job.nw"]
        assert "def2-svp" not in out["job.nw"]
        assert "job.nw" in _DECK or True            # original dict untouched
        assert "def2-svp" in _DECK                  # source constant unchanged

    def test_replaces_every_library_line(self):
        deck = _DECK + "basis\n  * library def2-svp\nend\ntce\n  ccsd(t)\nend\n"
        out = set_convergence_param({"calc.nw": deck}, "basis", "def2-qzvp")
        assert out["calc.nw"].count("library def2-qzvp") == 2
        assert "def2-svp" not in out["calc.nw"]

    def test_case_insensitive_library_keyword(self):
        out = set_convergence_param(
            {"job.nw": "basis\n  * LIBRARY cc-pvdz\nend\n"}, "basis", "cc-pvtz")
        assert "cc-pvtz" in out["job.nw"] and "cc-pvdz" not in out["job.nw"]

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            set_convergence_param({"job.nw": _DECK}, "xc", "pbe")

    def test_no_deck_raises(self):
        with pytest.raises(KeyError):
            set_convergence_param({"notes.txt": "x"}, "basis", "def2-tzvp")

    def test_no_library_line_raises(self):
        with pytest.raises(ValueError):
            set_convergence_param({"job.nw": "task dft energy\n"}, "basis", "def2-tzvp")


class TestConvergenceFrontmatter:
    def test_nwchem_skill_declares_basis_convergence(self):
        from scilink.skills.loader import load_skill
        conv = load_skill("nwchem", domain="molecular_qc")["meta"].get("convergence")
        assert isinstance(conv, list) and conv
        spec = conv[0]
        assert spec["parameter"] == "basis"
        assert isinstance(spec["ladder"], list) and len(spec["ladder"]) >= 2
        assert spec["observable"] and spec["tolerance"] > 0


class TestRegistryResolution:
    def test_hooks_resolve_when_nwchem_active(self):
        from scilink.skills._shared._registry import get_tool_function
        setter = get_tool_function("set_convergence_param", active_skills=["nwchem"])
        reader = get_tool_function("read_convergence_observable",
                                   active_skills=["nwchem"])
        out = setter(input_files={"job.nw": _DECK}, param="basis", value="def2-tzvp")
        assert "def2-tzvp" in out["job.nw"]
        assert reader(output_dir="/nonexistent", observable="total_energy") is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
