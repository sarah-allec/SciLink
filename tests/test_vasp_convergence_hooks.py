"""Tests for the VASP convergence hooks + the `convergence:` frontmatter.

The param-setter is pure string manipulation (no VASP needed). The frontmatter
and registry tests confirm the engine-neutral driver can discover the spec and
the hooks for the active engine. The observable extractor needs real vasprun.xml
output and is validated on the cluster, not here.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.skills.periodic_dft.vasp.vasp_convergence import (  # noqa: E402
    set_convergence_param,
)


class TestSetConvergenceParam:
    def test_replaces_existing_encut(self):
        deck = {"INCAR": "PREC = Accurate\nENCUT = 400\nISMEAR = 0\n",
                "POSCAR": "..."}
        out = set_convergence_param(deck, "ENCUT", 600)
        assert "ENCUT = 600" in out["INCAR"]
        assert "ENCUT = 400" not in out["INCAR"]
        assert out["INCAR"].count("ENCUT") == 1
        assert deck["INCAR"].count("400") == 1   # original not mutated

    def test_appends_encut_when_absent(self):
        deck = {"INCAR": "PREC = Accurate\n", "POSCAR": "..."}
        out = set_convergence_param(deck, "ENCUT", 500)
        assert "ENCUT = 500" in out["INCAR"]

    def test_kspacing_sets_incar_and_drops_kpoints(self):
        deck = {"INCAR": "PREC = Accurate\n", "KPOINTS": "mesh\n0\nG\n4 4 4\n",
                "POSCAR": "..."}
        out = set_convergence_param(deck, "k-points", 0.2)
        assert "KSPACING = 0.2" in out["INCAR"]
        assert "KPOINTS" not in out          # dropped so KSPACING governs
        assert "KPOINTS" in deck             # original untouched

    def test_replaces_existing_kspacing(self):
        deck = {"INCAR": "KSPACING = 0.5\nPREC = Accurate\n"}
        out = set_convergence_param(deck, "k-points", 0.15)
        assert "KSPACING = 0.15" in out["INCAR"]
        assert "0.5" not in out["INCAR"]

    def test_case_insensitive_existing_key(self):
        deck = {"INCAR": "encut = 400\n"}
        out = set_convergence_param(deck, "ENCUT", 600)
        assert "ENCUT = 600" in out["INCAR"]
        assert "400" not in out["INCAR"]

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            set_convergence_param({"INCAR": ""}, "SIGMA", 0.1)

    def test_missing_incar_raises(self):
        with pytest.raises(KeyError):
            set_convergence_param({"POSCAR": "..."}, "ENCUT", 500)


class TestConvergenceFrontmatter:
    def test_vasp_skill_declares_convergence_block(self):
        from scilink.skills.loader import load_skill
        meta = load_skill("vasp", domain="periodic_dft")["meta"]
        conv = meta.get("convergence")
        assert isinstance(conv, list) and conv, "no convergence block in vasp.md"
        params = {c["parameter"] for c in conv}
        assert "ENCUT" in params and "k-points" in params
        for c in conv:
            assert isinstance(c["ladder"], list) and len(c["ladder"]) >= 2
            assert c["observable"] and c["tolerance"] > 0


class TestRegistryResolution:
    def test_hooks_resolve_when_vasp_active(self):
        from scilink.skills._shared._registry import get_tool_function
        setter = get_tool_function("set_convergence_param", active_skills=["vasp"])
        reader = get_tool_function("read_convergence_observable",
                                   active_skills=["vasp"])
        # setter is callable end-to-end
        out = setter(input_files={"INCAR": "PREC = Accurate\n"},
                     param="ENCUT", value=520)
        assert "ENCUT = 520" in out["INCAR"]
        # reader returns None for a missing run dir rather than raising
        assert reader(output_dir="/nonexistent", observable="energy_per_atom") is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
