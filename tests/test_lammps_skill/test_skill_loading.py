# tests/test_lammps_skill/test_skill_loading.py
"""
Tests for the LAMMPS skill file: correct loading, section parsing,
and content requirements.
"""

import pytest
from scilink.skills.loader import load_skill, list_skills, list_all_skills


class TestSkillDiscovery:

    def test_lammps_in_md_simulation_domain(self):
        skills = list_skills(domain="molecular_dynamics")
        assert "lammps" in skills

    def test_md_simulation_in_all_skills(self):
        all_skills = list_all_skills()
        assert "molecular_dynamics" in all_skills
        assert "lammps" in all_skills["molecular_dynamics"]


class TestSkillLoading:

    @pytest.fixture
    def skill(self):
        return load_skill("lammps", domain="molecular_dynamics")

    def test_loads_without_error(self, skill):
        assert skill is not None

    def test_has_name(self, skill):
        assert skill["name"] == "lammps"

    @pytest.mark.parametrize("section", [
        "overview", "planning", "analysis",
        "interpretation", "validation", "implementation",
    ])
    def test_has_required_section(self, skill, section):
        assert section in skill
        assert len(skill[section]) > 0, f"Section '{section}' is empty"


class TestSkillContent:
    """Verify skill contains the critical decision-making information."""

    @pytest.fixture
    def skill(self):
        return load_skill("lammps", domain="molecular_dynamics")

    # ── Overview ──

    def test_overview_mentions_data_file(self, skill):
        assert "data file" in skill["overview"].lower()

    # ── Planning ──

    def test_planning_has_unit_systems(self, skill):
        p = skill["planning"]
        assert "metal" in p
        assert "real" in p

    def test_planning_has_atom_styles(self, skill):
        p = skill["planning"]
        assert "atomic" in p
        assert "full" in p
        assert "charge" in p

    def test_planning_has_pair_styles(self, skill):
        p = skill["planning"]
        for style in ["eam", "tersoff", "buck", "lj", "reaxff"]:
            assert style in p.lower(), f"Missing pair_style family: {style}"

    def test_planning_has_damping_by_units(self, skill):
        p = skill["planning"]
        assert "Tdamp" in p or "tdamp" in p.lower()

    def test_planning_has_ensembles(self, skill):
        p = skill["planning"]
        assert "NPT" in p
        assert "NVT" in p

    # ── Validation ──

    def test_validation_has_forbidden_patterns(self, skill):
        v = skill["validation"]
        assert "forbidden" in v.lower() or "Forbidden" in v
        for pattern in ["kspace", "bond_style", "qeq"]:
            assert pattern in v.lower(), f"Missing forbidden pattern: {pattern}"

    def test_validation_has_command_ordering(self, skill):
        v = skill["validation"]
        assert "order" in v.lower()

    def test_validation_has_parameter_ranges(self, skill):
        v = skill["validation"]
        assert "metal" in v
        assert "real" in v

    # ── Implementation ──

    def test_implementation_has_metal_template(self, skill):
        impl = skill["implementation"]
        assert "eam" in impl.lower()
        assert "units metal" in impl

    def test_implementation_has_biomolecular_template(self, skill):
        impl = skill["implementation"]
        assert "atom_style full" in impl
        assert "units real" in impl

    def test_implementation_has_reaxff_template(self, skill):
        impl = skill["implementation"]
        assert "reaxff" in impl.lower()
        assert "qeq" in impl.lower()

    # ── Analysis ──

    def test_analysis_has_detection_logic(self, skill):
        a = skill["analysis"]
        assert "atomic" in a
        assert "bond" in a.lower()

    # ── Interpretation ──

    def test_interpretation_has_common_errors(self, skill):
        interp = skill["interpretation"]
        assert "Lost atoms" in interp or "lost atoms" in interp.lower()


class TestCustomSkillPath:
    """Test loading skill from an arbitrary file path."""

    def test_load_from_path(self, tmp_path):
        custom = tmp_path / "my_engine.md"
        custom.write_text(
            "## Overview\nCustom engine.\n\n"
            "## Planning\nUse default.\n\n"
            "## Analysis\nParse the file.\n\n"
            "## Interpretation\nCheck output.\n\n"
            "## Validation\nVerify syntax.\n\n"
            "## Implementation\nTemplate here.\n"
        )
        skill = load_skill(str(custom), domain="molecular_dynamics")
        assert skill["name"] == "my_engine"
        assert "Custom engine" in skill["overview"]

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_skill("/no/such/file.md", domain="molecular_dynamics")

    def test_nonexistent_name_raises(self):
        with pytest.raises(FileNotFoundError):
            load_skill("nonexistent_engine", domain="molecular_dynamics")
