"""run_composition_series: the spine that maps one protocol over N structures.

Stubs the heavy stages (build_box -> Packmol, FF parameterization -> OpenFF,
deck generation -> LLM, executor -> engine) and asserts the ORCHESTRATION: each
member is built in its own directory, one deck is generated and fanned out over
every member's own structure, and the fan-out reaches the refinement campaign.
"""

import os
from unittest import mock

import pytest

import scilink.agents.sim_agents.simulation_pipeline as sp


def _members():
    return [
        {"name": "eis 20%", "components": [
            {"name": "water", "smiles": "O", "count": 600},
            {"name": "eis", "smiles": "CCS(=O)(=O)C(C)C", "count": 20}]},
        {"name": "eis 40%", "components": [
            {"name": "water", "smiles": "O", "count": 400},
            {"name": "eis", "smiles": "CCS(=O)(=O)C(C)C", "count": 40}]},
        {"name": "eis 50%", "components": [
            {"name": "water", "smiles": "O", "count": 300},
            {"name": "eis", "smiles": "CCS(=O)(=O)C(C)C", "count": 50}]},
    ]


def _patch_build_and_ff(tmp_path):
    """build_box writes a per-member 'data' file; _parameterize_structure types it.

    Each member's typed contents embed its own directory name, so the fan-out
    assertion can prove members carry their OWN structure, not a shared one.
    """
    def fake_build_box(components, density=None, box=None, working_dir=".", **kw):
        os.makedirs(working_dir, exist_ok=True)
        coord = os.path.join(working_dir, "packed.xyz")
        with open(coord, "w") as f:
            f.write(f"coords for {os.path.basename(working_dir)}\n")
        return {"structure_file": coord, "manifest_file": "m.json",
                "box": 30.0, "density": density or 1.0, "n_atoms": 3,
                "n_molecules": sum(c["count"] for c in components)}

    def fake_parameterize(structure_path, software, output_dir, **kw):
        typed = os.path.join(output_dir, "system.data")
        with open(typed, "w") as f:
            f.write(f"TYPED DATA for {os.path.basename(output_dir)}\n")
        return {"status": "success", "structure_file": typed,
                "force_field_files": {"ff.params": "SHARED FF"},
                "summary": {"status": "success", "backend": "openff",
                            "n_atoms": 3, "total_charge": 0.0}}

    return (
        mock.patch("scilink.skills.structure_generation.condensed.build_box.build_box",
                   fake_build_box),
        mock.patch.object(sp, "_parameterize_structure", fake_parameterize),
    )


def _fake_generate_inputs(**kw):
    """One protocol deck that reads_data the shared structure filename."""
    return {"status": "success", "entry_file": "run.lammps",
            "input_files": {
                "run.lammps": "read_data system.data\nrun 1000\n",
                "system.data": "REP DATA",         # representative's own box
                "ff.params": "SHARED FF"}}


def test_series_builds_each_member_and_fans_one_deck(tmp_path):
    p_bb, p_ff = _patch_build_and_ff(tmp_path)
    with p_bb, p_ff, mock.patch.object(sp, "_generate_inputs", _fake_generate_inputs):
        out = sp.run_composition_series(
            "MD of aqueous Zn(OTf)2 with EIS cosolvent", _members(),
            software="lammps", density=1.05, validate=False,
            output_dir=str(tmp_path / "series"))

    assert out["final_status"] == "generated"
    assert len(out["members"]) == 3
    assert all(m["status"] == "success" for m in out["members"])

    stages = out["stages"]
    assert len(stages) == 1 and stages[0]["parallel"] is True
    fan = stages[0]["members"]
    assert len(fan) == 3

    # Every member runs the SAME deck...
    decks = {m["input_files"]["run.lammps"] for m in fan}
    assert decks == {"read_data system.data\nrun 1000\n"}
    # ...but reads its OWN typed box under the deck's read_data name.
    datas = [m["input_files"]["system.data"] for m in fan]
    assert datas[0] != datas[1] != datas[2]
    assert all("TYPED DATA for" in d for d in datas)
    # ...and shares the one force field.
    assert all(m["input_files"]["ff.params"] == "SHARED FF" for m in fan)


def test_series_writes_members_to_isolated_directories(tmp_path):
    p_bb, p_ff = _patch_build_and_ff(tmp_path)
    with p_bb, p_ff, mock.patch.object(sp, "_generate_inputs", _fake_generate_inputs):
        sp.run_composition_series(
            "protocol", _members(), software="lammps", validate=False,
            output_dir=str(tmp_path / "series"))

    base = tmp_path / "series"
    # slugified member names, each its own dir with its own typed structure
    for slug in ("eis_20", "eis_40", "eis_50"):
        assert (base / slug / "system.data").is_file()


def test_series_runs_campaign_when_executor_given(tmp_path):
    p_bb, p_ff = _patch_build_and_ff(tmp_path)
    captured = {}

    def fake_run_campaign(stages, executor, critic, policy, ctx, **kw):
        captured["n_members"] = len(stages[0].phases)
        captured["parallel"] = stages[0].parallel
        return {"status": "success"}

    with p_bb, p_ff, \
         mock.patch.object(sp, "_generate_inputs", _fake_generate_inputs), \
         mock.patch("scilink.agents.sim_agents.refinement.run_campaign",
                    fake_run_campaign), \
         mock.patch("scilink.agents.sim_agents.critics.RunCritic"):
        out = sp.run_composition_series(
            "protocol", _members(), software="lammps", validate=False,
            output_dir=str(tmp_path / "series"),
            executor=object(), run_command="lmp -in {script}")

    assert out["final_status"] == "success"
    assert out["refinement"]["status"] == "success"
    # the fan-out reached the campaign: one parallel stage, one phase per member
    assert captured["n_members"] == 3
    assert captured["parallel"] is True


def test_series_validates_the_shared_deck_once(tmp_path):
    """The protocol deck is validated a single time (it is shared), and the
    verdict is recorded — the check that would flag a bad deck pre-run."""
    p_bb, p_ff = _patch_build_and_ff(tmp_path)
    seen = {}

    class FakeInputValidator:
        n = 0

        def __init__(self, **kw):
            pass

        def validate(self, input_files, system_description, skill=None, domain=None):
            FakeInputValidator.n += 1
            seen["files"] = set(input_files)
            return {"status": "success", "verdict": "good"}

    with p_bb, p_ff, \
         mock.patch.object(sp, "_generate_inputs", _fake_generate_inputs), \
         mock.patch("scilink.agents.sim_agents.critics.InputValidator",
                    FakeInputValidator):
        out = sp.run_composition_series(
            "protocol", _members(), software="lammps",
            output_dir=str(tmp_path / "series"))

    assert FakeInputValidator.n == 1, "deck should be validated exactly once"
    assert "run.lammps" in seen["files"]
    assert out["input_validation"]["verdict"] == "good"
    assert "deck_validation" in out["steps_completed"]


def test_series_skips_validation_when_disabled(tmp_path):
    p_bb, p_ff = _patch_build_and_ff(tmp_path)

    class BoomValidator:
        def __init__(self, **kw):
            raise AssertionError("validator must not be constructed when validate=False")

    with p_bb, p_ff, \
         mock.patch.object(sp, "_generate_inputs", _fake_generate_inputs), \
         mock.patch("scilink.agents.sim_agents.critics.InputValidator", BoomValidator):
        out = sp.run_composition_series(
            "protocol", _members(), software="lammps", validate=False,
            output_dir=str(tmp_path / "series"))

    assert "deck_validation" not in out["steps_completed"]
    assert "input_validation" not in out


def test_series_needs_at_least_two_members(tmp_path):
    out = sp.run_composition_series(
        "protocol", [{"name": "only", "components": [{"name": "w", "smiles": "O", "count": 1}]}],
        output_dir=str(tmp_path / "s"))
    assert out["final_status"] == "failed_series"
    assert "at least two" in out["error"]


def test_series_fails_loudly_when_a_member_build_fails(tmp_path):
    def bad_build(*a, **k):
        raise RuntimeError("packmol exploded")

    with mock.patch("scilink.skills.structure_generation.condensed.build_box.build_box",
                    bad_build):
        out = sp.run_composition_series(
            "protocol", _members(), software="lammps",
            output_dir=str(tmp_path / "s"))

    assert out["final_status"] == "failed_member_build"
    assert "packmol exploded" in out["error"]


def test_bad_deck_from_index_is_rejected(tmp_path):
    out = sp.run_composition_series(
        "protocol", _members(), software="lammps",
        deck_from=9, output_dir=str(tmp_path / "s"))
    assert out["final_status"] == "failed_series"
    assert "deck_from" in out["error"]
