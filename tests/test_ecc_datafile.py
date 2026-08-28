"""ECC scaling of a LAMMPS data file: scale net-charged molecules, keep neutrality.

Applied to the FINAL data file (OpenFF rejects pre-scaled charges), grouped by
molecule-ID so an originally-neutral box stays neutral.
"""

from scilink.agents.sim_agents._engine_inputs import _scale_datafile_lines


def _data(atoms_block, style="full", with_bonds=True):
    lines = ["LAMMPS data\n", "\n", "8 atoms\n", "\n", "Masses\n", "\n",
             "1 1.008\n", "\n", f"Atoms # {style}\n", "\n"]
    lines += [l if l.endswith("\n") else l + "\n" for l in atoms_block]
    if with_bonds:
        lines += ["\n", "Bonds\n", "\n", "1 1 1 2\n"]
    return lines


def test_scales_ions_keeps_neutral_and_neutrality():
    # mol 1 = water (neutral, 3 atoms), mol 2 = Zn2+ (+2), mol 3,4 = OTf- (-1 each)
    atoms = [
        "1 1 1  0.4  0 0 0",
        "2 1 2 -0.8  1 0 0",
        "3 1 1  0.4  0 1 0",
        "4 2 3  2.0  5 5 5",
        "5 3 4 -1.0  8 8 8",
        "6 4 4 -1.0  2 2 2",
    ]
    lines = _data(atoms)
    out, changed = _scale_datafile_lines(lines, 0.75)
    # pull the charge column (index 3) back out per atom id
    charges = {}
    started = False
    for ln in out:
        if ln.startswith("Atoms"):
            started = True; continue
        if started and ln.strip() and ln.split()[0].isdigit() and len(ln.split()) >= 4:
            p = ln.split(); charges[int(p[0])] = float(p[3])
        elif started and ln.strip().startswith("Bonds"):
            break
    # water (mol 1) untouched
    assert charges[1] == 0.4 and charges[2] == -0.8 and charges[3] == 0.4
    # ions scaled by 0.75
    assert abs(charges[4] - 1.5) < 1e-9
    assert abs(charges[5] - (-0.75)) < 1e-9 and abs(charges[6] - (-0.75)) < 1e-9
    assert changed == 3
    # whole box still neutral
    assert abs(sum(charges.values())) < 1e-9


def test_no_molecule_column_is_noop():
    # atom_style 'charge' has no molecule column -> cannot group -> unchanged
    atoms = ["1 2  2.0  5 5 5", "2 4 -1.0 8 8 8"]
    lines = _data(atoms, style="charge", with_bonds=False)
    out, changed = _scale_datafile_lines(lines, 0.75)
    assert changed == 0 and out == lines


def test_no_atoms_section_is_noop():
    lines = ["LAMMPS data\n", "\n", "Masses\n", "\n", "1 1.008\n"]
    out, changed = _scale_datafile_lines(lines, 0.5)
    assert changed == 0 and out == lines


def test_preserves_trailing_image_flags():
    atoms = ["1 2 3 2.0 5 5 5 0 1 -1"]
    lines = _data(atoms, with_bonds=False)
    out, _ = _scale_datafile_lines(lines, 0.5)
    row = next(l for l in out if l.startswith("1 2 3"))
    assert row.split()[4:] == ["5", "5", "5", "0", "1", "-1"]   # image flags kept
    assert abs(float(row.split()[3]) - 1.0) < 1e-9
