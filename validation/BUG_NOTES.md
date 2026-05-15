# Validation-panel bug log

Bugs surfaced by the lattice-constant panel runs on Deception. One file
per bug; short and dated.

---

## 2026-05-14 — MgO POSCAR has interleaved species → VASP rejects

### Symptom

All three modes (`a_forced`, `b_agent_select`, `c_bare_goal`) failed VASP
for `mgo_rocksalt` and only `mgo_rocksalt`. The other three systems
(`si_diamond`, `cu_fcc`, `c_diamond`) all completed normally in every
mode.

VASP's `vasp.out` for every MgO cell ended with the same message:

```
ERROR: number of potentials on File POTCAR incompatible with number
of species  INCAR :    8  POTCAR :    2

  ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----
```

### Diagnosis

The generated POSCAR had its species line and counts line interleaved:

```
Mg O  Mg O  Mg O  Mg O           ← line 6 (species symbols)
  1   1   1   1   1   1   1   1  ← line 7 (ion counts per species)
```

VASP reads this as 8 distinct species (each `Mg` and each `O` token a
separate species). The per-cell `submit.sbatch` we generate assembles
POTCAR from the *unique* element list (`for el in Mg O`), so POTCAR has
2 species — hence the 8-vs-2 mismatch.

Single-element systems (Si, Cu, C) don't expose this — line 6 is a
single token regardless of atom ordering, so it always parses as 1
species. The bug needs a multi-element cell to surface, which is why
the rest of the panel ran fine.

### Root cause

`validation/run_panel.py::_write_poscar()` calls
`ase.io.write(..., format="vasp", direct=True)` without `sort=True`.
The atoms come from `ase.build.bulk("MgO", "rocksalt", cubic=True)`,
which returns the conventional rock-salt cell with atoms ordered
`Mg, O, Mg, O, …` (the natural NaCl-stacking order). ASE's VASP writer
preserves that order verbatim, so the POSCAR's species line gets one
token per atom.

### Fix

One-line change to `_write_poscar` — pass `sort=True`:

```python
def _write_poscar(atoms, path: str) -> None:
    from ase.io import write
    write(path, atoms, format="vasp", direct=True, sort=True)
```

This groups atoms by element symbol before writing, so the species line
becomes `Mg O` and the counts line `4 4`.

### Scope of the fix

This patch covers the **validation harness**. Outside the harness, the
production path generates POSCARs through `StructureGenerator`, which
emits an LLM-authored ASE script that does its own `atoms.write(...)`.
That path is a separate concern — the LLM may or may not include
`sort=True` depending on the prompt and example code. If we start seeing
the same symptom from non-harness multi-element runs, the right fix is
to either (a) inject a sort step in the generated script template, or
(b) post-process the emitted POSCAR. Worth keeping an eye on but not
something to chase pre-emptively.

### Earned value

The panel earned its keep here: a single-element benchmark — even one
that swept many systems — would never have caught this. Multi-element
coverage is the load-bearing part of the panel.
