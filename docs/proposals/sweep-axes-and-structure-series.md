# Proposal: sweep axes — structure-varying campaigns + parameterized box building

**Status:** PROPOSAL — for discussion (Sarah ↔ Maxim). Not canonical. Nothing
here is in CLAUDE.md yet; once aligned, the direction-level slice lands in
CLAUDE.md and the detailed design becomes a `docs/` design note.

**Motivation:** A composition series — the same protocol run across
0/20/30/40/50 % cosolvent — is not expressible today. The fan-out machinery
(`expand_parameter_sweep`) varies a **scalar in the deck** while members
**share one structure**; a composition series is the mirror image (structure
varies, protocol shared). This is not a niche request: concentration series,
finite-size scaling, defect-density series, and multi-candidate screening all
have this shape. The gap also hides a silent wrong answer (Part 3).

---

## Part 1 — Vocabulary: a sweep axis targets a stage

The generalization that makes this tractable: **every sweep axis affects
exactly one pipeline stage.**

| Axis target | Varies | Example | Status |
|---|---|---|---|
| `deck` | a scalar in the engine input | temperature, pressure, strain rate, umbrella window | **exists** |
| `structure` | the built system | composition, concentration, box size, defect density | **this proposal** |
| `force_field` | the parameterization | water model, charge method, FF version | future (seam only) |

The `force_field` axis is not built here, but it is a real study shape
("TIP3P vs SPC/E vs OPC") and is the reason the design names the *target*
rather than special-casing composition. If the seam accommodates all three and
we implement one, the abstraction is general; if it only accommodates
composition, it is use-case scaffolding.

**Why the axis must be explicit.** Today the target is implied (always `deck`).
Making it a declared field is what turns Part 3's silent wrong answer into a
hard error.

---

## Part 2 — What is already true (this corrects the record)

Three findings from reading the code, two of which contradict what we believed:

**1. Charges are already consistent across a series — for free.**
`build_interchange.py:131-143` derives partial charges per *unique component*
from **SMILES only**: `Molecule.from_smiles(smiles)` →
`assign_partial_charges(nagl_model)`. No coordinates, counts, box, or
conformers enter. NAGL is an inference-only graph net (a deterministic AM1-BCC
surrogate), so identical species → identical charges, bit-for-bit, every run.
`charged_unique` (`:127-150`) depends only on the SMILES set; counts merely
stamp copies. **The recorded belief that per-run NAGL charge derivation drives
run-to-run variation is wrong.** Charge reuse across a series would be a pure
optimization, not a correctness fix.

**2. The real variance is the LLM authoring a fresh packmol script each run.**
Molecule counts and packing exist *only* inside LLM-generated code. This is the
same fragility `nacl_aq` caught red-handed (a hand-rolled numpy packing loop
whose overlap check compared only heavy-atom centres, letting hydrogens
overlap; the deterministic min-distance check flagged it, the LLM validator
passed it). PR #292 made codegen environment-aware and stated a packmol
preference, but the packing is still LLM-authored and regenerated per run.

**3. The fan-out data model already promises per-member structures.**
`Stage`'s own docstring (`refinement.py:104-118`) states that what members
differ by "lives in each member's `input_files`, authored by the generator,
**never here**", and describes fan-out members as differing "only in their
inputs". `Phase.input_files` is per-member (`:81-101`); `_collect_stages`
(`simulation_pipeline.py:511-590`, fan-out branch `:575-584`) and
`run_campaign` never inspect, share, or dedupe the structure file.

**Only the producer violates the contract.** `_assemble_fanout_stage`
(`md_simulation_agent.py:24-54`) does `input_files = dict(shared_files)` and
then overrides only the deck entry. So this proposal does not invent a
concept — it makes the producer honor a contract that is already written down
and already supported downstream.

---

## Part 3 — What breaks today

**The sweep cannot reach the structure builder — for an ordering reason.**
In `_run_workflow_once` the order is: Step 1 structure generation → one
`structure_path` (`simulation_pipeline.py:296`); Step 1.5 force field → one
`ParameterizedSystem` (`:306-321`); Step 2 `_generate_inputs` → MD agent →
sweep planned and expanded (`:345`). The box is packed and typed long before
the sweep exists.

**Silent wrong answer.** `expand_parameter_sweep`
(`lammps.py:1194-1224`) is `base_script.replace(placeholder, str(value))` —
nothing more. If a planner emitted `variable_parameter: "cosolvent_fraction"`,
it would dutifully splice `40` into `__SWEEP__` **in the deck**, producing N
identical-composition runs with a nonsense number in a LAMMPS command. No
error, no warning. The plan schema scopes the axis to deck quantities in prose
only.

**No parameterized box builder.** `condensed` is markdown-only — verified: the
bundle contains just `condensed.md` and `__init__.py`, and **no
`structure_generation` skill ships a `TOOL_SPEC` at all**, while
`force_field/openff` and `molecular_dynamics/lammps` both do. There is no way
to ask for "these five boxes, everything identical but the EIS fraction".

**Scalar spine.** `structure_path` threads as a scalar
(`simulation_pipeline.py:296 → 306 → 317 → 345`).

**Gates validate member 0 only.** `run_campaign` approves
`runnable[0].phases[0].input_files` and dry-runs that same first phase
(`refinement.py:847-859`), with the comment "Validates the first phase's
setup, which staged phases share." Correct for a temperature sweep; wrong for a
composition series, where a bad pack in member 3 sails straight through. It is
fail-open, so this is a coverage gap rather than a crash.

**Filename collision.** `build_interchange` writes
`working_dir/system_interchange.json` at a fixed name (`:177`); N members in
one working dir clobber each other.

---

## Part 4 — The design

### 4a. `build_box` — a parameterized condensed builder

A `TOOL_SPEC`-bearing sibling in `scilink/skills/structure_generation/condensed/`:

```
build_box(components, density=None, box=None, seed=..., tolerance=...)
  components: [{name, smiles, count}]   # same vocabulary as the FF side
  → writes coordinates + components.json ; returns paths + realized box/density
```

This is deliberately the **mirror image of `build_interchange(components,
coordinates_file, force_field, …)`** — same `components` vocabulary, one stage
earlier. That symmetry is the strongest evidence the design follows the grain
of the existing contract rather than cutting across it.

It also makes `components.json` a **tool output** rather than a prose
instruction the LLM is asked to remember (today: a "Components manifest
(required)" bullet in `condensed.md`, consumed at
`simulation_pipeline.py:149-170` where **order is load-bearing** and enforced
only by a downstream atom-count check at `build_interchange.py:158-163`).

Per the `TOOL_SPEC` promotion rule in CLAUDE.md — promote when the code must run
deterministically, is numerically sensitive, or is a reusable stage — box
packing hits all three, and `nacl_aq` is direct evidence the prose rung is
failing here. **Expose the knobs** (`density`, `seed`, `tolerance`) per the
no-locked-knobs rule, with adaptive defaults so a no-arg call still works.

**Scope honesty:** `build_box` is **condensed-class-specific**. A crystal alloy
sweep (supercell/occupancy) or a biomolecular series (fetch + solvate) would
each need their own parameterized builder. That is the skill architecture
working as designed — but this proposal delivers *a builder for one class plus a
general per-member-structure seam*, not one universal builder. `build_box`
is independently justified regardless of series: it makes every condensed
generation reproducible.

### 4b. The series contract

A structure axis cannot reuse `variable_values` (a list of scalars) — each
member needs a full component list. Proposed member spec:

```
series = [ {name, components: [{name, smiles, count}], density?, box?}, … ]
```

**One field, two authors.** `series` is a normal input that either a human or
the LLM may fill — not a fork. This mirrors `run_simulation`'s existing
behavior, where `scale=None` / `software=None` are routed from the description
but a supplied value is honored as-is (and CLAUDE.md's authoritative `skill`
vs non-binding `skill_hint` pair):

- `series=None` → derived from the description (the LLM names the points).
- `series=[...]` → honored exactly (the experimentalist names the points).

**But the counts are always computed deterministically**, whoever names the
points. "40 vol% cosolvent at 1.0 g/cc → N molecules" is arithmetic, and
LLM-authored stoichiometry is precisely where `nacl_aq` burned us. So the LLM
may say *which* compositions; a helper decides *how many molecules*. These are
separate concerns and only the second one is load-bearing for reproducibility.

### 4c. Spine: scalar → list

Map Steps 1 and 1.5 over members; Step 2 assembles one fan-out:

- Step 1: `build_box` per member (identical args but `components`) → N coords.
- Step 1.5: `build_interchange` per member with **identical** `force_field` /
  `extra_force_fields` / `nagl_model`. Given Part 2's determinism, this alone
  guarantees the shared-FF property. Needs a distinct `working_dir` per member
  to fix the `:177` collision.
- Step 2: `_assemble_fanout_stage` accepts a per-member structure and overrides
  that entry in `input_files` alongside the deck.

`_collect_stages` and `run_campaign` need **zero edits**. The existing
single-structure path is the degenerate case (N=1), which is the backcompat
story.

Useful existing seam: `_run_workflow_once` already accepts
`structure_file: Optional[str]` and skips generation when supplied
(`simulation_pipeline.py:194, 269-274`), with `caller_supplied_structure`
tracked at `:438`.

### 4d. Gate every distinct structure

Extend the pre-run and dry-run gates to cover each member with a distinct
structure (dedupe by content hash so a temperature sweep still gates once).
Without this, a composition series' per-member packs are ungated.

---

## Part 5 — Scope

**Build now:** `build_box` (4a) · per-member structures through the producer
(4c) · explicit axis target + hard error when a structure axis routes to deck
substitution (Part 3's silent wrong answer) · per-structure gating (4d).

**Seam only, not built:** the `force_field` axis.

**Explicitly out:** parameterized builders for crystal / molecular /
biomolecular classes.

---

## Part 6 — Open questions

1. ~~**Who authors the series?**~~ **SETTLED (2026-07-17):** both, via one
   field — `series=None` derives from the description, `series=[...]` is
   honored (4b). This was originally posed as caller-declared *vs* LLM-inferred;
   that was a false fork, fusing two separable questions. Naming the composition
   points is safe for either author; computing molecule counts is always the
   deterministic helper's job. Only the second constrains reproducibility.
2. **Where do counts get computed?** A `composition_from_spec` helper, or a
   `build_box` that accepts concentrations directly? (Not the LLM — see above.)
   Open only as to *which* surface carries the arithmetic.
3. **Does `build_box` supersede LLM codegen for condensed, or coexist?** If the
   tool covers the common cases, codegen becomes the fallback for exotic boxes.
   Retiring codegen entirely is more reproducible but less open-ended.
4. **Should `ParameterizedSystem` split?** It currently fuses FF choice with one
   box (`_parameterized_system.py:64-125` binds `components`,
   `coordinates_file`, `box`, `n_atoms`, `interchange_path` in one object), so
   it is a per-structure artifact, not a reusable FF handle. Part 2 means we do
   not *need* the split (charges are deterministic), but a `force_field` axis
   later might want it.
5. **Series-level acceptance.** `min_success` quorum exists per fan-out, but a
   composition *trend* with a missing point may be scientifically useless even
   at 4/5. Should a series declare all-required?
