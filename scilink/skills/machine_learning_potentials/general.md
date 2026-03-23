## Overview

Machine learning interatomic potentials (MLIPs) learn the potential energy
surface from quantum mechanical reference data.  They achieve near-DFT
accuracy at a fraction of the cost for systems up to ~10,000 atoms.

Key distinctions from classical force fields:
- No fixed functional form — the model learns energy/force relationships
- Accuracy depends on training data coverage, not parameter fitting
- Units in LAMMPS: **metal** (eV, Å, ps) — NOT real (kcal/mol, Å, fs)
- Require LAMMPS compiled with ML-IAP package or backend-specific plugin

## Planning

When to recommend MLIPs over classical force fields:
- Reactive chemistry (bond breaking/formation)
- Novel materials without established FF parameters
- High-accuracy energy barriers (catalysis, diffusion)
- Metal/oxide interfaces
- Phase transitions
- Systems where classical FF accuracy is insufficient

When to prefer classical force fields:
- Large-scale sampling (>50k atoms) without GPU resources
- Well-parameterized biomolecular systems (proteins in water)
- Routine production runs where speed matters more than accuracy
- Systems with no available DFT reference data and no pretrained coverage

Pretrained-first strategy:
1. Always try a foundation model first (zero cost to deploy)
2. Run a short simulation (10–50 ps)
3. Evaluate uncertainty and observables
4. Fine-tune only if quality is insufficient
5. Train from scratch only as last resort

## Validation

Quality assessment signals (combine all three):

1. MLIP uncertainty:
   - Per-atom force variance or descriptor distance
   - >10% of frames flagged as extrapolation → model is unreliable
   - High max forces (>10 eV/Å) suggest out-of-distribution geometry

2. Physical observables:
   - Density within 5% of experimental/DFT value → acceptable
   - Energy drift <1% over simulation → stable potential
   - Temperature fluctuations consistent with ensemble

3. Acceptance thresholds:
   - Energy MAE < 5 meV/atom (bulk), < 10 meV/atom (surfaces)
   - Force MAE < 100 meV/Å
   - If force MAE > 200 meV/Å: insufficient training data

## Implementation

Active learning loop:
1. Run MD with pretrained model
2. Identify high-uncertainty configurations
3. Run DFT on those configurations (VASP, CP2K, etc.)
4. Fine-tune model on augmented dataset
5. Repeat until uncertainty is acceptable

DFT data requirements for fine-tuning:
- Minimum: ~50 diverse structures with energy + forces
- Recommended: ~200 structures spanning the relevant PES region
- Include: equilibrium, compressed, expanded, high-temperature snapshots
