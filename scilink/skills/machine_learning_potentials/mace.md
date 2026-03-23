## Overview

MACE (Multi-ACE) is an equivariant message-passing neural network potential.
It is the recommended default backend because it provides production-ready
foundation models.

Foundation models:
- mace-mp-0: Trained on MPtrj (~150k inorganic materials). Covers most of
  the periodic table. Use for metals, oxides, ceramics, semiconductors.
- mace-mp-0b: Medium-accuracy variant, 3x faster inference.
- mace-off23: Trained on SPICE dataset. Covers organic molecules, peptides,
  drug-like compounds. Use for C/H/N/O/S/P/halogen systems.

## Planning

Model selection:
- Elements are inorganic (metals, oxides) → mace-mp-0
- Elements are organic (C, H, N, O, S, P, halogens) → mace-off23
- Mixed system → mace-mp-0 (broader element coverage)
- Need speed over accuracy → mace-mp-0b

Fine-tuning hyperparameters:
- r_max: 5.0 Å (default); increase to 6.0 for layered materials
- forces_weight: 100 (default); reduce to 10 if DFT forces are noisy
- batch_size: 4 (16 GB GPU), 8-16 (40+ GB GPU)
- learning_rate: 0.001 for fine-tuning (10x lower than from-scratch)
- max_num_epochs: 100 for fine-tuning (vs 200+ from scratch)

## Validation

MACE-specific thresholds:
- Energy MAE < 2 meV/atom: excellent
- Energy MAE 2-5 meV/atom: good (suitable for most applications)
- Energy MAE 5-10 meV/atom: acceptable (may affect barrier heights)
- Energy MAE > 10 meV/atom: poor (fine-tune or retrain)

- Force MAE < 50 meV/Å: excellent
- Force MAE 50-100 meV/Å: good
- Force MAE 100-200 meV/Å: marginal
- Force MAE > 200 meV/Å: unacceptable

## Implementation

LAMMPS configuration:
    pair_style  mace no_domain_decomposition
    pair_coeff  * * /path/to/model.model El1 El2 El3

Requirements:
- LAMMPS with ML-IAP package compiled against libtorch
- OR: lammps-mace plugin (pip install lammps-mace)
- Check: lmp -help | grep mace

Known limitations:
- no_domain_decomposition restricts to single-node MPI
- Inference speed: ~1 ms/atom/step on CPU, ~0.01 ms/atom/step on GPU
- Maximum practical system size: ~10k atoms (CPU), ~50k atoms (GPU)
