"""Electronic Continuum Correction (ECC) charge scaling — engine-neutral.

A fixed-charge (non-polarizable) force field omits electronic polarization,
which in condensed ionic phases screens ion-ion and ion-solvent interactions and
governs the transport (viscosity, conductivity, diffusion) of concentrated or
multivalent electrolytes. ECC approximates that screening in a mean field by
scaling the charges of the IONS by a factor ``f = 1/sqrt(eps_el) ~ 0.75``
(``eps_el ~ 1.78`` for many solvents), leaving neutral species' internal charges
intact so a solvent keeps its full dipole. Because an originally-neutral box's
ionic charges already sum to zero, scaling every net-charged species by the SAME
factor preserves overall neutrality exactly — no renormalization needed.

This module is pure arithmetic on per-molecule charge lists, so it is unit-
testable without any force-field toolkit; a backend calls it on the charges it
assigned. Selecting a scaling factor and WHEN to apply it is a modelling
judgment that lives in the force_field skill guidance + the caller, not here.
"""

from __future__ import annotations

from typing import List, Sequence

# The canonical ECC / ECCR value, 1/sqrt(1.78); a sensible default when the
# caller does not supply one. Reported values range ~0.7-0.85 by system.
DEFAULT_ECC_FACTOR = 0.75


def is_ionic(charges: Sequence[float], tol: float = 1e-6) -> bool:
    """True when a molecule's partial charges sum to a non-zero net charge."""
    return abs(sum(charges)) > tol


def scale_ionic_charges(charges_per_molecule: List[Sequence[float]],
                        factor: float, *, tol: float = 1e-6) -> List[List[float]]:
    """Scale the partial charges of each net-charged (ionic) molecule by ``factor``.

    Neutral molecules (``|sum(charges)| <= tol``) are returned unchanged, so a
    solvent keeps its full dipole; net-charged species scale uniformly, which
    preserves the system's total charge (an originally-neutral box stays
    neutral). Returns a new list of lists; the inputs are not mutated.
    """
    if factor is None:
        raise ValueError("scale_ionic_charges: factor must be a number")
    out: List[List[float]] = []
    for q in charges_per_molecule:
        out.append([c * factor for c in q] if is_ionic(q, tol) else list(q))
    return out
