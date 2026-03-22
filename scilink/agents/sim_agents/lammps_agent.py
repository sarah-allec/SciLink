# agents/sim_agents/lammps_agent.py
"""
Backward-compatible wrapper around MDSimulationAgent.

Existing code using LAMMPSSimulationAgent continues to work unchanged.
New code should use MDSimulationAgent directly.
"""

import warnings
from typing import Dict, Any, Optional
from .md_simulation_agent import MDSimulationAgent


class LAMMPSSimulationAgent(MDSimulationAgent):
    """
    LAMMPS-specific simulation agent.

    This is a thin wrapper around MDSimulationAgent with skill="lammps"
    pre-loaded and the old parameter names preserved.

    .. deprecated::
        Use ``MDSimulationAgent(skill="lammps")`` directly.
    """

    def __init__(self, working_dir: str, **kwargs):
        kwargs.setdefault("skill", "lammps")
        super().__init__(working_dir=working_dir, **kwargs)

    def generate_simulation(
        self,
        data_file: Optional[str] = None,
        structure_file: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Accept both old 'data_file' and new 'structure_file' param names."""
        resolved = structure_file or data_file
        if resolved is None:
            raise ValueError("Must provide structure_file (or data_file)")
        if data_file and not structure_file:
            warnings.warn(
                "data_file is deprecated, use structure_file",
                DeprecationWarning,
                stacklevel=2,
            )
        return super().generate_simulation(structure_file=resolved, **kwargs)

    def generate_staged_simulation(
        self,
        data_file: Optional[str] = None,
        structure_file: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        resolved = structure_file or data_file
        if resolved is None:
            raise ValueError("Must provide structure_file (or data_file)")
        if data_file and not structure_file:
            warnings.warn(
                "data_file is deprecated, use structure_file",
                DeprecationWarning,
                stacklevel=2,
            )
        return super().generate_staged_simulation(
            structure_file=resolved, **kwargs
        )
