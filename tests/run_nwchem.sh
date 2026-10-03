#!/bin/bash --login
# NWChem launch wrapper for the convergence live test, mirroring the benchmark's
# run_vasp.sh pattern: set up NWChem's own module environment here (a login shell
# so `module` is defined), decoupled from the conda env the Python driver runs
# in. The executor invokes this in each ladder rung's directory, where it has
# already written job.nw. Cluster-specific scaffolding — not for the PR.
module purge
module load gcc openmpi/4.1.8 nwchem
export NWCHEM_NWPW_LIBRARY=/share/apps/nwchem/7.2.3/share/libraryps/
export NWCHEM_BASIS_LIBRARY=/share/apps/nwchem/7.2.3/share/libraries/
exec mpirun --mca btl self,vader nwchem job.nw
