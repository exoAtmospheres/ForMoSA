'''
ForMoSA // run script
'''
# ----------------------------------------------------------------------------------------------------------------------
## IMPORTS
import sys
import os
import mpi4py
# external package setup
os.environ["OMP_NUM_THREADS"] = "1"  # to not have numpy parallelize on its own

# Import ForMoSA
from ForMoSA.global_file import GlobFile
from ForMoSA.nested_sampling.nested_sampling import launch_nested_sampling

# Launch MPI
# ----------------------------------------------------------------------------------------------------------------------
config_file_path = '/home/mravet/Documents/These/FORMOSA/OUTPUTS/COCONUT2b/ATMO2020++/2025-05-15_Flamingos2_chi2classic_long_globalcov/config_file_ref.ini'
global_params = GlobFile(config_file_path)
launch_nested_sampling(global_params)
