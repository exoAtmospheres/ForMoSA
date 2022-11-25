'''
ForMoSA v6 run script

Pre_layer to open the config file and extract all the needed information.
Easy to understand and simple access for the new uses. THIS IS THE GOAL

@author: P. Palma-Bifani
'''
# ----------------------------------------------------------------------------------------------------------------------
'''
TO-DOs:

- run Hip 654
- make different default config files for the different grids and different run_ForMoSA_default.py

- make ForMoSA a package (almost)
- Check ForMoSA Licence
- Check ForMoSA Pip install and read the docs
- with example of setup (jupyter) and webpage and all that...
- add timeing bars and stuff like that: F2 interp_grid, Run ForMoSA

Licallo & Library:
- licallo setup bash to run code
- create way and path to copy licalo <-> mac
- run all the SINFONI library 

Science Code Updates:
- Save bayesian inference for model comparison and selection..
- do better work with uncertainties? -> included in flux calibration but beyond too?
- make dynasty and ultranest work
- make flux calibration uncertainty work
- make cpd work

- create good plotting script for spectra, residuals, cornerplots, comparisons, etc...

THIS IS A SIGNIFICANT PRODUCT OF MY PHD!!!!!!!!!
'''
# ----------------------------------------------------------------------------------------------------------------------
## IMPORTS
import sys

# Import ForMoSA
base_path = '/Users/simonpetrus/PycharmProjects/ForMoSA_v.1.0/'     # Give the path to ForMoSA to be able to import it. No need when this will be a pip package
sys.path.insert(1, base_path)
from master_main_utilities import yesno
from master_main_utilities import GlobFile
from adapt.adapt_obs_mod import launch_adapt
from nested_sampling.nested_sampling import launch_nested_sampling

# ----------------------------------------------------------------------------------------------------------------------
## USER configuration path
print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('-> Configuration of environment')
if len(sys.argv) == 1:
    print('Where is your configuration file?')
    config_file_path = input()
else:
    config_file_path = sys.argv[1]
print()

# ----------------------------------------------------------------------------------------------------------------------
## CONFIG_FILE reading and defining global parameters
global_params = GlobFile(config_file_path)                          # To access any param.: global_params.parameter_name
# ----------------------------------------------------------------------------------------------------------------------
## Run ForMoSA
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('-> Initializing ForMoSA')
print()

if len(sys.argv) == 1:
    y_n_par = yesno('Do you want to adapt the grid too your data? (y/n)')
else:
    y_n_par = sys.argv[2]

if y_n_par == 'y':
    launch_adapt(global_params, justobs='no')
else:
    launch_adapt(global_params, justobs='yes')


print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('-> Nested sampling')
print()
# Run S5 for Nested Sampling
launch_nested_sampling(global_params)

print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('-> Voilà, on est prêt')
print()