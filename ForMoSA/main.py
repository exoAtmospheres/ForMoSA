'''
ForMoSA run script

Here we open the config file and extract all the needed information.
Easy to understand and simple access for the new users.

@authors: S. Petrus, P. Palma-Bifani and Matthieu Ravet
'''
# ----------------------------------------------------------------------------------------------------------------------
## IMPORTS
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import sys

# Import ForMoSA
from ForMoSA.utils import yesno
from ForMoSA.global_file import GlobFile
from ForMoSA.adapt.adapt_obs_mod import launch_adapt
from ForMoSA.nested_sampling.nested_sampling import launch_nested_sampling

if __name__ == '__main__':
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

    # create output directory if needed
    if not os.path.exists(global_params.result_path):
        os.makedirs(global_params.result_path, exist_ok=True)

    # ----------------------------------------------------------------------------------------------------------------------
    ## Run ForMoSA
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('-> Initializing ForMoSA')
    print()

    if len(sys.argv) == 1:
        y_n_par = yesno('Do you want to adapt the grid to your data? (y/n)')
    else:
        y_n_par = sys.argv[2]

    if y_n_par == 'y':
        launch_adapt(global_params, adapt_model=True)
    else:
        launch_adapt(global_params, adapt_model=False)

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