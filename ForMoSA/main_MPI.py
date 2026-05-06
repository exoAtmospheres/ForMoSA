'''
ForMoSA // run script
'''
# ----------------------------------------------------------------------------------------------------------------------
## IMPORTS
import shutil
import sys
import gc
import os
import glob
import mpi4py
from mpi4py import MPI
import numpy as np
from astropy.io import fits
# external package setup
os.environ["OMP_NUM_THREADS"] = "1"  # to not have numpy parallelize on its own

# Import ForMoSA
from ForMoSA.global_file import GlobFile
from ForMoSA.nested_sampling.nested_sampling import launch_nested_sampling

# Utils
def leave_block_out(wavs, dlam=0.1):
    """
    Given a list of wavelength arrays (wavs) and a block size (dlam),
    this function generates configurations to leave out a block of data
    of size dlam from each dataset in a sliding window fashion.
    Each configuration specifies which parts of each dataset to include
    and which parts to exclude based on the current block position.

    Args:
        wavs     (list of np.ndarray): List of wavelength arrays for each dataset.
        dlam                  (float): Size of the block to leave out.
    Returns:
        wave_limits  (list of tuples): Each tuple contains configurations for each dataset,
                        specifying which parts to include and exclude.
    """
    # Extract global min and max
    global_min = min([w.min() for w in wavs])
    global_max = max([w.max() for w in wavs])

    wave_limits = []
    start = global_min
    while start <= global_max:
        stop = start + dlam
        config = []
        
        # Iterate on each wavelength array
        for w in wavs:
            wmin, wmax = w.min(), w.max()

            # Case 1: block is outside the wavelength range
            if stop <= wmin or start >= wmax:
                config.append(f'{wmin}, {wmax}')

            # Case 2: block is fully inside the wavelength range
            elif start > wmin and stop < wmax:
                config.append(f'{wmin}, {start} / {stop}, {wmax}')

            # Case 3: block overlaps with the left edge of the wavelength range
            elif start <= wmin and stop < wmax:
                config.append(f'{stop}, {wmax}')

            # Case 4: block overlaps with the right edge of the wavelength range
            elif start > wmin and stop >= wmax:
                config.append(f'{wmin}, {start}')

            # Case 5: block fully covers the wavelength range
            elif start <= wmin and stop >= wmax:
                config.append('-2, -1')  # Irrealistic values to ignore this dataset
        
        # Append to the list only if at least one dataset is not fully excluded or not all datasets are excluded
        if not all(c == '-2, -1' for c in config) and not all(c == f'{w.min()}, {w.max()}' for c, w in zip(config, wavs)):
            wave_limits.append(config)
        start += dlam

    return wave_limits


def leave_one_out(wav, dlam=0.005):
    """
    Given an array (wav), this function generates configurations
    to leave out one dataset at a time. Each configuration specifies which datasets
    to include and which one to exclude.

    Args:
        wav     (np.ndarray): Wavelength arrays for each dataset.
        dlam          (float): Size of the block to leave out around each dataset.
    Returns:
        wav_fit  (list of str): Each string contains configurations for each dataset,
                        specifying which parts to include and exclude.
    Author: Paulina Palma-Bifani
    """
    # Prepare wav fit lists
    wav_fit = []
    for i in range(len(wav)):
        elements_before = np.array(wav[:i])
        elements_after = np.array(wav[i+1:])

        range_1 = ''
        range_2 = ''

        if len(elements_before)>0:
            low_1 = elements_before[0] - dlam
            high_1 = elements_before[-1] + dlam
            #print(str(low_1)+', '+str(high_1))
            range_1 = str(low_1)+', '+str(high_1) + ' / '

        if len(elements_after)>0:

            low_2 = elements_after[0] - dlam
            high_2 = elements_after[-1] + dlam
            #print(str(low_2)+', '+str(high_2))
            range_2 = str(low_2)+', '+str(high_2)

        if len(elements_after)==0:
            range_1=range_1[:-3]

        wav_fit.append(range_1 + range_2)
    return wav_fit

# Launch MPI
# ----------------------------------------------------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

config_file_path = '/home/mravet/Documents/These/FORMOSA/OUTPUTS/Delorme1ABb/Sonora/2026-04-30_all_noUVES/config_file_ref.ini'
if rank == 0:
    global_params = GlobFile(config_file_path)
else:
    global_params = None

# Broadcast to all ranks
global_params = comm.bcast(global_params, root=0)
launch_nested_sampling(global_params)





# #  - - - - - - - 
# #  Multi inversions

# config_paths = [
# '/home/mravet/Documents/These/FORMOSA/OUTPUTS/SR12C/ATMO3000/2026-04-24_MIRI+photo_BB/config_file_ref.ini',
# '/home/mravet/Documents/These/FORMOSA/OUTPUTS/SR12C/ATMO3000/2026-04-24_MIRI+photo_BB_Av/config_file_ref.ini',
# '/home/mravet/Documents/These/FORMOSA/OUTPUTS/SR12C/ATMO3000/2026-04-24_MIRI+photo_BB_Av_noisescaling/config_file_ref.ini',
# '/home/mravet/Documents/These/FORMOSA/OUTPUTS/SR12C/ATMO3000/2026-04-24_photo_BB_Av/config_file_ref.ini'
# ]

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# for config_file_path in reversed(config_paths):

#     if rank == 0:
#         print(f"\n=== Running config: {config_file_path} ===\n")
#         global_params = GlobFile(config_file_path)
#     else:
#         global_params = None

#     # Broadcast to all ranks
#     global_params = comm.bcast(global_params, root=0)

#     # Run nested sampling
#     launch_nested_sampling(global_params)

#     # Synchronize before next run
#     comm.Barrier()

#     # Optional: clean memory
#     del global_params


# # Launch MPI for leave-one-out / leave-block-out
# # ----------------------------------------------------------------------------------------------------------------------

# # Prepare wav fit lists

# # Choose which inversion
# base_path = '/home/mravet/Documents/These/FORMOSA/OUTPUTS/COCONUT2b/ATMO2020++/2025-09-24_All_leave-block-out_chi2noisescaling/'

# # First run with all the data
# global_params = GlobFile(base_path + 'config_file_ref.ini')
# # Launch the inversion (not parallelized)
# # launch_nested_sampling(global_params=global_params)


# # Get wavelength arrays
# wav_all = []
# for indobs, obs in enumerate(sorted(glob.glob(global_params.observation_path))):
#     obs_name = os.path.splitext(os.path.basename(obs))[0]
#     obs_dict = dict(np.load(os.path.join(global_params.result_path, f'spectrum_obs_{obs_name}.npz'), allow_pickle=True))
#     # Store
#     wav_all.append(obs_dict['wav_spectro'])

# # Leave-block-out arrays
# wav_fit = leave_block_out(wav_all)

# # Loop the leave-one-out
# for i in range(len(wav_fit)):

#     # Create reference config file
#     global_params = GlobFile(base_path + 'config_file_ref.ini')

#     # [config_path]
#     # Change the result path to avoid overwriting
#     global_params.result_path = base_path + f'run_loo_{i}/'
#     global_params.config['config_path']['result_path'] = base_path + f'run_loo_{i}/'
#     # Create the directory if it does not exist
#     if not os.path.exists(global_params.config['config_path']['result_path']):
#         os.makedirs(global_params.config['config_path']['result_path'])
#     # shutil.copy(base_path + "spectrum_obs_1_COCONUTS_2b_GPI_Flamingos2.npz", base_path + f'run_loo_{i}/')
#     # shutil.copy(base_path + "spectrum_obs_2_COCONUTS_2b_NIRSpec_f290_g395H.npz", base_path + f'run_loo_{i}/')
#     # shutil.copy(base_path + "spectrum_obs_3_COCONUTS_2b_MIRI_LRS_corr.npz", base_path + f'run_loo_{i}/')

#     # [config_inversion]
#     global_params.wav_fit = wav_fit[i]
#     global_params.config['config_inversion']['wav_fit'] = wav_fit[i]

#     # Save changes
#     global_params.config.filename = global_params.result_path + f'config_file_ref_loo_{i}.ini'
#     global_params.config.write()

#     # - - - -

#     # Run FORMOSA

#     # Launch the inversion (not parallelized)
#     launch_nested_sampling(global_params=global_params)

#     del global_params
#     gc.collect()