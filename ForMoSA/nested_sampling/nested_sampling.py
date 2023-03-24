import numpy as np
import nestle
import time
import xarray as xr
import pickle

from nested_sampling.nested_modif_spec import modif_spec
from nested_sampling.nested_prior_function import uniform_prior, gaussian_prior


def loglike(theta, theta_index, global_params, for_plot='no'):
    """
    

    Args:
        
    Returns:
        

    Author: Simon Petrus
    """
    # Recovery of the spectroscopy and photometry data
    spectrum_obs = np.load(global_params.result_path + '/spectrum_obs.npz', allow_pickle=True)
    wav_obs_merge = spectrum_obs['obs_merge'][0]
    flx_obs_merge = spectrum_obs['obs_merge'][1]
    err_obs_merge = spectrum_obs['obs_merge'][2]
    if 'obs_pho' in spectrum_obs.keys():
        wav_obs_phot = np.asarray(spectrum_obs['obs_pho'][0], dtype=float)
        flx_obs_phot = np.asarray(spectrum_obs['obs_pho'][1], dtype=float)
        err_obs_phot = np.asarray(spectrum_obs['obs_pho'][2], dtype=float)
    else:
        wav_obs_phot = np.asarray([], dtype=float)
        flx_obs_phot = np.asarray([], dtype=float)
        err_obs_phot = np.asarray([], dtype=float)

    # Recovery of the spectroscopy and photometry model
    path_grid_m = global_params.adapt_store_path + 'adapted_grid_merge_' + global_params.grid_name + '_nonan.nc'
    path_grid_p = global_params.adapt_store_path + 'adapted_grid_phot_' + global_params.grid_name + '_nonan.nc'
    ds = xr.open_dataset(path_grid_m, decode_cf=False, engine='netcdf4')
    grid_merge = ds['grid']
    ds.close()
    ds = xr.open_dataset(path_grid_p, decode_cf=False, engine='netcdf4')
    grid_phot = ds['grid']
    ds.close()

    # Calculation of the likelihood for each sub-spectrum defined by the parameter 'wav_fit'
    for ns_u_ind, ns_u in enumerate(global_params.wav_fit.split('/')):
        min_ns_u = float(ns_u.split(',')[0])
        max_ns_u = float(ns_u.split(',')[1])
        ind_grid_merge_sel = np.where((grid_merge['wavelength'] >= min_ns_u) & (grid_merge['wavelength'] <= max_ns_u))
        ind_grid_phot_sel = np.where((grid_phot['wavelength'] >= min_ns_u) & (grid_phot['wavelength'] <= max_ns_u))

        # Cutting of the grid on the wavelength grid defined by the parameter 'wav_fit'
        grid_merge_cut = grid_merge.sel(wavelength=grid_merge['wavelength'][ind_grid_merge_sel])
        grid_phot_cut = grid_phot.sel(wavelength=grid_phot['wavelength'][ind_grid_phot_sel])

        # Interpolation of the grid at the theta parameters set
        if global_params.par3 == 'NA':
            if len(grid_merge_cut['wavelength']) != 0:
                flx_mod_merge_cut = grid_merge_cut.interp(par1=theta[0], par2=theta[1],
                                                          method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_merge_cut = []
            if len(grid_phot_cut['wavelength']) != 0:
                flx_mod_phot_cut = grid_phot_cut.interp(par1=theta[0], par2=theta[1],
                                                        method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_phot_cut = []
        elif global_params.par4 == 'NA':
            if len(grid_merge_cut['wavelength']) != 0:
                flx_mod_merge_cut = grid_merge_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                          method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_merge_cut = []
            if len(grid_phot_cut['wavelength']) != 0:
                flx_mod_phot_cut = grid_phot_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                        method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_phot_cut = []
        elif global_params.par5 == 'NA':
            if len(grid_merge_cut['wavelength']) != 0:
                flx_mod_merge_cut = grid_merge_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                          method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_merge_cut = []
            if len(grid_phot_cut['wavelength']) != 0:
                flx_mod_phot_cut = grid_phot_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                        method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_phot_cut = []
        else:
            if len(grid_merge_cut['wavelength']) != 0:
                flx_mod_merge_cut = grid_merge_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                          par5=theta[4],
                                                          method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_merge_cut = []
            if len(grid_phot_cut['wavelength']) != 0:
                flx_mod_phot_cut = grid_phot_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                        par5=theta[4],
                                                        method="linear", kwargs={"fill_value": "extrapolate"})
            else:
                flx_mod_phot_cut = []


        # Re-merging of the data and interpolated synthetic spectrum to a wavelength grid defined by the parameter 'wav_fit'
        ind_merge = np.where((wav_obs_merge >= min_ns_u) & (wav_obs_merge <= max_ns_u))
        wav_obs_phot = np.array(wav_obs_phot,dtype=float)
        ind_phot = np.where((wav_obs_phot >= min_ns_u) & (wav_obs_phot <= max_ns_u))
        if ns_u_ind == 0:
            wav_obs_merge_ns_u = wav_obs_merge[ind_merge]
            flx_obs_merge_ns_u = flx_obs_merge[ind_merge]
            err_obs_merge_ns_u = err_obs_merge[ind_merge]
            flx_mod_merge_ns_u = flx_mod_merge_cut
            wav_obs_phot_ns_u = wav_obs_phot[ind_phot]
            flx_obs_phot_ns_u = flx_obs_phot[ind_phot]
            err_obs_phot_ns_u = err_obs_phot[ind_phot]
            flx_mod_phot_ns_u = flx_mod_phot_cut
        else:
            wav_obs_merge_ns_u = np.concatenate((wav_obs_merge_ns_u, wav_obs_merge[ind_merge]))
            flx_obs_merge_ns_u = np.concatenate((flx_obs_merge_ns_u, flx_obs_merge[ind_merge]))
            err_obs_merge_ns_u = np.concatenate((err_obs_merge_ns_u, err_obs_merge[ind_merge]))
            flx_mod_merge_ns_u = np.concatenate((flx_mod_merge_ns_u, flx_mod_merge_cut))
            wav_obs_phot_ns_u = np.concatenate((wav_obs_phot_ns_u, wav_obs_phot[ind_phot]))
            flx_obs_phot_ns_u = np.concatenate((flx_obs_phot_ns_u, flx_obs_phot[ind_phot]))
            err_obs_phot_ns_u = np.concatenate((err_obs_phot_ns_u, err_obs_phot[ind_phot]))
            flx_mod_phot_ns_u = np.concatenate((flx_mod_phot_ns_u, flx_mod_phot_cut))

    # Modification of the synthetic spectrum with the extra-grid parameters
    modif_spec_chi2 = modif_spec(global_params, theta, theta_index,
                                 wav_obs_merge_ns_u,  flx_obs_merge_ns_u,  err_obs_merge_ns_u,  flx_mod_merge_ns_u,
                                 wav_obs_phot_ns_u,  flx_obs_phot_ns_u, err_obs_phot_ns_u,  flx_mod_phot_ns_u)

    #print(modif_spec_chi2)
    # Merging the spectroscopy with photometry in order to calculate the likelihood
    #flx_obs_chi2 = np.array(np.concatenate((modif_spec_chi2[1], flx_obs_phot_ns_u)),dtype=float)
    #err_obs_chi2 = np.array(np.concatenate((modif_spec_chi2[2], err_obs_phot_ns_u)),dtype=float)
    #flx_mod_chi2 = np.array(np.concatenate((modif_spec_chi2[3], modif_spec_chi2[5])),dtype=float)
    flx_obs_chi2 = np.array(np.concatenate((modif_spec_chi2[1], modif_spec_chi2[5])), dtype=float)
    err_obs_chi2 = np.array(np.concatenate((modif_spec_chi2[2], modif_spec_chi2[6])), dtype=float)
    flx_mod_chi2 = np.array(np.concatenate((modif_spec_chi2[3], modif_spec_chi2[7])), dtype=float)
    #print('flx, mod, chi2', flx_mod_chi2)
    #print('array-0',modif_spec_chi2[0])
    #print('array-1',modif_spec_chi2[1])
    #print('array-2',modif_spec_chi2[2])
    #print('array-3',modif_spec_chi2[3])
    #print('array-4',modif_spec_chi2[4])
    #print('array-5',modif_spec_chi2[5])
    # Calculation of the chi2 (used to calculate the likelihood)
    chisq = np.sum(((flx_obs_chi2 - flx_mod_chi2) / err_obs_chi2) ** 2)
    #print(chisq)

    if for_plot == 'no':
        return -chisq/2.
    else:
        return modif_spec_chi2


def prior_transform(theta, theta_index, lim_param_grid, global_params):
    """
    

    Args:
        
    Returns:
        

    Author: Simon Petrus
    """
    prior = []
    if global_params.par1 != 'NA':
        prior_law = global_params.par1[0]
        if prior_law != 'constant':
            if prior_law == 'uniform':
                prior_par1 = uniform_prior([float(global_params.par1[1]), float(global_params.par1[2])], theta[0])
            if prior_law == 'gaussian':
                prior_par1 = gaussian_prior([float(global_params.par1[1]), float(global_params.par1[2])], theta[0])
            if prior_par1 < lim_param_grid[0][0]:
                prior_par1 = lim_param_grid[0][0]
            elif prior_par1 > lim_param_grid[0][1]:
                prior_par1 = lim_param_grid[0][1]
            prior.append(prior_par1)
    if global_params.par2 != 'NA':
        prior_law = global_params.par2[0]
        if prior_law != 'constant':
            if prior_law == 'uniform':
                prior_par2 = uniform_prior([float(global_params.par2[1]), float(global_params.par2[2])], theta[1])
            if prior_law == 'gaussian':
                prior_par2 = gaussian_prior([float(global_params.par2[1]), float(global_params.par2[2])], theta[1])
            if prior_par2 < lim_param_grid[1][0]:
                prior_par2 = lim_param_grid[1][0]
            elif prior_par2 > lim_param_grid[1][1]:
                prior_par2 = lim_param_grid[1][1]
            prior.append(prior_par2)
    if global_params.par3 != 'NA':
        prior_law = global_params.par3[0]
        if prior_law != 'constant':
            if prior_law == 'uniform':
                prior_par3 = uniform_prior([float(global_params.par3[1]), float(global_params.par3[2])], theta[2])
            if prior_law == 'gaussian':
                prior_par3 = gaussian_prior([float(global_params.par3[1]), float(global_params.par3[2])], theta[2])
            if prior_par3 < lim_param_grid[2][0]:
                prior_par3 = lim_param_grid[2][0]
            elif prior_par3 > lim_param_grid[2][1]:
                prior_par3 = lim_param_grid[2][1]
            prior.append(prior_par3)
    if global_params.par4 != 'NA':
        prior_law = global_params.par4[0]
        if prior_law != 'constant':
            if prior_law == 'uniform':
                prior_par4 = uniform_prior([float(global_params.par4[1]), float(global_params.par4[2])], theta[3])
            if prior_law == 'gaussian':
                prior_par4 = gaussian_prior([float(global_params.par4[1]), float(global_params.par4[2])], theta[3])
            if prior_par4 < lim_param_grid[3][0]:
                prior_par4 = lim_param_grid[3][0]
            elif prior_par4 > lim_param_grid[3][1]:
                prior_par4 = lim_param_grid[3][1]
            prior.append(prior_par4)
    if global_params.par5 != 'NA':
        prior_law = global_params.par5[0]
        if prior_law != 'constant':
            if prior_law == 'uniform':
                prior_par5 = uniform_prior([float(global_params.par5[1]), float(global_params.par5[2])], theta[4])
            if prior_law == 'gaussian':
                prior_par5 = gaussian_prior([float(global_params.par5[1]), float(global_params.par5[2])], theta[4])
            if prior_par5 < lim_param_grid[4][0]:
                prior_par5 = lim_param_grid[4][0]
            elif prior_par5 > lim_param_grid[4][1]:
                prior_par5 = lim_param_grid[4][1]
            prior.append(prior_par5)

    if global_params.r != 'NA':
        prior_law = global_params.r[0]
        if prior_law != 'constant':
            ind_theta_r = np.where(theta_index == 'r')
            if prior_law == 'uniform':
                prior_r = uniform_prior([float(global_params.r[1]), float(global_params.r[2])], theta[ind_theta_r[0][0]])
            if prior_law == 'gaussian':
                prior_r = gaussian_prior([float(global_params.r[1]), float(global_params.r[2])], theta[ind_theta_r[0][0]])
            prior.append(prior_r)
    if global_params.d != 'NA':
        prior_law = global_params.d[0]
        if prior_law != 'constant':
            ind_theta_d = np.where(theta_index == 'd')
            if prior_law == 'uniform':
                prior_d = uniform_prior([float(global_params.d[1]), float(global_params.d[2])], theta[ind_theta_d[0][0]])
            if prior_law == 'gaussian':
                prior_d = gaussian_prior([float(global_params.d[1]), float(global_params.d[2])], theta[ind_theta_d[0][0]])
            prior.append(prior_d)
    if global_params.rv != 'NA':
        prior_law = global_params.rv[0]
        if prior_law != 'constant':
            ind_theta_rv = np.where(theta_index == 'rv')
            if prior_law == 'uniform':
                prior_rv = uniform_prior([float(global_params.rv[1]), float(global_params.rv[2])], theta[ind_theta_rv[0][0]])
            if prior_law == 'gaussian':
                prior_rv = gaussian_prior([float(global_params.rv[1]), float(global_params.rv[2])], theta[ind_theta_rv[0][0]])
            prior.append(prior_rv)
    if global_params.av != 'NA':
        prior_law = global_params.av[0]
        if prior_law != 'constant':
            ind_theta_av = np.where(theta_index == 'av')
            if prior_law == 'uniform':
                prior_av = uniform_prior([float(global_params.av[1]), float(global_params.av[2])], theta[ind_theta_av[0][0]])
            if prior_law == 'gaussian':
                prior_av = gaussian_prior([float(global_params.av[1]), float(global_params.av[2])], theta[ind_theta_av[0][0]])
            prior.append(prior_av)
    if global_params.vsini != 'NA':
        prior_law = global_params.vsini[0]
        if prior_law != 'constant':
            ind_theta_vsini = np.where(theta_index == 'vsini')
            if prior_law == 'uniform':
                prior_vsini = uniform_prior([float(global_params.vsini[1]), float(global_params.vsini[2])], theta[ind_theta_vsini[0][0]])
            if prior_law == 'gaussian':
                prior_vsini = gaussian_prior([float(global_params.vsini[1]), float(global_params.vsini[2])], theta[ind_theta_vsini[0][0]])
            prior.append(prior_vsini)
    if global_params.ld != 'NA':
        prior_law = global_params.ld[0]
        if prior_law != 'constant':
            ind_theta_ld = np.where(theta_index == 'ld')
            if prior_law == 'uniform':
                prior_ld = uniform_prior([float(global_params.ld[1]), float(global_params.ld[2])], theta[ind_theta_ld[0][0]])
            if prior_law == 'gaussian':
                prior_ld = gaussian_prior([float(global_params.ld[1]), float(global_params.ld[2])], theta[ind_theta_ld[0][0]])
            prior.append(prior_ld)

    return prior


def launch_nested_sampling(global_params):
    """
    

    Args:
        
    Returns:
        

    Author: Simon Petrus
    """
    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine='netcdf4')

    # Count the number of free parameters and identify the parameter position in theta
    if global_params.par1 != 'NA':
        theta_index = ['par1']
        lim_param_grid = [[min(ds['par1'].values), max(ds['par1'].values)]]
    else:
        theta_index = []
        lim_param_grid = []
    if global_params.par2 != 'NA':
        theta_index.append('par2')
        lim_param_grid.append([min(ds['par2'].values), max(ds['par2'].values)])
    if global_params.par3 != 'NA':
        theta_index.append('par3')
        lim_param_grid.append([min(ds['par3'].values), max(ds['par3'].values)])
    if global_params.par4 != 'NA':
        theta_index.append('par4')
        lim_param_grid.append([min(ds['par4'].values), max(ds['par4'].values)])
    if global_params.par5 != 'NA':
        theta_index.append('par5')
        lim_param_grid.append([min(ds['par5'].values), max(ds['par5'].values)])
    n_free_parameters = len(ds.attrs['key'])
    if global_params.r != 'NA' and global_params.r[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('r')
    if global_params.d != 'NA' and global_params.d[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('d')
    if global_params.rv != 'NA' and global_params.rv[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('rv')
    if global_params.av != 'NA' and global_params.av[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('av')
    if global_params.vsini != 'NA' and global_params.vsini[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('vsini')
    if global_params.ld != 'NA' and global_params.ld[0] != 'constant':
        n_free_parameters += 1
        theta_index.append('ld')
    theta_index = np.asarray(theta_index)

    if global_params.ns_algo == 'nestle':
        tmpstot1 = time.time()
        loglike_gp = lambda theta: loglike(theta, theta_index, global_params)
        prior_transform_gp = lambda theta: prior_transform(theta, theta_index, lim_param_grid, global_params)
        result = nestle.sample(loglike_gp, prior_transform_gp, n_free_parameters, callback=nestle.print_progress,npoints=int(float(global_params.npoint)))
        tmpstot2 = time.time()-tmpstot1
        print('\n')
        print('########     The code spent ' + str(tmpstot2) + ' sec to run   ########')
        print(result.summary())
        print('\n\n')
    
    if global_params.ns_algo == 'dynesty':
        from dynesty import NestedSampler

        # initialize our nested sampler
        #sampler = NestedSampler(loglike, ptform, ndim)

    with open(global_params.result_path + '/result_' + global_params.ns_algo + '.pic', 'wb') as f1:
        pickle.dump(result, f1)
    
    print(' ')
    print('-> Voilà, on est prêt')

    return 

# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    from main_utilities import GlobFile

    # USER configuration path
    print()
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('-> Configuration of environment')
    print('Where is your configuration file?')
    config_file_path = input()
    print()

    # CONFIG_FILE reading and defining global parameters
    global_params = GlobFile(config_file_path)  # To access any param.: global_params.parameter_name
    print()
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('-> Nested sampling')
    print()
    launch_nested_sampling(global_params)
