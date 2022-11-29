from __future__ import print_function, division
import numpy as np
import corner
import matplotlib.pyplot as plt
import xarray as xr
import pickle
from matplotlib.figure import Figure
import sys
# Import ForMoSA
base_path = '/Users/simonpetrus/PycharmProjects/ForMoSA_v.1.0/'     # Give the path to ForMoSA to be able to import it. No need when this will be a pip package
sys.path.insert(1, base_path)
from master_main_utilities import GlobFile
from nested_sampling.modif_spec import modif_spec

# ----------------------------------------------------------------------------------------------------------------------


def corner_posterior(global_params, save='no'):
    """
    Plot the posterior distribution of each parameter explored by the nested sampling.

    Args:
        global_params: Class containing each parameter used in ForMoSA
        save: If ='yes' save the figure in a pdf format
    Returns:

    Author: Simon Petrus
    """
    figure_corner = Figure(figsize=(8, 8))

    with open(global_params.result_path + '/result_' + global_params.ns_algo + '.pic', 'rb') as open_pic:
        result = pickle.load(open_pic)
        samples = result.samples
        weights = result.weights

    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine='netcdf4')
    attrs = ds.attrs
    extra_parameters = [
                        ['r', 'R', r'(R$\mathrm{_{Jup}}$)'],
                        ['d', 'd', '(pc)'],
                        ['rv', 'RV', r'(km.s$\mathrm{^{-1}}$)'],
                        ['av', 'Av', '(mag)'],
                        ['vsini', 'v.sin(i)', r'(km.s$\mathrm{^{-1}}$)'],
                        ['ld', 'limb darkening', '']
                        ]

    tot_list_param_title = []
    theta_index = []
    if global_params.par1 != 'NA':
        tot_list_param_title.append(attrs['title'][0] + ' ' + attrs['title'][0])
        theta_index.append('par1')
    if global_params.par2 != 'NA':
        tot_list_param_title.append(attrs['title'][1] + ' ' + attrs['unit'][1])
        theta_index.append('par2')
    if global_params.par3 != 'NA':
        tot_list_param_title.append(attrs['title'][2] + ' ' + attrs['unit'][2])
        theta_index.append('par3')
    if global_params.par4 != 'NA':
        tot_list_param_title.append(attrs['title'][3] + ' ' + attrs['unit'][3])
        theta_index.append('par4')
    if global_params.par5 != 'NA':
        tot_list_param_title.append(attrs['title'][4] + ' ' + attrs['unit'][4])
        theta_index.append('par5')

    if global_params.r != 'NA' and global_params.r[0] != 'constant':
        tot_list_param_title.append(extra_parameters[0][1] + ' ' + extra_parameters[0][2])
        theta_index.append('r')
    if global_params.d != 'NA' and global_params.d[0] != 'constant':
        tot_list_param_title.append(extra_parameters[1][1] + ' ' + extra_parameters[1][2])
        theta_index.append('d')
    if global_params.rv != 'NA' and global_params.rv[0] != 'constant':
        tot_list_param_title.append(extra_parameters[2][1] + ' ' + extra_parameters[2][2])
        theta_index.append('rv')
    if global_params.av != 'NA' and global_params.av[0] != 'constant':
        tot_list_param_title.append(extra_parameters[3][1] + ' ' + extra_parameters[3][2])
        theta_index.append('av')
    if global_params.vsini != 'NA' and global_params.vsini[0] != 'constant':
        tot_list_param_title.append(extra_parameters[4][1] + ' ' + extra_parameters[4][2])
        theta_index.append('vsini')
    if global_params.ld != 'NA' and global_params.ld[0] != 'constant':
        tot_list_param_title.append(extra_parameters[5][1] + ' ' + extra_parameters[5][2])
        theta_index.append('ld')
    theta_index = np.asarray(theta_index)

    posterior_to_plot = []
    for res, result in enumerate(samples):
        if global_params.r != 'NA':
            if global_params.r[0] == "constant":
                r_picked = float(global_params.r[1])
            else:
                ind_theta_r = np.where(theta_index == 'r')
                r_picked = result[ind_theta_r[0]]
            lum = np.log10(4 * np.pi * (r_picked * 69911000.) ** 2 * result[0] ** 4 * 5.670e-8 / 3.83e26)
            result = np.concatenate((result, np.asarray([lum])))
        posterior_to_plot.append(result)
    if global_params.r != 'NA':
        tot_list_param_title.append(r'log(L/L$\mathrm{_{\odot}}$)')

    posterior_to_plot = np.array(posterior_to_plot)
    corner.corner(posterior_to_plot,
                  fig=figure_corner,
                  weights=weights,
                  labels=tot_list_param_title,
                  range=[0.9999] * len(tot_list_param_title),
                  bins=50,
                  quiet=False,
                  top_ticks=False,
                  plot_datapoints=False,
                  plot_density=True,
                  show_titles=True,
                  title_fmt='.2f',
                  title_kwargs=dict(fontsize=9),
                  contour_kwargs=dict(cmap='viridis', colors=None, linewidths=0.7),
                  label_kwargs=dict(fontsize=9)
                  )
    for ax in figure_corner.get_axes():
        ax.tick_params(axis='both', labelsize=9, which='both', direction='in', top=True, right=True)
    if save != 'no':
        figure_corner.savefig(global_params.result_path + '/result_' + global_params.ns_algo + '_corner.pdf',
                              bbox_inches='tight', dpi=300)

# ----------------------------------------------------------------------------------------------------------------------


def get_spec(theta, theta_index, global_params, for_plot='no'):
    # Recovery of the spectroscopy and photometry data
    spectrum_obs = np.load(global_params.result_path + '/spectrum_obs.npz', allow_pickle=True)
    wav_obs_merge = spectrum_obs['obs_merge'][0]
    flx_obs_merge = spectrum_obs['obs_merge'][1]
    err_obs_merge = spectrum_obs['obs_merge'][2]
    if 'obs_pho' in spectrum_obs.keys():
        wav_obs_phot = np.asarray(spectrum_obs['obs_pho'][0])
        flx_obs_phot = np.asarray(spectrum_obs['obs_pho'][1])
        err_obs_phot = np.asarray(spectrum_obs['obs_pho'][2])
    else:
        wav_obs_phot = np.asarray([])
        flx_obs_phot = np.asarray([])
        err_obs_phot = np.asarray([])

    # Recovery of the spectroscopy and photometry model
    path_grid_m = global_params.adapt_store_path + '/adapted_grid_merge_' + global_params.grid_name + '_nonan.nc'
    path_grid_p = global_params.adapt_store_path + 'adapted_grid_phot_' + global_params.grid_name + '_nonan.nc'
    ds = xr.open_dataset(path_grid_m, decode_cf=False, engine='netcdf4')
    grid_merge = ds['grid']
    ds.close()
    ds = xr.open_dataset(path_grid_p, decode_cf=False, engine='netcdf4')
    grid_phot = ds['grid']
    ds.close()

    if global_params.par3 == 'NA':
        if len(grid_merge['wavelength']) != 0:
            flx_mod_merge = grid_merge.interp(par1=theta[0], par2=theta[1],
                                                      method="linear", kwargs={"fill_value": "extrapolate"})
        else:
            flx_mod_merge = []
        if len(grid_phot['wavelength']) != 0:
            flx_mod_phot = grid_phot.interp(par1=theta[0], par2=theta[1],
                                                    method="linear", kwargs={"fill_value": "extrapolate"})
        else:
            flx_mod_phot = []
    elif global_params.par4 == 'NA':
        if len(grid_merge['wavelength']) != 0:
            flx_mod_merge = grid_merge.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                      method="linear", kwargs={"fill_value": "extrapolate"})
        else:
            flx_mod_merge = []
        if len(grid_phot['wavelength']) != 0:
            flx_mod_phot = grid_phot.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                    method="linear", kwargs={"fill_value": "extrapolate"})
        else:
            flx_mod_phot = []
    elif global_params.par5 == 'NA':
        if len(grid_merge['wavelength']) != 0:
            flx_mod_merge = grid_merge.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                      method="linear", kwargs={"fill_value": "extrapolate"})
        else:
            flx_mod_merge = []
        if len(grid_phot['wavelength']) != 0:
            flx_mod_phot = grid_phot.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                    method="linear", kwargs={"fill_value": "extrapolate"})
        else:
            flx_mod_phot = []
    else:
        if len(grid_merge['wavelength']) != 0:
            flx_mod_merge = grid_merge.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                      par5=theta[4],
                                                      method="linear", kwargs={"fill_value": "extrapolate"})
        else:
            flx_mod_merge = []
        if len(grid_phot['wavelength']) != 0:
            flx_mod_phot = grid_phot.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                    par5=theta[4],
                                                    method="linear", kwargs={"fill_value": "extrapolate"})
        else:
            flx_mod_phot = []

    # Modification of the synthetic spectrum with the extra-grid parameters
    modif_spec_chi2 = modif_spec(global_params, theta, theta_index,
                                 wav_obs_merge, flx_obs_merge, err_obs_merge, flx_mod_merge,
                                 wav_obs_phot, flx_obs_phot, err_obs_phot, flx_mod_phot)

    return modif_spec_chi2


def plot_fit(global_params, save='no'):
    """
    Plot the best fit comparing with the data.

    Args:
        global_params: Class containing each parameter used in ForMoSA
        save: If ='yes' save the figure in a pdf format
    Returns:

    Author: Simon Petrus
    """
    figure_fit = plt.figure(figsize=(10, 5))

    inter_ext_g = 0.1
    inter_ext_d = 0.1
    inter_ext_h = 0.05
    inter_ext_b = 0.15

    b_fits = inter_ext_b
    g_fits = inter_ext_g
    l_fits = 1-inter_ext_g-inter_ext_d
    h_fits = 1-inter_ext_b-inter_ext_h
    fits = figure_fit.add_axes([g_fits, b_fits, l_fits, h_fits])
    fits.set_xlabel(r'Wavelength (um)', labelpad=5)
    fits.set_ylabel(r'Flux (W.m-2.um-1)', labelpad=5)

    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine='netcdf4')

    with open(global_params.result_path + '/result_' + global_params.ns_algo + '.pic', 'rb') as ns_result:
        result = pickle.load(ns_result)
    samples = result.samples
    logl = result.logl
    ind = np.where(logl == max(logl))
    theta_best = samples[ind][0]

    if global_params.par1 != 'NA':
        theta_index = ['par1']
    if global_params.par2 != 'NA':
        theta_index.append('par2')
    if global_params.par3 != 'NA':
        theta_index.append('par3')
    if global_params.par4 != 'NA':
        theta_index.append('par4')
    if global_params.par5 != 'NA':
        theta_index.append('par5')
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
    spectra = get_spec(theta_best, theta_index, global_params, for_plot='yes')

    if global_params.model_name == 'SONORA':
        col_pair = 'peru'
    if global_params.model_name == 'ATMO':
        col_pair = 'darkgreen'
    if global_params.model_name == 'BTSETTL':
        col_pair = 'firebrick'
    if global_params.model_name == 'EXOREM':
        col_pair = 'mediumblue'
    if global_params.model_name == 'DRIFTPHOENIX':
        col_pair = 'darkviolet'
    fits.plot(spectra[0], spectra[1], c='k')
    fits.scatter(spectra[4], spectra[5], c='k')
    fits.plot(spectra[0], spectra[3], c=col_pair)
    fits.scatter(spectra[4], spectra[7], c=col_pair)

    for ns_u_ind, ns_u in enumerate(global_params.wav_fit.split('/')):
        min_ns_u = float(ns_u.split(',')[0])
        max_ns_u = float(ns_u.split(',')[1])
        fits.fill_between([min_ns_u, max_ns_u],
                          [min(min(spectra[1]), min(spectra[3]))],
                          [max(max(spectra[1]), max(spectra[3]))],
                          color='y',
                          alpha=0.2
                          )

    # plt.show()
    if save != 'no':
        figure_fit.savefig(global_params.result_path + '/result_' + global_params.ns_algo + '_fit.pdf',
                           bbox_inches='tight', dpi=300)

# ----------------------------------------------------------------------------------------------------------------------


# USER configuration path
print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('-> Plot the results')
if len(sys.argv) == 1:
    print('Where is your configuration file?')
    config_file_path = input()
else:
    config_file_path = sys.argv[1]
print()

# CONFIG_FILE reading and defining global parameters
global_params = GlobFile(config_file_path)  # To access any param.: global_params.parameter_name
print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('-> Plotting')
print()
corner_posterior(global_params, save='yes')
plot_fit(global_params, save='yes')
