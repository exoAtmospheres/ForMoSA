import os.path
import tkinter as tk
from tkinter import *
import json
import xarray as xr
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from ForMoSA.interface.tabcontrol.utilities import isfloat


def plot_fit(dico_interface_tk, x_place, y_place):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    # Figure definition
    frame_plot = tk.Frame(dico_interface_tk['tab_fit'])
    frame_plot.place(x=x_place, y=y_place, anchor=NW)
    figure = Figure(figsize=(7.7, 5.7), dpi=100)
    inter_pan_v = 0.09
    inter_ext_g = 0.1
    inter_ext_d = 0.02
    inter_ext_h = -0.04
    inter_ext_b = 0.1
    b_reso = inter_ext_b
    g_reso = inter_ext_g
    l_reso = 1 - inter_ext_g - inter_ext_d
    h_reso = 1 - inter_ext_b - inter_ext_h - inter_pan_v
    ax_nested_sampling = figure.add_axes([g_reso, b_reso, l_reso, h_reso])
    dico_interface_tk['ax_nested_sampling'] = ax_nested_sampling
    bar = FigureCanvasTkAgg(figure, frame_plot)
    NavigationToolbar2Tk(bar, frame_plot)
    bar.get_tk_widget().pack(expand=YES)

    # Calculate and plot the data adapted for the fit
    from ForMoSA.interface.utilities.globfile_interface import GlobFile
    global_params = GlobFile(dico_interface_tk['config_file_path'])
    global_params.model_path = dico_interface_param['model_path']
    global_params.adapt_method = dico_interface_param['adapt_method']
    global_params.observation_path = dico_interface_param['observation_path']
    global_params.wav_for_adapt = dico_interface_param['wav_for_adapt']
    global_params.wav_for_continuum = dico_interface_param['wav_for_continuum']
    global_params.continuum_sub = dico_interface_param['continuum_sub']
    global_params.custom_reso = dico_interface_param['custom_reso']
    global_params.result_path = dico_interface_param['result_path']
    if os.path.exists(global_params.model_path) and os.path.exists(global_params.observation_path):
        from ForMoSA.adapt.adapt_obs_mod import launch_adapt
        launch_adapt(global_params, justobs='yes')
        spectrum_obs = np.load(global_params.result_path + '/spectrum_obs.npz', allow_pickle=True)
        obs_merge = spectrum_obs['obs_merge']
        obs_pho = spectrum_obs['obs_pho']
        ax_nested_sampling.plot(obs_merge[0], obs_merge[1], label='Observation', c='royalblue', alpha=0.8, zorder=-1)
        ax_nested_sampling.scatter(obs_pho[0], obs_pho[1], label='Observation', c='royalblue', alpha=0.8, zorder=-1)

        # Calculate and plot the model adapted for the fit
        ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine='netcdf4')
        grid = ds['grid']
        attr = ds.attrs
        ds.close()
        for key_ind, key in enumerate(attr['key']):
            dico_interface_param[attr['par'][key_ind]+'_choice_fit_val'] = \
                dico_interface_tk[attr['par'][key_ind]+'_choice_fit'].get()
            if key_ind == 0:
                grid = grid.sel(par1=float(dico_interface_param[attr['par'][key_ind]+'_choice_fit_val']))
            if key_ind == 1:
                grid = grid.sel(par2=float(dico_interface_param[attr['par'][key_ind]+'_choice_fit_val']))
            if key_ind == 2:
                grid = grid.sel(par3=float(dico_interface_param[attr['par'][key_ind]+'_choice_fit_val']))
            if key_ind == 3:
                grid = grid.sel(par4=float(dico_interface_param[attr['par'][key_ind]+'_choice_fit_val']))
            if key_ind == 4:
                grid = grid.sel(par5=float(dico_interface_param[attr['par'][key_ind]+'_choice_fit_val']))
        wav_model = grid['wavelength'].values
        res_mod = attr['res']
        flx_model = grid.values
        from ForMoSA.adapt.extraction_functions import adapt_model
        mod_merge, mod_pho = adapt_model(global_params, wav_model, flx_model, res_mod)
        ck = (np.sum(mod_merge * obs_merge[1] / (obs_merge[2] ** 2))) / (np.sum((mod_merge/obs_merge[2]) ** 2))
        if len(obs_merge[0]) != 0:
            ax_nested_sampling.plot(obs_merge[0], mod_merge * ck, label='Model', c='firebrick', alpha=0.8, zorder=-1)
        if len(obs_pho[0]) != 0:
            ax_nested_sampling.scatter(obs_pho[0], mod_pho * ck, label='Model', c='firebrick', alpha=0.8, zorder=-1)

        # Plot parameters
        ax_nested_sampling.set_ylabel('Flux (W.m-2.s-1)')
        ax_nested_sampling.set_xlabel('Wavelength (um)')
        ax_nested_sampling.legend(loc='upper center', ncol=2).set_zorder(22)

        for axe in ['wav', 'flx']:
            if axe+'_lim_select_fit' in dico_interface_param.keys():
                lim = dico_interface_param[axe+'_lim_select_fit']
                if lim != '':
                    lim = lim.split(',')
                    if isfloat(lim[0]) is True and isfloat(lim[1]) is True:
                        if float(lim[0]) < float(lim[1]):
                            if axe == 'wav':
                                ax_nested_sampling.set_xlim(float(lim[0]), float(lim[1]))
                            if axe == 'flx':
                                ax_nested_sampling.set_ylim(float(lim[0]), float(lim[1]))

        # Plot the range used to adapt the data and the grid
        from ForMoSA.interface.tabcontrol.utilities import calc_band
        calc_band(dico_interface_tk, 'wav_for_adapt')

        with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
            dico_interface_param = json.load(dico)

        for c, cut in enumerate(dico_interface_param['wav_for_adapt_tab'][0]):
            if c == 0:
                ax_nested_sampling.fill_between([dico_interface_param['wav_for_adapt_tab'][0][c],
                                                 dico_interface_param['wav_for_adapt_tab'][1][c]],
                                                [min(obs_merge[1]), min(obs_merge[1])],
                                                [max(obs_merge[1]), max(obs_merge[1])], color='k', alpha=0.1,
                                                zorder=-20, label='$\mathrm{\Delta\lambda}$ for the fit')
            else:
                ax_nested_sampling.fill_between([dico_interface_param['wav_for_adapt_tab'][0][c],
                                                 dico_interface_param['wav_for_adapt_tab'][1][c]],
                                                [min(obs_merge[1]), min(obs_merge[1])],
                                                [max(obs_merge[1]), max(obs_merge[1])], color='k', alpha=0.1,
                                                zorder=-20)

        # Plot the range used for the fit
        calc_band(dico_interface_tk, 'wav_fit')

        with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
            dico_interface_param = json.load(dico)

        for c, cut in enumerate(dico_interface_param['wav_fit_tab'][0]):
            if c == 0:
                ax_nested_sampling.fill_between([dico_interface_param['wav_fit_tab'][0][c],
                                                 dico_interface_param['wav_fit_tab'][1][c]],
                                                [min(obs_merge[1]), min(obs_merge[1])],
                                                [max(obs_merge[1]), max(obs_merge[1])], color='y', alpha=0.1,
                                                zorder=-20, label='$\mathrm{\Delta\lambda}$ for the fit')
            else:
                ax_nested_sampling.fill_between([dico_interface_param['wav_fit_tab'][0][c],
                                                 dico_interface_param['wav_fit_tab'][1][c]],
                                                [min(obs_merge[1]), min(obs_merge[1])],
                                                [max(obs_merge[1]), max(obs_merge[1])], color='y', alpha=0.1,
                                                zorder=-20)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as fp:
        json.dump(dico_interface_param, fp)
