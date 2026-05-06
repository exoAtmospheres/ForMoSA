import os.path
import tkinter as tk
from tkinter import *
import json
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy.interpolate import interp1d
from astropy.io import fits
import xarray as xr

from ForMoSA.interface.tabcontrol.utilities import isfloat


def plot_adapt(dico_interface_tk, x_place, y_place):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    # Figure definition
    frame_plot = tk.Frame(dico_interface_tk['tab_adapt_grid_data'])
    frame_plot.place(x=x_place, y=y_place, anchor=NW)
    figure = Figure(figsize=(7.3, 5.7), dpi=100)
    inter_pan_v = 0.09
    inter_ext_g = 0.1
    inter_ext_d = 0.02
    inter_ext_h = 0.04
    inter_ext_b = 0.1
    b_reso = inter_ext_b
    g_reso = inter_ext_g
    l_reso = 1 - inter_ext_g - inter_ext_d
    h_reso = 1 - inter_ext_b - inter_ext_h - inter_pan_v
    ax_resolution = figure.add_axes([g_reso, b_reso, l_reso, 0.33*h_reso])
    dico_interface_tk['ax_resolution'] = ax_resolution
    ax_modif = figure.add_axes([g_reso, b_reso+0.33*h_reso+inter_pan_v, l_reso, 0.67*h_reso])
    dico_interface_tk['ax_modif'] = ax_modif
    bar = FigureCanvasTkAgg(figure, frame_plot)
    NavigationToolbar2Tk(bar, frame_plot)
    bar.get_tk_widget().pack(expand=YES)

    # Bring back the observation and model
    if os.path.exists(dico_interface_tk['observation_path_entry_select'].get()) and \
            os.path.exists(dico_interface_param['model_path']):
        with fits.open(dico_interface_tk['observation_path_entry_select'].get()) as hdul:
            wav_obs = hdul[1].data['WAV']
            flx_obs = hdul[1].data['FLX']
            res_obs = hdul[1].data['RES']

        ds = xr.open_dataset(dico_interface_tk['model_path_entry_select'].get(), decode_cf=False, engine='netcdf4')
        wav_mod = ds['wavelength'].values
        ds.close()
        dwav_mod = wav_mod[1:] - wav_mod[:-1]
        dwav_mod = np.concatenate((dwav_mod, dwav_mod[-2:-1]))
        res_mod = wav_mod / dwav_mod
        res_mod[-1] = res_mod[-2]

        min_res_tab = []
        max_res_tab = []
        # Plot observation data and resolution
        ax_modif.plot(wav_obs, flx_obs, label='Observation', c='royalblue', alpha=0.8)
        ax_resolution.plot(wav_obs, res_obs, label='Observation', c='royalblue', alpha=0.8)
        min_res_tab.append(min(res_obs))
        max_res_tab.append(max(res_obs))

        # Plot model resolution
        f_res_mod_to_obs = interp1d(wav_mod, res_mod, fill_value="extrapolate")
        res_mod_to_obs = f_res_mod_to_obs(wav_obs)
        ax_resolution.plot(wav_obs, res_mod_to_obs, label='Model', c='firebrick', alpha=0.8)
        min_res_tab.append(min(res_mod_to_obs))
        max_res_tab.append(max(res_mod_to_obs))

        # Plot custom resolution if chosen
        if dico_interface_param['custom_reso'] != 'NA':
            res_custom = wav_obs * 0 + float(dico_interface_param['custom_reso'])
            ax_resolution.plot(wav_obs, res_custom, c='forestgreen', label='Custom', alpha=0.8)
            min_res_tab.append(min(res_custom))
            max_res_tab.append(max(res_custom))

        # Plot the resolution target (minimum at each wavelength)
        if dico_interface_param['custom_reso'] != 'NA':
            ax_resolution.plot(wav_obs, np.min(np.asarray([res_obs, res_mod_to_obs, res_custom]), axis=0), '--',
                               c='k', label='Target resolution')
        else:
            ax_resolution.plot(wav_obs, np.min(np.asarray([res_obs, res_mod_to_obs]), axis=0), '--',
                               c='k', label='Target resolution')

        # Plot parameters
        ax_modif.set_ylabel('Flux (W.m-2.s-1)')
        ax_resolution.set_ylabel('Spectral resolution')
        ax_resolution.set_yscale('log')
        ax_resolution.set_xlabel('Wavelength (um)')
        min_compare_reso = min(min_res_tab) / 2
        max_compare_reso = 2 * max(max_res_tab)
        ax_modif.legend(loc='upper center', ncol=4)
        ax_resolution.set_xlabel('Wavelength (um)')
        ax_resolution.set_ylim(min_compare_reso, max_compare_reso)
        ax_resolution.set_ylabel('Spectral resolution')
        ax_resolution.legend(loc='upper center', ncol=4)

        for axe in ['wav', 'flx']:
            if axe+'_lim_select_adapt' in dico_interface_param.keys():
                lim = dico_interface_param[axe+'_lim_select_adapt']
                if lim != '':
                    lim = lim.split(',')
                    if isfloat(lim[0]) is True and isfloat(lim[1]) is True:
                        if float(lim[0]) < float(lim[1]):
                            if axe == 'wav':
                                ax_modif.set_xlim(float(lim[0]), float(lim[1]))
                                ax_resolution.set_xlim(float(lim[0]), float(lim[1]))
                            if axe == 'flx':
                                ax_modif.set_ylim(float(lim[0]), float(lim[1]))

        # Plot the range used to adapt the data and the grid
        from ForMoSA.interface.tabcontrol.utilities import calc_band
        calc_band(dico_interface_tk, 'wav_for_adapt')

        with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
            dico_interface_param = json.load(dico)

        for c, cut in enumerate(dico_interface_param['wav_for_adapt_tab'][0]):
            if c == 0:
                ax_modif.fill_between([dico_interface_param['wav_for_adapt_tab'][0][c],
                                       dico_interface_param['wav_for_adapt_tab'][1][c]], [min(flx_obs), min(flx_obs)],
                                      [max(flx_obs), max(flx_obs)], color='k', alpha=0.1, zorder=-20,
                                      label='$\mathrm{\Delta\lambda}$ for the fit')
            else:
                ax_modif.fill_between([dico_interface_param['wav_for_adapt_tab'][0][c],
                                       dico_interface_param['wav_for_adapt_tab'][1][c]], [min(flx_obs), min(flx_obs)],
                                      [max(flx_obs), max(flx_obs)], color='k', alpha=0.1, zorder=-20)

        # Plot the continuum to subtract
        if dico_interface_param['continuum_sub'] != 'NA':
            from ForMoSA.interface.tabcontrol.utilities import calc_continuum
            calc_continuum(dico_interface_tk)

            with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
                dico_interface_param = json.load(dico)

            for c, cut in enumerate(dico_interface_param['continuum_sub_wav']):
                if c == 0:
                    ax_modif.plot(dico_interface_param['continuum_sub_wav'][c],
                                  dico_interface_param['continuum_sub_flx'][c], c='orange', label='Continuum')
                else:
                    ax_modif.plot(dico_interface_param['continuum_sub_wav'][c],
                                  dico_interface_param['continuum_sub_flx'][c], c='orange')

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as fp:
        json.dump(dico_interface_param, fp)
