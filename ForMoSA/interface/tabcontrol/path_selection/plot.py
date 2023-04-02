import os.path
import tkinter as tk
from tkinter import *
import json
import xarray as xr
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy.interpolate import interp1d
from astropy.io import fits

from ForMoSA.interface.tabcontrol.utilities import isfloat


def plot_obs_vs_model(dico_interface_tk, x_place, y_place):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    frame_plot = tk.Frame(dico_interface_tk['tab_path_selection'])
    frame_plot.place(x=x_place, y=y_place, anchor=NW)
    figure = Figure(figsize=(8.8, 3.6), dpi=100)
    ax_observation_v_model = figure.add_axes([0.06, 0.15, 0.91, 0.77])
    dico_interface_tk['ax_observation_v_model'] = ax_observation_v_model
    bar = FigureCanvasTkAgg(figure, frame_plot)
    NavigationToolbar2Tk(bar, frame_plot)
    bar.get_tk_widget().pack(expand=YES)

    if os.path.exists(dico_interface_param['model_path']):
        ds = xr.open_dataset(dico_interface_param['model_path'], decode_cf=False, engine='netcdf4')
        grid = ds['grid']
        attr = ds.attrs
        ds.close()

        test = 1
        for key_ind, key in enumerate(attr['key']):
            dico_interface_param[attr['par'][key_ind]+'_choice_data_model_val'] = \
                dico_interface_tk[attr['par'][key_ind]+'_choice_data_model'].get()
            if isfloat(dico_interface_param[attr['par'][key_ind]+'_choice_data_model_val']) is True:
                if key_ind == 0:
                    grid = grid.sel(par1=float(dico_interface_param[attr['par'][key_ind]+'_choice_data_model_val']))
                if key_ind == 1:
                    grid = grid.sel(par2=float(dico_interface_param[attr['par'][key_ind]+'_choice_data_model_val']))
                if key_ind == 2:
                    grid = grid.sel(par3=float(dico_interface_param[attr['par'][key_ind]+'_choice_data_model_val']))
                if key_ind == 3:
                    grid = grid.sel(par4=float(dico_interface_param[attr['par'][key_ind]+'_choice_data_model_val']))
                if key_ind == 4:
                    grid = grid.sel(par5=float(dico_interface_param[attr['par'][key_ind]+'_choice_data_model_val']))
            else:
                test *= 0
        if test == 1:
            with fits.open(dico_interface_tk['observation_path_entry_select'].get()) as hdul:
                wav_obs = hdul[1].data['WAV']
                flx_obs = hdul[1].data['FLX']
                err_obs = hdul[1].data['ERR']
                res_obs = hdul[1].data['RES']
            wav_model = grid['wavelength']
            flx_model = grid.values
            f_model_to_obs = interp1d(wav_model, flx_model, fill_value='extrapolate')
            flx_model_to_norm = f_model_to_obs(wav_obs)
            ck = (np.sum(flx_model_to_norm * flx_obs / (err_obs ** 2))) / (np.sum((flx_model_to_norm/err_obs) ** 2))
            ax_observation_v_model.plot(wav_model, flx_model*ck, label='Model', c='firebrick', alpha=0.8)
            ind_no_pho = np.where(res_obs != 0.0)
            ind_pho = np.where(res_obs == 0.0)
            ax_observation_v_model.plot(wav_obs[ind_no_pho], flx_obs[ind_no_pho], label='Observation', c='royalblue',
                                        alpha=0.8)
            ax_observation_v_model.scatter(wav_obs[ind_pho], flx_obs[ind_pho], label='Observation', c='royalblue',
                                           alpha=0.8)
            ax_observation_v_model.legend(loc='upper center', ncol=2)

        ax_observation_v_model.set_ylabel('Flux (W.m-2.s-1)')
        ax_observation_v_model.set_xlabel('Wavelength (um)')
        for axe in ['wav', 'flx']:
            if axe+'_lim_select_data_model' in dico_interface_param.keys():
                lim = dico_interface_param[axe+'_lim_select_data_model']
                if lim != '':
                    lim = lim.split(',')
                    if isfloat(lim[0]) is True and isfloat(lim[1]) is True:
                        if float(lim[0]) < float(lim[1]):
                            if axe == 'wav':
                                ax_observation_v_model.set_xlim(float(lim[0]), float(lim[1]))
                            if axe == 'flx':
                                ax_observation_v_model.set_ylim(float(lim[0]), float(lim[1]))

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as fp:
        json.dump(dico_interface_param, fp)
