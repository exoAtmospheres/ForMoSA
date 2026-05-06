import os.path
from tkinter import ttk
import tkinter as tk
from tkinter import *
import json
import xarray as xr
from astropy.io import fits
import numpy as np


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def define_grid_parameter_to_plot(dico_interface_tk, parent):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    fram_select_param = tk.Frame(dico_interface_tk[parent])
    dico_interface_tk['fram_select_param'] = fram_select_param
    fram_select_param.grid(row=0, column=0, sticky=W)
    label_title = tk.Label(fram_select_param, text='Parameters of the model to plot:', font='Arial 13 bold')
    label_title.grid(row=0, column=0, columnspan=2, sticky=W)

    if parent == 'frame_select_data_model_param_plot':
        key_choice = '_choice_data_model'
    elif parent == 'frame_fit_param_plot':
        key_choice = '_choice_fit'
    else:
        pass

    if os.path.exists(dico_interface_param['model_path']):
        ds = xr.open_dataset(dico_interface_param['model_path'], decode_cf=False, engine='netcdf4')
        attr = ds.attrs
        for key_ind, key in enumerate(attr['key']):
            par_title = tk.Label(fram_select_param,
                                 text=attr['title'][key_ind] + ' ' + attr['unit'][key_ind] + ': ')
            par_title.grid(row=key_ind + 1, column=0, sticky=W)
            parameter_tab = []
            for parameter_val in ds[key].values:
                parameter_tab.append(parameter_val)
            par_choice = ttk.Combobox(fram_select_param, state='readonly', values=parameter_tab, width=13)
            par_choice.grid(row=key_ind + 1, column=1, sticky=W)
            dico_interface_tk[attr['par'][key_ind] + key_choice] = par_choice
            if attr['par'][key_ind] + key_choice + '_val' in dico_interface_param.keys():
                if isfloat(dico_interface_param[attr['par'][key_ind] + key_choice + '_val']) is True:
                    if float(dico_interface_param[attr['par'][key_ind] + key_choice + '_val']) in parameter_tab:
                        par_choice.set(dico_interface_param[attr['par'][key_ind] + key_choice + '_val'])
                    else:
                        par_choice.set('Select a ' + attr['title'][key_ind])
                else:
                    par_choice.set('Select a ' + attr['title'][key_ind])
            else:
                par_choice.set('Select a ' + attr['title'][key_ind])

        button_param_mod = tk.Button(fram_select_param, text='Plot',
                                     command=lambda: validate_define_grid_parameter_to_plot(dico_interface_tk, attr,
                                                                                            parent))
        button_param_mod.grid(row=len(ds.coords.keys()) + 1, column=1, sticky=W)
        ds.close()


def validate_define_grid_parameter_to_plot(dico_interface_tk, attr, parent, plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    if parent == 'frame_select_data_model_param_plot':
        key_choice = '_choice_data_model'
    elif parent == 'frame_fit_param_plot':
        key_choice = '_choice_fit'
    else:
        pass

    for key_ind, key in enumerate(attr['key']):
        dico_interface_param[attr['par'][key_ind]+key_choice+'_val'] = \
            dico_interface_tk[attr['par'][key_ind]+key_choice].get()

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as fp:
        json.dump(dico_interface_param, fp)

    if plot == 'yes':
        if parent == 'frame_select_data_model_param_plot':
            from ForMoSA.interface.tabcontrol.path_selection.plot import plot_obs_vs_model
            plot_obs_vs_model(dico_interface_tk,
                              dico_interface_tk['frame_select_data_model_param_plot'].winfo_width() + 25, 210)
        elif parent == 'frame_fit_param_plot':
            from ForMoSA.interface.tabcontrol.fit.plot import plot_fit
            plot_fit(dico_interface_tk, dico_interface_tk['frame_param_fit'].winfo_width() + 25, 0)
        else:
            pass


def define_xlim_ylim(dico_interface_tk, parent):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    if parent == 'frame_select_data_model_param_plot':
        key_choice = '_data_model'
    elif parent == 'frame_adapt_grid_data_param_plot':
        key_choice = '_adapt'
    elif parent == 'frame_fit_param_plot':
        key_choice = '_fit'
    else:
        pass

    axe_lim_tab = [['wav_lim_select'+key_choice, 'Wavelength limits (um)', '12', 0, 0],
                   ['flx_lim_select'+key_choice, 'Flux limits (W.m-2.um-1)', '12', 1, 0]]

    frame_axe_xy_lim = tk.Frame(dico_interface_tk[parent])
    dico_interface_tk['frame_axe_xy_lim' + key_choice] = frame_axe_xy_lim
    frame_axe_xy_lim.grid(row=2, column=0, sticky=W)
    for axe_lim in axe_lim_tab:
        frame_axe_lim = tk.Frame(frame_axe_xy_lim)
        frame_axe_lim.grid(row=axe_lim[3], column=axe_lim[4], sticky=W)

        frame_axe_lim_title = tk.Frame(frame_axe_lim)
        frame_axe_lim_title.grid(row=0, column=0, sticky=W)
        dico_interface_tk[axe_lim[0]+'_title'] = frame_axe_lim_title
        label_axe_lim_title = tk.Label(frame_axe_lim_title, text=axe_lim[1], font='Arial 13 bold')
        label_axe_lim_title.grid(row=0, column=0, sticky=W)
        frame_axe_lim_buttons = tk.Frame(frame_axe_lim)
        frame_axe_lim_buttons.grid(row=1, column=0, sticky=W)
        entry_selec_axe_lim = tk.Entry(frame_axe_lim_buttons, width=axe_lim[2])
        dico_interface_tk[axe_lim[0]+'_entry_select'] = entry_selec_axe_lim
        entry_selec_axe_lim.grid(row=0, column=0, sticky=W)

        button_validate = tk.Button(frame_axe_lim_buttons, text='->',
                                    command=lambda: validate_xlim_ylim(dico_interface_tk, axe_lim_tab, parent))
        button_validate.grid(row=0, column=1, sticky=W)

        if axe_lim[0] in dico_interface_param.keys():
            dico_interface_tk[axe_lim[0]+'_entry_select'].delete(0, tk.END)
            dico_interface_tk[axe_lim[0]+'_entry_select'].insert(0, dico_interface_param[axe_lim[0]])


def validate_xlim_ylim(dico_interface_tk, axe_lim_tab, parent, plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    dico_interface_param[axe_lim_tab[0][0]] = dico_interface_tk[axe_lim_tab[0][0]+'_entry_select'].get()
    dico_interface_param[axe_lim_tab[1][0]] = dico_interface_tk[axe_lim_tab[1][0]+'_entry_select'].get()

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as fp:
        json.dump(dico_interface_param, fp)

    if plot == 'yes':
        if parent == 'frame_select_data_model_param_plot':
            from ForMoSA.interface.tabcontrol.path_selection.plot import plot_obs_vs_model
            plot_obs_vs_model(dico_interface_tk,
                              dico_interface_tk['frame_select_data_model_param_plot'].winfo_width() + 25, 210)
        elif parent == 'frame_adapt_grid_data_param_plot':
            from ForMoSA.interface.tabcontrol.adapt_grid_data.plot import plot_adapt
            plot_adapt(dico_interface_tk,
                       dico_interface_tk['frame_param_adapt_grid_data'].winfo_width() + 25, 0)
        elif parent == 'frame_fit_param_plot':
            from ForMoSA.interface.tabcontrol.fit.plot import plot_fit
            plot_fit(dico_interface_tk, dico_interface_tk['frame_param_fit'].winfo_width() + 25, 0)
        else:
            pass


def calc_band(dico_interface_tk, band):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    with fits.open(dico_interface_tk['observation_path_entry_select'].get()) as hdul:
        wav_obs = hdul[1].data['WAV']
    band_for_reso = dico_interface_param[band].split('/')

    adapt_band_min_tab = []
    adapt_band_max_tab = []
    for w_ind, w in enumerate(band_for_reso):
        w_min = float(w.split(',')[0])
        w_max = float(w.split(',')[1])
        ind_ran = np.where((w_min <= wav_obs) & (wav_obs <= w_max))
        w_ran = wav_obs[ind_ran]
        if len(w_ran) != 0:
            adapt_band_min_tab.append(min(w_ran))
            adapt_band_max_tab.append(max(w_ran))

    dico_interface_param[band+'_tab'] = [adapt_band_min_tab, adapt_band_max_tab]

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as fp:
        json.dump(dico_interface_param, fp)


def calc_continuum(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    from ForMoSA.interface.utilities.globfile_interface import GlobFile
    global_params = GlobFile(dico_interface_tk['config_file_path'])
    global_params.observation_path = dico_interface_param['observation_path']
    global_params.wav_for_adapt = dico_interface_param['wav_for_adapt']
    global_params.wav_for_continuum = dico_interface_param['wav_for_continuum']
    global_params.continuum_sub = dico_interface_param['continuum_sub']

    from ForMoSA.adapt.extraction_functions import adapt_observation_range
    obs_cut, obs_pho, obs_cut_ins, obs_pho_ins = adapt_observation_range(global_params)
    wav_continuum = []
    flx_continuum = []
    from ForMoSA.adapt.extraction_functions import continuum_estimate
    for c, cut in enumerate(obs_cut):
        continuum = continuum_estimate(global_params, cut[0], cut[0], cut[1], np.array(cut[3], dtype=float), 'obs')
        wav_continuum.append(cut[0].tolist())
        flx_continuum.append(continuum.tolist())
    dico_interface_param['continuum_sub_wav'] = wav_continuum
    dico_interface_param['continuum_sub_flx'] = flx_continuum

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as fp:
        json.dump(dico_interface_param, fp)


def replot_all(dico_interface_tk):

    from ForMoSA.interface.tabcontrol.path_selection.plot import plot_obs_vs_model
    plot_obs_vs_model(dico_interface_tk,
                      dico_interface_tk['frame_select_data_model_param_plot'].winfo_width() + 25, 210)
    from ForMoSA.interface.tabcontrol.adapt_grid_data.plot import plot_adapt
    plot_adapt(dico_interface_tk,
               dico_interface_tk['frame_param_adapt_grid_data'].winfo_width() + 25, 0)
    from ForMoSA.interface.tabcontrol.fit.plot import plot_fit
    plot_fit(dico_interface_tk, dico_interface_tk['frame_param_fit'].winfo_width() + 25, 0)
