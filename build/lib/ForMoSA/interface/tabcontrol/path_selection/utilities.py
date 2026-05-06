import json
import tkinter as tk
from tkinter import *
import xarray as xr
import os
from tkinter import filedialog
from astropy.io import fits
import numpy as np


def button_observation_path(dico_interface_tk, choose_mode='choose', plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    if choose_mode == 'choose':
        dico_interface_tk['observation_path_entry_select'].delete(0, tk.END)
        dico_interface_tk['observation_path_entry_select'].insert(0, filedialog.askopenfilename())
    try:
        with fits.open(dico_interface_tk['observation_path_entry_select'].get()) as hdul:
            wav_obs = hdul[1].data['WAV']
            flx_obs = hdul[1].data['FLX']
            err_obs = hdul[1].data['ERR']
            res_obs = hdul[1].data['RES']
            if isinstance(wav_obs, np.ndarray) and \
                    isinstance(flx_obs, np.ndarray) and \
                    isinstance(err_obs, np.ndarray) and \
                    isinstance(res_obs, np.ndarray):
                dico_interface_param['observation_path'] = dico_interface_tk['observation_path_entry_select'].get()
                label_title = tk.Label(dico_interface_tk['observation_path_title'],
                                       text="Observation file's path [observation_path]:",
                                       font='Arial 13 bold', fg='green')
                label_title.grid(row=0, column=0, sticky=W)
            else:
                label_title = tk.Label(dico_interface_tk['observation_path_title'],
                                       text="Observation file's path [observation_path]:",
                                       font='Arial 13 bold', fg='red')
                label_title.grid(row=0, column=0, sticky=W)
    except:
        label_title = tk.Label(dico_interface_tk['observation_path_title'],
                               text="Observation file's path [observation_path]:",
                               font='Arial 13 bold', fg='red')
        label_title.grid(row=0, column=0, sticky=W)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param, dico)

    if plot == 'yes':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)


def button_adapt_store_path(dico_interface_tk, choose_mode='choose', plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    if choose_mode == 'choose':
        dico_interface_tk['adapt_store_path_entry_select'].delete(0, tk.END)
        dico_interface_tk['adapt_store_path_entry_select'].insert(0, filedialog.askopenfilename())

    if os.path.isdir(dico_interface_tk['adapt_store_path_entry_select'].get()):
        if dico_interface_tk['adapt_store_path_entry_select'].get()[-1] == '/':
            dico_interface_param['adapt_store_path'] = dico_interface_tk['adapt_store_path_entry_select'].get()
        else:
            dico_interface_param['adapt_store_path'] = dico_interface_tk['adapt_store_path_entry_select'].get() + '/'
        label_title = tk.Label(dico_interface_tk['adapt_store_path_title'],
                               text="Path to store the adapted model grid [adapt_store_path]:",
                               font='Arial 13 bold', fg='green')
        label_title.grid(row=0, column=0, sticky=W)
    else:
        label_title = tk.Label(dico_interface_tk['adapt_store_path_title'],
                               text="Path to store the adapted model grid [adapt_store_path]:",
                               font='Arial 13 bold', fg='red')
        label_title.grid(row=0, column=0, sticky=W)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param, dico)

    if plot == 'yes':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)


def button_result_path(dico_interface_tk, choose_mode='choose', plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    if choose_mode == 'choose':
        dico_interface_tk['result_path_entry_select'].delete(0, tk.END)
        dico_interface_tk['result_path_entry_select'].insert(0, filedialog.askopenfilename())

    if os.path.isdir(dico_interface_tk['result_path_entry_select'].get()):
        if dico_interface_tk['result_path_entry_select'].get()[-1] == '/':
            dico_interface_param['result_path'] = dico_interface_tk['result_path_entry_select'].get()
        else:
            dico_interface_param['result_path'] = dico_interface_tk['result_path_entry_select'].get() + '/'
        label_title = tk.Label(dico_interface_tk['result_path_title'],
                               text="Path to store the results [result_path]:",
                               font='Arial 13 bold', fg='green')
        label_title.grid(row=0, column=0, sticky=W)
    else:
        label_title = tk.Label(dico_interface_tk['result_path_title'],
                               text="Path to store the results [result_path]:",
                               font='Arial 13 bold', fg='red')
        label_title.grid(row=0, column=0, sticky=W)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param, dico)

    if plot == 'yes':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)


def button_model_path(dico_interface_tk, choose_mode='choose', plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    if choose_mode == 'choose':
        dico_interface_tk['model_path_entry_select'].delete(0, tk.END)
        dico_interface_tk['model_path_entry_select'].insert(0, filedialog.askopenfilename())

    try:
        ds = xr.open_dataset(dico_interface_tk['model_path_entry_select'].get(), decode_cf=False, engine='netcdf4')
        wav_mod = ds['wavelength'].values
        ds.close()
        dwav_mod = wav_mod[1:] - wav_mod[:-1]
        dwav_mod = np.concatenate((dwav_mod, dwav_mod[-2:-1]))
        res_mod = wav_mod / dwav_mod
        res_mod[-1] = res_mod[-2]
        dico_interface_param['model_path'] = dico_interface_tk['model_path_entry_select'].get()
        model_name = dico_interface_tk['model_path_entry_select'].get().split('/')
        model_name = model_name[-1]
        model_name = model_name.split('.nc')
        model_name = model_name[0]
        dico_interface_param['model_name'] = model_name
        if isinstance(wav_mod, np.ndarray) and isinstance(res_mod, np.ndarray):
            label_title = tk.Label(dico_interface_tk['model_path_title'],
                                   text="Model grid's path [model_path]:",
                                   font='Arial 13 bold', fg='green')
            label_title.grid(row=0, column=0, sticky=W)
        else:
            label_title = tk.Label(dico_interface_tk['model_path_title'],
                                   text="Model grid's path [model_path]:",
                                   font='Arial 13 bold', fg='red')
            label_title.grid(row=0, column=0, sticky=W)
    except:
        label_title = tk.Label(dico_interface_tk['model_path_title'],
                               text="Model grid's path [model_path]:",
                               font='Arial 13 bold', fg='red')
        label_title.grid(row=0, column=0, sticky=W)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param, dico)

    if plot == 'yes':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)


def validate_path_selection(dico_interface_tk, ini='no'):

    button_observation_path(dico_interface_tk, choose_mode='check', plot='no')
    button_adapt_store_path(dico_interface_tk, choose_mode='check', plot='no')
    button_result_path(dico_interface_tk, choose_mode='check', plot='no')
    button_model_path(dico_interface_tk, choose_mode='check', plot='no')

    if ini == 'no':
        dico_interface_tk['root_tabcontrol'].destroy()
        from ForMoSA.interface.tabcontrol.interface import tabcontrol_interface
        tabcontrol_interface(dico_interface_tk)
