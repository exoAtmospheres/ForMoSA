import os.path
from tkinter import ttk
import json
import xarray as xr
import tkinter as tk
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
import numpy as np

from ForMoSA.interface.tabcontrol.utilities import isfloat
from ForMoSA.interface.tabcontrol.grid_priors.prior_function import uniform_prior, gaussian_prior


def define_grid_prior_box(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    frame_master = tk.Frame(dico_interface_tk['frame_grid_priors'])
    dico_interface_tk['grid_prior_box'] = frame_master
    frame_master.pack()
    frame_title = tk.Frame(frame_master)
    frame_title.grid(row=0, column=0, sticky=W)
    dico_interface_tk['grid_prior_title'] = frame_title
    label_title = tk.Label(frame_title, text="Definition of the priors for the grid's parameters", font='Arial 13 bold')
    label_title.grid(row=0, column=0, sticky=W)

    if os.path.exists(dico_interface_param['model_path']):
        ds = xr.open_dataset(dico_interface_param['model_path'], decode_cf=False, engine='netcdf4')
        attr = ds.attrs
        row_gp = 1
        col_gp = 0
        for key_p_ind, key_p in enumerate(attr['key']):
            define_grid_prior_parameter(dico_interface_tk, key_p, key_p_ind, attr['title'][key_p_ind],
                                        attr['unit'][key_p_ind], row_gp, col_gp)
            prior_fct_test_grid(dico_interface_tk, key_p, attr['title'][key_p_ind], attr['unit'][key_p_ind], key_p_ind)
            col_gp += 1
            if col_gp == 3:
                col_gp = 0
                row_gp += 1


def define_grid_prior_parameter(dico_interface_tk, key, key_p_ind, title, unit, row_gp, col_gp):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    frame_prior_parameter = tk.Frame(dico_interface_tk['grid_prior_box'])
    dico_interface_tk[key + '_frame'] = frame_prior_parameter
    frame_prior_parameter.grid(row=row_gp, column=col_gp, padx=5, pady=5, sticky=W)

    frame_console = tk.Frame(frame_prior_parameter)
    frame_console.grid(row=0, column=0, sticky=E)

    frame_param_title = tk.Frame(frame_console)
    dico_interface_tk[key + '_title'] = frame_param_title
    frame_param_title.grid(row=0, column=0, sticky=E)
    param_title = tk.Label(frame_param_title, text=title + ' ' + unit + ' [par'+str(key_p_ind+1)+']: ',
                           font='Arial 13 bold')
    param_title.grid(row=0, column=0, sticky=E)

    frame_select = tk.Frame(frame_console)
    frame_select.grid(row=1, column=0, sticky=E)

    prior_fct = ['constant', 'uniform', 'gaussian', '/!/custom 1/!/', '/!/custom 2/!/', '/!/custom 3/!/']
    choice_prior_fct = ttk.Combobox(frame_select, state='readonly', values=prior_fct, width=7)
    dico_interface_tk[key + '_choice_prior_fct'] = choice_prior_fct
    choice_prior_fct.grid(row=0, column=0, sticky=W)

    entry_prior_fct_arg = tk.Entry(frame_select, width=9)
    dico_interface_tk[key + '_entry_prior_fct_arg'] = entry_prior_fct_arg
    entry_prior_fct_arg.grid(row=1, column=0, sticky=W)

    prior_fct_arg_tab = dico_interface_param[key][1:]
    if dico_interface_param[key] != 'NA':
        dico_interface_tk[key + '_choice_prior_fct'].set(dico_interface_param[key][0])
        prior_fct_arg = ''
        for arg_ind, arg in enumerate(prior_fct_arg_tab):
            prior_fct_arg += arg + ','
        dico_interface_tk[key + '_entry_prior_fct_arg'].delete(0, tk.END)
        dico_interface_tk[key + '_entry_prior_fct_arg'].insert(0, prior_fct_arg[:-1])
    else:
        grid_prior_default(dico_interface_tk, key, title, unit, key_p_ind)

    button_prior_fct_test = tk.Button(frame_select, text='init.',
                                      command=lambda: grid_prior_default(dico_interface_tk, key, title, unit,
                                                                         key_p_ind))
    button_prior_fct_test.grid(row=0, column=1, sticky=W)

    button_prior_fct_test = tk.Button(frame_select, text='plot',
                                      command=lambda: prior_fct_test_grid(dico_interface_tk, key, title, unit,
                                                                          key_p_ind))
    button_prior_fct_test.grid(row=1, column=1, sticky=W)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param, dico)


def grid_prior_default(dico_interface_tk, key, title, unit, key_p_ind):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    dico_interface_tk[key + '_choice_prior_fct'].set('uniform')
    ds = xr.open_dataset(dico_interface_param['model_path'], decode_cf=False, engine='netcdf4')
    range_param = ds[key].values
    dico_interface_tk[key + '_entry_prior_fct_arg'].delete(0, tk.END)
    dico_interface_tk[key + '_entry_prior_fct_arg'].insert(0, str(min(range_param)) + ', ' + str(max(range_param)))
    prior_fct_test_grid(dico_interface_tk, key, title, unit, key_p_ind)


def prior_fct_test_grid(dico_interface_tk, key, title, unit, key_p_ind):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    frame_plot_prior = tk.Frame(dico_interface_tk[key + '_frame'])
    frame_plot_prior.grid(row=0, column=1, padx=5, pady=5, sticky=E)
    figure = Figure(figsize=(2.1, 1.43), dpi=100)
    ax_priors = figure.add_axes([0.1, 0.35, 0.8, 0.57])
    ax_priors.set_yticks([], [])
    dico_interface_tk['ax_' + key] = ax_priors
    ax_priors.set_xlabel(title + ' ' + unit)
    bar = FigureCanvasTkAgg(figure, frame_plot_prior)
    bar.get_tk_widget().pack(expand=YES)

    prior_fct = dico_interface_tk[key + '_choice_prior_fct'].get()
    prior_fct_arg = dico_interface_tk[key + '_entry_prior_fct_arg'].get()
    prior_fct_arg = prior_fct_arg.split(',')
    ds = xr.open_dataset(dico_interface_param['model_path'], decode_cf=False, engine='netcdf4')
    param_tab = ds[key].values
    if prior_fct == 'constant':
        if len(prior_fct_arg) == 1:
            if isfloat(prior_fct_arg[0]) is True:
                constant_prior = float(prior_fct_arg[0])
                if min(param_tab) < constant_prior < max(param_tab):
                    dico_interface_tk['ax_' + key].axvline(x=constant_prior, ymin=0, ymax=1, color='royalblue')
                    dico_interface_tk['ax_' + key].set_xlim(0.5 * constant_prior, 1.5 * constant_prior)
                    dico_interface_param[key] = [prior_fct, prior_fct_arg]
                    param_title = tk.Label(dico_interface_tk[key + '_title'],
                                           text=title + ' ' + unit + ' [par' + str(key_p_ind + 1) + ']: ',
                                           font='Arial 13 bold', fg='green')
                    param_title.grid(row=0, column=0, sticky=E)
                else:
                    param_title = tk.Label(dico_interface_tk[key + '_title'],
                                           text=title + ' ' + unit + ' [par' + str(key_p_ind + 1) + ']: ',
                                           font='Arial 13 bold', fg='red')
                    param_title.grid(row=0, column=0, sticky=E)
            else:
                param_title = tk.Label(dico_interface_tk[key + '_title'],
                                       text=title + ' ' + unit + ' [par' + str(key_p_ind + 1) + ']: ',
                                       font='Arial 13 bold', fg='red')
                param_title.grid(row=0, column=0, sticky=E)
        else:
            param_title = tk.Label(dico_interface_tk[key + '_title'],
                                   text=title + ' ' + unit + ' [par' + str(key_p_ind + 1) + ']: ',
                                   font='Arial 13 bold', fg='red')
            param_title.grid(row=0, column=0, sticky=E)

    elif prior_fct in ['uniform', 'gaussian']:
        if len(prior_fct_arg) == 2:
            if isfloat(prior_fct_arg[0]) is True and isfloat(prior_fct_arg[1]) is True:
                if prior_fct == 'uniform':
                    min_prior = float(prior_fct_arg[0])
                    max_prior = float(prior_fct_arg[1])
                elif prior_fct == 'gaussian':
                    mu = float(prior_fct_arg[0])
                    sigma = float(prior_fct_arg[1])
                    min_prior = mu - 5 * sigma
                    max_prior = mu + 5 * sigma
                if min_prior < max(param_tab) and max_prior > min(param_tab) and min_prior < max_prior:
                    theta = np.random.uniform(low=0., high=1.0, size=(1000000,))
                    if prior_fct == 'uniform':
                        test_val_tab = uniform_prior(prior_fct_arg,  theta)
                    elif prior_fct == 'gaussian':
                        test_val_tab = gaussian_prior(prior_fct_arg,  theta)
                    dico_interface_tk['ax_' + key].hist(test_val_tab, range=(min_prior, max_prior), bins=100,
                                                        histtype='step', color='royalblue')
                    dico_interface_param[key + '_prior_fct'] = prior_fct
                    dico_interface_param[key + '_prior_fct_arg'] = prior_fct_arg
                    dico_interface_tk['ax_' + key].axvline(x=min(param_tab), ymin=0, ymax=1, color='r')
                    dico_interface_tk['ax_' + key].axvline(x=max(param_tab), ymin=0, ymax=1, color='r')
                    dico_interface_tk['ax_' + key].axvline(x=min(param_tab) - 0.05*(max_prior-min_prior),
                                                           ymin=0, ymax=1, color='r', alpha=0.66)
                    dico_interface_tk['ax_' + key].axvline(x=max(param_tab) + 0.05*(max_prior-min_prior),
                                                           ymin=0, ymax=1, color='r', alpha=0.66)
                    dico_interface_tk['ax_' + key].axvline(x=min(param_tab) - 0.1*(max_prior-min_prior),
                                                           ymin=0, ymax=1, color='r', alpha=0.33)
                    dico_interface_tk['ax_' + key].axvline(x=max(param_tab) + 0.1*(max_prior-min_prior),
                                                           ymin=0, ymax=1, color='r', alpha=0.33)

                    dico_interface_tk['ax_' + key].set_xlim(min_prior - 0.1 * (max_prior - min_prior),
                                                            max_prior + 0.1 * (max_prior - min_prior))

                    dico_interface_param[key] = [prior_fct, prior_fct_arg]
                    param_title = tk.Label(dico_interface_tk[key + '_title'],
                                           text=title + ' ' + unit + ' [par' + str(key_p_ind + 1) + ']: ',
                                           font='Arial 13 bold', fg='green')
                    param_title.grid(row=0, column=0, sticky=E)
                else:
                    param_title = tk.Label(dico_interface_tk[key + '_title'],
                                           text=title + ' ' + unit + ' [par' + str(key_p_ind + 1) + ']: ',
                                           font='Arial 13 bold', fg='red')
                    param_title.grid(row=0, column=0, sticky=E)
            else:
                param_title = tk.Label(dico_interface_tk[key + '_title'],
                                       text=title + ' ' + unit + ' [par' + str(key_p_ind + 1) + ']: ',
                                       font='Arial 13 bold', fg='red')
                param_title.grid(row=0, column=0, sticky=E)
        else:
            param_title = tk.Label(dico_interface_tk[key + '_title'],
                                   text=title + ' ' + unit + ' [par' + str(key_p_ind + 1) + ']: ',
                                   font='Arial 13 bold', fg='red')
            param_title.grid(row=0, column=0, sticky=E)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param, dico)


def validate_grid_prior(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    ds = xr.open_dataset(dico_interface_param['model_path'], decode_cf=False, engine='netcdf4')
    attr = ds.attrs
    for key_p_ind, key_p in enumerate(attr['key']):
        prior_fct_test_grid(dico_interface_tk, key_p, attr['title'][key_p_ind], attr['unit'][key_p_ind], key_p_ind)
