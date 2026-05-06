from tkinter import ttk
import json
import tkinter as tk
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
import numpy as np

from ForMoSA.interface.tabcontrol.utilities import isfloat
from ForMoSA.interface.tabcontrol.grid_priors.prior_function import uniform_prior, gaussian_prior


def define_extra_grid_prior_box(dico_interface_tk):

    frame_master = tk.Frame(dico_interface_tk['frame_extra_grid_priors'])
    dico_interface_tk['extra_grid_prior_box'] = frame_master
    frame_master.pack()
    frame_title = tk.Frame(frame_master)
    frame_title.grid(row=0, column=0, sticky=W)
    dico_interface_tk['extra_grid_prior_title'] = frame_title
    label_title = tk.Label(frame_title, text="Definition of the priors for the extra grid parameters",
                           font='Arial 13 bold')
    label_title.grid(row=0, column=0, sticky=W)

    row_gp = 1
    col_gp = 0
    for par_e_ind, par_e in enumerate([['r', 'R', '(RJup)'],
                                       ['d', 'd', '(pc)'],
                                       ['rv', 'RV', '(km/s)'],
                                       ['vsini', 'v.sin(i)', '(km/s)'],
                                       ['ld', 'limb darkening', ''],
                                       ['av', 'Av', '(mag)']
                                       ]):
        title = par_e[1]
        unit = par_e[2]
        define_extra_prior_parameter(dico_interface_tk, par_e[0], title, unit, row_gp, col_gp)
        prior_fct_test_extra(dico_interface_tk, par_e[0], title, unit)
        col_gp += 1
        if col_gp == 3:
            col_gp = 0
            row_gp += 1


def define_extra_prior_parameter(dico_interface_tk, key, title, unit, row_gp, col_gp):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    frame_prior_parameter = tk.Frame(dico_interface_tk['extra_grid_prior_box'])
    dico_interface_tk[key + '_frame'] = frame_prior_parameter
    frame_prior_parameter.grid(row=row_gp, column=col_gp, padx=5, pady=5, sticky=E)

    frame_console = tk.Frame(frame_prior_parameter)
    frame_console.grid(row=0, column=0, sticky=E)

    frame_param_title = tk.Frame(frame_console)
    dico_interface_tk[key + '_title'] = frame_param_title
    frame_param_title.grid(row=0, column=0, sticky=E)
    param_title = tk.Label(frame_param_title, text=title + ' ' + unit + ' ['+key+']: ', font='Arial 13 bold')
    param_title.grid(row=0, column=0, sticky=E)

    frame_select = tk.Frame(frame_console)
    frame_select.grid(row=1, column=0, sticky=E)

    prior_fct = ['NA', 'constant', 'uniform', 'gaussian', '/!/custom 1/!/', '/!/custom 2/!/', '/!/custom 3/!/']
    choice_prior_fct = ttk.Combobox(frame_select, state='readonly', values=prior_fct, width=7)
    dico_interface_tk[key + '_choice_prior_fct'] = choice_prior_fct
    choice_prior_fct.grid(row=0, column=0, sticky=E)

    entry_prior_fct_arg = tk.Entry(frame_select, width=9)
    dico_interface_tk[key + '_entry_prior_fct_arg'] = entry_prior_fct_arg
    entry_prior_fct_arg.grid(row=1, column=0, sticky=E)

    if dico_interface_param[key] != 'NA':
        dico_interface_tk[key + '_choice_prior_fct'].set(dico_interface_param[key][0])
        prior_fct_arg_tab = dico_interface_param[key][1:]
        prior_fct_arg = ''
        for arg_ind, arg in enumerate(prior_fct_arg_tab):
            print(prior_fct_arg)
            prior_fct_arg += arg + ','
        dico_interface_tk[key + '_entry_prior_fct_arg'].delete(0, tk.END)
        dico_interface_tk[key + '_entry_prior_fct_arg'].insert(0, prior_fct_arg[:-1])
    else:
        dico_interface_tk[key + '_choice_prior_fct'].set('NA')
        dico_interface_tk[key + '_entry_prior_fct_arg'].delete(0, tk.END)
        dico_interface_tk[key + '_entry_prior_fct_arg'].insert(0, 'NA')

    button_prior_fct_test = tk.Button(frame_select, text='init.',
                                      command=lambda: extra_prior_default(dico_interface_tk, key, title, unit))
    button_prior_fct_test.grid(row=0, column=1, sticky=E)
    button_prior_fct_test = tk.Button(frame_select, text='plot',
                                      command=lambda: prior_fct_test_extra(dico_interface_tk, key, title, unit))
    button_prior_fct_test.grid(row=1, column=1, sticky=E)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param, dico)


def extra_prior_default(dico_interface_tk, key, title, unit):

    dico_interface_tk[key + '_choice_prior_fct'].set('NA')
    dico_interface_tk[key + '_entry_prior_fct_arg'].delete(0, tk.END)
    dico_interface_tk[key + '_entry_prior_fct_arg'].insert(0, 'NA')
    prior_fct_test_extra(dico_interface_tk, key, title, unit)


def prior_fct_test_extra(dico_interface_tk, key, title, unit):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    if key in ['r', 'd']:
        bg_color = 'light blue'
    elif key in ['vsini', 'ld']:
        bg_color = 'wheat1'
    else:
        bg_color = None
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
    if prior_fct == 'constant':
        if len(prior_fct_arg) == 1:
            if isfloat(prior_fct_arg[0]) is True:
                constant_prior = float(prior_fct_arg[0])
                test = 1
                if key in ['r', 'd', 'vsini'] and constant_prior <= 0:
                    test *= 0
                elif key in ['ld'] and constant_prior <= 0 or key in ['ld'] and constant_prior >= 1:
                    test *= 0
                if test == 1:
                    dico_interface_tk['ax_' + key].axvline(x=constant_prior, ymin=0, ymax=1, color='royalblue')
                    dico_interface_tk['ax_' + key].set_xlim(0.5 * constant_prior, 1.5 * constant_prior)
                    dico_interface_param[key] = [prior_fct, prior_fct_arg]
                    param_title = tk.Label(dico_interface_tk[key + '_title'],
                                           text=title + ' ' + unit + ' ['+key+']: ',
                                           font='Arial 13 bold', fg='green', bg=bg_color)
                    param_title.grid(row=0, column=0, sticky=E)

                else:
                    param_title = tk.Label(dico_interface_tk[key + '_title'],
                                           text=title + ' ' + unit + ' ['+key+']: ',
                                           font='Arial 13 bold', fg='red', bg=bg_color)
                    param_title.grid(row=0, column=0, sticky=E)

            else:
                if prior_fct_arg[0] in ['NA', '']:
                    dico_interface_param[key] = 'NA'
                    dico_interface_tk[key + '_entry_prior_fct_arg'].delete(0, tk.END)
                    dico_interface_tk[key + '_entry_prior_fct_arg'].insert(0, 'NA')
                    param_title = tk.Label(dico_interface_tk[key + '_title'],
                                           text=title + ' ' + unit + ' ['+key+']: ',
                                           font='Arial 13 bold', fg='green', bg=bg_color)
                    param_title.grid(row=0, column=0, sticky=E)
                else:
                    param_title = tk.Label(dico_interface_tk[key + '_title'],
                                           text=title + ' ' + unit + ' ['+key+']: ',
                                           font='Arial 13 bold', fg='red', bg=bg_color)
                    param_title.grid(row=0, column=0, sticky=E)
        else:
            if prior_fct_arg[0] in ['NA', '']:
                dico_interface_param[key] = 'NA'
                dico_interface_tk[key + '_entry_prior_fct_arg'].delete(0, tk.END)
                dico_interface_tk[key + '_entry_prior_fct_arg'].insert(0, 'NA')
                param_title = tk.Label(dico_interface_tk[key + '_title'],
                                       text=title + ' ' + unit + ' ['+key+']: ',
                                       font='Arial 13 bold', fg='green', bg=bg_color)
                param_title.grid(row=0, column=0, sticky=E)
            else:
                param_title = tk.Label(dico_interface_tk[key + '_title'],
                                       text=title + ' ' + unit + ' ['+key+']: ',
                                       font='Arial 13 bold', fg='red', bg=bg_color)
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
                test = 1
                if key in ['r', 'd', 'vsini'] and max_prior <= 0 or min_prior >= max_prior:
                    test *= 0
                elif key in ['ld'] and max_prior <= 0 or key in ['ld'] and min_prior >= 1 or min_prior >= max_prior:
                    test *= 0

                if test == 1:
                    theta = np.random.uniform(low=0., high=1.0, size=(1000000,))
                    if prior_fct == 'uniform':
                        test_val_tab = uniform_prior(prior_fct_arg, theta)
                    elif prior_fct == 'gaussian':
                        test_val_tab = gaussian_prior(prior_fct_arg, theta)
                    dico_interface_tk['ax_' + key].hist(test_val_tab, range=(min_prior, max_prior), bins=100,
                                                        histtype='step', color='royalblue')
                    if key in ['r', 'd', 'vsini']:
                        dico_interface_tk['ax_' + key].axvline(x=0, ymin=0, ymax=1, color='r')
                        dico_interface_tk['ax_' + key].axvline(x=0 - 0.05 * max_prior, ymin=0, ymax=1, color='r',
                                                               alpha=0.66)
                        dico_interface_tk['ax_' + key].axvline(x=0 - 0.1 * max_prior, ymin=0, ymax=1, color='r',
                                                               alpha=0.33)
                    elif key == 'ld':
                        dico_interface_tk['ax_' + key].axvline(x=0, ymin=0, ymax=1, color='r')
                        dico_interface_tk['ax_' + key].axvline(x=1, ymin=0, ymax=1, color='r')
                        dico_interface_tk['ax_' + key].axvline(x=0 - 0.05, ymin=0, ymax=1, color='r', alpha=0.66)
                        dico_interface_tk['ax_' + key].axvline(x=1 + 0.05, ymin=0, ymax=1, color='r', alpha=0.66)
                        dico_interface_tk['ax_' + key].axvline(x=0 - 0.1, ymin=0, ymax=1, color='r', alpha=0.33)
                        dico_interface_tk['ax_' + key].axvline(x=1 + 0.1, ymin=0, ymax=1, color='r', alpha=0.33)
                    else:
                        pass
                    dico_interface_tk['ax_' + key].set_xlim(min_prior - 0.1 * (max_prior - min_prior),
                                                            max_prior + 0.1 * (max_prior - min_prior))

                    dico_interface_param[key] = [prior_fct, prior_fct_arg]
                    param_title = tk.Label(dico_interface_tk[key + '_title'],
                                           text=title + ' ' + unit + ' ['+key+']: ',
                                           font='Arial 13 bold', fg='green', bg=bg_color)
                    param_title.grid(row=0, column=0, sticky=E)
                else:
                    param_title = tk.Label(dico_interface_tk[key + '_title'],
                                           text=title + ' ' + unit + ' ['+key+']: ',
                                           font='Arial 13 bold', fg='red', bg=bg_color)
                    param_title.grid(row=0, column=0, sticky=E)
            else:
                if prior_fct_arg[0] in ['NA']:
                    dico_interface_param[key] = 'NA'
                    dico_interface_tk[key + '_entry_prior_fct_arg'].delete(0, tk.END)
                    dico_interface_tk[key + '_entry_prior_fct_arg'].insert(0, 'NA')
                    param_title = tk.Label(dico_interface_tk[key + '_title'],
                                           text=title + ' ' + unit + ' ['+key+']: ',
                                           font='Arial 13 bold', fg='green', bg=bg_color)
                    param_title.grid(row=0, column=0, sticky=E)
                else:
                    param_title = tk.Label(dico_interface_tk[key + '_title'],
                                           text=title + ' ' + unit + ' ['+key+']: ',
                                           font='Arial 13 bold', fg='red', bg=bg_color)
                    param_title.grid(row=0, column=0, sticky=E)
        else:
            if prior_fct_arg[0] in ['']:
                dico_interface_param[key] = 'NA'
                dico_interface_tk[key + '_entry_prior_fct_arg'].delete(0, tk.END)
                dico_interface_tk[key + '_entry_prior_fct_arg'].insert(0, 'NA')
                param_title = tk.Label(dico_interface_tk[key + '_title'],
                                       text=title + ' ' + unit + ' ['+key+']: ',
                                       font='Arial 13 bold', fg='green', bg=bg_color)
                param_title.grid(row=0, column=0, sticky=E)
            else:
                param_title = tk.Label(dico_interface_tk[key + '_title'],
                                       text=title + ' ' + unit + ' ['+key+']: ',
                                       font='Arial 13 bold', fg='red', bg=bg_color)
                param_title.grid(row=0, column=0, sticky=E)

    if prior_fct_arg[0] == 'NA' or prior_fct == 'NA':
        dico_interface_param[key] = 'NA'
        dico_interface_tk[key + '_choice_prior_fct'].set('NA')
        dico_interface_tk[key + '_entry_prior_fct_arg'].delete(0, tk.END)
        dico_interface_tk[key + '_entry_prior_fct_arg'].insert(0, 'NA')
        param_title = tk.Label(dico_interface_tk[key + '_title'],
                               text=title + ' ' + unit + ' [' + key + ']: ',
                               font='Arial 13 bold', fg='gray', bg=bg_color)
        param_title.grid(row=0, column=0, sticky=E)

    # Test doublets
    if dico_interface_param['r'] == 'NA':
        if dico_interface_param['d'] != 'NA':
            param_title = tk.Label(dico_interface_tk['r_title'],
                                   text='R (RJup) [r]: ',
                                   font='Arial 13 bold', fg='red', bg=bg_color)
            param_title.grid(row=0, column=0, sticky=E)
    if dico_interface_param['d'] == 'NA':
        if dico_interface_param['r'] != 'NA':
            param_title = tk.Label(dico_interface_tk['d_title'],
                                   text='d (pc) [d]: ',
                                   font='Arial 13 bold', fg='red', bg=bg_color)
            param_title.grid(row=0, column=0, sticky=E)
    if dico_interface_param['vsini'] == 'NA':
        if dico_interface_param['ld'] != 'NA':
            param_title = tk.Label(dico_interface_tk['vsini_title'],
                                   text='v.sin(i) (km/s) [vsini]: ',
                                   font='Arial 13 bold', fg='red', bg=bg_color)
            param_title.grid(row=0, column=0, sticky=E)
    if dico_interface_param['ld'] == 'NA':
        if dico_interface_param['vsini'] != 'NA':
            param_title = tk.Label(dico_interface_tk['ld_title'],
                                   text='limb darkening [ld]: ',
                                   font='Arial 13 bold', fg='red', bg=bg_color)
            param_title.grid(row=0, column=0, sticky=E)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param, dico)


def validate_extra_grid_prior(dico_interface_tk):

    for par_e_ind, par_e in enumerate([['r', 'R', '(RJup)'],
                                       ['d', 'd', '(pc)'],
                                       ['rv', 'RV', '(km/s)'],
                                       ['vsini', 'v.sin(i)', '(km/s)'],
                                       ['ld', 'limb darkening', ' '],
                                       ['av', 'Av', '(mag)']
                                       ]):
        prior_fct_test_extra(dico_interface_tk, par_e[0], par_e[1], par_e[2])
