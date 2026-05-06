import json
import tkinter as tk
from tkinter import *

from ForMoSA.interface.tabcontrol.utilities import isfloat


def switch_ns_algo(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    frame_ns_algo = tk.Frame(dico_interface_tk['frame_param_fit'])
    frame_ns_algo.grid(row=0, column=0, sticky=W)
    frame_ns_algo_title = tk.Frame(frame_ns_algo)
    frame_ns_algo_title.grid(row=0, column=0, sticky=W)
    dico_interface_tk['frame_ns_algo_title'] = frame_ns_algo_title
    label_ns_algo_title = tk.Label(frame_ns_algo_title, text='Nested sampling algorithm [ns_algo]:',
                                   font='Arial 13 bold')
    label_ns_algo_title.grid(row=0, column=0, sticky=W)
    frame_ns_algo_switch = tk.Frame(frame_ns_algo)
    dico_interface_tk['frame_ns_algo_switch'] = frame_ns_algo_switch
    frame_ns_algo_switch.grid(row=1, column=0, sticky=W)

    dico_interface_tk['ns_algo_choice'] = tk.StringVar(dico_interface_tk['frame_ns_algo_switch'],
                                                       dico_interface_param['ns_algo'])

    algo_tab_title = ['nestle', 'dynesty', 'ultranest']
    algo_tab_key = ['nestle', 'dynesty', 'ultranest']

    for key_ind, key_a in enumerate(algo_tab_title):
        radio_but = tk.Radiobutton(frame_ns_algo_switch, text=algo_tab_title[key_ind], value=algo_tab_key[key_ind],
                                   variable=dico_interface_tk['ns_algo_choice'],
                                   command=lambda: validate_switch_ns_algo(dico_interface_tk))
        radio_but.grid(row=0, column=key_ind)
        button_algo_par = tk.Button(frame_ns_algo_switch, text='/!/Param/!/',
                                    command=lambda: algo_parameters())
        button_algo_par.grid(row=1, column=key_ind)


def validate_switch_ns_algo(dico_interface_tk, plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    dico_interface_param['ns_algo'] = dico_interface_tk['ns_algo_choice'].get()

    label_ns_algo_title = tk.Label(dico_interface_tk['frame_ns_algo_title'],
                                   text='Nested sampling algorithm [ns_algo]:',
                                   font='Arial 13 bold', fg='green')
    label_ns_algo_title.grid(row=0, column=0, sticky=W)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param,  dico)

    if plot == 'yes':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)


def algo_parameters():
    pass


def define_wav_fit(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    fram_wav_fit = tk.Frame(dico_interface_tk['frame_param_fit'])
    dico_interface_tk['fram_wav_fit'] = fram_wav_fit
    fram_wav_fit.grid(row=2, column=0, sticky=W)
    frame_wav_fit_title = tk.Frame(fram_wav_fit)
    frame_wav_fit_title.grid(row=0, column=0, sticky=W)
    dico_interface_tk['frame_wav_fit_title'] = frame_wav_fit_title
    label_wav_fit_title = tk.Label(frame_wav_fit_title,
                                   text='Number of living points [wav_fit]:',
                                   font='Arial 13 bold')
    label_wav_fit_title.grid(row=0, column=0, sticky=W)
    frame_wav_fit_buttons = tk.Frame(fram_wav_fit)
    frame_wav_fit_buttons.grid(row=1, column=0, sticky=W)
    entry_selec_wav_fit = tk.Entry(frame_wav_fit_buttons, width=30)
    dico_interface_tk['wav_fit_entry_select'] = entry_selec_wav_fit
    entry_selec_wav_fit.grid(row=0, column=0, sticky=W)

    entry_selec_wav_fit.insert(0, dico_interface_param['wav_fit'])
    if dico_interface_param['wav_fit'] == '0.95, 2.5':
        label_wav_fit_title = tk.Label(dico_interface_tk['frame_wav_fit_title'],
                                       text='Number of living points [wav_fit]:',
                                       font='Arial 13 bold', fg='orange')
        label_wav_fit_title.grid(row=0, column=0, sticky=W)

    button_validate = tk.Button(frame_wav_fit_buttons, text='->',
                                command=lambda: validate_wav_fit(dico_interface_tk))
    button_validate.grid(row=0, column=1, sticky=W)


def validate_wav_fit(dico_interface_tk, plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    wav_fit = dico_interface_tk['wav_fit_entry_select'].get()
    wav_fit = wav_fit.split('/')
    test = 1
    for wav_fit_range in wav_fit:
        wav_fit_range = wav_fit_range.split(',')
        if isfloat(wav_fit_range[0]) and isfloat(wav_fit_range[1]):
            if float(wav_fit_range[0]) < float(wav_fit_range[1]):
                pass
            else:
                test = 0
        else:
            test = 0
    if test == 1:
        dico_interface_param['wav_fit'] = dico_interface_tk['wav_fit_entry_select'].get()
        label_wav_fit_title = tk.Label(dico_interface_tk['frame_wav_fit_title'],
                                       text='Wavelength range used for the fit (um) [wav_fit]:',
                                       font='Arial 13 bold', fg='green')
        label_wav_fit_title.grid(row=0, column=0, sticky=W)
    else:
        label_wav_fit_title = tk.Label(dico_interface_tk['frame_wav_fit_title'],
                                       text='Wavelength range used for the fit (um) [wav_fit]:',
                                       font='Arial 13 bold', fg='red')
        label_wav_fit_title.grid(row=0, column=0, sticky=W)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param,  dico)

    if plot == 'yes':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)


def define_npoint(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    fram_npoint = tk.Frame(dico_interface_tk['frame_param_fit'])
    dico_interface_tk['fram_npoint'] = fram_npoint
    fram_npoint.grid(row=4, column=0, sticky=W)
    frame_npoint_title = tk.Frame(fram_npoint)
    frame_npoint_title.grid(row=0, column=0, sticky=W)
    dico_interface_tk['frame_npoint_title'] = frame_npoint_title
    label_npoint_title = tk.Label(frame_npoint_title,
                                  text='Number of living points [npoint]:',
                                  font='Arial 13 bold')
    label_npoint_title.grid(row=0, column=0, sticky=W)
    frame_npoint_buttons = tk.Frame(fram_npoint)
    frame_npoint_buttons.grid(row=1, column=0, sticky=W)
    entry_selec_npoint = tk.Entry(frame_npoint_buttons, width=12)
    dico_interface_tk['npoint_entry_select'] = entry_selec_npoint
    entry_selec_npoint.grid(row=0, column=0, sticky=W)

    entry_selec_npoint.insert(0, dico_interface_param['npoint'])
    button_validate = tk.Button(frame_npoint_buttons, text='->',
                                command=lambda: validate_npoint(dico_interface_tk))
    button_validate.grid(row=0, column=1, sticky=W)


def validate_npoint(dico_interface_tk, plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    npoint = dico_interface_tk['npoint_entry_select'].get()
    if isfloat(npoint):
        dico_interface_param['npoint'] = int(npoint)
        label_npoint_title = tk.Label(dico_interface_tk['frame_npoint_title'],
                                      text='Number of living points [npoint]:',
                                      font='Arial 13 bold', fg='green')
        label_npoint_title.grid(row=0, column=0, sticky=W)
    else:
        label_npoint_title = tk.Label(dico_interface_tk['frame_npoint_title'],
                                      text='Number of living points [npoint]:',
                                      font='Arial 13 bold', fg='red')
        label_npoint_title.grid(row=0, column=0, sticky=W)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param,  dico)

    if plot == 'yes':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)


def validate_fit(dico_interface_tk, ini='no'):

    validate_switch_ns_algo(dico_interface_tk, plot='no')
    validate_wav_fit(dico_interface_tk, plot='no')
    validate_npoint(dico_interface_tk, plot='no')
    if ini == 'no':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)
