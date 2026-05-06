import json
import tkinter as tk
from tkinter import *

from ForMoSA.interface.tabcontrol.utilities import isfloat


def define_wav_for_adapt(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    fram_wav_for_adapt = tk.Frame(dico_interface_tk['frame_param_adapt_grid_data'])
    dico_interface_tk['fram_wav_for_adapt'] = fram_wav_for_adapt
    fram_wav_for_adapt.grid(row=0, column=0, sticky=W)
    frame_wav_for_adapt_title = tk.Frame(fram_wav_for_adapt)
    frame_wav_for_adapt_title.grid(row=0, column=0, sticky=W)
    dico_interface_tk['frame_wav_for_adapt_title'] = frame_wav_for_adapt_title
    label_wav_for_adapt_title = tk.Label(frame_wav_for_adapt_title,
                                         text='Wavelength range to adapt the grid (um) [wav_for_adapt]:',
                                         font='Arial 13 bold')
    label_wav_for_adapt_title.grid(row=0, column=0, sticky=W)
    frame_wav_for_adapt_buttons = tk.Frame(fram_wav_for_adapt)
    frame_wav_for_adapt_buttons.grid(row=1, column=0, sticky=W)
    entry_selec_wav_for_adapt = tk.Entry(frame_wav_for_adapt_buttons, width=30)
    dico_interface_tk['wav_for_adapt_entry_select'] = entry_selec_wav_for_adapt
    entry_selec_wav_for_adapt.grid(row=0, column=0, sticky=W)

    entry_selec_wav_for_adapt.insert(0, dico_interface_param['wav_for_adapt'])
    if dico_interface_param['wav_for_adapt'] == '0.95, 2.5':
        label_wav_for_adapt_title = tk.Label(dico_interface_tk['frame_wav_for_adapt_title'],
                                             text='Wavelength range to adapt the grid (um) [wav_for_adapt]:',
                                             font='Arial 13 bold', fg='orange')
        label_wav_for_adapt_title.grid(row=0, column=0, sticky=W)

    button_validate = tk.Button(frame_wav_for_adapt_buttons, text='->',
                                command=lambda: validate_wav_for_adapt(dico_interface_tk))
    button_validate.grid(row=0, column=1, sticky=W)


def validate_wav_for_adapt(dico_interface_tk, plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    wav_for_adapt = dico_interface_tk['wav_for_adapt_entry_select'].get()
    wav_for_adapt = wav_for_adapt.split('/')
    test = 1
    for wav_for_adapt_range in wav_for_adapt:
        wav_for_adapt_range = wav_for_adapt_range.split(',')
        if isfloat(wav_for_adapt_range[0]) and isfloat(wav_for_adapt_range[1]):
            if float(wav_for_adapt_range[0]) < float(wav_for_adapt_range[1]):
                pass
            else:
                test = 0
        else:
            test = 0
    if test == 1:
        dico_interface_param['wav_for_adapt'] = dico_interface_tk['wav_for_adapt_entry_select'].get()
        label_wav_for_adapt_title = tk.Label(dico_interface_tk['frame_wav_for_adapt_title'],
                                             text='Wavelength range to adapt the grid (um) [wav_for_adapt]:',
                                             font='Arial 13 bold', fg='green')
        label_wav_for_adapt_title.grid(row=0, column=0, sticky=W)
    else:
        label_wav_for_adapt_title = tk.Label(dico_interface_tk['frame_wav_for_adapt_title'],
                                             text='Wavelength range to adapt the grid (um) [wav_for_adapt]:',
                                             font='Arial 13 bold', fg='red')
        label_wav_for_adapt_title.grid(row=0, column=0, sticky=W)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param,  dico)

    if plot == 'yes':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)


def switch_opt_box_adapt_method(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    frame_adapt_method = tk.Frame(dico_interface_tk['frame_param_adapt_grid_data'])
    frame_adapt_method.grid(row=2, column=0, sticky=W)
    frame_adapt_method_title = tk.Frame(frame_adapt_method)
    frame_adapt_method_title.grid(row=0, column=0, sticky=W)
    dico_interface_tk['frame_adapt_method_title'] = frame_adapt_method_title
    label_adapt_method_title = tk.Label(frame_adapt_method_title, text='Resolution adjustment method [adapt_method]:',
                                        font='Arial 13 bold')
    label_adapt_method_title.grid(row=0, column=0, sticky=W)

    frame_switch_adapt_method = tk.Frame(frame_adapt_method)
    dico_interface_tk['frame_switch_adapt_method'] = frame_switch_adapt_method
    frame_switch_adapt_method.grid(row=1, column=0, sticky=W)

    dico_interface_tk['adapt_method_choice'] = tk.StringVar(dico_interface_tk['frame_switch_adapt_method'],
                                                            dico_interface_param['adapt_method'])

    radio_but1 = tk.Radiobutton(dico_interface_tk['frame_switch_adapt_method'], text='Re-sample the models (defaut)',
                                value='by_sample', variable=dico_interface_tk['adapt_method_choice'],
                                command=lambda: validate_adapt_method(dico_interface_tk))
    radio_but1.grid(row=0, column=0, sticky=W)
    radio_but2 = tk.Radiobutton(dico_interface_tk['frame_switch_adapt_method'], text='Reduce the resolution',
                                value='by_reso', variable=dico_interface_tk['adapt_method_choice'],
                                command=lambda: validate_adapt_method(dico_interface_tk))
    radio_but2.grid(row=0, column=2, sticky=W)


def validate_adapt_method(dico_interface_tk, plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    dico_interface_param['adapt_method'] = dico_interface_tk['adapt_method_choice'].get()

    label_adapt_method_title = tk.Label(dico_interface_tk['frame_adapt_method_title'],
                                        text='Resolution adjustment method [adapt_method]:',
                                        font='Arial 13 bold', fg='green')
    label_adapt_method_title.grid(row=0, column=0, sticky=W)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param,  dico)

    if plot == 'yes':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)


def define_custom_reso(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    fram_custom_reso = tk.Frame(dico_interface_tk['frame_param_adapt_grid_data'])
    dico_interface_tk['fram_custom_reso'] = fram_custom_reso
    fram_custom_reso.grid(row=3, column=0, sticky=W)
    frame_custom_reso_title = tk.Frame(fram_custom_reso)
    frame_custom_reso_title.grid(row=0, column=0, sticky=W)
    dico_interface_tk['frame_custom_reso_title'] = frame_custom_reso_title
    label_custom_reso_title = tk.Label(frame_custom_reso_title,
                                       text='Custom target resolution (default: NA) [custom_reso]:',
                                       font='Arial 13 bold')
    label_custom_reso_title.grid(row=0, column=0, sticky=W)
    frame_custom_reso_buttons = tk.Frame(fram_custom_reso)
    frame_custom_reso_buttons.grid(row=1, column=0, sticky=W)
    entry_selec_custom_reso = tk.Entry(frame_custom_reso_buttons, width=12)
    dico_interface_tk['custom_reso_entry_select'] = entry_selec_custom_reso
    entry_selec_custom_reso.grid(row=0, column=0, sticky=W)

    entry_selec_custom_reso.insert(0, dico_interface_param['custom_reso'])

    button_validate = tk.Button(frame_custom_reso_buttons, text='->',
                                command=lambda: validate_custom_reso(dico_interface_tk))
    button_validate.grid(row=0, column=1, sticky=W)


def validate_custom_reso(dico_interface_tk, plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    custom_reso = dico_interface_tk['custom_reso_entry_select'].get()

    if isfloat(custom_reso) or custom_reso == 'NA':
        dico_interface_param['custom_reso'] = custom_reso
        label_custom_reso_title = tk.Label(dico_interface_tk['frame_custom_reso_title'],
                                           text='Custom target resolution (default: NA) [custom_reso]:',
                                           font='Arial 13 bold', fg='green')
        label_custom_reso_title.grid(row=0, column=0, sticky=W)
    else:
        label_custom_reso_title = tk.Label(dico_interface_tk['frame_custom_reso_title'],
                                           text='Custom target resolution (default: NA) [custom_reso]:',
                                           font='Arial 13 bold', fg='red')
        label_custom_reso_title.grid(row=0, column=0, sticky=W)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param,  dico)

    if plot == 'yes':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)


def define_continuum_sub(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    fram_continuum_sub = tk.Frame(dico_interface_tk['frame_param_adapt_grid_data'])
    dico_interface_tk['fram_continuum_sub'] = fram_continuum_sub
    fram_continuum_sub.grid(row=5, column=0, sticky=W)
    frame_continuum_sub_title = tk.Frame(fram_continuum_sub)
    frame_continuum_sub_title.grid(row=0, column=0, sticky=W)
    dico_interface_tk['frame_continuum_sub_title'] = frame_continuum_sub_title
    label_continuum_sub_title = tk.Label(frame_continuum_sub_title,
                                         text='Resolution of the continuum (default: NA) [continuum_sub]:',
                                         font='Arial 13 bold')
    label_continuum_sub_title.grid(row=0, column=0, sticky=W)
    frame_continuum_sub_buttons = tk.Frame(fram_continuum_sub)
    frame_continuum_sub_buttons.grid(row=1, column=0, sticky=W)
    entry_selec_continuum_sub = tk.Entry(frame_continuum_sub_buttons, width=12)
    dico_interface_tk['continuum_sub_entry_select'] = entry_selec_continuum_sub
    entry_selec_continuum_sub.grid(row=0, column=0, sticky=W)

    entry_selec_continuum_sub.insert(0, dico_interface_param['continuum_sub'])


def define_wav_for_continuum(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    fram_wav_for_continuum = tk.Frame(dico_interface_tk['frame_param_adapt_grid_data'])
    dico_interface_tk['fram_wav_for_continuum'] = fram_wav_for_continuum
    fram_wav_for_continuum.grid(row=6, column=0, sticky=W)
    frame_wav_for_continuum_title = tk.Frame(fram_wav_for_continuum)
    frame_wav_for_continuum_title.grid(row=0, column=0, sticky=W)
    dico_interface_tk['frame_wav_for_continuum_title'] = frame_wav_for_continuum_title
    label_wav_for_continuum_title = tk.Label(frame_wav_for_continuum_title,
                                             text='Wavelengths used (um) [wav_for_continuum]:',
                                             font='Arial 13 bold')
    label_wav_for_continuum_title.grid(row=0, column=0, sticky=W)
    frame_wav_for_continuum_buttons = tk.Frame(fram_wav_for_continuum)
    frame_wav_for_continuum_buttons.grid(row=1, column=0, sticky=W)
    entry_selec_wav_for_continuum = tk.Entry(frame_wav_for_continuum_buttons, width=12)
    dico_interface_tk['wav_for_continuum_entry_select'] = entry_selec_wav_for_continuum
    entry_selec_wav_for_continuum.grid(row=0, column=0, sticky=W)

    entry_selec_wav_for_continuum.insert(0, dico_interface_param['wav_for_continuum'])

    button_validate = tk.Button(frame_wav_for_continuum_buttons, text='->',
                                command=lambda: validate_wav_for_continuum(dico_interface_tk))
    button_validate.grid(row=0, column=1, sticky=W)


def validate_wav_for_continuum(dico_interface_tk, plot='yes'):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    continuum_sub = dico_interface_tk['continuum_sub_entry_select'].get()
    wav_for_continuum = dico_interface_tk['wav_for_continuum_entry_select'].get()

    if continuum_sub == 'NA':
        dico_interface_param['continuum_sub'] = continuum_sub
        label_continuum_sub_title = tk.Label(dico_interface_tk['frame_continuum_sub_title'],
                                             text='Resolution of the continuum (default: NA) [continuum_sub]:',
                                             font='Arial 13 bold', fg='green')
        label_continuum_sub_title.grid(row=0, column=0, sticky=W)
        if wav_for_continuum == 'NA':
            dico_interface_param['wav_for_continuum'] = wav_for_continuum
            label_wav_for_continuum_title = tk.Label(dico_interface_tk['frame_wav_for_continuum_title'],
                                                     text='Wavelengths used (um) [wav_for_continuum]:',
                                                     font='Arial 13 bold', fg='green')
            label_wav_for_continuum_title.grid(row=0, column=0, sticky=W)
        else:
            wav_for_continuum = wav_for_continuum.split('/')
            test = 1
            for wav_for_continuum_range in wav_for_continuum:
                wav_for_continuum_range = wav_for_continuum_range.split(',')
                if isfloat(wav_for_continuum_range[0]) and isfloat(wav_for_continuum_range[1]):
                    if float(wav_for_continuum_range[0]) < float(wav_for_continuum_range[1]):
                        pass
                    else:
                        test = 0
                else:
                    test = 0
            if test == 1:
                dico_interface_param['wav_for_continuum'] = dico_interface_tk['wav_for_continuum_entry_select'].get()
                label_wav_for_continuum_title = tk.Label(dico_interface_tk['frame_wav_for_continuum_title'],
                                                         text='Wavelengths used (um) [wav_for_continuum]:',
                                                         font='Arial 13 bold', fg='green')
                label_wav_for_continuum_title.grid(row=0, column=0, sticky=W)
            else:
                label_wav_for_continuum_title = tk.Label(dico_interface_tk['frame_wav_for_continuum_title'],
                                                         text='Wavelengths used (um) [wav_for_continuum]:',
                                                         font='Arial 13 bold', fg='red')
                label_wav_for_continuum_title.grid(row=0, column=0, sticky=W)

    elif isfloat(continuum_sub):
        dico_interface_param['continuum_sub'] = continuum_sub
        label_continuum_sub_title = tk.Label(dico_interface_tk['frame_continuum_sub_title'],
                                             text='Resolution of the continuum (default: NA) [continuum_sub]:',
                                             font='Arial 13 bold', fg='green')
        label_continuum_sub_title.grid(row=0, column=0, sticky=W)
        if wav_for_continuum == 'NA':
            label_wav_for_continuum_title = tk.Label(dico_interface_tk['frame_wav_for_continuum_title'],
                                                     text='Wavelengths used (um) [wav_for_continuum]:',
                                                     font='Arial 13 bold', fg='red')
            label_wav_for_continuum_title.grid(row=0, column=0, sticky=W)
        else:
            wav_for_continuum = wav_for_continuum.split('/')
            test = 1
            for wav_for_continuum_range in wav_for_continuum:
                wav_for_continuum_range = wav_for_continuum_range.split(',')
                if isfloat(wav_for_continuum_range[0]) and isfloat(wav_for_continuum_range[1]):
                    if float(wav_for_continuum_range[0]) < float(wav_for_continuum_range[1]):
                        pass
                    else:
                        test = 0
                else:
                    test = 0
            if test == 1:
                dico_interface_param['wav_for_continuum'] = dico_interface_tk['wav_for_continuum_entry_select'].get()
                label_wav_for_continuum_title = tk.Label(dico_interface_tk['frame_wav_for_continuum_title'],
                                                         text='Wavelengths used (um) [wav_for_continuum]:',
                                                         font='Arial 13 bold', fg='green')
                label_wav_for_continuum_title.grid(row=0, column=0, sticky=W)
            else:
                label_wav_for_continuum_title = tk.Label(dico_interface_tk['frame_wav_for_continuum_title'],
                                                         text='Wavelengths used (um) [wav_for_continuum]:',
                                                         font='Arial 13 bold', fg='red')
                label_wav_for_continuum_title.grid(row=0, column=0, sticky=W)
    else:
        label_continuum_sub_title = tk.Label(dico_interface_tk['frame_continuum_sub_title'],
                                             text='Resolution of the continuum (default: NA) [continuum_sub]:',
                                             font='Arial 13 bold', fg='red')
        label_continuum_sub_title.grid(row=0, column=0, sticky=W)
        label_wav_for_continuum_title = tk.Label(dico_interface_tk['frame_wav_for_continuum_title'],
                                                 text='Wavelengths used (um) [wav_for_continuum]:',
                                                 font='Arial 13 bold', fg='red')
        label_wav_for_continuum_title.grid(row=0, column=0, sticky=W)

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as dico:
        json.dump(dico_interface_param,  dico)

    if plot == 'yes':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)


def validate_adapt_grid_data(dico_interface_tk, ini='no'):

    validate_wav_for_adapt(dico_interface_tk, plot='no')
    validate_adapt_method(dico_interface_tk, plot='no')
    validate_custom_reso(dico_interface_tk, plot='no')
    validate_wav_for_continuum(dico_interface_tk, plot='no')
    if ini == 'no':
        from ForMoSA.interface.tabcontrol.utilities import replot_all
        replot_all(dico_interface_tk)
