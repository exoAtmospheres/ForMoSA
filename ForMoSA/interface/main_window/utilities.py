from tkinter import *
import tkinter as tk
from tkinter import filedialog
import json
import os
import ForMoSA


def retrieve_previous_config(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    path_configuration_file_template = os.path.abspath(ForMoSA.__file__)
    path_configuration_file_template = path_configuration_file_template.split('__init__.py')
    path_configuration_file_template = path_configuration_file_template[0] + 'interface/' \
                                                                             'utilities/configuration_file.ini'
    if not os.path.isfile(dico_interface_tk['config_file_path']):
        os.system('cp ' + path_configuration_file_template + ' ' + dico_interface_tk['config_file_path'])
        warn_cp_wind = tk.Tk()
        screenwidth = warn_cp_wind.winfo_screenwidth()
        screenheight = warn_cp_wind.winfo_screenheight()
        windwidth = 500
        windheight = 100
        d_width = int((screenwidth - windwidth) / 2)
        d_height = int((screenheight - windheight) / 2)
        warn_cp_wind.geometry(str(windwidth)+"x"+str(windheight)+"+"+str(d_width)+"+"+str(d_height))
        warn_cp_wind.title(' /!\   WARNING   /!\ ')
        label_no_config = tk.Label(warn_cp_wind, text='No configuration_file.ini found. \n' 
                                                      'A template has been created.\n', font='Arial 14')
        label_no_config.place(x=250, y=40, anchor="center")
        button_ok = tk.Button(warn_cp_wind, text='OK', command=lambda: warn_cp_wind.destroy())
        button_ok.place(x=250, y=75, anchor="center")
        warn_cp_wind.wait_window(dico_interface_tk['root_config_file_path'])

    from ForMoSA.interface.utilities.globfile_interface import GlobFile
    global_params = GlobFile(dico_interface_tk['config_file_path'])
    dico_interface_param['observation_path'] = global_params.observation_path
    dico_interface_param['adapt_store_path'] = global_params.adapt_store_path
    dico_interface_param['result_path'] = global_params.result_path
    dico_interface_param['model_path'] = global_params.model_path
    dico_interface_param['wav_for_adapt'] = global_params.wav_for_adapt
    dico_interface_param['adapt_method'] = global_params.adapt_method
    dico_interface_param['custom_reso'] = global_params.custom_reso
    dico_interface_param['continuum_sub'] = global_params.continuum_sub
    dico_interface_param['wav_for_continuum'] = global_params.wav_for_continuum
    dico_interface_param['wav_fit'] = global_params.wav_fit
    dico_interface_param['ns_algo'] = global_params.ns_algo
    dico_interface_param['npoint'] = global_params.npoint
    dico_interface_param['par1'] = global_params.par1
    dico_interface_param['par2'] = global_params.par2
    dico_interface_param['par3'] = global_params.par3
    dico_interface_param['par4'] = global_params.par4
    dico_interface_param['par5'] = global_params.par5
    dico_interface_param['r'] = global_params.r
    dico_interface_param['d'] = global_params.d
    dico_interface_param['rv'] = global_params.rv
    dico_interface_param['av'] = global_params.av
    dico_interface_param['vsini'] = global_params.vsini
    dico_interface_param['ld'] = global_params.ld

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as fp:
        json.dump(dico_interface_param, fp)


def button_config_file_path(dico_interface_tk, choose_mode='choose'):

    if choose_mode == 'choose':
        dico_interface_tk['config_file_path_entry_select'].delete(0, tk.END)
        config_file_path_entry_select_askdirectory = filedialog.askdirectory()
        dico_interface_tk['config_file_path_entry_select'].insert(0, config_file_path_entry_select_askdirectory)
    if os.path.isdir(dico_interface_tk['config_file_path_entry_select'].get()):
        dico_interface_tk['config_file_path_folder'] = dico_interface_tk['config_file_path_entry_select'].get() + '/'
        dico_interface_tk['config_file_path'] = dico_interface_tk['config_file_path_folder'] + 'configuration_file.ini'
        if not os.path.isfile(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json'):
            dico_interface_param = {}
            with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as fp:
                json.dump(dico_interface_param, fp)
        retrieve_previous_config(dico_interface_tk)
    else:
        pass


def button_vali_config_file_path_fct(dico_interface_tk):

    if dico_interface_tk['config_file_path_entry_select'].get()[-1] != '/':
        dico_interface_tk['config_file_path_folder'] = dico_interface_tk['config_file_path_entry_select'].get() + '/'
    else:
        dico_interface_tk['config_file_path_folder'] = dico_interface_tk['config_file_path_entry_select'].get()
    dico_interface_tk['config_file_path'] = dico_interface_tk['config_file_path_folder'] + 'configuration_file.ini'

    if not os.path.exists(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json'):
        dico_interface_param = {}
        with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json', 'w') as fp:
            json.dump(dico_interface_param, fp)

    retrieve_previous_config(dico_interface_tk)
    from ForMoSA.interface.tabcontrol.interface import tabcontrol_interface
    tabcontrol_interface(dico_interface_tk)
    # dico_interface_tk['root_config_file_path'].destroy()


def button_launch_adapt_fct(dico_interface_tk):
    if 'config_file_path' in dico_interface_tk.keys():
        from ForMoSA.main_utilities import GlobFile
        global_params = GlobFile(dico_interface_tk['config_file_path'])
        from ForMoSA.adapt.adapt_obs_mod import launch_adapt
        launch_adapt(global_params, justobs='no')
        label_formosa_launcher = tk.Label(dico_interface_tk['frame_formosa_launcher'],
                                          text='Launch ForMoSA:',
                                          font='Arial 15')
        label_formosa_launcher.grid(sticky=SW)
    else:
        label_formosa_launcher = tk.Label(dico_interface_tk['frame_formosa_launcher'],
                                          text='Launch ForMoSA: You need to define a configuration_file.ini',
                                          font='Arial 15', fg='red')
        label_formosa_launcher.grid(sticky=SW)


def button_launch_fit_fct(dico_interface_tk):
    if 'config_file_path' in dico_interface_tk.keys():
        from ForMoSA.main_utilities import GlobFile
        global_params = GlobFile(dico_interface_tk['config_file_path'])
        from ForMoSA.nested_sampling.nested_sampling import launch_nested_sampling
        launch_nested_sampling(global_params)
        label_formosa_launcher = tk.Label(dico_interface_tk['frame_formosa_launcher'],
                                          text='Launch ForMoSA:',
                                          font='Arial 15')
        label_formosa_launcher.grid(sticky=SW)
    else:
        label_formosa_launcher = tk.Label(dico_interface_tk['frame_formosa_launcher'],
                                          text='Launch ForMoSA: You need to define a configuration_file.ini',
                                          font='Arial 15', fg='red')
        label_formosa_launcher.grid(sticky=SW)
