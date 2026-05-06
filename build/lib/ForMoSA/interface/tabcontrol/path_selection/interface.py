import tkinter as tk
from tkinter import *
from tkinter import ttk
import json
from PIL import ImageTk


def path_selection_interface(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the paths
    frame_paths = tk.Frame(dico_interface_tk['tab_path_selection'])
    frame_paths.place(x=0, y=0, anchor=NW)
    dico_interface_tk['frame_paths'] = frame_paths
    path_list = [['observation_path', "Observation file's path [observation_path]:", 0, 0],
                 ['adapt_store_path', "Path to store the adapted model grid [adapt_store_path]:", 1, 0],
                 ['result_path', "Path to store the results [result_path]:", 2, 0],
                 ['model_path', "Model grid's path [model_path]:", 3, 0]
                 ]
    for path in path_list:
        key = path[0]
        title = path[1]
        row = path[2]
        col = path[3]
        frame_master = tk.Frame(dico_interface_tk['frame_paths'])
        frame_master.grid(row=row, column=col, sticky=W)
        frame_title = tk.Frame(frame_master)
        frame_title.grid(row=0, column=0, sticky=W)
        dico_interface_tk[key + '_title'] = frame_title
        label_title = tk.Label(frame_title, text=title, font='Arial 13 bold')
        label_title.grid(row=0, column=0, sticky=W)
        frame_buttons = tk.Frame(frame_master)
        frame_buttons.grid(row=1, column=0, sticky=W)
        entry_selec = tk.Entry(frame_buttons, width=111)
        dico_interface_tk[key + '_entry_select'] = entry_selec
        entry_selec.grid(row=0, column=1, sticky=W)

        if key == 'observation_path':
            entry_selec.insert(0, dico_interface_param['observation_path'])
            if dico_interface_param['observation_path'] == \
                    '/Users/ppalmabifani/Desktop/exoAtm/c0_ForMoSA/ForMoSA/DEMO/inputs/test_data_ABPicb.fits':
                label_title = tk.Label(frame_title, text=title, font='Arial 13 bold', fg='orange')
                label_title.grid(row=0, column=0, sticky=W)
            from ForMoSA.interface.tabcontrol.path_selection.utilities import button_observation_path
            button_open = tk.Button(frame_buttons, text='Open',
                                    command=lambda: button_observation_path(dico_interface_tk))
            button_vali = tk.Button(frame_buttons, text='->',
                                    command=lambda: button_observation_path(dico_interface_tk, 'select'))

        if key == 'adapt_store_path':
            entry_selec.insert(0, dico_interface_param['adapt_store_path'])
            if dico_interface_param['adapt_store_path'] == \
                    '/Users/ppalmabifani/Desktop/exoAtm/c0_ForMoSA/ForMoSA/DEMO/outputs/adapted_grid/':
                label_title = tk.Label(frame_title, text=title, font='Arial 13 bold', fg='orange')
                label_title.grid(row=0, column=0, sticky=W)
            from ForMoSA.interface.tabcontrol.path_selection.utilities import button_adapt_store_path
            button_open = tk.Button(frame_buttons, text='Open',
                                    command=lambda: button_adapt_store_path(dico_interface_tk, dico_interface_param))
            button_vali = tk.Button(frame_buttons, text='->',
                                    command=lambda: button_adapt_store_path(dico_interface_tk, 'select'))

        if key == 'result_path':
            entry_selec.insert(0, dico_interface_param['result_path'])
            if dico_interface_param['result_path'] == \
                    '/Users/ppalmabifani/Desktop/exoAtm/c0_ForMoSA/ForMoSA/DEMO/outputs/':
                label_title = tk.Label(frame_title, text=title, font='Arial 13 bold', fg='orange')
                label_title.grid(row=0, column=0, sticky=W)
            from ForMoSA.interface.tabcontrol.path_selection.utilities import button_result_path
            button_open = tk.Button(frame_buttons, text='Open', command=lambda: button_result_path(dico_interface_tk))
            button_vali = tk.Button(frame_buttons, text='->', command=lambda: button_result_path(dico_interface_tk,
                                                                                                 'select'))

        if key == 'model_path':
            entry_selec.insert(0, dico_interface_param['model_path'])
            if dico_interface_param['model_path'] == \
                    '/Users/ppalmabifani/Desktop/exoAtm/c0_ForMoSA/ForMoSA/DEMO/outputs/':
                label_title = tk.Label(frame_title, text=title, font='Arial 13 bold', fg='orange')
                label_title.grid(row=0, column=0, sticky=W)
            from ForMoSA.interface.tabcontrol.path_selection.utilities import button_model_path
            button_open = tk.Button(frame_buttons, text='Open', command=lambda: button_model_path(dico_interface_tk))
            button_vali = tk.Button(frame_buttons, text='->', command=lambda: button_model_path(dico_interface_tk,
                                                                                                'select'))
        button_open.grid(row=0, column=0, sticky=W)
        button_vali.grid(row=0, column=2, sticky=W)
    frame_paths.update()
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the button to check all
    frame_butt_vali = tk.Frame(dico_interface_tk['tab_path_selection'])
    y_place = 10 + frame_paths.winfo_height()
    frame_butt_vali.place(x=0, y=y_place, anchor=NW)
    from ForMoSA.interface.tabcontrol.path_selection.utilities import validate_path_selection
    button_validate = tk.Button(frame_butt_vali, text='Check all', font='Arial 13 bold', fg='green',
                                command=lambda: validate_path_selection(dico_interface_tk))
    button_validate.grid(row=0, column=0, sticky=W)
    frame_butt_vali.update()
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame for parameters for the plot
    frame_select_data_model_param_plot = tk.Frame(dico_interface_tk['tab_path_selection'])
    dico_interface_tk['frame_select_data_model_param_plot'] = frame_select_data_model_param_plot
    y_place = y_place + 20 + frame_butt_vali.winfo_height()
    frame_select_data_model_param_plot.place(x=0, y=y_place, anchor=NW)
    from ForMoSA.interface.tabcontrol.utilities import define_grid_parameter_to_plot
    define_grid_parameter_to_plot(dico_interface_tk, 'frame_select_data_model_param_plot')
    frame_select_data_model_param_plot.update()
    ttk.Separator(frame_select_data_model_param_plot, orient=HORIZONTAL).grid(row=1, column=0, pady=5, sticky='ew')
    # Frame of xlim and ylim
    from ForMoSA.interface.tabcontrol.utilities import define_xlim_ylim
    define_xlim_ylim(dico_interface_tk, 'frame_select_data_model_param_plot')
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Separator
    canvas = Canvas(dico_interface_tk['tab_path_selection'], width=frame_select_data_model_param_plot.winfo_width(),
                    height=1, bg='black')
    canvas.place(x=0, y=y_place-10, anchor=W)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of le plot
    from ForMoSA.interface.tabcontrol.path_selection.plot import plot_obs_vs_model
    plot_obs_vs_model(dico_interface_tk, frame_select_data_model_param_plot.winfo_width() + 25, 210)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the logo
    import os
    import ForMoSA
    path_logo = os.path.abspath(ForMoSA.__file__)
    path_logo = path_logo.split('__init__.py')
    path_logo = path_logo[0] + 'interface/utilities/formosa_logo_small.png'
    logo_tk = ImageTk.PhotoImage(master=dico_interface_tk['tab_path_selection'], file=path_logo)
    logo_label = tk.Label(dico_interface_tk['tab_path_selection'], image=logo_tk)
    logo_label.image = logo_tk
    logo_label.place(x=0, y=608, anchor=SW)

    validate_path_selection(dico_interface_tk, ini='yes')
