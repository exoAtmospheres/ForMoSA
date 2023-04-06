import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import webbrowser
import json
import os
import ForMoSA

def openurl(url):
    webbrowser.open_new(url)


def generate_config_file_interface(dico_interface_tk):

    dico_interface_tk['root_config_file_path'] = tk.Tk()
    screenwidth = dico_interface_tk['root_config_file_path'].winfo_screenwidth()
    screenheight = dico_interface_tk['root_config_file_path'].winfo_screenheight()
    windwidth = 1100
    windheight = 600
    d_width = int((screenwidth - windwidth) / 2)
    d_height = int((screenheight - windheight) / 2)
    dico_interface_tk['root_config_file_path'].geometry(str(windwidth) + "x" + str(windheight) + "+" +
                                                        str(d_width) + "+" + str(d_height))
    dico_interface_tk['root_config_file_path'].title('ForMoSA interface')
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the title
    frame_title = tk.Frame(dico_interface_tk['root_config_file_path'])
    frame_title.place(x=40, y=60, anchor=NW)
    label_title_a = tk.Label(frame_title, text='Welcome to the ForMoSA interface', font='Arial 40 bold')
    label_title_a.grid(row=0, column=0, sticky=W)
    label_title_b = tk.Label(frame_title, text='Forward Modeling for Spectral Analysis', font='Arial 20 italic')
    label_title_b.grid(row=1, column=0, sticky=W)
    # Frame of the logo
    width_logo, height_logo = int(1245/5), int(745/5)
    path_logo = os.path.abspath(ForMoSA.__file__)
    path_logo = path_logo.split('__init__.py')
    path_logo = path_logo[0] + 'interface/utilities/formosa_logo.png'
    logo = Image.open(path_logo)
    logo = logo.resize((width_logo, height_logo))
    logo_tk = ImageTk.PhotoImage(logo)
    logo_label = tk.Label(dico_interface_tk['root_config_file_path'], image=logo_tk)
    logo_label.image = logo_tk
    logo_label.place(x=1070, y=30, anchor=NE)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the question
    frame_quest_ini = tk.Frame(dico_interface_tk['root_config_file_path'])
    frame_quest_ini.place(x=40, y=170, anchor=NW)
    label_quest_a = tk.Label(frame_quest_ini,
                             text='Please, give the path of the directory that contains your configuration_file.ini',
                             font='Arial 15 bold')
    label_quest_a.grid(row=0, column=0, sticky=SW)
    label_quest_b = tk.Label(frame_quest_ini,
                             text='If the configuration_file.ini does not exists, a template will be generated.',
                             font='Arial 15')
    label_quest_b.grid(row=1, column=0, sticky=SW)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the open + bar + ->
    frame_selection = tk.Frame(dico_interface_tk['root_config_file_path'])
    frame_selection.place(x=40, y=220, anchor=NW)
    # Frame of the open
    from ForMoSA.interface.main_window.utilities import button_config_file_path
    button_open = tk.Button(frame_selection, text='Open', command=lambda: button_config_file_path(dico_interface_tk))
    button_open.grid(row=0, column=0, sticky=W)
    # Frame of the bar
    entry_bar = tk.Entry(frame_selection, width=60)
    dico_interface_tk['config_file_path_entry_select'] = entry_bar
    entry_bar.grid(row=0, column=1, sticky=W)

    path_previous_configuration_file = os.path.abspath(ForMoSA.__file__)
    path_previous_configuration_file = path_previous_configuration_file.split('__init__.py')
    path_previous_configuration_file = path_previous_configuration_file[0] + 'interface/utilities/' \
                                                                             'dico_previous_configuration_file.json'
    if os.path.exists(path_previous_configuration_file):
        with open(path_previous_configuration_file) as dico:
            dico_previous_configuration_file = json.load(dico)
        previous_configuration_file = dico_previous_configuration_file['previous_configuration_file']
        entry_bar.insert(0, previous_configuration_file)
    # Frame of the ->
    button_vali = tk.Button(frame_selection, text='->', command=lambda: button_config_file_path(dico_interface_tk,
                                                                                                'select'))
    button_vali.grid(row=0, column=2, sticky=W)
    frame_validate_path_config_file = tk.Frame(dico_interface_tk['root_config_file_path'])
    frame_validate_path_config_file.grid(row=2, column=0, padx=30, pady=20, sticky=W)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------

    canvas = Canvas(dico_interface_tk['root_config_file_path'],
                    width=1050, height=1, bg='black')
    canvas.place(x=20, y=270, anchor=W)

    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the configuration_file.ini maker
    frame_configuration_file_maker = tk.Frame(dico_interface_tk['root_config_file_path'])
    frame_configuration_file_maker.place(x=40, y=300, anchor=NW)
    label_configuration_file_maker = tk.Label(frame_configuration_file_maker,
                                              text='Modify and check your configuration_file.ini:',
                                              font='Arial 15')
    label_configuration_file_maker.grid(sticky=SW)
    from ForMoSA.interface.main_window.utilities import button_vali_config_file_path_fct
    button_vali = tk.Button(frame_configuration_file_maker, text='configuration_file.ini maker', font='Arial 13 bold',
                            command=lambda: button_vali_config_file_path_fct(dico_interface_tk), height=3)
    button_vali.grid(row=1, sticky=W)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the ForMoSA launcher
    frame_formosa_launcher = tk.Frame(dico_interface_tk['root_config_file_path'])
    dico_interface_tk['frame_formosa_launcher'] = frame_formosa_launcher
    frame_formosa_launcher.place(x=40, y=400, anchor=NW)
    label_formosa_launcher = tk.Label(dico_interface_tk['frame_formosa_launcher'], text='Launch ForMoSA:',
                                      font='Arial 15')
    label_formosa_launcher.grid(sticky=SW)
    from ForMoSA.interface.main_window.utilities import button_launch_adapt_fct
    button_launch_adapt = tk.Button(frame_formosa_launcher, text='Grid adaptation', font='Arial 13 bold',
                                    command=lambda: button_launch_adapt_fct(dico_interface_tk),
                                    width=15, height=3)
    button_launch_adapt.grid(row=1, sticky=W)
    from ForMoSA.interface.main_window.utilities import button_launch_fit_fct
    button_launch_fit = tk.Button(frame_formosa_launcher, text='Inversion', font='Arial 13 bold',
                                  command=lambda: button_launch_fit_fct(dico_interface_tk),
                                  width=15, height=3)
    button_launch_fit.grid(row=1, column=1, sticky=W)
    frame_formosa_launcher.update()
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the plot maker
    frame_plot_maker = tk.Frame(dico_interface_tk['root_config_file_path'])
    frame_plot_maker.place(x=40, y=500, anchor=NW)
    label_plot_maker = tk.Label(frame_plot_maker, text='Gerenate your plots:', font='Arial 15')
    label_plot_maker.grid(sticky=SW)
    button_vali = tk.Button(frame_plot_maker, text='Plot maker /!\\', font='Arial 13 bold',
                            command=lambda: button_vali_config_file_path_fct(dico_interface_tk),
                            height=3)
    button_vali.grid(row=1, sticky=W)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the grid downloading
    frame_grid_download = tk.Frame(dico_interface_tk['root_config_file_path'])
    frame_grid_download.place(x=1070, y=300, anchor=NE)
    label_grid_download = tk.Label(frame_grid_download, text='Download the model grids:', font='Arial 15')
    label_grid_download.grid(sticky=SE)
    frame_grid_download_buttons = tk.Frame(frame_grid_download)
    frame_grid_download_buttons.grid(row=1, sticky=SE)
    url_btsettl = 'https://zenodo.org/record/7671384#.ZCnKB-zMKrc'
    button_btsettl = tk.Button(frame_grid_download_buttons, text='BT-SETTL', font='Arial 13 bold',
                               command=lambda aurl=url_btsettl: openurl(url_btsettl),
                               height=3)
    button_btsettl.grid(sticky=E)
    url_exorem = 'https://zenodo.org/record/7670904#.ZCnKG-zMKrc'
    button_exorem = tk.Button(frame_grid_download_buttons, text='Exo-REM', font='Arial 13 bold',
                              command=lambda aurl=url_exorem: openurl(url_exorem),
                              height=3)
    button_exorem.grid(row=0, column=1, sticky=W)
    url_atmo = 'https://zenodo.org/record/7670904#.ZCnKG-zMKrc'
    button_atmo = tk.Button(frame_grid_download_buttons, text='ATMO /!\\', font='Arial 13 bold',
                            command=lambda aurl=url_atmo: openurl(url_atmo),
                            height=3)
    button_atmo.grid(row=1, sticky=E)
    url_sonora = 'https://zenodo.org/record/7670904#.ZCnKG-zMKrc'
    button_sonora = tk.Button(frame_grid_download_buttons, text='SONORA /!\\', font='Arial 13 bold',
                              command=lambda aurl=url_sonora: openurl(url_sonora),
                              height=3)
    button_sonora.grid(row=1, column=1, sticky=W)
    url_driftphoenix = 'https://zenodo.org/record/7670904#.ZCnKG-zMKrc'
    button_driftphoenix = tk.Button(frame_grid_download_buttons, text='DRIFT-PHOENIX /!\\', font='Arial 13 bold',
                                    command=lambda aurl=url_driftphoenix: openurl(url_driftphoenix),
                                    height=3)
    button_driftphoenix.grid(row=2, columnspan=2, sticky=E)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the documentation
    frame_documentation = tk.Frame(dico_interface_tk['root_config_file_path'])
    frame_documentation.place(x=1070, y=500, anchor=NE)
    label_documentation = tk.Label(frame_documentation, text='Access the documentation:', font='Arial 15')
    label_documentation.grid(sticky=SW)
    url_documentation = 'https://formosa.readthedocs.io/en/latest/#'
    button_documentation = tk.Button(frame_documentation, text='Documentation', font='Arial 13 bold',
                                     command=lambda aurl=url_documentation: openurl(url_documentation),
                                     height=3)
    button_documentation.grid(row=1, sticky=E)
