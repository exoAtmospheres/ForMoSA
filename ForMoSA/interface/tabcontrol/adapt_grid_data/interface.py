import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import ImageTk


def adapt_grid_data_interface(dico_interface_tk):

    # Frame of the parameters
    frame_adapt_grid_data = tk.Frame(dico_interface_tk['tab_adapt_grid_data'])
    frame_adapt_grid_data.place(x=0, y=0, anchor=NW)
    frame_param_adapt_grid_data = tk.Frame(frame_adapt_grid_data)
    dico_interface_tk['frame_param_adapt_grid_data'] = frame_param_adapt_grid_data
    frame_param_adapt_grid_data.grid(row=1, column=0)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame for the definition of the wavelengths for the grid adaptation
    from ForMoSA.interface.tabcontrol.adapt_grid_data.utilities import define_wav_for_adapt
    define_wav_for_adapt(dico_interface_tk)
    ttk.Separator(frame_param_adapt_grid_data, orient=HORIZONTAL).grid(row=1, pady=5, sticky='ew')
    # Frame for the choice of the resolution methode
    from ForMoSA.interface.tabcontrol.adapt_grid_data.utilities import switch_opt_box_adapt_method
    switch_opt_box_adapt_method(dico_interface_tk)
    # Frame for the definition of the custom resolution
    from ForMoSA.interface.tabcontrol.adapt_grid_data.utilities import define_custom_reso
    define_custom_reso(dico_interface_tk)
    ttk.Separator(frame_param_adapt_grid_data, orient=HORIZONTAL).grid(row=4, padx=0, pady=5, sticky='ew')
    # Frame for the definition of the resolution for the continuum subtraction
    from ForMoSA.interface.tabcontrol.adapt_grid_data.utilities import define_continuum_sub
    define_continuum_sub(dico_interface_tk)
    # Frame for the definition of the wavelengths uses for the continuum subtraction
    from ForMoSA.interface.tabcontrol.adapt_grid_data.utilities import define_wav_for_continuum
    define_wav_for_continuum(dico_interface_tk)
    frame_param_adapt_grid_data.update()
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the button to check all
    frame_butt_vali = tk.Frame(dico_interface_tk['tab_adapt_grid_data'])
    y_place = 10 + frame_param_adapt_grid_data.winfo_height()
    frame_butt_vali.place(x=0, y=y_place, anchor=NW)
    from ForMoSA.interface.tabcontrol.adapt_grid_data.utilities import validate_adapt_grid_data
    button_validate = tk.Button(frame_butt_vali, text='Check all', font='Arial 13 bold', fg='green',
                                command=lambda: validate_adapt_grid_data(dico_interface_tk))
    button_validate.grid(row=0, column=0, sticky=W)
    frame_butt_vali.update()
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame for parameters for the plot
    frame_adapt_grid_data_param_plot = tk.Frame(dico_interface_tk['tab_adapt_grid_data'],
                                                width=frame_param_adapt_grid_data.winfo_width())
    dico_interface_tk['frame_adapt_grid_data_param_plot'] = frame_adapt_grid_data_param_plot
    y_place = y_place + 20 + frame_butt_vali.winfo_height()
    frame_adapt_grid_data_param_plot.place(x=0, y=y_place, anchor=NW)
    from ForMoSA.interface.tabcontrol.utilities import define_xlim_ylim
    define_xlim_ylim(dico_interface_tk, 'frame_adapt_grid_data_param_plot')
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of xlim and ylim
    dico_interface_tk['frame_axe_xy_lim_adapt'].update()
    canvas = Canvas(dico_interface_tk['tab_adapt_grid_data'],
                    width=dico_interface_tk['frame_axe_xy_lim_adapt'].winfo_width(), height=1, bg='black')
    canvas.place(x=0, y=y_place-10, anchor=W)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of le plot
    from ForMoSA.interface.tabcontrol.adapt_grid_data.plot import plot_adapt
    plot_adapt(dico_interface_tk, frame_param_adapt_grid_data.winfo_width() + 25, 0)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the logo
    import os
    import ForMoSA
    path_logo = os.path.abspath(ForMoSA.__file__)
    path_logo = path_logo.split('__init__.py')
    path_logo = path_logo[0] + 'interface/utilities/formosa_logo_small.png'
    logo_tk = ImageTk.PhotoImage(master=dico_interface_tk['tab_path_selection'], file=path_logo)
    logo_label = tk.Label(dico_interface_tk['tab_adapt_grid_data'], image=logo_tk)
    logo_label.image = logo_tk
    logo_label.place(x=0, y=608, anchor=SW)

    validate_adapt_grid_data(dico_interface_tk, ini='yes')
