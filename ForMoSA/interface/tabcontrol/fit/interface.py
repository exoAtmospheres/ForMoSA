import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import ImageTk


def fit_interface(dico_interface_tk):

    # Frame of the parameters
    frame_fit = tk.Frame(dico_interface_tk['tab_fit'])
    frame_fit.place(x=0, y=0, anchor=NW)
    frame_param_fit = tk.Frame(frame_fit)
    dico_interface_tk['frame_param_fit'] = frame_param_fit
    frame_param_fit.grid(row=1, column=0)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame for the selection of the nested sampling algorithm
    from ForMoSA.interface.tabcontrol.fit.utilities import switch_ns_algo
    switch_ns_algo(dico_interface_tk)
    ttk.Separator(frame_param_fit, orient=HORIZONTAL).grid(row=1, padx=0, pady=5, sticky='ew')
    # Frame for the definition of the wavelength range used for the fit
    from ForMoSA.interface.tabcontrol.fit.utilities import define_wav_fit
    define_wav_fit(dico_interface_tk)
    ttk.Separator(frame_param_fit, orient=HORIZONTAL).grid(row=3, padx=0, pady=5, sticky='ew')
    # Frame for the definition of the number of living points used for the fit
    from ForMoSA.interface.tabcontrol.fit.utilities import define_npoint
    define_npoint(dico_interface_tk)
    frame_param_fit.update()
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the button to check all
    frame_butt_vali = tk.Frame(dico_interface_tk['tab_fit'])
    y_place = 10 + frame_param_fit.winfo_height()
    frame_butt_vali.place(x=0, y=y_place, anchor=NW)
    from ForMoSA.interface.tabcontrol.fit.utilities import validate_fit
    button_validate = tk.Button(frame_butt_vali, text='Check all', font='Arial 13 bold', fg='green',
                                command=lambda: validate_fit(dico_interface_tk))
    button_validate.grid(row=0, column=0, sticky=W)
    frame_butt_vali.update()
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame for parameters for the plot
    frame_fit_param_plot = tk.Frame(dico_interface_tk['tab_fit'])
    dico_interface_tk['frame_fit_param_plot'] = frame_fit_param_plot
    y_place = y_place + 20 + frame_butt_vali.winfo_height()
    frame_fit_param_plot.place(x=0, y=y_place, anchor=NW)
    from ForMoSA.interface.tabcontrol.utilities import define_grid_parameter_to_plot
    define_grid_parameter_to_plot(dico_interface_tk, 'frame_fit_param_plot')
    frame_fit_param_plot.update()
    ttk.Separator(frame_fit_param_plot, orient=HORIZONTAL).grid(row=1, column=0, pady=5, sticky='ew')
    # Frame of xlim and ylim
    from ForMoSA.interface.tabcontrol.utilities import define_xlim_ylim
    define_xlim_ylim(dico_interface_tk, 'frame_fit_param_plot')
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Separator
    canvas = Canvas(dico_interface_tk['tab_fit'], width=frame_fit_param_plot.winfo_width(),
                    height=1, bg='black')
    canvas.place(x=0, y=y_place-10, anchor=W)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of le plot
    from ForMoSA.interface.tabcontrol.fit.plot import plot_fit
    plot_fit(dico_interface_tk, frame_param_fit.winfo_width() + 25, 0)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the logo
    import os
    import ForMoSA
    path_logo = os.path.abspath(ForMoSA.__file__)
    path_logo = path_logo.split('__init__.py')
    path_logo = path_logo[0] + 'interface/utilities/formosa_logo_small.png'
    logo_tk = ImageTk.PhotoImage(master=dico_interface_tk['tab_path_selection'], file=path_logo)
    logo_label = tk.Label(dico_interface_tk['tab_fit'], image=logo_tk)
    logo_label.image = logo_tk
    logo_label.place(x=0, y=608, anchor=SW)

    validate_fit(dico_interface_tk, ini='yes')
