import tkinter as tk
from tkinter import *
from PIL import ImageTk


def grid_priors_interface(dico_interface_tk):

    # Frame of the parameters
    frame_grid_priors = tk.Frame(dico_interface_tk['tab_grid_priors'])
    dico_interface_tk['frame_grid_priors'] = frame_grid_priors
    frame_grid_priors.place(x=0, y=0, anchor=NW)
    from ForMoSA.interface.tabcontrol.grid_priors.utilities import define_grid_prior_box
    define_grid_prior_box(dico_interface_tk)
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the logo
    import os
    import ForMoSA
    path_logo = os.path.abspath(ForMoSA.__file__)
    path_logo = path_logo.split('__init__.py')
    path_logo = path_logo[0] + 'interface/utilities/formosa_logo_small.png'
    logo_tk = ImageTk.PhotoImage(master=dico_interface_tk['tab_path_selection'], file=path_logo)
    logo_label = tk.Label(dico_interface_tk['tab_grid_priors'], image=logo_tk)
    logo_label.image = logo_tk
    logo_label.place(x=0, y=608, anchor=SW)
    logo_label.update()
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # Frame of the button to check all
    frame_butt_vali = tk.Frame(dico_interface_tk['tab_grid_priors'])
    x_place = 20 + logo_label.winfo_width()
    y_place = 608 - logo_label.winfo_height()/2
    frame_butt_vali.place(x=x_place, y=y_place, anchor=W)
    from ForMoSA.interface.tabcontrol.grid_priors.utilities import validate_grid_prior
    button_validate = tk.Button(frame_butt_vali, text='Check all', font='Arial 13 bold', fg='green',
                                command=lambda: validate_grid_prior(dico_interface_tk))
    button_validate.grid(row=0, column=0, sticky=W)
