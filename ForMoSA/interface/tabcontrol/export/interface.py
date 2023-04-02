import tkinter as tk
from tkinter import *
from PIL import ImageTk


def export_interface(dico_interface_tk):

    # Frame of the check and import
    frame_check_import = tk.Frame(dico_interface_tk['tab_export'])
    dico_interface_tk['frame_check_import'] = frame_check_import
    frame_check_import.place(x=0, y=0, anchor=NW)

    from ForMoSA.interface.tabcontrol.export.utilities import export_check
    export_check(dico_interface_tk)
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
