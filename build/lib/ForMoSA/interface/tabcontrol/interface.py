import tkinter as tk
from tkinter import *
from tkinter import ttk

from ForMoSA.interface.tabcontrol.path_selection.interface import path_selection_interface


def tabcontrol_interface(dico_interface_tk):

    dico_interface_tk['root_tabcontrol'] = tk.Tk()
    screenwidth = dico_interface_tk['root_tabcontrol'].winfo_screenwidth()
    screenheight = dico_interface_tk['root_tabcontrol'].winfo_screenheight()
    windwidth = 1182
    windheight = 670
    d_width = int((screenwidth - windwidth) / 2)
    d_height = int((screenheight - windheight) / 2)
    dico_interface_tk['root_tabcontrol'].geometry(str(windwidth) + "x" + str(windheight) + "+" +
                                                  str(d_width) + "+" + str(d_height))
    dico_interface_tk['root_tabcontrol'].title('configuration_file.ini maker')

    frame_tabcontrol = tk.Frame(dico_interface_tk['root_tabcontrol'])
    frame_tabcontrol.grid(row=0, column=0, sticky=W)

    tabcontrol = ttk.Notebook(frame_tabcontrol)
    tab_path_selection = ttk.Frame(tabcontrol, width=windwidth-50, height=windheight-60)
    tabcontrol.add(tab_path_selection, text='Path selection')
    dico_interface_tk['tab_path_selection'] = tab_path_selection
    path_selection_interface(dico_interface_tk)

    tab_adapt_grid_data = ttk.Frame(tabcontrol, width=windwidth-50, height=windheight-60)
    tabcontrol.add(tab_adapt_grid_data, text='Adapt grid-data')
    dico_interface_tk['tab_adapt_grid_data'] = tab_adapt_grid_data
    from ForMoSA.interface.tabcontrol.adapt_grid_data.interface import adapt_grid_data_interface
    adapt_grid_data_interface(dico_interface_tk)

    tab_fit = ttk.Frame(tabcontrol, width=windwidth-50, height=windheight-60)
    tabcontrol.add(tab_fit, text='Inversion')
    dico_interface_tk['tab_fit'] = tab_fit
    from ForMoSA.interface.tabcontrol.fit.interface import fit_interface
    fit_interface(dico_interface_tk)

    tab_grid_priors = ttk.Frame(tabcontrol, width=windwidth-50, height=windheight-60)
    tabcontrol.add(tab_grid_priors, text="Grid's priors")
    dico_interface_tk['tab_grid_priors'] = tab_grid_priors
    from ForMoSA.interface.tabcontrol.grid_priors.interface import grid_priors_interface
    grid_priors_interface(dico_interface_tk)

    tab_extra_grid_priors = ttk.Frame(tabcontrol, width=windwidth-50, height=windheight-60)
    tabcontrol.add(tab_extra_grid_priors, text="Extra grid's priors")
    dico_interface_tk['tab_extra_grid_priors'] = tab_extra_grid_priors
    from ForMoSA.interface.tabcontrol.extra_grid_priors.interface import extra_grid_priors_interface
    extra_grid_priors_interface(dico_interface_tk)

    tab_export = ttk.Frame(tabcontrol, width=windwidth-50, height=windheight-60)
    tabcontrol.add(tab_export, text='Export')
    dico_interface_tk['tab_export'] = tab_export
    from ForMoSA.interface.tabcontrol.export.interface import export_interface
    export_interface(dico_interface_tk)

    tabcontrol.pack(expand=1, fill="both")

    dico_interface_tk['root_tabcontrol'].wait_window(dico_interface_tk['root_config_file_path'])
