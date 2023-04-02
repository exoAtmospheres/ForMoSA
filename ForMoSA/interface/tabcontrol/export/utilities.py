import tkinter as tk
from tkinter import *
import json


def export_check(dico_interface_tk):

    frame_check_export_button = tk.Frame(dico_interface_tk['tab_export'])
    frame_check_export_button.place(x=30, y=30, anchor=NW)
    frame_check_export_button_title = tk.Label(frame_check_export_button,
                                               text='    To export your configuration_file.ini:',
                                               font='Arial 20 bold')
    frame_check_export_button_title.grid(row=0, column=0, sticky=E)

    button_check_export_button = tk.Button(frame_check_export_button, text='Exportation', font='Arial 15 bold',
                                           command=lambda: export_check_button(dico_interface_tk), height=3, width=30)
    button_check_export_button.grid(row=0, column=1, padx=30, sticky=E)


def export_check_button(dico_interface_tk):

    with open(dico_interface_tk['config_file_path_folder'] + 'dico_interface_param.json') as dico:
        dico_interface_param = json.load(dico)

    from ForMoSA.interface.tabcontrol.path_selection.utilities import validate_path_selection
    validate_path_selection(dico_interface_tk, ini='yes')
    from ForMoSA.interface.tabcontrol.adapt_grid_data.utilities import validate_adapt_grid_data
    validate_adapt_grid_data(dico_interface_tk, ini='yes')
    from ForMoSA.interface.tabcontrol.fit.utilities import validate_fit
    validate_fit(dico_interface_tk, ini='yes')
    from ForMoSA.interface.tabcontrol.grid_priors.utilities import validate_grid_prior
    validate_grid_prior(dico_interface_tk)
    from ForMoSA.interface.tabcontrol.extra_grid_priors.utilities import validate_extra_grid_prior
    validate_extra_grid_prior(dico_interface_tk)

    # with open(dico_interface_tk['config_file_path_folder'] + 'test.ini', 'w') as file_o:
    with open(dico_interface_tk['config_file_path'], 'w') as file_o:
        file_o.write("[config_path]\n")
        file_o.write("\t\t\t# Path to the observed spectrum file\n")
        file_o.write("\t\t\tobservation_path = " + dico_interface_param['observation_path'] + "\n")
        file_o.write("\n")
        file_o.write("\t\t\t# Path to store your interpolated grid\n")
        file_o.write("\t\t\tadapt_store_path = " + dico_interface_param['adapt_store_path'] + "\n")
        file_o.write("\n")
        file_o.write("\t\t\t# Path where you wish to store your results\n")
        file_o.write("\t\t\tresult_path = " + dico_interface_param['result_path'] + "\n")
        file_o.write("\n")
        file_o.write("\t\t\t# Path of your initial grid of models\n")
        file_o.write("\t\t\tmodel_path = " + dico_interface_param['model_path'] + "\n")
        file_o.write("\n")

        file_o.write("[config_adapt]\n")
        file_o.write("\t\t\t# Wavelength range used for the extraction of data and adaptation of the grid "
                     "(separated windows can be defined)\n")
        file_o.write("\t\t\t# Format : 'window1_min,window1_max / window2_min,window2_max / ... / "
                     "windown_min,windown_max'\n")
        file_o.write("\t\t\twav_for_adapt = '" + dico_interface_param['wav_for_adapt'] + "'\n")
        file_o.write("\n")
        file_o.write("\t\t\t# Method used to adapt the synthetic spectra to the data\n")
        file_o.write("\t\t\t# example : 'by_reso' : A Gaussian law is convolved to the spectrum to decrease the "
                     "resolution as a function of the\n")
        file_o.write("\t\t\t# wavelength\n")
        file_o.write("\t\t\t#           'by_sample' : The spectrum is directly re-sampled to the "
                     "wavelength grid of the data, using the module\n")
        file_o.write("\t\t\t# python spectres\n")
        file_o.write("\t\t\tadapt_method = " + dico_interface_param['adapt_method'] + "\n")
        file_o.write("\n")
        file_o.write("\t\t\t# Custom target resolution to reach (optional). The final resolution will be the "
                     "lowest between this custom resolution,\n")
        file_o.write("\t\t\t# the resolution of the data, and the resolution of the models, for each wavelength.\n")
        file_o.write("\t\t\t# Format : float or 'NA'\n")
        file_o.write("\t\t\tcustom_reso = " + dico_interface_param['custom_reso'] + "\n")
        file_o.write("\n")
        file_o.write("\t\t\t# Continuum subtraction. If a float is given, the value will give the "
                     "approximated spectral resolution of the continuum.\n")
        file_o.write("\t\t\t# Format : 'float' or 'NA'\n")
        file_o.write("\t\t\tcontinuum_sub = " + dico_interface_param['continuum_sub'] + "\n")
        file_o.write("\n")
        file_o.write("\t\t\t# Wavelength range used for the estimate of the continuum "
                     "(separated windows can be defined).\n")
        file_o.write("\t\t\t# Format : 'window1_max / window2_min,window2_max / ... / windown_min'\n")
        file_o.write("\t\t\t# Note : the 'window1_min' and 'windown_max' are defined as the edges of "
                     "'wav_for_adapt'.\n")
        file_o.write("\t\t\t# If the entire wavelength range defined by the parameter 'wav_for_adapt' "
                     "needs to be continuum-subtracted: 'full'\n")
        file_o.write("\t\t\twav_for_continuum = '" + dico_interface_param['wav_for_continuum'] + "'\n")
        file_o.write("\n")

        file_o.write("[config_inversion]\n")
        file_o.write("\t\t\t# Wavelength range used for the fit during the nested sampling procedure "
                     "(separated windows can be defined).\n")
        file_o.write("\t\t\t# Format : 'window1_min,window1_max / window2_min,window2_max / ... / "
                     "windown_min,windown_max'\n")
        file_o.write("\t\t\twav_fit = '" + dico_interface_param['wav_fit'] + "'\n")
        file_o.write("\n")
        file_o.write("\t\t\t# Nested sampling algorithm used.\n")
        file_o.write("\t\t\t# Format : 'nestle' or 'dynesty' or 'ultranest' (/!\ only nestle currently /!\)\n")
        file_o.write("\t\t\tns_algo = " + dico_interface_param['ns_algo'] + "\n")
        file_o.write("\n")
        file_o.write("\t\t\t# Number of living points during the nested sampling procedure.\n")
        file_o.write("\t\t\t# Format : 'integer'\n")
        file_o.write("\t\t\tnpoint = " + str(dico_interface_param['npoint']) + "\n")
        file_o.write("\n")

        file_o.write("[config_parameter]\n")
        file_o.write("\t\t\t# Definition of the prior function of each parameter explored by the grid. "
                     "Please refer to the documentation to check\n")
        file_o.write("\t\t\t# the parameter space explore by each grid.\n")
        file_o.write("\t\t\t# Format : 'function', function_param1, function_param2\n")
        file_o.write("\t\t\t# Example : 'uniform', min, max\n")
        file_o.write("\t\t\t#           'constant', value\n")
        file_o.write("\t\t\t#           'gaussian', mu, sigma\n")
        file_o.write("\t\t\t#           'NA' if the grid cover a lower number of parameters\n")

        if dico_interface_param['par1'] == 'NA':
            par1_export = 'NA'
        else:
            par1_export = ''
            for par1_ind, par1_interface in enumerate(dico_interface_param['par1']):
                if par1_ind == 0:
                    par1_export += par1_interface
                else:
                    for par1_export_2 in par1_interface:
                        par1_export += ', ' + par1_export_2
        if dico_interface_param['par2'] == 'NA':
            par2_export = 'NA'
        else:
            par2_export = ''
            for par2_ind, par2_interface in enumerate(dico_interface_param['par2']):
                if par2_ind == 0:
                    par2_export += par2_interface
                else:
                    for par2_export_2 in par2_interface:
                        par2_export += ', ' + par2_export_2
        if dico_interface_param['par3'] == 'NA':
            par3_export = 'NA'
        else:
            par3_export = ''
            for par3_ind, par3_interface in enumerate(dico_interface_param['par3']):
                if par3_ind == 0:
                    par3_export += par3_interface
                else:
                    for par3_export_2 in par3_interface:
                        par3_export += ', ' + par3_export_2
        if dico_interface_param['par4'] == 'NA':
            par4_export = 'NA'
        else:
            par4_export = ''
            for par4_ind, par4_interface in enumerate(dico_interface_param['par4']):
                if par4_ind == 0:
                    par4_export += par4_interface
                else:
                    for par4_export_2 in par4_interface:
                        par4_export += ', ' + par4_export_2
        if dico_interface_param['par5'] == 'NA':
            par5_export = 'NA'
        else:
            par5_export = ''
            for par5_ind, par5_interface in enumerate(dico_interface_param['par5']):
                if par5_ind == 0:
                    par5_export += par5_interface
                else:
                    for par5_export_2 in par5_interface:
                        par5_export += ', ' + par5_export_2

        file_o.write("\t\t\tpar1 = " + par1_export + "\n")
        file_o.write("\t\t\tpar2 = " + par2_export + "\n")
        file_o.write("\t\t\tpar3 = " + par3_export + "\n")
        file_o.write("\t\t\tpar4 = " + par4_export + "\n")
        file_o.write("\t\t\tpar5 = " + par5_export + "\n")
        file_o.write("\n")
        file_o.write("\t\t\t# Definition of the prior function of each extra-grid parameter. r is the "
                     "radius (MJup, >0), d is the distance (pc, >0),\n")
        file_o.write("\t\t\t# rv is the radial velocity (km.s-1), av is the extinction (mag), vsini is the "
                     "projected rotational velocity (km.s-1, >0),\n")
        file_o.write("\t\t\t# and ld is the limb darkening (0-1).\n")
        file_o.write("\t\t\t# Format : 'function', function_param1, function_param2\n")
        file_o.write("\t\t\t# Example : 'uniform', min, max\n")
        file_o.write("\t\t\t#           'constant', value\n")
        file_o.write("\t\t\t#           'gaussian', mu, sigma\n")
        file_o.write("\t\t\t#           'NA' if you do not want to constrain this parameter\n")
        if dico_interface_param['r'] == 'NA':
            parr_export = 'NA'
        else:
            parr_export = ''
            for parr_ind, parr_interface in enumerate(dico_interface_param['r']):
                if parr_ind == 0:
                    parr_export += parr_interface
                else:
                    for parr_export_2 in parr_interface:
                        parr_export += ', ' + parr_export_2
        if dico_interface_param['d'] == 'NA':
            pard_export = 'NA'
        else:
            pard_export = ''
            for pard_ind, pard_interface in enumerate(dico_interface_param['d']):
                if pard_ind == 0:
                    pard_export += pard_interface
                else:
                    for pard_export_2 in pard_interface:
                        pard_export += ', ' + pard_export_2
        if dico_interface_param['rv'] == 'NA':
            parrv_export = 'NA'
        else:
            parrv_export = ''
            for parrv_ind, parrv_interface in enumerate(dico_interface_param['rv']):
                if parrv_ind == 0:
                    parrv_export += parrv_interface
                else:
                    for parrv_export_2 in parrv_interface:
                        parrv_export += ', ' + parrv_export_2
        if dico_interface_param['av'] == 'NA':
            parav_export = 'NA'
        else:
            parav_export = ''
            for parav_ind, parav_interface in enumerate(dico_interface_param['av']):
                if parav_ind == 0:
                    parav_export += parav_interface
                else:
                    for parav_export_2 in parav_interface:
                        parav_export += ', ' + parav_export_2
        if dico_interface_param['vsini'] == 'NA':
            parvsini_export = 'NA'
        else:
            parvsini_export = ''
            for parvsini_ind, parvsini_interface in enumerate(dico_interface_param['vsini']):
                if parvsini_ind == 0:
                    parvsini_export += parvsini_interface
                else:
                    for parvsini_export_2 in parvsini_interface:
                        parvsini_export += ', ' + parvsini_export_2
        if dico_interface_param['ld'] == 'NA':
            parld_export = 'NA'
        else:
            parld_export = ''
            for parld_ind, parld_interface in enumerate(dico_interface_param['ld']):
                if parld_ind == 0:
                    parld_export += parld_interface
                else:
                    for parld_export_2 in parld_interface:
                        parld_export += ', ' + parld_export_2
        file_o.write("\t\t\tr = " + parr_export + "\n")
        file_o.write("\t\t\td = " + pard_export + "\n")
        file_o.write("\t\t\trv = " + parrv_export + "\n")
        file_o.write("\t\t\tav = " + parav_export + "\n")
        file_o.write("\t\t\tvsini = " + parvsini_export + "\n")
        file_o.write("\t\t\tld = " + parld_export + "\n")
        file_o.write("\n")

        file_o.write("[config_nestle]\n")
        file_o.write("\t\t\t# For details on these parameters, please see: http://kylebarbary.com/nestle/index.html\n")
        file_o.write("\n")
        file_o.write("\t\t\tmechanic = 'static'  # Sampler “super-classes” of dynesty\n")
        file_o.write("\t\t\t# e.g. 'static' / 'dynamic' # the number of living point is fixed / variates\n")
        file_o.write("\n")
        file_o.write("\t\t\tmethod = 'multi'  # Reduction of the parameters space\n")
        file_o.write("\t\t\t# e.g. 'single' / 'multi' #single-ellipsoidal / multi-ellipsoidal\n")
        file_o.write("\n")
        file_o.write("\t\t\tmaxiter = None  # Stopping criterions\n")
        file_o.write("\t\t\tmaxcall = None\n")
        file_o.write("\t\t\tdlogz = None\n")
        file_o.write("\t\t\tdecline_factor = 0.1\n")
        file_o.write("\n")
        file_o.write("\t\t\tupdate_interval = None  # Divers\n")
        file_o.write("\t\t\tnpdim = None\n")
        file_o.write("\t\t\trstate = None\n")
        file_o.write("\t\t\tcallback = None\n")
        file_o.write("\n")

        file_o.write("[config_dynesty]\n")
        file_o.write("\t\t\t# For details please see: https://dynesty.readthedocs.io/en/latest/api.html and "
                     "https://dynesty.readthedocs.io/en/latest/api.html\n")
        file_o.write("\n")
        file_o.write("\t\t\tmechanic = 'dynamic'  # Sampler “super-classes” of dynesty\n")
        file_o.write("\t\t\t# e.g. 'static' / 'dynamic'  # the number of living point is fixed / variates\n")
        file_o.write("\n")
        file_o.write("\t\t\tbound = 'multi'  # Bounding Options\n")
        file_o.write("\t\t\tbootstrap = 0\n")
        file_o.write("\t\t\tenlarge = None\n")
        file_o.write("\t\t\tupdate_interval = None\n")
        file_o.write("\n")
        file_o.write("\t\t\tsample = 'auto'  # Sampling Options\n")
        file_o.write("\t\t\tfirst_update = None  # Early-Time Behavior\n")
        file_o.write("\n")
        file_o.write("\t\t\tperiodic = None  # Special Boundary Conditions\n")
        file_o.write("\t\t\treflective = None\n")
        file_o.write("\n")
        file_o.write("\t\t\tpool = None  # Parallelization\n")
        file_o.write("\t\t\tqueue_size = None\n")
        file_o.write("\t\t\tuse_pool = None\n")
        file_o.write("\n")
        file_o.write("\t\t\tmaxiter = None  # Running Internally (static)\n")
        file_o.write("\t\t\tmaxcall = None\n")
        file_o.write("\t\t\tdlogz = None\n")
        file_o.write("\t\t\tlogl_max = inf\n")
        file_o.write("\t\t\tn_effective = inf\n")
        file_o.write("\t\t\tadd_live = True\n")
        file_o.write("\t\t\tprint_progress = True\n")
        file_o.write("\t\t\tprint_func = None\n")
        file_o.write("\t\t\tsave_bounds = True\n")
        file_o.write("\n")
        file_o.write("\t\t\tmaxiter_init = None  # Running Internally (dynamic)\n")
        file_o.write("\t\t\tmaxcall_init = None\n")
        file_o.write("\t\t\tdlogz_init = 0.01\n")
        file_o.write("\t\t\tlogl_max_init = inf\n")
        file_o.write("\t\t\tn_effective_init = inf\n")
        file_o.write("\t\t\tnlive_batch = 500\n")
        file_o.write("\t\t\twt_function = None\n")
        file_o.write("\t\t\twt_kwargs = None\n")
        file_o.write("\t\t\tmaxiter_batch = None\n")
        file_o.write("\t\t\tmaxcall_batch = None\n")
        file_o.write("\t\t\tmaxbatch = None\n")
        file_o.write("\t\t\tstop_function = None\n")
        file_o.write("\t\t\tstop_kwargs = None\n")
        file_o.write("\t\t\tuse_stop = True\n")
        file_o.write("\n")
        file_o.write("\t\t\tnpdim = None  # DIVERS\n")
        file_o.write("\t\t\trstate = None\n")
        file_o.write("\t\t\tlive_points = None\n")
        file_o.write("\t\t\tlogl_args = None\n")
        file_o.write("\t\t\tlogl_kwargs = None\n")
        file_o.write("\t\t\tptform_args = None\n")
        file_o.write("\t\t\tptform_kwargs = None\n")
        file_o.write("\t\t\tgradient = None\n")
        file_o.write("\t\t\tgrad_args = None\n")
        file_o.write("\t\t\tgrad_kwargs = None\n")
        file_o.write("\t\t\tcompute_jac = False\n")
        file_o.write("\t\t\tvol_dec = 0.5\n")
        file_o.write("\t\t\tvol_check = 2.0\n")
        file_o.write("\t\t\twalks = 25\n")
        file_o.write("\t\t\tfacc = 0.5\n")
        file_o.write("\t\t\tslices = 5\n")
        file_o.write("\t\t\tfmove = 0.9\n")
        file_o.write("\t\t\tmax_move = 100\n")
        file_o.write("\t\t\tupdate_func = None\n")
        file_o.write("\n")

        file_o.write("[config_ultranest]\n")
        file_o.write("\t\t\t# For details: https://dynesty.readthedocs.io/en/latest/api.html and "
                     "https://dynesty.readthedocs.io/en/latest/api.html\n")
        file_o.write("\n")
        file_o.write("\t\t\tmechanic = 'dynamic'  # Sampler “super-classes” of dynesty\n")
        file_o.write("\t\t\t# e.g. 'static' / 'dynamic' # the number of living point is fixed / variates\n")
        file_o.write("\n")
        file_o.write("\t\t\tbound = 'multi'  # Bounding Options.\n")
        file_o.write("\t\t\tbootstrap = 0\n")
        file_o.write("\t\t\tenlarge = None\n")
        file_o.write("\t\t\tupdate_interval = None\n")
        file_o.write("\n")
        file_o.write("\t\t\tsample = 'auto'  # Sampling Options\n")
        file_o.write("\t\t\tfirst_update = None  # Early-Time Behavior\n")
        file_o.write("\n")
        file_o.write("\t\t\tperiodic = None  # Special Boundary Conditions\n")
        file_o.write("\t\t\treflective = None\n")
        file_o.write("\n")
        file_o.write("\t\t\tpool = None  # Parallelization\n")
        file_o.write("\t\t\tqueue_size = None\n")
        file_o.write("\t\t\tuse_pool = None\n")
        file_o.write("\n")
        file_o.write("\t\t\tmaxiter = None  # Running Internally (static)\n")
        file_o.write("\t\t\tmaxcall = None\n")
        file_o.write("\t\t\tdlogz = None\n")
        file_o.write("\t\t\tlogl_max = inf\n")
        file_o.write("\t\t\tn_effective = inf\n")
        file_o.write("\t\t\tadd_live = True\n")
        file_o.write("\t\t\tprint_progress = True\n")
        file_o.write("\t\t\tprint_func = None\n")
        file_o.write("\t\t\tsave_bounds = True\n")
        file_o.write("\n")
        file_o.write("\t\t\tmaxiter_init = None  # Running Internally (dynamic)\n")
        file_o.write("\t\t\tmaxcall_init = None\n")
        file_o.write("\t\t\tdlogz_init = 0.01\n")
        file_o.write("\t\t\tlogl_max_init = inf\n")
        file_o.write("\t\t\tn_effective_init = inf\n")
        file_o.write("\t\t\tnlive_batch = 500\n")
        file_o.write("\t\t\twt_function = None\n")
        file_o.write("\t\t\twt_kwargs = None\n")
        file_o.write("\t\t\tmaxiter_batch = None\n")
        file_o.write("\t\t\tmaxcall_batch = None\n")
        file_o.write("\t\t\tmaxbatch = None\n")
        file_o.write("\t\t\tstop_function = None\n")
        file_o.write("\t\t\tstop_kwargs = None\n")
        file_o.write("\t\t\tuse_stop = True\n")
        file_o.write("\n")
        file_o.write("\t\t\tnpdim = None  # DIVERS\n")
        file_o.write("\t\t\trstate = None\n")
        file_o.write("\t\t\tlive_points = None\n")
        file_o.write("\t\t\tlogl_args = None\n")
        file_o.write("\t\t\tlogl_kwargs = None\n")
        file_o.write("\t\t\tptform_args = None\n")
        file_o.write("\t\t\tptform_kwargs = None\n")
        file_o.write("\t\t\tgradient = None\n")
        file_o.write("\t\t\tgrad_args = None\n")
        file_o.write("\t\t\tgrad_kwargs = None\n")
        file_o.write("\t\t\tcompute_jac = False\n")
        file_o.write("\t\t\tvol_dec = 0.5\n")
        file_o.write("\t\t\tvol_check = 2.0\n")
        file_o.write("\t\t\twalks = 25\n")
        file_o.write("\t\t\tfacc = 0.5\n")
        file_o.write("\t\t\tslices = 5\n")
        file_o.write("\t\t\tfmove = 0.9\n")
        file_o.write("\t\t\tmax_move = 100\n")
        file_o.write("\t\t\tupdate_func = None\n")

    # Save the path of the previous configuration file modified
    import os
    import ForMoSA
    path_previous_configuration_file = os.path.abspath(ForMoSA.__file__)
    path_previous_configuration_file = path_previous_configuration_file.split('__init__.py')
    path_previous_configuration_file = path_previous_configuration_file[0] + 'interface/utilities/' \
                                                                             'dico_previous_configuration_file.json'
    dico_previous_configuration_file = {'previous_configuration_file': dico_interface_tk['config_file_path_folder']}
    with open(path_previous_configuration_file, 'w') as dico:
        json.dump(dico_previous_configuration_file, dico)

    # from ForMoSA.plotting.plotting_class import PlottingForMoSA
    # import matplotlib.pyplot as plt
    # plotForMoSA = PlottingForMoSA(dico_interface_tk['config_file_path'], 'magenta')
    # plotForMoSA.plot_corner(levels_sig=[0.997, 0.95, 0.68], bins=100, quantiles=(0.16, 0.5, 0.84), burn_in=0)
    # plt.show()
    #
    # plotForMoSA.plot_chains()
    # plt.show()
    #
    # plotForMoSA.plot_radar()
    # plt.show()
    #
    # plotForMoSA._get_spectra()
    # plt.show()