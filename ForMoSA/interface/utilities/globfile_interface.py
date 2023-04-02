from configobj import ConfigObj


class GlobFile:
    '''Import config variables and make them GLOBAL FORMOSA VARIABLES

    Author: Paulina Palma-Bifani'''

    def __init__(self, config_file_path):
        # Generate the confog object
        config = ConfigObj(config_file_path, encoding='utf8')
        # self.obsname = obsname
        self.config = config

        ## Read CONFIG:
        # [config_path] (4)
        self.observation_path = config['config_path']['observation_path']
        self.adapt_store_path = config['config_path']['adapt_store_path']
        self.result_path = config['config_path']['result_path']
        self.model_path = config['config_path']['model_path']
        grid_name = config['config_path']['model_path'].split('/')
        grid_name = grid_name[-1]
        grid_name = grid_name.split('.nc')
        grid_name = grid_name[0]
        self.grid_name = grid_name
        model_name = grid_name.split('_')
        model_name = model_name[0]
        self.model_name = model_name

        # [config_adapt] (5)
        self.wav_for_adapt = config['config_adapt']['wav_for_adapt']
        self.adapt_method = config['config_adapt']['adapt_method']
        self.custom_reso = config['config_adapt']['custom_reso']
        self.continuum_sub = config['config_adapt']['continuum_sub']
        self.wav_for_continuum = config['config_adapt']['wav_for_continuum']

        # [config_inversion] (3)
        self.wav_fit = config['config_inversion']['wav_fit']
        self.ns_algo = config['config_inversion']['ns_algo']
        self.npoint = config['config_inversion']['npoint']

        # [config_parameter] (11)
        self.par1 = config['config_parameter']['par1']
        self.par2 = config['config_parameter']['par2']
        self.par3 = config['config_parameter']['par3']
        self.par4 = config['config_parameter']['par4']
        self.par5 = config['config_parameter']['par5']
        self.r = config['config_parameter']['r']
        self.d = config['config_parameter']['d']
        self.rv = config['config_parameter']['rv']
        self.av = config['config_parameter']['av']
        self.vsini = config['config_parameter']['vsini']
        self.ld = config['config_parameter']['ld']

        # # [config_nestle] (10 but 3 relevant)  (n_ prefix for params)
        # self.n_mechanic = config['config_nestle']['mechanic']
        # self.n_method = config['config_nestle']['method']
        # self.n_maxiter = int(config['config_nestle']['maxiter'])
        # self.n_maxcall = eval(config['config_nestle']['maxcall'])
        # self.n_dlogz = eval(config['config_nestle']['dlogz'])
        # self.n_decline_factor = eval(config['config_nestle']['decline_factor'])
        # self.n_update_interval = eval(config['config_nestle']['update_interval'])
        # self.n_npdim = eval(config['config_nestle']['npdim'])
        # self.n_rstate = eval(config['config_nestle']['rstate'])

        # [config_dinesty] & [config_ultranest] CHECK THIS

        # ## create OUTPUTS Sub-Directories: interpolated grids and results
        # stock_result_sub_dir = self.stock_result_raw + self.name_obs + '_' + self.model_name[
        #                                                                      :-4] + self.data_type  # sub_directory: obsname_grid_datatype
        # self.stock_interp_grid = stock_result_sub_dir + '/interp_grid'  # sub_sub directory to save interp grid (one interpolation for grid and data type, full wavelength covarage)
        #
        # self.path_grid_management = self.base_path + stock_result_sub_dir + '/interp_grid/grid_management'
        # os.makedirs(self.base_path + stock_result_sub_dir + '/interp_grid' + '/grid_management', exist_ok=True)
        #
        # subsub_dir_name = self.nest_samp_algo + '_' + self.NS_using_band + '_Res' + str(
        #     self.R_by_wl[2]) + '_'  # subsub_directory: nestle_band_ResOBS_params_date
        # subsub_dir_name = subsub_dir_name + self.free_params + '_t' + time.strftime("%Y%m%d_%H%M%S")
        # stock_result_subsub_dir = stock_result_sub_dir + '/' + subsub_dir_name
        # self.stock_result = self.base_path + stock_result_subsub_dir
        # os.makedirs(self.base_path + stock_result_subsub_dir)

        # ## Save CONFIG file with updated params for current run in OUTPUT subsub directory
        #
        # print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        # print('-> Saving new configuration')
        # print()
        #
        # config_current = self.result_path + '/past_config.ini'
        # config.filename = config_current
        # config['config_path']['stock_interp_grid'] = stock_interp_grid
        # config['config_path']['stock_result'] = stock_result_subsub_dir
        # config.write()
        #
        # print('Saved config: --- ' + config_current + ' ---')
        # print()
