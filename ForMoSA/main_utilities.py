from configobj import ConfigObj
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
def yesno(text):
    '''
    Function to interact with the terminal and decide for different options when running ForMoSA (Loop to repeat question if answer is different to 'y' or 'n).

    Args:
        text    (str): (y/n) answer in the terminall in interactive mode
    Returns:
        asw     (str): answer y or n

    Author: Simon Petrus
    '''
    print(text)
    asw = input()
    if asw in ['y', 'n']:
        return asw
    else:
        return yesno()

# ----------------------------------------------------------------------------------------------------------------------
def diag_mat(rem=[], result=np.empty((0, 0))):
    '''
    Function to concatenate and align iterativly block matrices (usefull during the extraction and the inversion).

    Args:
        rem        (list): matrices to be add iterativly (use diag([mat1, mat2]))
        result    (array): final array with each sub-matrices aligned allong the diagonal
    Returns:
        diag_mat (matrix): Generated diagonal matrix
        (If rem input is empty, it wull return an empy array)

    Author : Ishigoya, Stack-overflow : https://stackoverflow.com/questions/42154606/python-numpy-how-to-construct-a-big-diagonal-arraymatrix-from-two-small-array
    '''
    if not rem:
        return result
    m = rem.pop(0)
    result = np.block(
        [
            [result, np.zeros((result.shape[0], m.shape[1]))],
            [np.zeros((m.shape[0], result.shape[1])), m],
        ]
    )
    return diag_mat(rem, result)

# ----------------------------------------------------------------------------------------------------------------------


class GlobFile:
    '''
    Class that import all the parameters from the config file and make them GLOBAL FORMOSA VARIABLES.
    
    Author: Paulina Palma-Bifani
    '''

    def __init__(self, config_file_path):
        # Generate the confog object
        config = ConfigObj(config_file_path, encoding='utf8')
        self.config=config

        ## Read CONFIG:
        # [config_path] (4)
        self.observation_path = config['config_path']['observation_path'] + '*'
        self.main_observation_path = config['config_path']['observation_path'] + '*' # Needs to be changed
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

        if type(config['config_adapt']['wav_for_adapt']) != list: # Create lists if only one obs in the loop 
            # [config_adapt] (5)
            self.wav_for_adapt = [config['config_adapt']['wav_for_adapt']]
            self.adapt_method = [config['config_adapt']['adapt_method']]
            self.custom_reso = [config['config_adapt']['custom_reso']]
            self.continuum_sub = [config['config_adapt']['continuum_sub']]
            self.wav_for_continuum = [config['config_adapt']['wav_for_continuum']]
            self.use_lsqr = [config['config_adapt']['use_lsqr']]

            # [config_inversion] (4)
            self.logL_type = [config['config_inversion']['logL_type']]
            self.wav_fit = [config['config_inversion']['wav_fit']]
        else:
            # [config_adapt] (5)
            self.wav_for_adapt = config['config_adapt']['wav_for_adapt']
            self.adapt_method = config['config_adapt']['adapt_method']
            self.custom_reso = config['config_adapt']['custom_reso']
            self.continuum_sub = config['config_adapt']['continuum_sub']
            self.wav_for_continuum = config['config_adapt']['wav_for_continuum']
            self.use_lsqr = config['config_adapt']['use_lsqr']

            # [config_inversion] (4)
            self.logL_type = config['config_inversion']['logL_type']
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
        self.alpha = config['config_parameter']['alpha']
        self.rv = config['config_parameter']['rv']
        self.av = config['config_parameter']['av']
        self.vsini = config['config_parameter']['vsini']
        self.ld = config['config_parameter']['ld']
        self.bb_T = config['config_parameter']['bb_T']
        self.bb_R = config['config_parameter']['bb_R']

        self.ck = None
        
        # [config_nestle] (5, some mutually exclusive)  (n_ prefix for params)
        self.n_method = config['config_nestle']['method']
        self.n_maxiter = eval(config['config_nestle']['maxiter'])
        self.n_maxcall = eval(config['config_nestle']['maxcall'])
        self.n_dlogz = eval(config['config_nestle']['dlogz'])
        self.n_decline_factor = eval(config['config_nestle']['decline_factor'])

        # [config_pymultinest]
        # self.p_n_params = config['config_pymultinest']['n_params']
        # self.p_n_clustering_params = config['config_pymultinest']['n_clustering_params']
        # self.p_wrapped_params = config['config_pymultinest']['wrapped_params']
        # self.p_importance_nested_sampling = config['config_pymultinest']['importance_nested_sampling']
        # self.p_multimodal = config['config_pymultinest']['multimodal']
        # self.p_const_efficiency_mode = config['config_pymultinest']['const_efficiency_mode']
        # self.p_evidence_tolerance = eval(config['config_pymultinest']['evidence_tolerance'])
        # self.p_sampling_efficiency = eval(config['config_pymultinest']['sampling_efficiency'])
        # self.p_n_iter_before_update = eval(config['config_pymultinest']['n_iter_before_update'])
        # self.p_null_log_evidence = eval(config['config_pymultinest']['null_log_evidence'])
        # self.p_max_modes = eval(config['config_pymultinest']['max_modes'])
        # self.p_mode_tolerance = eval(config['config_pymultinest']['mode_tolerance'])
        # self.p_seed = eval(config['config_pymultinest']['seed'])
        # self.p_verbose = config['config_pymultinest']['verbose']
        # self.p_resume = config['config_pymultinest']['resume']
        # self.p_context = eval(config['config_pymultinest']['context'])
        # self.p_write_output = config['config_pymultinest']['write_output']
        # self.p_log_zero = eval(config['config_pymultinest']['log_zero'])
        # self.p_max_iter = eval(config['config_pymultinest']['max_iter'])
        # self.p_init_MPI = config['config_pymultinest']['init_MPI']
        # self.p_dump_callback = config['config_pymultinest']['dump_callback']
        # self.p_use_MPI = config['config_pymultinest']['use_MPI']
        
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
        # config.filename = ' '
        # config['config_path']['stock_interp_grid'] = stock_interp_grid
        # config['config_path']['stock_result'] = stock_result_subsub_dir
        # config.write()
        #
        # print('Saved config: --- ' + config_current + ' ---')
        # print()