_   _   _   _   _   _   _   _   _ 

This file compiles the new additions and comments to help you navigate through the changes

_   _   _   _   _   _   _   _   _

IN THIS FOLDER :
    - config_exemple.ini : Exemple of a typical config file working with the new additions (changes with previous config files are spotted with
    the note ''NEW'')
    - config_exemple_MOSAIC.ini : Exemple of a config file working with the new additions set in MOSAIC
    - NEWS_README.txt : This file






COVARIANCE MATRIX :
    You are now able to use data that incorporate covariance matrix in ForMoSA. Be carefull, even if the format is correct
    (i.e. hdul[1].data['COVARIANCE']), you also need to specify logL_type = 'chi2_covariance' if you want your input covariance matrix to run
    through the inversion (chech the next paragraph for more info on this)


LOGLIKE :
    You are now able to select the likelihood mapping you wish to use during the inversion. This is usefull to deal with the variability of 
    format / resolution / reduction / ... observed accros observational spectra. Here, I will descibe each likelihood function and their common
    uses :
        - logL_type = 'chi2_classic' is the classical chi2 already in place in ForMoSA, i.e. logL = -0.5 * sum( res/err ^2 ). It is
        usefull when you have data points that are spectrally uncorrelated (or at least assume them to be)
        - logL_type = 'chi2_covariance' is the generalised version of the classical chi2, i.e. logL = -0.5 * sum( res^T C⁻1 res ).
        It is usefull when you have knowledge on your spectral covariances (covariance matrix C)
        - logL_type = 'CCF_Brogi' is the CCF mapping introduced by Brogi et al. 2019, i.e. logL = -N/2 * log(Sf2 - 2*R + Sg2). It
        is usefull when you assume gaussian and spectrally constant noise + self-calibrated (no continuum) data
        - logL_type = 'CCF_Zucker' is the CCF mapping introduced by Zucker et al. 2003, i.e. logL = -N/2 * np.log(1-C2). It
        is usefull when you assume gaussian and spectrally constant noise + self-calibrated (no continuum) data   
        - logL_type = 'CCF_custom' is a CCF mapping created by Matthieu Ravet, i.e. logL = -N/(2*sigma2_weight) * (Sf2 + Sg2 - 2*R). It
        is usefull when you assume gaussian and spectrally constant noise + self-calibrated (no continuum) data    


MOSAIC :
    Multimodal Option for Spectral Analysis and Improved Constraints (MOSAIC) is the main new addition in ForMoSA. To activate this option,
    simply set the parameter observation_format = 'MOSAIC' in the config file. You can follow the config_exemple_MOSAIC.ini file as an
    exemple. Here, I will describe the few things you need to be carefull when using this option:

    - observation_path : this parameter need to be your absolute path to a folder containing all the observation files you want to invert
    (in the .fits format of ForMoSA). Exemple " observation_path = '/home/mravet/Documents/These/FORMOSA/INPUTS/DATA/MULTI-MODAL/*' " /!\ note
    that the * at the end of the path is important

    - wav_for_adapt : You need to select multiple windows for each or your observations. Exemple " wav_for_adapt = '2.25,4.17' , '4.1,4.4' "

    - continuum_sub : you need to specify it for each observation now. Exemple " continuum_sub = 'NA', 'NA', 'NA' " if you don't want to substract
    the continuum for any observation or " continuum_sub = '100', 'NA', 'NA' " if you want to only substract a R=100 continuum for the first observation.
    /!\ as of now, the continuum removal is only done using the parameter wav_for_continuum = 'full', i.e. using all the wavelength range of the observation
    in the loop (an option to properly select the desired wavelength range will be added in the near future)

    - wav_for_continuum : Similar to wav_for_adapt

    - logL_type : like with the continuum_sub parameter, you now need to specify the likelihood function you want to use for each observation.
    Exemple : " logL_type = 'chi2_classic', 'chi2_classic', 'chi2_covariance' ". A verification is here done before the inversion to check whether
    you inputed the right logL funtion with the desired observation (you can redefine your choice at this step)

    - wav_fit : this parameter works as intended. You can (and should) sub-divid you interval for each of you observation. For exemple :
    " wav_fit = '2.25,2.5 / 2.85,4.17' , '4.1,4.4' "

    The grid is adapted for the needs of each observation separatly which mean you will end up with separate adapted_grid.nc and spectrum.npz.
    However, since the inversion is done in parallel, the output result_nestle.pic will be unique.

=> This option is still in development. Any feedbacks / advices are welcomed !
NOTE : MOSAIC mode can also be use on single observation. It works the exact same way but by doing this you will be notify more frequently
with check-ups (i.e. check up if the spectral resolution you use for the continuum substraction is the one you want + check up if the likelihood
function selected is the one you want). For now these check ups are only added when running MOSAIC for convinience and also to maintain a stable
root base in ForMoSA (as MOSAIC is still under tests)






Below are listed in tree structure, the changes in each directory/file :

ForMoSA

    |_ adapt :

                adapt_grid.py : * new MOSAIC path added

                adapt_obs_mod.py : * new function launch_adapt_MOSAIC() to adapt multiple data separatly
                                   * new lines to treat covariance matrix (merge + invertion + store)
                                   * new checkup added to verify the structure of the covariance matrix (i.e. no <0 values on the diagonal)

                extraction_function.py : * modification of the adapt_observation_range() function. Now it's able to take as input .fits file
                                         in two formats: wavelength either in ['WAV'] or ['WAVELENGTH'] (i.e. GRAVITY format)
                                                         flux either in ['FLX'] or ['FLUX']
                                                         errors either in ['ERR'] or ['COVARIANCE']
                                                         resolution either in ['RES'] or ['RESOLUTION']
                                                         instrument either in ['INS'] or ['INSTRUMENT']
                                         * new MOSAIC path added

    |_ interface :

                NO CHANGES

    |_ nested_sampling

            NEW nested_logL_functions.py : python script containing all log-likelihood functions used during the nested sampling

            NEW nested_MOSAIC.py : separate python script to adapt the inversion when using the MOSAIC mode

                nested_sampling.py : * modification of the loglike() function to take into account the covariance matrix + MOSAIC option
                                     * modification of the launch_nested_sampling() function to check the correct use of likelihood functions for
                                     each dataset when using MOSAIC

    |_ phototeque :

                NO CHANGES

    |_ plotting :

                NO CHANGES

    main.py : * new checkup to see if the MOSAIC is used (i.e. redirection to launch_adapt_MOSAIC() instead of the usual launch_adapt())

    main_utilities : * 3 new variables : main_observation_path (a copy of the observation path to be kept outside the loop of MOSAIC)
                                         observation_format (set to 'NA' by default, can be set to 'MOSAIC' in the config file to initiate
                                         the MOSAIC)
                                         logL_type (selection parameter to choose which likelihood to chose from the library in
                                         nested_logL_functions.py, for now, 4 options are possible :
                                         'chi2_classic' or 'CCF_Brogi' or 'CCF_Lockwood' or 'CCF_custom' or 'chi2_covariance')