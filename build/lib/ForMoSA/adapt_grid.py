from __future__ import print_function, division
import numpy as np
import xarray as xr
import time
# import matplotlib.pyplot as plt

from adapt.extraction_functions import adapt_model, decoupe

# ----------------------------------------------------------------------------------------------------------------------


def adapt_grid(global_params, wav_obs_spec, wav_obs_phot):
    """
    Adapt the synthetic spectra of a grid to make them comparable with data.

    Args:
        global_params: Class containing each parameter
        wav_obs_spec: Merged wavelength grid of the data
        wav_obs_phot: Wavelengths of the photometry points
    Returns:

    Author: Simon Petrus
    """
    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine="netcdf4")
    wav_mod_nativ = ds["wavelength"].values
    grid = ds['grid']
    attr = ds.attrs
    grid_np = grid.to_numpy()
    if len(attr['par']) == 2:
        grid_new_np = np.full((len(wav_obs_spec),
                               len(grid["par1"].values),
                               len(grid["par2"].values)), np.nan)
        grid_phot_new_np = np.full((len(wav_obs_phot),
                                    len(grid["par1"].values),
                                    len(grid["par2"].values)), np.nan)
        tot_par = len(grid["par1"].values) * len(grid["par2"].values)
    if len(attr['par']) == 3:
        grid_new_np = np.full((len(wav_obs_spec),
                               len(grid["par1"].values),
                               len(grid["par2"].values),
                               len(grid["par3"].values)), np.nan)
        grid_phot_new_np = np.full((len(wav_obs_phot),
                                    len(grid["par1"].values),
                                    len(grid["par2"].values),
                                    len(grid["par3"].values)), np.nan)
        tot_par = len(grid["par1"].values) * len(grid["par2"].values) * len(grid["par3"].values)
    if len(attr['par']) == 4:
        grid_new_np = np.full((len(wav_obs_spec),
                               len(grid["par1"].values),
                               len(grid["par2"].values),
                               len(grid["par3"].values),
                               len(grid["par4"].values)), np.nan)
        grid_phot_new_np = np.full((len(wav_obs_phot),
                                    len(grid["par1"].values),
                                    len(grid["par2"].values),
                                    len(grid["par3"].values),
                                    len(grid["par4"].values)), np.nan)
        tot_par = len(grid["par1"].values) * len(grid["par2"].values) * len(grid["par3"].values) * len(grid["par4"].values)
    if len(attr['par']) == 5:
        grid_new_np = np.full((len(wav_obs_spec),
                               len(grid["par1"].values),
                               len(grid["par2"].values),
                               len(grid["par3"].values),
                               len(grid["par4"].values),
                               len(grid["par5"].values)), np.nan)
        grid_phot_new_np = np.full((len(wav_obs_phot),
                                    len(grid["par1"].values),
                                    len(grid["par2"].values),
                                    len(grid["par3"].values),
                                    len(grid["par4"].values),
                                    len(grid["par5"].values)), np.nan)
        tot_par = len(grid["par1"].values) * len(grid["par2"].values) * len(grid["par3"].values) * len(grid["par4"].values) * len(grid["par5"].values)
    i_tot = 1
    follow_print_title = ''
    for par_t in attr['title']:
        follow_print_title += par_t + ' \t- \t'
    for p1_i, p1 in enumerate(grid['par1'].values):
        for p2_i, p2 in enumerate(grid['par2'].values):
            if len(attr['par']) > 2:
                for p3_i, p3 in enumerate(grid['par3'].values):
                    if len(attr['par']) > 3:
                        for p4_i, p4 in enumerate(grid['par4'].values):
                            if len(attr['par']) > 4:
                                for p5_i, p5 in enumerate(grid['par5'].values):
                                    time1 = time.time()
                                    model_to_adapt = grid_np[:, p1_i, p2_i, p3_i, p4_i, p5_i]
                                    nan_mod_ind = ~np.isnan(model_to_adapt)
                                    if len(np.where(nan_mod_ind is False)[0]) == 0:
                                        flx_mod_extract, mod_pho = adapt_model(global_params, wav_mod_nativ,
                                                                               model_to_adapt, attr['res'])
                                        grid_new_np[:, p1_i, p2_i, p3_i, p4_i, p5_i] = flx_mod_extract
                                        grid_phot_new_np[:, p1_i, p2_i, p3_i, p4_i, p5_i] = mod_pho
                                    else:

                                        print('The extraction of the model : '+attr['title'][0]+'=' + str(p1) +
                                              ', '+attr['title'][1]+'=' + str(p2) +
                                              ', '+attr['title'][2]+'=' + str(p3) +
                                              ', '+attr['title'][3]+'=' + str(p4) +
                                              ', '+attr['title'][4]+'=' + str(p5) +
                                              '   failed')
                                    print(str(p1_i + 1) + '/' + str(len(grid['par1'].values)) + ' \t- \t' +
                                          str(p2_i + 1) + '/' + str(len(grid['par2'].values)) + ' \t- \t' +
                                          str(p3_i + 1) + '/' + str(len(grid['par3'].values)) + ' \t- \t' +
                                          str(p4_i + 1) + '/' + str(len(grid['par4'].values)) + ' \t- \t' +
                                          str(p5_i + 1) + '/' + str(len(grid['par5'].values)) + ' \t- \t' +
                                          '      Estimated time : ' + str(int(decoupe((tot_par - i_tot) *
                                                                                      (time.time() - time1))[0])) + 
                                          'h : ' + str(int(decoupe((tot_par - i_tot) * (time.time() - time1))[1])) + 
                                          'm : ' + str(int(decoupe((tot_par - i_tot) * (time.time() - time1))[2])) + 
                                          's')
                                    line_up = '\033[1A'
                                    line_clear = '\x1b[2K'
                                    print(line_up, end=line_clear)
                                    i_tot += 1
                            else:
                                time1 = time.time()
                                model_to_adapt = grid_np[:, p1_i, p2_i, p3_i, p4_i]
                                nan_mod_ind = ~np.isnan(model_to_adapt)
                                if len(np.where(nan_mod_ind is False)[0]) == 0:
                                    flx_mod_extract, mod_pho = adapt_model(global_params, wav_mod_nativ,
                                                                           model_to_adapt, attr['res'])
                                    grid_new_np[:, p1_i, p2_i, p3_i, p4_i] = flx_mod_extract
                                    grid_phot_new_np[:, p1_i, p2_i, p3_i, p4_i] = mod_pho
                                else:

                                    print('The extraction of the model : ' + attr['title'][0] + '=' + str(p1) +
                                          ', ' + attr['title'][1] + '=' + str(p2) +
                                          ', ' + attr['title'][2] + '=' + str(p3) +
                                          ', ' + attr['title'][3] + '=' + str(p4) +
                                          '   failed')
                                print(str(p1_i + 1) + '/' + str(len(grid['par1'].values)) + ' \t- \t' +
                                      str(p2_i + 1) + '/' + str(len(grid['par2'].values)) + ' \t- \t' +
                                      str(p3_i + 1) + '/' + str(len(grid['par3'].values)) + ' \t- \t' +
                                      str(p4_i + 1) + '/' + str(len(grid['par4'].values)) + ' \t- \t' +
                                      '      Estimated time : ' + str(int(decoupe((tot_par - i_tot) *
                                                                                  (time.time() - time1))[0]))
                                      + 'h : ' + str(int(decoupe((tot_par - i_tot) *
                                                                 (time.time() - time1))[1]))
                                      + 'm : ' + str(int(decoupe((tot_par - i_tot) *
                                                                 (time.time() - time1))[2]))
                                      + 's')
                                line_up = '\033[1A'
                                line_clear = '\x1b[2K'
                                print(line_up, end=line_clear)
                                i_tot += 1
                    else:
                        time1 = time.time()
                        model_to_adapt = grid_np[:, p1_i, p2_i, p3_i]
                        nan_mod_ind = ~np.isnan(model_to_adapt)
                        if len(np.where(nan_mod_ind is False)[0]) == 0:
                            flx_mod_extract, mod_pho = adapt_model(global_params, wav_mod_nativ, model_to_adapt,
                                                                   attr['res'])
                            grid_new_np[:, p1_i, p2_i, p3_i] = flx_mod_extract
                            grid_phot_new_np[:, p1_i, p2_i, p3_i] = mod_pho
                        else:

                            print('The extraction of the model : ' + attr['title'][0] + '=' + str(p1) +
                                  ', ' + attr['title'][1] + '=' + str(p2) +
                                  ', ' + attr['title'][2] + '=' + str(p3) +
                                  '   failed')
                        print(str(p1_i + 1) + '/' + str(len(grid['par1'].values)) + ' \t- \t' +
                              str(p2_i + 1) + '/' + str(len(grid['par2'].values)) + ' \t- \t' +
                              str(p3_i + 1) + '/' + str(len(grid['par3'].values)) + ' \t- \t' +
                              '      Estimated time : ' + str(int(decoupe((tot_par - i_tot) *
                                                                          (time.time() - time1))[0]))
                              + 'h : ' + str(int(decoupe((tot_par - i_tot) *
                                                         (time.time() - time1))[1]))
                              + 'm : ' + str(int(decoupe((tot_par - i_tot) *
                                                         (time.time() - time1))[2]))
                              + 's')
                        line_up = '\033[1A'
                        line_clear = '\x1b[2K'
                        print(line_up, end=line_clear)
                        i_tot += 1
            else:
                time1 = time.time()
                model_to_adapt = grid_np[:, p1_i, p2_i]
                nan_mod_ind = ~np.isnan(model_to_adapt)
                if len(np.where(nan_mod_ind is False)[0]) == 0:
                    flx_mod_extract, mod_pho = adapt_model(global_params, wav_mod_nativ, model_to_adapt, attr['res'])
                    grid_new_np[:, p1_i, p2_i] = flx_mod_extract
                    grid_phot_new_np[:, p1_i, p2_i] = mod_pho
                else:

                    print('The extraction of the model : ' + attr['title'][0] + '=' + str(p1) +
                          ', ' + attr['title'][1] + '=' + str(p2) +
                          '   failed')
                print(str(p1_i + 1) + '/' + str(len(grid['par1'].values)) + ' \t- \t' +
                      str(p2_i + 1) + '/' + str(len(grid['par2'].values)) + ' \t- \t' +
                      '      Estimated time : ' + str(int(decoupe((tot_par - i_tot) *
                                                                  (time.time() - time1))[0]))
                      + 'h : ' + str(int(decoupe((tot_par - i_tot) *
                                                 (time.time() - time1))[1]))
                      + 'm : ' + str(int(decoupe((tot_par - i_tot) *
                                                 (time.time() - time1))[2]))
                      + 's')
                line_up = '\033[1A'
                line_clear = '\x1b[2K'
                print(line_up, end=line_clear)
                i_tot += 1

    if len(attr['par']) == 2:
        ds_new = xr.Dataset(data_vars=dict(grid=(["wavelength", "par1", "par2"], grid_new_np)),
                            coords={"wavelength": wav_obs_spec,
                                    "par1": grid["par1"].values,
                                    "par2": grid["par2"].values},
                            attrs=attr)
        ds_new_phot = xr.Dataset(data_vars=dict(grid=(["wavelength", "par1", "par2"], grid_phot_new_np)),
                                 coords={"wavelength": wav_obs_phot,
                                         "par1": grid["par1"].values,
                                         "par2": grid["par2"].values},
                                 attrs=attr)
    if len(attr['par']) == 3:
        ds_new = xr.Dataset(data_vars=dict(grid=(["wavelength", "par1", "par2", "par3"], grid_new_np)),
                            coords={"wavelength": wav_obs_spec,
                                    "par1": grid["par1"].values,
                                    "par2": grid["par2"].values,
                                    "par3": grid["par3"].values},
                            attrs=attr)
        ds_new_phot = xr.Dataset(data_vars=dict(grid=(["wavelength", "par1", "par2", "par3"], grid_phot_new_np)),
                                 coords={"wavelength": wav_obs_phot,
                                         "par1": grid["par1"].values,
                                         "par2": grid["par2"].values,
                                         "par3": grid["par3"].values},
                                 attrs=attr)
    if len(attr['par']) == 4:
        ds_new = xr.Dataset(data_vars=dict(grid=(["wavelength", "par1", "par2", "par3", "par4"], grid_new_np)),
                            coords={"wavelength": wav_obs_spec,
                                    "par1": grid["par1"].values,
                                    "par2": grid["par2"].values,
                                    "par3": grid["par3"].values,
                                    "par4": grid["par4"].values},
                            attrs=attr)
        ds_new_phot = xr.Dataset(data_vars=dict(grid=(["wavelength", "par1", "par2", "par3", "par4"], 
                                                      grid_phot_new_np)),
                                 coords={"wavelength": wav_obs_phot,
                                         "par1": grid["par1"].values,
                                         "par2": grid["par2"].values,
                                         "par3": grid["par3"].values,
                                         "par4": grid["par4"].values},
                                 attrs=attr)

    if len(attr['par']) == 5:
        ds_new = xr.Dataset(data_vars=dict(grid=(["wavelength", "par1", "par2", "par3", "par4", "par5"], grid_new_np)),
                            coords={"wavelength": wav_obs_spec,
                                    "par1": grid["par1"].values,
                                    "par2": grid["par2"].values,
                                    "par3": grid["par3"].values,
                                    "par4": grid["par4"].values,
                                    "par5": grid["par5"].values},
                            attrs=attr)
        ds_new_phot = xr.Dataset(data_vars=dict(grid=(["wavelength", "par1", "par2", "par3", "par4", "par5"], 
                                                      grid_phot_new_np)),
                                 coords={"wavelength": wav_obs_phot,
                                         "par1": grid["par1"].values,
                                         "par2": grid["par2"].values,
                                         "par3": grid["par3"].values,
                                         "par4": grid["par4"].values,
                                         "par5": grid["par5"].values},
                                 attrs=attr)

    print()
    print()
    print()
    print('-> The possible holes in the grid are interpolated: ')
    print()
    for key_ind, key in enumerate(attr['key']):
        print(str(key_ind+1) + '/' + str(len(attr['key'])))
        ds_new = ds_new.interpolate_na(dim=key, method="linear", fill_value="extrapolate", limit=None,
                                       max_gap=None)
        ds_new_phot = ds_new_phot.interpolate_na(dim=key, method="linear", fill_value="extrapolate", limit=None,
                                                 max_gap=None)

    ds_new.to_netcdf(global_params.adapt_store_path + '/adapted_grid_merge_' + global_params.grid_name + '_nonan.nc',
                     format='NETCDF4',
                     engine='netcdf4',
                     mode='w')
    ds_new_phot.to_netcdf(global_params.adapt_store_path + '/adapted_grid_phot_' + global_params.grid_name + '_nonan.nc',
                          format='NETCDF4',
                          engine='netcdf4',
                          mode='w')
    print()
    print('-> The possible holes have been interpolated.')
