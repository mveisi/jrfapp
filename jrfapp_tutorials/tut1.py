#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:33:11 2023

@author: soroush
"""

from jrfapp.main_classes import Initialize
from jrfapp.main_classes import Jrfapp_station
import os
#%%
### Folders that need to be defined:

## Path to the data folder. This folder must contain your stations folder data. 
data_folder = '/mnt/4TB/soroush/CB_network'
###############################################################################
## Path to the folder which contains references models. The folder must only 
## contain .dat files. 
model_folder = '/home/soroush/my_py_rf/model_folder'
###############################################################################
## Path to the specific refrence initial model. These option used because you may want to
## use different model as the reference model. 
model_name='/home/soroush/my_py_rf/model_folder/continent_tibet_model.dat'
###############################################################################
## Path to the coordinates file. This file must contain station name, lat and 
## long of the stations. Note that the station name in this file, the station 
## name in the data_folder and station name given to thr Jrfapp_station class 
## must be same.
station_coordinate_file= 'CB_coordinates'
###############################################################################
## The folder which package use for saving the output of stations. 
output_folder='/mnt/4TB/soroush/CB_st/syn_st'
###############################################################################

#%%
init_obj = Initialize(model_name= 'halfspace',
        output_folder=output_folder, 
        gauss_width= 3.5)
print(init_obj.output_folder)
#%%
which_to_pert = []
which_to_pert.append([6, 12,  -0.6])
which_to_pert.append([18, 24,  0.6])
which_to_pert.append([30, 36, -0.6])
which_to_pert.append([50, 60,  0.6])

init_obj.create_synthetic(which_to_pert)
init_obj.plot_model('initial')
init_obj.plot_model('synthetic')
#%%
jrfapp_stobj = Jrfapp_station(init_obj, name = 'synthetic', 
                                  noise_level= 0.0)
jrfapp_stobj_file_name = jrfapp_stobj.save_file(file_name= 'syn_no_noise_gs_bf_inv')
#%%
jrfapp_stobj.invert_data(inv_method = 'grid_search',
                 stack_name = 'synthetic',
                 ndivide_list = [-1, 1, -2, 2, -3, 3, -4, 4],
                 finer_ndivide= [-2, 2, -3, 3, -4, 4, -5, 5],
                 nmodel = 6, nthread = 12)

jrfapp_stobj.plot_results(fig_name= 'GS_out.png')
jrfapp_stobj_file_name = jrfapp_stobj.save_file(file_name= 'syn_no_noise_gs_af_inv')
#%%
import pickle
jrfapp_stobj_file_name = '/home/soroush/rf_shallow_codes/makran_data/pkg_test/syn_no_noise_gs_af_inv'
with open(jrfapp_stobj_file_name, 'rb') as f1:
    jrfapp_stobj = pickle.load(f1)

jrfapp_stobj.invert_data(inv_method = 'PSO',
                stack_name = 'synthetic',
                PSO_nparticle = 4, PSO_maxiter = 2, 
                nthread = 12)
jrfapp_stobj_file_name = jrfapp_stobj.save_file(file_name= 'syn_no_noise_pso')
jrfapp_stobj.plot_results(fig_name= 'PSO_out.png')

jrfapp_stobj_file_name = '/home/soroush/rf_shallow_codes/makran_data/pkg_test/syn_no_noise_pso'
with open(jrfapp_stobj_file_name, 'rb') as f1:
    jrfapp_stobj = pickle.load(f1)























































