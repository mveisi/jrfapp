#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:58:02 2023

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
station_coordinate_file= '/home/soroush/my_py_rf/CB_coordinates'
###############################################################################
## The folder which package use for saving the output of stations. 
output_folder='/mnt/4TB/soroush/CB_st/syn_st'
#%%
init_obj = Initialize(network_name='CB', 
                                    station_coordinate_file= station_coordinate_file,
                                    data_folder= data_folder, 
                                    layering= [4, 5, 5], 
                                    model_folder = model_folder,
                                    output_folder= output_folder, 
                                    model_name= model_name,
                                    random_seed= 250)
#%%
jrfapp_stobj = Jrfapp_station(init_obj, name = 'CAD_P')
jrfapp_stobj_file_name = jrfapp_stobj.save_file(file_name='real_data_cad_gs_bf_inv.bin')
#%%
jrfapp_stobj.invert_data(inv_method = 'PSO',
                stack_name = "K0 Stack joint_harmonic after app_vel criteria",
                ndivide_list = [-1, 1, -2, 2, -3, 3, -4, 4], 
                PSO_nparticle = 2, PSO_maxiter = 2, 
                finer_ndivide= [-2, 2, -3, 3, -4, 4, -5, 5],
                nthread = 6)
jrfapp_stobj_file_name = jrfapp_stobj.save_file(file_name='real_data_cad_pso_af_inv.bin')
#%%























