#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 20:15:48 2022

@author: soroush
"""

import inverse_routine as iv 
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import inv_pso_thickness as ipt
import copy
import utils as ut
# import mpi_inv_pso_thickness as ipt
#%% PSO inversion
def inv_all(app_vel_obs_master, rf_obs_master, slowness_input,vel_param, 
            filt_list, ndivide_list = [2, 3, 4, 5], 
            stacked = False,
            name_pso_inp = False, synthetic = False, 
            vel_synth_init= False,
            use_pso = True, 
            use_app_vel = False,
            inv_bf = 2, inv_af = 25, 
            rf_method_sens = 'waterlevel',
            app_weight = 5,
            rf_method = 'iterative',
            rf_weight = 2.0,
            smooth_fac = 0.50,
            damp_fac = 0.5000,
            gauss_par = 2.5,
            nsamp = 1024, 
            dt = 0.05,
            PSO_nparticle = 30, 
            PSO_maxiter = 15,
            PSO_nthread = 4, 
            close_fig = True, 
            layers_thickness_syn = False, 
            rf_normalize = 1,
            finer_ndivide = '',
            save_dir_root= '/home/soroush/rf_shallow_codes/my_py_rf/output_of_forwftry/'):
    if (save_dir_root != False):
        if (save_dir_root[-1] != '/'):
            save_dir_root = save_dir_root + '/'
        if (os.path.isdir(save_dir_root)):
            pass 
        else:
            os.mkdir(save_dir_root)
            
    vel_param.cross_to_array = ndivide_list
    vel_param_init = copy.deepcopy(vel_param)
    
    vel_s = vel_param_init.vel_s.copy()
    if (use_app_vel):
        # vel_s[0] = app_vel_obs_master[0]
        vel_s[-1] = app_vel_obs_master[-1]
    layers_thickness = copy.deepcopy(vel_param_init.layer_thickness)
    layer_thickness_abs = copy.deepcopy(vel_param_init.layer_thickness_abs)
    #%% inv_info 
    inv_bf = inv_bf
    inv_af = inv_af


    gauss_par = gauss_par
    rf_normalize = rf_normalize
    vel_init = vel_s.copy()
    waterlevel_val = 0.01
    rf_method = rf_method
    
        
    
    #%% inv_pso 
    if (use_pso == True):
        if (name_pso_inp == False):
            if (type(finer_ndivide) != list):
                finer_ndivide = ndivide_list.copy()
            print('Starting PSO algorithm, Please wait...')
            vel_param_PSO = vel_param
            inv_frame_pso = ipt.Invert_thickness_pso(vel_init = vel_s, 
                         layers_thickness_init =layers_thickness, 
                         vel_param_initial= vel_param_init,
                         rf_obs= rf_obs_master, app_obs= app_vel_obs_master,
                         filt_list= filt_list, 
                         init_boundary = vel_param_PSO.init_boundary,
                         init_dif_boundary= vel_param_PSO.init_dif_boundary, 
                         init_layers= vel_param_PSO.PSO_layering,
                         num_procs= PSO_nthread, norm_cond=0.001,
                         pso_nparticle = PSO_nparticle, pso_max_iter = PSO_maxiter,
                         pso_c1 = 2, pso_c2 = 2.1, 
                         inv_bf = inv_bf, inv_af = inv_af,
                         pso_vw = 0.1, app_weight = app_weight,
                         rf_weight = rf_weight,
                         smooth_fac = smooth_fac,
                         damp_fac = damp_fac,
                         dt=dt,rf_method = rf_method,
                         rf_method_sens = rf_method_sens,
                         gauss_par = gauss_par,
                         rf_normalize = rf_normalize,
                         nsamp=nsamp, tshift=10.0,
                         slowness=slowness_input, 
                         print_particle_info=False, 
                         save_dir_root = save_dir_root)
            
            tthickness_final = inv_frame_pso.final_tthickness
            name = ('tthickness_pso.bin')
            name_pso = save_dir_root + name
            print("PSO result saved to "+ name_pso)
            file1 = open(name_pso, "wb") 
            pickle.dump(inv_frame_pso, file1)
            file1.close
        elif (os.path.isfile(name_pso_inp)):
            name_pso = name_pso_inp
            with open(name_pso, 'rb') as f1:
                inv_frame_pso = pickle.load(f1)
    
        if (inv_frame_pso.gbest_cond < 1.0):
            #creating a model based on PSO results
            vel_from_pso = np.unique(inv_frame_pso.gbest_vel)
            flayer = 0.0
            vs_info_from_PSO = []
            for i, boundary in enumerate(inv_frame_pso.gbest_boundary):
                vs_info_from_PSO.append([flayer, boundary, vel_from_pso[i]])
                flayer = boundary
            vs_info_from_PSO.append([inv_frame_pso.gbest_boundary[-1], 
                                     layer_thickness_abs[-2], 
                                     vel_from_pso[-1]])
            vs_info_from_PSO.append([layer_thickness_abs[-2], 
                                     layer_thickness_abs[-1], 
                                     vel_from_pso[-1]])
            layer_info = ut.find_layer_info(init_boundary= \
                                            inv_frame_pso.gbest_boundary, 
                            layering_4_cal= vel_param_init.initial_layering, 
                            max_depth_4_cal= layer_thickness_abs[-2])
            vel_param_af_PSO = \
                ut.Vel_paramterize(vel_info = vs_info_from_PSO, 
                                          layer_info = layer_info, 
                                          vp_to_vs= vel_param_init.vp_to_vs)
            
            
            
            
            
            vel_init = vel_param_af_PSO.vel_s
            lthickness = vel_param_af_PSO.layer_thickness
            tthickness_final = vel_param_af_PSO.time_thickness
            ndivide_list_accurate = finer_ndivide.copy()
            
            ndivide_list = ndivide_list_accurate.copy()
        else:
            print('COULDNT FIND A GOOD MODEL WITH PSO, PLEASE CHECK YOUR' +
                  ' INPUT PARAMETERS, USING INITIAL MODEL INPUTS')
            vel_init = vel_s.copy()
            lthickness = layers_thickness.copy()
            vel_param_af_PSO = vel_param_init
         
    elif (use_pso == False):
        vel_estimate_pso = vel_s.copy()
        name_pso = name_pso_inp 
        vel_init = vel_s.copy()
        lthickness= layers_thickness.copy()
        tthickness_initial_input = ut.cal_time_thickness(vel_init, layers_thickness, 
                              vp_to_vs = vel_param_init.vp_to_vs, 
                              slowness = slowness_input)
        

    #%% iterative inv part 
    
    
    all_iter_vel = []
    iter_vel = []
    iter_vel.append(lthickness)
    iter_vel.append(vel_init)
    
    app_weight = app_weight
    rf_weight = rf_weight
    smooth_fac = smooth_fac
    damp_fac = damp_fac
    
    app_weight_init = app_weight
    rf_weight_init = rf_weight
    smooth_fac_init = smooth_fac
    damp_fac_init = damp_fac   
    
    if ((isinstance(ndivide_list, list) == True) and \
        (len(ndivide_list)> 0)):
        iter_all = 0
        idum = 0
       
        for ndivide in ndivide_list:
            if (ndivide > 0):
                idum += 1 
                smooth_fac = smooth_fac_init + idum * (smooth_fac * 20/100)
                damp_fac = damp_fac_init + idum * (damp_fac * 10/100)
                print('===============================================\n'+
                      'dividing layers to '+str(ndivide)+ '\n'+
                'Runing with increased smoothing and damping factors. \n' +
                      ' smooth factor is :' + str(smooth_fac)+ 
                      ' damp factor is : '+
                      str(damp_fac))
                folder_name = ('Divide_to_'+str(ndivide)+
                               '_increased_factors'+'/')
            else:
                smooth_fac = smooth_fac_init
                damp_fac = damp_fac_init
                ndivide = -1 * ndivide
                print('===============================================\n'+
                      'dividing layers to '+str(ndivide)+ '\n'+
                'Runing with initial smoothing and damping factors. \n' +
                      ' smooth factor is :' + str(smooth_fac)+ 
                      ' damp factor is : '+
                      str(damp_fac))
                folder_name = ('Divide_to_'+str(ndivide)+
                               '_initial_factors'+'/')
            iter_all += 1
            if (use_pso):
                folder_name_pso = os.path.join(save_dir_root, 
                                               'PSO_pseudo_model_output')
                if (os.path.isdir(folder_name_pso)):
                    pass 
                else:
                    os.mkdir(folder_name_pso)
                save_dir_run = os.path.join(folder_name_pso, folder_name)
                tthickness_final = vel_param_af_PSO.time_thickness
            else:
                save_dir_run = os.path.join(save_dir_root, folder_name)
                tthickness_final = tthickness_initial_input
            vel_out, lthickness_out= divide_tthickness(
                              vel_estimate= vel_init, 
                              tthickness_final= tthickness_final,
                              slowness= slowness_input,
                              ndivide = ndivide)
             
            # tthickness_final = inv_frame_pso.final_tthickness
            # vel_out, lthickness_out= divide_tthickness(
            #                   vel_estimate= vel_estimate, 
            #                   tthickness_final= tthickness_final_inv_nodivide,
            #                   slowness= slowness_input,
            #                   ndivide = ndivide)
            if (synthetic == False):
                iter_vel = []
                iter_vel.append(lthickness_out)
                iter_vel.append(vel_out)
                inv_frame_iter_div = inv_real(vel_init= vel_out, 
                        layers_thickness= lthickness_out, 
                        rf_obs= rf_obs_master, app_obs= app_vel_obs_master, 
                     app_weight= app_weight, rf_weight = rf_weight, 
                     smooth_factor= smooth_fac, damp_factor= damp_fac, 
                     filt_list= filt_list, nsamp =nsamp, dt = dt, 
                     rf_method= rf_method, close_fig = close_fig,
                     slowness= slowness_input, save_dir= save_dir_run, 
                     gauss_par= gauss_par, waterlevel= waterlevel_val, 
                     rf_normalize= rf_normalize, 
                     inv_time_rf1= inv_bf, inv_time_rf2= inv_af, 
                     cal_sens= True)          
               
               
            else:
                #must map vel synthetic to new layers
                iter_vel = []
                iter_vel.append(lthickness_out)
                iter_vel.append(vel_out)
                inv_frame_iter_div = inv_syn(vel_init= vel_out, 
                                        layers_thickness= lthickness_out, 
                     rf_obs= rf_obs_master, app_obs= app_vel_obs_master, 
                     app_weight= app_weight, rf_weight = rf_weight, 
                     smooth_factor= smooth_fac, damp_factor= damp_fac, 
                     filt_list= filt_list, nsamp =nsamp, dt = dt, 
                     rf_method= rf_method, close_fig = close_fig,
                     layers_thickness_syn = layers_thickness_syn,
                     slowness= slowness_input, save_dir= save_dir_run, 
                     gauss_par= gauss_par, waterlevel= waterlevel_val, 
                     rf_normalize= rf_normalize, 
                     inv_time_rf1= inv_bf, inv_time_rf2= inv_af,
                     vel_synth= vel_synth_init, synthetic= True, 
                     cal_sens= True)
            obj_af_inv = (inv_frame_iter_div.dif_rf_norm_curr + 
                             inv_frame_iter_div.dif_app_norm_curr)
           
            rf_cond = (inv_frame_iter_div.dif_rf_norm_curr)
            app_cond = (inv_frame_iter_div.dif_app_norm_curr)
            iter_vel.append(inv_frame_iter_div.vel_s_estimate)
            iter_vel.append(rf_cond)
            iter_vel.append(app_cond)
            iter_vel.append(inv_frame_iter_div.rf_curve_best_curr)
            iter_vel.append(inv_frame_iter_div.app_curve_best_curr)
            iter_vel.append(inv_frame_iter_div.best_lthickness)
            iter_vel.append(inv_frame_iter_div.norm_all_iter)
            all_iter_vel.append(iter_vel)
            
            all_cond = []
            for el in all_iter_vel:
                cond = el[3] + el[4]
                all_cond.append(cond)
            all_cond = np.array(all_cond)
            ind = np.argwhere(np.min(all_cond) == all_cond)[0][0]
            inv_info_d = all_iter_vel[ind].copy()
           
            inv_info = {}
            inv_info['all_iter'] = all_iter_vel.copy()
            inv_info['vel_param'] = vel_param_init
            inv_info['best_inv'] = inv_info_d 
            inv_info['name_pso'] = name_pso
    return(inv_info)



#%%
def divide_tthickness(vel_estimate, tthickness_final,
                       slowness,
                       ndivide = 2.0):
    lthickness_out = []
    vel_out = []
    for k in range(len(tthickness_final) -1):
        vel, lthick = divide_tthickness_layer(tthickness_final[k],
                                              vel_layer= vel_estimate[k], 
                                              slowness= slowness,
                                              ndivide= ndivide)
        for i in range(len(vel)):
            vel_out.append(vel[i])
            lthickness_out.append(lthick[i])
    vel_out.append(vel_estimate[-1])
    lthickness_out.append(0.0)
    return(vel_out, lthickness_out)
def divide_tthickness_layer(tt, vel_layer, slowness, 
                            ndivide = 2):
    tt_new = tt / ndivide
    lthickness_layer = cal_lthickness_layer(tt_new, vel_layer, slowness)
    vel_new = [] 
    lthickness_new = [] 
    for i in range(ndivide):
        vel_new.append(vel_layer)
        lthickness_new.append(lthickness_layer)
    return(vel_new, lthickness_new)



def cal_lthickness_layer(tt, vel_l, slowness): 
    numerator = tt
    denomerator1 = (np.sqrt((vel_l**-2.0) -
                               (slowness) ** 2.0))
    denomerator2 = (np.sqrt(((1.732 *vel_l)**-2.0) -
                          (slowness) ** 2.0))
    denomerator = denomerator1 - denomerator2
    lthickness_layer = (numerator / denomerator)
    return(lthickness_layer)    
def cal_lthickness(tthickness, vel, slowness):
    lthickness = [] 
    idum = -1
    for el in vel:
        idum += 1
        numerator = tthickness[idum]
        denomerator = (np.sqrt((el**-2.0) -
                               (slowness) ** 2.0))
        lthickness.append(numerator / denomerator)
    return(lthickness)
    
def cal_tthickness(vel, layer_thickenss, slowness):
   tps = [] 
   idum = -1
   for el in vel:
       idum += 1
       numerator1 = (np.sqrt((el**-2.0) -
                              (slowness) ** 2.0))
       numerator2 = (np.sqrt(((1.732 *el)**-2.0) -
                              (slowness) ** 2.0))
       tps.append(layer_thickenss[idum] * (numerator1 - numerator2))
   return(tps)

def inv_real(vel_init, layers_thickness, 
              rf_obs, app_obs, 
              app_weight, rf_weight, smooth_factor, damp_factor, 
              filt_list, nsamp, dt, rf_method, 
              slowness, save_dir, gauss_par, waterlevel,close_fig, 
              rf_normalize, inv_time_rf1, inv_time_rf2, cal_sens= True):
    inv_frame_iter = iv.Invert_joint_iter(vel_init,
                            layers_thickness= layers_thickness,
                            rf_obs= rf_obs,
                            app_obs = app_obs,
                            app_weight= app_weight, rf_weight = rf_weight,
                            smooth_factor= smooth_factor,
                            damp_factor= damp_factor,
                            filt_list=filt_list,
                            Cd_array_rf = False, 
                            Cd_array_app = False,
                            norm_cond=0.01,
                            amon_method=True,
                            nsamp = nsamp, dt =dt, rf_method=rf_method,
                            tshift=10.0,
                            no_damp = False, 
                            no_smooth=False,
                            max_iter=10,
                            slowness=slowness,
                            jacob = 0,
                            out_kind='best',
                            save_dir=save_dir,
                            gauss_par = gauss_par,
                            waterlevel = waterlevel,
                            rf_normalize = rf_normalize,
                            inv_time_rf1 = inv_time_rf1,
                            inv_time_rf2 = inv_time_rf2,
                            cal_sens = cal_sens, 
                            obs_normalize=True, 
                            synthetic = False,
                            print_out=False,
                            close_fig = close_fig,
                            save_to_dict = False,
                            force_jacob_cal = True,
                            folder_jacob_save = 'other')
    return(inv_frame_iter)

def inv_syn(vel_init, layers_thickness, 
              rf_obs, app_obs, 
              app_weight, rf_weight, smooth_factor, damp_factor, 
              filt_list, nsamp, dt, rf_method, layers_thickness_syn,
              slowness, save_dir, gauss_par, waterlevel, close_fig,
              rf_normalize, inv_time_rf1, inv_time_rf2, 
              vel_synth, cal_sens= True, synthetic= True):
    inv_frame_iter = iv.Invert_joint_iter(vel_init,
                            layers_thickness= layers_thickness,
                            rf_obs= rf_obs,
                            app_obs = app_obs,
                            app_weight= app_weight, rf_weight = rf_weight,
                            smooth_factor= smooth_factor,
                            damp_factor= damp_factor,
                            filt_list=filt_list,
                            Cd_array_rf = False, 
                            Cd_array_app = False,
                            norm_cond=0.01,
                            amon_method=True,
                            nsamp = nsamp, dt =dt, rf_method=rf_method,
                            layers_thickness_syn = layers_thickness_syn,
                            tshift=10.0,
                            no_damp = False, 
                            no_smooth=False,
                            max_iter=10,
                            slowness=slowness,
                            jacob = 0,
                            out_kind='best',
                            close_fig = close_fig,
                            vel_synth=vel_synth, 
                            save_dir=save_dir,
                            gauss_par = gauss_par,
                            waterlevel = waterlevel,
                            rf_normalize = rf_normalize,
                            inv_time_rf1 = inv_time_rf1,
                            inv_time_rf2 = inv_time_rf2,
                            cal_sens = cal_sens, 
                            obs_normalize=True, 
                            synthetic = synthetic,
                            print_out=False,
                            save_to_dict = False,
                            force_jacob_cal = True,
                            folder_jacob_save = 'other')
    return(inv_frame_iter)

def cal_nlayer_new(nlayer_per, cross_to = 1.2):
    nlayer_curr = int(np.round(nlayer_per * cross_to))
    # if ((nlayer_curr == nlayer_per) and (cross_to > 1.0)):
        # nlayer_curr = nlayer_curr + 1
    return(nlayer_curr)
    
    
    
    
    