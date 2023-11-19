#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 14:09:50 2022

@author: soroush
"""

'''
inverse routine for H/v'''





from rf.deconvolve import deconv_waterlevel, deconv_iterative
import shutil
import rftan_classes as rc
import matplotlib
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import timeit
import pickle
import copy
import utils as ut
#%%
#iterative ls jacobson
class Invert_joint_iter:
    def __init__(self, vel_init, layers_thickness,
                 rf_obs, app_obs, jacob,
                 filt_list, norm_cond=0.15,
                 Cd_array_rf =False, Cd_array_app =False, 
                 Cb_array = False, Cm_array = False,
                 pert_val = 0.5, 
                 dt=0.05,rf_weight = 1.0, app_weight = 1.0,
                 max_iter = 100,
                 nsamp=1024, tshift=10.0,
                 slowness=0.04,
                 smooth_factor=4.0, damp_factor = 1.0,
                 out_kind = 'iterative',
                 no_damp = False, no_smooth = False,
                 amon_method = False,
                 vel_synth=False,
                 save_dir=False,
                 gauss_par = 2.5,
                 rf_method = 'iterative',
                 waterlevel = 0.01,
                 rf_normalize = None,
                 inv_time_rf1 = 5.0,
                 inv_time_rf2 = 25.0,
                 force_last_layer = False,
                 force_first_layer = False,
                 synthetic = False,
                 layers_thickness_syn = False, 
                 cal_sens=True, 
                 obs_normalize = False, 
                 save_to_dict = False,
                 close_fig = True, 
                 force_jacob_cal = True,
                 print_out = True,
                 folder_jacob_save = 'current'
                 ,plot_fig = True):
        self.force_last_layer = force_last_layer
        self.print_out = print_out
        self.folder_jacob_save = folder_jacob_save
        self.plot_fig = plot_fig
        self.obs_normalize = obs_normalize
        self.amon_method = amon_method
        self.no_damp = no_damp 
        self.no_smooth = no_smooth
        self.out_kind = out_kind
        self.max_iter = max_iter
        self.pert_val = pert_val
        self.save_to_dict = save_to_dict
        self.inv_time_rf1 = inv_time_rf1
        self.inv_time_rf2 = inv_time_rf2
        self.waterlevel = waterlevel
        self.gauss_par = gauss_par 
        self.rf_method = rf_method 
        self.rf_normalize = rf_normalize
        self.save_dir = save_dir
        self.rf_obs = rf_obs
        self.app_obs = app_obs
        self.dt = dt
        self.tshift = tshift
        self.synthetic = synthetic
        self.nsamp = nsamp
        self.slowness = slowness
        self.filt_list = filt_list
        self.norm_cond = norm_cond
        self.jacob = jacob
        self.layers_thickness = layers_thickness
        self.vel_init = vel_init
        self.nlayer = len(self.vel_init)
        self.unknown_param = np.zeros(shape=(len(self.vel_init),))
        self.damp_factor = damp_factor **2.0
        self.smooth_factor = smooth_factor **2.0
        self.app_weight = app_weight **2.0
        self.rf_weight = rf_weight **2.0
        self.Cm_array = Cm_array 
        self.Cd_array_rf = Cd_array_rf
        self.Cd_array_app = Cd_array_app
        self.Cb_array = Cb_array
        self.force_jacob_cal = force_jacob_cal
        self.cal_sens = cal_sens
        self.force_first_layer = force_first_layer
        self.layers_thickness_syn = layers_thickness_syn
        self.close_fig = close_fig
        self.norm_all_iter = []
        if (vel_synth):
            self.synthetic = True
            self.vel_syn = vel_synth
            if ((len(vel_synth) == len(vel_init)) and
                np.array(self.layers_thickness_syn).any() == False):
                self.layers_thickness_syn = self.layers_thickness.copy()
        if (self.save_to_dict):
            self.output_dict = {}
        if (self.save_dir != False):
            if (self.save_dir[-1] != '/'):
                self.save_dir = self.save_dir + '/'
            if (os.path.isdir(self.save_dir)):
                pass 
            else:
                os.mkdir(self.save_dir)
        self.damp_factor_virgin = self.damp_factor
        self.smooth_factor_virgin = self.smooth_factor 
        self.app_weight_virgin = self.app_weight 
        self.rf_weight_virgin = self.rf_weight
        self.layer_thickness_initial = self.layers_thickness.copy()
        self.fix_rf_obs_from_zero()
        #fixing first layer:
        # self.vel_init[0] = self.app_obs[0]
        # self.print_out = True
        
        

        
        
        
        
        
        
        start = timeit.default_timer()
        self.inv_LS()
        stop = timeit.default_timer()
        self.runtime = stop - start
        if (self.plot_fig):
            self.plot_estimates(save_to_dir= True)
        
    def inv_LS(self):
        self.iter_all = 0
        self.stop_run = False
        while (self.stop_run == False):
            self.iter_all += 1
            self.run_LS()
            if (self.out_kind == 'best'):
                self.find_cond_best()
            self.check_for_stop()
            self.norm_all_iter.append([self.dif_rf_norm_curr,
                                       self.dif_app_norm_curr,
                                       self.dif_smooth_norm_curr,
                                       self.dif_damp_norm_curr,
                                       self.dif_all_norm_curr,
                                       self.iter_all])
            if (self.print_out):
                print('======== iter '+str(self.iter_all)+
                  ' from '+str(self.max_iter)+'===============')
                print("new velocity assigned with reduction= " + 
                  str(self.cond_curr) + '\n'+
                  'rf obj is = '+str(self.dif_rf_norm_curr) + 
                  ' initial was = '+str(self.dif_rf_norm_init) + '\n'+
                  'app obj is = '+str(self.dif_app_norm_curr) + 
                  ' initial was = '+str(self.dif_app_norm_init) + '\n'+
                  'smooth obj is = '+str(self.dif_smooth_norm_curr)+ 
                  ' initial was = '+str(self.dif_smooth_norm_init) + '\n'+
                  'damp obj is = '+str(self.dif_damp_norm_curr)+ 
                  ' initial was = '+str(self.dif_damp_norm_init) + '\n'+
                  'final obj is = '+str(self.dif_all_norm_curr)+
                  ' initial was = '+str(self.norm_init) +'\n'+
                  'mean difference was = '+str(self.mean_dif_vel))
            
            
    def run_LS(self):
        if (self.iter_all == 1):
            self.vel_per = self.vel_init.copy()
            self.lthickness_per = self.layers_thickness.copy() 
            self.lthickness_cur = self.layers_thickness.copy()
            self.initial_cal_cond()
            self.vel_curr = self.calculate_vel(vel_p= self.vel_per)
            self.lthickness_cur = \
                self.update_layer_thickness(curr_vel = self.vel_curr, 
                                        per_vel= self.vel_per, 
                                        per_lthickness = self.lthickness_cur)
            self.vel_per = copy.deepcopy(self.vel_init)
            self.mean_dif_vel = np.mean(np.abs(np.array(self.vel_curr) - 
                                               np.array(self.vel_per)))
            
        else:
            self.vel_per = copy.deepcopy(self.vel_curr)
            if (self.cal_sens == False):
                (app_calculated, rf_calculated, rf_calculated_4_plot) =\
                    self.cal_app_rf_syn(self.vel_per)
            
            else:
            #with sens
                (app_calculated, rf_calculated, jacob_rf, jacob_app, 
                 rf_calculated_4_plot) = \
                    self.cal_app_rf_syn_ws(self.vel_per)
                
                
                (jacob_app, jacob_rf,
                          _, 
                          _)= self.normalize_jacob(jacob_app,
                          jacob_rf,
                          self.smooth_mat_virgin, 
                          self.damp_mat_virgin)
                self.jacob_app = copy.deepcopy(jacob_app)
                self.jacob_rf = copy.deepcopy(jacob_rf)
            #==============================================
            
            dif_app = np.array(self.app_obs) - np.array(app_calculated)
            dif_rf = np.array(self.rf_obs) - np.array(rf_calculated)
            smooth_obs = self.create_smooth_obs(self.smooth_mat_virgin)
            (damp_mat, damp_obs) = self.create_damp_obs() 
            
            (dif_app, dif_rf, 
             smooth_obs, damp_obs) = self.normalize_obs(dif_app,
                               dif_rf,
                               smooth_obs, 
                               damp_obs)
                
               
            self.dif_app = dif_app.copy()
            self.dif_rf = dif_rf.copy() 
            self.smooth_obs = smooth_obs.copy() 
            self.damp_obs = damp_obs.copy()
            
            # # ##test part ==============================
            # self.test_vel(self.jacob_app, self.jacob_rf, 
            #               self.smooth_mat, self.damp_mat, 
            #               self.dif_app, self.dif_rf, 
            #               self.smooth_obs, self.damp_obs 
            #               , curr_iter = self.iter_all)
             # ##========================================
        
            self.jacob_all = np.vstack((self.jacob_app, self.jacob_rf))
            self.obs_all = np.concatenate((self.dif_app, self.dif_rf))
            
            
            
            self.vel_curr = self.calculate_vel(vel_p= self.vel_per)
            self.lthickness_cur = \
               self.update_layer_thickness(curr_vel = self.vel_curr, 
                                       per_vel= self.vel_per, 
                                       per_lthickness = self.lthickness_cur)
            self.mean_dif_vel = np.mean(np.abs(self.vel_curr - self.vel_per))
    def initial_cal_cond(self):
        self.create_weight_mats()
        smooth_mat = self.create_smooth(derivative= 'second')
        self.smooth_mat_virgin = smooth_mat.copy()
        smooth_obs = self.create_smooth_obs(self.smooth_mat_virgin)
        self.smooth_obs_virgin = copy.deepcopy(smooth_obs)
        (damp_mat, damp_obs) = self.create_damp_obs() 
        self.damp_mat_virgin = copy.deepcopy(damp_mat)
        self.damp_obs_virgin = copy.deepcopy(damp_obs)
        
        if (self.force_jacob_cal):
            (app_calculated, rf_calculated, jacob_rf, jacob_app, 
             rf_calculated_4_plot) = \
                self.cal_app_rf_syn_ws(self.vel_init)
            self.save_jacobs(app_calculated, 
                             rf_calculated, 
                             jacob_rf, 
                             jacob_app)
        else:
            (app_calculated, rf_calculated, 
                             jacob_rf, 
                             jacob_app) = self.load_jacobs()
        
        
        self.normalize_factors(jacob_app,
                               jacob_rf,
                               self.smooth_mat_virgin, 
                               self.damp_mat_virgin)
        #initiate parameters
        self.app_calculated_curr = copy.deepcopy(app_calculated)
        self.rf_calculated_curr = copy.deepcopy(rf_calculated)
        self.app_calculated_init = copy.deepcopy(app_calculated)
        self.rf_calculated_init = copy.deepcopy(rf_calculated)
        self.rf_calculated_init_4_plot = np.zeros(shape = 
                                                  np.shape(self.time_rf))
        dif_len = len(self.time_rf) - len(rf_calculated)
        self.rf_calculated_init_4_plot[:dif_len] = 0.0
        self.rf_calculated_init_4_plot[dif_len:] = rf_calculated
        
        self.jacob_rf_virgin = copy.deepcopy(jacob_rf)
        self.jacob_app_virgin = copy.deepcopy(jacob_app)
        dif_app = np.array(self.app_obs) - np.array(app_calculated)
        dif_rf = np.array(self.rf_obs) - np.array(rf_calculated)
        self.dif_app_virgin = copy.deepcopy(dif_app)
        self.dif_rf_virgin = copy.deepcopy(dif_rf)
        
        
        ###test part ==============================   
        # self.test_vel(self.jacob_app_virgin, 
        #               self.jacob_rf_virgin, 
        #               self.smooth_mat_virgin, self.damp_mat_virgin,
        #               self.dif_app_virgin, self.dif_rf_virgin, 
        #               self.smooth_obs_virgin, self.damp_obs_virgin, 
        #               curr_iter= -1)
        ###========================================
        
        
        (jacob_app, jacob_rf,
                     smooth_mat, 
                     damp_mat)= self.normalize_jacob(self.jacob_app_virgin,
                     self.jacob_rf_virgin,
                     self.smooth_mat_virgin, 
                     self.damp_mat_virgin)
        self.smooth_mat = copy.deepcopy(smooth_mat)
        self.damp_mat = copy.deepcopy(damp_mat)
        self.jacob_app = copy.deepcopy(jacob_app) 
        self.jacob_rf = copy.deepcopy(jacob_rf)
        
        (dif_app, dif_rf, 
         smooth_obs, damp_obs) = self.normalize_obs(self.dif_app_virgin,
                           self.dif_rf_virgin,
                           self.smooth_obs_virgin, 
                           self.damp_obs_virgin)
        ##test part ==============================
        # self.test_vel(jacob_app, jacob_rf, 
        #               smooth_mat, damp_mat, dif_app, 
        #               dif_rf, smooth_obs, damp_obs, curr_iter = self.iter_all)
        ##========================================
        
        self.dif_app = copy.deepcopy(dif_app)
        self.dif_rf = copy.deepcopy(dif_rf) 
        self.smooth_obs = copy.deepcopy(smooth_obs)
        self.damp_obs = copy.deepcopy(damp_obs)
        
        
        
        self.dif_rf_norm_init = self.cal_norm(dif_rf)
        self.dif_app_norm_init = self.cal_norm(dif_app)
        
        self.dif_smooth_norm_init = self.cal_norm(self.smooth_obs)
        self.dif_damp_norm_init = self.cal_norm(self.damp_obs)
        

        
        self.jacob_all = np.vstack((self.jacob_app, self.jacob_rf))
        self.obs_all = np.concatenate((dif_app, dif_rf))

        self.obj_init_rf = self.cal_norm(dif_rf)
        self.obj_init_app = self.cal_norm(dif_app)
        self.norm_init = self.cal_norm(self.obs_all)
        
        self.dif_all_norm_per = self.norm_init
        self.vel_s_best_per = self.vel_init.copy() 
        self.cond_best_per = self.norm_init 
        self.rf_curve_best_per = rf_calculated
        self.app_curve_best_per = app_calculated
        
        self.dif_all_norm_best = self.norm_init * 20
        self.vel_s_best_curr = self.vel_init.copy() 
        self.cond_best_curr = self.norm_init 
        self.rf_curve_best_curr = rf_calculated
        self.app_curve_best_curr = app_calculated
    def create_smooth(self, derivative = 'second'):
         smooth_mat = np.zeros(shape=(len(self.vel_init), len(self.vel_init)))
         if (derivative == 'second'):
             for i in range(len(self.vel_init)):
                 for j in range(len(self.vel_init)):
                     if ((i == j) and (j-1 >= 0) and (j+1) < len(self.vel_init)):
                         smooth_mat[i, j] = -2.0
                     elif ((i == j) and (j-1 > 0) and (j+1) == len(self.vel_init)):
                         smooth_mat[i, j] = -1.0
                     elif ((i == j) and (j-1 == -1) and (j+1) < len(self.vel_init)):
                         smooth_mat[i, j] = -1.0
                     elif ((i == j-1) or (i == j+1)):
                         smooth_mat[i, j] = 1.0
                     else:
                         smooth_mat[i, j] = 0.0
             smooth_mat = smooth_mat
         elif (derivative == 'first'):
             for i in range(len(self.vel_init) -1 ):
                 for j in range(len(self.vel_init)):
                     if ((i==j)):
                         smooth_mat[i,j] = -1.0
                         smooth_mat[i, j+1]= 1.0
                     
             smooth_mat[len(self.vel_init)-1,len(self.vel_init)-1] = -1.0
             smooth_mat = smooth_mat
         return(smooth_mat)
    def create_smooth_obs(self, smooth_mat):
         smooth_obs = np.zeros(shape=(len(self.vel_init),))
         zero_smooth = np.zeros(shape=(len(self.vel_init),))
         vel_smooth = np.array(self.vel_per)
         smooth_obs = zero_smooth - np.matmul(smooth_mat, 
                                              vel_smooth)
         return(smooth_obs)
    def create_damp_obs(self):
         if (self.force_first_layer):
            self.vel_init[0] = self.app_obs[0]
         damp_mat = np.diag(np.ones(shape = (self.nlayer,)))
         damp_mat = damp_mat
         damp_obs = np.zeros(shape=(len(self.vel_init),))
         damp_obs = np.array(self.vel_init) - np.array(self.vel_per)
         
         #fixing layer 1 vel 
         # damp_mat[0,:] = damp_mat[0,:] * 1000.0
         return(damp_mat, damp_obs)
    def save_jacobs(self, app_calculated, 
                    rf_calculated, 
                    jacob_rf, 
                    jacob_app):
        if (self.folder_jacob_save == 'current'):
            folder = self.save_dir 
        else:
            folder = os.path.join(os.getcwd(), 'dummy_folder_for_inverse')
        if os.path.isdir(folder):
            pass
        else:
            os.mkdir(folder)
        name = folder + 'jacob_rf.bin'
        file1 = open(name, "wb") 
        pickle.dump(jacob_rf, file1)
        file1.close
        
        name = folder + 'jacob_app.bin'
        file1 = open(name, "wb") 
        pickle.dump(jacob_app, file1)
        file1.close
        
        name = folder + 'app_calculated.bin'
        file1 = open(name, "wb") 
        pickle.dump(app_calculated, file1)
        file1.close
        
        name = folder + 'rf_calculated.bin'
        file1 = open(name, "wb") 
        pickle.dump(rf_calculated, file1)
        file1.close
    def fix_rf_obs_from_zero(self, samp_shift = 15):
        if (self.inv_time_rf1 != 0.0):
            ind_zero = int(self.inv_time_rf1 * np.round(1 / self.dt, 1))
            self.rf_obs_4_plot = self.rf_obs.copy()
            self.rf_obs = self.rf_obs[ind_zero - samp_shift:].copy()
            
            
    def load_jacobs(self):
        if (self.folder_jacob_save == 'current'):
            folder = self.save_dir 
        else:
            folder = '/home/soroush/rf_shallow_codes/my_py_rf/inverse_input/'
            
        name = folder + 'jacob_rf.bin'
        with open(name, 'rb') as f1:
            jacob_rf = pickle.load(f1)
        f1.close    
        
        name = folder + 'jacob_app.bin'
        with open(name, 'rb') as f1:
            jacob_app = pickle.load(f1)
        f1.close
        
        name = folder + 'app_calculated.bin'
        with open(name, 'rb') as f1:
            app_calculated = pickle.load(f1)
        f1.close
        
        name = folder + 'rf_calculated.bin'
        with open(name, 'rb') as f1:
            rf_calculated = pickle.load(f1)
        f1.close
        return(app_calculated, rf_calculated, jacob_rf, jacob_app)
    def create_weight_mats(self):
        if (np.array(self.Cm_array).any() == False):
            self.Cm_array = np.ones(shape= (self.nlayer,))
            if (self.force_last_layer):
                self.Cm_array[-1] = 0.001
            if (self.force_first_layer):
                self.Cm_array[0] = 0.001
        if (np.array(self.Cd_array_rf).any() == False):
            self.Cd_array_rf = np.ones(shape= (len(self.rf_obs),))
        if (np.array(self.Cd_array_app).any() == False):
            self.Cd_array_app = np.ones(shape= (len(self.app_obs),))
        if (np.array(self.Cb_array).any() == False):
            self.Cb_array = np.ones(shape= (self.nlayer,))
        
        Cm = np.diag(self.Cm_array)
        Cb = np.diag(self.Cb_array)
        Cd_array = np.concatenate((self.Cd_array_app, self.Cd_array_rf))
        Cd = np.diag(Cd_array)
        
        self.Cmm1 = np.linalg.inv(Cm)
        self.Cdm1 = np.linalg.inv(Cd)
        self.Cbm1 = np.linalg.inv(Cb)  
    
    def normalize_factors(self, jacob_app, jacob_rf, smooth_mat, damp_mat):
        if (self.obs_normalize):
            [napp, _] = np.shape(jacob_app)
            [nrf, _] = np.shape(jacob_rf)
            nl = self.nlayer
        else:
            napp = 1
            nrf = 1
            nl = 1
        j_app = jacob_app * np.sqrt(1 / napp) 
        j_rf = jacob_rf * np.sqrt(1 / nrf)
        j_damp = damp_mat * np.sqrt(1 / nl)
        j_smooth = smooth_mat * np.sqrt(1 / nl)
        
        max_rf_mat = np.max(abs(j_rf))
        max_app_mat = np.max(abs(j_app))
        max_damp_mat = np.max(abs(j_damp))
        max_smooth_mat = np.max(abs(j_smooth))
        
        self.app_weight = self.app_weight 
        self.rf_weight = ((max_app_mat / max_rf_mat)**2.0) * \
            self.rf_weight_virgin 
        self.damp_factor = ((max_app_mat / max_damp_mat)**2.0) *\
            self.damp_factor_virgin
        self.smooth_factor = ((max_app_mat / max_smooth_mat)**2.0) *\
            self.smooth_factor_virgin
        # print(123)
        
        
    def normalize_jacob(self, jacob_app, jacob_rf, smooth_mat, damp_mat):
        if (self.obs_normalize):
            [napp, _] = np.shape(jacob_app)
            [nrf, _] = np.shape(jacob_rf)
            nl = self.nlayer
        else:
            napp = 1
            nrf = 1
            nl = 1
        
        j_app = jacob_app * np.sqrt(self.app_weight / napp) 
        j_rf = jacob_rf * np.sqrt(self.rf_weight / nrf)
        j_damp = damp_mat * np.sqrt(self.damp_factor / nl)
        j_smooth = smooth_mat * np.sqrt(self.smooth_factor / nl)
        return(j_app, j_rf, j_smooth, j_damp)
    def normalize_obs(self, dif_app, dif_rf, dif_smooth, dif_damp):
        if (self.obs_normalize):
            napp = len(dif_app)
            nrf = len(dif_rf)
            nl = self.nlayer
        else:
            napp = 1
            nrf = 1
            nl = 1  
        d_app = dif_app * np.sqrt(self.app_weight / napp)
        d_rf = dif_rf * np.sqrt(self.rf_weight / nrf) 
        d_smooth = dif_smooth * np.sqrt(self.smooth_factor / nl)
        d_damp = dif_damp * np.sqrt(self.damp_factor / nl)
        return(d_app, d_rf, d_smooth, d_damp)
    def cal_norm(self, dif):
        dif_norm = np.sqrt(np.sum(dif**2.0))
        return(dif_norm)
    def calculate_vel(self, vel_p):
        
        #first expression
        G = self.jacob_all.copy() 
        B = self.smooth_mat.copy()
        D = self.damp_mat.copy()
        
        
        Gt = np.transpose(G)
        Bt = np.transpose(B)
        
        
        GtCdm1 = np.matmul(Gt, self.Cdm1)
        GtCdm1G = np.matmul(GtCdm1, G)
        
        BtCbm1 = np.matmul(Bt, self.Cbm1)
        BtCbm1B = np.matmul(BtCbm1, B)
        
        DtCmm1 = np.matmul(D, self.Cmm1)
        DtCmm1D = np.matmul(DtCmm1, D)
        
        FE = GtCdm1G + BtCbm1B + DtCmm1D 
        FE_inv = np.linalg.inv(FE)
        
        
        #second expression
        delta_d = self.obs_all.copy()
        delta_b = self.smooth_obs.copy() 
        delta_m = self.damp_obs.copy() 
        
        GtCdm1delta_d = np.matmul(GtCdm1, delta_d)
        BtCbm1delta_b = np.matmul(BtCbm1, delta_b)
        DtCmm1delta_m = np.matmul(DtCmm1, delta_m)
        
        SE = GtCdm1delta_d + BtCbm1delta_b + DtCmm1delta_m
        
        #cal velocity
        vel_new = vel_p + np.matmul(FE_inv, SE)
        
        bad_model = False
        for el in vel_new:
            if ((el > 6.0) or (el < 0.5)):
                bad_model = True 
        if (bad_model):
            vel_new = vel_p.copy()
            vel_new = np.array(vel_new)
            
        return(vel_new)
    
    def find_cond_best(self):
        self.cal_cond()
        
        if (self.dif_all_norm_curr < self.dif_all_norm_best):
            self.dif_all_norm_best = self.dif_all_norm_curr
            self.vel_s_best_curr = self.vel_curr.copy() 
            self.cond_best_curr = self.cond_curr 
            self.rf_curve_best_curr = self.rf_calculated_curr
            self.app_curve_best_curr = self.app_calculated_curr
            
            
            self.best_rf_cond = self.dif_rf_norm_curr
            self.best_app_cond = self.dif_app_norm_curr
            self.best_dif_rf = self.dif_rf 
            self.best_dif_app = self.dif_app
            
            self.best_lthickness = self.lthickness_cur
            self.vel_s_estimate = self.vel_curr.copy()
            self.vel_s_norm = self.cal_norm(self.vel_curr)
            self.best_cond_estimate = self.cond_curr
            self.best_rf_curve_estimate = self.rf_curve_best_curr
            self.best_app_curve_estimate = self.app_curve_best_curr
            self.best_af_iter = self.iter_all
    def cal_cond(self):
        (app_calculated, rf_calculated, rf_calculated_4_plot) = \
            self.cal_app_rf_syn(self.vel_curr)
        self.app_calculated_curr = app_calculated.copy()
        # self.rf_calculated_curr = rf_calculated.copy()
        self.rf_calculated_curr = rf_calculated_4_plot.copy()
        
        
        
        
        self.dif_rf_norm_curr = self.cal_norm(self.dif_rf)
        self.dif_app_norm_curr = self.cal_norm(self.dif_app)
        
        self.dif_smooth_norm_curr = self.cal_norm(self.smooth_obs)
        self.dif_damp_norm_curr = self.cal_norm(self.damp_obs)
            
        dif_all = np.concatenate((self.dif_app, self.dif_rf))
        self.dif_all_norm_curr = self.cal_norm(dif_all)
        self.cond_curr = self.dif_all_norm_curr / self.norm_init
    def cal_app_rf_syn_ws(self, vel_s):
        fw_cal = Forward_cal(vel_s, self.lthickness_cur,
                             self.filt_list,
                             tshift=self.tshift,
                             nsamp=self.nsamp,
                             dt=self.dt,
                             inv_time_rf1 = self.inv_time_rf1, 
                             inv_time_rf2 = self.inv_time_rf2,
                             waterlevel= self.waterlevel,
                             slowness=self.slowness,
                             gauss_par = self.gauss_par,
                             rf_method_sens = 'waterlevel',
                             rf_method = self.rf_method,
                             rf_normalize = self.rf_normalize,
                             saving_directory= self.save_dir+'out_rf')
        app_calculated = fw_cal.apparant_vel_org.copy()
        rf_calculated = fw_cal.rf_r_4_inv.copy()
        rf_calculated_4_plot = fw_cal.rf_r_from_ind1.copy()
        self.time_rf = fw_cal.time_cuted
        # old sens =======================================
        # fw_cal.cal_sensivity(ref = 'vel_s', pert=self.pert_val)
        # fw_cal.cal_sensivity(ref = 'vel_p', pert= self.pert_val)
        # fw_cal.cal_sensivity(ref = 'rho', pert= self.pert_val)
        # fw_cal.cal_jacobian_rf()
        # fw_cal.cal_jacobian_app_vel()
        
        #new sense =======================================
        fw_cal.cal_sensivity_jacobson(ref = 'vel_s', pert=self.pert_val, 
                                          cal_rho_vp=True, plotter= False)
        fw_cal.cal_jacobian_rf_jacobson()
        fw_cal.cal_jacobian_app_vel_jacobson()
        
        #=================================================
        

        jacob_rf = fw_cal.jacobian_rf.copy()
        jacob_app = fw_cal.jacobian_app_vel.copy()
        return(app_calculated, rf_calculated, jacob_rf, jacob_app, 
               rf_calculated_4_plot)
    
    
    def cal_app_rf_syn(self, vel_s):
        fw_cal = Forward_cal(vel_s, self.lthickness_cur,
                             self.filt_list,
                             tshift=self.tshift,
                             nsamp=self.nsamp,
                             dt=self.dt,
                             inv_time_rf1 = self.inv_time_rf1, 
                             inv_time_rf2 = self.inv_time_rf2,
                             waterlevel= self.waterlevel,
                             slowness=self.slowness,
                             gauss_par = self.gauss_par,
                             rf_method = self.rf_method,
                             rf_normalize = self.rf_normalize,
                             saving_directory= self.save_dir+'out_rf')
        # fw_cal.run_code(cal_rho_vp = True, cal_ref= 'vel_s', 
        #                 rf_method_run = self.rf_method)
        app_calculated = fw_cal.apparant_vel_org.copy()
        rf_calculated = fw_cal.rf_r_4_inv.copy()
        rf_calculated_4_plot = fw_cal.rf_r_from_ind1.copy()
        self.time_rf = fw_cal.time_cuted
        return(app_calculated, rf_calculated, 
               rf_calculated_4_plot)
    def check_for_stop(self):
        self.stop_run = False 
        if (self.iter_all > self.max_iter):
            if (self.out_kind == 'iterative'):
                self.close_iter()
            self.stop_run = True
        else:
            if (self.cond_curr < self.norm_cond):
                if (self.out_kind == 'iterative'):
                    self.close_iter()
                self.stop_run = True
    def close_iter(self):
        self.best_lthickness = self.lthickness_cur
        self.vel_s_estimate = self.vel_curr.copy() 
        self.best_cond_estimate = self.cond_curr
        self.best_rf_curve_estimate = self.rf_calculated_curr
        self.best_app_curve_estimate = self.app_calculated_curr
        self.best_af_iter = self.iter_all
    def update_layer_thickness(self, curr_vel, per_vel, 
                               per_lthickness):
        
        curr_lthickness = []
        for i in range(self.nlayer -1):
            # print(self.slowness, per_vel[i], 1.732 *per_vel[i], 
            #       curr_vel[i], .732 *curr_vel[i])
            numerator1 = (np.sqrt((per_vel[i]**-2.0) -
                                  (self.slowness) ** 2.0))
            numerator2 = (np.sqrt(((1.732 *per_vel[i])**-2.0) -
                                  (self.slowness) ** 2.0))
            denominator1 =(np.sqrt((curr_vel[i]**-2.0) -
                                  (self.slowness) ** 2.0))
            denominator2 = (np.sqrt(((1.732 *curr_vel[i])**-2.0) -
                                  (self.slowness) ** 2.0))
            numerator = numerator1 - numerator2 
            denominator = denominator1 - denominator2 
            
            curr_lthickness.append(per_lthickness[i] *
                                   (numerator/ denominator))
        curr_lthickness.append(0)
        return(curr_lthickness)
    def plot_estimates(self, niter=1, save_to_dir= False):
        
        label_size = 18
        tick_size = 14
        lthick_abs = ut.find_lthickness_abs(self.best_lthickness)
        (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
            self.find_xminmax(lthick_abs, self.vel_s_estimate)
        ##ploting figure
        fig, axes = plt.subplots(figsize = (32, 20),
                                 nrows=1, ncols = 3,
                     gridspec_kw={'width_ratios': [1, 0.5, 0.5]}, 
                     facecolor='#FAEBD7')
        fig.suptitle('Norm = '+ '{0:4.2f}'.format(self.best_cond_estimate) + 
                     ', After iteration no: ' + str(self.best_af_iter), 
                     fontsize = 32)
        ##ploting velocities
        axes[0].vlines(v_line_x, y_min, y_max, colors='navy', lw = 3.0, 
                   label=' Estimated Velocity')
        axes[0].hlines(h_line_y, x_min, x_max, colors='navy', lw = 3.0)
        lthick_abs_init = ut.find_lthickness_abs(self.layer_thickness_initial)       
        (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
            self.find_xminmax(lthick_abs_init, self.vel_init)
            
        axes[0].vlines(v_line_x, y_min, y_max, colors='red', ls= '--',lw = 1, 
                   label=' Initial Velocity')
        axes[0].hlines(h_line_y, x_min, x_max, colors='red', ls= '--',lw = 1)
        if (self.synthetic):
            lthick_abs_syn = ut.find_lthickness_abs(self.layers_thickness_syn)
            (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
                self.find_xminmax(lthick_abs_syn, self.vel_syn)
            axes[0].vlines(v_line_x, y_min, y_max, colors='k', lw = 1.5, 
                   label=' Synthetic Velocity')
            axes[0].hlines(h_line_y, x_min, x_max,colors='k', lw = 1.5)
            axes[0].grid(color = 'gray', which= 'both', alpha = 0.5, 
                         axis = 'both')
            axes[0].legend(fontsize=18)
        axes[0].set_ylim([0, lthick_abs[-2] + 8])
        axes[0].set_ylim(axes[0].get_ylim()[::-1])
        # axes[0].set_xlim([1.5, 5.0])
        axes[0].grid(color = 'gray', which= 'both', alpha = 0.5, axis = 'both')
        axes[0].set_title('Estimated Velocity',
                  fontsize=label_size)
        axes[0].set_xlabel('S Velocity (km/s)', size = label_size)
        axes[0].set_ylabel('Depth (km)', size = label_size)
        ## ploting RFS
        axes[1].set_title('Estimated receiver function',
               fontsize=label_size)
        axes[1].plot(self.rf_obs_4_plot,
                     self.time_rf,
                     label='Observed',color='black',
                     lw = 4.0, alpha = 0.6)
        axes[1].plot(self.rf_calculated_init_4_plot, 
                     self.time_rf, ls= '--',
                     label='Initial', 
                     lw=1.6, color = 'red', alpha = 0.4)
        axes[1].plot(self.best_rf_curve_estimate, 
                     self.time_rf,
                     label='Estimate', 
                     lw=3.2, color = 'navy', alpha = 1.0)
        
        
        axes[1].tick_params(axis="x", labelsize= tick_size)
        axes[1].tick_params(axis="y", labelsize= tick_size)
        axes[1].set_ylabel('Time (s)', size = label_size)
        axes[1].set_ylim([max(self.time_rf), 
                          min(self.time_rf)])
        axes[1].grid(color = 'gray', which= 'both', alpha = 0.5, axis = 'both')
        
        #plotting app_curve 
        axes[2].set_title('Estimated Apparent curve with', 
               fontsize=label_size)
        axes[2].plot(self.app_obs,
                     self.filt_list,
                     label='Observed',color = 'black',
                     lw = 4, alpha = 0.6)
        
        axes[2].plot(self.app_calculated_init,
                     self.filt_list,
                     ls= '--', label='Initial',color = 'red',
                     lw = 1.6, alpha = 0.4)
        axes[2].plot(self.best_app_curve_estimate, 
                     self.filt_list,
                     label='Estimate', 
                     lw = 3.2, color = 'navy', alpha = 1.0)
        
        axes[2].tick_params(axis="x", labelsize= tick_size)
        axes[2].tick_params(axis="y", labelsize= tick_size)
        axes[2].set_ylabel('Filter Periods (s)', size = label_size)
        axes[2].set_ylim(axes[1].get_ylim())
        axes[2].grid(color = 'gray', which= 'both', alpha = 0.5, axis = 'both')
        if (save_to_dir):
            name = ('velocity_smW_'+str(self.smooth_factor)+
                    '_rfW_'+str(self.rf_weight)+
                    '_dW_'+str(self.damp_factor)+
                    '_appW'+str(self.app_weight)+'_.png')
            name = self.save_dir + name
            plt.savefig(name)
            if (self.close_fig):
                fig.clf()
                plt.cla()
                plt.close(fig)
    
    def find_xminmax(self, depth, vel):
        y_min = []
        y_max = []
        x_min = []
        x_max = []
        v_line_x = vel.copy()
        for i in range(len(depth)):
            if ( i == 0):
                y_min.append([0])
                y_max.append([depth[i]])
            else:
                y_min.append([depth[i-1]])
                y_max.append([depth[i]])
        for i in range(len(depth) - 1):
            x_min.append(vel[i])
            x_max.append(vel[i+1])
        h_line_y = y_max[0:len(y_max) - 1].copy()
        return(x_min, x_max, y_min, y_max, v_line_x, h_line_y)
    def test_vel(self, jacob_app, jacob_rf, 
                 smooth_mat, damp_mat, dif_app, 
                 dif_rf, smooth_obs, damp_obs, curr_iter):
        labels_vel = []
        out_all = np.zeros(shape = (self.nlayer, 17))
        out_all[:,0] = self.vel_per.copy()
        labels_vel.append('vel_init')
        out_all[:,1] = self.vel_syn.copy()
        labels_vel.append('vel_syn')
        
        out = np.linalg.lstsq(damp_mat, damp_obs)
        vel_damp = out[0] 
        vel_damp_abs = vel_damp + self.vel_per
        out_all[:,2] = vel_damp_abs.copy()
        labels_vel.append('vel_damp')
        
        out = np.linalg.lstsq(smooth_mat , smooth_obs)
        vel_smooth = out[0] 
        vel_smooth_abs = vel_smooth + self.vel_per
        out_all[:,3] = vel_smooth_abs.copy()
        labels_vel.append('vel_smooth')
        
        
        mat = np.vstack((smooth_mat, damp_mat))
        obs = np.concatenate((smooth_obs, damp_obs))
        out = np.linalg.lstsq(mat, obs)
        vel_smooth_damp = out[0] 
        vel_smooth_damp_abs = vel_smooth_damp + self.vel_per
        out_all[:,4] = vel_smooth_damp_abs.copy()
        labels_vel.append('vel_smooth_damp')
        
        
        out = np.linalg.lstsq(jacob_app, dif_app)
        vel_app = out[0] 
        vel_app_abs = vel_app + self.vel_per
        out_all[:,5] = vel_app_abs.copy()
        labels_vel.append('vel_app')
        
        
        mat = np.vstack((jacob_app, damp_mat))
        obs = np.concatenate((dif_app, damp_obs))
        out = np.linalg.lstsq(mat, obs)
        vel_app_damped = out[0]
        vel_app_damped_abs = vel_app_damped + self.vel_per
        out_all[:,6] = vel_app_damped_abs.copy()
        labels_vel.append('vel_app_damped')
        
        
        mat = np.vstack((jacob_app, smooth_mat))
        obs = np.concatenate((dif_app, smooth_obs))
        out = np.linalg.lstsq(mat, obs)
        vel_app_smoothed = out[0]
        vel_app_smoothed_abs = vel_app_smoothed + self.vel_per
        out_all[:,7] = vel_app_smoothed_abs.copy()
        labels_vel.append('vel_app_smoothed')
        
        
        mat = np.vstack((jacob_app, damp_mat))
        obs = np.concatenate((dif_app, damp_obs))
        mat = np.vstack((mat, smooth_mat))
        obs = np.concatenate((obs, smooth_obs))
        out = np.linalg.lstsq(mat, obs)
        vel_app_damped_smoothed = out[0]
        vel_app_damped_smoothed_abs = vel_app_damped_smoothed + self.vel_per
        out_all[:,8] = vel_app_damped_smoothed_abs.copy()
        labels_vel.append('vel_app_damped_smoothed')
        
        
        out = np.linalg.lstsq(jacob_rf, dif_rf)
        vel_rf = out[0] 
        # vel_rf[-1] = 0
        vel_rf_abs = vel_rf + self.vel_per
        out_all[:,9] = vel_rf_abs.copy()
        labels_vel.append('vel_rf')
        
        
        mat = np.vstack((jacob_rf, damp_mat))
        obs = np.concatenate((dif_rf, damp_obs))
        out = np.linalg.lstsq(mat, obs)
        vel_rf_damped = out[0]
        vel_rf_damped_abs = vel_rf_damped + self.vel_per 
        out_all[:,10] = vel_rf_damped_abs.copy()
        labels_vel.append('vel_rf_damped')
        
        
        mat = np.vstack((jacob_rf, smooth_mat))
        obs = np.concatenate((dif_rf, smooth_obs))
        out = np.linalg.lstsq(mat, obs)
        vel_rf_smoothed = out[0]
        vel_rf_smoothed_abs = vel_rf_smoothed + self.vel_per
        out_all[:,11] = vel_rf_smoothed_abs.copy()
        labels_vel.append('vel_rf_smoothed')
        
        
        mat = np.vstack((jacob_rf, damp_mat))
        obs = np.concatenate((dif_rf, damp_obs))
        mat = np.vstack((mat, smooth_mat))
        obs = np.concatenate((obs, smooth_obs))
        out = np.linalg.lstsq(mat, obs)
        vel_rf_damped_smoothed = out[0]
        vel_rf_damped_smoothed_abs = vel_rf_damped_smoothed + self.vel_per
        out_all[:,12] = vel_rf_damped_smoothed_abs.copy()
        labels_vel.append('vel_rf_damped_smoothed')
        
        
        mat = np.vstack((jacob_app, jacob_rf))
        obs = np.concatenate((dif_app, dif_rf))
        out = np.linalg.lstsq(mat, obs)
        vel_app_rf = out[0] 
        vel_app_rf_abs = vel_app_rf + self.vel_per
        out_all[:,13] = vel_app_rf_abs.copy()
        labels_vel.append('vel_app_rf')
        
        
        mat = np.vstack((jacob_app, jacob_rf))
        obs = np.concatenate((dif_app, dif_rf))
        mat = np.vstack((mat, smooth_mat))
        obs = np.concatenate((obs, smooth_obs))
        out = np.linalg.lstsq(mat, obs)
        vel_app_rf_smoothed = out[0] 
        vel_app_rf_smoothed_abs = vel_app_rf_smoothed + self.vel_per
        out_all[:,14] = vel_app_rf_smoothed_abs.copy()
        labels_vel.append('vel_app_rf_smoothed')
        
        
        mat = np.vstack((jacob_app, jacob_rf))
        obs = np.concatenate((dif_app, dif_rf))
        mat = np.vstack((mat, damp_mat))
        obs = np.concatenate((obs, damp_obs))
        out = np.linalg.lstsq(mat, obs)
        vel_app_rf_damped = out[0] 
        vel_app_rf_damped_abs = vel_app_rf_damped + self.vel_per
        out_all[:,15] = vel_app_rf_damped_abs.copy()
        labels_vel.append('vel_app_rf_damped')
        
        
        mat = np.vstack((jacob_app, jacob_rf))
        obs = np.concatenate((dif_app, dif_rf))
        mat = np.vstack((mat, damp_mat))
        obs = np.concatenate((obs, damp_obs))
        mat = np.vstack((mat, smooth_mat))
        obs = np.concatenate((obs, smooth_obs))
        out = np.linalg.lstsq(mat, obs)
        vel_app_rf_damped_smoothed = out[0] 
        vel_app_rf_damped_smoothed_abs = \
            vel_app_rf_damped_smoothed + self.vel_per
        out_all[:,16] = vel_app_rf_damped_smoothed_abs.copy()
        labels_vel.append('vel_rf_damped_smoothed')
        ut.plot_matrix(mat)
        
        
        for i in range(17):
            plt.figure(figsize= (8, 12))
            plt.ylim(3, 4.5)
            plt.scatter(range(self.nlayer), out_all[:,0])
            plt.plot(range(self.nlayer), out_all[:,0], label = labels_vel[0])
            plt.scatter(range(self.nlayer), out_all[:,1])
            plt.plot(range(self.nlayer), out_all[:,1], label = labels_vel[1])
            plt.scatter(range(self.nlayer), out_all[:,i])
            plt.plot(range(self.nlayer), out_all[:,i], label = labels_vel[i])
            plt.legend()
            plt.title('for iter= '+str(curr_iter))
            plt.grid(True)
        a = 1
        
    



#%%
class Forward_cal:
    def __init__(self, vel_s, layers_thickness, filt_list, kind='rftn',
                 gauss_par=2.5, dt=0.05,
                 nsamp=1024, tshift=10.0,
                 slowness=0.04, app_vel_obs='def',
                 cor_out='ZRT', filter_array='cal',
                 rf_method = 'iterative', rf_method_sens = 'waterlevel',
                 filter_dif_rf_ind= False,
                 waterlevel = 0.01,
                 rf_normalize= 1,
                 inv_time_rf1 = 5.0, inv_time_rf2 = 25,
                 vel_p = 'cal', rho='cal', 
                 noise_level = 0,
                 saving_directory= os.path.join(os.getcwd(), 'outforw'),
                 ):
        self.rf_normalize = rf_normalize
        if (app_vel_obs == 'def'):
            self.apparant_vel_obs = app_vel_obs
        else:
            self.apparant_vel_obs = app_vel_obs.copy()
        self.inv_time_rf1 = inv_time_rf1 
        self.inv_time_rf2 = inv_time_rf2
        self.waterlevel = waterlevel
        self.vel_s = vel_s.copy()
        self.vel_s_org = vel_s.copy()
        self.layers_thickness = layers_thickness
        self.filt_list = filt_list
        self.kind = kind
        self.gauss_par = gauss_par
        self.dt = dt
        self.nsamp = nsamp
        self.tshift = tshift
        self.slowness = slowness
        self.saving_directory = saving_directory
        self.cor_out = 'ZRT'
        self.filter_array = filter_array
        self.rf_method = rf_method
        self.rf_method_sens = rf_method_sens
        self.filter_dif_rf_ind = filter_dif_rf_ind
        self.sensivity = {}
        self.noise_level = noise_level
        max_depth = 0
        self.thickness_abs = []
        for dh in self.layers_thickness:
            max_depth = max_depth + dh
            self.thickness_abs.append(max_depth)
        self.max_depth = max_depth
        if (self.saving_directory[-1] != '/'):
            self.saving_directory = self.saving_directory + '/'
        if (vel_p != 'cal'):
            self.vel_p_org = vel_p 
        if (rho != 'cal'):
            self.rho_org = rho 
        if ((vel_p == 'cal') and (rho == 'cal')):
            self.run_code(first_run=True, cal_rho_vp= True,
                          rf_method_run =self.rf_method,
                          cal_ref='vel_s')
            self.vel_p_org = self.vel_p.copy()
            self.rho_org = self.rho.copy()
            self.vel_s_org = self.vel_s.copy()
        else:
            self.run_code(first_run=True, cal_rho_vp= False, 
                          rf_method_run =self.rf_method)
        
    def run_code(self, cal_rho_vp=True,
                 cal_ref='vel_s',
                 first_run=False, rf_method_run = 'waterlevel'):

        if (os.path.isdir(self.saving_directory)):
            shutil.rmtree(self.saving_directory)
            os.mkdir(self.saving_directory)
        else:
            os.mkdir(self.saving_directory)
        self.create_model(cal_rho_vp=cal_rho_vp, cal_ref=cal_ref)
        if (self.kind == 'rftn'):
            tr_dum = self.run_rftn()
            # self.time_vec = tr_dum[:,0] - self.tshift
            # self.ind1_4_rf = np.argwhere(abs(self.time_vec + self.inv_time_rf1)
            #                              == min(abs(self.time_vec +
            #                                         self.inv_time_rf1)))[0][0]
            # self.ind2_4_rf = np.argwhere(abs(self.time_vec - self.inv_time_rf2)
            #                              == min(abs(self.time_vec -
            #                                         self.inv_time_rf2)))[0][0]
            # self.time_cuted = self.time_vec[self.ind1_4_rf:self.ind2_4_rf]
            # tr_dum_c = tr_dum[self.ind1_4_rf:self.ind2_4_rf, :]
            # tr_dum = tr_dum_c.copy()
        elif (self.kind == 'raysum'):
            tr_dum = self.run_raysum()
            # self.time_vec = self.traces[:,0] - self.tshift
            # self.ind1_4_rf = np.argwhere(abs(self.time_vec + self.inv_time_rf1)
            #                              == min(abs(self.time_vec +
            #                                         self.inv_time_rf1)))[0][0]
            # self.ind2_4_rf = np.argwhere(abs(self.time_vec - self.inv_time_rf2)
            #                              == min(abs(self.time_vec -
            #                                         self.inv_time_rf2)))[0][0]
            # self.time_cuted = self.time_vec[self.ind1_4_rf:self.ind2_4_rf]
            # tr_dum_c = tr_dum[self.ind1_4_rf:self.ind2_4_rf, :]
            # tr_dum = tr_dum_c.copy()
        if (self.noise_level != 0.0):
            tr_dum = self.add_white_noise(tr_dum)
        
        self.tr_dum = copy.deepcopy(tr_dum)
            
        curve_out = App_curve(tr_dum, self.filt_list, nsamp=self.nsamp,
                              dt=self.dt, slowness=self.slowness,
                              gauss_par_4_rf=self.gauss_par,
                              rf_method=rf_method_run,
                              tshift_initial=self.tshift,
                              waterlevel=self.waterlevel,
                              tshift=self.inv_time_rf1,
                              filter_array=self.filter_array, amp_at=0.0,
                              normalize= self.rf_normalize)
        curve_out.cal_rf()
        self.curve_out = curve_out
        if (self.filter_array == 'cal'):
            curve_out.retrive_filters()
            self.filter_array = curve_out.filter_array
            curve_out.find_vel_rfs2()
        else:
            curve_out.find_vel_rfs2() 
    
        self.apparant_vel = []
        self.apparant_vel = curve_out.vel_app.copy()
        if (first_run):
            self.apparant_vel_org = curve_out.vel_app.copy()
            self.traces = tr_dum.copy()
            
            self.traces_rf_z = curve_out.rf_z.copy()
            self.traces_rf_r = curve_out.rf_r.copy()
            self.time_vec = tr_dum[:,0] - self.tshift
            self.ind0_4_rf = int(self.tshift * np.round(1 / self.dt, 1))
            self.ind1_4_rf = np.argwhere(abs(self.time_vec + self.inv_time_rf1)
                                         == min(abs(self.time_vec +
                                                    self.inv_time_rf1)))[0][0]
            self.ind2_4_rf = np.argwhere(abs(self.time_vec - self.inv_time_rf2)
                                         == min(abs(self.time_vec -
                                                    self.inv_time_rf2)))[0][0]
            # self.time_cuted = self.time_vec[self.ind1_4_rf:self.ind2_4_rf]
            self.time_cuted = self.time_vec[self.ind1_4_rf:self.ind2_4_rf]
            self.rf_r_from_ind1 = self.cut_rf(self.traces_rf_r)
            self.rf_z_from_ind1 = self.cut_rf(self.traces_rf_z)
            
            self.rf_r_4_inv = self.cut_rf_from_zero(self.traces_rf_r)
            self.rf_z_4_inv = self.cut_rf_from_zero(self.traces_rf_z)
            # self.rf_r_4_inv = self.traces_rf_r
    def cut_rf(self, rf):
        rf_cutted = rf[self.ind1_4_rf:self.ind2_4_rf]
        return(rf_cutted)
    def cut_rf_from_zero(self, rf, samp_shift = 15):
        rf_cutted = rf[self.ind0_4_rf - samp_shift:self.ind2_4_rf]
        return(rf_cutted)
    def add_white_noise(self, tr_dum):
        # Set a target SNR
        noise_level = self.noise_level
        trace_radial = tr_dum[:, 2].copy()
        trace_transverse = tr_dum[:, 3].copy()
        trace_vertical = tr_dum[:, 4].copy()
        
        traces =[] 
        traces.append(trace_radial)
        traces.append(trace_vertical)
        traces_after_noise = []
        for trace in traces:
            
            # Calculate signal power and convert to dB 
            sig_avg = np.mean(trace)
            sig_avg_db = 10 * np.log10(sig_avg) 
            target_snr_db = sig_avg_db * noise_level/100
            # Calculate noise according to [2] then convert to watts
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            # Generate an sample of white noise
            mean_noise = 0
            noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), 
                                     len(trace))
            # Noise up the original signal
            
            trace_after_noise = trace + noise
            traces_after_noise.append(trace_after_noise)
        tr_dum2 = tr_dum.copy()
        tr_dum2[:,2] = traces_after_noise[0].copy()
        tr_dum2[:,4] = traces_after_noise[1].copy()
        
        # plt.figure()
        # plt.plot(tr_dum2[:,0], tr_dum[:,2], label = 'r_initial')
        # plt.plot(tr_dum2[:,0], tr_dum2[:,2], 
        #          label = 'r_noisy', alpha = 0.5)
        # plt.legend()
        
        # plt.figure()
        # plt.plot(tr_dum2[:,0], tr_dum[:,4], label = 'z_initial')
        # plt.plot(tr_dum2[:,0], tr_dum2[:,4], 
        #          label = 'z_noisy', alpha = 0.5)
        # plt.legend()
        tr_dum = []
        tr_dum = tr_dum2.copy()
        return(tr_dum)
    
    def cal_dif_app(self):
        if (self.apparant_vel_obs == 'def'):
            self.apparant_vel_obs = self.apparant_vel
        self.dif_app = []
        for i in range(len(self.apparant_vel)):
            self.dif_app.append(self.apparant_vel[i] -
                                self.apparant_vel_obs[i])
    def plot_rf_app_sens(self, rf_r_back, app_vel_back,sensivity_filt,
                         p_layer,ref= 'vel_s',
                         pert = 0.1,
                         bf = 5):
        fig, axs = plt.subplots(nrows=2, ncols=2,figsize=(16,8))
        t_dum = self.time_vec + bf 
        ind1 = np.argwhere(abs(t_dum) == np.min(abs(t_dum)))[0][0]
        t_dum = self.time_vec - self.inv_time_rf2
        # t_dum = self.time_vec
        ind2 = np.argwhere(abs(t_dum) == np.min(abs(t_dum)))[0][0]
        
        t_dum = self.time_vec 
        ind1_app = np.argwhere(abs(t_dum) == np.min(abs(t_dum)))[0][0]
        t_dum = self.time_vec - np.max(self.filt_list)
        ind2_app = np.argwhere(abs(t_dum) == np.min(abs(t_dum)))[0][0]
        
        axs[0,0].plot(self.time_vec[ind1:ind2],rf_r_back[ind1:ind2], color = 'blue'
                      ,lw= 1.5, label= 'rf_r_init')
        axs[0,0].plot(self.time_vec[ind1:ind2],self.curve_out.rf_r[ind1:ind2],
                      color = 'red'
                      , lw = 0.7, label= 'rf_r_perturbed')
        axs[0,0].legend()
        
        axs[0,0].set_title('RF before and after perturbation')
        axs[0, 0].grid(True, color='0.6', dashes=(5, 2, 1, 2))
        dif_rf_r = np.array(self.curve_out.rf_r)- np.array(rf_r_back)
        axs[0,1].plot(self.time_vec[ind1_app:ind2], dif_rf_r[ind1_app:ind2])
        axs[0,1].set_title('difference of RF before and after perturbation')
        axs[0,1].grid(True, color='0.6', dashes=(5, 2, 1, 2))
        axs[1,0].plot(self.filt_list, app_vel_back, color = 'blue',
                      lw = 1.5, label = 'app vel before pert')
        axs[1,0].plot(self.filt_list, app_vel_back, color = 'red',
                      lw = 0.7, label = 'app vel after pert')
        axs[1,0].legend()
        axs[1,0].set_title('app vel before and after perturbation')
        axs[1,0].grid(True, color='0.6', dashes=(5, 2, 1, 2))
        dif_app_vel = np.array(self.apparant_vel)- np.array(app_vel_back) 
        axs[1,1].plot(self.filt_list, dif_app_vel)
        axs[1,1].set_title('difference of app vel before '+
                           ' and after perturbation')
        axs[1,1].grid(True, color='0.6', dashes=(5, 2, 1, 2))
        fig.suptitle('perturbed layer= '+ str(p_layer) + ' at depth '+
                     str(self.thickness_abs[p_layer])+ '\n'+
                     'ref= '+ref +' perturbation= '+str(pert))
        
    def update_layer_thickness(self, curr_par, per_par, 
                            per_lthickness, par_kind = 'vel_s'):
        curr_lthickness = []
        if (par_kind == 'vel_s'):
            per_vel = per_par.copy()
            curr_vel = curr_par.copy()
        elif (par_kind == 'vel_p'):
            per_vel = []
            curr_vel = []
            for el in per_par:
                per_vel.append(el / 1.732 )
            for el in curr_par:
                curr_vel.append(el / 1.732)
            # per_vel = per_par / 1.732
            # curr_vel = curr_par / 1.732
        elif (par_kind == 'rho'):
            per_vel = []
            curr_vel = []
            for el in per_par:
                per_vel.append(((el - 0.541) / 0.36) / 1.732)
            for el in curr_par:
                curr_vel.append(((el - 0.541) / 0.36) / 1.732)
            # per_vel = ((per_par - 0.541) / 0.36) / 1.732
            # curr_vel = ((curr_par - 0.541) / 0.36) / 1.732
        for i in range(self.nlayer -1):
            numerator1 = (np.sqrt((per_vel[i]**-2.0) -
                                  (self.slowness) ** 2.0))
            numerator2 = (np.sqrt(((1.732 *per_vel[i])**-2.0) -
                                  (self.slowness) ** 2.0))
            denominator1 =(np.sqrt((curr_vel[i]**-2.0) -
                                   (self.slowness) ** 2.0))
            denominator2 = (np.sqrt(((1.732 *curr_vel[i])**-2.0) -
                                    (self.slowness) ** 2.0))
            numerator = numerator1 - numerator2 
            denominator = denominator1 - denominator2 
            
            curr_lthickness.append(per_lthickness[i] *
                                   (numerator/ denominator))
        curr_lthickness.append(0)
        return(curr_lthickness)
    def cal_sensivity_jacobson(self, ref='vel_s', pert=0.1, 
                      plotter = False, 
                      gauss_filt_alpha= 1.5,
                      cal_rho_vp=True):
        self.run_code(first_run=True, cal_rho_vp= True,
                      rf_method_run =self.rf_method_sens,
                      cal_ref='vel_s')
        filter_dif_rf_ind = self.filter_dif_rf_ind
        rf_r_back = self.traces_rf_r.copy()
        rf_z_back = self.traces_rf_z.copy()
        apparant_vel_back = self.apparant_vel.copy()
        sens_mat = np.zeros(shape=(len(self.filt_list), self.nlayer))
        sens_rf_mat = np.zeros(shape=(len(self.rf_r_4_inv), self.nlayer))
        if (ref == 'vel_s'):
            layers_thickness_back = self.layers_thickness.copy()
            vel_s_back = self.vel_s.copy()
            self.dif_app_all = []
            self.dif_rf_all  = []
            if (filter_dif_rf_ind):
                self.dif_rf_all_filtered = []
            for i in range(self.nlayer):
                pert_percent = (self.vel_s[i] / 100.0) * pert 
                self.vel_s[i] = self.vel_s[i] + pert_percent
                lthickness = self.update_layer_thickness(
                    curr_par= self.vel_s.copy(), 
                    per_par= vel_s_back.copy(),
                    per_lthickness = layers_thickness_back.copy(),
                    par_kind='vel_s')
                
                self.layers_thickness = lthickness.copy()
                thickness_abs = self.cal_thick_abs(lthickness)
                self.run_code(cal_rho_vp=cal_rho_vp, cal_ref=ref,
                              rf_method_run = self.rf_method_sens)
                dif_app = (np.array(self.apparant_vel) -
                           np.array(apparant_vel_back))
                self.dif_app_all.append(dif_app)
                for j in range(len(self.filt_list)):
                    # if (self.layers_thickness[i] > 0):
                    #     thickness = self.layers_thickness[i]
                    # else:
                    #     thickness = 80.0
                    sensivity_filt = dif_app[j] / (pert_percent )
                    if (self.layers_thickness[i] == 0.0):
                        # sensivity_filt = sensivity_filt/ 40
                        sensivity_filt = sensivity_filt
                    # sensivity_filt = dif_app[j] / (pert_percent * thickness)
                    
                    sens_mat[j, i] = sensivity_filt
                if (plotter):
                    self.plot_rf_app_sens(rf_r_back = rf_r_back,
                                              app_vel_back= apparant_vel_back,
                                              sensivity_filt= sensivity_filt,
                                              p_layer= i,
                                              ref = ref,
                                              pert = pert_percent)
                # self.run_code(cal_rho_vp=True, cal_ref=ref)
                rf_r_cutted = rf_r_back
                cur_rf_r_cutted= self.curve_out.rf_r
                
                rf_r_cutted = self.cut_rf_from_zero(rf_r_back)
                cur_rf_r_cutted = self.cut_rf_from_zero(self.curve_out.rf_r)
                #u fked up here
                # dif_rf = np.array(rf_r_cutted) - np.array(cur_rf_r_cutted)
                dif_rf =  np.array(cur_rf_r_cutted) - np.array(rf_r_cutted)
                if (filter_dif_rf_ind):
                    dif_rf_filtered = self.filter_dif_rf(dif_rf,
                                          gauss_filt_alpha=gauss_filt_alpha)
                    self.dif_rf_all_filtered.append(dif_rf_filtered.copy())
                if (filter_dif_rf_ind):
                    dif_rf_filtered_4_sens_cal = dif_rf_filtered.copy()
                else:
                    dif_rf_filtered_4_sens_cal = dif_rf.copy()
                for k in range(len(dif_rf_filtered_4_sens_cal)):
                    # if (self.layers_thickness[i] > 0):
                    #     thickness = self.layers_thickness[i]
                    # else:
                    #     thickness = 1.0
                    sensivity_sample = (dif_rf_filtered_4_sens_cal[k] 
                                        / (pert_percent))
                    # sensivity_sample = dif_rf[k] / (pert_percent )
                    sens_rf_mat[k, i] = sensivity_sample
                
            
                
                self.vel_s = []
                self.vel_s = vel_s_back.copy()
                self.traces_rf_r = [] 
                self.traces_rf_r = rf_r_back.copy()
                self.traces_rf_z = []
                self.traces_rf_z = rf_z_back.copy()
                self.layers_thickness = [] 
                self.layers_thickness = layers_thickness_back.copy()
                
                
            self.sens_vel_s = sens_mat
            self.sens_rf_vel_s = sens_rf_mat
            
        self.apparant_vel = apparant_vel_back.copy()
        self.traces_rf_r = rf_r_back.copy()
    def cal_thick_abs(self, thickness):
        thickness_abs = []
        sumd = 0.0 
        for el in thickness:
            sumd = sumd + el 
            thickness_abs.append(sumd)
        return(thickness_abs)
    def filter_dif_rf(self, dif_rf, gauss_filt_alpha = 2.5):
        dt_gf = self.dt 
        nft_gf = len(dif_rf)
        f0_gf = gauss_filt_alpha
        gf = ut._gauss_filter(dt_gf, nft_gf, f0_gf, waterlevel=None)
        filtered_dif_rf = ut._apply_filter(dif_rf, gf)
        return(filtered_dif_rf)
        
        
        
    def plot_diff_at_certain_freq(self, freq_ind):
        fig = plt.figure(figsize=(16, 10))
        x = []
        x.append(0)
        for i in range(1, len(self.layers_thickness)):
            x.append(x[i-1] + self.layers_thickness[i])
        x[-1] = self.max_depth + 4
        dif_all_arr = np.array(self.dif_app_all)
        what_to_plot = np.zeros(len(x))
        for i in range(len(self.dif_app_all)):
            for j in range(len(self.filt_list)):
                if (freq_ind == j):
                    freq = self.filt_list[freq_ind]
                    what_to_plot = (dif_all_arr[:, j])
        plt.title('difference of app velocity for period = '+str(freq)+'\n'
                  + ' versus Vs perturbation of different depth', fontsize=18)
        plt.xlabel('Depth (km)', fontsize=18)
        plt.ylabel('diff app velocity', fontsize=18)
        plt.plot(x, what_to_plot)
        plt.scatter(x, what_to_plot)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))

    def plot_dif_app(self, depth_ind='all'):
        y = []
        y.append(0)
        for i in range(1, len(self.layers_thickness)):
            y.append(y[i-1] + self.layers_thickness[i])
        y[-1] = self.max_depth + 4
        norm_c = matplotlib.colors.Normalize(vmin=min(y),
                                             vmax=max(y))
        fig = plt.figure(figsize=(16, 10))
        idum = -1
        for row in range(len(self.dif_app_all)):
            if (depth_ind == 'all'):
                what_plot = self.dif_app_all[row]
                idum += 1
                clrs = cm.jet(norm_c(y[idum]))
                plt.plot(self.filt_list, what_plot,
                         color=clrs)
            else:
                if (row == depth_ind):
                    what_plot = self.dif_app_all[row]
                    idum += 1
                    clrs = cm.jet(norm_c(y[idum]))
                    plt.plot(self.filt_list, what_plot,
                             color=clrs)

        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        if (depth_ind == 'all'):
            plt.title('difference between apparant velocities\n'
                      + 'colored by depth of perturbed layer', fontsize=18)
            sm = plt.cm.ScalarMappable(cmap='jet', norm=norm_c)
            sm.set_array([])
            clb_label2 = 'Dept (km)'
            cb = plt.colorbar(sm, label=clb_label2,
                              fraction=0.02, pad=0.04, location='right')
        else:
            plt.title('difference between apparant velocities\n'
                      + 'for perturbed Vs at depth= '+str(y[depth_ind]), fontsize=18)
        plt.ylabel('difference of app velocity', fontsize=18)
        plt.xlabel('filter period (s)', fontsize=18)
    def plot_sensivity(self, ref='vel_s', freq_ind='all'):
        norm_c = matplotlib.colors.Normalize(vmin=min(self.filt_list),
                                             vmax=max(self.filt_list))
        y = []
        y.append(0)
        for i in range(1, len(self.layers_thickness)):
            y.append(y[i-1] + self.layers_thickness[i])
        y[-1] = self.max_depth + 4

        if (ref == 'vel_s'):
            to_plot = self.sens_vel_s
            title = 'Sensivity kernel of S velocity'
        elif (ref == 'rho'):
            to_plot = self.sens_rho
            title = 'Sensivity kernel of rho'
        elif (ref == 'vel_p'):
            to_plot = self.sens_vel_p
            title = 'Sensivity kernel of P velocity'
        fig = plt.figure(figsize=(8, 16))
        plt.yticks(ticks=y)
        idum = -1
        for row in range(len(to_plot)):
            if (freq_ind == 'all'):
                what_plot = to_plot[row, :]
                idum += 1
                clrs = cm.jet(norm_c(self.filt_list[idum]))
                plt.plot(what_plot, y,
                         color=clrs)
            else:
                if (row == freq_ind):
                    what_plot = to_plot[row, :]
                    idum += 1
                    clrs = cm.jet(norm_c(self.filt_list[idum]))
                    plt.plot(what_plot, y,
                             color=clrs)

        plt.gca().invert_yaxis()
        sm = plt.cm.ScalarMappable(cmap='jet', norm=norm_c)
        sm.set_array([])
        clb_label2 = 'Filter Periods'
        cb = plt.colorbar(sm, label=clb_label2,
                          fraction=0.02, pad=0.04, location='right')
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        # plt.set_facecolor('#FAEBD7')
        # plt.rcParams['axes.facecolor']='#FAEBD7'
        plt.title(title)
        plt.ylabel('Depth (km)')

    def plot_jacobian_app(self, vel_only=False):
        extent = [0, self.max_depth, self.filt_list[-1], self.filt_list[0]]
        fig = plt.subplots(figsize=(8, 12), nrows=1,
                           ncols=1, dpi=150)
        # if (vel_only == False):
        #     plt.imshow(self.jacobian_app_vel, extent=extent, 
        #                aspect = self.nlayer / len(self.filt_list))
        # else:
        #     plt.imshow(self.jacobian_vel_only, extent=extent,
        #                aspect = self.nlayer / len(self.filt_list))
        if (vel_only == False):
            plt.imshow(self.jacobian_app_vel, extent=extent, 
                           )
        else:
            plt.imshow(self.jacobian_vel_only, extent=extent,
                           )

        plt.xlabel('Depth (km)')
        plt.ylabel('Filter Period (sec)')
        plt.title('app vel Jacobian matrix')
        plt.colorbar(label='Kernel val',
                     fraction=0.02, pad=0.04)
    def plot_jacobian_rf(self):
        extent = [0, self.max_depth, len(self.rf_r_4_inv), 0]
        fig = plt.subplots(figsize=(8, 12), nrows=1,
                           ncols=1, dpi=150)
        plt.imshow(self.jacobian_rf, extent=extent, 
                   aspect=self.nlayer / len(self.rf_r_4_inv))
        plt.xlabel('Depth (km)')
        plt.ylabel('Sample')
        plt.title('rf Jacobian matrix')
        plt.colorbar(label='Kernel val',
                     fraction=0.02, pad=0.04)
    def cal_observ_error(self, normalize = False,
                         plotter = False):
        jacob = self.jacobian_rf.copy()
        [nrow, ncol] = np.shape(jacob)
        self.std_rf_samp = []
        self.rms_rf_samp = []
        for i in range(nrow):
            if (normalize):
                jacob_row = jacob[i,:] / np.max(abs(jacob[i,:]))
            else:
                jacob_row = jacob[i,:]
            if (plotter):
                plt.figure()
                lb = ('rf jacob for samp='+str(i)+' '+str(i * self.dt))
                plt.plot(jacob_row, label= lb)
                plt.legend()
            self.std_rf_samp.append(np.std(jacob_row))
            self.rms_rf_samp.append(ut.cal_rms(jacob_row))
        
        self.std_app_samp = []
        self.rms_app_samp = []
        jacob = self.jacobian_app_vel.copy()
        [nrow, ncol] = np.shape(jacob)
        for i in range(nrow):
            if (normalize):
                jacob_row = jacob[i,:] / np.max(abs(jacob[i,:]))
            else:
                jacob_row = jacob[i,:]
            if (plotter):
                plt.figure()
                lb = ('app jacob for filt='+str(i))
                plt.plot(jacob_row, label= lb)
                plt.legend()
            self.std_app_samp.append(np.std(jacob_row))
            self.rms_app_samp.append(ut.cal_rms(jacob_row))
    def plot_matrix(self, mat):
        extent = [0, self.max_depth, len(mat), 0]
        fig = plt.subplots(figsize=(8, 12), nrows=1,
                           ncols=1, dpi=150)
        plt.imshow(mat, extent=extent, 
                   aspect=self.nlayer / len(mat))
        plt.xlabel('Depth (km)')
        plt.ylabel('Row index')
        plt.colorbar(label='Kernel val',
                     fraction=0.02, pad=0.04)
    def run_rftn(self):
        gauss_width = (2.0 * np.pi) * self.gauss_par
        rftn_traces = rc.Rftn_init(phase='P', ray_p=self.slowness,
                                   delta=self.dt,
                                   nsamp=self.nsamp,
                                   model= self.model_name,
                                   model_name_implicit='syn_fow_model_rftan.mod',
                                   alpha=gauss_width,
                                   x2length=True,
                                   timeshift=self.tshift,
                                   out_name='rftn_output',
                                   saving_directory=self.saving_directory)
        tr_dum = rftn_traces.make_all()
        return(tr_dum)


    def create_model(self, cal_rho_vp=True, cal_ref='vel_s'):
        if (len(self.vel_s) < 1):
            print("check the number of velocity")
        else:
            nlayer = len(self.vel_s)
        if (nlayer != len(self.layers_thickness)):
            print("the number of layer is not match with" +
                  " the number of velocity")
        if (self.layers_thickness[-1] != 0.0):
            print("your layer thickness dont have any half space" +
                  " i will add a half space with vs = 4.6")
            self.layers_thickness.append(0)
            self.vel_s.append(4.6)
            nlayer = len(self.vel_s)
        if (cal_rho_vp == True):
            self.cal_vp_rho_simple(ref=cal_ref)
        else:
            if (cal_ref == 'vel_s'):
                self.vel_p = self.vel_p_org.copy() 
                self.rho = self.rho_org.copy()
            elif (cal_ref == 'rho'):
                self.vel_p = self.vel_p_org.copy() 
                self.vel_s = self.vel_s_org.copy()
            elif (cal_ref == 'vel_p'):
                self.vel_s = self.vel_s_org.copy() 
                self.rho = self.rho_org.copy()
        self.nlayer = nlayer

        if (self.kind == 'rftn'):
            model = []
            for i in range(nlayer):
                vs = self.vel_s[i]
                vp = self.vel_p[i]
                rho = self.rho[i]
                dz = self.layers_thickness[i]
                l = rc.Layer_rftan(thickness=dz,
                                   vel_s=vs,
                                   vel_p=vp,
                                   density=rho)
                model.append(l)
            self.model_name = self.saving_directory + 'syn_fow_model_rftan.mod'
            model_rftan = rc.Model_rftan(layers=model,
                                         name= self.model_name)
            model_rftan.create_model_file()
        self.model = model

    def cal_vp_rho(self):
        ''' p velocity relation
        Vp (km/sec) = 0.9409 + 2.0947Vs - 0.8206Vs**2
            + 0.2683Vs**3 - 0.0251Vs**4

        rho(g/cm3)= 1.6612 Vp - 0.4721 Vp**2
            + 0.0671 Vp**3 - 0.0043 Vp**4 + 0.000106 Vp**5 .   
        derivativse:
            Delta_Vp = (dVp / dVs) delta_Vs =
                (2.0947 - 2 * 0.8206 * Vs + 3 * 0.2683 Vs**2 
                 - 4 * 0.0251 Vs**3) * delta_Vs
            Delta_rho = [(drho / dVp) * (dVp / dVs)] * delta_Vs = 

            [(1.6612 - 2 * 0.4721 * Vp + 3 * 0.0671 * Vp ** 2 
              - 4 * 0.0043 * Vp **3 + 5 * 0.000106 * Vp ** 4) *
             (2.0947 - 2 * 0.8206 * Vs + 3 * 0.2683 Vs**2 
              - 4 * 0.0251 Vs**3)] * delta_Vs





            '''
        vel_p = []
        density = []
        R_alpha = []
        R_rho = []
        for el in self.vel_s:
            vp = (0.9409 + 2.0947 * el - 0.8206 * el**2
                  + 0.2683 * el**3 - 0.0251 * el**4)
            rho = (1.6612 * vp - 0.4721*vp**2
                   + 0.0671 * vp**3 - 0.0043 * vp**4 + 0.000106*vp**5)
            R_alpha_z = (2.0947 - 2 * 0.8206 * el + 3 * 0.2683 * el**2
                         - 4 * 0.0251 * el**3)
            R_rho_z = ((1.6612 - 2 * 0.4721 * vp + 3 * 0.0671 * vp ** 2
                        - 4 * 0.0043 * vp ** 3 + 5 * 0.000106 * vp ** 4) *
                       (2.0947 - 2 * 0.8206 * el + 3 * 0.2683 * el**2
                        - 4 * 0.0251 * el**3))
            vel_p.append(vp)
            density.append(rho)
            R_alpha.append(R_alpha_z)
            R_rho.append(R_rho_z)

        self.vel_p = vel_p
        self.rho = density
        self.R_alpha = R_alpha
        self.R_rho = R_rho

    def cal_vp_rho_simple(self, ref='vel_s'):
        ''' p velocity relation
        Vp (km/sec) = 1.732 * Vs
            Christensen and Mooney (1995)
            for crystaline crust 5.5 < Vp < 7.5
        rho(g/cm3)=  0.541 + 0.3601 * Vp.  

        Derivatives:
            Delta_Vp = (dVp / dVs) delta_Vs =
                (1.732) * delta_Vs
            Delta_rho = [(drho / dVp) * (dVp / dVs)] * delta_Vs = 
                [(0.3601) * (1.732)] delta_Vs

        Gardner et al., 1974
        for sediments 1.5 < Vp < 6.1
        rho(g/cm3)= 1.74V ** (0.25)




            '''
        vel_s = []
        vel_p = []
        density = []
        R_alpha = []
        R_rho = []
        if (ref == 'vel_s'):
            for el in self.vel_s:

                vp = 1.732 * el
                rho = 0.541 + 0.360 * vp
                vel_p.append(vp)
                density.append(rho)
                vel_s.append(el)
                R_alpha_z = 1.732
                R_rho_z = (0.3601 * 1.732)
                R_alpha.append(R_alpha_z)
                R_rho.append(R_rho_z)
            self.vel_p = vel_p
            self.rho = density
            self.R_alpha = R_alpha
            self.R_rho = R_rho
        elif (ref == 'rho'):
            for el in self.rho:
                vp = (el - 0.541) / 0.36
                vs = vp / 1.732
                vel_p.append(vp)
                vel_s.append(vs)
                density.append(el)
                R_alpha_z = 1.732
                R_rho_z = (0.3601 * 1.732)
                R_alpha.append(R_alpha_z)
                R_rho.append(R_rho_z)
            self.vel_s = vel_s.copy()
            self.vel_p = vel_p.copy()
            self.rho = density.copy()
            self.R_alpha = R_alpha.copy()
            self.R_rho = R_rho.copy()
        elif (ref == 'vel_p'):
            for el in self.vel_p:
                vs = el / 1.732
                rho = 0.541 + 0.360 * el
                vel_p.append(el)
                vel_s.append(vs)
                density.append(rho)
                R_alpha_z = 1.732
                R_rho_z = (0.3601 * 1.732)
                R_alpha.append(R_alpha_z)
                R_rho.append(R_rho_z)
            self.vel_s = vel_s.copy()
            self.vel_p = vel_p.copy()
            self.rho = density.copy()
            self.R_alpha = R_alpha.copy()
            self.R_rho = R_rho.copy()

    def cal_jacobian_app_vel(self):
        jacobian_app_vel = np.zeros(shape=(len(self.filt_list), self.nlayer))
        mat1 = self.sens_vel_s.copy()
        mat2 = self.sens_vel_p.copy()
        mat3 = self.sens_rho.copy()
        jacobian_app_vel = mat1 + (mat2 * self.R_alpha) + \
            (mat3 * self.R_rho)
        self.jacobian_app_vel = jacobian_app_vel.copy()
    def cal_jacobian_rf(self, reg_alpha = 1.0):
        jacobian_rf = np.zeros(shape=(len(self.rf_r_4_inv), self.nlayer))
        mat1 = self.sens_rf_vel_s.copy()
        mat2 = self.sens_rf_vel_p.copy()
        mat3 = self.sens_rf_rho.copy() 
        jacobian_rf = mat1 + (mat2 * self.R_alpha) + \
            (mat3 * self.R_rho)
        self.jacobian_rf = jacobian_rf.copy()
    def cal_jacobian_app_vel_jacobson(self):
        jacobian_app_vel = np.zeros(shape=(len(self.filt_list), self.nlayer))
        mat1 = self.sens_vel_s.copy()
        jacobian_app_vel = mat1
        self.jacobian_app_vel = jacobian_app_vel.copy()
    def cal_jacobian_rf_jacobson(self, reg_alpha = 1.0):
        jacobian_rf = np.zeros(shape=(len(self.rf_r_4_inv), self.nlayer))
        mat1 = self.sens_rf_vel_s.copy() 
        jacobian_rf = mat1.copy()
        self.jacobian_rf = jacobian_rf.copy()
    def merge_jacob(self, rf_weight= 1.0):
        jacobian = np.vstack((self.jacobian_app_vel, self.jacobian_rf))
        self.jacobian = jacobian.copy()


# %% class for traces in baz
class App_curve:
    def __init__(self, trs, filt_list, nsamp=1024,
                 dt=0.05, slowness=0.04, gauss_par_4_rf=1.0,
                 rf_method='waterlevel', waterlevel=0.01, tshift=10.0,
                 filter_array='cal', amp_at=0.0,tshift_initial = 10.0,
                 normalize = None, filt_kind = 'cosine'):
        self.filt_kind = filt_kind
        self.normalize = normalize
        self.trs = trs
        self.filt_list = filt_list
        self.nsamp = nsamp
        self.dt = dt
        self.slowness = slowness
        self.gauss_par_4_rf = gauss_par_4_rf
        self.rf_method = rf_method
        self.waterlevel = waterlevel
        self.tshift = tshift
        self.tshift_initial = tshift_initial
        self.time_vec = self.trs[:, 0] - self.tshift_initial
        self.amp_at = amp_at
        if (filter_array == 'cal'):
            self.retrive_filters()
        else:
            self.filter_array = filter_array
        time_to_look = self.time_vec - self.amp_at

        ind = np.argwhere(abs(time_to_look)
                          == min(abs(time_to_look)))[0][0]
        self.onset_ind = ind

    def cal_rf(self):
        trace_radial = self.trs[:, 2].copy()
        trace_transverse = self.trs[:, 3].copy()
        trace_vertical = self.trs[:, 4].copy()
        sample_rate = 1 / self.dt
        dr = trace_radial
        dz = trace_vertical

        src = dz
        rsp_list = [dr, dz]
        if (self.rf_method == 'waterlevel'):
            out = deconv_waterlevel(rsp_list, src, sample_rate , 
                                    waterlevel=self.waterlevel, 
                                gauss=self.gauss_par_4_rf,
                              tshift=self.tshift_initial, length=None, 
                              normalize=self.normalize, 
                              nfft=None,
                              return_info=False)
        elif (self.rf_method == 'iterative'):
            out, _, _ = deconv_iterative(rsp_list, src, sampling_rate= sample_rate,
                                         tshift= self.tshift_initial, 
                                         gauss = self.gauss_par_4_rf,
                                         normalize=self.normalize,
                                         mute_shift=True, 
                                         itmax = 200)
        dr = out[0]
        dr = dr.real
        dz = out[1]
        dz = dz.real
        
        
        dr_fixed, is_it_good = self.check_for_samp_shift(dr, dz)
        
        self.rf_r = dr_fixed
        self.rf_z = dz
    def check_for_samp_shift(self, dr, dz):
        is_it_good = False
        onset_ind_z = np.argwhere((dz) == np.max((dz)))[0][0]
        dz_4_check = dz[0: onset_ind_z + 22]
        dr_4_check = dr[0: onset_ind_z + 22]
        
        ind_z = np.argwhere((dz_4_check) == np.max((dz_4_check)))[0][0]
        ind_r = np.argwhere((dr_4_check) == np.max((dr_4_check)))[0][0]
        
        
        dr_fixed = np.zeros(shape=(len(dr),))
        if (ind_z == ind_r):
            is_it_good = True
            return(dr, is_it_good)
        elif (abs(ind_z - ind_r) <= 20):
            is_it_good = False
            dif_ind = ind_z - ind_r 
            if (dif_ind < 0):
                for i in range(len(dr)):
                    if (i - dif_ind <= len(dr) - 1):
                        dr_fixed[i] = dr[i-dif_ind]
                    else:
                        dr_fixed[i] = 0.0
                    # if (i + dif_ind < 0):
                    #     dr_fixed[i] = 0.0
            else:
                for i in range(len(dr)):
                    if (i+dif_ind < (len(dr) - 1)):
                        dr_fixed[i] = dr[i+dif_ind]
                    else:
                        dr_fixed[i] = 0.0
            
            ind_zero_z = (np.argwhere((dz) ==
                                    np.max((dz)))[0][0])
            ind_zero_r = (np.argwhere((dr_fixed) ==
                                    np.max((dr_fixed)))[0][0])
            if (ind_zero_z == ind_zero_r):
                is_it_good = True
            
            return(dr_fixed, is_it_good)
        else:
            return(dr, is_it_good)
    def retrive_filters(self):
        n_sample = self.nsamp
        filt = np.zeros(n_sample)
        time_filt = self.time_vec
        filter_array = []
        if (self.filt_kind == 'cosine'):
            for el in self.filt_list:
                if (el > 0.0):
                    filt = np.zeros(len(time_filt))
                    idum = -1
                    for t in time_filt:
                        idum += 1
                        if (abs(t) < el):
                            filt[idum] = np.cos((np.pi/2.0) * ((t) / el)) ** 2.0
                        else:
                            filt[idum] = 0.0
                    filter_array.append(filt)
                else:
                    if (el < 0.0):
                        print("filter period must be greater than 0")
                    else:
                        filt = np.zeros(len(time_filt))
                        ind = np.argwhere(abs(time_filt) ==
                                  min(abs(time_filt)))[0][0]
                        filt[ind] = 1.0
                    filter_array.append(filt)
            self.filter_array = filter_array
        elif (self.filt_kind == 'gauss'):
            for el in self.filt_list:
                gauss_freq= 1 / (np.pi * 2 * (el / 2))
                filt = ut._gauss_filter_shifted(self.dt, n_sample, gauss_freq,
                                 waterlevel=None)
                filter_array.append(filt)
            self.filter_array = filter_array

    def find_vel_rfs2(self):
        time_to_look = self.time_vec - self.amp_at

        ind = np.argwhere(abs(time_to_look)
                          == min(abs(time_to_look)))[0][0]
        self.onset_ind = ind
        vel_app = []
        for filt in self.filter_array:                    
            s1 = np.matmul(self.rf_r, filt)
            s2 = np.matmul(self.rf_z, filt)
            amp2 = s1/s2
            theta_rad2 = np.arctan(amp2)
            v_s2 = np.sin(theta_rad2 / 2.0) / self.slowness
            vel_app.append(v_s2)
            
            
            
            # 
        self.vel_app = vel_app

    def find_vel_rfs(self):
        vel_app = []
        n_sample = self.nsamp
        filt = np.zeros(n_sample)
        time_filt = self.time_vec
        for el in self.filt_list:
            if (el > 0.0):
                filt = np.zeros(len(time_filt))
                idum = -1
                for t in time_filt:
                    idum += 1
                    if (abs(t) < el):
                        filt[idum] = np.cos((np.pi/2.0) * ((t) / el)) ** 2.0
                    else:
                        filt[idum] = 0.0
            else:
                if (el < 0.0):
                    print("filter period must be greater than 0")
                else:
                    filt = np.zeros(len(time_filt))
                    ind = np.argwhere(abs(time_filt) ==
                                      min(abs(time_filt)))[0][0]
                    filt[ind] = 1.0
            rf_r_filtered = self.filter_rf(self.rf_r, filt)
            rf_z_filtered = self.filter_rf(self.rf_z, filt)
            amp = (rf_r_filtered[self.onset_ind] /
                   rf_z_filtered[self.onset_ind])
            theta_rad = np.arctan(amp)
            v_s = np.sin(theta_rad / 2.0) / self.slowness
            vel_app.append(v_s)
        self.vel_app = vel_app

    def filter_rf(self, rf_to_filt, filt):
        n = len(self.time_vec)
        sig_fft = np.fft.fft(rf_to_filt, n)
        filt_fft = np.fft.fft(filt, n)
        filtered_signal = sig_fft * filt_fft
        filtered_signal_timed = np.fft.ifft(filtered_signal)
        filtered_signal_timed = np.real(filtered_signal_timed)
        filtered_signal_time = np.zeros(len(filtered_signal_timed))
        shift = self.onset_ind
        idum = -1
        for i in range(len(filtered_signal_timed)):
            idum += 1
            if (idum + shift < len(filtered_signal_timed)):
                ind = idum + shift
            else:
                ind = (idum + shift) - len(filtered_signal_timed)
            filtered_signal_time[i] = filtered_signal_timed[ind]
        return(filtered_signal_time)
def cal_resolution(jacob, title_ref = 'RF'):
    
    
    j = jacob.copy()
    jt = np.transpose(j)
    jtj = np.matmul(jt, j)
    jtjm1 = np.linalg.inv(jtj)
    jtjm1jt = np.matmul(jtjm1, jt)
    
    [nrow, ncol] = np.shape(j)
    I_row = np.ones(shape=(nrow,nrow))
    I_col = np.ones(shape=(ncol,ncol))
    
    data_res = np.matmul(j,jtjm1jt)
    ut.plot_matrix(data_res, title_fig='Data resolution matrix of '
                   +title_ref)
    data_res_diag = np.abs(np.diag(data_res - I_row))
    ut.scatter_line_plot(data_res_diag, title='Data resolution diag of '
                         +title_ref)
    
    model_res = np.matmul(jtjm1jt,j)
    ut.plot_matrix(model_res, title_fig='Model resolution matrix of '
                   +title_ref)
    model_res_diag = np.abs(np.diag(model_res- I_col))
    ut.scatter_line_plot(model_res_diag, title='Model resolution diag of '
                         +title_ref)
    return(data_res, model_res)
    
    
    
    
    