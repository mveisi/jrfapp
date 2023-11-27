#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 16:02:50 2022

classes we are going to use in our code
"""

import numpy as np
import os
import copy
import obspy as op
import math
import rf
from matplotlib.colors import LinearSegmentedColormap
from obspy import taup
from scipy.fftpack import fft, ifft, next_fast_len
from matplotlib import cm
import matplotlib
import io
from PIL import Image
import PIL
from rf.deconvolve import deconv_waterlevel, deconv_iterative
import pickle
from jrfapp import inverse_routine as iv 
import shutil
from scipy.optimize import curve_fit
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import multiprocessing as mp
from jrfapp import main_inv_all as mia
import timeit
from matplotlib.colors import ListedColormap, BoundaryNorm
from jrfapp import utils as ut
# from sklearn.metrics import mean_squared_error
#creating geom file
# np.random.seed(155)


# def scatter_lene_vs(a, b):
#     Na= len(a)
#     Nb = len(b)
#     if (Na == Nb):

#%%
class Initialize:
    def __init__(self, station_coordinate_file='', model_folder = '', 
                 data_folder='', 
                 output_folder='', model_name='iasp91',
                 layering='default', force_layer_info = None, 
                 gauss_width=3.5,
                 nsamp=2048, dt=0.05,
                 inv_bf=2.0, inv_af=45,
                 filter_periods='default', network_name='default',
                 app_weight=5, rf_weight=2.5,
                 smooth_fac=1.00, damp_fac=0.5000,
                 kind='synthetic', freq_min_bpfilt=0.05,
                 freq_max_bpfilt=2.5, min_app_vel=1.5,
                 max_app_vel=5.0, min_app_vel_last = False, 
                 increase_cod = False,
                 max_depth=70.0, 
                 vp_to_vs=1.732, slowness=0.04, 
                 random_seed = None):
        
        """
        
        
        Parameters
        ----------
        station_coordinate_file : str, optional
            The path to the stations coordinate file. This file must contain
            3 columns (station name, station latitude, station longitude).
            Remove any header in the file. The default is '' for synthetic run 
            without any noise. 
        model_folder : str, optional 
            The full path to the folder containing initial models (model_folder)
            if not provided, code read model internally. 
            the defualt is ''. 
        data_folder : str, optional
            Path to the folder containing the dataset. This folder must contain 
            subfolders with the name of stations. Each station folder must 
            contain event folders with 3 waveforms for each component (ZNE). 
            Example for folders is
            {data_folder}/{station_folder}/{event_folder}/*BHZ.SAC *BHE.SAC *BHN.SAC
            data_folder/CHBR/2017119105924/2017119105924_BHE.SAC,
            data_folder/CHBR/2017119105924/2017119105924_BHN.SAC,  
            data_folder/CHBR/2017119105924/2017119105924_BHZ.SAC.
            The default is '' for synthetic without noise.
            
        output_folder : str, optional
            Path to the output folder. The default is current_directory/outputs/.
        model_name : str, optional
            Model name used in the inversion and synthetic generation.
            Model name can be 
            iasp91, sedimentary, continent, halfspace, or path to the model file.
            Example of a model file could be found at 
            /model_folder/*_model.dat. The default is 'iasp91'. 
            File should be created according to this format
            MIN_DEPTH   MAX_DEPTH   VELOCITY    MAX_DEPTH_CHANGE_MIN    MAX_DEPTH_CHANGE_MAX    VELOCITY_CHANGE
            0.0         20.0        3.36        -5                      5                       0.5
            20.0        35.0        3.75        -5                      5                       0.5
            35.0        120.0       4.47         0                      0                       0.5
            This is the IASP91 file. It means from 0.0 to 20.0 km, the velocity is 
            3.36. The 20.0 km is the major boundary because the value of MAX_DEPTH_CHANGE_MIN and 
            MAX_DEPTH_CHANGE_MIN is defined. These values will be used in PSO and 
            mean initial model run. The maximum depth must be greater than 110.
        
        layering : list, optional
            Define the number of layers according to the velocity model. 
            This is the initial number of layers, and all of the values are 
            crossed to maximum value of ndivide_list in random grid search and PSO run.
            The layering is applied according to the velocity model. 
            The layers which have MAX_DEPTH_CHANGE_MIN or MAX_DEPTH_CHANGE_MAX 
            in the velocity model are considered to be reference velocity changes. 
            The default values of layering for IASP91 is [2, 3, 3], which means 
            2 layers from 0.0 to 20.0 km, 3 layers from 20.0 to 35.0 km, 
            and 3 layers from 35.0 to 70 km. Thus, the length of the list 
            you provided here must be number_of_major_boundary + 1.
            The default is 'default' for the layering is [3, 4, 3] for an input 
            model defined by user or halfspace, [4, 2, 3] for IASP91, [2, 3, 3] for sedimentary, 
            and [4, 3, 3] for continent model.
        force_layer_info: list, optional
            if you want to define a specific parametersing model you can define
            it here. This is usefull when you estimate a velocity model for a 
            station and observed many flactuation in several part of model. This
            flactuation means that those part of model is overparameterised. 
            This is also usefull when you want to increase number of layers in 
            a specific part of model. The definition is as follow
            [ [initial_depth1, final_depth1, nlayer1],  
            [initial_depth1, final_depth1, nlayer1], ...]
            for example, 
            force_layer_info = [[0, 15, 3], 
                                [15, 35, 3], 
                                [35, 45, 2], 
                                [45, 70, 2]]
            means that you want to have 3 layer betwen 0 to 15 km, 
            3 layer from 15 to 35 and so on and so forth. 
            The default is None for the not forcing any layer_info and calculate
            layer_info from layering. 
        gauss_width : double, optional
            Gaussian width used in RF calculation. The default is 3.5.
        nsamp : int, optional
            Number of samples per trace in inversion. The default is 2048.
        dt : double, optional
            Delta time of traces. ALL DATA WILL BE DECIMATED TO THIS VALUE.
            The default is 0.05.
        inv_bf : double, optional
            Time before P arrival for the start point of RF time series inversion.
            The default is 2.0.
        inv_af : double, optional
            Time after P arrival for the start point of RF time series inversion.
            The default is 45.
        filter_periods : str or list, optional
            Filter list used in apparent velocity calculation. You can assign it by
            giving a list such as [min, max, step]. 
            The default is 'default'.
        network_name : str, optional
            Your network name. The default is 'default'.
        app_weight : double, optional
            Apparent velocity weight in inversion. The default is 5.
        rf_weight : double, optional
            RFR weight in inversion. The default is 2.5.
        smooth_fac : double, optional
            Smoothing factor. The default is 1.00.
        damp_fac : double, optional
            Damping factor. The default is 0.5000.
        kind : str, optional
            'real' for real data run and 'synthetic' for synthetic run. The default is 'synthetic'.
        freq_min_bpfilt : double, optional
            Minimum frequency of Butterworth filter applied to the rel dataset.
            The default is 0.05.
        freq_max_bpfilt : double, optional
            Maximum frequency of Butterworth filter applied to the rel dataset.
            The default is 2.5.
        min_app_vel : double, optional
            Minimum value of apparent velocity used in filtering bad RF before
            stacking. 
            The default is 1.5.
        max_app_vel : double, optional
            Maximum value of apparent velocity used in filtering bad RF before
            stacking. 
            The default is 5.00.
        min_app_vel_last : double, optional
            Minimum allowed value of the apparent velocity at the last period.
            This parameter use as the criteria for chosing RFs for stacking. 
            This is a optional criteria and should be defined accroding to the 
            minimum expected S-velocity of the bottom layer. 
            The default value is False. 
        increase_cod : bool, Optional
            Another criteria for chosing RFs for stacking. True means that 
            RFs will be included in the stacking which their apparent velocity 
            of the last period is greater than the apparent velocity of the 
            10th previous apparent velocity filter period 
            (app_vel[-1] > app_vel[-10]). This will remove any RFs which affected
            by the noise in the corresponding last period. 
            The default value is False. 
        max_depth : float, optional
            Define the maximum depth of the model. It must be at least 40 km 
            less than the maximum depth defined in the velocity model.
            The default is 70.00.
        vp_to_vs : float, optional
            Define the ratio of compressional velocity to shear velocity.
            The default is 1.732.
        slowness : float, optional
            Slowness that will be used for creating a synthetic model. 
            This will be ignored for real data inversion or when a synthetic model
            created with the real dataset RFs with noise.
            The default is 0.04.
        random_seed : int, optional
            random_seed for repeating same results in different runs.
            This should be an integer. 
            The default value is None for not giving any strick
            random seed. 

        Returns
        -------
        None
        
        
        """
        
        self.station_coordinate_file= station_coordinate_file 
        self.data_folder = data_folder 
        self.output_folder = output_folder
        self.model_name = model_name
        self.gauss_width =gauss_width
        self.nsamp = nsamp
        self.dt = dt
        self.inv_bf = inv_bf
        self.inv_af = inv_af
        self.filter_periods = filter_periods
        self.network_name = network_name
        self.app_weight = app_weight
        self.rf_weight = rf_weight
        self.smooth_fac = smooth_fac
        self.damp_fac = damp_fac
        self.kind = kind
        self.freq_min_bpfilt = freq_min_bpfilt
        self.freq_max_bpfilt = freq_max_bpfilt
        self.min_app_vel= min_app_vel
        self.min_app_vel_last = min_app_vel_last
        self.increase_cod = increase_cod
        self.max_app_vel= max_app_vel
        self.layering = layering
        self.max_depth = max_depth
        self.vp_to_vs = vp_to_vs
        self.slowness = slowness
        self.tshift = 10.0
        self.synthetic_model_created = False
        self.force_layer_info = force_layer_info
        self.random_seed = random_seed
        self.model_folder = model_folder
        if (self.random_seed != None):
            np.random.seed(self.random_seed)
            
        
        
        self.gauss_par = self.gauss_width / (2 * np.pi)
        self._create_coordinates()
        self._create_st_folders()
        self._create_filt_list()
        self._manage_folders()
        self._read_velocity_model()
        self._parameterize_velocity()
        self._create_vel_param_interpolate()
        self._create_time_vec()
        
        
    def _create_coordinates(self):
        if (len(self.station_coordinate_file) > 2):    
            fl_to_read = self.station_coordinate_file
            if (os.path.isfile(fl_to_read)):
                coord = []
                with open(fl_to_read, 'r') as f:
                    for l in f:
                        coord.append(l.split())
                f.close()
                coor = []
                coordinate = []
                st_name = []
                for el in coord:
                    coor.append([el[0], float(el[1]), float(el[2])])
                    st_name.append(el[0])
                    coordinate.append([float(el[1]), float(el[2])])
                network_coordinate = coor
                self.network_coordinate = network_coordinate.copy()
                self.st_name = st_name.copy()
                self.coordinate = coordinate.copy()
                
            else:
                raise Exception(self.station_coordinate_file + 
                                ' file not found')
        else:
            self.st_name = ['ASAD']
            self.coordinate = [[25, 60]]
    def _create_st_folders(self):
        
        
        if (self.data_folder == ''):
            self.st_folder = ''
            self.kind = 'synthetic'
            return 
        elif (len(self.data_folder) < 1):
            raise Exception('You need to define data_folder')
        stations_folder = [f for f in os.listdir(self.data_folder)
                        if os.path.isdir(os.path.join(self.data_folder, f))]
        st_folder = []
        for el in stations_folder:
            for stname in self.st_name:
                if (stname == el):
                    st_folder.append(os.path.join(self.data_folder, el))
        self.st_folder = st_folder
    def _create_filt_list(self):
        if (self.filter_periods == 'default'):
            fl_list1 = np.arange(0.0, 3, 0.1)
            fl_list2 = np.arange(3.25, 25, 0.25)
            filt_list = np.concatenate((fl_list1, fl_list2))
        elif (type(self.filter_periods) == list):
            try:
                filt_list = np.arange(self.filter_periods[0], 
                                      self.filter_periods[1], 
                                      self.filter_periods[2])
            except:
                print('Check your input filter period')
                fl_list1 = np.arange(0.0, 3, 0.1)
                fl_list2 = np.arange(3.25, 25, 0.25)
                filt_list = np.concatenate((fl_list1, fl_list2))
        self.filt_list = filt_list
    def _manage_folders(self):
        if (len(self.data_folder) < 1):
            self.data_folder = os.getcwd()
        if (len(self.output_folder) < 1):
            cwd = os.getcwd()
            self.output_folder = os.path.join(cwd, 'output_folder')
        if (os.path.isdir(self.output_folder)):
            pass
        else:
            os.mkdir(self.output_folder)
    def _read_velocity_model(self):
        model = []
        if (os.path.isfile(self.model_name)):
            model_path = self.model_name
        else:
            model_path = ''
        if (os.path.isdir(self.model_folder)):
            if (self.model_name == 'iasp91'):
                model_path = os.path.join(self.model_folder, 'iasp91_model.dat')
            elif (self.model_name == 'sedimentary'):
                model_path = os.path.join(self.model_folder,
                                          'sedimentary_model.dat')
            elif (self.model_name == 'continent'):
                model_path = os.path.join(self.model_folder, 
                                          'continent_model.dat')
            elif (self.model_name == 'halfspace'):
                model_path = os.path.join(self.model_folder, 
                                          'halfspace_model.dat')
        elif (self.model_folder == ''):
            model_d = get_model_list(self.model_name)
            model_path = ''
        
        else:
            model_path = self.model_name
        
        if (model_path != ''):
            if (os.path.isfile(model_path)):
                model_d = []
                with open(model_path, 'r') as f:
                    for i, l in enumerate(f):
                        model_d.append(l.split())
                f.close()
            else:
                raise Exception('File not found '+ model_path)
            
        
        model = copy.deepcopy(model_d)
        vs_info = []
        init_boundary = []
        init_dif_boundary = []
        velocity_change_constraint = []
        for model_line in model:
            if (is_digit(model_line[0])):
                vs_info.append([float(model_line[0]), float(model_line[1]), 
                                float(model_line[2])])
                velocity_change_constraint.append(float(model_line[5]))
                if ((abs(float(model_line[3])) + 
                     abs(float(model_line[4]))) > 0.0):
                    init_boundary.append(float(model_line[1]))
                    init_dif_boundary.append([float(model_line[3]), 
                                              float(model_line[4])])
        
            
        
        self.vs_info = vs_info.copy()
        self.init_boundary = init_boundary.copy()
        self.init_dif_boundary = init_dif_boundary.copy()
        self.velocity_change_constraint = velocity_change_constraint.copy()
        
        
        
        

        
        if (self.force_layer_info != None):
            correct = 0
            for el in self.force_layer_info:
                if (len(el) == 3):
                    correct += 1
            if (correct == len(self.force_layer_info)):
                print('Using layer info provided in force_layer_info\n '+
                      'The layering is ignored.')
                layer_info = copy.deepcopy(self.force_layer_info)
                self.max_depth = layer_info[-1][1]
                layering_4_run_forced = find_layering(init_boundary= 
                                                self.init_boundary, 
                                                vel_info= self.vs_info,
                                                layer_info = layer_info, 
                                                vp_to_vs= self.vp_to_vs)
                
                layering_4_run = copy.deepcopy(layering_4_run_forced)
        else:
            if (self.layering == 'default'):
                if (self.model_name == 'IASP91'):
                    layering_4_run = [4, 2, 3]
                elif (self.model_name == 'sedimentary'):
                     layering_4_run = [2, 3, 3]
                elif (self.model_name == 'continent'):
                    layering_4_run = [4, 3, 3]
                else:
                    layering_4_run = [3, 4, 3]
                    
                    
            else:
                if (len(self.layering)  == len(self.init_boundary) + 1):
                    layering_4_run = self.layering
                else:
                    raise Exception('The length of layering must equal to \
                                     the length of major boundary + 1')
            layer_info = ut.find_layer_info(init_boundary= self.init_boundary, 
                                               layering_4_cal= layering_4_run, 
                                               max_depth_4_cal= self.max_depth)
                
            
        
        self.layer_info = copy.deepcopy(layer_info)
        self.layering_4_run = copy.deepcopy(layering_4_run)
        
    def _parameterize_velocity(self):
        vel_param = ut.Vel_paramterize(vel_info = self.vs_info, 
                                            layer_info = self.layer_info, 
                                            vp_to_vs= self.vp_to_vs)
        self.initial_vel_param = vel_param
        
        self.initial_velocity = self.initial_vel_param.vel_s.copy()
        self.initial_layer_thickness = \
            self.initial_vel_param.layer_thickness.copy()
        self.initial_layer_thickness_abs = \
            self.initial_vel_param.layer_thickness_abs.copy()
        self.initial_time_thickness = self.initial_vel_param.time_thickness.copy()
    def _create_vel_param_interpolate(self):
        layer_info = [[0.0, self.max_depth, int(self.max_depth / 2)]]
        vel_param = ut.Vel_paramterize(vel_info = self.vs_info, 
                                            layer_info = layer_info, 
                                            vp_to_vs= self.vp_to_vs)
        self.interpolate_vel_param = vel_param
    def _create_time_vec(self):
        time_vec = np.arange(-self.inv_bf, self.inv_af, self.dt)
        self.time_vec = time_vec
        
    def create_synthetic(self, which_to_perturb, equal_layering = False):
        '''
        Parameters
        ----------
        which_to_perturb : TYPE = list
            This list define the perturbation parameters for creating 
            synthetic model. The syntax is :
                [start_depth, end_depth, velocity_perturbation]
            for example [2, 8, -0.2] means the shear velocity of synthetic model
            is equal to shear velocity of the input model for 2 to 8 km - 0.2.
            The which_to_perturb list could contain several anomaly. 
            For example:
                [[2, 8, -0.2], [10, 14, 0.4], [33, 40, 0.5]]
            
        equal_layering : TYPE = bool, optional
            the model file parameterize according to the layering provided 
            when initilize the code. Thus if which_to_perturb parameter doesnt
            match the initial layering the synthetic model wont generated in a 
            right way. For example if layering defined 4 layer in 0 to 20 km (
                i.e., four 5 km layer) we cant define an anomaly in the depth 
            ranges of 2 to 8 km. Therefore, if your anomaly doesnt match the 
            initial layering, you must use equal_layering= False, otherwise
            you can use equal_layering= True, but i dont recommend it. 
            The default is False.

        Returns
        -------
        None.

        '''
        if (equal_layering == True):
            vel_param_synthetic = ut.Vel_paramterize(vel_info = self.vs_info, 
                                                layer_info = self.layer_info, 
                                                vp_to_vs= self.vp_to_vs)
            vel_param_synthetic.perturb_vel(which_to_perturb)
        else:
            layer_info_syn = []
            layer_info_syn.append([0, self.max_depth, int(self.max_depth)])
            vel_param_synthetic = ut.Vel_paramterize(vel_info = self.vs_info, 
                                                layer_info = layer_info_syn, 
                                                vp_to_vs= self.vp_to_vs)
            vel_param_synthetic.perturb_vel(which_to_perturb)
        self.synthetic_model_created = True
        self.synthetic_vel_param = vel_param_synthetic
        self.synthetic_layer_thickness = \
            self.synthetic_vel_param.layer_thickness.copy()
        self.synthetic_velocity = \
            self.synthetic_vel_param.perturbed_vel.copy()
        self.synthetic_layer_thickness_abs = \
            self.synthetic_vel_param.layer_thickness_abs.copy()
        self.synthetic_time_thickness = \
            self.synthetic_vel_param.time_thickness.copy()
    def plot_model(self, model_p = 'initial'):
        '''
        Parameters
        ----------
        model_p : TYPE = str, optional
            ploting model. 
            model_p can be:
                initial : for ploting initial velocity model.
                synthetic : fro ploting synthetic model.
                
                
            The default is 'initial'.

        Returns
        -------
        None.

        '''
        
        if (model_p == 'initial'):
            lthickness = self.initial_layer_thickness.copy()
            lthickness_abs = self.initial_layer_thickness_abs.copy()
            velocity = self.initial_velocity.copy()
        elif (model_p == 'synthetic'):
            if (self.synthetic_model_created):
                lthickness = self.synthetic_layer_thickness.copy()
                lthickness_abs = self.synthetic_layer_thickness_abs.copy()
                velocity = self.synthetic_velocity.copy()
            else:
                raise Exception('Create a synthetic model first')
                return
        syn_forw = iv.Forward_cal(vel_s= velocity, 
                                             layers_thickness = lthickness, 
                                             filt_list = self.filt_list,
                                             gauss_par= self.gauss_par,
                                             dt = self.dt,
                                             nsamp = self.nsamp, 
                                             tshift = self.tshift,
                                             slowness= self.slowness,
                                             inv_time_rf1 = self.inv_bf, 
                                             inv_time_rf2 = self.inv_af,
                                             noise_level =0.0,
                                             rf_normalize= 1,
                                             saving_directory = \
                                                 self.output_folder+'/dummy_st/')
        
        self.tr_dum = syn_forw.tr_dum
        
        # shutil.rmtree(self.output_folder+'/dummy_st/')
            
        app_vel = np.array(syn_forw.apparant_vel_org) 
        rf = np.array(syn_forw.rf_r_from_ind1.copy())
        time_vec = np.arange(-self.inv_bf, self.inv_af, self.dt)
        
        
        
        if (model_p == 'initial'):
            self.rf_initial = copy.deepcopy(rf)
            self.time_vec_initial = copy.deepcopy(time_vec)
            plotter_d(velocity, time_vec, lthickness_abs, 
                      rf, app_vel, self.filt_list, 
                      save_name= os.path.join(self.output_folder, 
                                              self.model_name+'.png'),
                      header = 'Initial Model')
        else:
            plotter_d(velocity, time_vec, lthickness_abs, 
                      rf, app_vel, self.filt_list, 
                      save_name= os.path.join(self.output_folder, 
                                              'syn_model.png'), 
                      header = 'Synthetic Model')
#%%
class Jrfapp_station:
    def __init__(self, init_obj, name, 
                 noise_level = 0.0, 
                 save_data = True,
                 save_rf = True,
                 force_load_data=False, 
                 force_load_rf=False,  
                 close_fig=True, 
                 review_station = False):
        '''
        Parameters
        ----------
        init_obj : TYPE = Initilize.
            The Initilize object =.
        name : TYPE = str
            station name. If station name couldnt be found in the 
            init_obj.st_name which created according to the coordinate input 
            file, this station considered as the synthetic station. Thus, 
            you can create station with arbitary name for example "synthetic"
            for a synthetic run, otherwise this station considered as the 
            real station.
        noise_level : TYPE = float, optional
            If you provide a value greater than zero in noise level, This station 
            will considered as the synthetic station. In this scenario if the name
            you provided match the one of the station name in your station_coordinate_file
            provided in the Initilize, code will use the data from this station.
            otherwise if the name doesnt match any station in your station_coordinate_file
            code will create a synthetic station with the noise_level you provided. 
        save_data : TYPE = bool, optional
            True for saving data. The default is True.
        save_rf : TYPE = bool, optional
            True for saving Receiver functions. The default is True.
        force_load_data : TYPE = bool, optional
            True for reading data from previously saved data. The default is False.
        force_load_rf : TYPE = bool, optional
            True for reading RFs from previously saved RFs. The default is False.
        close_fig : TYPE = bool, optional
            True for closing figures created after definition of station.
            Dont change this to False unless you want to review your RFs and 
            app_vel calculated for this station. Changing this to False when 
            you are creating several station could potentially result in memory. 
            overload. 
            The default is True.
        review_station : TYPE = bool, optional
            If true the code will create two folder that represent all the 
            RFR and Vs,app that passes all the criteria for including traces 
            in the stacking. This is usefull when you want to review a station
            and look for any inconstancy in the RFs. 
            
        Returns
        -------
        None.

        '''
        self.init_obj = init_obj
        self.name = name 
        self.noise_level = noise_level
        self.save_data = save_data
        self.save_rf = save_rf 
        self.force_load_data = force_load_data
        self.force_load_rf = force_load_rf
        self.close_fig = close_fig
        self.review_station = review_station
        
        self._check_st()
        if (self.n_good_event > 2):
            self._real_st()
            if ((self.noise_level > 0.0) and (self.st_kind == 'real')):
                print('Creating a synthetic data with the dataset of '
                      + self.name+' station.')
                self._real_st_syn()
            
            
        
    def _check_st(self):
        if (self.name in self.init_obj.st_name):
            self.st_kind = 'real'
        else:
            self.st_kind = 'synthetic'
            self.n_good_event = 0
        if (self.st_kind == 'real'):
            for el in self.init_obj.st_folder:
                if (self.name in el):
                    st_folder = el
            event_folders = [os.path.join(st_folder, f) for f in os.listdir(st_folder)
                            if os.path.isdir(os.path.join(st_folder, f))]
            
            self.st_folder_cur = st_folder + '/'
            n_wave_events = []
            for event_fl in event_folders:
                sac_fls = [f for f in os.listdir(event_fl)
                                if (os.path.isfile(os.path.join(event_fl, f)) and 
                                    (f.endswith('SAC')))]
                n_wave_events.append(len(sac_fls))
            n_good_event = len(np.argwhere(np.array(n_wave_events) == 3))
            print('Creating station '+self.name+ ' \n'+
                  'found '+str(len(event_folders))+ ' events from which \n'+
                  ' '+ str(n_good_event) + ' had 3 waveforms')
            
           
                
            
            if (n_good_event > 2):
                self.n_good_event = n_good_event
            else:
                raise Exception('Couldnt Initialize station '+self.name+ ' check the following path' + 
                      st_folder +' make sure events folders exist')
        else:
            if (self.init_obj.synthetic_model_created):
                vel_syn = copy.deepcopy(self.init_obj.synthetic_velocity)
                layer_syn = copy.deepcopy(self.init_obj.synthetic_layer_thickness)
            else:
                layer_syn = copy.deepcopy(self.init_obj.initial_layer_thickness)
                vel_syn = copy.deepcopy(self.init_obj.initial_velocity)
            
            out_folder = os.path.join(self.init_obj.output_folder, self.name)
            self.st_obj = iv.Forward_cal(vel_s= vel_syn, 
                                         layers_thickness = layer_syn, 
                                         filt_list = self.init_obj.filt_list,
                                         gauss_par= self.init_obj.gauss_par,
                                         dt = self.init_obj.dt,
                                         nsamp = self.init_obj.nsamp, 
                                         tshift = self.init_obj.tshift,
                                         slowness= self.init_obj.slowness,
                                         inv_time_rf1 = self.init_obj.inv_bf, 
                                         inv_time_rf2 = self.init_obj.inv_af,
                                         noise_level = self.noise_level,
                                         saving_directory = out_folder)
            self.st_obj.time_vec = self.st_obj.time_cuted
    def _real_st(self):
        st_obj = Station_sor_4_pkg(network_name = self.init_obj.network_name, 
                                  stname=self.name,
                                  st_folder= self.st_folder_cur,
                                  layers_thickness= self.init_obj.initial_layer_thickness, 
                                  filt_list = self.init_obj.filt_list,
                                  vel_init= self.init_obj.initial_velocity,
                                  network_coordinate = self.init_obj.network_coordinate,
                                  to_sample = 1 / self.init_obj.dt, 
                                  gauss_filt = self.init_obj.gauss_par, 
                                  bf_p = self.init_obj.inv_bf,
                                  af_p = self.init_obj.inv_af,
                                  freq_min_bpfilt = self.init_obj.freq_min_bpfilt,
                                  freq_max_bpfilt = self.init_obj.freq_max_bpfilt,
                                  max_app_vel= self.init_obj.max_app_vel, 
                                  min_app_vel=self.init_obj.min_app_vel,
                                  min_app_vel_last= self.init_obj.min_app_vel_last,
                                  increase_cond = self.init_obj.increase_cod,
                                  save_data = self.save_data,
                                  save_rf = self.save_rf,
                                  force_load_data=self.force_load_data, 
                                  force_load_rf=self.force_load_rf,  
                                  close_fig=True, 
                                  review_station= self.review_station, 
                                  save_folder=self.init_obj.output_folder + '/')
        self.st_obj = st_obj
    def _real_st_syn(self):
        if (self.init_obj.synthetic_model_created):
            vel_syn = self.init_obj.synthetic_velocity.copy()
            layer_syn = self.init_obj.synthetic_layer_thickness.copy()
        else:
            layer_syn = self.init_obj.initial_layer_thickness.copy()
            vel_syn = self.init_obj.initial_velocity.copy()
        self.st_obj.create_syn(vel_syn = vel_syn,
                          lthickness= layer_syn,
                          noise_level = self.noise_level)
    def get_st_obj(self):
        return(self.st_obj)
    def save_file(self, file_name = 'jrfappst.bin'):
        if (hasattr(self.st_obj, 'save_folder')):
            fl_name = os.path.join(self.st_obj.save_folder, 
                           'jrfapp_st_obj'+file_name)
        else:
            fl_name = os.path.join(self.st_obj.saving_directory, 
                           'jrfapp_st_obj'+file_name)
        file1 = open(fl_name, "wb") 
        pickle.dump(self, file1)
        file1.close()
        print('Saved to '+ fl_name)
        return(fl_name)
    
    
    

        
        
        
        
        
    def invert_data(self, inv_method = 'PSO',
                    stack_name = 'Linear Stack after app_vel criteria',
                    ndivide_list = [-1, 1, -2, 2, -3, 3, -4, 4], 
                    nmodel = 200, nthread = 2, 
                    PSO_nparticle = 30, PSO_maxiter = 15, 
                    finer_ndivide = '', load_from_previous = False):
        '''
        

        Parameters
        ----------
        inv_method : TYPE = str, optional
            The inversion method define the sudo-initial model estimation.
            Possible method are grid_search and PSO.
            The default is 'PSO'.
        stack_name : TYPE = str, optional
            The station RFs stacked with different criteria and you can choose
            which stack you want to use for inversion. possible stacking methods are
            
            1- "Linear Stack before app_vel criteria", which is stacked RFs before
            removing RFs which produce apparant velocities lower than 
            min_app_vel and greater than max_app_vel defined in the Initialize 
            class. I dont recommend to use this because unrealistic values for 
            apparant velocity are possibly generated by noise in dataset. 
            
            2- "Linear Stack after app_vel criteria", which is stacked RFs before
            removing RFs which produce apparant velocities lower than 
            min_app_vel and greater than max_app_vel defined in the Initialize 
            class. This is the recommended parameter for inversion. 
            
            3- "Phase Weighted stack after app_vel criteria", which is phase 
            weighted stack of RFs after removing RFs which produce apparant
            velocities lower than min_app_vel and greater than max_app_vel
            defined in the Initialize class. I dont recommend to use this
            because moveout correction didnt applied to RFs in the phase 
            weighted stacking process and most of the multiple is removed 
            from RF time series. 
            
            4- "K0 Stack joint_harmonic before app_vel criteria", K0 stack is 
            based on removing first order variation of azimuthal difference in 
            RFs. The basic idea is to keep the bulk value of each phase in the 
            RFR time series. However, this stacking method need reliable dataset
            to constraint the azimuthal variation. This stack is the K0 stack before
            removing RFs which produce apparant velocities lower than 
            min_app_vel and greater than max_app_vel defined in the Initialize 
            class. I dont recommend to use this unless you have enough back azimuth
            variation in a low-noise level dataset.
            
            5- "K0 Stack joint_harmonic after app_vel criteria", like previous but
            after removing RFs which produce apparant velocities lower than 
            min_app_vel and greater than max_app_vel defined in the Initialize 
            class. You can use this when you have a very good back azimuth 
            variation.
            
            6- "Weighted Stack according to Number of trace in each bin 
            before app_vel criteria", linear stack but weighted according to the
            number of traces in each bins of back azimuth. In the process of 
            creating Station the back azimuth divided into 24 bins and RFs in 
            each bins stacked. This stacking method uses the number of RFs in 
            each stack as the weight for stacking all 24 bins. This 
            stack calculated before removing RFs which produce apparant velocities
            lower than min_app_vel and greater than max_app_vel defined in the
            Initialize class. I dont recommend using this cause of noises that
            produce unrealistic apparant velocity.
            
            7- "Weighted Stack according to Number of trace in each bin 
            after app_vel criteria", same as 6 but after removing RFs which 
            produce apparant velocities lower than min_app_vel and greater 
            than max_app_vel defined in the Initialize class. You can use this
            when your RFs dominated by a specific back azimuth. 
            
            8- "Linear Stack of bins before app_vel criteria", same as 1 but 
            each bins have a weight equal to 1.
            
            9- "Linear Stack of bins after app_vel criteria", same as 8 but after 
            removing RFs which produce apparant velocities lower than min_app_vel 
            and greater than max_app_vel defined in the Initialize class. 
            
            10- "synthetic", for inverting synthetic data made from real dataset.
            
            You can compare all the stacking method in a figure created in each 
            station folder with the starting name of comparision_of_stacks_. 
        
            The default is 'Linear Stack after app_vel criteria'.
        ndivide_list : TYPE = list, optional
            This list tell the inversion how to change layering in different 
            stage of joint inversion. Minus sign means we want to use "increased
             damping and smoothing factor" which is the function of number of 
             layers. The value represent the factor crossed to initial layering
             defined in Initialize class. For example [-1, 1, -2, 2, -3, 3, -4, 4]
             and a layering = [2, 3, 3] imply that after sudo-initial model 
             defined, we want to invert data with [2, 3, 3] layering and 
             increased factors, [2, 3, 3] layering and inputs factors, 
             [4, 6, 6] layering and increased factor, [4, 6, 6] layering and 
             input factors and so on. The output model is the best fit model 
             from inversions defined by this parameters. 
            The default is [-1, 1, -2, 2, -3, 3, -4, 4] which means 8 run. 
        nmodel : TYPE = int, optional
            Number of model used in grid search method. this parmeter only affect
            when you are using inv_method = "grid_search". The large number 
            of nmodel could potentially overflow memory.
            The default is 200. 
        nthread : TYPE = int, optional
            Number of threat that will be used in multiprocessor computing. 
            For maximum performance the mod(nmodel, nthread) should be 0. 
            For PSO mod(PSO_nparticle * PSO_maxiter, nthread) should be 0. 
            The default is 2.
        PSO_nparticle : TYPE = int, optional
            Number of PSO method particle. Increasing this may result in a 
            better sudo-initial model estimation with the cost of more 
            computational power (This parameter only important when 
            inv_method = "PSO"). 
            The default is 30. 
        PSO_maxiter : TYPE = int, optional
            Number of maximum iteration in PSO algorithm. Increasing this may 
            result in a better sudo-initial model estimation with the cost of 
            more computational power (This parameter only important when 
            inv_method = "PSO"). 
        finer_divide : TYPE = list, optional
            The division list of the final model. if defined the final 
            pseudo-initial model invert with this division, otherwise
            code will use ndivide_list as the finer. The structres of 
            this list is similar to the ndivide_list. 
            The default is ''.
        load_from_previous : TYPE = bool, optional
            If True code will load the previous saved run. 
            You can alternatively give a full path to a file which you previously 
            saved the inversion information.
            The default is False.

        Returns
        -------
        None.

        '''
        
        
        self.inv_method = inv_method 
        self.stack_name_to_inv = stack_name
        self.ndivide_list = ndivide_list
        self.nthread = nthread
        self.nmodel = nmodel
        self.PSO_nparticle = PSO_nparticle
        self.PSO_maxiter = PSO_maxiter
        self.finer_ndivide = finer_ndivide
        
        if (type(finer_ndivide) != list):
            self.finer_ndivide = self.ndivide_list.copy()
        
        start_inv_time = timeit.default_timer()
        if (self.inv_method == 'grid_search'):
            vel_param_list = self._define_grid_models()
            self.vel_param_list = vel_param_list
            self._inv_grid_search(vel_param_list, load_from_previous)
        elif (self.inv_method == 'PSO'):
            self._inv_pso(load_from_previous)
                
            
            
        
        else:
            raise Exception('inv_method can only be "PSO" or "grid_search"')
            
        end_inv_time = timeit.default_timer()
        print('Run time was '+ str(end_inv_time - start_inv_time) + ' second')
            
            
            
    
    
    
    
    
    
    
    
    
    def _inv_pso(self, load_from_previous = False):
        num_procs = self.nthread
        (kind, app_vel_obs, rf_r_data, slowness, output_folder_inv) = \
            self._get_rf_app_obs()
        self.output_folder_inv = output_folder_inv
        iter_all_arr = np.arange(1)
        if (load_from_previous == False):
            pso_name = False
        elif (load_from_previous == True):
            pso_name = os.path.join(output_folder_inv, 'tthickness_pso.bin')
            print('loading from previous run of PSO. loading\n' + 
                  pso_name)
        else:
            pso_name = load_from_previous
        initial_vel_param = copy.deepcopy(self.init_obj.initial_vel_param)
        args = zip(iter_all_arr, 
                   [num_procs] * len(iter_all_arr),
                   [kind] * len(iter_all_arr),
                   [self.PSO_nparticle] * len(iter_all_arr), 
                   [self.PSO_maxiter] * len(iter_all_arr), 
                   [self.init_obj] * len(iter_all_arr),
                   [app_vel_obs] * len(iter_all_arr),
                   [rf_r_data]* len(iter_all_arr),
                   [slowness] * len(iter_all_arr), 
                   [self.ndivide_list]* len(iter_all_arr),
                   [output_folder_inv]* len(iter_all_arr), 
                   [num_procs] * len(iter_all_arr),
                   [initial_vel_param] * len(iter_all_arr), 
                   [self.finer_ndivide] * len(iter_all_arr), 
                   [pso_name] * len(iter_all_arr))
        result = map(_pso_inv_parser, args)
        for el in result:
            self.inv_info = el
            self.st_obj.inv_info = el
        self.plot_results(fig_name= 'inversion_of_PSO_pseudo_initial.png')
        self._pso_plotter()
        
        
    def _inv_grid_search(self, vel_param_list, 
                         load_from_previous = False):
        num_procs = self.nthread
        # # # Create a multiprocessing pool with the number of processors
        (kind, app_vel_obs, rf_r_data, slowness, output_folder_inv) = \
            self._get_rf_app_obs()
        self.output_folder_inv = output_folder_inv
        if (load_from_previous == False):
            iter_all_arr = np.arange(self.nmodel)
            file_to_save = os.path.join(output_folder_inv,'grid_search_workplace')
            file_to_save = os.path.join(file_to_save, 'gs_inv_info_list.bin')
            st_obj_dummy_path = os.path.join(output_folder_inv, 'st_obj_dummy.bin')
            file1 = open(st_obj_dummy_path, "wb") 
            pickle.dump(self.st_obj, file1)
            file1.close()
            del self.st_obj
            ## for parallel run
            args = (iter_all_arr, num_procs,
                       kind, self.init_obj,
                       app_vel_obs, rf_r_data,
                       slowness, self.ndivide_list,
                       output_folder_inv, vel_param_list, 
                       False)
            gs_inv_info_path_list = _gs_inv_parser(args)
            
            gs_inv_info_list = []
            for inv_info_path in gs_inv_info_path_list:
                with open(inv_info_path, 'rb') as f1:
                    gs_inv_info_list.append(pickle.load(f1))
            file1 = open(file_to_save, "wb") 
            pickle.dump(gs_inv_info_list, file1)
            file1.close()
            
            with open(st_obj_dummy_path, 'rb') as f1:
                self.st_obj = pickle.load(f1)
            os.remove(st_obj_dummy_path)
            ##=========================================
            ## for loading from inv_info from indivisual model from a previous run
            ## in case that your run interupted and u want to load that models you
            ## should as follow. 
            ## copy all of your inv_info for different models into a new folder
            ## which saved into /grid_search_workspace/Run_model../inv_info'no'.bin
            ## create a list with each row represent full path of models inv_info
            ## that you copied into new folder.
            ## save this list using pickle and give its path here. uncomment lines
            ## below and run for low number of model. for exampe 4 model.
            ## code will include your previous full run and incorporate 
            ## all the previous models.
            # path_to_invinfo_list = '/home/soroush/rf_shallow_codes/makran_data/pkg_test_real_rgs/all_inv_info_list_fpr.bin'
            # with open(path_to_invinfo_list, 'rb') as f1:
            #     inv_info_list_path = pickle.load(f1)
            # for el in inv_info_list_path:
            #     with open(el, 'rb') as f1:
            #         inv_info = pickle.load(f1)
            #         vel_param_list.append(inv_info['vel_param'])
            #         gs_inv_info_list.append(inv_info)
            # self.nmodel = len(gs_inv_info_list)
            # file1 = open(file_to_save, "wb") 
            # pickle.dump(gs_inv_info_list, file1)
            # file1.close()
            # ##===================================================
            self.gs_inv_info_list_path = file_to_save
            
            print('all grid search model info saved into \n'+ file_to_save)
        elif (load_from_previous == True):
            file_to_load = os.path.join(output_folder_inv,'grid_search_workplace')
            file_to_load = os.path.join(file_to_load, 'gs_inv_info_list.bin')
            print('loading from previous run of RGS. loading\n' + 
                  file_to_load)
            self.gs_inv_info_list_path = file_to_load
        else:
            self.gs_inv_info_list_path = load_from_previous
        gs_inv_info_list = self._gs_plotter()
        self._find_best_model_from_gs(gs_inv_info_list)
        iter_all_arr = np.arange(1)
        args = zip(iter_all_arr, 
                [num_procs] * len(iter_all_arr),
                [kind] * len(iter_all_arr),
                [self.init_obj] * len(iter_all_arr),
                [app_vel_obs] * len(iter_all_arr), 
                [rf_r_data]* len(iter_all_arr),[slowness] * len(iter_all_arr), 
                [self.finer_ndivide]* len(iter_all_arr), 
                [output_folder_inv]* len(iter_all_arr), 
                [self.best_vel_param_grid_search] * len(iter_all_arr),
                [True] * len(iter_all_arr))
        
        print('best initial model found from grid search start to invert '+
              'data with this model')
        result = map(_gs_inv_parser, args)
        
        for el in result:
            self.inv_info = el
            self.st_obj.inv_info = el
        self.plot_results(fig_name= 'best_of_random_grid_search.png')
        self._calculate_according_to_mean_vel(file_name = 
                                              'result_of_mean_model.png')
        

    def _find_best_model_from_gs(self, gs_inv_info_list):
        layer_info_4_interpolate = [[0.0, self.init_obj.max_depth, 
                                    int(self.init_obj.max_depth/ 2)]]
        vel_param_4_interpolate = \
            ut.Vel_paramterize(vel_info = self.init_obj.vs_info, 
                                layer_info = layer_info_4_interpolate, 
                                vp_to_vs= self.init_obj.vp_to_vs)
        lthickness_abs_4_interpolate = vel_param_4_interpolate.layer_thickness_abs
        
        
        
        self.vel_param_list = []
        vel_interpolate_list = []
        for el in gs_inv_info_list:
            vel_param = el['vel_param']
            vel_grid = el['best_inv'][2]
            lthickness_grid = el['best_inv'][7]
            v_s_interp = interp_velocity(lthickness= lthickness_grid
                                                      , velocity= vel_grid, 
                            lthickness_abs_input= lthickness_abs_4_interpolate)
            vel_param.vel_grid = vel_grid
            vel_param.lthickness_grid = lthickness_grid
            vel_param.v_s_estimate_interp = v_s_interp
            vel_interpolate_list.append(v_s_interp)
            self.vel_param_list.append(vel_param)
        vel_interpolate_array = np.array(vel_interpolate_list)
        vel_interpolate_array = np.transpose(vel_interpolate_array)
        vel_mean = []
        for i in range(len(vel_interpolate_array[:, 0])):
            vel_mean.append(np.mean(vel_interpolate_array[i, :]))
        dif_norm_all = []
        for v_param in self.vel_param_list:
            dif = np.array(v_param.v_s_estimate_interp - vel_mean)
            v_param.dif_norm = np.linalg.norm(dif)
            dif_norm_all.append(np.linalg.norm(dif))
        ind_best = np.argwhere(dif_norm_all == np.min(dif_norm_all))[0][0]
        self.best_vel_param_grid_search = self.vel_param_list[ind_best]
        
            
            
        
        
        
    
        
        
    def _calculate_according_to_mean_vel(self, file_name = 'mean_model.png'):
        (kind, app_vel_obs, rf_r_data, slowness, output_folder_inv) = \
            self._get_rf_app_obs()
        if (kind == 'Synthetic_Inversion_from_real_data'):
            title = 'Synthetic inversion using real datasets'
            time_vec = self.st_obj.syn_l_stack[0].time_vec
            rf_std = self.st_obj.syn_l_stack[0].rf_std
        elif (kind == 'Synthetic_Inversion'):
            title = 'Synthetic inversion '
            time_vec = self.st_obj.time_cuted
            rf_std = np.zeros(shape = np.shape(time_vec))
        else:
            title = 'Inversion of '+ self.stack_name_to_inv
            # title = self.stack_name_to_inv
            time_vec = \
                self.st_obj.stacked_dict[self.stack_name_to_inv][0].time_vec
            rf_std = self.st_obj.stacked_dict[self.stack_name_to_inv][0].rf_std
        time_vec_d = time_vec - 25.0
        ind_fig_plot = len(time_vec)
        if (self.inv_method == 'grid_search'):
            mean_model_folder = os.path.join(output_folder_inv, 'mean_model_rgs')
            vel_param = copy.deepcopy(self.mean_vel_param)
            mean_vel = copy.deepcopy(self.mean_vel_af_gs)
        else:
            mean_model_folder = os.path.join(output_folder_inv, 'mean_model_pso')
            vel_param = copy.deepcopy(self.mean_vel_param)
            mean_vel = copy.deepcopy(self.mean_vel_af_PSO)
            
        syn_forw = iv.Forward_cal(vel_s= mean_vel, 
                                             layers_thickness = vel_param.layer_thickness, 
                                             filt_list = self.init_obj.filt_list,
                                             gauss_par= self.init_obj.gauss_par,
                                             dt = self.init_obj.dt,
                                             nsamp = self.init_obj.nsamp, 
                                             tshift = self.init_obj.tshift,
                                             slowness= slowness,
                                             inv_time_rf1 = self.init_obj.inv_bf, 
                                             inv_time_rf2 = self.init_obj.inv_af,
                                             noise_level =0.0,
                                             saving_directory = \
                                                 mean_model_folder)
        rf_cal = syn_forw.rf_r_from_ind1
        app_cal = syn_forw.apparant_vel_org
        dif_rf = np.array(rf_cal) - np.array(rf_r_data)
        dif_rf_norm = np.linalg.norm(dif_rf)
        dif_app = np.array(app_cal) - np.array(app_vel_obs)
        dif_app_norm = np.linalg.norm(dif_app)
        self.inv_info['mean_vel_inv'] = []
        self.inv_info['mean_vel_inv'].append(vel_param.layer_thickness)
        self.inv_info['mean_vel_inv'].append(mean_vel)
        self.inv_info['mean_vel_inv'].append(mean_vel)
        self.inv_info['mean_vel_inv'].append(dif_rf_norm)
        self.inv_info['mean_vel_inv'].append(dif_app_norm)
        self.inv_info['mean_vel_inv'].append(rf_cal)
        self.inv_info['mean_vel_inv'].append(app_cal)
        self.inv_info['mean_vel_inv'].append(vel_param.layer_thickness)
        self.inv_info['mean_vel_inv'].append(vel_param)
        fig, axes = plt.subplots(figsize = (32, 20),
                                 nrows=1, ncols = 3,
                                 gridspec_kw={'width_ratios': [1, 0.5, 0.5]}, 
                                 facecolor='#FAEBD7')
        
        norm = (dif_rf_norm + dif_app_norm)
        
        subfig_title = ('Results for Interpolated velocity' + 
                        ', norm = '+str(round(norm, 2)))
        plt.suptitle(subfig_title, 
                                            fontsize = 32)
        info = self.inv_info['mean_vel_inv']
        self._axes_ploter(axes, info, rf_r_data, app_vel_obs, 
                          time_vec, rf_std, ind_fig_plot,
                         low_smooth = True, subfig_title= subfig_title, 
                         init_model_color = 'red')
        if ('Synthetic' in kind):
            (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
                ut.find_xminmax(self.init_obj.synthetic_layer_thickness_abs, 
                             self.init_obj.synthetic_velocity)
            axes[0].vlines(v_line_x, y_min, y_max, colors='black', 
                           lw = 4.0)
            axes[0].hlines(h_line_y, x_min, x_max,colors='black', 
                           lw = 4.0)
        
        fl_name = os.path.join(output_folder_inv, file_name)
        fig.savefig(fl_name, dpi = 250)
        
        
        
        
        
        
        
        
        
        
        
        
    def _get_rf_app_obs(self):
        st_obj = self.get_st_obj()
        if ((self.stack_name_to_inv == 'synthetic') 
            and (self.st_kind == 'real')):
            kind = 'Synthetic_Inversion_from_real_data'
            if (self.noise_level > 0.0):
                rf_to_inv  = st_obj.syn_l_stack
                app_vel_obs= rf_to_inv[0].app_vel 
                rf_r_data = rf_to_inv[0].data[:len(st_obj.l_stack[0].data)]
                slowness = rf_to_inv[0].ray_p_s_to_km
                output_folder_inv = st_obj.save_folder+str(kind)+'/'
            else:
                raise Exception('You need to define a noise_level > 0.0 for synthetic '+
                       'inversion of this station')
        elif ((self.stack_name_to_inv != 'synthetic') 
            and (self.st_kind == 'real')):
            stacked_present = False
            for key in st_obj.stacked_dict.keys():
                if (self.stack_name_to_inv  == key):
                    stacked_present = True
                    key_to_inv = key
            
            if (stacked_present == False):
                raise Exception('Couldnt find '+ self.stack_name_to_inv+'.')
            kind = key_to_inv.replace(" ", "__")
            rf_to_inv  = st_obj.stacked_dict[self.stack_name_to_inv]
            app_vel_obs= rf_to_inv[0].app_vel 
            rf_r_data = rf_to_inv[0].data[:len(st_obj.l_stack[0].data)]
            slowness = rf_to_inv[0].ray_p_s_to_km
            output_folder_inv = st_obj.save_folder+str(kind)+'/'           
        else:
            kind = 'Synthetic_Inversion'
            app_vel_obs = st_obj.apparant_vel_org
            rf_r_data = st_obj.rf_r_from_ind1
            slowness = st_obj.slowness
            output_folder_inv = os.path.join(st_obj.saving_directory, kind) 
        if (os.path.isdir(output_folder_inv)):
            pass 
        else:
            os.mkdir(output_folder_inv)
        return(kind, app_vel_obs, rf_r_data, slowness, output_folder_inv)
              
    def _define_grid_models(self):
        
        nmodel_all = self.nmodel
        init_lthickness_abs = \
            copy.deepcopy(self.init_obj.initial_layer_thickness_abs)
        init_velocity = copy.deepcopy(self.init_obj.initial_velocity)
        init_boundary = copy.deepcopy(self.init_obj.init_boundary)
        init_dif_boundary = copy.deepcopy(self.init_obj.init_dif_boundary)
        max_pert_all = copy.deepcopy(self.init_obj.velocity_change_constraint)
        vs_info = copy.deepcopy(self.init_obj.vs_info)
        layer_info = copy.deepcopy(self.init_obj.layer_info)
        
        
        
        vel_param_list = []
        for i in range(nmodel_all):
            vs_info_d = copy.deepcopy(vs_info)
            layer_info_d = copy.deepcopy(layer_info)
                

            new_boundary = []
            for i, el in enumerate(init_boundary):
                boundary_change = np.random.uniform(init_dif_boundary[i][0],
                                                    init_dif_boundary[i][1])
                n_boundary = el + boundary_change
                for j in range(len(vs_info)):
                    if (vs_info[j][1] == el):
                        vs_info_d[j][1] = n_boundary
                        vs_info_d[j+1][0] = n_boundary
                for j in range(len(layer_info)):
                    if (layer_info[j][1] == el):
                        layer_info_d[j][1] = n_boundary
                        layer_info_d[j+1][0] = n_boundary
                new_boundary.append(n_boundary)
            
                
            vel_param_n = ut.Vel_paramterize(vel_info = vs_info_d, 
                                    layer_info = layer_info_d, 
                                    vp_to_vs= self.init_obj.vp_to_vs)
            
            lthickness_abs = copy.deepcopy(vel_param_n.layer_thickness_abs)
            velocity = copy.deepcopy(vel_param_n.vel_s)
            max_pert = []
            for i in range(len(vs_info)):
                for el in lthickness_abs:
                    if ((el > vs_info[i][0]) and (el <= vs_info[i][1])):
                        max_pert.append(max_pert_all[i])
            
            
            vel_s_init_4_perturb = init_velocity.copy()
            vel_s_perturbed = perturb_vel_4_pkg(lthickness_abs,
                                              velocity,
                                              max_pert = max_pert,
                                              max_val_pn = 0.15, 
                                              lezhendre_ind = True)
            vel_param_n.vel_s = vel_s_perturbed
            vel_param_list.append(vel_param_n)
        return(vel_param_list)
    
    
    def plot_results(self, fig_name = 'Inversion_outputs.png'):
        inv_info = self.inv_info.copy()
        (kind, app_vel_obs, rf_r_data, slowness, output_folder_inv) = \
            self._get_rf_app_obs()
        if (kind == 'Synthetic_Inversion_from_real_data'):
            title = 'Synthetic inversion using real datasets'
            time_vec = self.st_obj.syn_l_stack[0].time_vec
            rf_std = self.st_obj.syn_l_stack[0].rf_std
        elif (kind == 'Synthetic_Inversion'):
            title = 'Synthetic inversion '
            time_vec = self.st_obj.time_cuted
            rf_std = np.zeros(shape = np.shape(time_vec))
        else:
            title = 'Inversion of '+ self.stack_name_to_inv
            # title = self.stack_name_to_inv
            time_vec = \
                self.st_obj.stacked_dict[self.stack_name_to_inv][0].time_vec
            rf_std = self.st_obj.stacked_dict[self.stack_name_to_inv][0].rf_std
        
        
        time_vec_d = time_vec - 25.0
        ind_fig_plot = np.argwhere(np.abs(time_vec_d) == 
                          np.min(np.abs(time_vec_d)))[0][0]
        nfig = len(inv_info['all_iter'])
        fig = plt.figure(constrained_layout=True, figsize=(32, 19.77))
        subfigs = fig.subfigures(2, int(nfig / 2), edgecolor = 'white', 
                                 linewidth= 0.1, frameon=True, 
                                 wspace=0.05, hspace=0.05)
        fig.suptitle(title, fontsize = 24)
        norms_list = []
        for i, el in enumerate(inv_info['all_iter']):
            norms_list.append(el[3] + el[4])
        norms_list = np.array(norms_list)
        ind_best = np.argwhere(norms_list == np.min(norms_list))[0][0]
        idum = 0
        divide_to_row = []
        divide_to_col = []
        for i, el in enumerate(self.finer_ndivide):
            if (np.mod(i, 2) == 1):
                divide_to_row.append(abs(el))
            else:
                divide_to_col.append(abs(el))
        kdum = -1
        for row_iter in np.arange(1):
            for col_iter in np.arange(int(nfig / 2)):
                kdum += 1
                subfig_title = ('Divided to '+str(divide_to_row[kdum])+ 
                                ' with Initial factors')
                ax_subfig1 = \
                    subfigs[row_iter,col_iter].subplots(nrows=1, ncols = 3,
                                gridspec_kw={'width_ratios': [1, 0.5, 0.5],
                                             'wspace': 0.01})
                
                if (idum == ind_best): 
                    subfigs[row_iter,col_iter].set_facecolor('0.85')
                norm = (inv_info['all_iter'][idum][3] + 
                        inv_info['all_iter'][idum][4])
                
                subfig_title = subfig_title + ', norm = '+str(round(norm, 2))
                subfigs[row_iter,col_iter].suptitle(subfig_title, 
                                                    fontsize = 20)
                info = inv_info['all_iter'][idum]
                self._axes_ploter(ax_subfig1, info, rf_r_data, app_vel_obs, 
                                  time_vec, rf_std, ind_fig_plot,
                                 low_smooth = True, subfig_title= subfig_title)
                idum += 2
        idum = 1
        mdum = -1
        for row_iter in np.arange(1, 2):
            for col_iter in np.arange(int(nfig / 2)):
                mdum += 1
                subfig_title = ('Divided to '+str(divide_to_col[mdum])+ 
                                ' with Increased factors')
                ax_subfig1 = \
                    subfigs[row_iter,col_iter].subplots(nrows=1, ncols = 3,
                                gridspec_kw={'width_ratios': [1, 0.5, 0.5], 
                                             'wspace': 0.01})
                subfigs[row_iter,col_iter].suptitle(subfig_title, 
                                                    fontsize = 20)
                
                if (idum == ind_best): 
                    subfigs[row_iter,col_iter].set_facecolor('0.85')
                norm = (inv_info['all_iter'][idum][3] + 
                        inv_info['all_iter'][idum][4])
                
                subfig_title = subfig_title + ', norm = '+str(round(norm, 2))
                subfigs[row_iter,col_iter].suptitle(subfig_title, 
                                                    fontsize = 20)
                info = inv_info['all_iter'][idum]
                self._axes_ploter(ax_subfig1, info, rf_r_data, app_vel_obs, 
                                  time_vec, rf_std, ind_fig_plot,
                                 low_smooth = True, subfig_title= subfig_title)
                idum += 2
        fl_name = os.path.join(output_folder_inv, fig_name)
        
        fig.savefig(fl_name, dpi = 250)
        
    def _pso_plotter(self, name = 'PSO_output.png'):
        label_size = 14
        syn_color = 'aqua'
        layer_info_4_interpolate = copy.deepcopy(self.init_obj.layer_info)
        for el in layer_info_4_interpolate:
            el[2] = el[2] * np.max(self.ndivide_list)
        vel_param_4_interpolate = \
            ut.Vel_paramterize(vel_info = self.init_obj.vs_info, 
                                layer_info = layer_info_4_interpolate, 
                                vp_to_vs= self.init_obj.vp_to_vs)
        lthickness_abs_interpolate = vel_param_4_interpolate.layer_thickness_abs
        
        self.mean_vel_param = copy.deepcopy(vel_param_4_interpolate)
        
        time_vec = self.st_obj.time_vec
        inv_info_pso = copy.deepcopy(self.inv_info)
        initial_lthick = inv_info_pso['best_inv'][0]
        initial_vel = inv_info_pso['best_inv'][1]
        initial_lthick_abs = ut.find_lthickness_abs(initial_lthick)
        final_lthick = inv_info_pso['best_inv'][7]
        final_vel = inv_info_pso['best_inv'][2]
        final_lthick_abs = ut.find_lthickness_abs(final_lthick)
        final_rf = inv_info_pso['best_inv'][5]
        final_app_vel = inv_info_pso['best_inv'][6]
        final_norm = inv_info_pso['best_inv'][3] + inv_info_pso['best_inv'][4]
        
        
        subfig, ax_subfig = plt.subplots(nrows=1, ncols = 6,
                            gridspec_kw={'width_ratios':[1, 1, 0.4, 0.8, 0.8, 0.8]},
                            figsize=(18, 10))
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.1)
        if ('synthetic' in self.stack_name_to_inv):
            v_s_interp_syn = interp_velocity(\
                            lthickness= self.init_obj.synthetic_layer_thickness
                            ,velocity= self.init_obj.synthetic_velocity
                            ,lthickness_abs_input= lthickness_abs_interpolate)
        else:
            v_s_interp_syn = interp_velocity(\
                            lthickness= initial_lthick
                            ,velocity= initial_vel
                            ,lthickness_abs_input= lthickness_abs_interpolate)
            
        v_s_interp_pso = interp_velocity(\
                        lthickness= final_lthick
                        ,velocity= final_vel
                        ,lthickness_abs_input= lthickness_abs_interpolate)
        self.mean_vel_af_PSO = copy.deepcopy(v_s_interp_pso)
        pso_dif = []
        for i in range(len(v_s_interp_syn)):
            pso_dif.append(abs(v_s_interp_pso[i] - v_s_interp_syn[i]))
        ax_subfig[2].plot(pso_dif,
                            lthickness_abs_interpolate - 
                            lthickness_abs_interpolate[0], 
                            color = 'red', lw= 1)
        ax_subfig[2].scatter(pso_dif,
                            lthickness_abs_interpolate - 
                            lthickness_abs_interpolate[0], 
                            color = 'red', lw= 1, s= 10)
        (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
           ut.find_xminmax(initial_lthick_abs, 
                           initial_vel)
        ax_subfig[0].vlines(v_line_x, y_min, y_max, colors='gray',
                           alpha = 0.9)
        ax_subfig[0].hlines(h_line_y, x_min, x_max, colors='gray',
                           alpha = 0.9, label = 'Pseudo-Initial model')
        
        (kind, app_vel_obs, rf_r_data, slowness, output_folder_inv) = \
            self._get_rf_app_obs()
        self.output_folder_inv = output_folder_inv
        ax_subfig[3].plot(rf_r_data, 
                          time_vec, 
                               color = syn_color, 
                               alpha = 1.0, lw = 2.4)
        if ('Synthetic' in kind):
            (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
                ut.find_xminmax(self.init_obj.synthetic_layer_thickness_abs, 
                                self.init_obj.synthetic_velocity)
            ax_subfig[1].vlines(v_line_x, y_min, y_max, colors=syn_color,
                                alpha = 1.0, lw = 2.5)
            ax_subfig[1].hlines(h_line_y, x_min, x_max, colors=syn_color,
                                alpha = 1.0, lw = 2.5)
            ax_subfig[4].plot(app_vel_obs,
                          self.init_obj.filt_list,
                               color = syn_color, 
                               alpha = 1.0, lw = 2.4, label = 'Synthetic')
        else:
            ax_subfig[4].plot(app_vel_obs,
                          self.init_obj.filt_list,
                               color = syn_color, 
                               alpha = 1.0, lw = 2.4, label = 'Observed')
        (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
            ut.find_xminmax(self.init_obj.initial_layer_thickness_abs, 
                            self.init_obj.initial_velocity)
        ax_subfig[0].vlines(v_line_x, y_min, y_max, colors='black',
                            alpha = 1.0, lw = 2.4)
        ax_subfig[0].hlines(h_line_y, x_min, x_max, colors='black',
                            alpha = 1.0, lw = 2.4, label = 'Initial Model')
        
        (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
            ut.find_xminmax(lthickness_abs_interpolate, 
                            v_s_interp_pso)
        ax_subfig[1].vlines(v_line_x, y_min, y_max, colors='red',
                            alpha = 1.0, lw = 2.0)
        ax_subfig[1].hlines(h_line_y, x_min, x_max,  colors='red',
                            alpha = 1.0, lw = 2.0)
        (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
            ut.find_xminmax(final_lthick_abs, 
                            final_vel)
        ax_subfig[1].vlines(v_line_x, y_min, y_max, colors='navy',
                            alpha = 1.0, lw = 2.0)
        ax_subfig[1].hlines(h_line_y, x_min, x_max,  colors='navy',
                            alpha = 1.0, lw = 2.0)
        ax_subfig[3].plot(final_rf, 
                          time_vec, 
                               color = 'navy', 
                               alpha = 0.8)
        ax_subfig[4].plot(final_app_vel,
                          self.init_obj.filt_list,
                               color = 'navy',
                               alpha = 0.8, label = 'PSO')
        self._calculate_according_to_mean_vel(file_name= 'PSO_interpolated.png')
        ax_subfig[3].plot(self.inv_info['mean_vel_inv'][5], 
                          time_vec, 
                               color = 'red', 
                               alpha = 0.8)
        ax_subfig[4].plot(self.inv_info['mean_vel_inv'][6],
                          self.init_obj.filt_list,
                               color = 'red',
                               alpha = 0.8, label = 'Interpolated PSO')
        ## adding pso gbest evolution
        tthickness_path = self.inv_info['name_pso']
        with open(tthickness_path, 'rb') as f1:
            tthickness = pickle.load(f1)
        all_particle = tthickness.all_particles



        obj_particle_all = np.zeros(shape = (len(all_particle[0]['iter_all_obj']), 
                                             len(all_particle)))
        for i, particle in enumerate(all_particle):
            for j in range(len(particle['iter_all_obj'])):
                iter_obj =  particle['iter_all_obj'][j]
                obj_particle_all[j, i] = iter_obj[1]
        gbest_cond = []
        gbest_cond.append(tthickness.init_obj_final)
        best_per_iter = 15.0
        for i in range(len(obj_particle_all)):
            best_this_iter = np.min(obj_particle_all[i, :])
            if (best_this_iter < best_per_iter):
                gbest_cond.append(best_this_iter)
                best_per_iter = best_this_iter
            else:
                gbest_cond.append(best_per_iter)
        
        ax_subfig[5].scatter(gbest_cond, range(len(gbest_cond)), s= 20, c= "blue")
        ax_subfig[5].plot(gbest_cond, range(len(gbest_cond)), color = "blue")
        
        
        ax_subfig[5].set_ylim([len(gbest_cond), 0])
        ax_subfig[5].set_xlim([np.min(gbest_cond) - 0.05, gbest_cond[0] + 0.1])
        ax_subfig[5].grid(True)
        ax_subfig[5].set_title("Cost function\n evolution", size = label_size)
        ax_subfig[5].set_xlabel("Cost function", size= label_size)
        ax_subfig[5].set_ylabel("Iter No", size = label_size)
        
        ax_subfig[0].set_xlabel('S Velocity (km/s)', size = label_size)
        ax_subfig[0].set_ylabel('Depth (km)', size = label_size)
        ax_subfig[0].set_ylim([0, self.init_obj.max_depth])
        ax_subfig[0].set_ylim(ax_subfig[0].get_ylim()[::-1])
        ax_subfig[0].set_xlim([2, 5.5])
        ax_subfig[0].grid(True)
        ax_subfig[0].set_title('Initial Models')
        ax_subfig[1].set_xlabel('S Velocity (km/s)', size = label_size)
        ax_subfig[1].set_ylim([0, self.init_obj.max_depth])
        ax_subfig[1].set_ylim(ax_subfig[1].get_ylim()[::-1])
        ax_subfig[1].set_xlim([1.5, 5.5])
        ax_subfig[1].grid(True)
        ax_subfig[1].set_title('Estimated Models')
        ax_subfig[2].set_ylim([self.init_obj.max_depth, 0])
        ax_subfig[2].set_xticks([0.2, 0.5])
        ax_subfig[2].grid(True)
        ax_subfig[2].set_title('Interpolated\n difference')
        ax_subfig[3].set_ylim([self.init_obj.inv_af, 
                               -self.init_obj.inv_bf])
        ax_subfig[3].set_ylabel('Time (s)', size = label_size)
        ax_subfig[3].grid(True)
        ax_subfig[3].set_title('Estimated\n RFs')
        ax_subfig[4].set_ylim(ax_subfig[3].get_ylim())
        ax_subfig[4].grid(True)
        ax_subfig[4].set_ylabel('Filter periods (s)', size = label_size)
        ax_subfig[4].set_xlabel('Apparent velocity (km/s)', size = label_size)
        ax_subfig[4].set_title('Estimated\n Apparent Velocities')
        ax_subfig[4].legend()
        ax_subfig[0].legend()
        
        fl_name = os.path.join(self.output_folder_inv, name)
        subfig.savefig(fl_name, dpi = 200)
    def _gs_plotter(self, name = 'gs_output.png'):
        label_size = 14
        time_vec = self.st_obj.time_vec
        fl_to_load = self.gs_inv_info_list_path
        syn_color = 'aqua'
        with open(fl_to_load, 'rb') as f1:
            gs_inv_info_list = pickle.load(f1)
        
        layer_info_4_interpolate = copy.deepcopy(self.init_obj.layer_info)
        for el in layer_info_4_interpolate:
            el[2] = el[2] * np.max(self.ndivide_list)
        # layer_info_4_interpolate = [[0.0, self.init_obj.max_depth, 
        #                             int(self.init_obj.max_depth)]]
        vel_param_4_interpolate = \
            ut.Vel_paramterize(vel_info = self.init_obj.vs_info, 
                                layer_info = layer_info_4_interpolate, 
                                vp_to_vs= self.init_obj.vp_to_vs)
        lthickness_abs_interpolate = vel_param_4_interpolate.layer_thickness_abs
        
        self.mean_vel_param = copy.deepcopy(vel_param_4_interpolate)
        
        
        # lthickness_abs_interpolate = \
        #     self.init_obj.interpolate_vel_param.layer_thickness_abs.copy()
        plot_info = {}
        plot_info['initial_vel'] = []
        plot_info['initial_lthickness'] = []
        plot_info['initial_lthickness_abs'] = []
        plot_info['final_vel'] = []
        plot_info['norm'] = []
        plot_info['final_lthickness'] = []
        plot_info['final_lthickness_abs'] = []
        plot_info['final_rf'] = []
        plot_info['final_app_vel'] = []
        plot_info['final_vel_interpolate'] = []
        iter_all_number = len(gs_inv_info_list[0]['best_inv'][8])
        plot_info['norm_all'] = np.zeros(shape = (iter_all_number, 
                                                  len(gs_inv_info_list)))
        self.vel_param_list = []
        for i, el in enumerate(gs_inv_info_list):
            self.vel_param_list.append(el['vel_param'])
            plot_info['norm'].append(el['best_inv'][3] + 
                                     el['best_inv'][4])
            plot_info['initial_lthickness'].append(el['best_inv'][0])
            plot_info['initial_vel'].append(el['best_inv'][1])
            plot_info['initial_lthickness_abs'].\
                append(ut.find_lthickness_abs(el['best_inv'][0]))
            plot_info['final_vel'].append(el['best_inv'][2])
            plot_info['final_lthickness'].append(el['best_inv'][7])
            plot_info['final_lthickness_abs'].\
                append(ut.find_lthickness_abs(el['best_inv'][7]))
            plot_info['final_rf'].append(el['best_inv'][5])
            plot_info['final_app_vel'].append(el['best_inv'][6])
            v_s_interp = interp_velocity(lthickness= el['best_inv'][7]
                            ,velocity= el['best_inv'][2]
                            ,lthickness_abs_input= lthickness_abs_interpolate)
            plot_info['final_vel_interpolate'].append(v_s_interp)
            for j, norm in enumerate(el['best_inv'][8]):    
                plot_info['norm_all'][j, i] = norm[4]
        
        all_vel_interp = np.array(plot_info['final_vel_interpolate'])
        
        nmodel, nlayer = np.shape(all_vel_interp)
        mean_vel = np.zeros(shape = (nlayer))
        std_vel = np.zeros(shape = (nlayer))
        for i in range(nlayer):
            layer_vel = []
            for j in range(nmodel):
                layer_vel.append(all_vel_interp[j, i])
            # mean_vel[i] = np.median(layer_vel)
            mean_vel[i] = np.mean(layer_vel)
            std_vel[i] = np.std(layer_vel)
        mean_vel = np.array(mean_vel)
        self.mean_vel_af_gs = copy.deepcopy(mean_vel)
        norm_mean = []
        for j in range(nmodel):
            norm_mean.append(np.linalg.norm(all_vel_interp[j, :] - 
                                            mean_vel))       
        norm_mean = np.array(norm_mean)
        ind_best = np.argwhere(norm_mean == np.min(norm_mean))[0][0]
        subfig, ax_subfig = plt.subplots(nrows=1, ncols = 6,
                            gridspec_kw={'width_ratios':[1, 1, 0.4, 0.8, 0.8, 0.8]},
                            figsize=(18, 10))
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.1)
        norm_to_work_d = plot_info['norm'].copy()
        # norm_to_work_d = norm_mean
        sorted_indices = np.argsort(norm_to_work_d)
        norm_to_work = np.sort(norm_to_work_d)
        for key in plot_info.keys():
            dummy_list = plot_info[key].copy()
            if (key == 'norm_all'):
                plot_info[key] = np.zeros(shape = (iter_all_number, 
                                                          len(gs_inv_info_list)))
                for j, index in enumerate(sorted_indices):
                    plot_info[key][:, j] = dummy_list[:, index]
            else:
                plot_info[key] = []
                for index in sorted_indices:
                    plot_info[key].append(dummy_list[index])
        # viridis=cm.get_cmap('rainbow', nmodel)
        norm= matplotlib.colors.BoundaryNorm(norm_to_work, 
                                                len(norm_to_work))
        cm = plt.cm.bwr(np.linspace(0, 1, len(norm_to_work)))
        cmap = ListedColormap(cm) 
        alpha_d = np.linspace(1, 0.3, len(norm_to_work))
        for i in np.arange(-1, -len(plot_info['norm']) - 1, -1):
            (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
                ut.find_xminmax(plot_info['initial_lthickness_abs'][i], 
                                plot_info['initial_vel'][i])
            ax_subfig[0].vlines(v_line_x, y_min, y_max, colors='gray',
                                alpha = 0.6)
            ax_subfig[0].hlines(h_line_y, x_min, x_max, colors='gray',
                                alpha = 0.6)
            (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
                ut.find_xminmax(plot_info['final_lthickness_abs'][i], 
                                plot_info['final_vel'][i])
            
            ax_subfig[1].vlines(v_line_x, y_min, y_max, colors=cm[i],
                                alpha = alpha_d[i])
            ax_subfig[1].hlines(h_line_y, x_min, x_max, colors=cm[i],
                                alpha = alpha_d[i])
            
            (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
                ut.find_xminmax(lthickness_abs_interpolate, 
                                mean_vel)
            ax_subfig[1].vlines(v_line_x, y_min, y_max, colors='red',
                                alpha = 1.0, lw = 2.0)
            ax_subfig[1].hlines(h_line_y, x_min, x_max,  colors='red',
                                alpha = 1.0, lw = 2.0)
            
            ax_subfig[3].plot(plot_info['final_rf'][i], 
                              time_vec, 
                                   color = cm[i], 
                                   alpha = alpha_d[i])
            ax_subfig[4].plot(plot_info['final_app_vel'][i],
                              self.init_obj.filt_list,
                                   color = cm[i],
                                   alpha = alpha_d[i])
            ax_subfig[5].scatter(plot_info['norm_all'][:, i], 
                                 np.arange(1, len(plot_info['norm_all'][:, i]) + 1), 
                                 s= 20, color= cm[i], alpha = alpha_d[i])
            ax_subfig[5].plot(plot_info['norm_all'][:, i], 
                                 np.arange(1, len(plot_info['norm_all'][:, i]) + 1), 
                                 color = cm[i], 
                                 alpha = alpha_d[i])
            
            
            
        if ('synthetic' in self.stack_name_to_inv):
            v_s_interp_syn = interp_velocity(\
                            lthickness= self.init_obj.synthetic_layer_thickness
                            ,velocity= self.init_obj.synthetic_velocity
                            ,lthickness_abs_input= lthickness_abs_interpolate)
            mean_dif = []
            for i in range(len(v_s_interp_syn)):
                mean_dif.append(abs(mean_vel[i] - v_s_interp_syn[i]))
            ax_subfig[2].plot(mean_dif,
                                lthickness_abs_interpolate - 
                                lthickness_abs_interpolate[0], 
                                color = 'red', lw= 1)
            ax_subfig[2].scatter(mean_dif,
                                lthickness_abs_interpolate - 
                                lthickness_abs_interpolate[0], 
                                color = 'red', lw= 1, s= 10)
            
            
            
            
            (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
                ut.find_xminmax(self.init_obj.synthetic_layer_thickness_abs, 
                                self.init_obj.synthetic_velocity)
            ax_subfig[1].vlines(v_line_x, y_min, y_max, colors=syn_color,
                                alpha = 1.0, lw = 2.5)
            ax_subfig[1].hlines(h_line_y, x_min, x_max, colors=syn_color,
                                alpha = 1.0, lw = 2.5)
        else:
            ax_subfig[2].plot(std_vel,
                                lthickness_abs_interpolate, 
                                color = 'red', lw= 1)
            ax_subfig[2].scatter(std_vel,
                                lthickness_abs_interpolate, 
                                color = 'red', lw= 1, s= 10)
        (kind, app_vel_obs, rf_r_data, slowness, output_folder_inv) = \
            self._get_rf_app_obs()
        ax_subfig[3].plot(rf_r_data, 
                          time_vec, 
                               color = syn_color, 
                               alpha = 1.0, lw = 2.4)
        ax_subfig[4].plot(app_vel_obs,
                          self.init_obj.filt_list,
                               color = syn_color, 
                               alpha = 1.0, lw = 2.4)
        (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
            ut.find_xminmax(self.init_obj.initial_layer_thickness_abs, 
                            self.init_obj.initial_velocity)
        ax_subfig[0].vlines(v_line_x, y_min, y_max, colors=syn_color,
                            alpha = 1.0, lw = 2.4)
        ax_subfig[0].hlines(h_line_y, x_min, x_max, colors=syn_color,
                            alpha = 1.0, lw = 2.4)
            
            
        ax_subfig[0].set_xlabel('S Velocity (km/s)', size = label_size)
        ax_subfig[0].set_ylabel('Depth (km)', size = label_size)
        ax_subfig[0].set_ylim([0, self.init_obj.max_depth])
        ax_subfig[0].set_ylim(ax_subfig[0].get_ylim()[::-1])
        ax_subfig[0].set_xlim([2, 5.5])
        ax_subfig[0].grid(True)
        ax_subfig[0].set_title('Initial Models')
        ax_subfig[1].set_xlabel('S Velocity (km/s)', size = label_size)
        ax_subfig[1].set_ylim([0, self.init_obj.max_depth])
        ax_subfig[1].set_ylim(ax_subfig[1].get_ylim()[::-1])
        ax_subfig[1].set_xlim([1.5, 5.5])
        ax_subfig[1].grid(True)
        ax_subfig[1].set_title('Estimated Models')
        ax_subfig[2].set_ylim([self.init_obj.max_depth, 0])
        ax_subfig[2].set_xticks([0.2, 0.5])
        ax_subfig[2].grid(True)
        ax_subfig[5].set_ylim([iter_all_number + 1, 1])
        ax_subfig[5].grid(True)
        ax_subfig[5].set_title("Objective function\n evolution", size = label_size)
        ax_subfig[5].set_xlabel("Objective function", size= label_size)
        ax_subfig[5].set_ylabel("Iter No", size = label_size)
        if ('synthetic' in self.stack_name_to_inv):
            ax_subfig[2].set_title('Mean\n difference')
        else:
            ax_subfig[2].set_title('STD\n')
        
        ax_subfig[3].set_ylabel('Time (s)', size = label_size)
        ax_subfig[3].set_ylim([self.init_obj.inv_af, 
                               -self.init_obj.inv_bf])
        ax_subfig[3].grid(True)
        ax_subfig[3].set_title('Estimated\n RFs')
        ax_subfig[4].set_ylabel('Filter periods (s)', size = label_size)
        ax_subfig[4].set_xlabel('Apparent velocity (km/s)', size = label_size)
        ax_subfig[4].set_ylim(ax_subfig[3].get_ylim())
        ax_subfig[4].grid(True)
        ax_subfig[4].set_title('Estimated\n Apparent Velocities')
        
        rounded_norm_to_work = [round(x, 4) for x in norm_to_work]
        rounded_norm_to_work = np.linspace(np.min(rounded_norm_to_work), 
                                           np.max(rounded_norm_to_work), 
                                           6)
        subfig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),ax = ax_subfig[5],
                        label='Objective function')
        fl_name = os.path.join(self.output_folder_inv, name)
        subfig.savefig(fl_name, dpi = 200)
        return(gs_inv_info_list)
    def _axes_ploter(self, axes, info, rf_obs, app_vel_obs, time_vec, rf_std,
                     ind_fig_plot,  subfig_title, low_smooth = True, 
                     init_model_color = 'black'):
        tick_size = 14
        label_size = 16
        lthick = info[7]
        vel = info[2]
        rf_norm = info[3]
        app_norm = info[4]
        rf = info[5]
        app_vel = info[6]
        rf_upper = rf_obs + rf_std
        rf_lower = rf_obs - rf_std
        lthick_abs = []
        depth = 0.0
        for i, el in enumerate(lthick):
            if (el == 0.0):
                depth = depth + 10
            else:
                depth = depth + el
            lthick_abs.append(depth)
        (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
            ut.find_xminmax(lthick_abs, vel)
        axes[0].vlines(v_line_x, y_min, y_max, colors=init_model_color, lw = 3.0)
        axes[0].hlines(h_line_y, x_min, x_max,colors=init_model_color, lw = 3.0)
        axes[0].set_xlabel('S Velocity (km/s)', size = label_size)
        axes[0].set_ylabel('Depth (km)', size = label_size)
        axes[0].set_ylim([0, lthick_abs[-2]])
        axes[0].set_ylim(axes[0].get_ylim()[::-1])
        axes[0].set_xlim([1.5, 5.5])
        axes[0].tick_params(axis="x", labelsize= tick_size)
        axes[0].tick_params(axis="y", labelsize= tick_size)
        axes[0].grid(True)
        
        axes[1].plot(rf_obs[:ind_fig_plot],
                        time_vec[:ind_fig_plot], 
                        color = 'black', lw= 3)
        
        axes[1].plot(rf_upper[:ind_fig_plot],
                        time_vec[:ind_fig_plot], 
                        color = 'black', lw= 0.1)
        axes[1].plot(rf_lower[:ind_fig_plot],
                        time_vec[:ind_fig_plot], 
                        color = 'black', lw= 0.1)
        axes[1].plot(rf[:ind_fig_plot],
                        time_vec[:ind_fig_plot], 
                        color = 'red', lw= 3)
        axes[1].fill_betweenx(y = time_vec[:ind_fig_plot], 
                              x1= rf_upper[:ind_fig_plot], 
                              x2= rf_lower[:ind_fig_plot], 
                              color = 'gray', alpha = 0.5)
        axes[1].tick_params(axis="x", labelsize= tick_size)
        axes[1].tick_params(axis="y", labelsize= tick_size)
        axes[1].set_ylabel('Time (s)', size = label_size)
        axes[1].set_ylim([max(time_vec[:ind_fig_plot]), 
                          min(time_vec[:ind_fig_plot])])
        axes[1].grid(True)
        
        axes[2].plot(app_vel_obs,
                        self.init_obj.filt_list, 
                        color = 'black', lw= 3)
        axes[2].plot(app_vel,
                        self.init_obj.filt_list, 
                        color = 'red', lw= 3)
        axes[2].tick_params(axis="x", labelsize= tick_size)
        axes[2].tick_params(axis="y", labelsize= tick_size)
        axes[2].set_ylabel('Filter Periods (s)', size = label_size)
        axes[2].set_ylim(axes[1].get_ylim())
        axes[2].grid(True)
        
            
        
        
        
        
                
    
#%%
def _pso_inv_parser(args):
    (iter_all, num_procs, kind, PSO_nparticle, PSO_maxiter, init_obj_d, 
     app_vel_to_inv, rf_to_inv ,slowness_stack, ndivide_list, save_dir_root,
     PSO_nthread, vel_param_list, finer_ndivide, pso_name) = args
    
    init_obj = copy.deepcopy(init_obj_d)
    vel_param_best = vel_param_list 
    folder_name = save_dir_root 
    
    
    vel_s_perturbed = vel_param_best.vel_s.copy()
    layers_thickness = vel_param_best.layer_thickness.copy()
    layer_thickness_abs = vel_param_best.layer_thickness_abs.copy()
    
    vel_param_run = vel_param_best
    vel_param_run.init_boundary = init_obj.init_boundary.copy()
    vel_param_run.init_dif_boundary = init_obj.init_dif_boundary.copy()
    vel_param_run.initial_layering = init_obj.layering_4_run.copy()
    pso_init_layer = init_obj.layering_4_run.copy()
    for i, el in enumerate(pso_init_layer):
        pso_init_layer[i] = el * np.max(np.abs(ndivide_list))
                            
    vel_param_run.PSO_layering = pso_init_layer.copy()

    
    syn_ind = False
    if ('Synthetic' not in kind):
        syn_ind = False
        vel_synth_init = init_obj.initial_velocity
        layer_thickness_syn = init_obj.initial_layer_thickness
    else:
        vel_synth_init = init_obj.synthetic_velocity
        layer_thickness_syn = init_obj.synthetic_layer_thickness
        syn_ind = True
    inv_info= mia.inv_all(app_vel_obs_master =app_vel_to_inv, 
                              vel_param= vel_param_run, 
                              rf_obs_master= rf_to_inv, 
                              slowness_input= slowness_stack,
                              ndivide_list= ndivide_list, 
                              stacked = True,
                              use_app_vel=False, 
                              name_pso_inp = pso_name,
                              synthetic = syn_ind,  
                              vel_synth_init= vel_synth_init,
                              use_pso = True, 
                              inv_bf = init_obj.inv_bf, 
                              inv_af = init_obj.inv_af, 
                              rf_method = 'iterative',
                              rf_method_sens = 'waterlevel',
                              dt = init_obj.dt,
                              layers_thickness_syn = layer_thickness_syn,
                              nsamp=init_obj.nsamp,
                              filt_list = init_obj.filt_list,
                              app_weight = init_obj.app_weight,
                              rf_weight = init_obj.rf_weight,
                              smooth_fac = init_obj.smooth_fac,
                              damp_fac = init_obj.damp_fac,
                              gauss_par = init_obj.gauss_par,
                              PSO_nparticle = PSO_nparticle, 
                              PSO_maxiter = PSO_maxiter,
                              PSO_nthread = PSO_nthread, 
                              close_fig = True, 
                              rf_normalize = 1,
                              finer_ndivide= finer_ndivide,
                              save_dir_root= folder_name)
    
    fl_inv_info = folder_name+'/inv_info'+str(iter_all)+'.bin'
    file1 = open(fl_inv_info, "wb") 
    pickle.dump(inv_info, file1)
    file1.close()
    return(inv_info) 
#%%
def perturb_vel_4_pkg(lthickness_abs, vel_s, max_pert,
                max_val_pn = 0.4, lezhendre_ind = True):
    
    
    thick_abs = lthickness_abs[:-1]
    max_lthickness = np.max(thick_abs)
    min_lthickness = np.min(thick_abs)
    if (lezhendre_ind):
        pn_input = []
        for el in thick_abs:
            pn_input.append(((el/max_lthickness)) -1)
        p3_x = []
        for el in pn_input:
            p3_x.append(0.5*((5 * el**3.0) - (3* el)))
        p3_x = np.array(p3_x)
        p3_x = np.array(p3_x) / np.max(abs(p3_x))
        p3_x = max_val_pn * p3_x
        
        # plt.plot(p3_x)
        
        vel_s_perturbed = []
        idum = -1
        for i, el in enumerate(vel_s[:len(vel_s)-1]):
            idum += 1
            vel_accept = False 
            while (vel_accept == False):
                vel = (el + p3_x[idum] +
                                    np.random.uniform(-max_pert[i], max_pert[i]))
                # vel = (el + p3_x[idum] +
                #                     np.random.normal(loc= 0.0, scale= max_pert))
                if ((vel > el - max_pert[i]) and (vel < el + max_pert[i])):
                    vel_accept = True
            vel_s_perturbed.append(vel)
        vel_s_perturbed.append(max(vel_s[-1], vel_s_perturbed[-1]))
    else:
        vel_s_perturbed = []
        for i, el in enumerate(vel_s[:len(vel_s)-1]):
            vel_accept = False 
            while (vel_accept == False):
                # vel = (el + np.random.normal(loc= 0.0, scale= max_pert))
                vel = (el + np.random.uniform(-max_pert[i], max_pert[i]))
                if ((vel > el+ max_pert[i]) and (vel < el - max_pert[i])):
                    vel_accept = True
            vel_s_perturbed.append(vel)
        vel_s_perturbed.append(max(vel_s[-1], vel_s_perturbed[-1]))
    # print(p3_x)
    
    return(vel_s_perturbed)

#%%
def _gs_inv_parser(args):
    (iter_all_arr, num_procs, kind, init_obj, app_vel_to_inv, 
    rf_to_inv ,slowness_stack, ndivide_list, save_dir_root,
    vel_param_list, final_run) = args
    
    inversion_pre_req = (init_obj.inv_bf, init_obj.inv_af, 
                         init_obj.dt, init_obj.nsamp, 
                         init_obj.filt_list, init_obj.app_weight, 
                         init_obj.rf_weight, init_obj.smooth_fac, 
                         init_obj.damp_fac, init_obj.gauss_par)
    syn_ind = False
    if ('Synthetic' not in kind):
        syn_ind = False
        vel_synth_init = init_obj.initial_velocity
        layer_thickness_syn = init_obj.initial_layer_thickness
    else:
        vel_synth_init = init_obj.synthetic_velocity
        layer_thickness_syn = init_obj.synthetic_layer_thickness
        syn_ind = True
    
    if (final_run):
        vel_param_best = vel_param_list
        ndivide_list_accurate = ndivide_list.copy()
        folder_name = os.path.join(save_dir_root,'RGS_pseudo_model_output') 
        
        vel_param_run = vel_param_best
        vel_s_perturbed = vel_param_best.vel_s.copy()
        layers_thickness = vel_param_best.layer_thickness.copy()
        layer_thickness_abs = vel_param_best.layer_thickness_abs.copy()
        (inv_bf, inv_af, 
         dt, nsamp, 
         filt_list, app_weight, 
         rf_weight, smooth_fac, 
         damp_fac, gauss_par) = inversion_pre_req
        inv_info= mia.inv_all(app_vel_obs_master =app_vel_to_inv, 
                                  vel_param= vel_param_run, 
                                  rf_obs_master= rf_to_inv, 
                                  slowness_input= slowness_stack,
                                  ndivide_list= ndivide_list, 
                                  stacked = True,
                                  use_app_vel=False, 
                                  name_pso_inp = False, synthetic = syn_ind,  
                                  vel_synth_init= vel_synth_init,
                                  use_pso = False, 
                                  inv_bf = inv_bf, 
                                  inv_af = inv_af, 
                                  rf_method = 'iterative',
                                  rf_method_sens = 'waterlevel',
                                  dt = dt,
                                  layers_thickness_syn = layer_thickness_syn,
                                  nsamp= nsamp,
                                  filt_list = filt_list,
                                  app_weight = app_weight,
                                  rf_weight = rf_weight,
                                  smooth_fac = smooth_fac,
                                  damp_fac = damp_fac,
                                  gauss_par = gauss_par,
                                  close_fig = True, 
                                  rf_normalize = 1,
                                  save_dir_root= folder_name)
        
        fl_inv_info = os.path.join(folder_name,'inv_info_best_gs.bin')
        file1 = open(fl_inv_info, "wb") 
        pickle.dump(inv_info, file1)
        file1.close()
        
        return(inv_info)
    else:
        grid_search_folder = os.path.join(save_dir_root,'grid_search_workplace')
        if (os.path.isdir(grid_search_folder)):
            pass 
        else:
            os.mkdir(grid_search_folder)
        ncol = int(len(iter_all_arr) / num_procs)
        nrow = num_procs
        iter_all_arr1 = iter_all_arr[:ncol * nrow]
        iter_all_arr2 = iter_all_arr[ncol * nrow:]
        iter_all_ar1_reshaped = iter_all_arr1.reshape((nrow, ncol))
        iter_all_arr_list = []
        for j in range(ncol):
            iter_all_arr_list.append(iter_all_ar1_reshaped[:, j])
        iter_all_arr_list.append(iter_all_arr2)
        
        gs_inv_info_path_list = []
        for iter_all_arr_short in iter_all_arr_list:
            args = zip(iter_all_arr_short, 
                       [inversion_pre_req] * len(iter_all_arr_short),
                       [num_procs] * len(iter_all_arr_short),
                       [syn_ind] * len(iter_all_arr_short),
                       [vel_synth_init] * len(iter_all_arr_short),
                       [layer_thickness_syn] * len(iter_all_arr_short),
                       [app_vel_to_inv] * len(iter_all_arr_short),
                       [rf_to_inv]* len(iter_all_arr_short),
                       [slowness_stack] * len(iter_all_arr_short), 
                       [ndivide_list]* len(iter_all_arr_short),
                       [grid_search_folder]* len(iter_all_arr_short), 
                       [vel_param_list] * len(iter_all_arr_short))
            
            
            pool = mp.Pool(num_procs)
            # result = map(par_gs, args)
            result = pool.map(_parallel_gs, args)
            for el in result:
                gs_inv_info_path_list.append(el)
            pool.close()
        
        return(gs_inv_info_path_list)
    
def _parallel_gs(args):
        (iter_all, inversion_pre_req,
               num_procs, syn_ind,
               vel_synth_init, layer_thickness_syn,
               app_vel_to_inv, rf_to_inv,
               slowness_stack, ndivide_list,
               grid_search_folder, 
               vel_param_list) = args
        
        (inv_bf, inv_af, 
         dt, nsamp, 
         filt_list, app_weight, 
         rf_weight, smooth_fac, 
         damp_fac, gauss_par) = inversion_pre_req
        cur_run_fl_name = ('Run_model_no_' + "{:03d}".format(iter_all))
        folder_name = os.path.join(grid_search_folder, cur_run_fl_name)
        folder_name = folder_name 
    
        vel_param_run = vel_param_list[iter_all]
        vel_s_perturbed = vel_param_list[iter_all].vel_s.copy()
        layers_thickness = vel_param_list[iter_all].layer_thickness.copy()
        layer_thickness_abs = vel_param_list[iter_all].layer_thickness_abs.copy()
        
        inv_info= mia.inv_all(app_vel_obs_master =app_vel_to_inv, 
                                  vel_param= vel_param_run, 
                                  rf_obs_master= rf_to_inv, 
                                  slowness_input= slowness_stack,
                                  ndivide_list= ndivide_list, 
                                  stacked = True,
                                  use_app_vel=False, 
                                  name_pso_inp = False, synthetic = syn_ind,  
                                  vel_synth_init= vel_synth_init,
                                  use_pso = False, 
                                  inv_bf = inv_bf, 
                                  inv_af = inv_af, 
                                  rf_method = 'iterative',
                                  rf_method_sens = 'waterlevel',
                                  dt = dt,
                                  layers_thickness_syn = layer_thickness_syn,
                                  nsamp= nsamp,
                                  filt_list = filt_list,
                                  app_weight = app_weight,
                                  rf_weight = rf_weight,
                                  smooth_fac = smooth_fac,
                                  damp_fac = damp_fac,
                                  gauss_par = gauss_par,
                                  close_fig = True, 
                                  rf_normalize = 1,
                                  save_dir_root= folder_name)
        
        fl_inv_info = folder_name+'/inv_info'+str(iter_all)+'.bin'
        file1 = open(fl_inv_info, "wb") 
        pickle.dump(inv_info, file1)
        file1.close()
        
        del inv_info
        return(fl_inv_info)
#%%
class Station_sor_4_pkg:
    def __init__(self, network_name, stname, 
                 st_folder, layers_thickness,
                 filt_list, network_coordinate, 
                 vel_init = False, 
                 gauss_filt = 2.5,
                 bf_p = 15,
                 af_p = 60,
                 ndivide_list = [-1, 1, -2, 2, -3, 3, -4, 4],
                 to_sample = 10,
                 dist_range = (25,95),
                 save_data = False, 
                 freq_min_bpfilt = 0.8,
                 freq_max_bpfilt = 'default',
                 save_rf=False, 
                 force_load_data = False, 
                 force_load_rf = False, 
                 max_app_vel = 3.0,
                 min_app_vel = 1.4,
                 min_app_vel_last = False, 
                 increase_cond = False, 
                 close_fig = True,
                 review_station= False,
                 save_folder = '/home/soroush/rf_shallow_codes/my_py_rf/real_data_outputs/'):
        # in this part i assign some input to the class
        self.info_for_save = {}
        self.review_station = review_station
        self.ndivide_list = ndivide_list
        self.min_app_vel = min_app_vel
        self.max_app_vel = max_app_vel
        self.min_app_vel_last = min_app_vel_last
        self.increase_cond = increase_cond
        self.filt_kind = 'cosine'
        self.freq_max_bpfilt = freq_max_bpfilt
        self.freq_min_bpfilt = freq_min_bpfilt
        self.network_coordinate = network_coordinate
        self.to_sample = to_sample
        self.to_dt = 1 / self.to_sample
        self.vel_init = vel_init
        self.rf_normalize = 1
        self.stname = stname 
        self.network_name = network_name 
        self.filt_list  = filt_list
        self.st_folder = st_folder
        self.rf_method = 'iterative' 
        self.rf_method_sens = 'waterlevel'
        self.tshift = bf_p 
        self.waterlevel = 0.01 
        self.gauss_filt = gauss_filt 
        self.bf_p = bf_p 
        self.af_p = af_p 
        self.dist_range= (25,95)
        self.save_folder = save_folder
        self.close_fig = close_fig 
        self.harmonic_cal = True
        self.nbin_normal = 24
        self.visual_inspected_file = 'visual_inspected_st.dat'
        self.harmonic_af_p = 20
        self.model_tp = taup.TauPyModel(model="iasp91")
        self.visual_insp = []
        if (self.review_station):
            print('Reviewing station is on, All RFRs and apparant_velocities' + 
                  ' will be plotted in '+
                  os.path.join(self.save_folder, 'review_station_rfs') + 
                  ' and '+ 
                  os.path.join(self.save_folder, 'review_station_apps')
                  + ' respectively. ' +
                  'The runtime will be increased accordingly.')
            print('==========================================================')
            print('==========================================================')
        if (os.path.isdir(self.save_folder)):
            if (os.path.isdir(self.save_folder+self.network_name+'_'+
                self.stname+'/')):
                pass 
            else:
                os.mkdir(self.save_folder+self.network_name+'_'+
                    self.stname+'/')
            
        else:
            os.mkdir(self.save_folder)
            os.mkdir(self.save_folder+self.network_name+'_'+
                self.stname+'/')
        self.save_folder = self.save_folder+self.network_name+'_'+\
            self.stname+'/'
        # here we are going to read data in the event folder in station folder
        if (force_load_data):
            self.load_from_file(what_to_load = 'all_data', file_to_load='')
        else:
            self.read_files()
            if (save_data):
                self.save_to_file(what_to_save = 'all_data', file_to_save= '')
        # cutting data according to b_p and af_p and assigning some info to trace
        
        # calculating RF
        if (force_load_rf):
            self.load_from_file(what_to_load='all_rf', file_to_load='')
        else:
            self.cut_trace_for_p()
            self.cal_rf()  
            if (save_rf):
                self.save_to_file(what_to_save = 'all_rf', file_to_save= '')
        #rfs calculated for this station, from now on we are going to cal 
        #app curve
        
        self.find_app_vel()
        if (self.harmonic_cal == True):
            self.binn_normal(bf_eval = True, kind= 'back_azimuth')
            self.find_harmonics(bf_eval= True, joint_invert=True)
            self.plot_rf_bf_eval(bf_eval = True)
            self.plot_app_vel_bf_eval(bf_eval = True)
        self.evaluate_app_vel()
        self.rf_stack() 
        self.assign_cor_coef()
        self.evaluate_cor_coef()
        self.find_app_vel_stack()
        if (self.harmonic_cal == True):
            self.binn_normal(bf_eval = False, kind= 'back_azimuth')
            self.find_harmonics(bf_eval= False, joint_invert=True)
            self.plot_rf_bf_eval(bf_eval = False)
            self.plot_app_vel_bf_eval(bf_eval = False)
            self.plot_rf_app_vel_final()
        self.binn_rf(nbin = 4, kind= 'back_azimuth')
        self.find_app_vel_stack_arr()
        if (force_load_data == False and force_load_rf == False):
            self.write_info()
    def write_info(self):
        info = []
        dum = 'Network name: '+self.network_name
        info.append(dum)
        dum = 'Station name: '+self.stname
        info.append(dum)
        dum = 'Number of event folders: '+str(self.info_for_save['N_folder'])
        info.append(dum)
        dum = ('Number of imported events: '+
               str(self.info_for_save['N_imported_events']))
        info.append(dum)
        dum = ('Number of preprocessed events: '+
               str(self.info_for_save['N_imported_events']))
        info.append(dum)
        dum = ('Number of calculated RFs: '+
               str(self.info_for_save['N_RFS_bf_app_vel_criteria']))
        info.append(dum)
        dum = ('Number of RFs pass the apparant '+
               'velocity boundary criteria: '+ 
               str(self.info_for_save['N_rf_after_app_vel_bf_std']))
        info.append(dum)
        dum = ('Number of RFs pass the apparant '+
               'velocity standard deviation criteria: '+ 
               str(self.info_for_save['N_rf_after_app_vel_af_std']))
        info.append(dum)
        dum = ('List of event folders that cant be imported')
        info.append(dum)
        for el in self.info_for_save['bad_events_for_read']:
            info.append(el)
        
        
        
        fl_name_ew = 'ST_'+self.stname+'_info.dat'
        fl_name = os.path.join(self.save_folder, fl_name_ew)
        file1 = open(fl_name, "w")
        for el in info:
            file1.write(el + '\n')
        file1.close()
        print('An info file saved in: '+fl_name)
    def create_custom_rf(self, custom_strm, 
                         bf_p = 1, af_p= 20, 
                         max_filt_list = 20):
        self.custom_stack_ind = True
        
        app_vel = []
        for tr in custom_strm:
            if (tr.stats.channel =='RFR'):
                rf_r = tr.data.copy()
                slowness = tr.ray_p_s_to_km
            elif (tr.stats.channel == 'RFZ'):
                rf_z = tr.data.copy()
        filt_list_dum = self.filt_list.copy()
        filt_list_dum = filt_list_dum - max_filt_list
        ind_to_pick = np.argwhere(abs(filt_list_dum) == 
                                  np.min(abs(filt_list_dum)))[0][0]
        filt_ok = False 
        while (filt_ok == False):
            if ((self.filt_list[ind_to_pick] - max_filt_list) > 0.0):
                ind_to_pick = ind_to_pick - 1
            else:
                filt_ok = True 
        self.custom_filt_list = self.filt_list[:ind_to_pick+1]
        custom_filter_array = self.filter_array[:ind_to_pick+1]
        self.custom_filter_array = []
        for filt in custom_filter_array:
            if (len(filt) > len(rf_r.data)):
                filt_cuted = filt[:len(rf_r.data)]
                self.custom_filter_array.append(filt_cuted)
        self.custom_bf_p = bf_p 
        self.custom_af_p = af_p
        
        idum = -1
        for filt in self.custom_filter_array:
            idum += 1
            s1 = np.matmul(rf_r, filt)
            s2 = np.matmul(rf_z, filt)
            amp2 = s1/s2
            theta_rad2 = np.arctan(amp2)
            v_s2 = np.sin(theta_rad2 / 2.0) / slowness
            app_vel.append(v_s2)    
        self.custom_stack = custom_strm.copy()
        for tr in self.custom_stack:
            tr.app_vel = app_vel.copy()
            tr.custom_filt_list = self.custom_filt_list
        self.stacked_dict['custom stack'] = self.custom_stack.copy()

    def find_app_vel_stack_arr(self):
        for l_stack in self.l_stack_array:
            app_vel = [] 
            app_vel = self.find_app_vel_func(l_stack)
        
            for tr in l_stack:
                tr.app_vel = app_vel.copy()  
            
            app_vel_all = []
            rf_all = []
            for ind in l_stack[0].inds:
                app_vel_all.append(self.all_rf_station[ind][1].app_vel)
                rf_all.append(self.all_rf_station[ind][1].data)
            
            
            
            app_info = []
            rf_info = []
            for i in range(len(app_vel_all[0])):
                app_samp = []
                for el in app_vel_all:
                    app_samp.append(el[i])
                app_info.append([np.mean(app_samp), np.std(app_samp), 
                                 np.median(app_samp)])
            app_info = np.array(app_info)
            
            for i in range(len(rf_all[0])):
                rf_samp = []
                for el in rf_all:
                    rf_samp.append(el[i])
                rf_info.append([np.mean(rf_samp), np.std(rf_samp), 
                                 np.median(rf_samp)])
            rf_info = np.array(rf_info)
            
            
            
            
            for tr in l_stack:
                tr.app_vel_mean = app_info[:,0]
                tr.app_vel_std = app_info[:,1]
                tr.app_vel_median = app_info[:,2]
                
                tr.rf_mean = rf_info[:,0]
                tr.rf_std = rf_info[:,1]
                tr.rf_median = rf_info[:,2]

            
                
            for pw_stack in self.pw_stack_array:
                app_vel = [] 
                app_vel = self.find_app_vel_func(pw_stack)
            
                for tr in pw_stack:
                    tr.app_vel = app_vel.copy()
                    tr.app_vel_mean = app_info[:,0]
                    tr.app_vel_std = app_info[:,1]
                    tr.app_vel_median = app_info[:,2]
                    
                    tr.rf_mean = rf_info[:,0]
                    tr.rf_std = rf_info[:,1]
                    tr.rf_median = rf_info[:,2]
    def evaluate_app_vel(self):
        #in this routine i moved rfs with negative app vel to 
        #trash.
        rf_fixed = []
        rf_trash = []
        for strm_rf in self.all_rf_station:
            if (self.increase_cond == False):
                if (self.min_app_vel_last == False):
                    if ((min(strm_rf[1].app_vel) < self.min_app_vel) 
                        or (strm_rf[1].app_vel[-1] > self.max_app_vel)):
                        rf_trash.append(strm_rf)
                    else:
                        rf_fixed.append(strm_rf)
                else:
                    if ((min(strm_rf[1].app_vel) < self.min_app_vel) 
                        or (strm_rf[1].app_vel[-1] > self.max_app_vel) or 
                        (strm_rf[1].app_vel[-1] < self.min_app_vel_last)):
                        rf_trash.append(strm_rf)
                    else:
                        rf_fixed.append(strm_rf)
            else:
                if (self.min_app_vel_last == False):
                    if ((min(strm_rf[1].app_vel) < self.min_app_vel) 
                        or (strm_rf[1].app_vel[-1] > self.max_app_vel) or 
                        (strm_rf[1].app_vel[-1] < strm_rf[1].app_vel[-10])):
                        rf_trash.append(strm_rf)
                    else:
                        rf_fixed.append(strm_rf)
                else:
                    if ((min(strm_rf[1].app_vel) < self.min_app_vel) 
                        or (strm_rf[1].app_vel[-1] > self.max_app_vel) or 
                        (strm_rf[1].app_vel[-1] < self.min_app_vel_last) or 
                        (strm_rf[1].app_vel[-1] < strm_rf[1].app_vel[-10])):
                        rf_trash.append(strm_rf)
                    else:
                        rf_fixed.append(strm_rf)
                
        self.all_rf_station = [] 
        self.all_rf_station = rf_fixed.copy() 
        self.rf_trash = rf_trash.copy()
        
        if (self.review_station):
            print('From '+ str(len(self.all_rf_station) + len(self.rf_trash)) +
                  ' calculated RFs, ' + str(len(self.all_rf_station)) + 
                  ' RFs passed the app_vel boundary criteria')
            
            
            number_of_passed_criteria = len(self.all_rf_station)
            folder_review = os.path.join(self.save_folder, 'review_station_apps')
            folder_review_wasnt_fix = os.path.join(folder_review, 'wasnt_good')
            folder_review_was_fix = os.path.join(folder_review, 'was_good')
            folders_to_work = [folder_review, folder_review_wasnt_fix,
                               folder_review_was_fix]
            for folder in folders_to_work:    
                if (os.path.isdir(folder)):
                    pass
                else:
                    os.mkdir(folder)
            
            ## for fixed
            for strm_good in self.all_rf_station:
                rfr = strm_good[1]
                t_vec = rfr.time_vec
                app_vel = rfr.app_vel
                fig, ax = plt.subplots(figsize = (24, 10), 
                                       nrows = 2, ncols= 1)
                ax[0].plot(t_vec, rfr.data)
                ax_0_xtick = np.linspace(-self.bf_p, self.af_p, 
                                      int(abs(self.bf_p)+abs(self.af_p) + 1))
                ax[0].set_xticks(ax_0_xtick)
                ax[0].grid(True)
                ax_1_xtick = np.linspace(0.0, round(np.max(self.filt_list)), 
                                      int(round(np.max(self.filt_list) + 1)))
                ax[1].set_xticks(ax_1_xtick)
                ax_1_ytick = np.linspace(0.0, self.max_app_vel + 1, 
                                      (round(self.max_app_vel + 1) * 2)+1)
                ax[1].set_yticks(ax_1_ytick)
                ax[1].plot(self.filt_list, app_vel)
                ax[1].hlines(self.min_app_vel, 
                             0.0, np.max(self.filt_list),
                             colors='red', lw = 3.0)
                ax[1].hlines(self.max_app_vel, 
                             np.max(self.filt_list) - 1, 
                             np.max(self.filt_list),
                             colors='red', lw = 3.0)
                ax[1].hlines(self.max_app_vel, 
                             self.filt_list[-6], 
                             np.max(self.filt_list),
                             colors='red', lw = 3.0)
                
                ax[1].fill_between(self.filt_list, y1 = self.min_app_vel, 
                                   y2 = np.min(app_vel),
                                   color = 'red', alpha = 0.4)
                if (np.max(app_vel) > np.max(ax_1_ytick)):
                    y2_fill = np.max(app_vel)
                else:
                    y2_fill =np.max(ax_1_ytick)
                ax[1].fill_between(self.filt_list[-6:], y1 = self.max_app_vel,
                                   y2 = y2_fill,
                                   color = 'red', alpha = 0.4)
                
                ax[1].grid(True)
                plot_name = os.path.join(folder_review_was_fix, 
                                         rfr.stats.folder_name[-13:-1])
                title_of_fig = ('trace : ' + rfr.stats.file_name + ' Baz : '+
                                str(rfr.stats.back_azimuth) + ' distance : '+
                                str(rfr.stats.distance))
                fig.suptitle(title_of_fig)
                fig.tight_layout()
                fig.savefig(plot_name + '.png', dpi = 100)
                plt.clf()
                plt.cla()
                plt.close('all')
                del fig 
                 
            ## for not fixed
            for strm_bad in self.rf_trash:
                rfr = strm_bad[1]
                t_vec = rfr.time_vec
                app_vel = rfr.app_vel
                fig, ax = plt.subplots(figsize = (24, 10), 
                                       nrows = 2, ncols= 1)
                ax[0].plot(t_vec, rfr.data)
                ax_0_xtick = np.linspace(-self.bf_p, self.af_p, 
                                      int(abs(self.bf_p)+abs(self.af_p) + 1))
                ax[0].set_xticks(ax_0_xtick)
                ax[0].grid(True)
                ax_1_xtick = np.linspace(0.0, round(np.max(self.filt_list)), 
                                      int(round(np.max(self.filt_list) + 1)))
                ax[1].set_xticks(ax_1_xtick)
                ax[1].set_xticks(ax_1_xtick)
                ax_1_ytick = np.linspace(0.0, self.max_app_vel + 1, 
                                      (round(self.max_app_vel + 1) * 2)+1)
                ax[1].set_yticks(ax_1_ytick)
                ax[1].plot(self.filt_list, app_vel)
                ax[1].hlines(self.min_app_vel, 
                             0.0, np.max(self.filt_list),
                             colors='red', lw = 3.0)
                ax[1].hlines(self.max_app_vel, 
                             self.filt_list[-6], 
                             np.max(self.filt_list),
                             colors='red', lw = 3.0)
                ax[1].fill_between(self.filt_list, y1 = self.min_app_vel, 
                                   y2 = np.min(app_vel),
                                   color = 'red', alpha = 0.4)
                if (np.max(app_vel) > np.max(ax_1_ytick)):
                    y2_fill = np.max(app_vel)
                else:
                    y2_fill =np.max(ax_1_ytick)
                ax[1].fill_between(self.filt_list[-6:], y1 = self.max_app_vel,
                                   y2 = y2_fill,
                                   color = 'red', alpha = 0.4)
                ax[1].grid(True)
                plot_name = os.path.join(folder_review_wasnt_fix, 
                                         rfr.stats.folder_name[-13:-1])
                title_of_fig = ('trace : ' + rfr.stats.file_name + ' Baz : '+
                                str(rfr.stats.back_azimuth) + ' distance : '+
                                str(rfr.stats.distance))
                fig.suptitle(title_of_fig)
                fig.tight_layout()
                fig.savefig(plot_name + '.png', dpi = 100)
                plt.clf()
                plt.cla()
                plt.close('all')
                del fig 
                
                    
                
        self.info_for_save['N_rf_after_app_vel_bf_std'] = len(self.all_rf_station)
        mean_vel = np.zeros(shape=(len(self.filt_list),))
        std_vel = np.zeros(shape=(len(self.filt_list),))
        median_vel = np.zeros(shape=(len(self.filt_list),))
        idum =-1
        for filt in self.filt_list:
            idum+=1
            vel_array = []
            for strm_rf in self.all_rf_station:
                vel_array.append(strm_rf[1].app_vel[idum])
            mean_vel[idum] = np.mean(vel_array)
            std_vel[idum] = np.std(vel_array)
            median_vel[idum] = np.median(vel_array)
        self.mean_vel = mean_vel 
        self.std_vel = std_vel 
        self.median_vel = median_vel
        rf_fixed = [] 
        for strm_rf in self.all_rf_station:
            good_strm = True
            app_vel = np.zeros(shape=(len(self.filt_list)))
            app_vel = strm_rf[1].app_vel.copy()
            for filt_counter in np.arange(len(self.filt_list)):
                upper = mean_vel[filt_counter] + 1.5*std_vel[filt_counter]
                lower = mean_vel[filt_counter] - 1.5*std_vel[filt_counter]
                if ((app_vel[filt_counter] > upper) or 
                        (app_vel[filt_counter] < lower)):
                    good_strm = False 
            if (good_strm):
                rf_fixed.append(strm_rf)
            else:
                self.rf_trash.append(strm_rf)
        if (self.review_station):
            print('From '+str(number_of_passed_criteria) + ', '+
                  str(len(rf_fixed)) +
                  ' are in the 1.5* standard deviation of apparant velocities')
        self.all_rf_station = [] 
        self.all_rf_station = rf_fixed.copy()
        self.info_for_save['N_rf_after_app_vel_af_std'] = len(self.all_rf_station)
    def find_app_vel(self):
        self.retrive_filters()
        for strm in self.all_rf_station:
            # print('finding apparant velocity for: \n'+strm[1].stats.folder_name)
            dummy_strm = op.Stream()
            for tr in strm:
                if (tr.stats.channel =='RFR'):
                    dummy_strm.append(tr.copy())
                elif (tr.stats.channel == 'RFZ'):
                    dummy_strm.append(tr.copy())
            app_vel = []
            app_vel = self.find_app_vel_func(dummy_strm)
            for tr in strm:
                tr.app_vel = app_vel 

    def find_app_vel_func(self, dummy_strm):
        app_vel = []
        for tr in dummy_strm:
            if (tr.stats.channel =='RFR'):
                rf_r = tr.data.copy()
                slowness = tr.ray_p_s_to_km
            elif (tr.stats.channel == 'RFZ'):
                rf_z = tr.data.copy()
        idum = -1
        for filt in self.filter_array:
            idum += 1
            # rf_r_filtered = self.filter_rf(rf_r, filt)
            # rf_z_filtered = self.filter_rf(rf_z, filt)
            # amp = (rf_r_filtered[self.onset_ind] /
            #        rf_z_filtered[self.onset_ind])
            
            
            # theta_rad = np.arctan(amp)
            # v_s = np.sin(theta_rad / 2.0) / slowness
            # app_vel.append(v_s)
            s1 = np.matmul(rf_r, filt)
            s2 = np.matmul(rf_z, filt)
            amp2 = s1/s2
            theta_rad2 = np.arctan(amp2)
            v_s2 = np.sin(theta_rad2 / 2.0) / slowness
            
            # print(v_s, v_s2, abs(v_s) - abs(v_s2), idum)
            app_vel.append(v_s2)
            
            
        return(app_vel)

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

    def assign_cor_coef(self):
        linear_stacked = self.l_stack[0].data
        pw_stacked = self.pw_stack[0].data 
        for strm in self.all_rf_station:
            for tr in strm:
                if (tr.stats.channel == 'RFR'):     
                    corr_coef_li = np.corrcoef(tr.data, linear_stacked)[0][1]
                    corr_coef_pw = np.corrcoef(tr.data, pw_stacked)[0][1]
            for tr in strm:
                tr.stats.corr_coef_li = corr_coef_li
                tr.stats.corr_coef_pw = corr_coef_pw
               
    def evaluate_cor_coef(self, ref= 'linear'):
        all_cor_coef_li = []
        all_cor_coef_pw = []
        for strm in self.all_rf_station:
            for tr in strm:
                if (tr.stats.channel == 'RFR'):
                    all_cor_coef_li.append(tr.stats.corr_coef_li)
                    all_cor_coef_pw.append(tr.stats.corr_coef_pw)
        
        mean_cor_coef_li = np.mean(all_cor_coef_li)
        mean_cor_coef_pw = np.mean(all_cor_coef_pw)
        
        std_cor_coef_li = np.std(all_cor_coef_li)
        std_cor_coef_pw = np.std(all_cor_coef_pw)
        
        if (ref == 'linear'):
            mean_cor = mean_cor_coef_li
            std_cor = std_cor_coef_li
            flag = ref
        else: 
            mean_cor = mean_cor_coef_pw
            std_cor = std_cor_coef_pw
            flag = ref
        for strm in self.all_rf_station:
            for tr in strm:
                if (tr.stats.channel == 'RFR'):
                    if (tr.stats.corr_coef_li >= (mean_cor
                                        + std_cor)):
                        tr.stats.corr_grade = 'A_'+flag
                    elif (tr.stats.corr_coef_li >= mean_cor):
                        tr.stats.corr_grade = 'B_'+flag
                    else:
                        tr.stats.corr_grade = 'C_'+flag
               

    def find_weighted_stack_from_binned(self, binned_stack_array):

        nrf_in_binned = []
        ray_p_in_binned = []
        baz_in_binned = []
        all_rf_r_binned = []
        all_rf_t_binned = []
        for l_stack in binned_stack_array:
            for tr in l_stack:
                if ("RFR" in tr.stats.channel):
                    all_rf_r_binned.append(tr.data)
                    nrf_in_binned.append(len(tr.baz_all))
                    ray_p_in_binned.append(tr.ray_p_s_to_km)
                    baz_in_binned.append(tr.stats.back_azimuth)
                elif ("RFT" in tr.stats.channel):
                    all_rf_t_binned.append(tr.data)
        mean_ray_p_binned = np.mean(ray_p_in_binned)
        mean_baz_in_binned = np.mean(baz_in_binned)
        all_rf_r_binned = np.array(all_rf_r_binned)
        all_rf_t_binned = np.array(all_rf_t_binned)
        
        
        mean_rf_r_sample = []
        mean_rf_r_sample_weighted = []
        mean_rf_t_sample = []
        mean_rf_t_sample_weighted = []
        for i in range(len(all_rf_r_binned[0, :])):
            mean_rf_r_sample.append(np.mean(all_rf_r_binned[:, i]))
            mean_rf_t_sample.append(np.mean(all_rf_t_binned[:, i]))
            dummy_weight_r = []
            dummy_weight_t = []
            for j in range(len(all_rf_r_binned[:, 0])):
                dummy_weight_r.append(all_rf_r_binned[j, i] *
                                      nrf_in_binned[j])
                dummy_weight_t.append(all_rf_t_binned[j, i] *
                                      nrf_in_binned[j])
            w_mean_r = np.sum(dummy_weight_r) / np.sum(nrf_in_binned)
            w_mean_t = np.sum(dummy_weight_t) / np.sum(nrf_in_binned)
            mean_rf_r_sample_weighted.append(w_mean_r)
            mean_rf_t_sample_weighted.append(w_mean_t)
        
        dummy_weight_ray_p = []
        dummy_weight_baz = []
        for i in range(len(all_rf_r_binned[:, 0])):
            dummy_weight_ray_p.append(ray_p_in_binned[i] * 
                                      nrf_in_binned[i])
            dummy_weight_baz.append(baz_in_binned[i] * 
                                      nrf_in_binned[i])
        mean_ray_p_binned_weighted = (np.sum(dummy_weight_ray_p)
                                      / np.sum(nrf_in_binned))
        mean_baz_in_binned_weighted = (np.sum(dummy_weight_baz)
                                      / np.sum(nrf_in_binned))
        
        ## for linear
        l_stack_all_rf_bf_eval_linear = binned_stack_array[0].copy()
        for tr in l_stack_all_rf_bf_eval_linear:
            if ("RFR" in tr.stats.channel):
                tr.data = np.array(mean_rf_r_sample)
                tr.baz_all = baz_in_binned
                tr.ray_p_l = ray_p_in_binned
                tr.ray_p_s_to_km = mean_ray_p_binned
                tr.stats.back_azimuth = mean_baz_in_binned
            elif ("RFT" in tr.stats.channel):
                tr.data = np.array(mean_rf_t_sample)
                tr.baz_all = baz_in_binned
                tr.ray_p_l = ray_p_in_binned
                tr.ray_p_s_to_km = mean_ray_p_binned
                tr.stats.back_azimuth = mean_baz_in_binned
            else:
                tr.baz_all = baz_in_binned
                tr.ray_p_l = ray_p_in_binned
                tr.ray_p_s_to_km = mean_ray_p_binned_weighted
                tr.stats.back_azimuth = mean_baz_in_binned_weighted
        app_vel = [] 
        app_vel = self.find_app_vel_func(l_stack_all_rf_bf_eval_linear)
        for tr in l_stack_all_rf_bf_eval_linear:
            tr.app_vel = app_vel
        
        
        ## for weighted 
        l_stack_all_rf_bf_eval_weighted = binned_stack_array[0].copy()
        for tr in l_stack_all_rf_bf_eval_weighted:
            if ("RFR" in tr.stats.channel):
                tr.data = np.array(mean_rf_r_sample_weighted)
                tr.baz_all = baz_in_binned
                tr.ray_p_l = ray_p_in_binned
                tr.ray_p_s_to_km = mean_ray_p_binned_weighted
                tr.stats.back_azimuth = mean_baz_in_binned_weighted
            elif ("RFT" in tr.stats.channel):
                tr.data = np.array(mean_rf_t_sample_weighted)
                tr.baz_all = baz_in_binned
                tr.ray_p_l = ray_p_in_binned
                tr.ray_p_s_to_km = mean_ray_p_binned_weighted
                tr.stats.back_azimuth = mean_baz_in_binned_weighted
            else:
                tr.baz_all = baz_in_binned
                tr.ray_p_l = ray_p_in_binned
                tr.ray_p_s_to_km = mean_ray_p_binned_weighted
                tr.stats.back_azimuth = mean_baz_in_binned_weighted
        app_vel = [] 
        app_vel = self.find_app_vel_func(l_stack_all_rf_bf_eval_weighted)
        for tr in l_stack_all_rf_bf_eval_weighted:
            tr.app_vel = app_vel
        return (l_stack_all_rf_bf_eval_linear, l_stack_all_rf_bf_eval_weighted)        
    def binn_normal(self, bf_eval, kind= 'back_azimuth'):
        if (bf_eval == True):
            print("Start to stack in back azimuth bins before evaluating"+
                  " app vel, no of bins is: "
                  +str(self.nbin_normal))
            self.nrf_bf_eval = len(self.all_rf_station)
        else:
            print("Start to stack in back azimuth bins after evaluating"+
                  " app vel, no of bins is: "
                  +str(self.nbin_normal))
            self.nrf_af_eval = len(self.all_rf_station)
        n_divide = self.nbin_normal + 1
        r_comp_all = op.Stream()
        z_comp_all = op.Stream()
        t_comp_all = op.Stream()
        baz_all = []
        ray_p_l = []
        for strm_rf in self.all_rf_station:
            for tr in strm_rf:
                if ('RFR' in tr.stats.channel):
                    r_comp_all.append(tr)
                    ray_p_l.append(tr.ray_p_s_to_km)
                    baz_all.append(tr.stats.back_azimuth)
                elif ('RFZ' in tr.stats.channel):
                    z_comp_all.append(tr)
                else:
                    t_comp_all.append(tr)
            if (kind == 'back_azimuth'):
                min_par = 0.0
                max_par = 360.0
                par = baz_all.copy()
            else:
                min_par = np.min(ray_p_l)
                max_par = np.max(ray_p_l)
                par = ray_p_l.copy()
        #start bining
        n_divide = self.nbin_normal + 1
        bin_corner = np.linspace(min_par, max_par, n_divide)
        bin_center = []
        for i in range(len(bin_corner) - 1):
            center = bin_corner[i] +\
                ((bin_corner[i+1] - bin_corner[i]) / 2)
            bin_center.append(center)
        bin_center = np.array(bin_center)
        # print(bin_center)
        bins_all = []
        inds_all = []
        lens_all = []
        bin_theory_center = []
        for i in range(len(bin_corner) - 1):
            inds = []
            bins = []
            for j in range(len(par)):
                if ((par[j] >= bin_corner[i]) and 
                    (par[j] < bin_corner[i+1])):
                    bins.append(par[j])
                    inds.append(j)
            if (len(inds) > 0): 
                bin_theory_center.append(bin_center[i])
                bins_all.append(bins)
                inds_all.append(inds)
        stacked_bins = []
        l_stack_array_bf_eval = []
        pw_stack_array_bf_eval = []
        
        idum = -1
        for el in inds_all:
           idum += 1
           l_stack, pw_stack = self.rf_stack_bins(el, kind)
           l_stack[0].bin_theory_center = bin_theory_center[idum]
           l_stack[1].bin_theory_center = bin_theory_center[idum]
           l_stack[2].bin_theory_center = bin_theory_center[idum]
           l_stack_array_bf_eval.append(l_stack)
           pw_stack_array_bf_eval.append(pw_stack)
        
        for l_stack in l_stack_array_bf_eval:
            app_vel = [] 
            app_vel = self.find_app_vel_func(l_stack)
        
            for tr in l_stack:
                tr.app_vel = app_vel.copy()  
            
        if (bf_eval == True):
            inds_all_rf_bf_eval = np.arange(len(self.all_rf_station))
            
            self.l_stack_all_rf_bf_eval, self.pw_stack_all_rf_bf_eval\
                = self.rf_stack_bins(inds_all_rf_bf_eval, kind)
            app_vel = []
            app_vel = self.find_app_vel_func(self.l_stack_all_rf_bf_eval)
            for tr in self.l_stack_all_rf_bf_eval:
                tr.app_vel = app_vel
            app_vel = []
            app_vel = self.find_app_vel_func(self.pw_stack_all_rf_bf_eval)
            for tr in self.pw_stack_all_rf_bf_eval:
                tr.app_vel = app_vel
            
            
            self.l_stack_all_rf_bf_eval_from_bins, \
                self.l_stack_all_rf_bf_eval_weighted = \
                self.find_weighted_stack_from_binned(binned_stack_array = 
                                            l_stack_array_bf_eval) 
          
                
            self.l_stack_array_harmonic_bf = l_stack_array_bf_eval
            self.pw_stack_array_harmonic_bf = pw_stack_array_bf_eval
        else:
            
            self.l_stack_all_rf_af_eval_from_bins, \
                self.l_stack_all_rf_af_eval_weighted = \
                self.find_weighted_stack_from_binned(binned_stack_array = 
                                            l_stack_array_bf_eval)
                
            self.l_stack_array_harmonic_af = l_stack_array_bf_eval
            self.pw_stack_array_harmonic_af = pw_stack_array_bf_eval

    def find_harmonics(self, bf_eval, joint_invert = False):
        
        if (bf_eval == True):
            stack_array = self.l_stack_array_harmonic_bf
        else:
            stack_array = self.l_stack_array_harmonic_af
            
        all_ray_p = []
        all_baz = []
        all_rf_r_bins = []
        all_rf_t_bins = []
        all_rf_phi_bins = []
        for l_stack in stack_array:
            for tr in l_stack:
                if (tr.stats.channel == "RFR"):
                    tr_r = tr.copy()
                    all_rf_r_bins.append(tr_r.data)
                    all_rf_phi_bins.append(tr_r.bin_theory_center)
                    
                    for ray_p in tr.ray_p_l:
                        all_ray_p.append(ray_p)
                    for baz in tr.baz_all:
                        all_baz.append(baz)
                    
                elif (tr.stats.channel == "RFT"):
                    tr_t = tr.copy()
                    all_rf_t_bins.append(tr_t.data)
                else:
                    tr_z = tr.copy()
        all_rf_r_bins = np.array(all_rf_r_bins)
        all_rf_t_bins = np.array(all_rf_t_bins)
        all_rf_phi_bins = np.array(all_rf_phi_bins)
        if (joint_invert == False):
            a_amp = []
            b_amp = []
            c_amp = []
            d_amp = []
            e_amp = []
            nsamp = len(all_rf_r_bins[0,:])

            for i in range(nsamp):
                x = all_rf_phi_bins
                y = all_rf_r_bins[:, i]
                popt, _ = curve_fit(objective_harmonic_r, x, y)
                
                a, b, c, d, e = popt 
                # error_val = y - objective_harmonic_r(x, a, b, c, d, e)
                # fig, ax = plt.subplots(nrows = 2, ncols= 1)
                # ax[0].plot(x, y, label = 'data')
                # ax[0].plot(x, objective_harmonic_r(x, a, b, c, d, e), 
                #            label = 'estimated')
                # ax[0].legend()
                # ax[1].scatter(x, error_val)
                
                
                a_amp.append(a)
                b_amp.append(b)
                c_amp.append(c)
                d_amp.append(c)
                e_amp.append(c)
            harmonic_r_k0_stack = a_amp.copy()
            harmonic_r_k1_cos_stack = b_amp.copy()
            harmonic_r_k1_sin_stack = c_amp.copy()
            harmonic_r_k2_cos_stack = d_amp.copy()
            harmonic_r_k2_sin_stack = e_amp.copy()
            
            b_amp = []
            c_amp = []
            d_amp = []
            e_amp = []
            nsamp = len(all_rf_r_bins[0,:])

            for i in range(nsamp):
                x = all_rf_phi_bins
                y = all_rf_t_bins[:, i]
                popt, _ = curve_fit(objective_harmonic_t, x, y)
                b, c, d, e = popt 
                b_amp.append(b)
                c_amp.append(c)
                d_amp.append(c)
                e_amp.append(c)

            harmonic_t_k1_cos_stack = b_amp.copy()
            harmonic_t_k1_sin_stack = c_amp.copy()
            harmonic_t_k2_cos_stack = d_amp.copy()
            harmonic_t_k2_sin_stack = e_amp.copy()
            
            b_fin = []
            c_fin = []
            d_fin = []
            e_fin = []
            for i in range(len(harmonic_r_k1_cos_stack)):
                b_fin.append((harmonic_r_k1_cos_stack[i] +  
                                     harmonic_t_k1_cos_stack[i]) / 2)
                c_fin.append((harmonic_r_k1_sin_stack[i]+ 
                                     harmonic_t_k1_sin_stack[i]) / 2)
                d_fin.append((harmonic_r_k2_cos_stack[i]+ 
                                     harmonic_t_k2_cos_stack[i]) / 2)
                e_fin.append((harmonic_r_k2_sin_stack[i]+ 
                                     harmonic_t_k2_sin_stack[i]) / 2)
        elif (joint_invert == True):
            a_amp, b_fin, c_fin, \
                d_fin, e_fin = find_harmonic_joint(all_rf_r_bins,
                                all_rf_t_bins, all_rf_phi_bins)
            harmonic_r_k0_stack = a_amp.copy()
        sample_stack_strm = stack_array[0].copy()
        for tr in sample_stack_strm:
            if (tr.stats.channel == "RFR"):
                tr.data = np.array(harmonic_r_k0_stack.copy())
                tr.k1_cos = np.array(b_fin)
                tr.k1_sin = np.array(c_fin)
                tr.k2_cos = np.array(d_fin)
                tr.k2_sin = np.array(e_fin)
                
                tr.ray_p_l = all_ray_p
                tr.baz_all = all_baz 
                tr.ray_p_s_to_km = np.mean(all_ray_p)
                tr.stats.back_azimuth = np.mean(all_baz)
                
            elif (tr.stats.channel == "RFT"):
                tr.k1_cos = np.array(b_fin)
                tr.k1_sin = np.array(c_fin)
                tr.k2_cos = np.array(d_fin)
                tr.k2_sin = np.array(e_fin)
                
                tr.ray_p_l = all_ray_p
                tr.baz_all = all_baz 
                tr.ray_p_s_to_km = np.mean(all_ray_p)
                tr.stats.back_azimuth = np.mean(all_baz)
            else:
                tr.ray_p_l = all_ray_p
                tr.baz_all = all_baz 
                tr.ray_p_s_to_km = np.mean(all_ray_p)
                tr.stats.back_azimuth = np.mean(all_baz)
        app_vel = self.find_app_vel_func(sample_stack_strm)
        for tr in sample_stack_strm:
            tr.app_vel = app_vel.copy()
        if (joint_invert == True):
            for tr in sample_stack_strm:
                tr.kind_harmonic = 'joint_invert'
        else:
            for tr in sample_stack_strm:
                tr.kind_harmonic = 'rfr_invert'
                
        if (bf_eval == True):
            self.l_stack_harmonic_k0_bf = sample_stack_strm.copy()
        else:
            self.l_stack_harmonic_k0_af = sample_stack_strm.copy()
 
    def plot_rf_bf_eval(self,  bf_eval, kind= "back_azimuth"):
        if (bf_eval == True):
            stack_array = self.l_stack_array_harmonic_bf
            # stack_array = self.l_stack_all_rf_bf_eval
            stacked_harmonic = self.l_stack_harmonic_k0_bf
            
        else:
            stack_array = self.l_stack_array_harmonic_af
            # stack_array = self.l_stack_all_rf_af_eval_weighted
            stacked_harmonic = self.l_stack_harmonic_k0_af
        n_divide = self.nbin_normal + 1
        bin_corner = np.linspace(0, 360, n_divide)
        bin_center = []
        for i in range(len(bin_corner) - 1):
            center = bin_corner[i] +\
                ((bin_corner[i+1] - bin_corner[i]) / 2)
            bin_center.append(center)
        bin_center = np.array(bin_center)
        fig = plt.figure(constrained_layout=True, figsize=(16, 14))
        if (bf_eval == True):
            fig.suptitle("Azimuthal Variation of RFs before app_vel criteria"+
                     " (for all filter, app_vel>= "+str(self.min_app_vel)+")"+
                     " for Station:\n "+
                     self.stname + ", No of Rfs = "+ 
                     str(self.nrf_bf_eval), fontsize=22)
        else:
            fig.suptitle("Azimuthal Variation of RFs after app_vel criteria"+
                     " (for all filter, app_vel>= "+str(self.min_app_vel)+")"+
                     " for Station:\n "+
                     self.stname + ", No of Rfs = "+ 
                     str(self.nrf_af_eval), fontsize=22)
        subfigs = fig.subfigures(2,2, width_ratios= (1, 1),
                                 height_ratios=(3/self.nbin_normal, 1), 
                                 edgecolor = 'black', 
                                 linewidth= 2)
        ax_subfig1 = subfigs[0,0].subplots(nrows=1, sharex=True, 
                                          subplot_kw=dict(frameon=False))
        ax_subfig2 = subfigs[0,1].subplots(nrows=1, sharex=True, 
                                          subplot_kw=dict(frameon=False))
        
        ax_subfig3 = subfigs[1,0].subplots(nrows=len(bin_center), sharex=True, 
                                          subplot_kw=dict(frameon=False))
        subfigs[0,0].suptitle('RFR', fontsize=16)
        # plt.subplots_adjust(hspace=.01)
        ax_subfig4 = subfigs[1, 1].subplots(nrows=len(bin_center), sharex=True, 
                                          subplot_kw=dict(frameon=False))
        subfigs[0, 1].suptitle('RFT', fontsize=16)
        
        idum = 0
        for i in range(len(bin_center)):
            idum -= 1
            b_cent = bin_center[i]
            ax_subfig3[idum].text(-3.5, 0, '{0:5.1f}'.format(b_cent), 
                           fontsize=9)
            ax_subfig3[idum].set_yticks([])
            for l_stack in stack_array:
                if (l_stack[0].bin_theory_center == b_cent):
                    for tr in l_stack:
                        if (tr.stats.channel == "RFR"):
                            tr_r = tr.copy()
                        elif (tr.stats.channel == "RFT"):
                            tr_t = tr.copy()
                        else:
                            tr_z = tr.copy()
                    ind_af = np.argwhere(tr_r.time_vec ==
                                         self.harmonic_af_p)[0][0]
                    x0 = tr_r.time_vec[:ind_af]
                    y0 = tr_r.data[:ind_af]
                    
                    n_involve = len(tr_r.baz_all)
                    
                    x1 = tr_t.time_vec[:ind_af]
                    y1 = tr_t.data[:ind_af]
                    
                    #### for R 
                    ax_subfig3[idum].plot(x0, y0, color= 'black', lw = .5)
                    ax_subfig3[idum].fill_between(x0, 0, y0, where=y0 > 0,
                                               color = 'red')
                    ax_subfig3[idum].fill_between(x0, 0, y0, where=y0 < 0,
                                               color = 'blue')
                    ax_subfig3[idum].yaxis.set_visible(False)
                    
                    ax_subfig3[idum].text(self.harmonic_af_p + 0.5, 0,
                                          '{0:5.1f}'.format(n_involve), 
                                   fontsize=9)
                    #### for T
                    ax_subfig4[idum].plot(x1, y1, color= 'black', lw = .5)
                    ax_subfig4[idum].fill_between(x1, 0, y1, where=y1 > 0,
                                               color = 'red')
                    ax_subfig4[idum].fill_between(x1, 0, y1, where=y1 < 0,
                                               color = 'blue')
                    ax_subfig4[idum].yaxis.set_visible(False)
                    ax_subfig4[idum].text(-3.5, 0, '{0:5.1f}'.format(b_cent), 
                                   fontsize=9)
                    ax_subfig4[idum].text(self.harmonic_af_p + 0.5, 0,
                                          '{0:5.1f}'.format(n_involve), 
                                  fontsize=9)

        #plotting K0 for r
        if (bf_eval == True):
            # l_stack_harmonic_k0 = self.l_stack_harmonic_k0_bf
            l_stack_harmonic_k0 = self.l_stack_all_rf_bf_eval
            nrf = self.nrf_bf_eval
        else:
            # l_stack_harmonic_k0 = self.l_stack_harmonic_k0_af
            l_stack_harmonic_k0 = self.l_stack_all_rf_af_eval_weighted
            nrf = self.nrf_af_eval
        for tr in l_stack_harmonic_k0:
            if ('RFR' in tr.stats.channel):
                xk0_r = tr.time_vec[:ind_af]
                yk0_r = tr.data[:ind_af]
                
            elif ('RFT' in tr.stats.channel):
                xk0_t = tr.time_vec[:ind_af]
                yk0_t = tr.data[:ind_af]
                
        
        ax_subfig1.plot(xk0_r, yk0_r, color= 'black', lw = .5)
        ax_subfig1.fill_between(xk0_r, 0, yk0_r, where=yk0_r > 0,
                                   color = 'red')
        ax_subfig1.fill_between(xk0_r, 0, yk0_r, where=yk0_r < 0,
                                   color = 'blue')
        ax_subfig1.yaxis.set_visible(False)
        ax_subfig1.text(-3.5, 0, 'K0 ', 
                        fontsize=14, color = 'white')
        ax_subfig1.text(self.harmonic_af_p + 0.5, 0,
                              '{0:5.1f}'.format(nrf), 
                       fontsize=9)
        
        #plotting K0 for t
        ax_subfig2.plot(xk0_t, yk0_t, color= 'black', lw = .5)
        ax_subfig2.fill_between(xk0_t, 0, yk0_t, where=yk0_t > 0,
                                   color = 'red')
        ax_subfig2.fill_between(xk0_t, 0, yk0_t, where=yk0_t < 0,
                                   color = 'blue')
        ax_subfig2.yaxis.set_visible(False)
        ax_subfig2.text(-3.5, 0, 'K0 ', 
                       fontsize=14, color = 'white')
        ax_subfig2.text(self.harmonic_af_p + 0.5, 0,
                              '{0:5.1f}'.format(nrf), 
                       fontsize=9)
        
        if (bf_eval == True):
            name = (self.save_folder + 'binned_stack_bf_eval'+
                    '{0:3.1f}'.format(self.gauss_filt * 2 * np.pi)+'.png')
        else:
            name = (self.save_folder + 'binned_stack_af_eval'+
                    '{0:3.1f}'.format(self.gauss_filt* 2 * np.pi)+'.png')
        plt.savefig(name, dpi = 250)
        if (self.close_fig == True):
            fig.clf()
            plt.cla()
            plt.close(fig)
                
    def plot_app_vel_bf_eval(self,  bf_eval, kind= "back_azimuth"):
        #### plotting figure for apparent velocity of binned stack 
        if (bf_eval == True):
            stack_array = self.l_stack_array_harmonic_bf
            # stack_array = self.l_stack_all_rf_bf_eval
            stacked_harmonic = self.l_stack_harmonic_k0_bf
            
        else:
            stack_array = self.l_stack_array_harmonic_af
            # stack_array = self.l_stack_all_rf_af_eval_weighted
            stacked_harmonic = self.l_stack_harmonic_k0_af
                
        n_divide = self.nbin_normal + 1
        bin_corner = np.linspace(0, 360, n_divide)
        bin_center = []
        for i in range(len(bin_corner) - 1):
            center = bin_corner[i] +\
                ((bin_corner[i+1] - bin_corner[i]) / 2)
            bin_center.append(center)
        bin_center = np.array(bin_center)
        fig = plt.figure(constrained_layout=True, figsize=(16, 12))
        if (bf_eval == True):
            fig.suptitle("Azimuthal Variation of RFs before app_vel criteria"+
                     " (for all filter, app_vel>= "+str(self.min_app_vel)+")"+
                     " for Station:\n "+
                     self.stname + ", No of Rfs = "+ 
                     str(self.nrf_bf_eval), fontsize=22)
        else:
            fig.suptitle("Azimuthal Variation of RFs after app_vel criteria"+
                     " (for all filter, app_vel>= "+str(self.min_app_vel)+")"+
                     " for Station:\n "+
                     self.stname + ", No of Rfs = "+ 
                     str(self.nrf_af_eval), fontsize=22)
        subfigs = fig.subfigures(2,2, width_ratios= (1, 1),
                                 height_ratios=(5/self.nbin_normal, 1), 
                                 edgecolor = 'black', 
                                 linewidth= 2)
        ax_subfig1 = subfigs[0,0].subplots(nrows=1, sharex=True, 
                                          subplot_kw=dict(frameon=False))
        ax_subfig2 = subfigs[0,1].subplots(nrows=1)
        
        ax_subfig3 = subfigs[1,0].subplots(nrows=len(bin_center), sharex=True, 
                                          subplot_kw=dict(frameon=False))
        subfigs[0,0].suptitle('RFR', fontsize=16)

        ax_subfig4 = subfigs[1, 1].subplots(nrows=1)
        subfigs[0, 1].suptitle('Apparent Velocity', fontsize=16)
        
        viridis=cm.get_cmap('viridis', len(bin_center))
        
        
       
        idum = 0
        for i in range(len(bin_center)):
            idum -= 1
            b_cent = bin_center[i]
            ax_subfig3[idum].text(-3.5, 0, '{0:5.1f}'.format(b_cent), 
                           fontsize=9)
            ax_subfig3[idum].set_yticks([])
            for l_stack in stack_array:
                if (l_stack[0].bin_theory_center == b_cent):
                    for tr in l_stack:
                        if (tr.stats.channel == "RFR"):
                            tr_r = tr.copy()
                        elif (tr.stats.channel == "RFT"):
                            tr_t = tr.copy()
                        else:
                            tr_z = tr.copy()
                    ind_af = np.argwhere(tr_r.time_vec ==
                                         self.harmonic_af_p)[0][0]
                    x0 = tr_r.time_vec[:ind_af]
                    y0 = tr_r.data[:ind_af]
                    
                    n_involve = len(tr_r.baz_all)
                    
                    x1 = self.filt_list
                    y1 = tr_r.app_vel
                    
                    #### for R 
                    ax_subfig3[idum].plot(x0, y0, color= 'black', lw = .5)
                    ax_subfig3[idum].fill_between(x0, 0, y0, where=y0 > 0,
                                               color = 'red')
                    ax_subfig3[idum].fill_between(x0, 0, y0, where=y0 < 0,
                                               color = 'blue')
                    ax_subfig3[idum].yaxis.set_visible(False)
                    # ax_subfig3[idum].text(-3.5, 0, '{0:5.1f}'.format(b_cent), 
                    #                fontsize=9)
                    ax_subfig3[idum].text(self.harmonic_af_p + 0.5, 0,
                                          '{0:5.1f}'.format(n_involve), 
                                   fontsize=9)
                    #### for App vel
                    
                    ax_subfig4.plot(x1, y1, color=viridis(i), lw = 2.5)
                

                    ax_subfig4.set_xlabel('Filter period')
                    ax_subfig4.set_ylabel('Apparent Velocity')
        
        
        norm= matplotlib.colors.BoundaryNorm(bin_center, 
                                              len(bin_center))
        sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ticks=bin_center ,ax= ax_subfig4,
                            label='Back Azimuth')
        if (bf_eval == True):
            # l_stack_harmonic_k0 = self.l_stack_harmonic_k0_bf
            l_stack_harmonic_k0 = self.l_stack_all_rf_bf_eval
            nrf = self.nrf_bf_eval
        else:
            # l_stack_harmonic_k0 = self.l_stack_harmonic_k0_af
            l_stack_harmonic_k0 = self.l_stack_all_rf_af_eval_weighted
            nrf = self.nrf_af_eval
        for tr in l_stack_harmonic_k0:
            if ('RFR' in tr.stats.channel):
                xk0_r = tr.time_vec[:ind_af]
                yk0_r = tr.data[:ind_af]
                
            elif ('RFT' in tr.stats.channel):
                xk0_t = tr.time_vec[:ind_af]
                yk0_t = tr.data[:ind_af]
                
        
        ax_subfig1.plot(xk0_r, yk0_r, color= 'black', lw = .5)
        ax_subfig1.fill_between(xk0_r, 0, yk0_r, where=yk0_r > 0,
                                   color = 'red')
        ax_subfig1.fill_between(xk0_r, 0, yk0_r, where=yk0_r < 0,
                                   color = 'blue')
        ax_subfig1.yaxis.set_visible(False)
        ax_subfig1.text(-3.5, 0, 'K0 ', 
                        fontsize=18, color = 'white')
        ax_subfig1.text(self.harmonic_af_p + 0.5, 0,
                              '{0:5.1f}'.format(nrf), 
                       fontsize=9)
        
        #plotting apparent velocities 
        x1 = self.filt_list
        y1 = l_stack_harmonic_k0[0].app_vel
        ax_subfig2.plot(x1, y1, color="black", lw = 2.5)
        #plotting apparent velocities in bottom figure
        x1 = self.filt_list
        y1 = l_stack_harmonic_k0[0].app_vel
        ax_subfig4.plot(x1, y1, color="red", lw = 4.5)
        
        if (bf_eval == True):
            name = (self.save_folder + 'binned_stack_app_vel_bf_eval'+
                    '{0:3.1f}'.format(self.gauss_filt * 2 * np.pi)+'.png')
        else:
            name = (self.save_folder + 'binned_stack_app_vel_af_eval'+
                    '{0:3.1f}'.format(self.gauss_filt* 2 * np.pi)+'.png')
        plt.savefig(name, dpi = 250)
        if (self.close_fig == True):
            fig.clf()
            plt.cla()
            plt.close(fig)
    
    def plot_rf_app_vel_final(self, af_p = 20, 
                              af_p_filt = "default"):
        if (af_p_filt == "default"):
            x_app_vel = self.filt_list
        else:
            fl_list = np.array(self.filt_list) - np.array(af_p_filt)
            ind_app_vel = np.argwhere(abs(fl_list) == np.min(abs(fl_list)))[0][0]
            x_app_vel = self.filt_list[:ind_app_vel]
        t_vec = self.time_vec - af_p 
        ind_x_rf = np.argwhere(abs(t_vec) == np.min(abs(t_vec)))[0][0]
        x_rf = self.time_vec[:ind_x_rf]
        stacked_dict = {}
        stacked_dict['Linear Stack before app_vel criteria'] = \
            self.l_stack_all_rf_bf_eval
        stacked_dict['Linear Stack after app_vel criteria'] = \
            self.l_stack
        stacked_dict['Phase Weighted stack after app_vel criteria'] = \
            self.pw_stack
        if (self.l_stack_harmonic_k0_bf[0].kind_harmonic == 'rfr_invert'):
            stacked_dict['K0 Stack before app_vel criteria'] = \
                self.l_stack_harmonic_k0_bf
        else:
            stacked_dict['K0 Stack joint_harmonic before app_vel criteria'] = \
                self.l_stack_harmonic_k0_bf
        if (self.l_stack_harmonic_k0_af[0].kind_harmonic == 'rfr_invert'):
            stacked_dict['K0 Stack after app_vel criteria'] = \
                self.l_stack_harmonic_k0_af
        else:
            stacked_dict['K0 Stack joint_harmonic after app_vel criteria'] = \
                self.l_stack_harmonic_k0_af
        stacked_dict['Weighted Stack according to Number of trace in each bin '+
                     'before app_vel criteria'] = \
            self.l_stack_all_rf_bf_eval_weighted
        stacked_dict['Weighted Stack according to Number of trace in each bin '+
                     'after app_vel criteria'] = \
            self.l_stack_all_rf_af_eval_weighted
        stacked_dict['Linear Stack of bins before app_vel criteria'] = \
            self.l_stack_all_rf_bf_eval_from_bins
        stacked_dict['Linear Stack of bins after app_vel criteria'] = \
            self.l_stack_all_rf_af_eval_from_bins
        
        if hasattr(self, "custom_stack"):
            stacked_dict['custom stack'] = self.custom_stack
        self.stacked_dict = stacked_dict
        
        nstack = len(self.stacked_dict)
        
        fig = plt.figure(constrained_layout=True, figsize=(14, 18), 
                         dpi= 150, facecolor='#FAEBD7')
        subfigs = fig.subfigures(nstack, 1, facecolor='#FAEBD7')
        max_array_rf = []
        min_array_rf = []
        for key in self.stacked_dict.keys():
            y_rf = self.stacked_dict[key][0].data[:ind_x_rf]
            max_array_rf.append(np.max(y_rf))
            min_array_rf.append(np.min(y_rf))
        max_y_rf = np.max(max_array_rf)
        min_y_rf = np.min(min_array_rf)
        y_rf_ticks = np.round(np.linspace(min_y_rf, max_y_rf, 6), 1)
        x_rf_ticks = np.round(np.arange(np.min(x_rf), np.max(x_rf), 2), 0)
       
        idum = -1
        for key in self.stacked_dict.keys():
            idum += 1
            y_rf = self.stacked_dict[key][0].data[:ind_x_rf]
            y_app_vel = self.stacked_dict[key][0].app_vel
            ax_subfig = subfigs[idum].subplots(nrows=1, ncols=2,
                                             subplot_kw=dict(frameon=True))
            
            ax_subfig[0].xaxis.set_major_locator(MultipleLocator(5))
            ax_subfig[0].xaxis.set_major_formatter('{x:.0f}')
            ax_subfig[0].xaxis.set_minor_locator(MultipleLocator(1))
            ax_subfig[0].set_yticks(y_rf_ticks)
            ax_subfig[0].plot(x_rf, y_rf, lw= 1.5, color = 'black')
            ax_subfig[0].fill_between(x_rf, 0, y_rf, where= y_rf > 0,
                                       color = 'red')
            
            ax_subfig[0].fill_between(x_rf, 0, y_rf, where= y_rf < 0,
                                       color = 'blue')
            ax_subfig[0].grid(color = 'k', which= 'both', alpha = 0.5, 
                          axis = 'both',dashes=(5, 2, 1, 2))
            if (key == 'Linear Stack after app_vel critia'):
                subfigs[idum].suptitle(key +
                                       " (this is what i used for inversion)"
                                       , fontsize=16)
            else:
                subfigs[idum].suptitle(key, fontsize=16)
            if (key == 'custom stack'):
                x_app_vel_custom = self.stacked_dict[key][0].custom_filt_list
                ax_subfig[1].plot(x_app_vel_custom, y_app_vel, lw= 1.5, color = 'black')
                ax_subfig[1].set_xticks(np.arange(np.max(self.filt_list)))
            else:
                ax_subfig[1].plot(x_app_vel, y_app_vel, lw= 1.5, color = 'black')
            ax_subfig[1].xaxis.set_major_locator(MultipleLocator(5))
            ax_subfig[1].xaxis.set_major_formatter('{x:.0f}')
            ax_subfig[1].xaxis.set_minor_locator(MultipleLocator(1))
            ax_subfig[1].grid(color = 'k', which= 'both', alpha = 0.5, 
                          axis = 'both',dashes=(5, 2, 1, 2))
        
        name = (self.save_folder + 'comparision_of_stacks_'+self.stname+"_"+
                '{0:3.1f}'.format(self.gauss_filt * 2 * np.pi)+'.png')
        if hasattr(self, "custom_stack"):
            name = (self.save_folder + 'comparision_of_stacks_with_custom_'
                    +self.stname+"_"+
                    '{0:3.1f}'.format(self.gauss_filt * 2 * np.pi)+'.png')
        plt.savefig(name)
        if (self.close_fig == True):
            fig.clf()
            plt.cla()
            plt.close(fig)
    
    def fit_harmonic_to_bins(self):
        all_rf_r_bins = []
        all_rf_t_bins = []
        all_rf_phi_bins = []
        for l_stack in self.l_stack_array_bf_eval:
            for tr in l_stack:
                if (tr.stats.channel == "RFR"):
                    tr_r = tr.copy()
                    all_rf_r_bins.append(tr_r.data)
                    all_rf_phi_bins.append(tr_r.bin_theory_center)
                elif (tr.stats.channel == "RFT"):
                    tr_t = tr.copy()
                    all_rf_t_bins.append(tr_t.data)
                else:
                    tr_z = tr.copy()
        all_rf_r_bins = np.array(all_rf_r_bins)
        all_rf_t_bins = np.array(all_rf_t_bins)
        all_rf_phi_bins = np.array(all_rf_phi_bins)
        
        a_amp = []
        b_amp = []
        c_amp = []
        
    def binn_rf(self, nbin = 4, kind= 'back_azimuth'):
        
        r_comp_all = op.Stream()
        z_comp_all = op.Stream()
        baz_all = []
        ray_p_l = []
        for strm_rf in self.all_rf_station:
            for tr in strm_rf:
                if ('RFR' in tr.stats.channel):
                    r_comp_all.append(tr)
                    ray_p_l.append(tr.ray_p_s_to_km)
                    baz_all.append(tr.stats.back_azimuth)
                elif ('RFZ' in tr.stats.channel):
                    z_comp_all.append(tr)
            if (kind == 'back_azimuth'):
                min_par = 0.0
                max_par = 360.0
                par = baz_all.copy()
            else:
                min_par = np.min(ray_p_l)
                max_par = np.max(ray_p_l)
                par = ray_p_l.copy()
        #start bining
        n_divide = nbin + 1
        bin_corner = np.linspace(min_par, max_par, n_divide)
        bin_center = []
        for i in range(len(bin_corner) - 1):
            center = bin_corner[i] +\
                ((bin_corner[i+1] - bin_corner[i]) / 2)
            bin_center.append(center)
        bin_center = np.array(bin_center)
        bins_all = []
        inds_all = []
        lens_all = []
        for i in range(len(bin_corner) - 1):
            inds = []
            bins = []
            for j in range(len(par)):
                if ((par[j] >= bin_corner[i]) and 
                    (par[j] < bin_corner[i+1])):
                    bins.append(par[j])
                    inds.append(j)
                    
            bins_all.append(bins)
            inds_all.append(inds)
            
        check_par = 0
        while (check_par == 0):
            check_par = 1
            lens_all = []
            
            idum = -1
            merge_ind = []
            for el in inds_all:
                idum += 1
                lens_all.append(len(el))
                if (len(el) >= 10):
                    check_par = check_par * 1
                else:
                    check_par = check_par * 0
                    if (idum + 1 == len(inds_all)):
                        if (len(inds_all[idum -1]) < len(inds_all[0])):
                            merge_ind.append([idum, idum-1])
                        else:
                            merge_ind.append([idum, 0])
                    else:
                        if (len(inds_all[idum -1]) < len(inds_all[idum+1])):
                            if (idum - 1 < 0):    
                                merge_ind.append([idum, len(inds_all) - 1])
                            else:
                                merge_ind.append([idum, idum - 1])
                                
                        else:
                            merge_ind.append([idum, idum+1])
            if (check_par == 0):
                good_inds = []
                inds_all_new = []
                merge_ref = merge_ind[0]
                for ind_merge in merge_ref:
                    for el2 in inds_all[ind_merge]:
                        inds_all_new.append(el2)
                
                for i in range(len(inds_all)):
                    is_it_in_merge = False
                    for el1 in merge_ind:
                        for el2 in el1:
                            if (i == el2):
                                is_it_in_merge = True
                    if (is_it_in_merge == False):
                        if (len(inds_all[i]) > 0):
                            good_inds.append(inds_all[i])
                good_inds.append(inds_all_new)
                
                whats_left = []
                for el in merge_ind:
                    for el2 in el:
                        if ((el2 != merge_ref[0]) and (el2 != merge_ref[1])):
                            whats_left.append(el2)
                unique_whats_left = np.sort(np.unique(whats_left))
                for el in unique_whats_left:
                    if (len(inds_all[el]) > 0):
                        good_inds.append(inds_all[el])
                inds_all = []
                inds_all = good_inds.copy()           
        stacked_bins = []
        self.l_stack_array = []
        self.pw_stack_array = []
        for el in inds_all:
           l_stack, pw_stack = self.rf_stack_bins(el, kind)
           self.l_stack_array.append(l_stack)
           self.pw_stack_array.append(pw_stack)
                        
    def rf_stack_bins(self, inds, kind = 'back_azimuth'):
        
        all_rf_st = []
        for el in inds:
            all_rf_st.append(self.all_rf_station[el])
        
        
        r_comp_all = op.Stream()
        z_comp_all = op.Stream()
        t_comp_all = op.Stream()
        ray_p_l = []
        baz_all = []
        for strm_rf in all_rf_st:
            for tr in strm_rf:
                if ('RFR' in tr.stats.channel):
                    r_comp_all.append(tr)
                    ray_p_l.append(tr.ray_p_s_to_km)
                    baz_all.append(tr.stats.back_azimuth)
                elif ('RFZ' in tr.stats.channel):
                    z_comp_all.append(tr)
                else:
                    t_comp_all.append(tr)
        
        data_all_r = []
        for tr in r_comp_all:
            data_all_r.append(tr.data)
        data_all_r= np.array(data_all_r)
        
        data_all_z = []
        for tr in z_comp_all:
            data_all_z.append(tr.data)
        data_all_z= np.array(data_all_z)
        
        data_all_t = []
        for tr in t_comp_all:
            data_all_t.append(tr.data)
        data_all_t= np.array(data_all_t)
        
        
        mean_samp = [] 
        std_samp = [] 
        median_samp = []
        for i in range(len(r_comp_all[0].data)):
            samp_val = []
            for tr in r_comp_all:
                samp_val.append(tr.data[i])
            mean_samp.append(np.mean(samp_val))
            std_samp.append(np.std(samp_val))
            median_samp.append(np.median(samp_val))
            
            
        mean_samp = mean_samp
        std_samp = std_samp 
        median_samp = median_samp
        
        
        rfz_linear_stack = z_comp_all.copy()
        rfr_linear_stack = r_comp_all.copy()
        rft_linear_stack = t_comp_all.copy()
        
        rfz_pw_stack = z_comp_all.copy()
        rfr_pw_stack = r_comp_all.copy()
        rft_pw_stack = t_comp_all.copy()
        
        
        rfz_linear_stack.stack(group_by= 'station', 
                                            stack_type='linear')
        stacked_z = stack(data_all_z, stack_type='linear')
        rfz_linear_stack[0].data = stacked_z
        
        rfr_linear_stack.stack(group_by= 'station', 
                                            stack_type='linear')
        stacked_r = stack(data_all_r, stack_type='linear')
        rfr_linear_stack[0].data = stacked_r
        
        rft_linear_stack.stack(group_by= 'station', 
                                            stack_type='linear')
        stacked_t = stack(data_all_t, stack_type='linear')
        rft_linear_stack[0].data = stacked_t
        
        rfr_pw_stack.stack(group_by= 'station', 
                                        stack_type=('pw',2))
        stacked_r = stack(data_all_r, stack_type=('pw', 2))
        rfr_pw_stack[0].data = stacked_r
        
        # plt.figure()
        # plt.plot(rfr_linear_stack[0].data - mean_samp, label = 'd_l_m')
        # plt.plot( mean_samp- rfr_pw_stack[0].data, label = 'd_p_m')
        # plt.plot(rfr_linear_stack[0].data - rfr_pw_stack[0].data, 
        #          label = 'd_l_p')
        # plt.legend()
        # plt.title(str(np.mean(baz_all)) + 'N= '+str(len(baz_all)))
        
        
        rfz_pw_stack.stack(group_by= 'station', 
                                        stack_type=('pw',2))
        stacked_z = stack(data_all_z, stack_type=('pw', 2))
        rfz_pw_stack[0].data = stacked_z
        
        rft_pw_stack.stack(group_by= 'station', 
                                        stack_type=('pw',2))
        stacked_t = stack(data_all_t, stack_type=('pw', 2))
        rft_pw_stack[0].data = stacked_t
       
        
        
        l_stack = op.Stream() 
        pw_stack = op.Stream() 
        
        l_stack.append(rfr_linear_stack[0].copy())
        l_stack.append(rfz_linear_stack[0].copy()) 
        l_stack.append(rft_linear_stack[0].copy()) 
        
        l_stack[0].baz_all = baz_all 
        l_stack[1].baz_all = baz_all
        l_stack[2].baz_all = baz_all
        
        l_stack[0].ray_p_l = ray_p_l
        l_stack[1].ray_p_l = ray_p_l
        l_stack[2].ray_p_l = ray_p_l
        
        l_stack[0].inds = inds
        l_stack[1].inds = inds
        l_stack[2].inds = inds
        
        pw_stack.append(rfr_pw_stack[0].copy())
        pw_stack.append(rfz_pw_stack[0].copy())
        pw_stack.append(rft_pw_stack[0].copy())
        
        pw_stack[0].baz_all = baz_all 
        pw_stack[1].baz_all = baz_all
        pw_stack[2].baz_all = baz_all
        
        pw_stack[0].ray_p_l = ray_p_l
        pw_stack[1].ray_p_l = ray_p_l
        pw_stack[2].ray_p_l = ray_p_l
        
        pw_stack[0].inds = inds
        pw_stack[1].inds = inds
        pw_stack[2].inds = inds
        for tr in l_stack:
            tr.stats.stack_method = 'linear'
            tr.onset_ind = self.all_rf_station[0][1].onset_ind
            tr.time_vec = self.all_rf_station[0][1].time_vec
            tr.ray_p_s_to_km = np.mean(ray_p_l)
            tr.stats.back_azimuth = np.mean(baz_all)
            tr.stats.stack_kind = kind
            tr.rf_mean = mean_samp
            tr.rf_std = std_samp 
            tr.rf_median = median_samp
        for tr in pw_stack:
            tr.stats.stack_method = 'phase weighted'
            tr.onset_ind = self.all_rf_station[0][1].onset_ind
            tr.time_vec = self.all_rf_station[0][1].time_vec
            tr.ray_p_s_to_km = np.mean(ray_p_l)
            tr.stats.back_azimuth = np.mean(baz_all)
            tr.stats.stack_kind = kind
            tr.rf_mean = mean_samp
            tr.rf_std = std_samp 
            tr.rf_median = median_samp
        l_stack = l_stack.copy()
        pw_stack = pw_stack.copy()
        return(l_stack, pw_stack)
  
    def save_to_file(self, what_to_save, file_to_save):
        data_saved =False
        rf_saved = False
        st_obj_saved = False
        if (what_to_save == 'all_data'):
            what_to_save = self.all_data_station 
            file_to_save = self.network_name +'_'+self.stname+'_'+ \
                'all_data_bf_rf.bin'
            data_saved = True
        elif (what_to_save == 'all_rf'):
            what_to_save = self.all_rf_station
            file_to_save = self.network_name +'_'+self.stname+'_'+ \
                'all_rf_gf_'+'{0:3.1f}'.format(self.gauss_filt* 2 * np.pi)+'.bin'
            rf_saved = True
        elif (what_to_save =='all'):
            what_to_save = self 
            file_to_save = self.network_name +'_'+self.stname+'_'+ \
                'st_object.bin'
            st_obj_saved = True
        file_to_save = self.save_folder + file_to_save 
        
        file1 = open(file_to_save, "wb") 
        pickle.dump(what_to_save, file1)
        file1.close()
        if (data_saved):
            print('all data saved for network= '+self.network_name+
              ' ,station= '+
              self.stname+' to:\n'+ file_to_save)
        elif (rf_saved):
            print('all rfs saved for network= '+self.network_name+
              ' ,station= '+
              self.stname+' to:\n'+ file_to_save)
        elif (st_obj_saved):
            print('station object saved for network= '+self.network_name+
              ' ,station= '+
              self.stname+' to:\n'+ file_to_save)
            return(file_to_save)

    def load_from_file(self, what_to_load, file_to_load):
        if (what_to_load == 'all_data'):
            file_to_load = self.network_name +'_'+self.stname+'_'+ \
                'all_data_bf_rf.bin'
            file_to_load = self.save_folder + file_to_load
            if (os.path.isfile(file_to_load)):
                with open(file_to_load, 'rb') as f1:
                    self.all_data_station = pickle.load(f1)
                print('all data laoded for network= '+self.network_name+
                      ' ,station= '+
                      self.stname+' loaded successfully')
            else:
                print('I CANT FIND THIS FILE: '+file_to_load)
        elif (what_to_load == 'all_rf'):
            file_to_load = self.network_name +'_'+self.stname+'_'+ \
                'all_rf_gf_'+'{0:3.1f}'.format(self.gauss_filt* 2 * np.pi)+'.bin'
            file_to_load = self.save_folder + file_to_load
            if (os.path.isfile(file_to_load)):
                with open(file_to_load, 'rb') as f1:
                    self.all_rf_station = pickle.load(f1)
                print('all RFs loaded for network= '+self.network_name+' ,station= '
                      +self.stname+
                      ' loaded successfully')
                self.nsamp = len(self.all_rf_station[0][0].data)
                self.time_vec = self.all_rf_station[0][0].time_vec
            else:
                print('I CANT FIND THIS FILE: '+file_to_load)    
            
        else:
           with open(file_to_load, 'rb') as f1:
               self.loaded_file = pickle.load(f1)

    def cal_rf(self):
        print('calculating RFs for network= '
              +self.network_name+', station= '+
              self.stname)
        all_rf_station = []
        idum = -1
        if (self.review_station):
            print('Number of data is : ' + str(len(self.all_data_station)))
        for strm_zrt in self.all_data_station:
            idum +=1
            # print('calculating RF for '+ strm_zrt[0].stats.folder_name+ 
                  # '  ind= '+str(idum))
            strm_rf, was_it_fix = self.calculate_receiver_function(strm_zrt)
            if (was_it_fix):
                all_rf_station.append(strm_rf)
            # for tr in strm_rf:
                # find_snr(tr, tr.stats.P_onset)
        if (self.review_station):
            print('Number of good RFs is : ' + str(len(all_rf_station)))
        
        self.all_rf_station = all_rf_station
        exmple_tr = self.all_rf_station[0][0]
        self.nsamp = exmple_tr.stats.npts
        self.time_vec = exmple_tr.time_vec.copy()
        self.onset_ind = exmple_tr.onset_ind
        self.info_for_save['N_RFS_bf_app_vel_criteria'] = len(self.all_rf_station)
        # self.save_rf_for_mrs_farzaneh()
        
    def save_rf_for_mrs_farzaneh(self):
        dummy_strm = self.all_rf_station.copy()
        for strm in dummy_strm:
            for tr in strm:
                if (tr.stats.channel == 'RFR'):
                    tr.stats.starttime = op.UTCDateTime(year = tr.stats.sac.nzyear,
                                                       julday = tr.stats.sac.nzjday, 
                                                       hour = tr.stats.sac.nzhour, 
                                                       minute = tr.stats.sac.nzmin, 
                                                       second = tr.stats.sac.nzsec,
                                                       microsecond = tr.stats.sac.nzmsec * 1000)
                    tr_r = tr.copy()
                    fl_name = ('c60_'+tr.stats.folder_name[-14:-1] + tr.stats.station+ '._'+
                               str(np.round(self.gauss_filt* 2.0 * np.pi, 2))+'.i.'+'eqr')
                    fl_name_full = tr.stats.folder_name + fl_name
                    
                    tr_r.write(fl_name_full, format = 'SAC')
                elif (tr.stats.channel == 'RFT'):
                    tr.stats.starttime = op.UTCDateTime(year = tr.stats.sac.nzyear,
                                                       julday = tr.stats.sac.nzjday, 
                                                       hour = tr.stats.sac.nzhour, 
                                                       minute = tr.stats.sac.nzmin, 
                                                       second = tr.stats.sac.nzsec,
                                                       microsecond = tr.stats.sac.nzmsec * 1000)
                    tr_t = tr.copy()
                    fl_name = ('c60_'+tr.stats.folder_name[-14:-1] + tr.stats.station+ '._'+
                               str(np.round(self.gauss_filt* 2.0 * np.pi, 2))+'.i.'+'eqt')
                    fl_name_full = tr.stats.folder_name + fl_name
                    
                    tr_t.write(fl_name_full, format = 'SAC')
                else:
                    tr.stats.starttime = op.UTCDateTime(year = tr.stats.sac.nzyear,
                                                       julday = tr.stats.sac.nzjday, 
                                                       hour = tr.stats.sac.nzhour, 
                                                       minute = tr.stats.sac.nzmin, 
                                                       second = tr.stats.sac.nzsec,
                                                       microsecond = tr.stats.sac.nzmsec * 1000)
                    tr_z = tr.copy()
                    fl_name = ('c60_'+tr.stats.folder_name[-14:-1] + tr.stats.station+ '._'+
                               str(np.round(self.gauss_filt* 2.0 * np.pi, 2))+'.i.'+'eqz')
                    fl_name_full = tr.stats.folder_name + fl_name
                    
                    tr_z.write(fl_name_full, format = 'SAC')
        
        
        
        

    def read_files(self):
        print('reading files for network = '+self.network_name+' ,station = '+
              self.stname)
    
        events_folder = [self.st_folder + f +'/' for f in os.listdir(self.st_folder)
                           if os.path.isdir(os.path.join(self.st_folder, f))]
        self.info_for_save['N_folder'] = len(events_folder)
        #reading all the ZNE traces from event folders in station folder
        self.info_for_save['bad_events_for_read'] = []
        all_data_station = []
        for event in events_folder:
            try:
                strm_4_rf, is_it_good = self.read_for_rf(event, trace= 'ZNE',
                                        sample_rate= self.to_dt)
                if (is_it_good):
                    all_data_station.append(strm_4_rf)
                else:
                    self.info_for_save['bad_events_for_read'].append(event)
            except:
                self.info_for_save['bad_events_for_read'].append(event)
        self.all_data_station = all_data_station
        self.info_for_save['N_imported_events'] = len(all_data_station)
        if (self.review_station):
            print('From '+str(len(events_folder))+' event folders, '+
                  str(len(all_data_station))+ 
                  ' events successfully imported for calculations')
    def cut_trace_for_p(self):
        print('preparing files for rf calculation for network = '
              +self.network_name+', station = '+
              self.stname)
        if (self.review_station):
            self.number_of_data_bf_cut_trace = len(self.all_data_station)
        remove_ind = []
        idum = -1
        all_data_station_dum= []
        for strm in self.all_data_station:
            idum += 1
            # print('preparing '+ strm[0].stats.folder_name+ 
            #         ' for RF calculation')
            
            is_it_good_val = 0
            for tr in strm:
                is_it_good = False
                tr.resample(sampling_rate= self.to_sample)
                tr.detrend(type= "demean")
                try:
                    is_it_good = find_onset_amp(tr, dist_range = 
                               self.dist_range,
                               bf_p= self.bf_p,
                               af_p = self.af_p)
                except:
                    continue
                if (is_it_good):
                    is_it_good_val += 1
                
            if (is_it_good_val == 3):
                snr_good = self.check_snr(strm)
                if (snr_good):
                    all_data_station_dum.append(strm)
                # all_data_station_dum.append(strm)
                # print(len(all_data_station_dum))
        self.all_data_station = []
        if (self.review_station):
            self.number_of_data_af_cut_trace = len(all_data_station_dum)
            print('From ' + str(self.number_of_data_bf_cut_trace) + ' trace, ' +
                  str(self.number_of_data_af_cut_trace) + ' was preprocessed and added to database')
        self.all_data_station = all_data_station_dum.copy()
        self.info_for_save['N_preprocessed'] = len(self.all_data_station)
          
    def check_snr(self, strm):
        for tr in strm:
            if ('Z' in tr.stats.channel):
                trz = tr.copy() 
            elif ('R' in tr.stats.channel):
                trr = tr.copy() 
            elif ('T' in tr.stats.channel):
                trt = tr.copy()
        if ((trr.stats.snr > 0.10) and (trz.stats.snr > 0.10)):
            return(True)
        else:
            return(False)
            
    def calculate_receiver_function(self, strm_input):

        for tr in strm_input:
            if ('Z' in tr.stats.channel):
                trz = tr.copy() 
            elif ('R' in tr.stats.channel):
                trr = tr.copy() 
            elif ('T' in tr.stats.channel):
                trt = tr.copy()
            else:
                print('THIS STREAM DOESNT HAVE ZRT CHECK IT')
        if (trz.stats.npts == 0):
            return(op.Stream(), False)
        # print(trz.stats.file_name)
        # if ('/home/soroush/rf_shallow_codes/makran_data/ZHDN/2018025035328/2018025035328_BHZ.SAC' ==
        #     trz.stats.file_name):
        #     a = 1
        sample_rate = trz.stats.sampling_rate
        src = trz
        rsp_list = [trr, trz, trt]
        if (self.rf_method == 'waterlevel'):
            out = deconv_waterlevel(rsp_list, src, sampling_rate= sample_rate , 
                                    waterlevel=self.waterlevel, 
                                gauss=self.gauss_filt,
                              tshift=self.tshift, length=None, 
                              normalize=self.rf_normalize, 
                              nfft=None,
                              return_info=False)
        elif (self.rf_method == 'iterative'):
            out, _, rms = deconv_iterative(rsp_list, src,
                                         sampling_rate= sample_rate,
                                         tshift= self.tshift,
                                         gauss= self.gauss_filt,
                                         mute_shift= True,
                                         normalize=self.rf_normalize)
            
        dr = out[0]
        dr = dr.real 
        dz = out[1].real
        dz = dz.real
        dt = out[2] 
        dt = dt.real
        if (self.review_station):
            fig = plt.figure(figsize=(16,8))
            plt.plot(trr.time_vec, dr, ls = '--',label = 'dr')
            plt.plot(trr.time_vec, dz, label = 'dz')
            plt.vlines(0, ymin = 0, ymax= 1.0, color = 'red')
            plt.grid(True)
            plt.legend()
        
        (dr_fixed, was_it_fix) = self.check_for_samp_shift(dr, dz, sample_rate)
        
        
        if ((self.review_station == True) and (was_it_fix)):
            plt.plot(trr.time_vec, dr_fixed, label = 'dr_fixed')
        if ((self.review_station == True) and (was_it_fix == False)):
            plt.plot(trr.time_vec, dr_fixed, label = 'dr_not_fixed')
        # plt.plot(trr.time_vec, dz, label = 'dz')
        # plt.vlines(0, ymin = 0, ymax= 0.4)
        if (self.review_station):
            folder_review = os.path.join(self.save_folder, 'review_station_rfs')
            folder_review_wasnt_fix = os.path.join(folder_review, 'wasnt_good')
            folder_review_was_fix = os.path.join(folder_review, 'was_good')
            folders_to_work = [folder_review, folder_review_wasnt_fix,
                               folder_review_was_fix]
            for folder in folders_to_work:    
                if (os.path.isdir(folder)):
                    pass
                else:
                    os.mkdir(folder)
                    
            if (was_it_fix):
                plt.legend()
                x_ticks = np.linspace(-self.bf_p, self.af_p, 
                                      int(abs(self.bf_p)+abs(self.af_p) + 1))
                plt.xticks(x_ticks)
                plot_name = os.path.join(folder_review_was_fix, 
                                         trr.stats.folder_name[-13:-1])
                title_of_fig = ('trace : ' + trr.stats.file_name + ' Baz : '+
                                str(tr.stats.back_azimuth) + ' distance : '+
                                str(tr.stats.distance))
                plt.title(title_of_fig)
                plt.tight_layout()
                plt.savefig(plot_name + '.png', dpi = 100)
            if (was_it_fix == False):
                plt.legend()
                plot_name = os.path.join(folder_review_wasnt_fix, 
                                         trr.stats.folder_name[-13:-1])
                title_of_fig = ('trace : ' + trr.stats.file_name + ' Baz : '+
                                str(tr.stats.back_azimuth) + ' distance : '+
                                str(tr.stats.distance))
                plt.title(title_of_fig)
                plt.tight_layout()
                plt.savefig(plot_name + '.png', dpi = 100)
            if (self.close_fig):
                fig.clf()
                plt.cla()
                plt.close(fig)
        rf_z = trz.copy() 
        rf_r = trr.copy() 
        rf_t = trt.copy()
        
        rf_z.data = dz.copy()
        rf_z.stats.channel = 'RFZ'
        rf_r.data = dr_fixed.copy() 
        rf_r.stats.channel = 'RFR'
        rf_t.data = dt.copy()
        rf_t.stats.channel = 'RFT'
        
        sample_trace = op.Trace()
        stime = sample_trace.stats.starttime
        strm_out = op.Stream() 
        strm_out.append(rf_z)
        strm_out.append(rf_r)
        strm_out.append(rf_t)
    
        for tr in strm_out:
            if (self.rf_method == 'iterative'):
                tr.stats.rf_rms = rms[-1]
            tr.stats.s_time_bf_rf = tr.stats.starttime 
            tr.stats.e_time_bf_rf = tr.stats.endtime
            tr.stats.starttime = stime
            tr.stats.was_it_fix = was_it_fix
            tr.stats.RF_method = self.rf_method
            tr.stats.RF_gauss = self.gauss_filt
            tr.stats.network = self.network_name 
            tr.stats.method_phase = 'P'
            
        
    
    
        return(strm_out, was_it_fix)  
    def check_for_samp_shift(self, dr, dz, sample_rate, 
                             time_to_check = 1):
        
        n_cut = int(sample_rate * time_to_check)
        is_it_good = False
        onset_ind_z = np.argwhere((dz) == np.max((dz)))[0][0]
        dz_4_check = dz[0: onset_ind_z + n_cut + 2]
        dr_4_check = dr[0: onset_ind_z + n_cut + 2]
        
        ind_z = np.argwhere((dz_4_check) == np.max((dz_4_check)))[0][0]
        ind_r = np.argwhere((dr_4_check) == np.max((dr_4_check)))[0][0]
        
        
        dr_fixed = np.zeros(shape=(len(dr),))
        if (ind_z == ind_r):
            is_it_good = True
            return(dr, is_it_good)
        elif (abs(ind_z - ind_r) <= n_cut):
            is_it_good = False
            dif_ind = ind_z - ind_r 
            if (dif_ind < 0):
                for i in range(len(dr)):
                    if (i - dif_ind <= (len(dr) - 1)):
                        dr_fixed[i] = dr[i-dif_ind]
                    else:
                        dr_fixed[i] = 0.0
            else:                
                for i in range(len(dr)):
                    if (i >= dif_ind):
                        dr_fixed[i] = dr[i-dif_ind]
                    else:
                        dr_fixed[i] = 0.0
            
            
            
            ind_zero_z = (np.argwhere((dz) ==
                                    np.max((dz)))[0][0])
            ### if you want to consider rfrs which have maximum
            ### at times that differ from p-arrival you can 
            ### change line below to 
            ### max_dr_fixed = np.max(dr_4_check)
            
            
            max_dr_fixed = np.max(dr_fixed)
            # max_dr_fixed = np.max(dr_4_check)
            normalized_dr_fixed = dr_fixed / max_dr_fixed
            for i in np.arange(0, ind_zero_z):
                # if (dr_fixed[i] < 0):
                #     dr_fixed[i] = 0
                normalized_dr_fixed[i] = dz[i]
            
            dr_fixed = normalized_dr_fixed * max_dr_fixed
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
                filt = _gauss_filter_shifted(self.dt, n_sample, gauss_freq,
                                 waterlevel=None)
                filter_array.append(filt)
            self.filter_array = filter_array   
    def plot_rf(self, component = 'RFR', af = 15):

        all_rf_to_plot = []
        for strm in self.all_rf_station:
            for tr in strm:
                if (component in tr.stats.channel):
                    all_rf_to_plot.append(tr)


        fig, axs = plt.subplots(figsize=(12,8), 
                                    nrows=len(all_rf_to_plot), sharex=True, 
                                          subplot_kw=dict(frameon=False), dpi = 350)
        fig.suptitle(' All rf component= '+component)

        plt.subplots_adjust(hspace=.001)
        baz_all = []
        for tr in all_rf_to_plot:
            baz_all.append(tr.stats.back_azimuth)
        sort_ind = sorted(range(len(baz_all)), key=lambda k: baz_all[k])

        time = tr.time_vec 
        time = time - af 
        ind = np.argwhere(np.min(abs(time)) == abs(time))[0][0]

        all_rf_to_plots = []
        for i in range(len(all_rf_to_plot)):
            all_rf_to_plots.append(all_rf_to_plot[sort_ind[i]])
        all_rf_to_plot = []
        all_rf_to_plot = all_rf_to_plots.copy()
        all_rf_to_plots = []

        idum = -1
        for tr in all_rf_to_plot:
            idum += 1
            x = tr.time_vec[:ind+1]
            y = tr.data[:ind+1] 
            
            axs[idum].plot(x, y, color= 'black', lw = .5)
            axs[idum].fill_between(x, 0, y, where=y>0, color = 'red')
            axs[idum].fill_between(x, 0, y, where=y<0, color = 'blue')
            axs[idum].yaxis.set_visible(False)
            axs[idum].text(-2, 0, '{0:5.1f}'.format(tr.stats.back_azimuth), 
                           fontsize=9)
        name = (self.save_folder + 'All_rf_'+self.rf_method
                +'{0:3.1f}'.format(self.gauss_filt* 2 * np.pi)+'.png')
        plt.savefig(name)
        if (self.close_fig == True):
            fig.clf()
            plt.cla()
            plt.close(fig)
    def plot_stack(self, component = 'RFR',
                   af = 15):
        fig, axs = plt.subplots(figsize=(8,12), 
                                    nrows=2, sharex=True, 
                                    subplot_kw=dict(frameon=True), dpi = 250)
        if (component == 'RFR'):
            li = self.l_stack[0]
            pw = self.pw_stack[0]
        else:
            li = self.l_stack[1]
            pw = self.pw_stack[1]
        time = li.time_vec 
        time = time - af 
        ind = np.argwhere(np.min(abs(time)) == abs(time))[0][0]
        
        stacked = []
        stacked.append(li)
        stacked.append(pw)
        idum = -1
        for tr in stacked:
            idum += 1
            x = tr.time_vec[:ind+1]
            y = tr.data[:ind+1] 
            if (idum == 0):
                axs[idum].set_title('Linear Stack')
            else:
                axs[idum].set_title('Phase weighted Stack')
            axs[idum].plot(x, y, color= 'black', lw = .5)
            axs[idum].fill_between(x, 0, y, where=y>0, color = 'red')
            axs[idum].fill_between(x, 0, y, where=y<0, color = 'blue')
            axs[idum].grid(True, axis = 'both')
        name = (self.save_folder + 'stacked_rf_'+self.rf_method
                +'{0:3.1f}'.format(self.gauss_filt* 2 * np.pi)+'.png')
        plt.savefig(name)
        if (self.close_fig == True):
            fig.clf()
            plt.cla()
            plt.close(fig)

    def plot_app_vel(self):
        baz_list = []
        app_list = []
        for el in self.all_rf_station:
            baz_list.append(el[1].stats.back_azimuth)
            app_list.append(el[1].app_vel)
        min_baz = np.min(baz_list)
        max_baz = np.max(baz_list)
        sort_ind = sorted(range(len(baz_list)), key=lambda k: baz_list[k])
        app_list_s = []
        baz_list_s = []
        for i in range(len(app_list)):
            app_list_s.append(app_list[sort_ind[i]])
            baz_list_s.append(baz_list[sort_ind[i]])
        app_list = []
        app_list = app_list_s.copy()
        app_list_s = []
        baz_list = []
        baz_list = baz_list_s.copy()
        baz_list_s = []
        
        
        viridis=cm.get_cmap('viridis', len(baz_list))
        fig = plt.figure(figsize=(12,8), dpi =200)
        for i in range(len(app_list)):    
            plt.plot(self.filt_list, app_list[i] 
                 ,color=viridis(i))
        plt.xlabel('Filter period')
        plt.ylabel('Apparent Velocity')
        norm= matplotlib.colors.BoundaryNorm(baz_list, 
                                             len(baz_list))
        sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ticks=baz_list ,label='Back Azimuth')
        name = (self.save_folder + 'App_vel_'+self.rf_method
                +'{0:3.1f}'.format(self.gauss_filt* 2 * np.pi)+'.png')
        plt.savefig(name)
        if (self.close_fig == True):
            fig.clf()
            plt.cla()
            plt.close(fig)

    def read_for_rf(self, folder_name, trace= 'ZRT', dist_range=(30,90), 
                    sample_rate = 0.1):
        file_list = [f for f in os.listdir(folder_name) 
                     if os.path.isfile(os.path.join(folder_name, f))]
        # print('reading from ' + folder_name)
        if (trace == 'ZRT'):
            switch_to_find = '.BH'
        else:
            switch_to_find = '.SAC'
        sac_files = []
        for el in file_list:
            if (switch_to_find in el):
                sac_files.append(el)
        if (len(sac_files) != 3):
            a = op.Stream()
            return(a, False)
        # print(sac_files)
        
        sac_strm = {}
        for el in sac_files:
            
            tr = op.read(folder_name + el)[0]
            if (len(tr.data) < 10):
                if (self.review_station):
                    print('Encounter problem in reading event in folder '+ 
                          folder_name + 
                          ' This event removed from calculations')
                return('False_strm', False)
            tr.stats.file_name = folder_name + el
            tr.stats.folder_name = folder_name
            tr.stats.kind ='Real_data'
            if ('R' in tr.stats.channel):
                tr.stats.channel = 'BHR'
                tr.stats.sac.kcmpnm = 'BHR'
                sac_strm['Radial'] = tr
            elif ('T' in tr.stats.channel):
                tr.stats.channel = 'BHT'
                tr.stats.sac.kcmpnm = 'BHT'
                sac_strm['Transverse'] = tr
            elif ('Z' in tr.stats.channel):
                tr.stats.channel = 'BHZ'
                tr.stats.sac.kcmpnm = 'BHZ'
                sac_strm['Vertical'] = tr
            elif ('N' in tr.stats.channel):
                tr.stats.channel = 'BHN'
                tr.stats.sac.kcmpnm = 'BHN'
                sac_strm['North'] = tr
            elif ('E' in tr.stats.channel):
                tr.stats.channel = 'BHE'
                tr.stats.sac.kcmpnm = 'BHE'
                sac_strm['East'] = tr
        if (trace == 'ZNE'):
            sac_strm_ZNE = op.Stream()
            sac_strm_ZNE.append(sac_strm['Vertical'])
            sac_strm_ZNE.append(sac_strm['North'])
            sac_strm_ZNE.append(sac_strm['East'])
        else:
            sac_strm_ZRT = op.Stream()
            sac_strm_ZRT.append(sac_strm['Vertical'])
            sac_strm_ZRT.append(sac_strm['Radial'])
            sac_strm_ZRT.append(sac_strm['Transverse'])
        # t = sac_strm['Vertical'].times()
        # fig, ax = plt.subplots(nrows=3, ncols= 1,
        #                                figsize=(12, 8))
        # ax[0].plot(t, sac_strm['Vertical'].data)
        # ax[1].plot(t, sac_strm['North'].data)
        # ax[2].plot(t, sac_strm['East'].data)
        
        
        freqmax = (1 / (sample_rate)) * 0.5 - 1
        if (trace == 'ZRT'):
            rf_st = rf.RFStream(sac_strm_ZRT)
        else:
            strm_zne = op.Stream()
            for tr in sac_strm_ZNE:
                # if (tr.stats.sac.nzjday == 0):
                #     tr.stats.sac.nzjday = 365
                # out = self.model_tp.get_travel_times_geo(
                #     source_depth_in_km= tr.stats.sac.evdp,
                #     source_latitude_in_deg=tr.stats.sac.evla,
                #     source_longitude_in_deg = tr.stats.sac.evlo,
                #     receiver_latitude_in_deg= tr.stats.sac.stla,
                #     receiver_longitude_in_deg = tr.stats.sac.stlo,
                #     phase_list='P')
                # # print(tr.stats.file_name)
                # t0 = out[0].time + tr.stats.sac.o
                tr.stats.event_depth = tr.stats.sac.evdp
                
                
                
                if hasattr(tr.stats.sac, 't1'):
                    stime = tr.stats.starttime + tr.stats.sac.t1 - 20
                    etime = tr.stats.starttime + tr.stats.sac.t1 + 60
                else:
                    stime = tr.stats.starttime + tr.stats.sac.t0 - 20
                    etime = tr.stats.starttime + tr.stats.sac.t0 + 60
                
                tr.trim(starttime = stime, endtime = etime)
                s_rate = tr.stats.sampling_rate
                n_data_must_be = (s_rate * 80) + 1
                if (len(tr.data) != n_data_must_be):
                    if (self.review_station):
                        print('The data for event in folder '+ 
                              folder_name + 
                              'is not consistence with other data.'+
                              ' This event removed from calculations')
                    return('False_strm', False)
                tr.detrend(type='linear')
                tr.detrend(type='demean')
                if (self.freq_max_bpfilt == 'default'):
                    tr.filter(type='bandpass', freqmin = self.freq_min_bpfilt, 
                              freqmax = freqmax, zerophase = True)
                else:
                    tr.filter(type='bandpass', freqmin = self.freq_min_bpfilt, 
                              freqmax = self.freq_max_bpfilt, zerophase = True)
                tr.detrend(type='linear')
                tr.detrend(type='demean')
                strm_zne.append(tr)
            # t = strm_zne[0].times()
            # fig, ax = plt.subplots(nrows=3, ncols= 1,
            #                                figsize=(12, 8))
            # for tr in strm_zne:
            #     if (tr.stats.channel == 'BHZ'):
            #         ax[0].plot(t, tr.data)
            #     if (tr.stats.channel == 'BHN'):
            #         ax[1].plot(t, tr.data)
            #     if (tr.stats.channel == 'BHE'):
            #         ax[2].plot(t, tr.data)
            rf_st = rf.RFStream(strm_zne)
            rf_st.rotate(method= 'NE->RT', 
                            back_azimuth=rf_st[0].stats.back_azimuth,
                            inclination=None)
            # t = rf_st[0].times()
            # fig, ax = plt.subplots(nrows=3, ncols= 1,
            #                                figsize=(12, 8))
            # for tr in strm_zne:
            #     if ('Z' in tr.stats.channel):
            #         ax[0].plot(t, tr.data)
            # for tr in rf_st:
            #     if ('R' in tr.stats.channel):
            #         ax[1].plot(t, tr.data)
            # for tr in rf_st:
            #     if ('T' in tr.stats.channel):
            #        ax[2].plot(t, tr.data)
                    
                    
                    
            rf_st_sorted = op.Stream()
            for tr in rf_st:
                if ('Z' in tr.stats.channel):
                    rf_st_sorted.append(tr)
            for tr in rf_st:
                if ('R' in tr.stats.channel):
                    rf_st_sorted.append(tr)
            for tr in rf_st:
                if ('T' in tr.stats.channel):
                    rf_st_sorted.append(tr)
            rf_st = rf_st_sorted        
            
            
            
            
        if (self.stname in self.visual_insp):
            tr = rf_st_sorted[0]
            if hasattr(tr.stats.sac, 't2'):
                return(rf_st, True)
            else:
                return(rf_st, False)
        else:
            return(rf_st, True) 

    def define_velocity_from_inv(self, inv_ref = 'best'):
        self.vel_final = {}
        self.vel_final['mean velocity'] = []
        self.vel_final['std velocity'] = []
        self.vel_final['median velocity'] = []
        inv_info_all = []
        for l_stack in self.l_stack_array:
            inv_info_all.append(l_stack[0].inv_info)
        if (inv_ref == 'best'):
            vel_all = []
            lthick_all = []
            for ii in inv_info_all:
                vel_all.append(ii['best_inv'][2])
                lthick_all.append(ii['best_inv'][0])
            self.vel_final['inv_ref'] = inv_ref
        else:
            iter_ref = inv_ref
            vel_all = []
            lthick_all = []
            for ii in inv_info_all:
                vel_all.append(ii['all_iter'][iter_ref][2])
                lthick_all.append(ii['all_iter'][iter_ref][0])
            self.vel_final['inv_ref'] = str(iter_ref)
        vel_all_smooth = []
        for i in range(len(vel_all)):
            v_smooth = self.interp_vel(lthick_all[i], 
                            vel_all[i])
            vel_all_smooth.append(v_smooth)
        for i in range(len(vel_all_smooth[0])):
            dum = []
            for vel in vel_all_smooth:
                dum.append(vel[i])
            mean_vel_samp = np.mean(dum)
            std_vel_samp = np.std(dum)
            median_vel_samp = np.median(dum)
            self.vel_final['mean velocity'].append(mean_vel_samp)
            self.vel_final['std velocity'].append(std_vel_samp)
            self.vel_final['median velocity'].append(median_vel_samp)
        self.velocity = self.vel_final['mean velocity']
        self.plot_velocity()

    def interp_vel(self, lthick, vel):
        dum1 = 0.0
        l_thickness_abs_best = []
        for i in range(len(lthick) - 1):
            dum1 = dum1 + lthick[i]
            l_thickness_abs_best.append(dum1)
        l_thickness_abs_best.append(dum1 + 50)
        lthickness_smooth = []
        for i in range(39):
            lthickness_smooth.append(1)
        lthickness_smooth.append(0)
        lthickness_smooth_abs = []
        dum1 = 0.0
        for i in range(len(lthickness_smooth) -1):
            dum1 = dum1 + lthickness_smooth[i]
            lthickness_smooth_abs.append(dum1)
        lthickness_smooth_abs.append(dum1 + 50)
        vel_smooth = np.interp(np.array(lthickness_smooth_abs),
                          np.array(l_thickness_abs_best),
                          np.array(vel))
        self.lthickness = lthickness_smooth
        self.lthickness_abs = lthickness_smooth_abs
        
        return(vel_smooth)
         
    def forward_vel(self, vel, lthickness):
        f_vel = iv.Forward_cal(vel_s= vel, 
                       layers_thickness = lthickness, 
                       filt_list= self.filt_list,
                       kind = 'rftn',
                       gauss_par= self.gauss_filt,
                       dt=0.05,nsamp=1024, tshift=10.0,
                       rf_method = self.rf_method,
                       slowness = self.l_stack[0].ray_p_s_to_km,
                       rf_normalize= self.rf_normalize,
                       inv_time_rf1= self.bf_p, 
                       inv_time_rf2= self.af_p)
        rf = f_vel.rf_r_4_inv
        app_curve = f_vel.apparant_vel_org
        return(rf, app_curve)

    def plot_velocity(self, max_vel_report = 30):
        self.max_vel_report = max_vel_report
        velocity = self.velocity
        lthick = self.lthickness
        lthick_abs = self.lthickness_abs
        x = [] 
        y = []
        idum = -1
        for el in lthick_abs:
            idum += 1
            if (el-lthick_abs[0] <= max_vel_report+4):
                x.append(velocity[idum])
                y.append(el -lthick_abs[0])
        fig = plt.figure(figsize=(4, 8),dpi = 300,
                             facecolor='#FAEBD7')
        plt.xlim([1.5, 4.5])
        plt.ylim([0, max_vel_report + 2])
        plt.gca().invert_yaxis()
        plt.xlabel('Velocity (km/s)', fontsize= 15)
        plt.ylabel('Depth (km)', fontsize= 15)
        plt.title('Estimated Velocity')
        plt.plot(x, y)
        plt.grid(True)
        self.fig_velocity = fig
        if (self.close_fig == True):
            fig.clf()
            plt.cla()
            plt.close(fig)

    def find_app_vel_stack(self):
        app_vel = [] 
        app_vel = self.find_app_vel_func(self.l_stack)
        
        for tr in self.l_stack:
            tr.app_vel = app_vel.copy()  
            tr.app_vel_mean = self.mean_vel
            tr.app_vel_median = self.std_vel
            tr.app_vel_std = self.median_vel
            
            

        a_vel= []
        for strm in self.all_rf_station:
            tr = strm[0]
            a_vel.append(tr.app_vel) 
            
                
        app_vel = [] 
        app_vel = self.find_app_vel_func(self.pw_stack)
        
        for tr in self.pw_stack:
            tr.app_vel = app_vel.copy()
            tr.app_vel_mean = self.mean_vel
            tr.app_vel_median = self.std_vel
            tr.app_vel_std = self.median_vel

    def find_app_vel_stack_syn(self):
        app_vel = [] 
        app_vel = self.find_app_vel_func(self.syn_l_stack)
        
        for tr in self.syn_l_stack:
            tr.app_vel = app_vel.copy()  
            tr.app_vel_mean = self.mean_vel
            tr.app_vel_median = self.std_vel
            tr.app_vel_std = self.median_vel
            
            

        a_vel= []
        for strm in self.syn_all_rf:
            tr = strm[0]
            a_vel.append(tr.app_vel) 
            
                
        app_vel = [] 
        app_vel = self.find_app_vel_func(self.syn_pw_stack)
        
        for tr in self.syn_pw_stack:
            tr.app_vel = app_vel.copy()
            tr.app_vel_mean = self.mean_vel
            tr.app_vel_median = self.std_vel
            tr.app_vel_std = self.median_vel

    def rf_stack(self, kind = 'real'):
        self.r_comp_all = op.Stream()
        self.z_comp_all = op.Stream()
        self.t_comp_all = op.Stream()
        ray_p_l = []
        baz_all = []
        if (kind == 'real'):
            for strm_rf in self.all_rf_station:
                for tr in strm_rf:
                    if ('RFR' in tr.stats.channel):
                        self.r_comp_all.append(tr)
                        ray_p_l.append(tr.ray_p_s_to_km)
                        baz_all.append(tr.stats.back_azimuth)
                    elif ('RFZ' in tr.stats.channel):
                        self.z_comp_all.append(tr)
                    else:
                        self.t_comp_all.append(tr)
        else:
            for strm_rf in self.syn_all_rf:
                for tr in strm_rf:
                    if ('RFR' in tr.stats.channel):
                        self.r_comp_all.append(tr)
                        ray_p_l.append(tr.ray_p_s_to_km)
                        baz_all.append(tr.stats.back_azimuth)
                    elif ('RFZ' in tr.stats.channel):
                        self.z_comp_all.append(tr)
        

        data_all_r = []
        for tr in self.r_comp_all:
            data_all_r.append(tr.data)
        data_all_r= np.array(data_all_r)
        
        data_all_z = []
        for tr in self.z_comp_all:
            data_all_z.append(tr.data)
        data_all_z= np.array(data_all_z)
        
        data_all_t = []
        for tr in self.t_comp_all:
            data_all_t.append(tr.data)
        data_all_t= np.array(data_all_t)
        
        mean_samp = [] 
        std_samp = [] 
        median_samp = []
        for i in range(len(self.r_comp_all[0].data)):
            samp_val = []
            for tr in self.r_comp_all:
                samp_val.append(tr.data[i])
            mean_samp.append(np.mean(samp_val))
            std_samp.append(np.std(samp_val))
            median_samp.append(np.median(samp_val))
            
            
        self.mean_samp = mean_samp
        self.std_samp = std_samp 
        self.median_samp = median_samp
        
        inds_all_stack = np.arange(len(baz_all))
        
        l_stack, pw_stack = self.rf_stack_bins(inds = inds_all_stack, 
                                               kind = 'back_Azimuth')
        if (kind == 'real'):
            self.l_stack = l_stack.copy()
            self.pw_stack = pw_stack.copy()
        else:
            self.syn_l_stack = l_stack.copy()
            self.syn_pw_stack = pw_stack.copy()

    def create_syn(self, vel_syn, lthickness, 
                   noise_level = 20):
        rf_normalize = 1
        nsamp = 2048
        waterlevel_val = 0.01
        dt = 0.05
        self.syn_all_rf = self.all_rf_station.copy()
        for strm in self.syn_all_rf:
            for el in strm:
                if ('RFR' == el.stats.channel):
                    tr = el
            slowness = tr.ray_p_s_to_km
            f_vel = iv.Forward_cal(vel_s= vel_syn, 
                           layers_thickness = lthickness, 
                           filt_list= self.filt_list,
                           kind = 'rftn',
                           gauss_par= self.gauss_filt,
                           dt=dt, nsamp=nsamp, tshift=10.0,
                           rf_method = self.rf_method,
                           slowness = slowness,
                           rf_normalize= self.rf_normalize,
                           inv_time_rf1= self.bf_p, 
                           inv_time_rf2= self.af_p,
                           noise_level = 20.0)
            # rf_r_with_noise = f_vel.rf_r_4_inv
            # rf_z_with_noise = f_vel.rf_z_4_inv
            rf_r_with_noise = f_vel.rf_r_from_ind1
            rf_z_with_noise = f_vel.rf_z_from_ind1
            app_curve_with_noise = f_vel.apparant_vel_org
            s_vel = iv.Forward_cal(vel_s= vel_syn, 
                           layers_thickness = lthickness, 
                           filt_list= self.filt_list,
                           kind = 'rftn',
                           gauss_par= self.gauss_filt,
                           dt=dt, nsamp=nsamp, tshift=10.0,
                           rf_method = self.rf_method,
                           slowness = slowness,
                           rf_normalize= self.rf_normalize,
                           inv_time_rf1= self.bf_p, 
                           inv_time_rf2= self.af_p,
                           noise_level = 0.0)
            # rf_r_without_noise = s_vel.rf_r_4_inv
            # rf_z_without_noise = s_vel.rf_z_4_inv
            rf_r_without_noise = s_vel.rf_r_from_ind1
            rf_z_without_noise = s_vel.rf_z_from_ind1
            app_curve_without_noise = s_vel.apparant_vel_org
            
            for el in strm:
                if ('RFR' == el.stats.channel):
                    el.data = rf_r_with_noise
                    el.rf_syn_without_noise = rf_r_without_noise
                    el.app_vel = app_curve_with_noise.copy()
                    el.app_vel_without_noise = app_curve_without_noise
                    el.vel_syn = vel_syn 
                    el.lthickness_syn = lthickness
                    el.stats.kind = 'rf_r_syn'
                elif ('RFZ' == el.stats.channel):
                    el.data = rf_z_with_noise
                    el.app_vel = app_curve_with_noise.copy()
                    el.vel_syn = vel_syn 
                    el.lthickness_syn = lthickness
                    el.rf_syn_without_noise = rf_z_without_noise
                    el.app_vel_without_noise = app_curve_without_noise
                    el.stats.kind = 'rf_z_syn'
        #finding syn_stack 
        self.rf_stack(kind= 'synthetic')
        self.find_app_vel_stack_syn()
#%%
def plotter_d(velocity, time_vec, layer_thickness_abs, rf, app_vel, filt_list,  
              save_name = 'model.png', header= 'Model'):
    subfig, ax_subfig = plt.subplots(nrows=1, ncols = 3,
                                     gridspec_kw={'width_ratios':[2, 1, 1]},
                                   figsize=(12, 8))
    
    (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
        ut.find_xminmax(layer_thickness_abs, velocity)



    ax_subfig[0].vlines(v_line_x, y_min, y_max, colors='black', lw = 3.0)
    ax_subfig[0].hlines(h_line_y, x_min, x_max,colors='black', lw = 3.0)
    ax_subfig[0].set_ylim([0, layer_thickness_abs[-2]])
    ax_subfig[0].set_ylim(ax_subfig[0].get_ylim()[::-1])
    ax_subfig[0].set_xlim([1.5, 5.5])
    ax_subfig[0].grid(True)
    
    ax_subfig[1].plot(rf,
                    time_vec, 
                    color = 'black', lw= 3)
    ax_subfig[1].set_ylim([max(time_vec), min(time_vec)])
    ax_subfig[1].grid(True)
    
    ax_subfig[2].plot(app_vel,
                    filt_list, 
                    color = 'black', lw= 3)
    ax_subfig[2].set_ylim(ax_subfig[1].get_ylim())
    ax_subfig[2].grid(True)
    subfig.suptitle(header)
    plt.savefig(save_name)
                
#%%
def smooth_vel(vel, derivative = 'second'):
     smooth_mat = np.zeros(shape=(len(vel), len(vel)))
     if (derivative == 'second'):
         for i in range(len(vel)):
             for j in range(len(vel)):
                 if ((i == j) and (j-1 >= 0) and (j+1) < len(vel)):
                     smooth_mat[i, j] = -2.0
                 elif ((i == j) and (j-1 > 0) and (j+1) == len(vel)):
                     smooth_mat[i, j] = -1.0
                 elif ((i == j) and (j-1 == -1) and (j+1) < len(vel)):
                     smooth_mat[i, j] = -1.0
                 elif ((i == j-1) or (i == j+1)):
                     smooth_mat[i, j] = 1.0
                 else:
                     smooth_mat[i, j] = 0.0
         smooth_mat = smooth_mat
     elif (derivative == 'first'):
         for i in range(len(vel) -1 ):
             for j in range(len(vel)):
                 if ((i==j)):
                     smooth_mat[i,j] = -1.0
                     smooth_mat[i, j+1]= 1.0
                 
         smooth_mat[len(vel)-1,len(vel)-1] = -1.0
         smooth_mat = smooth_mat
    
     vel_smoothed = vel + np.matmul(smooth_mat, vel) 
     return(vel_smoothed)
def cal_rf_energy(vel, thickness, rf, time_rf, weight):
    thickness_abs = []
    sumd = 0.0
    for el in thickness:
        sumd = sumd + el 
        thickness_abs.append(sumd)
    tps = cal_tps(vel, thickness)
    tppPs = cal_tppPs(vel, thickness)
    tppSs = cal_tppSs(vel, thickness)

    energy_tps = cal_energy(rf, time_rf, tps)
    energy_tppPs = cal_energy(rf, time_rf, tppPs)
    energy_tppSs = cal_energy(rf, time_rf, tppSs)
    
    energy_all = weight[0] * energy_tps + weight[1] * energy_tppPs -\
        weight[2] * energy_tppSs
    all_t = []
    for i in range(len(tps)):
        all_t.append([tps[i],tppPs[i], tppSs[i]])
    return (energy_all,  all_t)



def cal_energy(rf, time_rf, t):
    
    t_abs = []
    sumd= 0.0
    for el in t:
        sumd = sumd + el 
        t_abs.append(sumd)
    
    energy_all = []
    for el in t_abs:
        ttime = time_rf - el
        ind = np.argwhere(np.abs(ttime) == np.min(np.abs(ttime)))[0][0]
        energy_all.append(rf[ind])
    return(np.sum(energy_all))
    
    
def cal_tps(vel, thickness, slowness = 0.04, vp_to_vs = 1.732):
    tps = [] 
    idum = -1
    for el in vel:
        idum += 1
        numerator1 = (np.sqrt((el**-2.0) -
                              (slowness) ** 2.0))
        numerator2 = (np.sqrt(((vp_to_vs *el)**-2.0) -
                              (slowness) ** 2.0))
        tps.append(thickness[idum] * (numerator1 - numerator2))
    return(tps)
def cal_tppPs(vel, thickness, slowness = 0.04, vp_to_vs = 1.732):
    tppPs = [] 
    idum = -1
    for el in vel:
        idum += 1
        numerator1 = (np.sqrt((el**-2.0) -
                              (slowness) ** 2.0))
        numerator2 = (np.sqrt(((vp_to_vs *el)**-2.0) -
                              (slowness) ** 2.0))
        tppPs.append(thickness[idum] * (numerator1 + numerator2))
    return(tppPs)
def cal_tppSs(vel, thickness, slowness = 0.04):
    tppSs = [] 
    idum = -1
    for el in vel:
        idum += 1
        numerator1 = (np.sqrt((el**-2.0) -
                              (slowness) ** 2.0))
        tppSs.append(thickness[idum] * 2 * (numerator1))
    return(tppSs)
def cal_plot_time_thickness(vel, lthickness, rf, time_rf, 
                            bf = 1.0, af = 6.0, slowness = 0.04, 
                            only_change = False, vp_to_vs= 1.732):
    
    lthickness_abs = [] 
    sumd = 0.0
    for el in lthickness:
        sumd = sumd + el 
        lthickness_abs.append(sumd)
    
    ttd = time_rf + bf
    ind1 = np.argwhere(np.abs(ttd) == np.min(np.abs(ttd)))[0][0]
    
    ttd = time_rf - af 
    ind2 = np.argwhere(np.abs(ttd) == np.min(np.abs(ttd)))[0][0] 
    
    time_rf_4plot = time_rf[ind1:ind2].copy()
    rf_4plot = rf[ind1:ind2].copy()
    time_thickness_dif = []
    idum = -1
    for i in range(len(vel) -1 ):
        idum += 1
        numerator1 = (np.sqrt((vel[i]**-2.0) -
                              (slowness) ** 2.0))
        numerator2 = (np.sqrt(((vp_to_vs *vel[i])**-2.0) -
                              (slowness) ** 2.0))
        time_thickness_dif.append(lthickness[idum] * (numerator1 - numerator2))
    time_thickness_dif.append(0)
    
    time_thickness = [] 
    sumd = 0.0
    for el in time_thickness_dif:
        sumd = sumd + el 
        time_thickness.append(sumd)
    if (only_change):
        time_thickness_4plot = [] 
        vel_p = vel[0]
        for i in np.arange(1, len(vel)):
            vel_c = vel[i]
            if (np.abs(vel_c - vel_p) > 0.1):
                time_thickness_4plot.append(time_thickness[i])
            vel_p = vel_c
    else: 
        time_thickness_4plot = time_thickness.copy()
    
    plt.figure(figsize = (8,12))
    plt.plot(time_rf_4plot, rf_4plot, color= 'navy', label = 'RF')
    for el in time_thickness_4plot:
        plt.vlines(el, ymin = np.min(rf), ymax = 0.5)
    plt.legend()
    plt.grid(True)
    return(time_thickness)
    
def update_layer_thickness_ps(curr_vel, per_vel, 
                        per_lthickness, 
                        slowness = 0.04, vp_to_vs = 1.732):
    curr_lthickness = []
    for i in range(len(per_lthickness)- 1):
        numerator1 = (np.sqrt((per_vel[i]**-2.0) -
                              (slowness) ** 2.0))
        numerator2 = (np.sqrt(((vp_to_vs *per_vel[i])**-2.0) -
                              (slowness) ** 2.0))
        denominator1 =(np.sqrt((curr_vel[i]**-2.0) -
                               (slowness) ** 2.0))
        denominator2 = (np.sqrt(((vp_to_vs *curr_vel[i])**-2.0) -
                                (slowness) ** 2.0))
        numerator = numerator1 - numerator2 
        denominator = denominator1 - denominator2 
        
        curr_lthickness.append(per_lthickness[i] *
                               (numerator/ denominator))
    curr_lthickness.append(0)
    return(curr_lthickness)


def cal_time_thickness_PS(vel_s, layer_thickness, 
                       vp_to_vs = 1.732, slowness = 0.04):
    time_thickness = []
    for i in range(len(layer_thickness) -1):
        vp = vel_s[i] * vp_to_vs
        f_term = np.sqrt((vel_s[i] ** -2.0)  - 
                          (slowness ** 2.0))
        s_term = np.sqrt((vp ** -2.0) - 
                         (slowness ** 2.0))
        tthickness = layer_thickness[i] * (f_term + s_term)
        time_thickness.append(tthickness)
    time_thickness.append(0)
    return(time_thickness)
        
        
        
        
def update_layer_thickness_ppPs(curr_vel, per_vel, 
                        per_lthickness, 
                        slowness = 0.04, vp_to_vs = 1.732):
    curr_lthickness = []
    for i in range(len(per_lthickness)- 1):
        numerator1 = (np.sqrt((per_vel[i]**-2.0) -
                              (slowness) ** 2.0))
        numerator2 = (np.sqrt(((vp_to_vs *per_vel[i])**-2.0) -
                              (slowness) ** 2.0))
        denominator1 =(np.sqrt((curr_vel[i]**-2.0) -
                               (slowness) ** 2.0))
        denominator2 = (np.sqrt(((vp_to_vs *curr_vel[i])**-2.0) -
                                (slowness) ** 2.0))
        numerator = numerator1 + numerator2 
        denominator = denominator1 + denominator2 
        
        curr_lthickness.append(per_lthickness[i] *
                               (numerator/ denominator))
    curr_lthickness.append(0)
    return(curr_lthickness)
def update_layer_thickness_ppSs(curr_vel, per_vel, 
                        per_lthickness, 
                        slowness = 0.04):
    curr_lthickness = []
    for i in range(len(per_lthickness)- 1):
        numerator1 = (np.sqrt((per_vel[i]**-2.0) -
                              (slowness) ** 2.0))
        denominator1 =(np.sqrt((curr_vel[i]**-2.0) -
                               (slowness) ** 2.0))
        numerator = numerator1 
        denominator = denominator1 
        
        curr_lthickness.append(per_lthickness[i] *
                               (numerator/ denominator))
    curr_lthickness.append(0)
    return(curr_lthickness)  

    
    
    
    

def leg_p(n, x): 
    if(n == 0):
        return 1 # P0 = 1
    elif(n == 1):
        return x # P1 = x
    else:
        return (((2 * n)-1)*x * leg_p(n-1, x)-(n-1)*leg_p(n-2, x))/float(n)
def cal_rand_pert(z, zmax, pert_vel = 0.2):
    x = (z/zmax) 
    lp = leg_p(3, x)
    rand_part = pert_vel * np.random.normal()
    out = lp + rand_part
    # out = lp 
    return(out)
def printMatrix(a):
   print ("Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]")
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(rows):
      for j in range(cols):
         print( "%6.f" %a[i,j])
      print()
   print()     

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img
def merge_figures(fig_list,
                  out_name ='dummy.png'):
    img = []
    for fig in fig_list:
        img.append(fig2img(fig))
    height = img[0].height
    width = int((img[0].width) / 10)
    for el in img:
        width = width + el.width 
    bg_color = (255,255,255)
    merged_image = Image.new('RGB',(width + 15,height),color = bg_color)
    pre_width = 0
    for i in range(len(img)):
         img_to_p = img[i]
         merged_image.paste(img_to_p, (pre_width, 0))
         pre_width = img[i].width + pre_width + 15
    return(merged_image)

def _gauss_filter_shifted(dt, nft, f0, waterlevel=None,
                       tshift = 10):
    """
    Gaussian filter with width f0

    :param dt: sample spacing in seconds
    :param nft: length of filter in points
    :param f0: Standard deviation of the Gaussian Low-pass filter,
        corresponds to cut-off frequency in Hz for a response value of
        exp(0.5)=0.607.
    :param waterlevel: waterlevel for eliminating very low values
        (default: no waterlevel)
    :return: array with Gaussian filter frequency response
    """
    f = np.fft.fftfreq(nft, dt)
    gauss_arg = -0.5 * (f/f0) ** 2
    if waterlevel is not None:
        gauss_arg = np.maximum(gauss_arg, waterlevel)
    gauss_f = np.exp(gauss_arg)
    shift_filt = _phase_shift_filter(nft, dt, tshift)
    gauss_f = shift_filt * gauss_f
    return(gauss_f)

def _phase_shift_filter(nft, dt, tshift):
    """
    Construct filter to shift an array to account for time before onset

    :param nft: number of points for fft
    :param dt: sample spacing in seconds
    :param tshift: time to shift by in seconds
    :return: shifted array
    """
    freq = np.fft.fftfreq(nft, d=dt)
    return np.exp(-2j * np.pi * freq * tshift)


def cal_tan_phi(ray_p, vp, vs, ro1, ro2):
    dum1 = ray_p * ((ro2/vs**2.0) +
            (2.0 * ro1 * np.sqrt((1/vs**2.0)-ray_p**2.0)
             * np.sqrt((1/vp**2.0) - ray_p **2.0)))
    dum2 = ro1 * (np.sqrt((1/vp**2.0)- ray_p **2.0)
                  *((1/vs**2.0)- 2*ray_p**2.0))
    tan_phi = dum1/dum2
    return(tan_phi)
    
def define_event(stla, stlo, baz):
    R = 6378.1 #Radius of the Earth
    d= 45 * 110
    if (baz <= 180):
        az = baz +180
    else:
        az = baz - 180
    az = math.radians(az)
    lat1 = math.radians(stla)
    lon1 = math.radians(stlo)
    lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
         math.cos(lat1)*math.sin(d/R)*math.cos(az))
    lon2 = lon1 + math.atan2(math.sin(az)*math.sin(d/R)*math.cos(lat1),
                 math.cos(d/R)-math.sin(lat1)*math.sin(lat2))
    return(lat2, lon2)
def create_z_stream(stream_in, gauss_filt = 0.5):
    tr = stream_in[0]
    tr_org = tr.copy() 
    tr_r = tr.copy() 
    tr_z = tr_org.copy()
    tr_z.data = np.zeros(len(tr_r.data))
    tr_z.data[tr.onset_ind] = 1.0
    if (gauss_filt):
        nfft = next_fast_len(2 * len(tr_z.data))
        gaussF = ut._gauss_filter(dt, nfft, gauss_filt)
        r_flt = ut._apply_filter(tr_z.data, gaussF)
        tr_z.data = r_flt
        
    tr_z.stats.channel = 'Z'
    tr_stream = op.Stream()
    tr_stream.append(tr_r)
    tr_stream.append(tr_z)
    return(tr_stream)

def cal_deltat_kanamori(dz, vs, ray_p):
    vp = 1.732 * vs 
    d1 = np.sqrt(vs ** -2 + ray_p ** 2.0)
    d2 = np.sqrt(vp ** -2 + ray_p ** 2.0)
    dt = dz * (d1 - d2)
    return(dt)
def cal_depth_kanamori(t, vs, ray_p):
    vp = 1.732 * vs
    d1 = np.sqrt(vs ** -2 + ray_p ** 2.0)
    d2 = np.sqrt(vp ** -2 + ray_p ** 2.0)
    depth = t / (d1 - d2)
    return(depth)
def cosine_filt(period_filt, s_rate):
    time_filt_abs = period_filt + 1
    n_sample = round((period_filt+1) / s_rate)
    filt = np.zeros(n_sample)
    time_filt = np.arange(-time_filt_abs, time_filt_abs, s_rate)
    filt = np.zeros(len(time_filt))
    idum = -1
    for t in time_filt:
        idum += 1
        if (abs(t) < period_filt):
            filt[idum] = np.cos((np.pi/2.0) * (abs(t) / period_filt) ) ** 2.0
        else:
            filt[idum] = 0.0
    return(filt)

def cosine_filt2(trace, period_filt):
    # time_filt_abs = period_filt + 1
    n_sample = len(trace.data)
    filt = np.zeros(n_sample)
    time_filt = trace.time_vec
    filt = np.zeros(len(time_filt))
    idum = -1
    for t in time_filt:
        idum += 1
        if (abs(t) < period_filt):
            filt[idum] = np.cos((np.pi/2.0) * ((t) / period_filt) ) ** 2.0
        else:
            filt[idum] = 0.0
    return(filt)

def cosine_filt3(period_filt, s_rate, time_bf, time_af):
    # time_filt_abs = period_filt + 1
    n_sample = round((time_bf + time_af)/ s_rate)
    filt = np.zeros(n_sample)
    time_filt = np.arange(-time_bf, time_af, s_rate)
    filt = np.zeros(len(time_filt))
    idum = -1
    for t in time_filt:
        idum += 1
        if (abs(t) < period_filt):
            filt[idum] = np.cos((np.pi/2.0) * (abs(t) / period_filt) ) ** 2.0
        else:
            filt[idum] = 0.0
    return(filt)

def find_amp(rst, per):
    time_vec = rst[0].time_vec
    rrf = rst[0].data
    zrf = rst[1].data
    out_vec_r = []
    out_vec_z = []
    if (per > 0):
        idum = -1
        for el in time_vec:
            idum += 1
            if (abs(el) <= per):
                out_vec_r.append(rrf[idum] * 
                                 np.cos((np.pi / 2.0) * el / per)**2.0)
                out_vec_z.append(zrf[idum] * 
                                 np.cos((np.pi / 2.0) * el / per)**2.0)
            else:
                out_vec_r.append(0.0)
                out_vec_z.append(0.0)
    else:
        idum = -1
        for i in range(len(rrf)):
            out_vec_r.append(rrf[i] * zrf[i])
            out_vec_z.append(zrf[i] * zrf[i])
            
        
    y = sum(out_vec_r)
    x = sum(out_vec_z)
    amp_zero = y / x
        
    return(amp_zero)
    
def normalize(sig):
    normed_sig = np.zeros(shape=len(sig))
    idum=-1
    for el in sig:
        idum +=1 
        normed_sig[idum] = el/max(sig)
    return(normed_sig)
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def convolve_filt(sig,filt, normalize = 0):
    filt_max_ind = np.argwhere(filt == max(filt))[0][0]
    af_conv_full = np.convolve(sig, filt, 'full')
    what_to_remove_b = filt_max_ind 
    what_to_remove_e = len(af_conv_full) -(len(filt) - filt_max_ind) +1 
    filtered_sig = af_conv_full[what_to_remove_b:what_to_remove_e]
    if (normalize == 1):
        idum = -1
        for el in filtered_sig:
            idum += 1
            filtered_sig[idum] = el / max(filtered_sig)
            
    return(filtered_sig)

# def filter_sig(sig, period, filter_type = 'cosine', method = 'frequency'):
def get_filter_time(tr, period, kind= 'butterworth', 
                    corner = 2, zerophase= True):
    z_tr = tr.copy()
    z_tr.data = np.zeros(len(tr.data))
    z_tr.data[tr.onset_ind] = 1.0
    if (kind == 'butterworth'):
        z_tr.filter('lowpass',corners=corner,freq = 1/period, 
                zerophase = zerophase)
    else:
        z_tr.filter('lowpass_cheby_2',freq = 1/period)
    return(z_tr.data)

def find_onset_amp(tr, dist_range = (25,95) ,
                   bf_p = 10, af_p = 60):
    
    
    
    # i did this in makran preprocesses 
    #====================================
    # if (tr.stats.sac.nzjday == 0):
    #     tr.stats.sac.nzjday= 365
    # refrence_time = UTCDateTime(year = tr.stats.sac.nzyear,
    #                             julday = tr.stats.sac.nzjday,
    #                             hour= tr.stats.sac.nzhour, 
    #                             minute= tr.stats.sac.nzmin,
    #                             second = tr.stats.sac.nzsec, 
    #                             microsecond= tr.stats.sac.nzmsec )
    # tr.stats.refrence_time = refrence_time
    
    # if ('user9' in tr.stats.sac):
    #     tr.stats.sac.mag = tr.stats.sac.user9
    # dist_m, baz, az = (op.geodetics
    #                    .base.calc_vincenty_inverse
    #                    (tr.stats.sac.stla,tr.stats.sac.stlo,
    #                     tr.stats.sac.evla,tr.stats.sac.evlo))
    # tr.distance_km = dist_m / 1000.0
    # tr.distance_deg = op.geodetics.base.kilometer2degrees(tr.distance_km)
    #====================================
    tr.distance_deg = tr.stats.distance
    if (tr.stats.sac.evdp > 1000):
        tr.stats.sac.evdp = tr.stats.sac.evdp / 1000.0
    if (tr.distance_deg > dist_range[0]) & (tr.distance_deg < dist_range[1]):
        
        model = taup.TauPyModel(model="iasp91")
        arrival_tr = model.get_pierce_points(tr.stats.sac.evdp,
                                         tr.distance_deg, 'P')
        ray_p_s_to_km = (arrival_tr[0].ray_param_sec_degree
                     / op.geodetics.base.degrees2kilometers(1))
        t_inc_ang = arrival_tr[0].incident_angle
        tr.t_inc_ang = t_inc_ang
        tr.ray_p_s_to_km = ray_p_s_to_km
        if hasattr(tr.stats.sac, 't1'):
            tr.stats.P_onset = tr.stats.starttime + 20
        else:
            tr.stats.P_onset = tr.stats.event_time + arrival_tr[0].time
        find_snr(tr, tr.stats.P_onset)
        tr.trim(tr.stats.P_onset - bf_p, tr.stats.P_onset + af_p- tr.stats.delta)
    else:  
        tr.t_inc_ang = 'Null'
        tr.ray_p_s_to_km = 'Null'
        return(False)
    if (tr.stats.sac.b < 0):
        time_vec = np.arange(tr.stats.sac.b,
                            tr.stats.sac.e,
                            tr.stats.delta)
    else:
        # begin = tr.stats.starttime - tr.stats.P_onset 
        # end = tr.stats.endtime - tr.stats.P_onset
        # time_vec = np.arange(begin,
        #                     end,
        #                     tr.stats.delta)
        time_vec = tr.times() - bf_p
    time_vec_UTC_d = []
    for el in time_vec:
        time_vec_UTC_d.append(tr.stats.P_onset + el)
    time_vec_UTC = np.array(time_vec_UTC_d)
    onset_ind = np.argwhere(abs(time_vec) == min(abs(time_vec)))[0][0]
    tr.time_vec = time_vec
    tr.time_vec_UTC = time_vec_UTC
    tr.onset_ind = onset_ind
    
    
    return(True)
    # r_amp_zero = tr.data[onset_ind]
    # tr.rrf_amp_zero = r_amp_zero

    
def find_snr(tr, ref_time, noise= [10,5], 
             signal = [2,20]):
    tr_n = copy.deepcopy(tr) 
    tr_s = copy.deepcopy(tr) 
    tr_n.trim(ref_time - noise[0], ref_time - noise[1])
    tr_s.trim(ref_time - signal[0], ref_time + noise[1])
    
    noise_data = tr_n.data 
    signal_data = tr_s.data 
    
    mean_noise = np.mean(noise_data)
    mean_signal = np.mean(signal_data)
    
    
    rms_n = np.std(noise_data)
    # rms_s = (1/ len(signal_data)) * np.sum((signal_data - mean_signal)**2.0)
    rms_s = np.std(signal_data)
    
    tr.stats.snr = rms_s/ rms_n
    
def find_onset_amp_strm(strm, dist_range = (25,95), time_utc = False):
    
    for el in strm:
        if ('Z' in el.stats.channel):
            tr = el.copy()
    
    refrence_time = op.UTCDateTime(year = tr.stats.sac.nzyear,
                                julday = tr.stats.sac.nzjday,
                                hour= tr.stats.sac.nzhour, 
                                minute= tr.stats.sac.nzmin,
                                second = tr.stats.sac.nzsec, 
                                microsecond= tr.stats.sac.nzmsec )
    tr.stats.refrence_time = refrence_time
    
    if ('user9' in tr.stats.sac):
        tr.stats.sac.mag = tr.stats.sac.user9
    if ('kuser2' in tr.stats.sac):
        tr.stats.sac.mag = tr.stats.sac.user2
    dist_m, baz, az = (op.geodetics
                       .base.calc_vincenty_inverse
                       (tr.stats.sac.stla,tr.stats.sac.stlo,
                        tr.stats.sac.evla,tr.stats.sac.evlo))
    tr.distance_km = dist_m / 1000.0
    tr.distance_deg = op.geodetics.base.kilometer2degrees(tr.distance_km)
    if (tr.stats.sac.evdp > 1000):
        tr.stats.sac.evdp = tr.stats.sac.evdp / 1000.0
    if (tr.distance_deg > dist_range[0]) & (tr.distance_deg < dist_range[1]):
        
        model = taup.TauPyModel(model="iasp91")
        arrival_tr = model.get_pierce_points(tr.stats.sac.evdp,
                                         tr.distance_deg, 'P')
        ray_p_s_to_km = (arrival_tr[0].ray_param_sec_degree
                     / op.geodetics.base.degrees2kilometers(1))
        t_inc_ang = arrival_tr[0].incident_angle
        tr.t_inc_ang = t_inc_ang
        tr.ray_p_s_to_km = ray_p_s_to_km
        tr.stats.P_onset = tr.stats.event_time + arrival_tr[0].time
    else:
        tr.t_inc_ang = 'Null'
        tr.ray_p_s_to_km = 'Null'
        for el in strm:
            el.stats.refrence_time = tr.stats.refrence_time
            el.stats.sac.mag = tr.stats.sac.mag 
            el.distance_km = tr.distance_km
            el.distance_deg = tr.distance_deg
            el.t_inc_ang = tr.t_inc_ang
            el.ray_p_s_to_km = tr.ray_p_s_to_km
        return
    if (tr.stats.sac.b < 0):
        time_vec = np.arange(tr.stats.sac.b,
                            tr.stats.sac.e,
                            tr.stats.delta)
    else:
        begin = tr.stats.starttime - tr.stats.P_onset 
        end = tr.stats.endtime - tr.stats.P_onset
        time_vec = np.arange(begin,
                            end,
                            tr.stats.delta)
    if (time_utc):
        time_vec_UTC_d = []
        for el in time_vec:
            time_vec_UTC_d.append(tr.stats.P_onset + el)
            time_vec_UTC = np.array(time_vec_UTC_d)
            tr.time_vec_UTC = time_vec_UTC
            
    onset_ind = np.argwhere(abs(time_vec) == min(abs(time_vec)))[0][0]
    tr.time_vec = time_vec
    tr.onset_ind = onset_ind
    r_amp_zero = tr.data[onset_ind]
    tr.rrf_amp_zero = r_amp_zero
    for el in strm:
        el.stats.refrence_time = tr.stats.refrence_time
        el.stats.sac.mag = tr.stats.sac.mag 
        el.distance_km = tr.distance_km
        el.distance_deg = tr.distance_deg
        el.t_inc_ang = tr.t_inc_ang
        el.ray_p_s_to_km = tr.ray_p_s_to_km
        el.time_vec = tr.time_vec
        if (time_utc):
            el.time_vec_UTC = tr.time_vec_UTC
        el.onset_ind = tr.onset_ind
        el.amp_zero = el.data[el.onset_ind]
def filter_trace(tr, period_filt, filt_kind = 'cosine', 
                 corner = 2,
                 method_filt = 'frequency',
                 zerophase = True,
                 plot_fig = 'both',
                 out_dir = os.getcwd() + '/out_filt_fig' ):
    dt = tr.stats.delta
    sig = tr.data.copy()
    dummy_tr = tr.copy()
    dummy_tr.data = np.zeros(len(tr.data))
    dummy_tr.data[tr.onset_ind] = tr.data[tr.onset_ind]
    t=tr.time_vec
    if (filt_kind == 'cosine'):
        filt = cosine_filt2(tr, period_filt)
    else:
        filt = get_filter_time(tr, period_filt, kind= filt_kind
                               ,corner= corner, 
                               zerophase= zerophase)
    if (method_filt == 'frequency') or (plot_fig == 'both'):
        n = len(t)
        sig_fft = np.fft.fft(sig,n)
        filt_fft = np.fft.fft(filt,n)
        freq = (1/(dt*n))* np.arange(n)
        L = np.arange(1,np.floor(n/2),dtype='int')
        filtered_signal = sig_fft * filt_fft
        filtered_signal_time = np.fft.ifft(filtered_signal)
        filtered_signal_time = np.real(filtered_signal_time)
        filtered_signal_timed = np.zeros(len(filtered_signal_time))
        sh = tr.onset_ind
        idum = -1
        for i in range(len(filtered_signal_time)):
            idum += 1
            if (idum + sh < len(filtered_signal_time)):
                ind = idum +sh
            else:
                ind = (idum + sh) - len(filtered_signal_time)
            filtered_signal_timed[i] =  filtered_signal_time[ind]
            
        if (plot_fig == True) or (plot_fig == 'both'):
            
            if (os.path.isdir(out_dir)):
                pass
            else:
                os.mkdir(out_dir)
                
            fig, axes= plt.subplots(figsize=(8,12), nrows=3, 
                                    ncols=2, dpi = 150)
            # plt.subplot(6,1,1)
            axes[0,0].set_title('Signal in time domain')
            axes[0,0].plot(t,sig,color='k', linewidth = 1.5, label='Sig')
            axes[0,0].set_xlim(t[0], t[-1])
            
            
            # plt.subplot(6,1,2)
            axes[0,1].set_title(filt_kind +
                                ' filter in time domain with period ' 
                      +str(period_filt)+'\n'+'(frequency = '+
                      str(1/period_filt)+')')
            axes[0,1].plot(t,filt,color='k', linewidth = 2, label='Filter')
            axes[0,1].set_xlim(t[0], t[-1])

            
            

            # plt.subplot(6,1,3)
            axes[1,0].set_title('Signal in frequency domain')
            axes[1,0].plot(freq[L],abs(sig_fft)[L],color='c', linewidth = 2, 
                     label='Signal in freq')
            axes[1,0].set_xlim(freq[L[0]], freq[L[-1]])

            # plt.subplot(6,1,4)
            axes[1,1].set_title(filt_kind +
                                ' filter in frequency domain with period ' 
                      +str(period_filt) +'\n'+'(frequency = '+
                      str(1/period_filt)+')')
            axes[1,1].plot(freq[L],abs(filt_fft)[L],color='c', linewidth = 2,
                     label='Filter in freq')
            axes[1,1].set_xlim(freq[L[0]], freq[L[-1]])


            # plt.subplot(6,1,5)
            axes[2,0].set_title('Filtered signal in frequency domain')
            axes[2,0].plot(freq[L],abs(filtered_signal[L]),color='c', linewidth = 2, 
                     label='Noisy')
            axes[2,0].set_xlim(freq[L[0]], freq[L[-1]])

            # plt.subplot(6,1,1)
            axes[2,1].set_title('Filtered signal in time domain')
            axes[2,1].plot(t,filtered_signal_timed,color='k', linewidth = 2,
                     label='Filter')
            axes[2,1].set_xlim(t[0], t[-1])
            plt.tight_layout(pad=.8)
            plt.savefig(out_dir+
                        'Filter_in_freq_domain' +str(period_filt) +'_'+ 
                        filt_kind +
                        '_frequency.png')
            if (plot_fig == 'both'):
                filtered_signal_time2 = convolve_filt(sig, 
                                                         filt, normalize= 0)
                dif = filtered_signal_time2 - filtered_signal_timed
                
                fig2, axes2= plt.subplots(figsize=(8,12), nrows=3, 
                                        ncols=1, dpi = 300)
                # plt.subplot(3,1,1)
                axes2[0].set_title('Filtered signal using fft in time domain ')
                axes2[0].plot(t,filtered_signal_timed,color='k', linewidth = 2, 
                         label='Filter')
                axes2[0].set_xlim(t[0], t[-1])
                
                # plt.subplot(3,1,2)
                axes2[1].set_title('Filtered signal using conv in time domain ')
                axes2[1].plot(t,filtered_signal_time2,color='k', linewidth = 2, 
                         label='Filter')
                axes2[1].set_xlim(t[0], t[-1])
                
                # plt.subplot(3,1,3)
                axes2[2].set_title('difference of Filtered'+
                          ' signal using conv and fft in time domain ')
                axes2[2].plot(t,dif,color='k', linewidth = 2, 
                         label='Filter')
                axes2[2].set_xlim(t[0], t[-1])
                plt.savefig(out_dir+
                            'dif_Filter_in_freq_domain' +str(period_filt)
                            +'_'+ filt_kind +
                            '_frequency.png')

    elif(method_filt == 'time'):
        filtered_signal_timed = convolve_filt(sig, filt, normalize= 0)
        
        if (plot_fig):
            if (os.path.isdir(out_dir)):
                pass
            else:
                os.mkdir(out_dir)
            plt.figure()
            plt.subplot(3,1,1)
            plt.title('Signal in time domain')
            plt.plot(t,sig,color='c', linewidth = 1.5, label='Sig')
            plt.xlim(t[0], t[-1])
            plt.subplot(3,1,2)
            plt.title(filt + ' filter in time domain with period ' 
                      +str(period_filt))
            plt.plot(t,filt,color='k', linewidth = 2, label='Filter')
            plt.xlim(t[0], t[-1])
            
            plt.subplot(3,1,3)
            plt.title('Filtered signal in time domain')
            plt.plot(t,filtered_signal_timed,color='k', linewidth = 2, label='Filter')
            plt.xlim(t[0], t[-1])
            plt.tight_layout(pad=.8)
            plt.savefig(out_dir+
                        'Filter_in_time_domain' +str(period_filt) +'_'+ 
                        filt_kind +
                        '_frequency.png')
    dummy_tr.data = filtered_signal_timed
            
    return(dummy_tr)
def filter_stream(stream_in, period_filt, filt_kind = 'cosine', 
                 corner = 2,
                 method_filt = 'frequency',
                 normalize = True,
                 zerophase = True,
                 plot_fig = 'both',
                 out_dir = os.getcwd() + '/out_filt_fig' ):
    stream_out = op.Stream()
    for el in stream_in:
        if (period_filt > 0.0):
            el2 = filter_trace(el, period_filt, filt_kind = filt_kind, 
                         corner = corner,
                         method_filt = method_filt,
                         zerophase = zerophase,
                         plot_fig = plot_fig,
                         out_dir = out_dir)
            stream_out.append(el2)
            if (normalize):
                stream_out.normalize()
        else:
            stream_out.append(el)
    return(stream_out)

def create_z_stream_wg(stream_in, gauss_filt= 0.5):
    for el in stream_in:
        if ('R' in el.stats.channel):
            tr = el.copy()
        if ('Z' in el.stats.channel):
            tr_z_rf = el
    tr_org = tr.copy() 
    tr_r = tr.copy() 
    tr_z = tr_org.copy()
    tr_z.data = np.zeros(len(tr_r.data))
    tr_z.data[tr.onset_ind] = 1.0
    if (gauss_filt):
        nfft = next_fast_len(2 * len(tr_z.data))
        dt = 1./ tr_r.stats.sampling_rate
        gaussF = ut._gauss_filter(dt, nfft, gauss_filt)
        # tshift = tr_r.stats.onset - tr_r.stats.starttime
        # shift_filt = _phase_shift_filter(nfft, dt, tshift= tshift)
        spec_src = fft(tr_z.data, nfft)
        
        r_flt = ifft(spec_src * gaussF)[:len(tr_z.data)]
        r_flt = r_flt * (1./ max(r_flt))
        r_flt = r_flt.real
        tr_z.data = r_flt 
    tr_z.stats.channel = 'BHZ'
    #for waterlevel thing 
    tr_z = tr_z_rf
    tr_stream = rf.RFStream()
    tr_stream.append(tr_r)
    tr_stream.append(tr_z)
    # print(max(stream_in[0].data - tr_z.data), 'here i am')
    return(tr_stream)
def find_velocity_from_strm(strm_input):
    for el in strm_input:
        if ('R' in el.stats.channel):
            tr = el.copy()
    strm_out = op.Stream()
    theta_d = (strm_input[0].data[tr.onset_ind] /
               strm_input[1].data[tr.onset_ind])
    theta_rad = np.arctan(theta_d)
    theta_degree = theta_rad * 180 / np.pi 
    v_s = np.sin(theta_rad/ 2.0) / tr.ray_p_s_to_km
    for el in strm_input:
        el.amp_zero = el.data[tr.onset_ind]
        el.theta_rad = theta_rad
        el.theta_deg = theta_degree
        el.polarazation_rad = theta_rad / 2.0
        el.polarazation_deg = theta_degree / 2.0
        el.vel_s = v_s
        strm_out.append(el)
    return(strm_out)
        
#%% 
def interp_velocity(lthickness, velocity, 
                    lthickness_abs_input):
    
    lthickness_abs = []
    sumd = 0
    for el in lthickness:
        sumd = sumd + el 
        lthickness_abs.append(sumd)
    vel_syn_interpolated = np.interp(np.array(lthickness_abs_input),
                          np.array(lthickness_abs),
                          np.array(velocity))
    return(vel_syn_interpolated)

def interp_velocity_layer_ref(lthickness, velocity, 
                    tthickness_abs_input):
    vs_interp = interp_velocity(lthickness= lthickness, 
                                velocity= velocity, 
                        tthickness_abs_input= tthickness_abs_input)
    
    lthickness_abs = []
    sumd = 0
    for el in lthickness:
        sumd = sumd + el 
        lthickness_abs.append(sumd)
    
    vel_syn_interpolated = []
    flayer = 0.0
    for i in range(len(tthickness_abs_input)):
        slayer = tthickness_abs_input[i]
        vel_layer = []
        for j in range(len(lthickness_abs)):
            if ((lthickness_abs[j] >= flayer) and
                (lthickness_abs[j] < slayer)):
                vel_layer.append(velocity[j])
        if (len(vel_layer) == 0):
            vel_syn_interpolated.append(-1)
        elif (len(vel_layer) == 1):
            vel_syn_interpolated.append(vel_layer[0])
        else:
            vel_syn_interpolated.append(np.mean(vel_layer))
        flayer = slayer
    
    
    vel_syn_interpolated = np.array(vel_syn_interpolated)
    for i in range(len(vel_syn_interpolated)):
        if (vel_syn_interpolated[i] == -1):
            vel_syn_interpolated[i] = vs_interp[i]
    return(vel_syn_interpolated)

def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))
def read_custom_rf(fl_name, coor):
    sample_tr_flr = "/home/soroush/rf_shallow_codes/my_py_rf/sample_rfr"
    sample_tr_flz = "/home/soroush/rf_shallow_codes/my_py_rf/sample_rfz"
    with open(sample_tr_flr, 'rb') as f1:
                sample_tr_r = pickle.load(f1)
    with open(sample_tr_flz, 'rb') as f1:
                sample_tr_z = pickle.load(f1)
    for name, lat, lon in coor:
        if (name in fl_name):
            sample_tr_r.stats.station = name 
            sample_tr_r.ray_p_s_to_km = 0.06 
            sample_tr_r.stats.station_longitude = lon 
            sample_tr_r.stats.station_latitude = lat 
            sample_tr_r.stats.stack_kind = "Harmonic"
            sample_tr_r.stats.RF_gauss = 3 / (2 * np.pi)
            
            
            
            sample_tr_z.stats.station = name 
            sample_tr_z.ray_p_s_to_km = 0.06 
            sample_tr_z.stats.station_longitude = lon 
            sample_tr_z.stats.station_latitude = lat 
            sample_tr_z.stats.stack_kind = "Harmonic"
            sample_tr_z.stats.RF_gauss = 3 / (2 * np.pi)

    fl_content = []
    with open(fl_name, 'r') as f:
        for l in f:
            fl_content.append(l.split())
    f.close()


    time = []
    harmonic_rfr = []
    for line in fl_content:
        time.append(float(line[0]))
        harmonic_rfr.append(float(line[1]))

    delta = time[1] - time[0]
    time.insert(0, time[0] - delta)
    time.insert(0, time[0] - delta)
    harmonic_rfr.insert(0, 0)
    harmonic_rfr.insert(0, 0)
    harmonic_rfr_virg = np.array(harmonic_rfr)

    ind_stop_time = np.argwhere(sample_tr_r.time_vec == 20.0)[0][0]
    time_fin_trace = sample_tr_r.time_vec[:ind_stop_time+1]
    harmonic_rfr = np.interp(np.array(time_fin_trace),
                         np.array(time),
                         np.array(harmonic_rfr_virg))

    harmonic_strm = op.Stream()
    harmonic_strm.append(sample_tr_r)
    harmonic_strm.append(sample_tr_z)
    for tr in harmonic_strm:
        s_time = tr.stats.starttime 
        e_time = tr.stats.starttime + 21.0
        tr.trim(starttime= s_time, 
                      endtime = e_time)
        tr.time_vec = time_fin_trace.copy()
        if (tr.stats.channel == 'RFR'):
            tr.data = harmonic_rfr.copy()
    harmonic_rfr_fixed, is_it_good = \
        check_for_samp_shift(harmonic_rfr, harmonic_strm[1].data)
    harmonic_strm[0].data = harmonic_rfr_fixed.copy()
    
    return harmonic_strm
def check_for_samp_shift(dr, dz):
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
        for i in np.arange(0, ind_zero_z):
            if (dr_fixed[i] < 0):
                dr_fixed[i] = 0
        
        ind_zero_r = (np.argwhere((dr_fixed) ==
                                np.max((dr_fixed)))[0][0])
        if (ind_zero_z == ind_zero_r):
            is_it_good = True
        
        return(dr_fixed, is_it_good)
    else:
        return(dr, is_it_good)
def objective_harmonic_r(phi, a, b, c, d, e):
    phi_rad = (np.pi / 180) * phi
    on = (a + 
          b* np.cos(phi_rad) + 
          c * np.sin(phi_rad) + 
          d * np.cos(2 * phi_rad) +
          e * np.sin(2 * phi_rad))
    return on
def objective_harmonic_t(phi, b, c, d, e):
    phi_rad = (np.pi / 180) * phi
    on = (b* (np.cos(phi_rad) + (np.pi/ 2))+ 
          c * (np.sin(phi_rad) + (np.pi/ 2)) + 
          d * (np.cos(2 * phi_rad) + (np.pi/ 2)) +
          e * (np.sin(2 * phi_rad) + (np.pi/ 2)))
    return on
##stack function from obspy
def stack(data, stack_type='linear'):
    """
    Stack data by first axis.

    :type stack_type: str or tuple
    :param stack_type: Type of stack, one of the following:
        ``'linear'``: average stack (default),
        ``('pw', order)``: phase weighted stack of given order
        (see [Schimmel1997]_, order 0 corresponds to linear stack),
        ``('root', order)``: root stack of given order
        (order 1 corresponds to linear stack).
    """
    if stack_type == 'linear':
        stack = np.mean(data, axis=0)
    elif stack_type[0] == 'pw':
        from scipy.signal import hilbert
        from scipy.fftpack import next_fast_len

        npts = np.shape(data)[1]
        nfft = next_fast_len(npts)
        anal_sig = hilbert(data, N=nfft)[:, :npts]
        norm_anal_sig = anal_sig / np.abs(anal_sig)
        phase_stack = np.abs(np.mean(norm_anal_sig, axis=0)) ** stack_type[1]
        stack = np.mean(data, axis=0) * phase_stack
    elif stack_type[0] == 'root':
        r = np.mean(np.sign(data) * np.abs(data)
                    ** (1 / stack_type[1]), axis=0)
        stack = np.sign(r) * np.abs(r) ** stack_type[1]
    else:
        raise Exception('stack type is not valid.')
    return stack
def cal_abs_thicknes_from_l_thickness(l_thickness):
    abs_thickness = []
    sumd = 0.0
    for el in l_thickness:
        sumd = sumd + el
        abs_thickness.append(sumd)
    return(np.array(abs_thickness))
def interpolate_velocity(abs_thickness_vel, vel, abs_thickness_ref):
    vel_out = np.interp(np.array(abs_thickness_ref),
                         np.array(abs_thickness_vel),
                         np.array(vel))
    return(vel_out)
def sort_objective(objective_dict,par_sort = 'dif_vel_syn'):
    a_list = objective_dict[par_sort].copy()
    indices = sorted(range(len(a_list)), 
                     key=lambda index: a_list[index])
    objective_dict_sorted = objective_dict.copy()
    for key in objective_dict.keys():
        objective_dict_sorted[key] = []
        for ind in indices:
            objective_dict_sorted[key].append(objective_dict[key][ind])
            
    return(objective_dict_sorted)
def find_harmonic_joint(all_rf_r_bins,
                    all_rf_t_bins, all_rf_phi_bins):
    jacob_harmonic = []
    for phi in all_rf_phi_bins:
        phi_rad = (np.pi / 180) * phi
        jacob_harmonic.append([1, np.cos(phi_rad), np.sin(phi_rad), 
                               np.cos(2 * phi_rad), np.sin(2 * phi_rad)])
    for phi in all_rf_phi_bins:
        phi_rad = (np.pi / 180) * phi
        jacob_harmonic.append([0, np.cos(phi_rad) + (np.pi/2), 
                               np.sin(phi_rad) + (np.pi/2), 
                               np.cos(2 * phi_rad) + (np.pi/2), 
                               np.sin(2 * phi_rad) + (np.pi/2)])
    jacob_harmonic = np.array(jacob_harmonic)
    # a_amp = []
    # b_amp = []
    # c_amp = []
    # d_amp = []
    # e_amp = []
    x_out = []
    for i in range(len(all_rf_r_bins[0, :])):
        b = []
        for amp_r in all_rf_r_bins[:, i]:
            b.append(amp_r)
        for amp_t in all_rf_t_bins[:, i]:
            b.append(amp_t)
        b = np.array(b)
        ata = np.matmul(np.transpose(jacob_harmonic), jacob_harmonic)
        inv_ata = np.linalg.inv(ata)
        atb = np.matmul(np.transpose(jacob_harmonic), b)
        x = np.matmul(inv_ata, atb)
        x_out.append(x)
    x_out = np.array(x_out)
    a_amp = x_out[:, 0]
    b_amp = x_out[:, 1]
    c_amp = x_out[:, 2]
    d_amp = x_out[:, 3]
    e_amp = x_out[:, 4]
    return(a_amp, b_amp, c_amp, d_amp, e_amp)
        
    
def is_digit(var):
    try:
        a = float(var)
        return(True)
    except ValueError:
        return(False)
#%%
import matplotlib.pyplot as plt
import numpy as np
import warnings


def fftPlot(sig, dt=None, plot=True, title= None):
    # Here it's assumes analytic signal (real signal...) - so only half of the axis is required

    if dt is None:
        dt = 1
        t = np.arange(0, sig.shape[-1])
        xLabel = 'samples'
    else:
        t = np.arange(0, sig.shape[-1]) * dt
        xLabel = 'freq [Hz]'

    if sig.shape[0] % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...")
        t = t[0:-1]
        sig = sig[0:-1]

    sigFFT = np.fft.fft(sig) / t.shape[0]  # Divided by size t for coherent magnitude

    freq = np.fft.fftfreq(t.shape[0], d=dt)

    # Plot analytic signal - right half of frequence axis needed only...
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[0:firstNegInd]
    sigFFTPos = 2 * sigFFT[0:firstNegInd]  # *2 because of magnitude of analytic signal

    if plot:
        plt.figure(figsize=(16,8))
        plt.plot(freqAxisPos, np.abs(sigFFTPos))
        plt.xlabel(xLabel)
        # plt.xscale("log")
        plt.xticks(np.arange(0, 10, 1))
        plt.ylabel('mag')
        plt.grid(True)
        if (title == None):
            plt.title('Analytic FFT plot')
        else:
            plt.title(title)
        # plt.show()
        plt.tight_layout()
        plt.savefig(title+'.png', dpi = 300)

    return sigFFTPos, freqAxisPos


if __name__ == "__main__":
    dt = 1 / 1000

    # Build a signal within Nyquist - the result will be the positive FFT with actual magnitude
    f0 = 200  # [Hz]
    t = np.arange(0, 1 + dt, dt)
    sig = 1 * np.sin(2 * np.pi * f0 * t) + \
        10 * np.sin(2 * np.pi * f0 / 2 * t) + \
        3 * np.sin(2 * np.pi * f0 / 4 * t) +\
        7.5 * np.sin(2 * np.pi * f0 / 5 * t)

    # Result in frequencies
    fftPlot(sig, dt=dt)
    # Result in samples (if the frequencies axis is unknown)
    fftPlot(sig)
        
            
#%%
def find_layering(init_boundary, vel_info, layer_info, vp_to_vs):
    vel_param = ut.Vel_paramterize(vel_info = vel_info, 
                                        layer_info = layer_info, 
                                        vp_to_vs= vp_to_vs)
    lthick_abs = copy.deepcopy(vel_param.layer_thickness_abs)
    
    layering = []
    layering_perv = 0
    for el in init_boundary:
        lthick_abs_d = copy.deepcopy(lthick_abs)
        lthick_abs_d = np.array(lthick_abs_d) - el
        ind = np.argwhere(lthick_abs_d <= 0)
        layering_cur = len(ind) - layering_perv
        layering_perv = layering_cur + layering_perv
        layering.append(layering_cur)
    
    lthick_abs_d = copy.deepcopy(lthick_abs)
    lthick_abs_d = np.array(lthick_abs_d) - lthick_abs[-2]
    ind = np.argwhere(lthick_abs_d <= 0)
    layering_cur = len(ind) - layering_perv
    layering.append(layering_cur)

    return(layering)

def get_model_list(param):
    if (param == 'halfspace'):
        model = [['MIN_DEPTH', 'MAX_DEPTH', 'VELOCITY', 'MAX_DEPTH_CHANGE_MIN', 
                  'MAX_DEPTH_CHANGE_MAX', 'VELOCITY_CHANGE'], ['0.0', '15.0', 
                  '3.0', '-5', '5', '0.6'], ['15.0', '40.0', '3.0', '-5', '5', 
                  '0.6'], ['40.0', '120.0', '4.1', '0', '0', '0.4']]
    elif (param == 'sedimentary'):
        model = [['MIN_DEPTH', 'MAX_DEPTH', 'VELOCITY', 'MAX_DEPTH_CHANGE_MIN',
                  'MAX_DEPTH_CHANGE_MAX', 'VELOCITY_CHANGE'], ['0.0', '3.0', 
                  '2.5', '0', '0', '0.5'], ['3.0', '10.0', '3.0', '-2', '5', 
                  '0.5'], ['10.0', '35.0', '3.5', '-5', '5', '0.6'], ['35.0', 
                  '40.0', '3.5', '0', '0', '0.6'], ['40.0', '90.0', '4.1', '0',
                        '0', '0.6'], ['90.0', '120.0', '4.2', '0', '0', '0.6']]
    elif (param == 'continent'):
        model = [['MIN_DEPTH', 'MAX_DEPTH', 'VELOCITY', 'MAX_DEPTH_CHANGE_MIN', 
                 'MAX_DEPTH_CHANGE_MAX', 'VELOCITY_CHANGE'], ['0.0', '20.0', 
                '3.36', '-5', '5', '0.5'], ['20.0', '40.0', '3.75', '-5', '8',
                '0.6'], ['40.0', '120.0', '4.47', '0', '0', '0.6']]
    elif (param == 'iasp91'):
        model = [['MIN_DEPTH', 'MAX_DEPTH', 'VELOCITY', 'MAX_DEPTH_CHANGE_MIN',
                  'MAX_DEPTH_CHANGE_MAX', 'VELOCITY_CHANGE'], ['0.0', '20.0',
                  '3.36', '-5', '5', '0.5'], ['20.0', '35.0', '3.75', '-5', '5',
                  '0.5'], ['35.0', '120.0', '4.47', '0', '0', '0.5']]
    return(model)

#%%
def creat_and_plot_model(init_obj, lthickness, velocity, 
                         save_dir = '', plotter_ind = False):
    '''
    

    Parameters
    ----------
    init_obj : TYPE
        Initial object created by main_classes.Initialize.
    lthickness : TYPE
        Layer thickness.
    velocity : TYPE
        Shear velocity.
    save_dir : TYPE, optional
        Saving directory. The default is ''.
    plotter_ind : TYPE, optional
        If True plot the model. The default is False.

    Returns
    -------
    None.

    '''
    if (save_dir == ''):
        saving_directory =  init_obj.output_folder+'/dummy_st/'
    else:
        saving_directory = save_dir
    if (os.path.isdir(saving_directory)):
        pass
    else: 
        try:
            os.mkdir(saving_directory)
        except FileNotFoundError:
            print('Your input path doesnt exist.')
            return
    
            
    
    syn_forw = iv.Forward_cal(vel_s= velocity, 
                                         layers_thickness = lthickness, 
                                         filt_list = init_obj.filt_list,
                                         gauss_par= init_obj.gauss_par,
                                         dt = init_obj.dt,
                                         nsamp = init_obj.nsamp, 
                                         tshift = init_obj.tshift,
                                         slowness= init_obj.slowness,
                                         inv_time_rf1 = init_obj.inv_bf, 
                                         inv_time_rf2 = init_obj.inv_af,
                                         noise_level =0.0,
                                         rf_normalize= 0,
                                         saving_directory = saving_directory)

    
    
    lthickness_abs = ut.find_lthickness_abs(lthickness)
    app_vel = np.array(syn_forw.apparant_vel_org) 
    rf = np.array(syn_forw.rf_r_from_ind1.copy())
    time_vec = np.arange(-init_obj.inv_bf, init_obj.inv_af, init_obj.dt)
    
    if (plotter_ind):
        plotter_d(velocity, time_vec, lthickness_abs, 
              rf, app_vel, init_obj.filt_list, 
              save_name= os.path.join(init_obj.output_folder, 
                                      'velocity_model.png'),
              header = 'Velocity Model')
    return(rf, time_vec, app_vel, init_obj.filt_list, velocity, 
           lthickness, lthickness_abs)
    