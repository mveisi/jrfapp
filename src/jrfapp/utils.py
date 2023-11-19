#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 08:06:46 2023

@author: soroush
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, next_fast_len


#%%
class Vel_paramterize:
    def __init__(self, vel_info, layer_info, vp_to_vs):
        self.vel_info = vel_info 
        self.layer_info = layer_info
        self.vp_to_vs = vp_to_vs
        
        self.create_layer(layer_info = False) 
        self.create_vel_s()
        self.create_time_thickness()
        self.create_vel_s_smooth()
        self.create_vel_init_real()
    
    
    
    def create_layer(self, layer_info = False):
        if (layer_info == False):
            layer_info_cal = self.layer_info 
        else:
            layer_info_cal = layer_info
        nlayer = 0
        for layer in layer_info_cal:
            nlayer = nlayer + layer[2] 
        nlayer = nlayer + 1
        
        layer_thickness = []
        
        idum = -1
        for layer in layer_info_cal:
            idum += 1
            l_ref = np.linspace(layer[0], layer[1], layer[2] + 1)
            for i in range(len(l_ref) - 1):
                layer_thickness.append(l_ref[i+1] - l_ref[i])
        layer_thickness_abs = []
        thickness = 0.0
        for layer in layer_thickness:
            thickness = thickness + layer 
            layer_thickness_abs.append(thickness)
        
        layer_thickness_abs.append(thickness + 40)
        layer_thickness.append(0)
        if (layer_info == False):
            self.nlayer = nlayer
            self.layer_thickness = layer_thickness 
            self.layer_thickness_abs = layer_thickness_abs
        else:
            return(nlayer, layer_thickness, layer_thickness_abs)
        
    def update_layer_thickness(self, layer_thickness_input):
        self.layer_thickness = layer_thickness_input.copy()
        layer_thickness_abs = []
        thickness = 0.0
        for layer in layer_thickness_input:
            thickness = thickness + layer
            layer_thickness_abs.append(thickness)
        layer_thickness_abs.append(thickness + 40)
        self.layer_thickness_abs = layer_thickness_abs.copy()
        
        
        
    def create_time_thickness(self):
        self.time_thickness = cal_time_thickness(self.vel_s,
                                                 self.layer_thickness, 
                                                 vp_to_vs= self.vp_to_vs)
    def create_vel_s(self):
        vel_s = []
        for i in range(self.nlayer):
            for vs_info in self.vel_info:
                if ((self.layer_thickness_abs[i] > vs_info[0]) and 
                    (self.layer_thickness_abs[i] <= vs_info[1])):
                        # print(self.layer_thickness_abs[i], vs_info[0], vs_info[1])
                        vel_s.append(vs_info[2])
            self.vel_s = vel_s.copy()
    def create_vel_s_smooth(self):
        for i in range(self.nlayer):
            min_vel = self.vel_info[0][2]
            max_vel = self.vel_info[-2][2]
            
            vel_s = np.linspace(min_vel, max_vel, self.nlayer - 1)
            vel_s = vel_s.tolist() 
            vel_s.append(self.vel_info[-1][2])
            self.vel_s_smooth = vel_s.copy()
    def perturb_vel(self, which_to_perturb):
        self.perturbed_vel = self.vel_s.copy()
        for i in range(self.nlayer):
            for pert in which_to_perturb:
                if ((self.layer_thickness_abs[i] > pert[0]) and 
                    (self.layer_thickness_abs[i] <= pert[1])):
                    self.perturbed_vel[i] = self.perturbed_vel[i] + pert[2]
    def perturb_vel_smooth(self, which_to_perturb):
        self.perturbed_vel_smooth = self.vel_s_smooth.copy()
        pert_vel = [] 
        for pert in which_to_perturb:
            pert_vel.append(-1)
        for i in range(self.nlayer):
            idum = -1
            for pert in which_to_perturb:
                idum += 1
                if ((self.layer_thickness_abs[i] > pert[0]) and 
                    (self.layer_thickness_abs[i] <= pert[1])):
                    if (pert_vel[idum] == -1):
                        pert_vel[idum] = self.perturbed_vel_smooth[i] + pert[2]
                        self.perturbed_vel_smooth[i] = \
                            self.perturbed_vel_smooth[i] + pert[2]
                    else:
                        self.perturbed_vel_smooth[i] = pert_vel[idum]
    def create_vel_init(self):
        self.vel_init = [] 
        vs_max = self.vel_s[-1]
        vs_min = self.vel_s[0]
        for vs in np.linspace(vs_min, vs_max, self.nlayer):
            self.vel_init.append(vs)
    def create_vel_init_real(self):
        self.vel_init = [] 
        vs_max = self.vel_s[-2]
        vs_min = self.vel_s[0]
        for vs in np.linspace(vs_min, vs_max, self.nlayer):
            self.vel_init.append(vs)
        self.vel_init[-1] = self.vel_s[-1]
    def map_to_new_layers(self, layer_info, kind = 'vel_s'):
        nlayer_n, layer_thickness_n, layer_thickness_abs_n = \
            self.create_layer(layer_info)
        vs_n = []
        vs_minmax_n = []
        vs_limit_n = []
        vs_minmax_old = np.array(self.vs_minmax)
        limit_old = np.array(self.vs_limit)
        if (kind == 'vel_s'):
            vs_old = np.array(self.vel_s)
        if (kind == 'vel_s_smooth'):
            vs_old = np.array(self.vel_s_smooth)
        if (kind == 'vel_pert_smooth'):
            vs_old = np.array(self.perturbed_vel_smooth)
        if (kind == 'vel_pert'):
            vs_old = np.array(self.perturbed_vel)
        if (kind == 'vel_init'):
            vs_old = np.array(self.vel_init)
        if (kind == 'vel_estimate'):
            vs_old = np.array(self.vel_estimate)
        
        thickness_old = np.array(self.layer_thickness)
        thickness_abs_old = np.array(self.layer_thickness_abs)
        for i in range(nlayer_n):
            vs = []
            limit = []
            minmax_1 = []
            minmax_2 = []
            for j in range(self.nlayer):
                if ( i == 0):
                    if (thickness_abs_old[j] <= layer_thickness_abs_n[i]):
                        vs.append(vs_old[j])
                        limit.append(limit_old[j])
                        minmax_1.append(vs_minmax_old[j][0])
                        minmax_2.append(vs_minmax_old[j][1])
                else:
                    if ((thickness_abs_old[j] <= layer_thickness_abs_n[i]) and
                         (thickness_abs_old[j] > layer_thickness_abs_n[i-1])):
                        vs.append(vs_old[j])
                        limit.append(limit_old[j])
                        minmax_1.append(vs_minmax_old[j][0])
                        minmax_2.append(vs_minmax_old[j][1])
            vs_n.append(np.mean(vs))
            vs_minmax_n.append([np.mean(minmax_1), np.mean(minmax_2)])
            vs_limit_n.append(np.mean(limit))
        return(nlayer_n, layer_thickness_n, layer_thickness_abs_n,
               vs_n, vs_minmax_n, vs_limit_n)
    def map_to_new_layers_high(self, layer_info, kind = 'vel_s'):
        nlayer_n, layer_thickness_n, layer_thickness_abs_n = \
            self.create_layer(layer_info)
        vs_n = []
        for i in range(nlayer_n):
            vs_n.append(-1)
        vs_minmax_n = []
        vs_limit_n = []
        vs_minmax_old = np.array(self.vs_minmax)
        limit_old = np.array(self.vs_limit)
        if (kind == 'vel_s'):
            vs_old = np.array(self.vel_s)
        if (kind == 'vel_pert'):
            vs_old = np.array(self.perturbed_vel)
        if (kind == 'vel_init'):
            vs_old = np.array(self.vel_init)
        if (kind == 'vel_estimate'):
            vs_old = np.array(self.vel_estimate)
        
        thickness_old = np.array(self.layer_thickness)
        thickness_abs_old = np.array(self.layer_thickness_abs)
        for i in range(self.nlayer):
            for j in range(nlayer_n):
                if ((layer_thickness_abs_n[j] <= thickness_abs_old[i]) and 
                    (vs_n[j] == -1)):
                        vs_n[j] = vs_old[i]
                        vs_minmax_n.append([vs_minmax_old[i][0], vs_minmax_old[i][1]])
                        vs_limit_n.append(limit_old[i])
        return(nlayer_n, layer_thickness_n, layer_thickness_abs_n,
               vs_n, vs_minmax_n, vs_limit_n)   
    def map_to_new_layers_high_4inv(self, layer_info, kind = 'vel_s'):
        nlayer_n, layer_thickness_n, layer_thickness_abs_n = \
            self.create_layer(layer_info)
        vs_n = []

        vs_minmax_n = []
        vs_limit_n = []
        vs_minmax_old = np.array(self.vs_minmax)
        limit_old = np.array(self.vs_limit)
        if (kind == 'vel_s'):
            vs_old = np.array(self.vel_s)
        if (kind == 'vel_pert'):
            vs_old = np.array(self.perturbed_vel)
        if (kind == 'vel_init'):
            vs_old = np.array(self.vel_init)
        if (kind == 'vel_estimate'):
            vs_old = np.array(self.vel_estimate)
        
        thickness_old = np.array(self.layer_thickness)
        thickness_abs_old = np.array(self.layer_thickness_abs)
        vs_n_arr = np.interp(np.array(layer_thickness_abs_n),
                             np.array(thickness_abs_old),
                             np.array(vs_old))
        for el in vs_n_arr:
            vs_n.append(el)
        return(nlayer_n, layer_thickness_n, layer_thickness_abs_n,
               vs_n)  
    
        
                        
                             



   

#%%
def find_layer_info(init_boundary, layering_4_cal, max_depth_4_cal):
    layer_info = []
    flayer = 0.0
    for i, el in enumerate(init_boundary):
        layer_info.append([flayer, el, layering_4_cal[i]])
        flayer = el
    layer_info.append([flayer, max_depth_4_cal, layering_4_cal[-1]])
    return(layer_info)
def cal_time_thickness(vel_s, layer_thickness, 
                       vp_to_vs = 1.732, slowness = 0.04):
    time_thickness = []
    for i in range(len(layer_thickness) -1):
        vp = vel_s[i] * vp_to_vs
        f_term = np.sqrt((vel_s[i] ** -2.0)  - 
                          (slowness ** 2.0))
        s_term = np.sqrt((vp ** -2.0) - 
                         (slowness ** 2.0))
        tthickness = layer_thickness[i] * (f_term - s_term)
        time_thickness.append(tthickness)
    time_thickness.append(0)
    return(time_thickness)
def _gauss_filter(dt, nft, f0, waterlevel=None):
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
    return np.exp(gauss_arg)
def _apply_filter(x, filt):
    """
    Apply a filter defined in frequency domain to a data array

    :param x: array of data to filter
    :param filter: filter to apply in frequency domain,
        e.g. from _gauss_filter()
    :return: real part of filtered array
    """
    nfft = len(filt)
    xf = fft(x, n=nfft)
    xnew = ifft(xf*filt, n=nfft)
    return xnew.real
def cal_rms(a):
    N = len(a)
    sumd = 0.0 
    for i in range(N):
        sumd = sumd + a[i]**2.0 
    rms = np.sqrt(sumd / N)
    return(rms)
def find_lthickness_abs(lthickness):
    lthick_abs = []
    depth = 0.0
    for i, el in enumerate(lthickness):
        if (el == 0.0):
            depth = depth + 10
        else:
            depth = depth + el
        lthick_abs.append(depth)
    return(lthick_abs)
def plot_matrix(mat, title_fig = None, value_title = None):
    nrow, ncol = np.shape(mat)
    extent = [0, ncol, nrow, 0]
    fig = plt.subplots(figsize=(8, 12), nrows=1,
                        ncols=1, dpi=150)
    plt.imshow(mat, extent=extent, 
                aspect=ncol / nrow)
    plt.xlabel('Column index')
    plt.ylabel('Row index')
    if (value_title == None):
        plt.colorbar(label='Matrix Values',
                  fraction=0.02, pad=0.04)
    else:
        plt.colorbar(label=value_title,
                  fraction=0.02, pad=0.04)
    if (title_fig != None):
        plt.title(title_fig)
def scatter_line_plot(a, 
                      title= 'Array scatter', 
                      xtitle = 'N',
                      ytitle = 'Amplitude'):
    N = len(a)
    
    std = np.std(a)
    rms = cal_rms(a)
    mean = np.mean(a)
    prob_info_title = ('std = '+'{0:4.2f}'.format(std)+
                     ', rms = '+'{0:4.2f}'.format(rms)+
                     ', mean ='+'{0:4.2f}'.format(mean))
    fig, axs = plt.subplots(figsize=(12, 8), nrows=1,
                        ncols=2)
    fig.suptitle(title + '\n' + prob_info_title)
    axs[0].scatter(range(N), a, s = 12, c= 'cyan')
    axs[0].plot(range(N), a, c= 'navy', alpha = 0.5)
    axs[0].grid(True)
    axs[0].set_xlabel(xtitle)
    axs[0].set_ylabel(ytitle)

    
    axs[1].hist(a, 50, density=False, facecolor='navy')
    axs[1].set_xlabel(ytitle)
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True)
def find_xminmax(depth, vel):
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