        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:19:28 2022

@author: soroush
"""

import numpy as np
import os
from obspy import read, UTCDateTime, Stream
import obspy as op
from obspy.io import sac
import math
import rf
from obspy import taup
import matplotlib.pyplot as plt

#%%
class Layer_rftan:
    def __init__(self, thickness, vel_s, density, vel_p = 'according to vp/vs', 
                 vp_to_vs = 1.732,
                 q_p = 0.0,
                 q_s = 0.0,
                 eta_p= 0.0,
                 eta_s = 0.0,
                 fref_p = 1.0,
                 fref_s = 1.0):
        self.thickness = thickness
        self.vel_s = vel_s
        self.s_vel = vel_s * 1000
        if (self.vel_s > 100):
            print('the unit of vvel_s should be km/s')
            self.vel_s = self.vel_s / 1000.0 
        
        self.vp_to_vs = vp_to_vs
        if (vel_p == 'according to vp/vs'):
            self.vel_p = self.vel_s * vp_to_vs
        else:
            self.vel_p = vel_p 
        self.density = density
        if (self.density > 100):
            print('the unit of vvel_s should be gr/cm3')
            self.density = self.density / 1000.0
        self.q_p = q_p 
        self.q_s = q_s 
        self.eta_p = eta_p 
        self.eta_s = eta_s 
        self.fref_p = fref_p 
        self.fref_s = fref_s
#%%
class Model_rftan:
    def __init__(self, layers, name='syn_fow_model_rftan.mod'):
        self.layers = layers
        if (isinstance(self.layers, list)):
            self.nlayer = len(self.layers)
        else: 
            self.nlayer = 1 
        self.name = name
        
    def create_model_file(self):
        header = ("MODEL\n"+
                  "MODEL FOR SYN RF\n"+
                  "ISOTROPIC\n"+ 
                  "KGS\n"+
                  "FLAT EARTH\n"+
                  "1-D\n"+ 
                  "CONSTANT VELOCITY\n" 
                  +"LINE08\n"
                  +"LINE09\n"+ 
                  "LINE10\n" 
                  +"LINE11\n"
                  +"HR VP VS RHO QP QS ETAP ETAS FREFP FREFS\n")
       
        if (self.nlayer == 1):
            print('your model only have 1 layer i will attach\n' +
                  'a half space beneath it')
            orginal = self.layers 
            mantle = Layer_rftan(0.0,4.7,3.3)
            self.layers = [orginal, mantle]
        '''
        example
        HR VP VS RHO QP QS ETAP ETAS FREFP FREFS
        40 6 3.5 2.5 0.0  0.0 0.0 0.0 1.0 1.0
        '''
        layer_line = []
        model_values = ''
        # check if everyting is ok
        model_ok = False 
        model_ok = self.check_model()
        
        if (model_ok):
            idum = -1
            for el in self.layers:
                idum += 1
                layer_line.append(el.thickness)
                layer_line.append(el.vel_p)
                layer_line.append(el.vel_s)
                layer_line.append(el.density)
                layer_line.append(el.q_p)
                layer_line.append(el.q_s)
                layer_line.append(el.eta_p)
                layer_line.append(el.eta_s)
                layer_line.append(el.fref_p)
                layer_line.append(el.fref_s)
                line_text = ''
                for el2 in layer_line:
                    line_text = line_text + "{:25.16f}".format(el2) 
                if (idum == (self.nlayer - 1)):
                    model_values = model_values + line_text 
                else:    
                    model_values = model_values + line_text + '\n' 
                layer_line = []
        else:
            print("bad model detected, changing velocities to half space")
            layer_line = []
            el = self.layers[0]
            layer_line.append(30)
            layer_line.append(5.71)
            layer_line.append(3.30)
            layer_line.append(2.60)
            layer_line.append(el.q_p)
            layer_line.append(el.q_s)
            layer_line.append(el.eta_p)
            layer_line.append(el.eta_s)
            layer_line.append(el.fref_p)
            layer_line.append(el.fref_s)
            line_text = ''
            for el2 in layer_line:
                line_text = line_text + "{:9.2f}".format(el2)     
            model_values = model_values + line_text + '\n' 
            layer_line = []
            
            
            
            layer_line.append(0)
            layer_line.append(7.14)
            layer_line.append(4.12)
            layer_line.append(3.11)
            layer_line.append(el.q_p)
            layer_line.append(el.q_s)
            layer_line.append(el.eta_p)
            layer_line.append(el.eta_s)
            layer_line.append(el.fref_p)
            layer_line.append(el.fref_s)
            line_text = ''
            for el2 in layer_line:
                line_text = line_text + "{:9.2f}".format(el2) 
            model_values = model_values + line_text 
            layer_line = []
        file_to_write = header +  model_values
        with open(self.name, 'w') as f:
            f.write(file_to_write)
        f.close()
            
    def check_model(self):
        thickness = []
        vel_p = []
        vel_s = []
        rho = []
        for el in self.layers:
            thickness.append(el.thickness)
            vel_p.append(el.vel_p)
            vel_s.append(el.vel_s)
            rho.append(el.density)
        is_it_ok_thickness = self.check_param(thickness, kind = "thickness")
        is_it_ok_vel_s = self.check_param(vel_s, kind = "vel_s")
        is_it_ok_vel_p = self.check_param(vel_p, kind = "vel_p")
        is_it_ok_rho = self.check_param(rho, kind = "rho")
        
        if ((is_it_ok_thickness == True) and 
            (is_it_ok_vel_s == True) and 
            (is_it_ok_vel_p == True) and 
            (is_it_ok_rho == True)):
            model_ok = True 
        else:
            # if (is_it_ok_vel_s == False):
            #     print('bad vel_s')
            #     print(vel_s)
            # elif (is_it_ok_vel_p == False):
            #     print('bad vel_p')
            #     print(vel_p)
            # elif (is_it_ok_rho == False):
            #     print('bad rho')
            #     print(rho)
            # else:
            #     print(thickness)
            model_ok = False 
        return(model_ok)
        
    def check_param(self, param, kind = 'vel_s'):
        is_it_ok = True 
        if (kind == 'thickness'):
            dum = np.nan
            for el in param:
                if ((el < 0) or (el == dum)):
                    is_it_ok = False 
        else:
            for el in param:
                dum = np.nan
                if ((el < 1) or (el == dum)):
                    is_it_ok = False 
        return(is_it_ok)
                    
        
        
        
        
            
            
#%%
class Rftn_init:
    def __init__(self, phase = 'P', ray_p = 0.05,
                 delta = 0.05, nsamp = 512, 
                 model = 'syn_fow_model_rftan.mod',
                 model_name_implicit = 'syn_fow_model_rftan.mod',
                 alpha= 1.0,
                 x2length = False,
                 timeshift = 5,
                 out_name = 'rftn_output'
        ,saving_directory = '/home/soroush/rf_shallow_codes/my_py_rf/rftan_out'):   
        '''
    USAGE: hrftn96 [-P] [-S] [-2] [-r] [-z] -RAYP p -ALP alpha -DT dt -NSAMP nsamp -M model
    -P           (default true )    Incident P wave
    -S           (default false)    Incident S wave
    -RAYP p      (default 0.05 )    Ray parameter in sec/km
    -DT dt       (default 1.0  )    Sample interval for synthetic
    -NSAMP nsamp (default 512  )    Number samples for synthetic
    -M   model   (default none )    Earth model name
    -ALP alp     (default 1.0  )    Number samples for synthetic
         H(f) = exp( - (pi freq/alpha)**2) 
         Filter corner ~ alpha/pi 
    -2           (default false)    Use 2x length internally
    -r           (default false)    Output radial   time series
    -z           (default false)    Output vertical time series
         -2  (default false) use double length FFT to
         avoid FFT wrap around in convolution 
    -D delay     (default 5 sec)    output delay sec before t=0
     SAC header values set by hrftn96
      B     :  delay
      USERO :  gwidth        KUSER0:  Rftn
      USER4 :  rayp (sec/km)
      USER5 :  fit in % (set at 100)
      KEVNM :  Rftn          KUSER1:  hrftn96
    The program creates the file names hrftn96.sac
    This is the receiver fucntion, Z or R trace according to the command line flag
    '''
    
    #hrftn96 [-P] [-S] [-2] [-r] [-z] -RAYP p -ALP alpha -DT dt -NSAMP nsamp -M model
        self.phase = phase
        self.ray_p = ray_p 
        self.model_name_implicit = model_name_implicit
        self.delta = delta 
        self.nsamp = nsamp 
        self.model = model 
        self.alpha = alpha 
        self.x2length = x2length
        self.timeshift = timeshift
        self.out_name = out_name
        if (saving_directory[-1] != '/'):
            saving_directory = saving_directory+'/'
        self.saving_directory = saving_directory
        
        if (os.path.isdir(self.saving_directory)):
            pass
        else:
            os.mkdir(self.saving_directory)
        
    def make_all(self):
        time, data_r = self.run_hrftn(cmp = 'R')
        
        time, data_z = self.run_hrftn(cmp = 'Z')
        
        baz_arr = np.zeros(len(data_r))
        data_t = np.zeros(len(data_r))
        
        mat_tr = np.zeros(shape=(len(data_r), 5))
        mat_tr[:,0] = time 
        mat_tr[:,1] = baz_arr
        mat_tr[:,2] = data_r 
        mat_tr[:,3] = data_t
        mat_tr[:,4] = data_z
        
        return(mat_tr)
        
        
        
    def run_hrftn(self, cmp = 'R'):
        alpha_for_name = str(self.alpha)
        alpha_for_name = alpha_for_name.replace('.', '_')
        ray_p_for_name = str(self.ray_p)
        ray_p_for_name = ray_p_for_name.replace('.', '_') 
        out_file_name = (self.saving_directory + 
                         self.out_name + '_'+ alpha_for_name +'_'
                         +ray_p_for_name + '_' + cmp+
                         '.sac')
        run_command = ''
        run_command = run_command + 'hrftn96 '
        run_command = run_command + '-'+self.phase+' '
        if (self.x2length):
            run_command = run_command + '-2 '
        if (cmp == 'R'):
            run_command = run_command + '-r '
        else:
            run_command = run_command + '-z '
        run_command = run_command + '-RAYP '+str(self.ray_p)+ ' '
        run_command = run_command + '-ALP '+str(self.alpha) + ' '
        run_command = run_command + '-DT '+str(self.delta) + ' '
        run_command = run_command + '-NSAMP '+str(self.nsamp) + ' '
        run_command = run_command + '-D '+str(self.timeshift) + ' '
        run_command = run_command + '-M '+ self.model_name_implicit
        #runing hrftan here, the program should be exported in the bashrc
        dir_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(self.saving_directory)
        
        os.system(run_command + '>/dev/null 2>&1')
        command_to_mv ='mv hrftn96.sac '+ out_file_name
        os.system(command_to_mv)
        os.chdir(dir_path)
        # print("==========")
        # os.system("cat syn_fow_model_rftn.mod")
        # print(out_file_name)
        # print(run_command)
        time, data = self.read_and_shift(out_file_name)
        return(time, data)
        
    def read_and_shift(self, out_file_name):
        # print(out_file_name)
        # print(self.model)
        rftan_sac = op.read(out_file_name)

        tr_rftan = rftan_sac[0]
        data_rftan = tr_rftan.data

        delta_af = tr_rftan.stats.delta 
        timeshift_af = abs(tr_rftan.stats.sac.b)
        nsamp_af = tr_rftan.stats.npts


        time = np.arange(-timeshift_af, ((nsamp_af * delta_af)- timeshift_af),
                          delta_af)
        zero_ind = np.argwhere(abs(time) == min(abs(time)))[0][0]
        onset_ind = np.argwhere(data_rftan == max(data_rftan))[0][0]
        # print(onset_ind)
        shift_data = (onset_ind - zero_ind)

        data_rftan_af_shift = np.zeros(len(data_rftan))
        for i in range(len(data_rftan)):
            if ((i - shift_data) > len(data_rftan) -1 ):
                shifted_sample = (i - shift_data) - (len(data_rftan)-1)
            else:
                shifted_sample = (i - shift_data)
            data_rftan_af_shift[shifted_sample] = data_rftan[i]
        time = time + timeshift_af
        return(time, data_rftan_af_shift)
        
        
        
    