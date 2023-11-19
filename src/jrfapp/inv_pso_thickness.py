

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
import inverse_routine as iv 
import multiprocessing as mp
import utils as ut

#%% PSO
class Invert_thickness_pso:
    def __init__(self, vel_init, layers_thickness_init,
                 rf_obs, app_obs, filt_list, 
                 vel_param_initial, 
                 init_boundary = [10, 35],
                 init_dif_boundary= [[-2, 5] , [-5, 5]], 
                 init_layers= [8, 12, 12],
                 num_procs= 12,
                 norm_cond=0.15,
                 pso_nparticle = 100, pso_max_iter = 200,
                 pso_c1 = 2, pso_c2 = 2.1, 
                 inv_bf =2, inv_af = 25,
                 pso_vw = 1.0, app_weight = 5,
                 rf_weight = 2.0,
                 smooth_fac = 0.5,
                 damp_fac = 0.5000,
                 dt=0.05, rf_method = 'waterlevel',
                 rf_method_sens = 'waterlevel',
                 gauss_par = 2.5,
                 rf_normalize = 0,
                 nsamp=1024, tshift=10.0,
                 slowness=0.04, 
                 print_particle_info = False, 
                 save_dir_root = '/home/soroush/rf_shallow_codes/my_py_rf/output_of_forwftry/'):
        self.save_dir_root = save_dir_root
        self.rf_normalize = rf_normalize
        self.gauss_par = gauss_par
        self.rf_method = rf_method
        self.rf_method_sens = rf_method_sens 
        self.inv_bf = inv_bf
        self.inv_af = inv_af 
        self.app_weight = app_weight
        self.rf_weight = rf_weight 
        self.damp_fac = damp_fac
        self.smooth_fac = smooth_fac
        self.pso_max_iter = pso_max_iter
        self.pso_nparticle = pso_nparticle
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2 
        self.pso_vw = pso_vw
        self.rf_obs = rf_obs
        self.app_obs = app_obs
        self.dt = dt
        self.tshift = tshift
        self.nsamp = nsamp
        self.slowness = slowness
        self.filt_list = filt_list
        self.norm_cond = norm_cond
        self.layers_thickness = layers_thickness_init
        self.vel_param_initial = vel_param_initial
        self.vel_init = vel_init
        self.nlayer = len(self.vel_init)
        self.unknown_param = np.zeros(shape=(len(self.vel_init),))
        self.print_particle_info = print_particle_info
        self.init_dif_boundary = init_dif_boundary 
        self.init_boundary = init_boundary 
        self.init_layers=  init_layers
        self.num_procs = num_procs
        self.print_particle_info = True
        # self.init_dif_boundary = [[-2, 5] , [-5, 5]] # for synthetic
        # self.init_boundary = [10, 37] #for synthetic
        

        self.bound_obj = []

        self.iter_cpu = 0
        
        self.all_outputs = {}
        # np.random.seed(100)
        self.pso_vw_iterative = np.linspace(self.pso_vw, 1.0, self.pso_max_iter + 1)
                
                
        start = timeit.default_timer()
        self.run_PSO()
        stop = timeit.default_timer()
        self.runtime = stop - start
        # self.plot_estimates(save_to_dir= True)  
        
        
        
        
    def run_PSO(self):
        self.iter_pso = 0
        # self.all_outputs['iter' + str(self.iter_pso)] = []
        self.initialize_pso()
        while (self.stop_run == False):
            self.iter_pso += 1
            self.update_gbest()
            print('======== iter '+str(self.iter_pso)+
                  ' from '+str(self.pso_max_iter + 1)+'===============')
            if (self.iter_pso == 1):
                print("new gbest assigned with reduction= " + 
                      str(self.gbest_obj_final / self.init_obj_final) + '\n'+
                      'final obj is = '+str(self.gbest_obj_final)+
                      ' initial was = '+str(self.init_obj_final)+'\n'+
                      'initial boundary was: '+   
                      str(self.init_boundary)+'\n'+
                      'new boundary is: '+ str(self.gbest_boundary))
            else:
                print("new gbest assigned with reduction= " + 
                      str(self.gbest_obj_final / self.init_obj_final) + '\n'+
                      'final obj is = '+str(self.gbest_obj_final)+
                      ' initial was = '+str(self.init_obj_final)+'\n'+
                      'initial boundary was: '+   
                      str(self.all_particles[self.gbest_ind]['per_boundary'])+'\n'+
                      'new boundary is: '+ str(self.gbest_boundary))
            print('======================================')
            self.check_for_stop()
            self.all_particles_backup = self.all_particles.copy()
            pool = mp.Pool(self.num_procs)
            nrun = np.arange(self.pso_nparticle)
            args = zip([self] * self.pso_nparticle, 
                       [self.all_particles] * self.pso_nparticle
                       ,nrun)
            result = pool.map(update_vel_wrapper, args)
            self.all_particles = []
            for part in result:
                self.all_particles.append(part)
            pool.close()
            # for particles in self.all_particles:
            #     self.update_vel(particles)
            # self.plot_all_estimates()
                
            
            
            
            
            
            
            
            
    def create_model(self, boundary = [15, 35], layers = [3, 3, 2], 
                     dif_boundary = [5, 5], phase = 'particle'):
        if (phase == 'initial'):
            vel_info = self.vel_param_initial.vel_info
            layering = self.vel_param_initial.PSO_layering
            layer_info_init = self.vel_param_initial.layer_info
            for i, el in enumerate(layer_info_init):
                el[2] = layering[i]
            layer_info = layer_info_init.copy()
            vp_to_vs = self.vel_param_initial.vp_to_vs
            
            self.vel_param = ut.Vel_paramterize(vel_info, 
                                                       layer_info, 
                                                       vp_to_vs)

            vel_info = self.vel_param_initial.vel_info 
            layer_info = self.vel_param_initial.layer_info 
            vp_to_vs = self.vel_param_initial.vp_to_vs 
            layer_info = [[0.0, layer_info[-1][1], int(layer_info[-1][1] / 2)]]
            vel_param_fine = ut.Vel_paramterize(vel_info, layer_info, vp_to_vs)
            
            layers_thickness_fine = vel_param_fine.layer_thickness.copy()
            velocity_fine = vel_param_fine.vel_s.copy()
            boundary_init_vel = np.unique(velocity_fine)
            layer_thickness_abs_fine = vel_param_fine.layer_thickness_abs
            
            velocity = self.vel_param.vel_s.copy()
            layer_thickness_abs = self.vel_param.layer_thickness_abs
            layers_thickness = self.vel_param.layer_thickness
            nlayer = len(velocity)
            
            jacob_app_vel = self.forward_cal_for_guess_vel(vel_param_fine)
            
            min_boundary = np.zeros(shape= (len(self.init_boundary)))
            max_boundary = np.zeros(shape= (len(self.init_boundary)))
            for i, el in enumerate(self.init_boundary):
                min_boundary[i] = el + self.init_dif_boundary[i][0]
                max_boundary[i] = el + self.init_dif_boundary[i][1]
            self.min_boundary = min_boundary
            self.max_boundary = max_boundary
            
            guessed_vel_list = []
            flayer = 0.0
            for i in range(len(boundary)):
                g_vel_list = []
                for j in range(len(layer_thickness_abs_fine)):
                    if ((layer_thickness_abs_fine[j] <= boundary[i]) and 
                        (layer_thickness_abs_fine[j] > flayer)):
                        g_vel = self.guess_vel_cal(jacob_app_vel, 
                                                vel_param_fine,  
                                                layer_thickness_abs_fine[j])
                        if ((g_vel > 2.0) and (g_vel <= 4.0)):
                            g_vel_list.append(g_vel)
                            # print(layer_thickness_abs_fine[j], g_vel)
                guessed_vel_list.append(np.mean(g_vel_list))
                flayer = boundary[i]
                # print('Mean was : '+ str(np.mean(g_vel_list)))
                
            shifted_lthick_abs = np.array(layer_thickness_abs_fine) - boundary[-1]
            
            ind_last_boundary = np.argwhere(np.abs(shifted_lthick_abs) == 
                                            np.min(np.abs(shifted_lthick_abs)))[0][0]
            g_vel_list = []
            for el in layer_thickness_abs_fine[ind_last_boundary:]:
                g_vel = self.guess_vel_cal(jacob_app_vel,
                                                     vel_param_fine, el)
                if ((g_vel > 2.0) and (g_vel <= 5.0)):
                    g_vel_list.append(g_vel)
                    # print(el, g_vel)
            guessed_vel_list.append(np.max(g_vel_list))
            # print('Mean was : '+ str(np.mean(g_vel_list)))
            # print('max was : '+ str(np.max(g_vel_list)))
                
            self.layers_thickness = layers_thickness
            self.layers_thickness_abs = layer_thickness_abs
            self.nlayer = nlayer
            boundary_dum = []
            boundary_dum.append(0)
            for el in boundary:
                boundary_dum.append(el)
            boundary_dum.append(layer_thickness_abs[-2])
            
            for j in range(len(boundary_dum) - 1):
                for i, el in enumerate(layer_thickness_abs):
                    if ((el <= boundary_dum[j + 1]) and 
                        (el > boundary_dum[j])):
                        velocity[i] = guessed_vel_list[j]
            velocity[-1] = guessed_vel_list[-1]
            self.vel_init = velocity.copy()
            (x_min, x_max, y_min, y_max, v_line_x, h_line_y) = \
                ut.find_xminmax(self.layers_thickness_abs, self.vel_init)


            fig = plt.figure(figsize = (6, 12))
            plt.vlines(v_line_x, y_min, y_max, colors='black', lw = 3.0)
            plt.hlines(h_line_y, x_min, x_max,colors='black', lw = 3.0)
            plt.ylim([self.layers_thickness_abs[-2], 0])
            plt.xlim([2, 5.5])
            plt.grid(True)
            plt.title('Initial Approximated Velocity')
            plt.savefig(self.save_dir_root+ 
                        'Initial_Approximated_velocity_of_PSO.png')
            
            # print(guessed_vel_list)
        else:
           vel_info = self.vel_param_initial.vel_info
           layering = self.vel_param_initial.PSO_layering
           layer_info_init = self.vel_param_initial.layer_info
           for i, el in enumerate(layer_info_init):
               el[2] = layering[i]
           
           layer_info_d = layer_info_init.copy()
           flayer = 0.0
           for i, el in enumerate(boundary):
               layer_info_d[i][0] = flayer
               layer_info_d[i][1] = el
               flayer = el
           layer_info_d[-1][0] = boundary[-1]
           vp_to_vs = self.vel_param_initial.vp_to_vs
           vel_param_d = ut.Vel_paramterize(vel_info, 
                                                      layer_info_d, 
                                                      vp_to_vs)
           velocity = self.vel_init.copy()
           layers_thickness = vel_param_d.layer_thickness.copy()
           layers_thickness_abs = vel_param_d.layer_thickness_abs.copy()
           nlayer = len(velocity)
           
           
           
          
        
        
        return(layers_thickness, velocity, nlayer, boundary)
    def forward_cal_for_guess_vel(self, vel_param):
        layers_thickness = vel_param.layer_thickness.copy()
        velocity = vel_param.vel_s.copy()
        layer_thickness_abs = vel_param.layer_thickness_abs
        ##creating sample forward for guessing velocity
        ftry_forw = iv.Forward_cal(velocity,
                       layers_thickness ,self.filt_list
                       ,kind = 'rftn'
                       ,gauss_par = self.gauss_par
                       ,dt = self.dt
                       ,nsamp = self.nsamp
                       ,rf_method = self.rf_method
                       ,tshift = self.tshift
                       ,slowness = self.slowness
                       ,waterlevel = 0.01
                       ,cor_out ='ZRT'
                       ,rf_method_sens = self.rf_method_sens
                       ,inv_time_rf1 = self.inv_bf, inv_time_rf2 = self.inv_af
                       ,filter_array = 'cal'
                       ,rf_normalize=self.rf_normalize
                       ,saving_directory= os.path.join(os.getcwd(), 'dummy_folder'))
        ftry_forw.cal_sensivity_jacobson(ref = 'vel_s', 
                                         cal_rho_vp=True, plotter= False)
        ftry_forw.cal_jacobian_rf_jacobson()
        ftry_forw.cal_jacobian_app_vel_jacobson()
        return(ftry_forw.jacobian_app_vel)
        
    def guess_vel_cal(self, jacob_app_vel, vel_param, depth):
        
        
        from scipy.signal import argrelextrema
        layer_thickness_abs_shifted = (np.array(vel_param.layer_thickness_abs) - 
                                       depth)

        ind = np.argwhere(abs(layer_thickness_abs_shifted) == 
              np.min(abs(layer_thickness_abs_shifted)))[0][0]

        signal = jacob_app_vel[:, ind] / np.max(jacob_app_vel[:, ind])
        maxima_indices = argrelextrema(signal, np.greater)[0]
        maxima_indices_list = maxima_indices.tolist()
        
        absolute_max_ind = np.argwhere(np.abs(signal) == 
                                       np.max(np.abs(signal)))[0][0]
        if (absolute_max_ind not in maxima_indices_list):
            maxima_indices_list.append(absolute_max_ind)
        maxima_indices = np.array(maxima_indices_list)
        # # Get the corresponding maximum values
        maxima_values = signal[maxima_indices]
        
        
        
        ###
        # plt.figure()
        # plt.plot(signal)
        # plt.scatter(maxima_indices, maxima_values)
        # fl_name = ('/home/soroush/rf_shallow_codes/my_py_rf/sens_2/' + 'sens'+
        #            str(depth)+ '.png')
        # plt.savefig(fl_name)
        
        
        # maxima_values = maxima_values / np.max(maxima_values)
        plt.plot(self.app_obs)
        vel_guessd = []
        for i in range(len(maxima_indices)):
            vel_guessd.append(self.app_obs[maxima_indices[i]])
        vel_guessd = np.array(vel_guessd)
        vel_guess_val = []
        for i in range(len(vel_guessd)):
            vel_guess_val.append(vel_guessd[i] * maxima_values[i])

        vel_guess = np.sum(vel_guess_val) / np.sum(maxima_values)
        
        return(vel_guess)
         
        
    def check_for_stop(self):
        if (self.iter_pso > self.pso_max_iter):
            self.stop_run = True 
            self.close_pso()
        if ((self.gbest_obj_final / self.init_obj_final) < self.norm_cond):
            self.stop_run = True 
            self.close_pso()
        
        
    def initialize_pso(self):
        self.objective_particle_all_iter = []
        self.iter_pso = 0 
        self.cal_kappa()
        self.layers_thickness, self.vel_init, self.nlayer, self.init_boundary = \
            self.create_model(boundary= self.init_boundary, 
                              layers= self.init_layers, dif_boundary= self.init_dif_boundary, 
                              phase= 'initial')
        self.call_obj_vel_init()
        self.all_particles = []
        ####unparrallel
        # for i in range(self.pso_nparticle):
        #     part = self.create_particles()
        #     self.all_particles.append(part)
        ### parallel
        pool = mp.Pool(self.num_procs)
        nrun = np.arange(self.pso_nparticle)
        args = zip([self] * self.pso_nparticle, nrun)
        result = pool.map(create_particles_wrapper, args)
        # result = map(create_particles_wrapper, self)
        for part in result:
            self.all_particles.append(part)
        pool.close()
        for part in self.all_particles:
            self.bound_obj.append([part['curr_boundary'], 
                                   part['curr_obj_final']])
        self.stop_run = False 
        
        
        
        
        
        
    def create_particles(self, nrun):
        # print(nrun)
        np.random.seed(100 * nrun + 1)
        particles = dict() 
        lthick, vel, boundary = self.rand_boundary()
        
        particles['per_lthickness'] = lthick
        particles['per_vel'] =  vel
        particles['per_boundary'] = boundary
        
        
        
        particles['curr_vel'] = particles['per_vel'].copy() 
       
        particles['curr_lthickness'] = particles['per_lthickness'].copy()

        (obj, tthickness, boundary) = self.cal_rf_energy(particles['per_vel'], 
                                                      particles['per_lthickness'], 
                                                      particles['per_boundary'],
                                                      nrun)
        particles['iter_all_obj'] = []
        particles['iter_all_obj'].append([boundary, obj])
        particles['per_tthickness'] = tthickness
        
        particles['per_obj_final'] = obj
        
        particles['curr_boundary'] = boundary

        
        
        particles['curr_obj_final'] = obj
        particles['curr_tthickness'] = tthickness.copy()
       
        list_zero = [] 
        for el in boundary:
            list_zero.append(0)
        
        particles['curr_unk_vel'] = list_zero
        particles['per_unk_vel'] = list_zero
        particles['pbest'] =  particles['curr_boundary']
        particles['pbest_obj_final'] = obj
        particles['gbest'] = list_zero
        if (self.print_particle_info):
            print('Particle created with boundary = '+str(boundary) +'\n' +
                  ' with energy = '+str(obj))
        return(particles)
        
    def update_gbest(self):
        self.curr_obj_final = np.zeros(shape=(self.pso_nparticle,))
        for i in range(self.pso_nparticle):
            self.curr_obj_final[i] = self.all_particles[i]['curr_obj_final']
        self.objective_particle_all_iter.append(self.curr_obj_final)
        ind = np.argwhere(self.curr_obj_final == 
                          np.min(self.curr_obj_final))[0][0]
        if (self.iter_pso == 1):
            per_gbest_obj = 10000
        else:
            per_gbest_obj = self.gbest_obj_final 
        
        cur_gbest_boundary = self.all_particles[ind]['curr_boundary']
        cur_gbest_vel = self.all_particles[ind]['curr_vel']
        cur_gbest_lthickness = self.all_particles[ind]['curr_lthickness']
        cur_gbest_obj = self.all_particles[ind]['curr_obj_final']
        
        if (cur_gbest_obj < per_gbest_obj):
            self.gbest_particle_ind = ind
            self.gbest_boundary = cur_gbest_boundary.copy()
            # print(self.gbest_boundary)
            # print(self.all_particles[ind]['curr_vel'])
            self.gbest_ind = ind 
            self.gbest_vel = self.all_particles[ind]['curr_vel'].copy()
            self.gbest_lthickness = cur_gbest_lthickness.copy()
            self.gbest_obj_final = cur_gbest_obj
            self.gbest_cond = self.gbest_obj_final / self.init_obj_final
            for i in range(self.pso_nparticle):
                self.all_particles[i]['gbest'] = \
                    self.gbest_boundary.copy()
        
        
        
        
    def update_vel(self, particles, nrun):

        v0 = particles['curr_unk_vel'].copy()
        pbest = particles['pbest'].copy()
        gbest = particles['gbest'].copy()
        boundary = particles['curr_boundary'].copy()
        
        
        
        v1 = []
        for i in np.arange(len(boundary)):
            r1 = random.uniform(0.0, 1.0)
            r2 = random.uniform(0.0, 1.0)
            dum1 = v0[i] * self.pso_vw_iterative[self.iter_pso - 1]
            # print('iter_pso is : ' +str(self.iter_pso) + 'index is : ' +
            #       str(self.iter_pso - 1) + 'value of index is : '+ 
            #       str(self.pso_vw_iterative[self.iter_pso - 1]))
            dum2 = self.pso_c1 * r1 * (pbest[i] - boundary[i])
            dum3 = self.pso_c2 * r2 * (gbest[i] - boundary[i])
            
            v1d = self.kappa * (dum1 + dum2 +dum3)
            v1.append(v1d)
        boundary_new = []
        for i in range(len(boundary)):
            accept_boundary = False 
            accept_boundary = self.bound_boundary((boundary[i] + v1[i]), i)
            if (accept_boundary == False):
                    v1[i] = 0.0
            boundary_new.append(boundary[i] + v1[i])
        
            
                    
        particles = self.update_particles(particles, 
                              v1, boundary_new, nrun)
        if (self.print_particle_info):
            print("current iteration is :" + str(self.iter_pso)+ '\n'+
                  'The boundary for this particle before update was: '+
                  str(boundary) + '\n' +
                  'the energy was: '+
                  str(particles['per_obj_final']) + '\n'+
                  'current gbest is :'+ str(particles['gbest']) +'\n'+
                  'current pbest for this particle is :' + str(particles['pbest'])+ '\n'+
                  'boundary after update is :'+
                  str(boundary_new) + '\n' + 
                  'the objective updated to : '+ str(particles['curr_obj_final']) +
                  '\n'+ '================================================')
       
        return(particles)
        
        
    def update_particles(self, particles, v1, boundary_new, nrun= 0):
        #maping curr to per:
        particles['per_lthickness'] = particles['curr_lthickness'].copy()
        particles['per_obj_final'] = particles['curr_obj_final']
        particles['per_unk_vel'] = particles['curr_unk_vel'].copy()
        particles['per_tthickness'] = particles['curr_tthickness'].copy()
        particles['per_boundary'] = particles['curr_boundary'].copy()
        
        
        
        #calculating new
        particles['curr_unk'] = boundary_new.copy() 
        particles['curr_unk_vel'] = v1.copy()
        
        
        layers_thickness, velocity, nlayer, boundary = \
            self.create_model(boundary= boundary_new, layers= self.init_layers, 
                              dif_boundary= self.init_dif_boundary)
        particles['curr_lthickness'] = layers_thickness.copy()
        particles['curr_boundary'] = boundary_new.copy()
        
        
        particles['curr_vel'] = velocity.copy()
        # print("========== next particle ================")
        # print ('calculating with new boundary :'+str(boundary_new))
        (obj, tthickness, boundary) = self.cal_rf_energy(particles['curr_vel'], 
                                                      particles['curr_lthickness'], 
                                                      particles['curr_boundary'], 
                                                      nrun)
        particles['iter_all_obj'].append([boundary, obj])
        
    
        particles['curr_all_t'] = tthickness
        particles['curr_obj_final'] = obj
        
        if (particles['curr_obj_final'] < particles['pbest_obj_final']):
            particles['pbest'] =   particles['curr_boundary'].copy()
            particles['pbest_obj_final'] =  particles['curr_obj_final']
        
        return(particles)
            
    def call_obj_vel_init(self):
        obj, tthickness, boundary = self.cal_rf_energy(self.vel_init, 
                                                      self.layers_thickness, 
                                                      self.init_boundary)
        
        self.init_obj_final = obj
        self.init_tthickness = tthickness
    def cal_rf_energy(self, vel, thickness, boundary, nrun = 0):
        
        thickness_abs = []
        sumd = 0.0
        for el in thickness:
            sumd = sumd + el 
            thickness_abs.append(sumd)
        app_weight = self.app_weight
        rf_weight = self.rf_weight
        smooth_fac = self.smooth_fac
        damp_fac = self.damp_fac
        gauss_par = self.gauss_par
        waterlevel_val = 0.01
        rf_normalize = self.rf_normalize
        rf_method = self.rf_method
        inv_bf = self.inv_bf
        inv_af = self.inv_af
        #### trying to divide to 4
        tthickness_final = self.cal_tthickness(vel = vel, 
                                               layer_thickenss= thickness)
        vel_out, lthickness_out= divide_tthickness(
                      vel_estimate= vel, 
                      tthickness_final= tthickness_final,
                      slowness= self.slowness,
                      ndivide = 1)
        
        
        iter_all = nrun
        save_dir_root = self.save_dir_root
        save_dir_pso = self.save_dir_root + 'pso_workplace/'
        if (os.path.isdir(save_dir_pso)):
            pass
        else:
            os.mkdir(save_dir_pso)
        folder_name = ('run_iter_'+str(iter_all)+
            '_PSO_dummy'+'/')
        save_dir_run = save_dir_pso +folder_name                                                                           
        inv_frame_iter = iv.Invert_joint_iter(vel_out,
                                          layers_thickness= lthickness_out,
                                          rf_obs= self.rf_obs,
                                          app_obs = self.app_obs,
                                          app_weight= app_weight,
                                          rf_weight = rf_weight,
                                          smooth_factor= smooth_fac,
                                          damp_factor= damp_fac,
                                          filt_list=self.filt_list,
                                          Cd_array_rf = False, 
                                          Cd_array_app = False,
                                          norm_cond=0.01,
                                          amon_method=True,
                                          nsamp = self.nsamp, 
                                          dt =self.dt, rf_method=rf_method,
                                          tshift=10.0,
                                          no_damp = False, 
                                          no_smooth=False,
                                          max_iter=9,
                                          slowness=self.slowness,
                                          jacob = 0,
                                          out_kind='best',
                                          save_dir=save_dir_run,
                                          gauss_par = gauss_par,
                                          waterlevel = waterlevel_val,
                                          rf_normalize = rf_normalize,
                                          inv_time_rf1 = inv_bf,
                                          inv_time_rf2 = inv_af,
                                          cal_sens = True, 
                                          obs_normalize=True, 
                                          synthetic = False,
                                          save_to_dict = False,
                                          force_jacob_cal = True,
                                          plot_fig = False,
                                          print_out = False, 
                                          folder_jacob_save = 'current')
        ## for relative obj
        # obj = inv_frame_iter.best_cond_estimate
        ##====
        ## for basolute obj 
        # part_out = {}
        # part_out['layer_thickness_init'] = lthickness_out
        # part_out['velocity_init'] = vel_out
        # part_out['rf_cond'] = inv_frame_iter.best_rf_cond
        # part_out['app_cond'] = inv_frame_iter.best_app_cond
        # part_out['estimated_vel'] = inv_frame_iter.estimated_vel
        # part_out['layer_thickness_output'] = inv_frame_iter.best_lthickness
        # part_out['rf_output'] = inv_frame_iter.best_rf_curve_estimate
        # part_out['app_vel_out'] = inv_frame_iter.best_app_curve_estimate
        
        
        
        obj = (inv_frame_iter.best_rf_cond + 
                   inv_frame_iter.best_app_cond)
        self.bound_obj.append([self.iter_pso, boundary, obj])
        # print('=================')
        # print(str([self.iter_pso, boundary, obj]))
        # print('=================')
        
        # dif_rf_200 = np.sqrt(np.sum(inv_frame_iter.best_dif_rf[:200]**2.0))
        
        ##==
        # print(obj)
        tthickness = self.cal_tthickness(vel, thickness)
        return(obj, tthickness, boundary)
    def cal_energy(self, t):
        
        t_abs = []
        sumd= 0.0
        for el in t:
            sumd = sumd + el 
            t_abs.append(sumd)
        energy_all = []
        for el in t_abs:
            ttime = self.time_rf - el
            ind = np.argwhere(np.abs(ttime) == np.min(np.abs(ttime)))[0][0]
            energy_all.append(self.rf_obs[ind])
        return(np.sum(energy_all))
        
        
   
    def call_obj_vel_final(self):
        (obj, tthickness, boundary) = self.cal_rf_energy(self.gbest_vel, 
                                                      self.gbest_lthickness, 
                                                      self.gbest_boundary)
        self.final_lthickness = self.gbest_lthickness.copy()
        self.final_vel = self.gbest_vel.copy()
        
        self.final_obj_final = obj
        
        self.final_tthickness = tthickness
        
        self.final_boundary = boundary
        
    def close_pso(self):
        self.call_obj_vel_final()
        self.vel_s_estimate = self.final_vel.copy()
        self.best_lthickness = self.final_lthickness.copy()
        self.best_obj_estimate = self.final_obj_final
        self.best_cond_estimate = self.final_obj_final / self.init_obj_final
        
       
        
    def cal_kappa(self):
        phi = self.pso_c1 + self.pso_c2
        kappa = 2.0/(np.abs((2.0-phi)-np.sqrt((phi**2.0) - (4.0*phi))))
        self.kappa = kappa
        # KAPPA=2.0/ABS((2.0-PHI)-(SQRT((PHI**2.0)-(4.0*PHI))))
        
        
        
    def cal_obj(self, dif_o_s):
         N= len(dif_o_s)
         obj = np.sum(dif_o_s**2.0) 
         obj = np.sqrt((1 / N) * obj)
         return(obj)
    def cal_obj_final(self, obj_app, obj_rf, obj_smooth):
        obj_final1 = self.rf_weight * obj_rf
        obj_final2 = self.app_weight * obj_app
        obj_final3 = self.smooth_factor * obj_smooth
        obj_final = obj_final1 + obj_final2+ obj_final3 
        return(obj_final, obj_final1, obj_final2, obj_final3)
    def rand_boundary(self):
        
        
        rand_boundary = []
        for i in range(len(self.init_boundary)):
            random_dif = np.random.uniform(self.min_boundary[i], 
                                           self.max_boundary[i])
            rand_boundary.append(random_dif)
        layers_thickness, velocity, nlayer, boundary = \
            self.create_model(boundary= rand_boundary, layers= self.init_layers, 
                          dif_boundary = self.init_dif_boundary)
        return(layers_thickness, velocity, boundary)
        
        
        
        
        
        
        
    def rand_tthickness(self):
        
        self.init_tthickness
        tthickness = []
        for i in np.arange(self.nlayer-1):
            accept_tthickness = False 
            while (accept_tthickness == False):
                random_dif = np.random.uniform(
                    self.tt_minmax[i, 0] * self.init_tthickness[i],
                    self.tt_minmax[i, 1] * self.init_tthickness[i])
                # random_dif = (random.uniform(
                #     self.tt_minmax[i, 0] * self.init_tthickness[i],
                #     self.tt_minmax[i, 1] * self.init_tthickness[i]))
                tt_layer = self.init_tthickness[i] + random_dif
                accept_tthickness = self.bound_tthickness(tt_layer, i)
            tthickness.append(tt_layer)
        tthickness.append(self.init_tthickness[-1])
        # print("creating particles")
        # print(self.init_tthickness, tthickness)
        return(tthickness)

    
    def bound_boundary(self, bb, i):
        accept_boundary = False
        if ((bb <= self.max_boundary[i]) and 
            (bb>= self.min_boundary[i])):
            accept_boundary = True
        return(accept_boundary)
    def bound_tthickness(self, tt, i):
       tt_min =  self.init_tthickness[i] - \
           (self.tt_limits[i] *self.init_tthickness[i])
       tt_max =  self.init_tthickness[i] + \
           (self.tt_limits[i] *self.init_tthickness[i])
       accept_tthickness = False 
       if (tt < tt_max and tt > tt_min):
           accept_tthickness = True 
           
       return(accept_tthickness)
    def cal_thickness_from_tthickness(self, vel, tthickness):
       layer_thickenss = []
       idum = -1
       for el in vel[:len(vel)-1]:
           idum += 1
           denominator1 = (np.sqrt((el**-2.0) -
                                 (self.slowness) ** 2.0))
           denominator2 = (np.sqrt(((1.732 *el)**-2.0) -
                                 (self.slowness) ** 2.0))
           layer_thickenss.append(tthickness[idum]/ 
                                  (denominator1 - denominator2))
       layer_thickenss.append(0)
       return(layer_thickenss)
    def cal_tthickness(self, vel, layer_thickenss):
       tps = [] 
       idum = -1
       for el in vel:
           idum += 1
           numerator1 = (np.sqrt((el**-2.0) -
                                 (self.slowness) ** 2.0))
           numerator2 = (np.sqrt(((1.732 *el)**-2.0) -
                                 (self.slowness) ** 2.0))
           tps.append(layer_thickenss[idum] * (numerator1 - numerator2))
       return(tps)
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
def create_particles_wrapper(args):
    obj, nrun = args
    part = obj.create_particles(nrun)
    return(part)
def update_vel_wrapper(args):
    obj, all_particle, nrun = args
    particle = all_particle[nrun]
    np.random.seed(50 * obj.iter_pso * nrun + 8)
    part = obj.update_vel(particle, nrun)
    return(part)
    
    