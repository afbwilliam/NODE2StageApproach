# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:27:53 2020

@author: Afbwi
"""
#Functions for integrating and tracking the history of ODE optimization
    #Using Autograd via pytorch

import torch
import numpy as np
import torch.nn as nn

def odeint(func, y0, t, z=[], rtol=1e-7, atol=1e-9, N_steps=6, method=None, options=None, params=None):    
    solver = Eul(func, y0, z, rtol, atol, N_steps, params)
    solution = solver.integrate(t)
    return solution
    
class Eul(object):
    def __init__(self, func, y0, z, atol, rtol, N_steps, params, **unused_kwargs):
#        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs
        self.func = func
        self.y0 = y0
        self.atol = atol
        self.rtol = rtol
        self.z = z
        self.N_steps = N_steps
        self.params = params
        
    def rk4_step_func(self, func, N_t, dt, y, z, t, k1=None):
        if k1 is None: k1 = func(N_t, y, z, t)
        k2 = func(N_t + dt / 2, y + dt * k1 / 2 , z, t)
        k3 = func(N_t + dt / 2, y + dt * k2 / 2, z, t)
        k4 = func(N_t + dt, y + dt * k3, z, t )
        return (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6) 
    
    def eul_step_func(self, func, N_t, dt, y, z, t, k1=None):
#        if z==None:
#            return func(N_t, y) * dt
#        func(N_t, y , z, t).T * torch.tensor(dt,dtype=torch.float32)
#        return (func(N_t, y, z, t).T *dt).T
        if self.params is not None:
            return func(N_t,y,self.params)*dt # Only works for Sense_vs_NoSense example
        else:
            return func(N_t,y,z,t)*dt

    def integrate(self,t):
        solution = [self.y0]
        y0 = self.y0
        z = self.z
        N_steps = self.N_steps;
        # Manually discretize space
        t_data = []
        if t.ndim == 1:
            for t0,t1 in zip(t[:-1],t[1:]):
                t_data.append(np.linspace(t0,t1,N_steps+1)[:-1]) #include then remove t in interval
            #batch_t = torch.stack(tuple(map(torch.stack, tuple(zip(*t_data)))))
            batch_t = torch.tensor(np.append(np.concatenate(np.stack(t_data)),t[-1]),dtype=torch.float32)
            
            for t0,t1 in zip(batch_t[:-1],batch_t[1:]):
                if len(self.z) != 0:
                    #Set index for Sf = N_t*66/(198)
                    ind = ((t0-min(t))*(len(t)-1)/(max(t)-min(t))).int()# 
                    z = self.z[:,ind,:]
                dy = self.step_func(self.func, t0, t1 - t0, y0, z, t)
                y1 = y0 + dy 
                if t1 in torch.tensor(t,dtype=torch.float32):
                    solution.append(y1)
                y0 = y1
            tup = tuple(map(torch.stack, tuple(zip(*solution))))

        if t.ndim == 2:
            N_runs = len(t[:,0]); #N_Bsteps = len(t[0,:])
            batch_t = []
            for j in range(0,N_runs):
                t_data = []
                for t0,t1 in zip(t[j,:-1],t[j,1:]):
                    t_data.append(np.linspace(t0,t1,N_steps+1)[:-1]) #include then remove t in interval
                batch_t.append(np.append(np.concatenate(np.stack(t_data)),t[j,-1]))
            batch_t = torch.tensor(np.stack(batch_t).T,dtype=torch.float32)

            for t0,t1 in zip(batch_t[:-1],batch_t[1:]):
                if len(self.z) != 0:
                    #This assumes t1-t0 is the same for all intervals
                    ind = ((t0[0]-min(t[0]))*(len(t[0])-1)/(max(t[0])-min(t[0]))).int()#.astype(int) #Set index for Sf = N_t*66/(198)
                    z = self.z[:,ind,:]
                t0 = t0.unsqueeze(1).repeat(1,len(y0[0,:]))
                t1 = t1.unsqueeze(1).repeat(1,len(y0[0,:]))
#                dt = (t1 - t0).unsqueeze(1).repeat(1,len(y0[0,:]))
                dy = self.step_func(self.func, t0, t1-t0, y0, z, t)
                # if sum(np.isnan(dy.detach().numpy())).any() > 0:
                #     print("nans on the loose."); break

                y1 = y0 + dy 
                if t1[0,0] in torch.tensor(t,dtype=torch.float32)[0]:
                    solution.append(y1)
                y0 = y1
            tup = tuple(map(torch.stack, tuple(zip(*solution))))
        return torch.stack(tup)
    
    def step_func(self, func, N_t, dt, y, z, t):
#        return self.rk4_step_func(func, N_t, dt, y, z, t)
        return self.eul_step_func(func, N_t, dt, y, z, t)

    @property
    def order(self):
        return 4

class Store(object):
    def __init__(self, momentum=0.99): #momentum term unused but adds a look of sophistication
        self.momentum = momentum
        self.reset()

    def reset(self):
        # parameters
        self.w1 = []; self.w2 = []; self.w3 = []; self.w4 = []; self.k = []
        # gradients
        self.w1g=[]; self.w2g=[]; self.w3g=[]; self.w4g=[]; self.kg=[]
        self.avg = 0
        # loss
        self.y_rawloss = torch.tensor([100],dtype=torch.float32)
        self.reg_loss = torch.tensor([100],dtype=torch.float32)
        self.dy_rawloss = torch.tensor([100],dtype=torch.float32)
        self.tot_loss = torch.tensor([100],dtype=torch.float32)
        self.y_loss = torch.tensor([100],dtype=torch.float32)
        self.dy_loss = torch.tensor([100],dtype=torch.float32)

        self.increase_reg = []; self.increase_tot = []
        
    def l_update(self,losses):
        with torch.no_grad():
            self.tot_loss = torch.cat([self.tot_loss,losses[0].unsqueeze(dim=0)])
            self.reg_loss = torch.cat([self.reg_loss,losses[1].unsqueeze(dim=0)])
            self.y_rawloss = torch.cat([self.y_rawloss,losses[2].unsqueeze(dim=0)])
            self.dy_rawloss = torch.cat([self.dy_rawloss,losses[3].unsqueeze(dim=0)])
            self.y_loss = torch.cat([self.y_loss,losses[4].unsqueeze(dim=0)])
            self.dy_loss = torch.cat([self.dy_loss,losses[5].unsqueeze(dim=0)])

    def reg_add(self,epoch):
        self.increase_reg.append(epoch)
    def tot_add(self,epoch):
        self.increase_tot.append(epoch)
        
    def update(self,p):
#        self.params.append(list(p.detach().numpy()))
#        self.grads.append(list(p.grad.detach().numpy()))
#        for p in params:
#            self.param
        self.w1.append(p[0].detach().numpy()*1); 
        self.w2.append(p[1].detach().numpy()*1); 
        self.w3.append(p[2].detach().numpy()*1);
        self.w4.append(p[3].detach().numpy()*1); 
        self.k.append(p[4].detach().numpy()*1);
        self.w1g.append(p[0].grad.detach().numpy()*1); 
        self.w2g.append(p[1].grad.detach().numpy()*1); 
        self.w3g.append(p[2].grad.detach().numpy()*1);
        self.w4g.append(p[3].grad.detach().numpy()*1); 
        self.kg.append(p[4].grad.detach().numpy()*1)
