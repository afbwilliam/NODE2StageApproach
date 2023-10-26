# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:48:20 2020

@author: Afbwi
"""
''' This code fits NODEs to Penicillin, Styrene, and Lotka-Volterra data.
  States and derivatives estimated from this code can be used to fit
  ODE model parameters in the pyomo code found in the same folder. 
  '''
import argparse
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import os
import imageio as io
import shutil

from AG_Funs import odeint, Store, \
    compute_losses, Check_point, MMScaler, MScaler
from sklearn.preprocessing import MinMaxScaler
from Visualize import visuals
import pickle
import sys

from Visualize import plot_LoVody, plot_LoVo



# =============================================================================
# # Set default hyperparameters and training conditions
# =============================================================================
#parser.add_argument('--data_size', type=int, default=1000)
parser = argparse.ArgumentParser('Styrene demo')
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=3220) #changed from 2000
parser.add_argument('--test_freq', type=int, default=4) #changed from 20
args = parser.parse_args()
# args.mod = 'LoVo'     #LoVo, Sty, Pen; Model to fit
args.viz = None         #Subplots, ClearGraph, None; Choose graphs to plot
args.onerun = False     #Reduces # of runs to 1
noise = 0.05            #0, 0.01, 0.05, 0.10; Level of noise (%/100)
# N_dat = 10            # cannot be larger than N_t - N_Bsteps, otherwise indices are repeated
N_Bsteps = 3            #5, N_dat -1 or length of t; Size of overlapping intervals
args.rtol=10**-6
# lamb = 1e-5;          #1e-4, 1e-5, 1e-6  
tot_lamb = 100          #Weighting for error term
MC_lamb = 100           #Weighting for error term     
learning_rate = 0.1*10  #Optimzer learning rate
# N_hnodes = 10         #10, 20; Number of hidden nodes
omitIC = 0              #Omit Initial Conition if >0
SIC = False; MIC = True #Single or multiple integration Intervals
save_data = False       #Whether to save results to csv file

# Create hyperparameter dataframe
columns = ['Model','Noise Lvl','Reg weight','Hid Nodes','Lrn rate','Activ Func',
           'Euler steps','Epochs','MSE Train','Sim Time','MSE true','MSE noisy',
           'MAE', 'MSE t_scld', 'MSE n_scld']
df_hyper = pd.DataFrame(columns=columns)

row = 0 # row of hyper_param df

# Test multiple hyperparameters
# for args.mod in ['Pen']:
#     for N_hlayers in [1,2]:
#         for N_hnodes in [10,20]:
#             for lamb in [1e-4,1e-6]:
#                 print('Mod, Noise, Hid Nodes, Reg:',args.mod,noise,N_hnodes,lamb)
# Test a single set of hyperparameters
for args.mod in ['Sty']:
    for N_hlayers in [1]:
        for N_hnodes in [10]:
            for lamb in [1e-4]:
                # Fix seed so that Neural ODE param init and noise is the same each time
                np.random.seed(84)
                torch.manual_seed(34) 
                print('Mod, Noise, Hid Nodes, Reg:',args.mod,noise,N_hnodes,lamb)
    
                # =============================================================================
                # # Import data
                # =============================================================================
                if args.mod == 'Pen':
                    file = 'data/PenBatch_0Noise.csv'
                    states = ['tB','tS','tP','tV']
                    controls = ['Sf','F']
                    derivs = ['dBdt','dSdt','dPdt','dVdt']
                    args.N_steps = 12; N_dat = 10
                #    lamb = 1e-5;       learning_rate = 0.1
                elif args.mod == 'Sty':
                    file = 'data/NonIso_PFR.csv'
                    states = ['Temp','EB','Sty','H2','BeEt','ToMe'] #df_sol.drop(columns=['t','Run']).columns.values
                    controls=[] #['Temp'] # dummy var
                    derivs = ['dTdt','dEBdt','dStydt','dH2dt','dBeEtdt','dToMedt']
                    args.N_steps = 6;   N_dat = 10
                elif args.mod == 'LoVo':
                    # idxes = [0,1,2,3,14,15,16] # sparse
                    file = 'data/LoVo.csv'
                    states = ['x','y']
                    controls = []#['x'] # dummyZ var
                    derivs = ['dxdt','dydt']
                    args.N_steps = 20;  N_dat = 20 #N_steps=10
                    # args.N_steps = 15;  N_dat = 8 #20
                    # args.N_steps = 50;  N_dat = 4;
                    # args.N_steps = 150; N_dat = 2 
                #    lamb = 1e-5;       learning_rate = 0.01

                df_sol = pd.read_csv(file)
                t = df_sol[df_sol['Run'] == 0]['t'].to_numpy()#[::-1]#[5:]#[:20]
                # Reduce to 1 Run: Used specifically for NN vs. NODE comparison.
                #   Also being used to fit small NODE in pyomo
                if args.onerun == True:
                    df_sol = df_sol[df_sol['Run'] == 0] 
                    print('Warning: Comparing NNs vs NODEs')
                
                N_t = len(t)                    # Number of time points
                N_runs = df_sol['Run'].max()+1  # Number of run conditions
                N_vars = len(states)            # Number of state vars
                N_zvars = len(controls)
                
                sol = np.ones((N_runs,N_t,N_vars));
                solz = np.ones((N_runs,N_t,N_zvars));
                sol_dy = np.ones((N_runs,N_t,N_vars));
                for N_run in range(0,N_runs):
                    sol[N_run,:,:] = df_sol[df_sol['Run'] == N_run][states]#[::-1]#[5:]#[:20]
                    solz[N_run,:,:] = df_sol[df_sol['Run'] == N_run][controls]#[::-1]#[5:]#[:20]
                    sol_dy[N_run,:,:] = df_sol[df_sol['Run'] == N_run][derivs]#[::-1]#[5:]#[:20]
                        
                if args.mod == 'Sty':
                    sol[:,:,0] = sol[:,:,0]-800
                

                # =============================================================================
                # # Prepare data for training
                # =============================================================================
                # Collect data -- Single IC
                true_y0 = torch.tensor(sol[:,0,:],dtype=torch.float32) # true init cond
                true_z = torch.tensor(solz[:,:,:],dtype=torch.float32) # forcing vars
                true_y = torch.tensor(sol[:,:,:],dtype=torch.float32)  # true data
                true_dy = torch.tensor(sol_dy[:,:,:],dtype=torch.float32)
                max_y = np.amax(np.amax(true_y.numpy(),axis=1),axis=0) #for adding noise
                min_y = np.amin(np.amin(true_y.numpy(),axis=1),axis=0) #for adding noise
                
                idx = np.round(np.linspace(0, len(t) -1, N_dat)).astype(int) # 10 data points
                idxes = idx
                if omitIC >= 1:
                    idx = idx[omitIC:] # Omit initial condition
                    N_dat = len(idx)
                
                true_t = t[idx]#[::-1].copy()
                
                # Add noise
                batch_y = true_y[:,idx,:]       # data + meas noise
                k = N_vars-1 if args.mod == 'Pen' else N_vars
                for i in range(0,k):
                    batch_y[:,:,i:i+1] = batch_y[:,:,i:i+1] + np.random.normal(loc=0.0,
                           scale=noise*(max_y[i]-min_y[i]),size=(N_runs,len(idx),1))
                
                # Create var for y data after adding noise, before rearranging and scaling
                torch_y = batch_y[:,:,:].detach().clone()
                
                # Scaling
                max_y = np.amax(np.amax(batch_y.numpy(),axis=1),axis=0) #before scaling batch_y
                min_y = np.amin(np.amin(batch_y.numpy(),axis=1),axis=0) #for scaling dy
                #min_y = np.zeros(N_vars)
                
                MMS = MMScaler(args) #MinMaxScaler()
                batch_y2d = MMS.fit_transform(batch_y.reshape((-1,N_vars)))#,
                
                batch_y = batch_y2d.reshape((N_runs,len(idx),N_vars)).clone().detach()
                true_y2d = true_y[:,idx,:].reshape((-1,N_vars)).clone().detach()
                torch_y2d = MMS.transform(torch_y.reshape((-1,N_vars)))#,       #noise added and scaled
                batch_y0 = torch.tensor(batch_y[:,0,:].numpy(),dtype=torch.float32,requires_grad=True)
                #batch_y0 = MMS.transform(true_y0)#,
                #                        dtype=torch.float32,requires_grad=True)
                
                if args.mod == 'Pen':
                    #Sf and F scaled w/state min+max
                    batch_z = true_z[:,:,:]
                    MMSz = MMScaler(args,torch.tensor([max_y[1],max_y[3]]),
                                        torch.tensor((min_y[1],min_y[3]))) 
                    # Must use same scaling as batch_y2d
                    batch_z2d = torch.stack(((true_z[:,:,0])/(max_y[1]- min_y[1]),
                                             (true_z[:,:,1])/(max_y[3] - min_y[3])),
                                                                axis=2).reshape((-1,N_zvars))
                #    batch_z2d = MMSz.transform(true_z[:,:,:].reshape((-1,N_zvars)))#.clone.detach()
                #                       dtype=torch.float32) # for scaling loss function
                    batch_z = batch_z2d.reshape((N_runs,len(t),N_zvars)).clone().detach()
                else:
                    batch_z = true_y.clone().detach() #dummy var for arg consistency
                
                # Collect data -- Multiple ICs
                true_MCy = []
                true_MCz = []
                true_MCt = []
                bat_MCy = []
                batch_MCz = []
                for j in range(0,N_runs):
                    for i in range(0,N_dat - N_Bsteps):#idx:#range(0,N_t - N_Bsteps):
                        true_MCt.append(t[idx][i:i+N_Bsteps+1])#[::-1])
                        true_MCy.append(sol[j][idx][i:i+N_Bsteps+1,:])
                        true_MCz.append(solz[j][idx][i:i+N_Bsteps+1,:])
                        bat_MCy.append(batch_y[j][i:i+N_Bsteps+1,:])
                        batch_MCz.append(batch_z[j][i:i+N_Bsteps+1,:])
                true_MCy = torch.tensor(np.stack(true_MCy,axis=0),dtype=torch.float32)
                true_MCz = torch.tensor(np.stack(true_MCz,axis=0),dtype=torch.float32)
                true_MCt = np.stack(true_MCt,axis=0)
                batch_MCy = torch.stack(bat_MCy,axis=0)
                batch_MCz = torch.stack(batch_MCz,axis=0)
                idx = np.round(np.linspace(0,N_Bsteps,N_Bsteps+1)).astype(int)
                N_runs = len(true_y[:,0,0]) # don't think this changes anything
                true_MCy0 = true_MCy[:,0,:].detach().clone()
                batch_MCy0 = torch.tensor(batch_MCy[:,0,:].numpy(),dtype=torch.float32,requires_grad=True)

                
                y0 = MMS.transform(torch_y[:,0,:])#sol[:,0,:]

                # sys.exit()
                # =============================================================================
                # # Train data-driven NODE
                # =============================================================================
                class ODEFunct(nn.Module):
                    def __init__(self,N_inputs,N_outputs,N_hnodes):
                        super(ODEFunct, self).__init__()
                        self.N_inputs = N_inputs;   self.N_outputs = N_outputs
                        # Single NN
                        if N_hlayers == 1:
                            self.net = nn.Sequential(
                                nn.Linear(N_inputs, N_hnodes),
                                nn.Tanh(),
                                nn.Linear(N_hnodes, N_outputs),)
                        if N_hlayers == 2:
                            self.net = nn.Sequential(
                                nn.Linear(N_inputs, N_hnodes),
                                nn.Tanh(),

                                nn.Linear(N_hnodes, N_hnodes),
                                nn.Tanh(),
                                nn.Linear(N_hnodes, N_outputs),)
                
                        for m in self.net.modules():
                            if isinstance(m, nn.Linear):
                                nn.init.normal_(m.weight, mean=0, std=0.2)
                                nn.init.constant_(m.bias, val=0)
                        self.k = torch.tensor([2.00,2.00,2.00],requires_grad=True) #13.2392
                        
                        
                    # Black-box Model
                    def forward(self, N_t, x, z=0, t=0):
                        terms = self.net(x)
                        return terms
                
                    if args.mod == 'Sty':
                        def forward(self, N_t, x, z, t):
                            # inputs = torch.cat(x,N_t)
                            # terms = self.net(inputs)
                            terms = self.net(x)
                            #use only explantory vars as inputs
                #            inputs = x[:,0:4]
                #            terms = self.net(inputs)
                #            dum = list(self.net.parameters())[1].detach().numpy()
                #            dum = list(self.net.parameters())[2].detach().numpy()
                            return terms
                    
                    if args.mod == 'Pen':
                        # No assumed physics
                        def forward(self, N_t, x, z, t):
                            dV = z[:,[1]]
                            D = z[:,1:]/x[:,3:]         # F/V
                #            inputs = torch.cat((x[:,0:3],D,z[:,0:1]),axis=1)
                            inputs = torch.cat((x[:,:],z[:,:]),axis=1) #Consistent dims for dy calcs
                            terms = self.net(inputs)
                            dy = torch.cat((terms,dV),axis=1)
                            return dy
                        # Assume mass balance
                        def forward(self, N_t, x, z, t):
                #            t_idx = np.tile(np.arange(0,7),2)
                #            dum = np.tile(np.arange(7,N_dat-N_Bsteps),2)
                            dV = z[:,[1]]
                #            D = (z[:,1]*(max_y[3]-min_y[3]) + min_y[3])/ \
                #                (x[:,3]*(max_y[3]-min_y[3]) + min_y[3]) # D = F/V
                            D = z[:,1]/x[:,3]
                            inputs = x[:,0:3] #Consistent dims for dy calcs
                            dilute = torch.stack(( x[:,0]*D, -(z[:,0]-x[:,1])*D, x[:,2]*(D+0.000) )).T
                            terms = self.net(inputs) - dilute                          # Single NN
                            
                            dy = torch.cat((terms,dV),axis=1)
                            return dy
                        
                
                
                
                def makedirs(dirname):
                    if os.path.exists(dirname):
                        shutil.rmtree(dirname)
                        time.sleep(0.5)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                
                    
                #args.viz = 'Subplots'
                if args.viz:
                    makedirs('Visuals')
                
                def train(t, batch_t, batch_y, true_y, batch_y0, batch_z,
                          batch_MCt, batch_MCy, batch_MCy0, batch_MCz,N_Bsteps=None):
                    def closure():
                        optimizer.zero_grad()
                        if SIC == True:# and itr > 100:
                            pred_y = odeint(funct, batch_y0, batch_t, batch_z[:,:,:],
                                        N_steps=args.N_steps, method='dopri5')
                        else:
                            pred_y = batch_y
                        if MIC == True:
                            pred_MCy = odeint(funct, batch_MCy0, batch_MCt, batch_MCz,
                                          N_steps=args.N_steps, method='dopri5')
                        else:
                            pred_MCy = batch_MCy
                        ### Compute losses
                        losses = compute_losses(funct,pred_y[:,:,:],
                                                batch_y,true_y,lamb)
                        lossMC = torch.mean((pred_MCy-batch_MCy)**2)                                            
                        
                        extra_loss = 0.0;

                        loss = losses[0]*tot_lamb + extra_loss*.01 + lossMC*MC_lamb
                        
                        
                        loss.backward()
                        return loss
                
                    if batch_z == None:
                        # For argument consistency
                        batch_z = batch_y.detach().clone()#[:,idx,:]
                    N_Bsteps = len(batch_y[0,:,0])
                
                # Training 
                    N_vars = len(batch_y[0,0,:]);    N_inputs = N_vars;    N_outputs = N_vars#+N_zvars
                    if args.mod == 'Pen':
                        N_inputs = 3; N_outputs = 3
                    if args.mod == 'Sty':
                        N_inputs = 6; N_outputs = 6
                    
                    funct = ODEFunct(N_inputs,N_outputs,N_hnodes)
                    pred_y = odeint(funct, 
                                    batch_y0, batch_t, batch_z[:,:,:], 
                                    N_steps=args.N_steps, method='dopri5')
                    pred_MCy = odeint(funct, batch_MCy0, batch_MCt, batch_MCz,
                                      N_steps=args.N_steps, method='dopri5')
                    lossMC = torch.mean((pred_MCy-batch_MCy)**2)                    
                    print(f'Prediction before training: f([[2., 1.0, 0.0]]) = {pred_y[0][-1].detach().numpy()}') # [1.379,.379,0.621,1.121]
                    loss = torch.mean(torch.abs(pred_y[:,:,:] - batch_y[:,:,:]))
                    loss = torch.mean((pred_y[:,:,0:] - batch_y[:,:,0:])**2)
                    print('Iter {:04d} | Abs Loss {:.6f}'.format(0, loss.item()))
                
                # Training loop    
                    ii = 0
                    if args.viz != None:
                        vis = visuals(args,N_vars)
                    if N_hlayers == 1:
                        # Single NN
                        params = list(funct.parameters()) #+list([batch_y0])#+ list([k])
                        [w1,w2,w3,w4] = funct.parameters()
                    if N_hlayers == 2:
                        # Single NN, 2 hidden layers
                        params = list(funct.parameters()) #+list([batch_y0])#+ list([k])
                        [w1,w2,w3,w4,w5,w6]  = funct.parameters()
                    history = Store()
                    
                    # optimizer = torch.optim.SGD(params,lr=learning_rate)
                    optimizer = torch.optim.LBFGS(params, lr=learning_rate,history_size=10, max_iter=4) #AG default                
                #    optimizer = torch.optim.LBFGS([func.k], lr=learning_rate,history_size=10, max_iter=4) #AG default
                    tot_time = time.time()
                    # bat_t = torch.tensor(batch_t[:,0:1],dtype=torch.float32)
                    for itr in range(0, args.niters + 1):
                        optimizer.zero_grad()
                                                
                        #Mech loss term: combined direct + indirect approach
                        extra_loss = 0.0
                        # Integrate all intervals simultaneously
                        if SIC == True:
                            pred_y = odeint(funct, batch_y0, batch_t, 
                                        batch_z[:,:,:],N_steps=args.N_steps)
                        else:
                            pred_y = batch_y
                        if MIC == True:
                            pred_MCy = odeint(funct, batch_MCy0, batch_MCt, batch_MCz,
                                          N_steps=args.N_steps, method='dopri5')
                        else:
                            pred_MCy = batch_MCy
                
                        if sum(np.isnan(w1.detach().numpy())).any() > 0:
                            print("nans on the loose."); break
 
                        ### Calculate loss, store loss, check loss progress
                        losses = compute_losses(funct,pred_y[:,:,:],
                                                batch_y,true_y,lamb)
                        lossMC = torch.mean((pred_MCy-batch_MCy)**2)                                            

                        losses[0] = losses[0]*tot_lamb + extra_loss*.01 + lossMC*MC_lamb
                        stop = Check_point(history,losses)
                        history.l_update(losses)
                        if stop == True:
                            break

                        losses[0].backward()
                        # Stop sim here to determine param gradients wrt objective
                        if itr == 0 or itr == 100:
                            dum = optimizer.param_groups[0]['params'][3].grad.numpy()

                        # Update parameters, reset gradients
                        optimizer.step(closure)
                        # optimizer.step()
                        with torch.no_grad():
                            # Calculate loss  w/o regularization
                            if args.mod == 'daMKM':
                                l = torch.mean((pred_y[:,:,:] - batch_y[:,:,:])**2)
                            else:
                                l = torch.mean((pred_y[:,:,:] - batch_y[:,:,:])**2)
                            # Print loss every 4 iterations
                            if itr % args.test_freq == 0:
                                pred_y2d = MMS.inverse_transform(odeint(funct,batch_y0,batch_t, \
                                        batch_z[:,:,:],N_steps=args.N_steps).reshape((-1,N_vars)))
                                pred_y = pred_y2d.reshape((N_runs,N_Bsteps,N_vars))
                #                pred_y2d = MMS.inverse_transform(odeint(funct, batch_y0[0:1,:], t)[0])
                #                pred_dy = funct(t, pred_y[0]).detach().numpy()*(MMS.max_y - MMS.min_y) #forward func
                                print('Iter{:04d} | Sqrd Loss {:.6f} | Tot Loss {:6f}'.format(
                                        itr, l.item(), losses[0].item(),))
                                if sum(np.isnan(pred_y.detach().numpy())).any() > 0:
                                    print("nans on the loose."); #break
                                if args.viz != None:
                                    vis.visualize(t, true_y, true_t, pred_y, idx, ii)
                                    ii += 1
                # For plotting training progression on a single graph
                #                plt.plot(t,pred_y2d[:,0],'g',t,pred_y2d[:,1],'g')
                    tot_time = time.time() - tot_time
                    print('Sim time: ',tot_time)
                    return funct, [itr,losses[0].detach().numpy(),tot_time]
                
                # Initiate training
                idx = np.round(np.linspace(0, len(t) -1, N_dat)).astype(int) # 10 data points
                if __name__ == '__main__':
                    if args.mod == 0:#'Pen' or args.mod == 'Sty':
                        funct,res = train(t,true_t,batch_y[:,:,:],true_y[:,:,:],batch_y0,batch_z[:,:,:])
                    else:
                        funct,res = train(t,true_t,batch_y[:,:,:],sol[:,:,:],batch_y0,batch_z[:,:,:],
                                          true_MCt, batch_MCy, batch_MCy0, batch_MCz )
                    
                # =============================================================================
                # # Simulate results
                # =============================================================================
                # Simulate results with single IC
                y = []
                yscld = []
                
                if args.mod == 'Pen':
                    z = MMSz.transform(torch.tensor(solz,dtype=torch.float32))
                else:
                    z = sol #dummy var
                # ***This code is not efficient with too many runs*** 
                with torch.no_grad():
                    for N_run in range(0,len(torch_y[:,0,0].numpy())):
                        yscld.append(odeint(
                                funct, y0[N_run:N_run+1,:], t, z[N_run:N_run+1,:,:], 
                                N_steps=5)[0])
                        y.append(MMS.inverse_transform(odeint(
                                funct, y0[N_run:N_run+1,:], t, z[N_run:N_run+1,:,:], 
                                N_steps=5)[0]))
                pred_yscld = torch.tensor(np.stack(yscld),dtype=torch.float32)
                pred_y = torch.tensor(np.stack(y),dtype=torch.float32)
                
                # Calculate MSEs, true and noisy
                j = df_sol['Run'].max()+1
                MSE_Tscld = torch.sum((pred_yscld[:,idx,:]-MMS.transform(torch_y))**2).numpy()/(j*len(idx)*N_vars)
                MSE_Nscld = torch.sum((pred_yscld[:,idx,:]-MMS.transform(torch_y))**2).numpy()/(j*len(idx)*N_vars)
                MSE_T = torch.sum((pred_y[:,idx,:] - sol[:,idx,:])**2).numpy()/(j*len(idx)*N_vars)#All vars
                MSE_N = torch.sum((pred_y[:,idx,:] - torch_y[:,:,:])**2).numpy()/(j*len(idx)*N_vars)#All vars
                MAE = np.zeros(N_vars+1)
                # Don't need MAE until stage 2
                for N_var in range(0,N_vars):
                    MAE[N_var] = torch.sum(abs(pred_y[:,idx,N_var] - sol[:,idx,N_var])).numpy()/(j*len(idx))
                MAE[N_var+1] = sum(MAE)/N_vars
                                
                df_hyper.loc[row] = [args.mod,noise,lamb,N_hnodes,learning_rate,'tanh',
                                args.N_steps,res[0],res[1],res[2],MSE_T,MSE_N,
                                MAE[N_vars], MSE_Tscld, MSE_Nscld]
                # Make predictions
                # dy/dt predictions
                dy = []
                for N_run in range(0,len(torch_y[:,0,0].numpy())):
                    dum = MMS.transform(true_y[N_run])#,dtype=torch.float32)
                    # dum = MMS.transform(pred_y[N_run])#,dtype=torch.float32)
                    dy.append(funct(t,dum,z[N_run],t).detach().numpy()*(MMS.max_y.numpy() - MMS.min_y.numpy()))
                pred_dy = np.stack(dy)   
                pred_dy = torch.tensor(pred_dy,dtype=torch.float32)
                # if MIC == False: # N_runs has not changed
                MAE_dy = torch.sum(abs(pred_dy[:,idx,:] - true_dy[:,idx,:])).numpy()/(N_runs*len(idx)*N_vars)
                MAE_dy = torch.sum(abs(pred_dy[0,idx,:] - true_dy[0,idx,:])).numpy()/(1*len(idx)*N_vars)
                # =============================================================================
                # # Save models
                # =============================================================================

                # Create df of state and deriv results
                df_pred_dt = pd.DataFrame(data=pred_dy.numpy().reshape((-1,N_vars)),
                                         columns=derivs)
                df_pred_dt['t'] = np.tile(t[:],len(torch_y[:,0,0]))
                df_pred_dt['Run'] = np.repeat(np.arange(len(torch_y[:,0,0])),len(t))
                
                df_pred_y = pd.DataFrame(data=pred_y.numpy().reshape(N_t*len(torch_y[:,0,0]),N_vars)*1,
                                      columns=states)
                df_pred_y['t'] = np.tile(t,len(torch_y[:,0,0]))
                df_pred_y['Run'] = np.repeat(np.arange(len(torch_y[:,0,0])),N_t)
                
                dum = torch_y2d 
                dum2 = MMS.inverse_transform(dum).numpy()                    
                
                df_batch_y = pd.DataFrame(data=dum2,
                                      columns=states)
                df_batch_y['t'] = np.tile(t[idx],len(torch_y[:,0,0]))
                df_batch_y['Run'] = np.repeat(np.arange(len(torch_y[:,0,0])),len(idx))

                # If noisy MSE of current hyperparms is lowest, save MSEs + model
                if df_hyper.loc[row]['MSE noisy'] <= \
                    df_hyper[(df_hyper['Noise Lvl'] == noise)&(df_hyper['Model'] == args.mod)]['MSE noisy'].min():
                    # Save NODE model
                    torch.save(funct,'data/NODE{}{}N.pt'.format(args.mod,int(noise*100)))
                    # Load NODE model
                    funct = torch.load('data/NODE{}{}N.pt'.format(args.mod,int(noise*100)))
                    
                    def save_obj(obj, name ):
                        with open('data/'+ name + '.pkl', 'wb') as f:
                            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
                    def load_obj(name ):
                        with open('data/' + name + '.pkl', 'rb') as f:
                            return pickle.load(f)
                    
                    od = funct.state_dict()
                    for key, value in od.items():
                        od[key] = value.numpy()                        
                    od['scalers'] = MMS.max_y.numpy()-MMS.min_y.numpy()
                    # Save NODE weights+biases
                    save_obj(od,'dictNODE{}{}N'.format(args.mod,int(noise*100)))
                    # Test loading weights+biases
                    # dum = load_obj('dictNODE{}{}N'.format(args.mod,int(noise*100)))

                    # Save batch, predicted dydt and predicted y data respecitvely
                    if save_data == True: 
                        df_batch_y.to_csv(r'data/batch{}_{}Noise.csv'.format(args.mod,int(noise*100)),index= None, header=True)
                        df_pred_dt.to_csv(r'data/pred{}_dt{}Noise.csv'.format(args.mod,int(noise*100)),index = None, header=True)  #Change based on noise, # of samples, MVs
                        df_pred_y.to_csv(r'data/pred{}_y{}Noise.csv'.format(args.mod,int(noise*100)),index = None, header=True) 

                
                row = row+1
                N_run = 0

# Save table of tested hyperparameters                
# df_hyper.to_csv(r'tuningdata/hyperparam_res.csv',index = None, header=True) 
# sys.exit()

pars = funct.k.detach().numpy().reshape(1,-1)

from Visualize import plot_LoVody, plot_LoVo, plot_temp, plot_pred, plot_dy, \
                plotMKM, plot_MKMdy, plot_robert, plot_MAPK, plot_MAPKdy
# =============================================================================
# # Plot data
# =============================================================================
N_run = 0 # Choose which run to plot
title = 'Fitted NODE Simulation'
if args.mod == 'Sty':
    pred_y[:,:,0] = pred_y[:,:,0]+800;
    torch_y[:,:,0] = torch_y[:,:,0]+800;
    sol[:,:,0] = sol[:,:,0]+800;
    plot_temp(t,pred_y[N_run],torch_y[N_run],sol[N_run],idx,title)
    plot_pred(t,pred_y[N_run],torch_y[N_run],sol[N_run],idx,title)
    plot_dy(t,pred_dy[N_run],true_dy[N_run],idx,title,noise)
if args.mod == 'Pen':
    args.viz = 'Subplots'
    viz = visuals(args,N_vars)
    # viz.final_vis(t,true_y[0],idx,torch_y[0],true_t,pred_y[0:1],noise)
    viz.single_graph(t,true_y[N_run],idx,torch_y[N_run],pred_y[N_run],noise)
if args.mod == 'LoVo':
    plot_LoVo(t,pred_y[N_run],torch_y[N_run],true_y[N_run],idx,title,noise,N_hnodes,final=True)
    plot_LoVody(t,pred_dy[N_run],true_dy[N_run],idx,title,noise,N_hnodes,final=True)


sys.exit()