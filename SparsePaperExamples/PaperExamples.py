# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:14:42 2020

@author: Afbwi
"""
# This code worked with python-3.6.9, pytorch=1.3.1
import argparse
import time
import numpy as np
import pandas as pd
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from AG_Funs import odeint, Store, MMScaler, Check_point
from AG_Funs import vdp, FHN, sir_model

# Configure a few plotting variables
Reg_size = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

# =============================================================================
# # Set default hyperparameters and training conditions
# =============================================================================
noise = 0.0  #0, 0.05, 0.10  # Noise added to data (%/100)
# N_Bsteps = 3  # 1,3,4,     # Number of dps + 1 for each subinterval
rtol = 1e-6  # 1e-5, 1e-6    # Relative improvement training termination criteria
SIC = False                  # Use a single integration interval
MIC = True                   # Use multiple integration intervals
save_data = False            # Whether to save results to csv file

learning_rate = 0.1  # 0.1 #Optimzer learning rate
lamb = 1e-6  # 1e-4, 1e-6  #Parameter regularization term
N_hnodes = 20  # 10, 20    #Number of hidden nodes
N_hlayers = 2  # 1, 2      #Number of hidden layers
# N_runs = 3   #1,3, 5, 10 #Number of runs

# Create hyperparameter dataframe
columns = ['Model', 'Data pts/run', 'Data pts/interval', 'Runs', 'Noise Lvl',
           'Reg weight', 'Layers', 'Hid Nodes', 'Lrn rate', 'Activ Func',
           'Euler steps', 'Epochs', 'MSE Train', 'Sim Time', 'MSE true alldat', 'MSE noisy alldat',
           'MAE true alldat', 'MSE_dy pred alldat', 'MAE_dy pred alldat', 
           'MSE_dy dat alldat', 'MAE_dy dat alldat','stop criteria',
           'MSE fitdat', 'MAE fitdat', 'MSE_dy dat fitdat ', 'MAE_dy dat fitdat', 
           'MSE_dy pred fitdat', 'MAE_dy pred fitdat']
df_hyper = pd.DataFrame(columns=columns)

# More user-defined training conditions
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--niters', type=int, default=1040)  # changed from 2000
parser.add_argument('--test_freq', type=int, default=4)  # changed from 20
args = parser.parse_args()
args.plot_training = False
args.mod = 'fhn'

# =============================================================================
# # General data processing
# =============================================================================
row = 0  # row of hyper_param df

controls = []
if args.mod == 'vdp':
    args.true_mod = vdp
    file = 'data/vdp.csv'
    states = ['x1', 'x2']
    derivs = ['dx1dt', 'dx2dt']
    args.N_steps = 18
    N_dat = 10
    args.N_steps = 20
    N_dat = 10
    # args.N_steps = 60; N_dat = 3; #Cannot fit even w/10 runs
    # args.N_steps = 50; N_dat = 4; #Cannot fit even w/10 runs
    args.N_steps = 40; N_dat = 5
    # args.N_steps = 30; N_dat = 7
    # args.N_steps = 5; N_dat = 100;
    # args.N_steps = 80
if args.mod == 'fhn':
    args.true_mod = FHN
    file = 'data/fhn.csv'
    states = ['x1', 'x2']
    derivs = ['dx1dt', 'dx2dt']
    # args.N_steps = 30
    args.N_steps = 30
    # args.N_steps = 50
    N_dat = 10
if args.mod == 'sir':
    args.true_mod = sir_model
    file = 'data/sir.csv'
    states = ['S', 'I', 'R']
    derivs = ['dSdt', 'dIdt', 'dRdt']
    args.N_steps = 25#18
    N_dat = 10
if args.plot_training == True:
    fig = plt.figure()
    ax = fig.add_subplot(111)

df_sol = pd.read_csv(file)
t = df_sol[df_sol['Run'] == 0]['t'].to_numpy()  # [:20]#[::-1]#[5:]#[:20]
Tot_runs = df_sol['Run'].max()+1


# Test multiple hyperparameters
# for N_runs in [1,3,5,10]:
#     for N_dat in [3,5,8,10]:
#         for noise in [0.00,0.05,0.10]:
# Test a single set of hyperparameters
for N_runs in [10]:                     #Options: [1,3,5,10]
    for N_dat in [10]:                  #Options: [3,5,8,10]
        for noise in [0.05]:            #Options: [0.00,0.05,0.10]
            for N_hnodes in [20]:       #Options: [10,15,20]
                for N_hlayers in [2]:   #Options: [1,2]
                    # Fix seed so that Neural ODE param init and noise is the same each time
                    torch.manual_seed(34)
                    np.random.seed(84)
                    # args.N_steps = 50
                    
                    #Choose # of Steps
                    N_Bsteps = N_dat - 1
                    # N_Bsteps = N_dat - 2
                    # N_Bsteps = 8
                    # N_Bsteps = int(np.floor(N_dat/2))
                    
                    dum = np.arange(0,N_runs*(N_dat-N_Bsteps))*1.0 #5 runs, 3 dps = 0-9 = len 10
                    CVruns = dum.reshape(1,len(dum))
                    
                    N_t = len(t)                         # Number to time points
                    N_runs = df_sol['Run'].min()+N_runs  # Number of run conditions
                    N_MCruns = (N_runs)*(N_dat-N_Bsteps) # Number of total integration intervals
                    N_vars = len(states)                 # Number of state vars
                    N_zvars = len(controls)              # Number of control vars
                    print(N_runs,N_dat,noise,N_hnodes,N_hlayers,N_Bsteps)
                    
                    sol = np.ones((Tot_runs, N_t, N_vars))
                    solz = np.ones((Tot_runs, N_t, N_zvars))
                    sol_dy = np.ones((Tot_runs, N_t, N_vars))
                    for N_run in range(0, Tot_runs):
                        sol[N_run, :, :] = df_sol[df_sol['Run'] ==
                                                  N_run][states]  # [:20]#[::-1]#[5:]#[:20]
                        solz[N_run, :, :] = df_sol[df_sol['Run'] ==
                                                   N_run][controls]  # [:20]#[::-1]#[5:]#[:20]
                        sol_dy[N_run, :, :] = df_sol[df_sol['Run'] ==
                                                     N_run][derivs]  # [:20]#[::-1]#[5:]#[:20]
        
                    # Collect data -- Single IC
                    true_y0 = torch.tensor(
                        sol[:, 0, :], dtype=torch.float32)  # true init cond
                    true_z = torch.tensor(
                        solz[:, :, :], dtype=torch.float32)  # forcing vars
                    true_y = torch.tensor(
                        sol[:, :, :], dtype=torch.float32)  # true data
                    true_dy = torch.tensor(sol_dy[:, :, :], dtype=torch.float32)
                    max_y = np.amax(np.amax(true_y.numpy(), axis=1),
                                    axis=0)  # for adding noise
                    min_y = np.amin(np.amin(true_y.numpy(), axis=1),
                                    axis=0)  # for adding noise
        
                    idx = np.round(np.linspace(0, len(t) - 1, N_dat)
                                   ).astype(int)  # 10 data points
        
                    true_t = t[idx]  # [::-1].copy()
             
                    # Add noise
                    noisy_y = true_y[:, :, :]*1.0       # data + meas noise
                    k = N_vars-1 if args.mod == 'Pen' else N_vars
                    for i in range(0, k):
                        noisy_y[:, :, i:i+1] = noisy_y[:, :, i:i+1] + np.random.normal(loc=0.0,
                                        scale=noise*(max_y[i]-min_y[i]), size=(Tot_runs, len(t), 1))
                        # Used to compare increasing the length of integration intervals
                        # batch_y[:,:,i:i+1] = batch_y[:,:,i:i+1] + np.random.normal(loc=0.0,
                        #        scale=noise*(max_y[i]-min_y[i]),size=(N_runs,len(idx),1))

                    batch_y = noisy_y[:, idx, :]
                    # Create 2d ground truth
                    true_dy2d = true_dy.reshape((-1, N_vars))
        
                    # Create var for y data after adding noise, before rearranging and scaling
                    torch_y = batch_y[:, :, :].detach().clone()
                    
                    # Scaling       
                    MMS = MMScaler(args)  # MinMaxScaler()
                    MMS.fit(batch_y[0:1])
                    batch_y2d = MMS.transform(batch_y.reshape((-1, N_vars)))  # ,
        
                    batch_y = batch_y2d.reshape(
                        (Tot_runs, len(idx), N_vars)).clone().detach()
                    true_y2d = true_y[:, idx, :].reshape((-1, N_vars)).clone().detach()
                    # ,       #noise added and scaled
                    torch_y2d = MMS.transform(torch_y.reshape((-1, N_vars)))
                    batch_y0 = torch.tensor(batch_y[:, 0, :].numpy(
                                ), dtype=torch.float32, requires_grad=True)
                    batch_z = true_y.clone().detach()  # dummy var for arg consistency
        
                    # Create a time variable specifically for plotting progression of NODE training
                    # Works when N_dat = 3, 4, or 10
                    if args.plot_training == True:
                        t_plot = []
                        for i in range(0, N_dat - N_Bsteps):
                            t_plot.append(t[idx[i]:idx[i+N_Bsteps]])
                            t[0:50].shape
                            t[25:75].shape
                            t[50:100].shape
                        # only works when all time intervals are the same
                        t_plot = np.stack(t_plot, axis=0)
        
                    # Collect data -- Multiple ICs
                    true_MCy = []
                    true_MCz = []
                    true_MCt = []  # shape: (Tot_runs*N_dat,N_Bsteps+1)
                    bat_MCy = []
                    batch_MCz = []
                    for j in range(0, Tot_runs):
                        # idx:#range(0,N_t - N_Bsteps):
                        for i in range(0, N_dat - N_Bsteps):
                            true_MCt.append(t[idx][i:i+N_Bsteps+1])  # [::-1])
                            true_MCy.append(sol[j][idx][i:i+N_Bsteps+1, :])
                            true_MCz.append(solz[j][idx][i:i+N_Bsteps+1, :])
                            bat_MCy.append(batch_y[j][i:i+N_Bsteps+1, :])
                            batch_MCz.append(batch_z[j][i:i+N_Bsteps+1, :])
                    true_MCy = torch.tensor(
                        np.stack(true_MCy, axis=0), dtype=torch.float32)
                    true_MCz = torch.tensor(
                        np.stack(true_MCz, axis=0), dtype=torch.float32)
                    true_MCt = np.stack(true_MCt, axis=0)
                    batch_MCy = torch.stack(bat_MCy, axis=0)
                    batch_MCz = torch.stack(batch_MCz, axis=0)
                    MCidx = np.round(np.linspace(0, N_Bsteps, N_Bsteps+1)).astype(int)
                    true_MCy0 = true_MCy[:, 0, :].detach().clone()
                    batch_MCy0 = torch.tensor(
                        batch_MCy[:, 0, :].numpy(), dtype=torch.float32, requires_grad=True)
        
                    y0 = torch.tensor(sol[:,0,:], dtype=torch.float32) # MMS.transform(torch_y[:, 0, :])
                    # sys.exit()
                    # =============================================================================
                    # # Set-up purely data-driven Neural ODE
                    # =============================================================================
                    class ODEFunct(nn.Module):  # Nueral ODE Function
                        def __init__(self):
                            super(ODEFunct, self).__init__()
                            if N_hlayers == 1:
                                self.net = nn.Sequential(
                                    nn.Linear(N_vars, N_hnodes),
                                    nn.Tanh(),
                                    nn.Linear(N_hnodes, N_vars),)
                            if N_hlayers == 2:
                                self.net = nn.Sequential(
                                    nn.Linear(N_vars, N_hnodes),
                                    nn.Tanh(),
                                    nn.Linear(N_hnodes, N_hnodes),
                                    nn.Tanh(),
                                    nn.Linear(N_hnodes, N_vars),)
        
                            for m in self.net.modules():
                                if isinstance(m, nn.Linear):
                                    nn.init.normal_(m.weight, mean=0, std=0.2)
                                    nn.init.constant_(m.bias, val=0)
                        # Black-box Model
                        def forward(self, N_t, x, z=0, t=0):
                            terms = self.net(x)
                            return terms
        
        
                    def closure():
                        optimizer.zero_grad()
                        if SIC == True:
                            pred_y = odeint(funct, batch_MCy0[0::N_dat-N_Bsteps], true_t,
                                            N_steps=args.N_steps)
                        else:
                            pred_y = batch_y
        
                        if MIC == True:
                            pred_MCy = odeint(funct, batch_MCy0, true_MCt, batch_MCz,
                                              N_steps=args.N_steps, method='dopri5')
                        else:
                            pred_MCy = batch_MCy  # for consistency
                        # Compute reg loss
                        reglos = 0
                        for param in funct.parameters():
                            reglos += torch.sum(param**2)
                        # reglos = torch.sum(w1**2) + torch.sum(w2**2) + \
                        #     torch.sum(w3**2) + torch.sum(w4**2)
                        # Compute data loss
        
                        # + torch.mean((pred_ys[idx,:] - batch_y)**2)
                        los = torch.mean((pred_y[0:N_runs, :, :] - batch_y[0:N_runs])**2)
                        lossMC = torch.mean((pred_MCy[CVruns]-batch_MCy[CVruns])**2)
                        losses = los + lossMC + lamb*reglos/N_params
        
                        losses.backward()
                        return losses
        
                    def compute_losses(model, pred_y, batch_y, true_y):
                        # All losses are 2-dimensional and indexed
                        # Regulaization loss
                        reglos = 0
                        for param in funct.parameters():
                            reglos += torch.sum(param**2)
                        # reglos = torch.sum(w1**2) + torch.sum(w2**2) + \
                        #     torch.sum(w3**2) + torch.sum(w4**2)
                        # Loss in loss function
                        tot_loss = torch.mean(((pred_y - batch_y))**2) + lamb*reglos
                        with torch.no_grad():
                            # Loss between prediction and noisy batch data
                            los_raw = torch.mean((pred_y - batch_y)**2)
                            # Loss between Derivative prediction and noisy batch data
                            # pred_dy = model(t, batch_y).detach().numpy()*(max_y - min_y)
                            # pred_dy = torch.tensor(pred_dy,dtype=torch.float32)
                            # torch.mean((pred_dy - batch_dy)**2) #this doesn't mean much
                            losdy_raw = torch.tensor(1, dtype=torch.float32)
        
                            # Loss between Derivative pred and true deriatives
                            # pred_dy = model(t, pred_y).detach().numpy()*(max_y - min_y) #forward func
                            # pred_dy = torch.tensor(pred_dy,dtype=torch.float32)
                            # torch.mean((pred_dy - true_dy2d)**2)
                            losdy = torch.tensor(1, dtype=torch.float32)
                        # Loss between pred and true data
                        with torch.no_grad():
                            true_y = MMS.transform(true_y)
                            los = torch.mean((pred_y - true_y)**2)
                        losses = [tot_loss, reglos, los_raw, losdy_raw, los, losdy]
                        stop = Check_point(history, losses, rtol)
                        history.l_update(losses)
                        return [tot_loss, stop]
        
                    # =============================================================================
                    # # Training of Neural ODE
                    # =============================================================================
                    funct = ODEFunct()
                    pred_y = odeint(funct, batch_y0, true_t,
                                    N_steps=args.N_steps, method='dopri5')
                    print(
                        f'Prediction before training: f([[2., 1.0, 0.0]]) = {pred_y[0][-1].detach().numpy()}')
                    loss = torch.mean(torch.abs(pred_y[0][:, :] - batch_y[0][:, :]))
                    loss = torch.mean((pred_y[:, :, 0:] - batch_y[:, :, 0:])**2)
                    print('Iter {:04d} | Abs Loss {:.6f}'.format(0, loss.item()))
        
                    if __name__ == '__main__':
                        ii = 0
        
                        params = list(funct.parameters()) + list([batch_MCy0])  + list([batch_y0])# + list([k])
                        N_params = 0
                        for param in params:
                            N_params += len(param)
                        # [w1,w2,w3,w4] = funct.parameters()
                        history = Store()
                        # Select optimizer
                    #    optimizer = torch.optim.SGD(params,lr=learning_rate)
                        optimizer = torch.optim.LBFGS(
                            params, lr=learning_rate, history_size=10, max_iter=4)  # AG default
                    #    optimizer = optim.RMSprop(params, lr=1e-2, weight_decay=0.5)
                        end = time.time()
                        tot_time = time.time()
        
                        for itr in range(0, args.niters + 0):
                            ii += 1
                            optimizer.zero_grad()
        
                            if SIC == True:
                                pred_y = odeint(funct, batch_MCy0[0::N_dat - N_Bsteps], true_t,
                                                N_steps=args.N_steps)
                                pred_y2d = pred_y[:, :, :].reshape((-1, N_vars))
                                # Includes data + reglos, for checking stopping criteria only
                                tot_loss, stop = compute_losses(funct, pred_y2d[0:N_runs*N_dat], 
                                                    batch_y2d[0:N_runs*N_dat], true_y2d[0:N_runs*N_dat])
                            else:
                                pred_y = batch_y
                                pred_y2d = batch_y2d
        
                            if MIC == True:
                                pred_MCy = odeint(funct, batch_MCy0, true_MCt, batch_MCz,
                                                  N_steps=args.N_steps, method='dopri5')
        
                                pred_MCy2d = pred_MCy[CVruns].reshape((-1, N_vars))
                                # Includes data + reglos, for checking stopping criteria only
                                tot_lossMC, stop = compute_losses(funct, pred_MCy2d,
                                                    batch_MCy[CVruns].reshape((-1, N_vars)),
                                                    true_MCy[CVruns].reshape((-1, N_vars)))
        
                            else:
                                pred_MCy = batch_MCy  # for consistency
                            # Compute reg loss
                            reglos = 0
                            for param in funct.parameters():
                                reglos += torch.sum(param**2)
                            # reglos = torch.sum(w1**2) + torch.sum(w2**2) + \
                            #     torch.sum(w3**2) + torch.sum(w4**2)
                            # Compute data loss
                            los = torch.mean((pred_y[0:N_runs, :, :] - batch_y[0:N_runs])**2)
                            lossMC = torch.mean((pred_MCy[CVruns]-batch_MCy[CVruns])**2)
                            losses = los + lossMC + lamb*reglos/N_params

                            if stop != False:
                                break
                            losses.backward()
                            optimizer.step(closure)
                    #        optimizer.step()
                            with torch.no_grad():
                                l = torch.mean(
                                    torch.abs(pred_y[0][:, :] - batch_y[0][:, :]))
                                if itr % args.test_freq == 0:
                                    # Print progress of training
                                    print('Iter {:04d} | Abs Loss {:.6f} | Tot Loss {:6f}'.format(
                                        itr, l.item(), losses.item()))
                                    # Plot progress of training
                                    if args.plot_training == True:
                                        plt.cla()
                                        plt.plot(t[:], true_y[0][:, 0], 'b--', t[idx], MMS.inverse_transform(batch_y[0])[:, 0:1], 'ko',
                                                 t[:], true_y[0][:, 1], 'r--', t[idx], MMS.inverse_transform(batch_y[0])[:, 1:2], 'ko')
                                        pred_y = MMS.inverse_transform(
                                            odeint(funct, batch_y0, t, N_steps=args.N_steps)[0])
                                        pred_MCy = MMS.inverse_transform(odeint(funct,
                                                                                batch_MCy0[0:N_dat -
                                                                                            N_Bsteps], t_plot,
                                                                                N_steps=args.N_steps, method='dopri5'))
                        #                pred_dy = funct(t, pred_y[0]).detach().numpy()*(max_y - min_y) #forward func
                                        # plt.figure()
                                        if SIC == True:
                                            plt.plot(
                                                t, pred_y[:, 0], 'g', t, pred_y[:, 1], 'g')
                                        if MIC == True:
                                            plt.plot(t_plot[:, :].T,
                                                      pred_MCy[0:N_dat -
                                                              N_Bsteps, :, 0].T, 'g',
                                                      t_plot[:, :].T,
                                                      pred_MCy[0:N_dat-N_Bsteps, :, 1].T, 'g')
                                        # Save figure
                                        plt.text(1,1,'Epoch: ' + str(itr),va='top',ha='right',
                                                 fontsize=MEDIUM_SIZE,transform=ax.transAxes)
                                        plt.xlabel('t')
                                        plt.ylabel('x,y')
                                        plt.savefig('GIF_visuals/{}{}stps_{:03d}'.format(args.mod,N_Bsteps,itr),dpi=300)
                                    # sys.exit()
        
                            end = time.time()  # duration of fitting
                        tot_time = time.time() - tot_time
                    print('Sim time: ', tot_time)
                    if itr == args.niters:
                        stop = 'epoch limit reached'
                        print('Warning: max # of iters reached')
                        # ii = args.iters
                    # sys.exit()
        
                    # =============================================================================
                    # # Simulate results
                    # =============================================================================
                    # Simulate results with single IC
                    y = []
                    yscld = []
        
                    with torch.no_grad():
                        pred_yscld = odeint(funct, batch_MCy0[0::N_dat-N_Bsteps], t, N_steps=args.N_steps)
                        pred_y = MMS.inverse_transform(pred_yscld)
        
                    # Calculate MSEs, true (T) and noisy (N), of state predictions
                    j = 10
                    MSE_T = torch.sum((pred_y[0:j, idx, :] - sol[0:j, idx, :])
                                      **2).numpy()/(j*len(idx)*N_vars)  # All vars
                    MSE_N = torch.sum((pred_y[0:j, idx, :] - torch_y[0:j, :, :])
                                      ** 2).numpy()/(j*len(idx)*N_vars)  # All vars
                    MAE_T = np.zeros(N_vars+1)
                    # Don't need MAE until stage 2
                    for N_var in range(0, N_vars):
                        MAE_T[N_var] = torch.sum(
                            abs(pred_y[0:j, idx, N_var] - sol[0:j, idx, N_var])).numpy()/(j*len(idx))
                    MAE_T[N_var+1] = sum(MAE_T)/N_vars  # Add avg MAE at the end
                    # Fitting dataset
                    j = N_runs
                    MSE_Tscldf = torch.sum(
                        (pred_yscld[0:j, idx, :]-batch_y[0:j, :, :])**2).numpy()/(j*len(idx)*N_vars)
                    MSE_Nf = torch.sum(
                        (pred_y[0:j, idx, :] - torch_y[0:j, :, :])**2).numpy()/(j*len(idx)*N_vars)  # All vars
                    MAE_Nf = np.zeros(N_vars+1)
                    # Don't need MAE until stage 2
                    for N_var in range(0, N_vars):
                        MAE_Nf[N_var] = torch.sum(
                            abs(pred_y[0:j, idx, N_var] - torch_y[0:j, :, N_var])).numpy()/(j*len(idx))
                    MAE_Nf[N_var+1] = sum(MAE_T)/N_vars  # Add avg MAE at the end
        
                    # Make dy/dt predictions
        
                    # Using measured data
                    j = 10  # Number of runs to include in error calc
                    with torch.no_grad():
                        # dum = MMS.transform(batch_y) #true_y)
                        #pred_dy = MMS.inverse_transform(funct(t, dum).detach().numpy())
                        pred_dy = funct(t, batch_y).detach().numpy() * \
                            (MMS.max_y.numpy() - MMS.min_y.numpy())
                    pred_dy = torch.tensor(pred_dy, dtype=torch.float32)
                    true_dy = np.zeros(batch_y.shape)
                    for N_run in range(0, df_sol['Run'].max()+1):
                        # *(MMS.max_y.numpy() - MMS.min_y.numpy())
                        true_dy[N_run] = args.true_mod(
                            t, torch_y[N_run]).detach().numpy()
        
                    # Calculate MSEs. Derivs from measurments. All run
                    MSE_dy = torch.sum(
                        (pred_dy[0:j, :, :] - true_dy[0:j, :, :])**2).numpy()/(j*len(idx)*N_vars)  # All vars
                    MAE_dy = torch.sum(
                        abs(pred_dy[0:j, :, :] - true_dy[0:j, :, :])).numpy()/(j*len(idx)*N_vars)
                    # Calculate MSEs. Derivs from measurments. Fitted runs
                    j = N_runs
                    MSE_dyf = torch.sum(
                        (pred_dy[0:j, :, :] - true_dy[0:j, :, :])**2).numpy()/(j*len(idx)*N_vars)  # All vars
                    MAE_dyf = torch.sum(
                        abs(pred_dy[0:j, :, :] - true_dy[0:j, :, :])).numpy()/(j*len(idx)*N_vars)
                    # sys.exit()
        
                    # Using NODE predicted states
                    j = 10
                    with torch.no_grad():
                        dum = MMS.transform(pred_y)  # scale pred_y again
                        #pred_dy = MMS.inverse_transform(funct(t, dum).detach().numpy())
                        pred_dy = funct(t, dum).detach().numpy() * \
                            (MMS.max_y.numpy() - MMS.min_y.numpy())
                    pred_dy = torch.tensor(pred_dy, dtype=torch.float32)
                    true_dy = np.zeros(pred_y.shape)
                    for N_run in range(0, df_sol['Run'].max()+1):
                        # *(MMS.max_y.numpy() - MMS.min_y.numpy())
                        true_dy[N_run] = args.true_mod(
                            t, pred_y[N_run]).detach().numpy()
        
                    # Calculate MSEs. Derivs from state predictions (i.e. not vs noisy measured data).
                    # All runs
                    MSE_dyp = torch.sum(
                        (pred_dy[0:j, idx, :] - true_dy[0:j, idx, :])**2).numpy()/(j*len(idx)*N_vars)  # All vars
                    MAE_dyp = torch.sum(
                        abs(pred_dy[0:j, idx, :] - true_dy[0:j, idx, :])).numpy()/(j*len(idx)*N_vars)
                    # Fitted runs
                    j = N_runs
                    MSE_dypf = torch.sum(
                        (pred_dy[0:j, idx, :] - true_dy[0:j, idx, :])**2).numpy()/(j*len(idx)*N_vars)  # All vars
                    MAE_dypf = torch.sum(
                        abs(pred_dy[0:j, idx, :] - true_dy[0:j, idx, :])).numpy()/(j*len(idx)*N_vars)
        
                    # Assemble calcs
                    df_hyper.loc[row] = [args.mod, N_dat, N_Bsteps+1, N_runs, noise, lamb, N_hlayers,
                                          N_hnodes, learning_rate, 'tanh',
                                          args.N_steps, itr, losses.detach().numpy(), tot_time, MSE_T,
                                          MSE_N, MAE_T[N_vars], MSE_dyp, MAE_dyp, MSE_dy, MAE_dy,
                                          stop, MSE_Nf, MAE_Nf[N_vars], MSE_dyf, MAE_dyf, MSE_dypf, MAE_dypf]
                    row = row+1
                    # =============================================================================
                    # # Save results in csv files
                    # =============================================================================
                    # Create df of state and deriv results
                    # 'NODE deriv predictions'
                    df_pred_dt = pd.DataFrame(data=pred_dy.reshape((-1, N_vars)),
                                              columns=derivs)
                    df_pred_dt['t'] = np.tile(t[:], len(torch_y[:, 0, 0]))
                    df_pred_dt['Run'] = np.repeat(np.arange(len(torch_y[:, 0, 0])), len(t))
                    # 'NODE state predictions'
                    df_pred_y = pd.DataFrame(data=pred_y.reshape(-1, N_vars)*1,
                                              columns=states)
                    df_pred_y['t'] = np.tile(t, len(torch_y[:, 0, 0]))
                    df_pred_y['Run'] = np.repeat(np.arange(len(torch_y[:, 0, 0])), N_t)
                    # '(Noisy) Training data
                    df_batch_y = pd.DataFrame(data=torch_y2d.numpy(),
                                              columns=states)
                    df_batch_y['t'] = np.tile(t[idx], len(torch_y[:, 0, 0]))
                    df_batch_y['Run'] = np.repeat(np.arange(len(torch_y[:, 0, 0])), len(idx))
                    # ' True derivs from NODE state predictions
                    df_true_dt = pd.DataFrame(data=true_dy.reshape((-1, N_vars)),
                                                columns=derivs)
                    df_true_dt['t'] = np.tile(t[:], len(torch_y[:, 0, 0]))
                    df_true_dt['Run'] = np.repeat(np.arange(len(torch_y[:,0,0])), len(t))
                    
                    # Save batch, predicted dydt and predicted y data respectively
                        # Save based on model, amount of noise, dps, and runs
                    if save_data == True:
                        df_batch_y.to_csv(r'data/batch{}_{}N{}Dps{}Rns{}HN{}HL{}stps.csv'.format(
                            args.mod, int(noise*100), N_dat, N_runs, N_hnodes, N_hlayers,N_Bsteps),
                            index=None, header=True)
                        df_pred_dt.to_csv(r'data/pred{}_dt{}N{}Dps{}Rns{}HN{}HL{}stps.csv'.format(
                            args.mod, int(noise*100), N_dat, N_runs, N_hnodes, N_hlayers,N_Bsteps),
                            index=None, header=True)  # Change based on noise, # of samples, MVs
                        df_pred_y.to_csv(r'data/pred{}_y{}N{}Dps{}Rns{}HN{}HL{}stps.csv'.format(
                            args.mod, int(noise*100), N_dat, N_runs, N_hnodes, N_hlayers,N_Bsteps),
                            index=None, header=True)
                        df_true_dt.to_csv(r'data/true{}_dt{}N{}Dps{}Rns{}HN{}HL{}stps.csv'.format(
                            args.mod, int(noise*100), N_dat, N_runs, N_hnodes, N_hlayers,N_Bsteps),
                            index=None, header=True)
                    
                    
                    # 'All (noisy) training and non-training data'
                    df_noisy_y = pd.DataFrame(data=noisy_y.reshape((-1, N_vars)).numpy(),
                                              columns=states)
                    df_noisy_y['t'] = np.tile(t, len(noisy_y[:, 0, 0]))
                    df_noisy_y['Run'] = np.repeat(
                        np.arange(len(noisy_y[:, 0, 0])), len(t))
                    if save_data == True:
                        df_noisy_y.to_csv(r'data/data{}_{}N.csv'.format(
                            args.mod, int(noise*100)), index=None, header=True)
                    # sys.exit()
# sys.exit()
# df_hyper.to_csv(r'data/hyperparam_res_{}_halfstps.csv'.format(args.mod),index = None, header=True) 
# =============================================================================
# # Let's visualize those results!
# =============================================================================
# Create a loop that saves incremental integrations
if args.plot_training == True:
    for i in range(1,N_dat+1):
        with torch.no_grad():
            N_run = 0
            pred_yscld = odeint(funct, batch_MCy0[0::N_dat-N_Bsteps], t, N_steps=args.N_steps)
            pred_y = MMS.inverse_transform(pred_yscld)
            pred_MCy = MMS.inverse_transform(odeint(funct,
                                        batch_MCy0[0:N_dat - N_Bsteps], t_plot,
                                        N_steps=args.N_steps, method='dopri5'))

        plt.figure()
        plt.plot(t[:], true_y[0][:, 0], 'b--', t[idx], MMS.inverse_transform(batch_y[0])[:, 0:1], 'ko',
                 t[:], true_y[0][:, 1], 'r--', t[idx], MMS.inverse_transform(batch_y[0])[:, 1:2], 'ko')
    
        plt.plot(t_plot[:, :].T,
                  pred_MCy[0:N_dat -
                          N_Bsteps, :, 0].T, 'g',
                  t_plot[:, :].T,
                  pred_MCy[0:N_dat-N_Bsteps, :, 1].T, 'g')
        plt.plot(t[0:int(100*i/N_dat)], pred_y[N_run, 0:int(100*i/N_dat), 0], 'k',linewidth=3)
        plt.plot(t[0:int(100*i/N_dat)], pred_y[N_run, 0:int(100*i/N_dat), 1], 'k',linewidth=3)
        
        # Save figure
        # plt.text(1,1,'Epoch: ' + str(itr),va='top',ha='right',
        #          fontsize=MEDIUM_SIZE,transform=ax.transAxes)
        plt.xlabel('t')
        plt.ylabel('x,y')
        plt.savefig('GIF_visuals/Sim_{}{}stps_{:03d}'.format(args.mod,N_Bsteps,i),dpi=300)


N_run = 0

plt.plot(t[:], true_y[N_run][:, 0], 'b', t[idx], true_y[N_run][idx, 0], 'ko',
         t[:], true_y[N_run][:, 1], 'b', t[idx], true_y[N_run][idx, 1], 'ko')
plt.plot(t, pred_y[N_run, :, 0], 'r--', t,
         pred_y[N_run, :, 1], 'r--', label='final')
plt.legend(['Predict x1', 'Predict x2'])
plt.title('Training progression Neural ODE')
plt.xlabel('t')
plt.ylabel('x1,x2')
plt.show()

# dy graphs: dy from true non-noisy data (more accurate than dy pred from pred states)
plt.figure()
dum = MMS.transform(true_y[N_run])
#pred_dy = MMS.inverse_transform(funct(t, dum).detach().numpy())
pred_dy = funct(t, dum).detach().numpy()
pred_dy = pred_dy*(MMS.max_y.numpy() - MMS.min_y.numpy())
true_dy = args.true_mod(t, true_y[N_run]).detach().numpy()#*(MMS.max_y.numpy() - MMS.min_y.numpy())
dum2 = np.mean((true_dy-pred_dy)**2)
plt.plot(t, pred_dy[:, 0], 'b', t, pred_dy[:, 1], 'r')
if N_vars > 2:
    plt.plot(t, pred_dy[:, 2], 'c')  # ,t,pred_dy[:,3],'g')
# plt.plot(t,pred_dy[:,4],'p')
plt.title('Trajectories NN+ODE vs true')
plt.xlabel('t')
plt.ylabel('dX/dt')
plt.plot(t[:], true_dy[:, 0], 'bo', t[:], true_dy[:, 1], 'ro')
plt.legend(['dxdt pred', 'dydt pred'])
if N_vars > 2:
    plt.plot(t[:], true_dy[:, 2], 'co')
    plt.legend(['dx/dt pred', 'dy/dt pred', 'dz/dt pred'])  # ,'dCddt','dCedt','true data'])
plt.show()

# dy graphs: dy from NODE state predictions (more accurate than dy pred from noisy data)
plt.figure()
dum = MMS.transform(pred_y[N_run])
#pred_dy = MMS.inverse_transform(funct(t, dum).detach().numpy())
pred_dy = funct(t, dum).detach().numpy()
pred_dy = pred_dy*(MMS.max_y.numpy() - MMS.min_y.numpy())
true_dy = args.true_mod(t, pred_y[N_run]).detach().numpy()#*(MMS.max_y.numpy() - MMS.min_y.numpy())
np.mean((true_dy-pred_dy)**2)
plt.plot(t, pred_dy[:, 0], 'b', t, pred_dy[:, 1], 'r')
if N_vars > 2:
    plt.plot(t, pred_dy[:, 2], 'c')  # ,t,pred_dy[:,3],'g')
# plt.plot(t,pred_dy[:,4],'p')
plt.title('Trajectories NN+ODE vs pred')
plt.xlabel('t')
plt.ylabel('dX/dt')
plt.plot(t[:], true_dy[:, 0], 'bo', t[:], true_dy[:, 1], 'ro')
plt.legend(['dxdt pred', 'dydt pred'])
if N_vars > 2:
    plt.plot(t[:], true_dy[:, 2], 'co')
    plt.legend(['dx/dt pred', 'dy/dt pred', 'dz/dt pred'])  # ,'dCddt','dCedt','true data'])
plt.show()

# y graphs
plt.figure()
plt.plot(t, pred_y[N_run, :, 0], 'b', t, pred_y[N_run, :, 1], 'r')
if N_vars > 2:
    plt.plot(t, pred_y[N_run, :, 2], 'c')  # ,t,pred_y[N_run,:,3],'g')
# plt.plot(t,pred_y[:,4],'p')
plt.title('NN+ODE vs meas data')
plt.xlabel('t')
plt.ylabel('X')
plt.plot(t[idx], MMS.inverse_transform(batch_y[N_run])[:, 0:1], 'bo',
         t[idx], MMS.inverse_transform(batch_y[N_run])[:, 1:2], 'ro')
plt.legend(['x pred', 'y pred'])
if N_vars > 2:
    plt.plot(t[idx], MMS.inverse_transform(batch_y[N_run])[:, 2:3], 'co')
# plt.plot(t[idx],true_y[0][idx,0],'bo',t[idx],true_y[0][idx,1],'ro',
#         t[idx],true_y[0][idx,2],'co')
    plt.legend(['x pred', 'y pred', 'z pred'])  # ,'Cd','Ce','true data'])
plt.show()


