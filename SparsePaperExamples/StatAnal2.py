# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:11:37 2023

@author: WilliamTBradley
"""

'''
Purpose of code is to compare Neural ODEs with splines (i.e., Figure 10 in paper)
'''


import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

# =============================================================================
# # User-defined analysis parameters
# =============================================================================
mod = 'fhn' #Either 'fhn', 'sir', or 'vdp'

# =============================================================================
# # Set up error comparison and compare errors of splines vs NODEs
# =============================================================================
# Define true models
def vdp(t, z,):
    x = z[:,0]
    y = z[:,1]
    mu=1.5
    dxdt = y
    dydt = mu*(1 - x**2)*y - x # mu*y - mu*x**2*y - x
    dydx=np.stack((dxdt,dydt),)
    return dydx
def sir_model(t,y,):
    S = y[:,0]
    I = y[:,1]
    # R = y[:,2]
    beta = 6 #0.3
    gamma = 2.3#0.1
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    dydx = np.stack((dSdt,dIdt,dRdt),)
    return dydx
def FHN(t,z):
    a =0.2; b=0.2; c=3;
    x1 = z[:,0]
    x2 = z[:,1]
    dx1dt = c*(x1 - x1**3/3 + x2)
    dx2dt = -(1/c)*(x1 - a + b*x2)
    dydx=np.stack((dx1dt,dx2dt),)
    return dydx

if mod == 'vdp':
    file = 'data/vdp.csv'
    states = ['x1', 'x2']
    derivs = ['dx1dt', 'dx2dt']
    N_vars = 2
    true_mod = vdp
if mod == 'sir':
    file = 'data/sir.csv'
    states = ['S','I','R']
    N_vars = 3
    true_mod = sir_model
if mod == 'fhn':
    file = 'data/fhn.csv'
    states = ['x1', 'x2']
    derivs = ['dx1dt', 'dx2dt']
    N_vars = 2
    true_mod = FHN
    
# Import true data
df_sol = pd.read_csv(file)
t = df_sol[df_sol['Run'] == 0]['t'].to_numpy()  # [:20]#[::-1]#[5:]#[:20]
Tot_runs = df_sol['Run'].max()+1
N_t = len(t)               

columns_MSE = ['N_dat','Runs','Noise','MSE_dy spl','MSE_dy node','Result']
df_MSE = pd.DataFrame(columns=columns_MSE)
file =  'data/hyperparam_res_{}.csv'.format(mod) 

# Compare spline and Neural ODE estimates
df_hyper = pd.read_csv(file)
for noise in [0.00,0.05,0.10]:
    for N_runs in [1,3,5,10]:
        for N_dat in [3,5,10]:
# for N_runs in [1,3,5,10]:
#     for N_dat in [3,5,10]:
#         for noise in [0.00,0.05,0.10]:
# for N_runs in [10]:#[1,3,5,10]:
#     for N_dat in [10]:#[3,5,10]:
#         for noise in [0.00]:#[0.00,0.05,0.10]:
            idx = np.round(np.linspace(0, len(t) - 1, N_dat)).astype(int)

            # NODEs
            df_abr = df_hyper[df_hyper['Runs'] == N_runs]
            df_abr = df_abr[df_abr['Noise Lvl'] == noise]
            # df_abr = df_abr[df_abr['stop criteria'] == 'converge']
            dum = df_abr[df_abr['Data pts/interval'] == N_dat - 1]['MSE fitdat']
            idxmin = df_abr[df_abr['Data pts/interval'] == N_dat - 1]['MSE fitdat'].idxmin()
            MSEn_dy = df_abr.loc[idxmin]['MSE_dy pred fitdat']
            # sample1 = df_abr[df_abr['Data pts/interval'] == 4]['MSE_dy pred fitdat'].to_numpy()

            # splines
            file = 'newcasestudies_csv_results/data{}_{}N{}dps{}rns.csv'.format(
                mod,int(noise*100),N_dat,N_runs)
            df_spl = pd.read_csv(file)
            
            if mod == 'vdp' or mod == 'fhn':
                if noise == 0:
                    spl_states = ['x1_spl','x2_spl']
                    spl_derivs = ['dx1_spl','dx2_spl']
                elif noise > 0:
                    spl_states = ['x1_spl_smoothed','x2_spl_smoothed']
                    spl_derivs = ['dx1_spl_smoothed','dx2_spl_smoothed']
            elif mod == 'sir':
                if noise == 0:
                    spl_states = ['S_spl','I_spl','R_spl']
                    spl_derivs = ['dS_spl','dI_spl','dR_spl']
                elif noise > 0:
                    spl_states = ['S_spl_smoothed','I_spl_smoothed','R_spl_smoothed']
                    spl_derivs = ['dS_spl_smoothed','dI_spl_smoothed','dR_spl_smoothed']              
            
            true_y = np.ones((N_runs, N_t, N_vars))
            true_dy = np.ones((N_runs, N_t, N_vars))
            pred_y = np.ones((N_runs, N_t, N_vars))
            pred_dy = np.ones((N_runs,N_t, N_vars))
            for N_run in range(0, N_runs):
                pred_y[N_run, :, :] = df_spl[df_spl['Run'] == N_run][spl_states]
                pred_dy[N_run, :, :] = df_spl[df_spl['Run'] == N_run][spl_derivs]
                
                true_y[N_run, :, :] = df_sol[df_sol['Run'] ==
                                          N_run][states]  # [:20]#[::-1]#[5:]#[:20]
                # true_dy[N_run, :, :] = df_sol[df_sol['Run'] ==
                #                               N_run][derivs]  # [:20]#[::-1]#[5:]#[:20]
                true_dy[N_run, :, :] = true_mod(t,pred_y[N_run]).T
                
            
            
            MSE_dy = np.mean((true_dy[:,idx,:] - pred_dy[:,idx,:])**2)
            MSE_y = np.mean((true_y[:,idx,:] - pred_dy[:,idx,:])**2)
            if MSE_dy < MSEn_dy:
                res = 0 # Neural ODE is 'worse'
            else:
                res = 1 # Neural is 'better'
            df_MSE.loc[len(df_MSE.index)] = [N_dat,N_runs,noise,MSE_dy,MSEn_dy,res]
# Print comparison
print(df_MSE)
# =============================================================================
# # Visualize Splines fits
# =============================================================================
N_run = 0
plt.figure()
plt.plot(t,pred_y[N_run,:,0],t,pred_y[N_run,:,1])
plt.plot(t[idx],true_y[N_run,idx,0],'x',t[idx],true_y[N_run,idx,1],'x')
plt.xlabel('t')
plt.ylabel('x1,x2')
plt.title('Fit')
plt.show()

plt.figure()
plt.plot(t,pred_dy[N_run,:,0],t,pred_dy[N_run,:,1])
plt.plot(t[idx],true_dy[N_run,idx,0],'x',t[idx],true_dy[N_run,idx,1],'x')
plt.xlabel('t')
plt.ylabel('dx1/dt,dx2/dt')
plt.title('Fit')
plt.show()

