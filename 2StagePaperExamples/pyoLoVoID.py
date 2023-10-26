# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:22:48 2020

@author: Afbwi
"""

# Objective find params of LoVo mod from NODE fit

import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyomo.environ import Var, Set, ConcreteModel, Param, Constraint,ConstraintList, Objective, \
        SolverFactory,TransformationFactory, exp, tanh
from scipy.integrate import odeint

Reg_size = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


np.random.seed(2016)
noise = 0.00
N_dat = 20
#N_Bsteps = 5
omitIC = True

# Import data
df_data = pd.read_csv(r'data/LoVo.csv')
#df_datadt = pd.read_csv(r'data/Pen_dt0.csv')
df_pred_dy = pd.read_csv(r'data/predLoVo_dt{}Noise.csv'.format(int(noise*100)))
df_pred_y = pd.read_csv(r'data/predLoVo_y{}Noise.csv'.format(int(noise*100)))
df_batch_y = pd.read_csv(r'data/batchLoVo_{}Noise.csv'.format(int(noise*100)))

N_runs = df_data['Run'].max()+1
t = df_data[df_data['Run'] == 0]['t'].to_numpy()
states = df_batch_y.drop(columns=['t','Run']).columns.values
derivs = df_pred_dy.drop(columns=['t','Run']).columns.values

#N_t = len(t)                    # Number to time points
N_runs = df_data['Run'].max()+1  # Number of run conditions
N_vars = len(states)            # Number of state vars
idx = np.round(np.linspace(0, len(t) - 1, N_dat)).astype(int) # 10 data points

N_t = len(t[idx])


true_y = np.ones((N_runs,len(t),N_vars))
pred_y = np.ones((N_runs,len(idx),N_vars))
pred_dy = np.ones((N_runs,len(idx),N_vars))
batch_y = np.ones((N_runs,len(idx),N_vars))
for N_run in range(0,N_runs):
    pred_y[N_run,:,:] = df_pred_y[df_pred_y['Run'] == N_run][states].iloc[idx]
    pred_dy[N_run,:,:] = df_pred_dy[df_pred_dy['Run'] == N_run][derivs].iloc[idx]
    true_y[N_run,:,:] = df_data[df_data['Run'] == N_run][states]
    batch_y[N_run,:,:] = df_batch_y[df_batch_y['Run'] == N_run][states]

#Create scaling factors
max_y = np.amax(np.amax(pred_y,axis=1),axis=0)
min_y = np.amin(np.amin(pred_y,axis=1),axis=0) #for adding noise

# Define inputs and outputs
y_dict = {}
dy_dict = {}
for itr in range(0,len(t[idx])):
    ii = t[idx][itr]
    for j in range(0,N_runs):
        for k in range(0,N_vars):
            y_dict[j,ii,k] = pred_y[:,:,:][j,itr,k]
            dy_dict[j,ii,k] = pred_dy[j,itr,k]/(max_y[k]-min_y[k])




# Create model
m = ConcreteModel()

# Defining Indexed Sets
m.t = Set(initialize=t[idx])
#m.run = Set(initialize=[0,1,2,3,4,5,6,7,8]) #Create indices for 9 runs
m.run = Set(initialize=range(0,N_runs)) #Create indices for 9 runs
m.var = Set(initialize=range(0,N_vars)) #state vars


# State Vars and Params
m.y = Param(m.run,m.t,m.var,initialize=y_dict)
m.dydt = Var(m.run,m.t,m.var)

m.p1 = Var(initialize=2); m.p2 = Var(initialize=2); m.p3 = Var(initialize=2)


# Define Constraints
def _dxdt(m,i,j):
    if i == 0 and omitIC ==1: #Do not include first and last points
        return Constraint.Skip
    if i > 4.75:
        return Constraint.Skip
    return m.dydt[j,i,0] ==  m.p1*m.y[j,i,0] - m.p2*m.y[j,i,0]*m.y[j,i,1]
m.dxdtcon = Constraint(m.t, m.run, rule=_dxdt)

def _dydt(m,i,j):
    if i == 0 and omitIC == 1: #Do not include first and last points
        return Constraint.Skip
    if i > 4.75:
        return Constraint.Skip
    return m.dydt[j,i,1] == -m.p3*m.y[j,i,1] + m.y[j,i,0]*m.y[j,i,1]
m.dydtcon = Constraint(m.t, m.run, rule=_dydt)

def _obj(m):
    return sum( (m.dydt[j,i,k]/(max_y[k] - min_y[k]) - dy_dict[j,i,k])**2 for j in m.run for i in m.t for k in m.var) #+ \
m.obj = Objective(rule=_obj)    

# Initialize
#m.p1, m.p2, m.p3 = params
for i in m.t:
    for j in m.run:
        m.dydt[j,i,0] = m.p1.value*m.y[j,i,0] - m.p2.value*m.y[j,i,0]*m.y[j,i,1]
        m.dydt[j,i,1] = -m.p2.value*m.y[j,i,1] + m.y[j,i,0]*m.y[j,i,1]


# Set up solver. Solve.
# Solve via ipopt
solver=SolverFactory('ipopt')
solver.options['max_iter'] = 4000
#solver.options['halt_on_ampl_error'] = "yes"
results = solver.solve(m,tee=True,keepfiles=True)
#results = solver.solve(m,tee=True,keepfiles=True, warmstart=True, tmpdir=r'./')


params = np.array([[1.5,2.0,3.4]]) #True params
params = np.array([[m.p1.value, m.p2.value, m.p3.value]])

np.savetxt('params/LoVopar{}.csv'.format(int(noise*100)), params, delimiter=',')
# sys.exit()
# =============================================================================
# # Post-processing NLP
# =============================================================================
def LoVo(z,t,p):
    p1, p2, p3 = p
    x = z[0]
    y = z[1]
    dxdt = p1*x - p2*x*y
    dydt = -p3*y + x*y
    return [dxdt,dydt]

def plot_sim(t,sim_y,batch_y,true_y,idx,title,noise,ls):
#    plt.figure()
    plt.plot(t,sim_y[:,0],'b',label='{}%'.format(int(noise*100)),
             linestyle=ls)
    plt.plot(t,sim_y[:,1],'r',label='{}%'.format(int(noise*100)),
             linestyle=ls)

#    plt.plot(t,MMS.transform(true_y[N_run,:,:])[:,1],'ko',label='True Ethylbenzene')
#    plt.plot(t,true_y[N_run,:,1],'kx',label='True Ethylbenzene')
    #plt.show()
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel('x,y')
    plt.plot(t[idx],batch_y[:,0:1],'bo',
             t[idx],batch_y[:,1:2],'ro',)
    plt.legend(loc='best')
#    plt.savefig('Visuals/{}N {}'.format(int(noise*100),title),dpi=300)
    plt.show()


# Checking Extrapolation
init = [1,1] #[1,1] or [1,3] 
t = np.linspace(0,10,101)
MSE = []; MAE = []
plt.figure()
for ls in ['solid']:
# for noise, ls in zip([0,0.01,0.05,.1],['solid','dotted','dashed','dashdot']):
    params = np.loadtxt('params/LoVopar{}.csv'.format(int(noise*100)), delimiter=',')
    sim_y = np.ones((N_runs,len(t),N_vars))
    y_true = np.ones((N_runs,len(t[idx]),N_vars))
    for j in range(0,N_runs):
        out = odeint(LoVo,init, t, args=(params,))
        sim_y[j,:,:] = out
        out = odeint(LoVo,init, t[idx], args=([1.5,2.0,3.4],))
        y_true[j,:,:] = out

    N_run = 0
    title = 'Fitted LoVo Extrapolation'
    plot_sim(t,sim_y[N_run], y_true[N_run,:,:], y_true[N_run],
             idx,title,noise,ls)
    MSE.append(np.sum((sim_y[:,idx,:] - y_true[:,:,:])**2)/ \
                     (N_runs*len(idx)*N_vars))
    MAE.append(np.sum(abs(sim_y[:,idx,:] - y_true[:,:,:]))/ \
                     (N_runs*len(idx)*N_vars))

if init[1] == 1:
    plt.savefig('Visuals/{}'.format(title),dpi=300)
if init[1] == 3:
    plt.savefig('Visuals/{}2'.format(title),dpi=300)












