# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 07:51:16 2020

@author: Afbwi
"""
#Description: This code solves the algebraic form of the Ethylbenzen model using
#           derivatives via the indirect approach
import idaes # imported before pyomo so the correct ipopt version is imported
import sys
import matplotlib.pyplot as plt
import pandas as pd; import numpy as np
from pyomo.environ import Var, Set, ConcreteModel, Param, Constraint,ConstraintList, Objective, \
        SolverFactory,TransformationFactory, exp, log, PositiveReals
from EthylbenPFRSim import odefun2, odefun1, plot_temp, plot_sol
from scipy.integrate import odeint
from Visualize import plot_dy, plot_pred
#from pyomo.dae import DerivativeVar, ContinuousSet
# With true derivs, obj can reach O(2e-13)

noise = 0.1
N_dat = 10
omitIC=True

# Import data
df_sol = pd.read_csv('data/NonIso_PFR.csv')
#df_sol2 = pd.read_csv(r'data/Iso_PFR.csv')
df_pred_dy = pd.read_csv(r'data/predSty_dt{}Noise.csv'.format(int(noise*100)))
df_pred_y = pd.read_csv(r'data/predSty_y{}Noise.csv'.format(int(noise*100)))
df_batch_y = pd.read_csv(r'data/batchSty_{}Noise.csv'.format(int(noise*100)))

df_pred_y['Temp'] = df_pred_y['Temp'] + 800


t = df_sol[df_sol['Run'] == 0]['t'].to_numpy()
states = df_batch_y.drop(columns=['t','Run']).columns.values
derivs = df_pred_dy.drop(columns=['t','Run']).columns.values

N_t = len(t)                    # Number to time points
N_runs = df_sol['Run'].max()+1  # Number of run conditions
N_vars = len(states)            # Number of state vars
idx = np.round(np.linspace(0, len(t) - 1, N_dat)).astype(int) # 10 data points

sol = np.ones((N_runs,N_t,N_vars));
pred_y = np.ones((N_runs,N_t,N_vars))
pred_dy = np.ones((N_runs,len(t),N_vars))
batch_y = np.ones((N_runs,len(idx),N_vars))
for N_run in range(0,N_runs):
    sol[N_run,:,:] = df_sol[df_sol['Run'] == N_run][states]
    pred_dy[N_run,:,:] = df_pred_dy[df_pred_dy['Run'] == N_run][derivs]
    pred_y[N_run,:,:] = df_pred_y[df_pred_y['Run'] == N_run][states]
    batch_y[N_run,:,:] = df_batch_y[df_batch_y['Run'] == N_run][states]
#pred_dy = df_pred_dy[derivs].to_numpy()
#batch_y = df_batch_y[states].to_numpy()

max_y = np.amax(np.amax(batch_y,axis=1),axis=0) #for adding noise
min_y = np.amin(np.amin(batch_y,axis=1),axis=0) #for adding noise
dum = pred_dy/(max_y-min_y)

dy = [odefun2(sol[:,0,:],t[0])]
for i in range(1,len(t)):
#    dy.append(odefun2(sol[:,idx,:][:,i,:],t[idx][i])) #NA
    dy.append(odefun2(sol[:,i,:],t[i])) #NA
true_dy = np.stack([np.stack(p, axis=1) for p in dy],axis=1)
dat_dy = true_dy[:,idx,:]
dat_dy = pred_dy[:,idx,:]

# Define inputs and outputs
y_dict = {}
dy_dict = {}
for i in range(0,len(t[idx])):
    ii = t[idx][i]
    for j in range(0,N_runs):
        for k in range(0,N_vars):
            y_dict[j,ii,k] = pred_y[:,idx,:][j,i,k]
            dy_dict[j,ii,k] = dat_dy[j,i,k]/(max_y[k]-min_y[k])

m = ConcreteModel()

m.t = Set(initialize=t[idx])#,bounds=(0,max(t)))
m.run = Set(initialize=range(0,N_runs)) #Create indices for 9 runs
m.var = Set(initialize=range(0,N_vars)) #state vars

# Define Paramters
m.b1 = Param(initialize=-17.34);     m.b2 = Param(initialize=-1.302e4); m.b3 = Param(initialize=5.051)
m.b4 = Param(initialize=-2.314e-10); m.b5 = Param(initialize=1.302e-6); m.b6=Param(initialize=-4.931e-3)
m.p = Param(initialize=2137);        m.o = Param(initialize=0.4)
m.Cpa = Param(initialize=299);       m.Cpb = Param(initialize=273); m.Cpc = Param(initialize=30);
m.Cpd = Param(initialize=201);       m.Cpe = Param(initialize=90);  m.Cpf = Param(initialize=249);
m.Cpg = Param(initialize=68);        m.Cpsteam = Param(initialize=40)
m.P0 = Param(initialize=2.4);        m.Fsteam = Param(initialize=48); m.Ft = Param(initialize=52);
#    r1st=p*(1-o)*torch.exp(-0.08539-10925/y[:,0])*(Peb-Pst*Ph2/Kp1);
#    r2=p*(1-o)*torch.exp(13.2392-25000/y[:,0])*(Peb);
#    r3=p*(1-o)*torch.exp(0.2961-11000/y[:,0])*(Peb*Ph2);

# Define Variables
# m.a = Var(initialize=-0.08539);  m.b = Var(initialize=13.2392);  m.c = Var(initialize=0.2961)
m.a = Var(initialize=2);    m.b = Var(initialize=2);       m.c = Var(initialize=2)
#m.d = Var(initialize=10925,within=PositiveReals); m.e = Var(initialize=25000,within=PositiveReals);   m.f = Var(initialize=11000,within=PositiveReals)
#m.a = Var(initialize=-1);    m.b = Var(initialize=1);       m.c = Var(initialize=1)
#m.d = Var(initialize=10.925,bounds=(0,100)); m.e = Var(initialize=25.000,bounds=(0,100));   
#m.f = Var(initialize=11.000,bounds=(0,100))
#m.a = Param(initialize=-0.08539);  m.b = Param(initialize=13.2392);  m.c = Param(initialize=0.2961)
m.d = Param(initialize=10925); m.e = Param(initialize=25000);   m.f = Param(initialize=11000)

m.Peb = Var(m.run,m.t); m.Pst = Var(m.run,m.t); m.Ph2 = Var(m.run,m.t)
m.Kp1 = Var(m.run,m.t)
m.r1st = Var(m.run,m.t);  m.r2 = Var(m.run,m.t);    m.r3 = Var(m.run,m.t)
m.dhrx1 = Var(m.run,m.t); m.dhrx2 = Var(m.run,m.t); m.dhrx3 = Var(m.run,m.t)
#m.dy1dx = Var(m.t,m.run); m.dy2dx = Var(m.t,m.run); m.dy3dx = Var(m.t,m.run)
#m.dy4dx = Var(m.t,m.run); m.dy5dx = Var(m.t,m.run); m.dy6dx = Var(m.t,m.run)
m.y = Param(m.run,m.t,m.var,initialize=y_dict)
m.dydx = Var(m.run,m.t,m.var)

m.r1pos = Var(m.run,m.t,)#initialize=1e-6,bounds=(1e-10,100))
m.r2pos = Var(m.run,m.t,)#initialize=1e-6,bounds=(1e-10,100))
m.r3pos = Var(m.run,m.t,)#initialize=1e-6,bounds=(1e-10,100))

m.r1neg = Var(m.run,m.t)#,initialize=1e-6)#,bounds=(1e-10,0))
m.r2neg = Var(m.run,m.t)#,initialize=1e-6)#,bounds=(1e-10,0))
m.r3neg = Var(m.run,m.t)#,initialize=1e-6)#,bounds=(1e-10,0))

# Define Constraints
def _Peb(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.Peb[j,ii] == m.y[j,ii,1]/(m.Ft*1)*(m.P0*1)
m.Pebcon = Constraint(m.t,m.run,rule=_Peb)

def _Pst(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.Pst[j,ii] == m.y[j,ii,2]/(m.Ft*1)*(m.P0*1)
m.Pstcon = Constraint(m.t,m.run,rule=_Pst)

def _Ph2(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.Ph2[j,ii] == m.y[j,ii,3]/(m.Ft*1)*(1*m.P0)
m.Ph2con = Constraint(m.t,m.run,rule=_Ph2)

def _Kp1(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.Kp1[j,ii] == exp(m.b1+m.b2/m.y[j,ii,0]+m.b3*log(m.y[j,ii,0])+ \
                ((m.b4*m.y[j,ii,0]+m.b5*1)*m.y[j,ii,0]+m.b6*1)*m.y[j,ii,0])
m.Kp1con = Constraint(m.t,m.run,rule=_Kp1)

def _r1st(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
#    return m.r1st[j,ii] == m.p*(1-m.o)*exp(m.a - m.d/m.y[j,ii,0])* \
#            (m.Peb[j,ii]-m.Pst[j,ii]*m.Ph2[j,ii]/m.Kp1[j,ii])
#    return log(m.r1pos[j,ii]) == (m.r1neg[j,ii])
    return (m.r1pos[j,ii]) == exp(m.r1neg[j,ii])
#    return (m.r1pos[j,ii]) == exp(m.a - m.d/m.y[j,ii,0]*1000)
m.r1stcon = Constraint(m.t,m.run,rule=_r1st)

def _r2(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
#    return m.r2[j,ii] == m.p*(1-m.o)*exp(m.b - m.e/m.y[j,ii,0]) * m.Peb[j,ii]
#    return log(m.r2pos[j,ii]) == (m.r2neg[j,ii]) 
    return (m.r2pos[j,ii]) == exp(m.r2neg[j,ii]) 
#    return (m.r2pos[j,ii]) == exp(m.b - m.e/m.y[j,ii,0]*1000) 
m.r2con = Constraint(m.t,m.run,rule=_r2)

def _r3(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
#    return m.r3[j,ii] == m.p*(1-m.o)*exp(m.c - m.f/m.y[j,ii,0])*m.Peb[j,ii]*m.Ph2[j,ii]
#    return log(m.r3pos[j,ii])== (m.r3neg[j,ii])
    return (m.r3pos[j,ii])== exp(m.r3neg[j,ii])
#    return (m.r3pos[j,ii])== exp(m.c - m.f/m.y[j,ii,0]*1000)
m.r3con = Constraint(m.t,m.run,rule=_r3)

def _dhrx1(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.dhrx1[j,ii] == 118000+(299-273-30)*(m.y[j,ii,0]-273)
m.dhrx1con = Constraint(m.t,m.run,rule=_dhrx1)

def _dhrx2(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.dhrx2[j,ii] == 105200 + (299-201-90)*(m.y[j,ii,0]-273)
m.dhrx2con = Constraint(m.t,m.run,rule=_dhrx2)

def _dhrx3(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.dhrx3[j,ii] == -53900 + (299+30-249-68)*(m.y[j,ii,0]-273)
m.dhrx3con = Constraint(m.t,m.run,rule=_dhrx3)

def _dy1dx(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.dydx[j,ii,0] == 1000*(-m.r1st[j,ii]*m.dhrx1[j,ii]-m.r2[j,ii]*m.dhrx2[j,ii]-m.r3[j,ii]*m.dhrx3[j,ii])/ \
            (m.y[j,ii,1]*(m.Cpa*1) + m.y[j,ii,2]*(m.Cpb*1) + m.y[j,ii,3]*(m.Cpc*1) + m.y[j,ii,4]*(m.Cpd*1) + \
             m.y[j,ii,4]*(m.Cpe*1) + m.y[j,ii,5]*(m.Cpf*1) + m.y[j,ii,5]*(m.Cpg*1) + (m.Fsteam*1)*m.Cpsteam)
m.dy1dxcon = Constraint(m.t,m.run,rule=_dy1dx)

def _dy2dx(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.dydx[j,ii,1] == (-m.r1st[j,ii] - m.r2[j,ii] - m.r3[j,ii])*1000
m.dy2dxcon = Constraint(m.t,m.run,rule=_dy2dx)

def _dy3dx(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.dydx[j,ii,2] == m.r1st[j,ii]*1000
m.dy3dxcon = Constraint(m.t,m.run,rule=_dy3dx)

def _dy4dx(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.dydx[j,ii,3] == (m.r1st[j,ii] - m.r3[j,ii])*1000
m.dy4dxcon = Constraint(m.t,m.run,rule=_dy4dx)

def _dy5dx(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.dydx[j,ii,4] == m.r2[j,ii]*1000
m.dy5dxcon = Constraint(m.t,m.run,rule=_dy5dx)

def _dy6dx(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.dydx[j,ii,5] == m.r3[j,ii]*1000
m.dy6dxcon = Constraint(m.t,m.run,rule=_dy6dx)

# Positivity constraints
def _r1stpos(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.r1st[j,ii] == m.r1pos[j,ii]*(m.p*(1-m.o)*(m.Peb[j,ii]-m.Pst[j,ii]*m.Ph2[j,ii]/m.Kp1[j,ii]))
m.r1stposcon = Constraint(m.t,m.run,rule=_r1stpos)

def _r2pos(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.r2[j,ii] == m.r2pos[j,ii]*(m.p*(1-m.o)* m.Peb[j,ii])
m.r2poscon = Constraint(m.t,m.run,rule=_r2pos)

def _r3pos(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return m.r3[j,ii] == m.r3pos[j,ii]*(m.p*(1-m.o)*m.Peb[j,ii]*m.Ph2[j,ii])
m.r3poscon = Constraint(m.t,m.run,rule=_r3pos)

# Negativity constraints
def _r1stneg(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return (m.a - m.d/m.y[j,ii,0]) == m.r1neg[j,ii]
m.r1stnegcon = Constraint(m.t,m.run,rule=_r1stneg)

def _r2neg(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return (m.b - m.e/m.y[j,ii,0]) == m.r2neg[j,ii]
m.r2negcon = Constraint(m.t,m.run,rule=_r2neg)

def _r3neg(m,ii,j):
    if ii == 0 and omitIC==1:
        return Constraint.Skip
    return (m.c - m.f/m.y[j,ii,0]) == m.r3neg[j,ii]
m.r3negcon = Constraint(m.t,m.run,rule=_r3neg)

# Objective function
def _obj(m):
    return sum( (m.dydx[j,ii,k]/(max_y[k] - min_y[k]) - dy_dict[j,ii,k])**2 for j in m.run for ii in m.t for k in m.var) #+ \
#    return sum( (m.y[j,ii,k]/(max_y[k]))**2 for for j in m.run for ii in m.t for k in m.var)
m.obj = Objective(rule=_obj)    

# Must initialize variables to avoid log(0)
for i in range(0,len(t[idx])):
    ii = t[idx][i]
    for j in range(0,N_runs):
        m.Peb[j,ii] = m.y[j,ii,1]/(m.Ft.value*1)*(m.P0.value*1)
        m.Pst[j,ii] = m.y[j,ii,2]/(m.Ft.value*1)*(m.P0.value*1)
        m.Ph2[j,ii] = m.y[j,ii,3]/(m.Ft.value*1)*(1*m.P0.value)
        m.Kp1[j,ii] = exp(m.b1.value+m.b2.value/m.y[j,ii,0]+m.b3.value*log(m.y[j,ii,0])+ \
                ((m.b4.value*m.y[j,ii,0]+m.b5.value*1)*m.y[j,ii,0]+m.b6.value*1)*m.y[j,ii,0])
        m.r1st[j,ii] = m.p.value*(1-m.o.value)*exp(m.a.value - m.d.value/m.y[j,ii,0])* \
            (m.Peb[j,ii].value - m.Pst[j,ii].value*m.Ph2[j,ii].value/m.Kp1[j,ii].value)
#    return log(m.r1st[j,ii]) == (m.a - m.d/m.y[j,ii,0])* \
#    log(m.p*(1-m.o)*(m.Peb[j,ii]-m.Pst[j,ii]*m.Ph2[j,ii]/m.Kp1[j,ii]))
        m.r2[j,ii] = m.p.value*(1-m.o.value)*exp(m.b.value - m.e.value/m.y[j,ii,0]) * m.Peb[j,ii].value
#    return log(m.r2[j,ii]) == log(m.p*(1-m.o)* m.Peb[j,ii])*(m.b - m.e/m.y[j,ii,0]) 
        m.r3[j,ii] = m.p.value*(1-m.o.value)*exp(m.c.value - m.f.value/m.y[j,ii,0])*m.Peb[j,ii].value*m.Ph2[j,ii].value
#    return log(m.r3[j,ii]) == log(m.p*(1-m.o))*(m.c - m.f/m.y[j,ii,0])
        m.dhrx1[j,ii] = 118000+(299-273-30)*(m.y[j,ii,0]-273)
        m.dhrx2[j,ii] = 105200 + (299-201-90)*(m.y[j,ii,0]-273)
        m.dhrx3[j,ii] = -53900 + (299+30-249-68)*(m.y[j,ii,0]-273)
        m.dydx[j,ii,0] = 1000*(-m.r1st[j,ii].value*m.dhrx1[j,ii].value-m.r2[j,ii].value*m.dhrx2[j,ii].value-m.r3[j,ii].value*m.dhrx3[j,ii].value)/ \
            (m.y[j,ii,1]*(m.Cpa.value*1) + m.y[j,ii,2]*(m.Cpb.value*1) + m.y[j,ii,3]*(m.Cpc.value*1) + m.y[j,ii,4]*(m.Cpd.value*1) + \
             m.y[j,ii,4]*(m.Cpe.value*1) + m.y[j,ii,5]*(m.Cpf.value*1) + m.y[j,ii,5]*(m.Cpg.value*1) + (m.Fsteam.value*1)*m.Cpsteam.value)
        m.dydx[j,ii,1] = (-m.r1st[j,ii].value - m.r2[j,ii].value - m.r3[j,ii].value)
        m.dydx[j,ii,2] = m.r1st[j,ii].value*1000
        m.dydx[j,ii,3] = (m.r1st[j,ii].value - m.r3[j,ii].value)*1000
        m.dydx[j,ii,4] = m.r2[j,ii].value*1000
        m.dydx[j,ii,5] = m.r3[j,ii].value*1000

for ii in m.t:
    for j in range(0,N_vars):
        dum = m.r1st[j,ii].value/(m.p.value*(1-m.o.value)*(m.Peb[j,ii].value-m.Pst[j,ii].value*m.Ph2[j,ii].value/m.Kp1[j,ii].value))
#        dum = m.r2[j,ii].value/(m.p.value*(1-m.o.value)* m.Peb[j,ii].value)
        dum = m.r3[j,ii].value/(m.p.value*(1-m.o.value)*m.Peb[j,ii].value*m.Ph2[j,ii].value)
#        for k in range(0,N_vars):
#            dum = m.y[j,ii,k].value
#        print(dum)

# Set up solver. Solve.
# Solve via ipopt
solver=SolverFactory('ipopt')
solver.options['max_iter'] = 800
solver.options['halt_on_ampl_error'] = "yes"
solver.options['warm_start_init_point'] = 'yes'
solver.options['bound_relax_factor'] = 0
results = solver.solve(m,tee=True,keepfiles=True)
#results = solver.solve(m,tee=True,keepfiles=True, warmstart=True, tmpdir=r'./')


params = [[m.a.value, m.b.value, m.c.value, m.d.value, m.e.value, m.f.value],
                   [-0.08539,13.2392,0.2961,10925,25000,11000]]
print('Fitted vs. true params\n', params[0], '\n', params[1])

dy = [odefun2(sol[:,0,:],t[0], params[0])]
for i in range(1,len(t)):
    dy.append(odefun2(sol[:,i,:],t[i], params[0])) #NA
sol_pred_dy = np.stack([np.stack(p, axis=1) for p in dy],axis=1)

N_run = 4
title = 'NN+ODE true vs estim'      #Noisy data vs NODE estimates
#plot_dy(t,pred_dy[N_run],true_dy[N_run],idx,title) #obj below
title = 'NN+ODE true vs pyo estim'  #Noisy data vs pyomo estimates
#plot_dy(t,sol_pred_dy[N_run],true_dy[N_run],idx,title)
title = 'NN+ODE estim vs pyo estim' #NODE estimates vs pyomo estimates
#plot_dy(t,sol_pred_dy[N_run],pred_dy[N_run],idx,title) #obj above

dum = pred_dy - sol_pred_dy
dum = sum(sum(sum(((pred_dy[:,idx,:] - true_dy[:,idx,:])/(max_y - min_y))**2)))
dum = sum(sum(sum(((pred_dy[:,idx,:] - sol_pred_dy[:,idx,:])/(max_y - min_y))**2)))

np.savetxt('params/Stypar{}.csv'.format(int(noise*100)), params, delimiter=',')
# sys.exit()
# =============================================================================
# # Calculate SSE
# =============================================================================

MSE = []; MAE = []
for ls in ['solid']:
# for noise, ls in zip([0,0.01,0.05,.1],['solid','dotted','dashed','dashdot']):
    params = np.loadtxt('params/Stypar{}.csv'.format(int(noise*100)), delimiter=',')
    sol_pred = np.ones((N_runs,len(t),N_vars))
    for j in range(0,N_runs):
        out = odeint(odefun1, sol[j,0,:], t,args=(params[0],))
        sol_pred[j,:,:] = out

    # Plotting
    title = 'Fitted ODE Simulation'
#    plot_temp(t,sol_pred[1,:,:],title); plot_sol(t,sol_pred[1,:,:],title)
    MSE.append(np.sum((sol_pred[:,idx,:] - sol[:,idx,:])**2)/ \
                     (N_runs*len(idx)*N_vars))
    MAE.append(np.sum(abs(sol_pred[:,idx,:] - sol[:,idx,:]))/ \
                     (N_runs*len(idx)*N_vars))


title = 'Fitted ODE Simulation'
plot_pred(t,sol_pred[N_run],batch_y[N_run],sol[N_run],idx,title)
