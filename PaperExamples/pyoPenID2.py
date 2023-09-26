# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:54:53 2019

@author: wbradley30
"""
#Description: This code solves the algebraic form of the Penicillin model using
#           derivs from the indirect approach. Objective funct = SSE(dxdt_pred-dxdt_true) 



import idaes # imported before pyomo so the correct ipopt version is imported
import sys
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyomo.environ import Var, Set, ConcreteModel, Param, Constraint,ConstraintList, Objective, \
        SolverFactory,TransformationFactory, exp, tanh
#from pyomo.dae import DerivativeVar, ContinuousSet
from scipy.integrate import odeint
from Visualize import plot_Pendy

np.random.seed(2016)
noise = 0.01
N_dat = 10
#N_Bsteps = 5
omitIC = True


def restruc2d1d(x):    
    x_new = x[:,0]
    N_runs = len(x[0,:]) #Assume N_runs are in the second dimension
    for i in range(1,N_runs):
        x_new = np.append(x_new, x[:,i],axis=0)
    x_new = x_new.reshape(len(x_new),1)        
    return x_new

# Import data
df_data = pd.read_csv(r'data/PenBatch_0Noise.csv') # True data
#df_datadt = pd.read_csv(r'data/Pen_dt0.csv')
df_pred_dy = pd.read_csv(r'data/predPen_dt{}Noise.csv'.format(int(noise*100)))   #pred from NODE
df_pred_y = pd.read_csv(r'data/predPen_y{}Noise.csv'.format(int(noise*100)))     #pred from NODE
df_batch_y = pd.read_csv(r'data/batchPen_{}Noise.csv'.format(int(noise*100)))    #data + noise

t = df_data[df_data['Run'] == 0]['t'].to_numpy()
states = df_batch_y.drop(columns=['t','Run']).columns.values
derivs = df_pred_dy.drop(columns=['t','Run']).columns.values

#N_t = len(t)                    # Number to time points
N_runs = df_data['Run'].max()+1  # Number of run conditions
N_vars = len(states)             # Number of state vars
idx = np.round(np.linspace(0, len(t) - 1, N_dat)).astype(int) # 10 data points

N_t = len(t[idx])

true_y = np.ones((N_runs,len(t),N_vars))
pred_y = np.ones((N_runs,len(idx),N_vars))
pred_dy = np.ones((N_runs,len(idx),N_vars))
for N_run in range(0,N_runs):
    pred_y[N_run,:,:] = df_pred_y[df_pred_y['Run'] == N_run][states].iloc[idx]
    pred_dy[N_run,:,:] = df_pred_dy[df_pred_dy['Run'] == N_run][derivs].iloc[idx]
    true_y[N_run,:,:] = df_data[df_data['Run'] == N_run][states]

#Create scaling factors
maxB = df_pred_y['tB'].max(axis=0)
maxS = df_pred_y['tS'].max(axis=0)
maxP = df_pred_y['tP'].max(axis=0)
#maxD = df_batch_y['tD'].max(axis=0)
maxV = df_pred_y['tV'].max(axis=0)

#Create scaled initial guesses
B_scaled = {};      dBdt_scaled = {};
S_scaled = {};      dSdt_scaled = {};
P_scaled = {};      dPdt_scaled = {};
V_scaled = {};      dVdt_scaled = {};
#batch_y = np.ones((N_runs,len(idx),N_vars))
F = {}
Sf = {}
D = {}
for N_run in range(0,N_runs):
    for i,ii in zip(idx,range(0,N_dat)):#range(0,len(t)):
#        B_scaled[i*3,N_run] = df_data[df_data['Run'] == N_run]['tB'].iloc[i]*(1/maxB)
#        S_scaled[i*3,N_run] = df_data[df_data['Run'] == N_run]['tS'].iloc[i]*(1/maxS)
#        P_scaled[i*3,N_run] = df_data[df_data['Run'] == N_run]['tP'].iloc[i]*(1/maxP)
#        V_scaled[i*3,N_run] = df_data[df_data['Run'] == N_run]['tV'].iloc[i]*(1/maxV)
        B_scaled[i*3,N_run] = df_pred_y[df_pred_y['Run'] == N_run]['tB'].iloc[i]*(1/maxB)
        S_scaled[i*3,N_run] = df_pred_y[df_pred_y['Run'] == N_run]['tS'].iloc[i]*(1/maxS)
        P_scaled[i*3,N_run] = df_pred_y[df_pred_y['Run'] == N_run]['tP'].iloc[i]*(1/maxP)
        V_scaled[i*3,N_run] = df_pred_y[df_pred_y['Run'] == N_run]['tV'].iloc[i]*(1/maxV)
        dBdt_scaled[i*3,N_run] = df_pred_dy[df_pred_dy['Run'] == N_run]['dBdt'].iloc[i]*(1/maxB)
        dSdt_scaled[i*3,N_run] = df_pred_dy[df_pred_dy['Run'] == N_run]['dSdt'].iloc[i]*(1/maxS)
        dPdt_scaled[i*3,N_run] = df_pred_dy[df_pred_dy['Run'] == N_run]['dPdt'].iloc[i]*(1/maxP)
        dVdt_scaled[i*3,N_run] = df_data[df_data['Run']==N_run]['F'].iloc[i]*(1/maxV)
#        dBdt_scaled[i*3,N_run] = df_data[df_data['Run'] == N_run]['tBdt'].iloc[i]*(1/maxB)
#        dSdt_scaled[i*3,N_run] = df_data[df_data['Run'] == N_run]['tSdt'].iloc[i]*(1/maxS)
#        dPdt_scaled[i*3,N_run] = df_data[df_data['Run'] == N_run]['tPdt'].iloc[i]*(1/maxP)
#        dVdt_scaled[i*3,N_run] = df_data[df_data['Run'] == N_run]['F'].iloc[i]*(1/maxV)
        F[i*3,N_run] = df_data[df_data['Run']==N_run]['F'].iloc[i]
        Sf[i*3,N_run] = df_data[df_data['Run']==N_run]['Sf'].iloc[i]
        D[i*3,N_run] = F[i*3,N_run]/maxV/V_scaled[i*3,N_run] 
        

# Create model
m = ConcreteModel()
# Defining Indexed Sets
m.t = Set(initialize=t[idx])

#m.run = Set(initialize=[0,1,2,3,4,5,6,7,8]) #Create indices for 9 runs
m.run = Set(initialize=range(0,N_runs)) #Create indices for 9 runs


# State Vars
#m.B = Var(m.t, m.run, bounds=(0,5), initialize=B_scaled) #Total cells/volume
#m.S = Var(m.t, m.run, bounds=(0,5), initialize=S_scaled)  #Glucose/volume
#m.P = Var(m.t, m.run, bounds=(0,5), initialize=P_scaled)  #Product/volume
#m.V = Var(m.t, m.run, bounds=(0,5), initialize=V_scaled)  #Volume
#m.D = Var(m.t, m.run)                       #Dilution: D = F/V

m.B = Param(m.t, m.run, mutable=True, initialize=B_scaled) #Total cells/volume
m.S = Param(m.t, m.run, mutable=True, initialize=S_scaled)  #Glucose/volume
m.P = Param(m.t, m.run, mutable=True, initialize=P_scaled)  #Product/volume
m.V = Param(m.t, m.run, mutable=True, initialize=V_scaled)#)  #Volume
m.D = Param(m.t, m.run, mutable=True, initialize=D)#)    

#m.rates = Var(m.t,m.run, m.outputs)  #only if indexing rates
m.mu = Var(m.t, m.run, bounds = (-9,9))
m.sigL = Var(m.t, m.run, bounds = (-9,9))
m.qp = Var(m.t, m.run,  bounds = (-9,9))
m.cL = Var(m.t, m.run, bounds = (-94,94))
m.m_x = Var(m.t, m.run, bounds = (-95,95))

#m.k = Var(bounds=(0,10),initialize=0.01)
#m.k.fixed = True

m.dBdt = Var(m.t, m.run, bounds = (-3,3))
m.dSdt = Var(m.t, m.run, bounds = (-3,3))
m.dPdt = Var(m.t, m.run, bounds = (-3,3))
m.dVdt = Var(m.t, m.run, bounds = (-3,3))
#
#m.dBdt = Param(m.t, m.run, mutable=True, initialize=dBdt_scaled)
#m.dSdt = Param(m.t, m.run, mutable=True, initialize=dSdt_scaled)
#m.dPdt = Param(m.t, m.run, mutable=True, initialize=dPdt_scaled)
#m.dVdt = Param(m.t, m.run, mutable=True, initialize=dVdt_scaled)

# Exogenous inputs
m.F = Param(m.t,m.run, mutable=True)
m.Sf = Param(m.t,m.run, mutable=True)

# Parameters
#else:
#    cLmax = 0.0519; kL = 0.05;     ki = 1.0; m_xm = 0.010; k = 0.0137
#    kp = 0.0001;    mu_m = 0.0099; kx = 0.3; qpm = 0.0837; Yps = 1.2; Yxs = 0.47
#         cLmax,  kL,   ki,  m_xm,  k,      kp,     mu_m,   kx,  qpm,    Yps, Yxs
#params = np.array([0.0084,0.05,1.0,0.029,0.010,0.0001,0.11,0.3,0.004,1.2,0.47]) # incorrect
params = [0.0519, 0.05, 1.0, 0.010, 0.0137, 0.0001, 0.0099, 0.3, 0.0837, 1.2, 0.47]
params = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])*2
#2,6
m.cLmax = Var(bounds = (0.001,10.0))
m.kL = Var(bounds = (0.01,10.0))
m.ki = Var(bounds = (0.1,20))
m.m_xm = Var(bounds = (0.001,10.0))
m.k = Var(bounds=(0,10),initialize=0.01)
m.kp = Var(bounds = (0.00001,10.0))
m.mu_m = Var(bounds = (0.001,20))
m.kx = Var(bounds = (0.1,10))
m.qpm = Var(bounds = (0.001,10.0))
m.Yps = Var(bounds = (0.1,20))
m.Yxs = Var(bounds = (0.1,20))

#m.cLmax = Param(mutable=True)
#m.kL = Param(mutable=True)
#m.ki = Param(mutable=True)
#m.m_xm = Param(mutable=True)
#m.k = Param(mutable=True)
#m.kp = Param(mutable=True)
#m.mu_m = Param(mutable=True)
#m.kx = Param(mutable=True)
#m.qpm = Param(mutable=True)
#m.Yps = Param(mutable=True)
#m.Yxs = Param(mutable=True)

#Establish initial conditions
#def _init_con(m):
#    for j in range(0,N_runs):
##        yield m.B[0,j] == B_scaled[0,j]
##        yield m.S[0,j] == 0
##        yield m.P[0,j] == 0
##        yield m.V[0,j] == 0.2/maxV
#m.init_con = ConstraintList(rule=_init_con)

# Missing Equation
def _D(m,i,j):
    return m.D[i,j]*m.V[i,j] == m.F[i,j]/maxV
#m.Dcon = Constraint(m.t, m.run, rule=_D)

# Equation C1
def _dBdt(m,i,j):
    if i == 0 and omitIC==1:
        return Constraint.Skip
    return m.dBdt[i,j] == m.B[i,j]*(m.mu[i,j] - m.D[i,j] - m.cL[i,j])                                          #+ m.s1[i,j]
m.dBdtcon = Constraint(m.t, m.run, rule=_dBdt)

# Equation C2
def _dSdt(m,i,j):
    if i == 0 and omitIC==1:
        return Constraint.Skip
    return m.dSdt[i,j]*maxS == -m.sigL[i,j]*m.B[i,j]*maxB + ((m.Sf[i,j] - m.S[i,j]*maxS)*m.D[i,j]) #+ m.s2[i,j]
m.dSdtcon = Constraint(m.t, m.run, rule=_dSdt)

# Equation C3
def _dPdt(m,i,j):
    if i == 0 and omitIC==1:
        return Constraint.Skip
    return m.dPdt[i,j]*maxP == m.qp[i,j]*m.B[i,j]*maxB - m.P[i,j]*maxP*(m.D[i,j] + m.k)               #m.c9*m.k)/maxB #+ m.s3[i,j]
m.dPdtcon = Constraint(m.t, m.run, rule=_dPdt)

# Equaction C4
def _dVdt(m,i,j):
    if i == 0 and omitIC==1:
        return Constraint.Skip
    return m.dVdt[i,j] == m.F[i,j]/maxV #+ m.s4[i,j]
#m.dVdtcon = Constraint(m.t, m.run, rule=_dVdt)

#Equation B1
def _mu(m,i,j):
    if i == 0 and omitIC==1:
        return Constraint.Skip
    return (m.mu[i,j])*(m.kx*m.B[i,j]*maxB +10) == m.mu_m*m.S[i,j]*maxS 
m.mucon = Constraint(m.t, m.run, rule=_mu)

#Equation B2
def _sigL(m,i,j):
    if i == 0 and omitIC==1:
        return Constraint.Skip
    return m.sigL[i,j]*m.Yxs*m.Yps== m.mu[i,j]*m.Yps + m.qp[i,j]*m.Yxs + m.m_x[i,j]*(m.Yxs*m.Yps)
m.sigLcon = Constraint(m.t, m.run, rule=_sigL)

#Equation B3
def _qp(m,i,j):
    if i == 0 and omitIC==1:
        return Constraint.Skip
    return m.qp[i,j]*((4*m.kp*3*m.ki + m.B[i,j]*maxB*m.S[i,j]*maxS*3*m.ki) \
            + m.B[i,j]*maxB*m.S[i,j]**2*maxS**2) == 1.5*m.qpm*m.S[i,j]*maxS*m.B[i,j]*maxB*3*m.ki
m.qpcon = Constraint(m.t, m.run, rule=_qp)

#Equation B3
def _cL(m,i,j):
    if i == 0 and omitIC==1:
        return Constraint.Skip
    return m.cL[i,j]*(m.kL + m.B[i,j]*maxB + 1) == m.cLmax*m.B[i,j]*maxB*exp(-m.S[i,j]*maxS/100)
m.cLcon = Constraint(m.t, m.run, rule=_cL)

def _m_x(m,i,j):
    if i == 0 and omitIC==1:
        return Constraint.Skip
    return m.m_x[i,j]*(m.B[i,j]*maxB + 10) == m.m_xm*m.B[i,j]*maxB
m.m_xcon = Constraint(m.t, m.run, rule=_m_x)


def _obj(m):
    return sum( (m.dBdt[i,j] - dBdt_scaled[i,j])**2 for i in m.t for j in m.run) + \
            sum( (m.dSdt[i,j] - dSdt_scaled[i,j])**2 for i in m.t for j in m.run) + \
            sum( (m.dPdt[i,j] - dPdt_scaled[i,j])**2 for i in m.t for j in m.run) 

#            sum( (m.D[i,j] - df_exp_scal.loc[i,j]['Dt'])**2 for i in m.t for j in m.run)
m.obj = Objective(rule=_obj)    


# Initialize
#         cLmax,  kL,   ki,  m_xm,  k,      kp,     mu_m,   kx,  qpm,    Yps, Yxs
#params = [0.0519, 0.05, 1.0, 0.010, 0.0137, 0.0001, 0.0099, 0.3, 0.0837, 1.2, 0.47]
m.cLmax,m.kL,m.ki,m.m_xm,m.k,m.kp,m.mu_m,m.kx, m.qpm, m.Yps, m.Yxs = params

for i in m.t: 
    for j in m.run:
        #Exogenous inputs
        m.Sf[i,j] = Sf[i,j]
        m.F[i,j] = F[i,j]
        # State var initialization is redundant if previously initialized
        m.B[i,j] = df_pred_y[df_pred_y['Run'] == j]['tB'].iloc[int(i/3)]*(1/maxB)
        m.S[i,j] = df_pred_y[df_pred_y['Run'] == j]['tS'].iloc[int(i/3)]*(1/maxS)
        m.P[i,j] = df_pred_y[df_pred_y['Run'] == j]['tP'].iloc[int(i/3)]*(1/maxP)
        m.V[i,j] = V_scaled[i,j]                #Scaled by Vmax instead of Dmax
        m.D[i,j] = m.F[i,j].value/maxV/m.V[i,j].value 
        #m.V[i,j] = F[i,j]/m.D[i,j].value
        

        m.mu[i,j] = m.mu_m.value*m.S[i,j].value*maxS / (m.kx.value*m.B[i,j].value*maxB + 10) 
        m.qp[i,j] = 1.5*m.qpm.value*m.S[i,j].value*maxS*m.B[i,j].value*maxB/ \
                    (4*m.kp.value + m.S[i,j].value*maxS*m.B[i,j].value*maxB* \
                     (1 + m.S[i,j].value*maxS/3*m.ki.value)) 
        m.sigL[i,j] = m.mu[i,j].value/m.Yxs.value + m.qp[i,j].value/m.Yps.value + m.m_xm.value
        m.cL[i,j] = m.cLmax.value*m.B[i,j].value*maxB*exp(-m.S[i,j].value*maxS/100) / \
                    (m.kL.value + m.B[i,j].value*maxB + 1)
        m.m_x[i,j] = m.m_xm.value*m.B[i,j].value*maxB / (m.B[i,j].value*maxB + 10)
        
#        m.dBdt[i,j] = m.B[i,j].value*(m.mu[i,j].value - m.D[i,j].value)
#        m.dSdt[i,j] = (-m.sigL[i,j].value*m.B[i,j].value + (m.Sf[i,j].value - m.S[i,j].value)*m.D[i,j].value)/maxS
#        m.dPdt[i,j] = (m.qp[i,j].value*m.B[i,j].value-m.P[i,j].value*(m.D[i,j].value - m.k.value))/maxP
        m.dVdt[i,j] = m.F[i,j].value/maxV
        
        
#        m.B[i,j].fixed = True
#        m.S[i,j].fixed = True
#        m.P[i,j].fixed = True
        #m.V[i,j].fixed = True
        #m.D[i,j].fixed = True


# Set up solver. Solve.
# Solve via ipopt
solver=SolverFactory('ipopt')
solver.options['max_iter'] = 4000
#solver.options['halt_on_ampl_error'] = "yes"
results = solver.solve(m,tee=True,keepfiles=True)
#results = solver.solve(m,tee=True,keepfiles=True, warmstart=True, tmpdir=r'./')

params = np.array([[m.cLmax.value, m.kL.value, m.ki.value, m.m_xm.value, m.k.value, m.kp.value, \
                   m.mu_m.value, m.kx.value, m.qpm.value, m.Yps.value, m.Yxs.value]])
#sys.exit()
#stophere = stopnow
# =============================================================================
# # Plot pyomo fit and scipy simulation
# =============================================================================
# Create variables for plotting.
arr_Bmod = restruc2d1d(np.asarray([m.B[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs)*maxB)
arr_Smod = restruc2d1d(np.asarray([m.S[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs)*maxS)
arr_Pmod = restruc2d1d(np.asarray([m.P[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs)*maxP)
arr_Vmod = restruc2d1d(np.asarray([m.V[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs)*maxV)
arr_Dmod = restruc2d1d(np.asarray([m.D[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs))
arr_mu = restruc2d1d(np.asarray([m.mu[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs))
arr_sigL= restruc2d1d(np.asarray([m.sigL[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs))
arr_qp = restruc2d1d(np.asarray([m.qp[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs))
arr_pred = np.concatenate((arr_Bmod,arr_Smod,arr_Pmod,arr_Dmod,arr_Vmod,arr_mu,arr_sigL,arr_qp),axis=1)

arr_t = np.tile(t[idx],N_runs)         #Repeats array N_run times
arr_runs = np.repeat(np.arange(N_runs),N_t)    #Repeats each N_run element N_t times, element by element
index = pd.MultiIndex.from_arrays([arr_t,arr_runs],names=['t','runs'])
df_mod = pd.DataFrame(data=arr_pred, index=index, columns = ['tB','tS','tP','tD','tV','mu','sigL','qp'] )
df_mod.sort_index(inplace=True)
idx2 = pd.IndexSlice

df_mod = pd.DataFrame(data=arr_pred, columns = ['tB','tS','tP','tD','tV','mu','sigL','qp'] )
df_mod['t'] = arr_t
df_mod['Run'] = arr_runs
df_mod['tBdt'] = restruc2d1d(np.asarray([m.dBdt[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs)*maxB)
df_mod['tSdt'] = restruc2d1d(np.asarray([m.dSdt[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs)*maxS)
df_mod['tPdt'] = restruc2d1d(np.asarray([m.dPdt[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs)*maxP)
df_mod['tVdt'] = restruc2d1d(np.asarray([m.dVdt[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs)*maxV)
#df_mod = pd.DataFrame(data=arr_Bmod,columns = ['tB'])
#df_mod['tS'] = restruc2d1d(np.asarray([m.S[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs)*maxS)
#df_mod['tS'] = restruc2d1d(np.asarray([m.P[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs)*maxP)
#df_mod['tS'] = restruc2d1d(np.asarray([m.D[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs))
#df_mod['tS'] = restruc2d1d(np.asarray([m.mu[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs))
#arr_sigL= restruc2d1d(np.asarray([m.sigL[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs))
#arr_qp = restruc2d1d(np.asarray([m.qp[i,j].value for i in m.t for j in m.run]).reshape(-1,N_runs))

params - [0.0519, 0.05, 1.0, 0.010, 0.0137, 0.0001, 0.0099, 0.3, 0.0837, 1.2, 0.47]
#max_vals = [maxB,maxS,maxP,maxD,maxmu,maxsigL,maxqp]


# Plot the results
def pyoplot(t,true_y,t2,pred_y,title):
    plt.figure()
    plt.subplot(221)
#    plt.title('Fit of NN-Hybrid vs. Experimental data')
    plt.plot(t,true_y[:,0],'k+',t2,pred_y[:,0])
    plt.ylabel('Biomass (gm/L)')
    
    plt.subplot(223)
    plt.plot(t,true_y[:,2],'k+',t2,pred_y[:,2])
    plt.xlabel('t (hr)')
    plt.ylabel('Penicillin (gm/L)')
    
    plt.subplot(222)
    plt.plot(t,true_y[:,1],'k+',t2,pred_y[:,1])
    plt.xlabel('t (hr)')
    plt.ylabel('Substrate (gm/L)')
    
    plt.subplot(224)
    plt.plot(t,true_y[:,3],'k+',t2,pred_y[:,3])
    plt.xlabel('t (hr)')
    plt.ylabel('Volume (L)')
    plt.legend(['Exp Data','Prediction'])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('data/PenID{}N'.format(int(noise*100)),dpi=300)
    plt.show()
    
def plotPen(t,yt_runs):
    plt.figure()
    plt.subplot(221); plt.ylabel('biomass (g/L)')
    plt.subplot(222); plt.ylabel('substrate (g/L)')
    plt.subplot(223); plt.ylabel('product (g/L)');   plt.xlabel('time (hr)')
    plt.subplot(224); plt.ylabel('volume (L)');      plt.xlabel('time (hr)')
    for N_run in range(0,N_runs):
        plt.subplot(221)
        #plt.title('Comparison of true and mechanistic model predictions')
        plt.plot(t,yt_runs[N_run,:,0],'k',)
        #plt.legend(['B','S','P','V'])
        plt.subplot(222)
        plt.plot(t,yt_runs[N_run,:,1],'k',)        
    #    plt.show()
        plt.subplot(223)
        plt.plot(t,yt_runs[N_run,:,2],'k',)
    #    plt.show()
        if len(yt_runs[0,:,0]) > 3:
            plt.subplot(224)
            plt.plot(t,yt_runs[N_run,:,3],'k',)
    plt.suptitle('All batches')
    plt.tight_layout()
    plt.savefig('data/Pen9Sims',dpi=300)
    plt.show()
    
N_run = 0
mod_y = df_mod[df_mod['Run'] == N_run][['tB','tS','tP','tV']].to_numpy()
#data = df_data[df_data['Run'] == N_run][['tB','tS','tP','tV']].to_numpy()
data = df_pred_y[df_pred_y['Run'] == N_run][['tB','tS','tP','tV']].to_numpy()
#dum = df_mod[df_mod['Run'] == N_run][['tBdt','tSdt','tPdt','tVdt']].to_numpy()
#data = df_data[df_data['Run'] == N_run][['tBdt','tSdt','tPdt','tVdt']].to_numpy()
pyoplot(t[idx],data[idx,:],t[idx],mod_y,'pyomo fit')

# =============================================================================
# Use params in mech mpd
# =============================================================================
# def odefunPen(N_t, y, z, t, params=None, extra=0):
def FBPen_true(z,N_t,Sf,F,t,params):
    ind = int(N_t*(len(t)-1)/(max(t))) #Set index for Sf = N_t*66/(198)
    if ind > len(t) - 1:        #Prevent inputs from exceeding time index (66)
        ind = len(t) - 1
#        pdb.set_trace()
    #Set inputs
    # if ind == len(t):
        # pdb.set_trace()
    B = z[0]
    S = z[1]
    P = z[2]
    V = z[3]
#    Sf = z[:,ind,0];    F=z[:,ind,1]
    Sf = Sf[ind];    F=F
    if params is not None:
        cLmax, kL, ki, m_xm, k, kp, mu_m, kx, qpm, Yps, Yxs = params
    else:
        cLmax = 0.0519      #hr-1 
        kL =    0.05
        ki = 1.0            #gm/L added
        m_xm = 0.010        #hr-1  
        k = 0.0137          #hr-1 
        kp = 0.0001         #gm/L
        mu_m = 0.0099       #hr-1 
        kx = 0.3
        qpm = 0.0837        #hr-1
        Yps = 1.2
        Yxs = 0.47
    D = F/V 
#    cLmax = 0.0084      #hr-1 
#    kL =    0.05
#    ki = 1.0            #gm/L added
#    m_xm = 0.029        #h-1  changed
#    k = 0.01          #hr-1 
#    kp = 0.0001         #gm/L
#    mu_m = 0.11       #hr-1 
#    kx = 0.3
#    qpm = 0.004        #hr-1 
    #kinetics
    mu = mu_m*S/(kx*B + 10.0)                   #
    cL = cLmax*B*exp(-S/100.0)/(kL+B+1)
    qp = 1.5*qpm*S*B/(4*kp + S*B*(1 + S/(3*ki)))
    m_x = m_xm*B/(B + 10.0)                     #
    sigL = mu/Yxs + qp/Yps + m_x                #

    #differential equations    
    dBdt = B*(mu-D-cL) 
    dSdt = -sigL*B + (Sf - S)*D
    dPdt = qp*B - P*(D + k)
    dVdt = F
    dydx=[dBdt[0],dSdt[0],dPdt[0],dVdt]
    return dydx

batch_y = np.ones((N_runs,N_t,N_vars))
sim_y = np.ones((N_runs,len(t),N_vars))
for N_run in range(0,N_runs):
    Sf = df_data[df_data['Run'] == N_run]['Sf'].to_numpy()
    Fss = df_data[df_data['Run'] == N_run]['F'].iloc[0]
    batch_y[N_run,:,:] = df_batch_y[df_batch_y['Run'] == N_run][['tB','tS','tP','tV']].to_numpy()
    out = odeint(FBPen_true,true_y[N_run,0,:],t,args=(Sf[:],Fss,t,params.T))
    sim_y[N_run,:,:] = out
# Graph Single Run
N_run = 0
pyoplot(t[idx],batch_y[N_run,:,:],t,sim_y[N_run],'Mech Mod w/Fitted Params')

# Graph All Runs
plotPen(t,sim_y)
#for N_run in range(0,N_runs):
#    pyoplot(t[idx],batch_y[N_run,:,:],t,pred_y[N_run])

# =============================================================================
# # Calculate Squared Errors
# =============================================================================
# Key: o = NODE_pred - Pyomo_fit;     s = MechSim_pred - NoNoiseTrue_y
BMSE_o = sum(abs(restruc2d1d(pred_y[:,:,0].T) - arr_Bmod))/(len(idx)*N_runs)
SMSE_o = sum(abs(restruc2d1d(pred_y[:,:,1].T) - arr_Smod))/(len(idx)*N_runs)
PMSE_o = sum(abs(restruc2d1d(pred_y[:,:,2].T) - arr_Pmod))/(len(idx)*N_runs)
dum = pred_y[:,:,1].T
dum = true_y[:,idx,1].T
BMSE_s = sum(sum(abs(sim_y[:,idx,0] - true_y[:,idx,0])))/(len(idx)*N_runs)
SMSE_s = sum(sum(abs(sim_y[:,idx,1] - true_y[:,idx,1])))/(len(idx)*N_runs)
PMSE_s = sum(sum(abs(sim_y[:,idx,2] - true_y[:,idx,2])))/(len(idx)*N_runs)
print('Pen MSE: ', PMSE_s)
# 0 index removes the numpy array
MSEs = np.array([[BMSE_o[0],BMSE_s], [SMSE_o[0],SMSE_s], [PMSE_o[0],PMSE_s]]).T
