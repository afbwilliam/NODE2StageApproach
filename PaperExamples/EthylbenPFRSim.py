# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:02:41 2020

@author: Afbwi
"""

#Ethylbenzene spyder version
# Purpose: Simulate EB PFR reactor for multiple random initial conditions

from scipy.integrate import odeint
import numpy as np
import math
import pandas as pd
import os; import sys

import matplotlib.pyplot as plt

#Reg_size = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


# Single run
def odefun1(y,t,par=[0],iso=0,test=0):
    #a = Ethylbenzene, b = styrene, c = hydrogen, d = Benzene,
    #e = Ethylene, f = Toluene, g = Meth
    b1=-17.34; b2=-1.302e4; b3=5.051;
    b4=-2.314e-10; b5=1.302e-6; b6=-4.931e-3;
    p=2137; o=0.4;  #Density of pellet:kg/m^3   #Void fraction
    Cpa=299; Cpb=273; Cpc=30; Cpd=201;  #Heat Capacity: J/(mol*K)
    Cpe=90; Cpf=249; Cpg=68; Cpsteam=40;
    P0=2.4; Fsteam=48; Ft=Fsteam+sum(y[1:]); #Inert steam pressure: atm, #Flow rate: mol/s

    # y(0)= Temp y(1)=Fa y(2)=Fb y(3)=Fc
    # y(4)=Fd=Fe y(5)=Ff=Fg
    # dydx(1)=dT/dV dydx(2)=ra=-r1st-r2-r3 
    # dydx(3)=rb=r1st dydx(4)=rc=r1st-r3
    # dydx(5)=rd=re=r2 dydx(6)=rf=rg=r3
    if par[0] == 0:
        par = [-0.08539,13.2392,0.2961,10925,25000,11000]
#        if test == 1:
#            par = [0.9181541,5.61968,1.344605,10925,25000,11000]
    Peb=y[1]/Ft*P0; Pst=y[2]/Ft*P0;  #atm
    Ph2=y[3]/Ft*P0;
    Kp1=math.exp(b1+b2/y[0]+b3*math.log(y[0])+((b4*y[0]+b5)*y[0]+b6)*y[0]); #atm
    if test == 1:
        Kp1=math.exp(b1+b2/y[0]+b3*math.log(y[0])); #atm
#        r1st=p*(1-o)*math.exp(par[0]-par[3]/y[0])*(Peb-Pst*Ph2/Kp1);
#        r2=p*(1-o)*math.exp(par[1]-par[4]/y[0])*(Peb);
#        r3=p*(1-o)*math.exp(par[2]-par[5]/y[0])*(Peb*Ph2);        
#    else:
    r1st=p*(1-o)*math.exp(par[0]-par[3]/y[0])*(Peb-Pst*Ph2/Kp1); #kmol/(m^3*s)
    r2=p*(1-o)*math.exp(par[1]-par[4]/y[0])*(Peb);
    r3=p*(1-o)*math.exp(par[2]-par[5]/y[0])*(Peb*Ph2);
    dhrx1=118000+(299-273-30)*(y[0]-273);       #kJ/kmol ethylbenzene
    dhrx2=105200+(299-201-90)*(y[0]-273);
    dhrx3=-53900+(299+30-249-68)*(y[0]-273);    #converting to degC?
    dydx1=1000*(-r1st*dhrx1-r2*dhrx2-r3*dhrx3)/((y[1]*Cpa+y[2]*Cpb+y[3]* \
               Cpc+y[4]*Cpd+y[4]*Cpe+y[5]*Cpf+y[5]*Cpg+Fsteam*Cpsteam)); #1000J/1kJ
    dydx2=(-r1st-r2-r3)*1000;                                              #1000mol/1kmol
    dydx3=r1st*1000;
    dydx4=(r1st-r3)*1000;
    dydx5=r2*1000;
    dydx6=r3*1000;
    if iso==1:
        dydx1=0
    dydx=[dydx1,dydx2,dydx3,dydx4,dydx5,dydx6]
    return dydx

# Multiple runs
def odefun2(y,t,par=[],iso=0):
    #a = Ethylbenzene, b = styrene, c = hydrogen, d = Benzene,
    #e = Ethylene, f = Toluene, g = Meth
    b1=-17.34; b2=-1.302e4; b3=5.051;
    b4=-2.314e-10; b5=1.302e-6; b6=-4.931e-3;
    p=2137; o=0.4;  #Density of pellet:kg/m^3   #Void fraction
    Cpa=299; Cpb=273; Cpc=30; Cpd=201;  #Heat Capacity: J/(mol*K)
    Cpe=90; Cpf=249; Cpg=68; Cpsteam=40;
    P0=2.4; Fsteam=48; Ft=Fsteam+np.sum(y[:,1:],axis=1); #Inert steam pressure: atm, #Flow rate: mol/s

    # y(0)= Temp y(1)=Fa y(2)=Fb y(3)=Fc
    # y(4)=Fd=Fe y(5)=Ff=Fg
    # dydx(1)=dT/dV dydx(2)=ra=-r1st-r2-r3 
    # dydx(3)=rb=r1st dydx(4)=rc=r1st-r3
    # dydx(5)=rd=re=r2 dydx(6)=rf=rg=r3
    if not par:
        par = [-0.08539,13.2392,0.2961,10925,25000,11000]

    Peb=y[:,1]/Ft*P0; Pst=y[:,2]/Ft*P0;  #atm
    Ph2=y[:,3]/Ft*P0;
    Kp1=np.exp(b1+b2/y[:,0]+b3*np.log(y[:,0])+((b4*y[:,0]+b5)*y[:,0]+b6)*y[:,0]); #atm
    r1st=p*(1-o)*np.exp(par[0]-par[3]/y[:,0])*(Peb-Pst*Ph2/Kp1);
    r2=p*(1-o)*np.exp(par[1]-par[4]/y[:,0])*(Peb);
    r3=p*(1-o)*np.exp(par[2]-par[5]/y[:,0])*(Peb*Ph2);
    dhrx1=118000+(299-273-30)*(y[:,0]-273);       #kJ/kmol ethylbenzene
    dhrx2=105200+(299-201-90)*(y[:,0]-273);
    dhrx3=-53900+(299+30-249-68)*(y[:,0]-273);    #converting to degC?
    dydx1=1000*(-r1st*dhrx1-r2*dhrx2-r3*dhrx3)/((y[:,1]*Cpa+y[:,2]*Cpb+y[:,3]* \
               Cpc+y[:,4]*Cpd+y[:,4]*Cpe+y[:,5]*Cpf+y[:,5]*Cpg+Fsteam*Cpsteam));
    dydx2=(-r1st-r2-r3)*1000;
    dydx3=r1st*1000;
    dydx4=(r1st-r3)*1000;
    dydx5=r2*1000;
    dydx6=r3*1000;
    if iso==1:
        dydx1=0
    dydx=[dydx1,dydx2,dydx3,dydx4,dydx5,dydx6]
    return dydx

def plot_temp(t,sol,title):
    plt.figure()
    plt.plot(t, sol[:, 0], 'b', label='Temperature')
    plt.legend(loc='best')
    plt.xlabel('Volume [m$^3$]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.savefig('Temp {}.png'.format(title),dpi=300)    
    plt.show()
def plot_sol(t,sol,title):
    plt.figure()
    plt.plot(t, sol[:, 1], 'g', label='Ethylbenzene')
    plt.plot(t, sol[:, 2], 'r', label='Styrene')
    plt.plot(t, sol[:, 3], 'b', label='Hydrogen')
    plt.plot(t, sol[:, 4], 'y', label='Benzene/Ethylene')
    plt.plot(t, sol[:, 5], 'm', label='Toluene/Methane')
    plt.legend(loc='best')
    plt.xlabel('Volume [m$^3$]')
    plt.ylabel('Molar Flowrate [mol/s]')
    plt.grid()
    plt.savefig('Conc {}.png'.format(title),dpi=300)
    plt.show()
def plot_soldy(t,sol,title):
    plt.figure()
    plt.plot(t, sol[:, 1], 'g', label='Ethylbenzene')
    plt.plot(t, sol[:, 2], 'r', label='Styrene')
    plt.plot(t, sol[:, 3], 'b', label='Hydrogen')
    plt.plot(t, sol[:, 4], 'y', label='Benzene/Ethylene')
    plt.plot(t, sol[:, 5], 'm', label='Toluene/Methane')
    plt.legend(loc='best')
    plt.xlabel('Volume [m$^3$]')
    plt.ylabel('dFdt [mol/s$^2$]')
    plt.grid()
    plt.savefig('Rates {}.png'.format(title),dpi=300)
    plt.show()
  
    
def CreateFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

np.random.seed(432)

if __name__ == '__main__':
    # Non Isothermal Model
    t = np.linspace(0, 12, 101)
    y0 = np.random.uniform(low=0,high=1,size=(6,6)) * (950-850,5-3,0,0,0,0) + (850,3,0,0,0,0)
    # y0 = np.array([[1050,5,0,0,0,0]])
    N_runs = len(y0[:,0])           #Number of pfr process conditions
    N_t = len(t)                    #Number of time measurments
    N_state_vars = len(y0[0,:])     #Number of state variables
    sol = np.ones((N_runs,len(t),N_state_vars))
    sol_dy = np.ones((N_runs,len(t),N_state_vars))
    for j in range(0,N_runs):
        out = odeint(odefun1, y0[j,:], t,args=([0],0.0,0.0))
        sol[j,:,:] = out
        out = odefun2(out, t,)
        sol_dy[j,:,:] = np.stack(out,axis=1)
    
    # Plotting
    title = 'True Profiles'
    N_run = 0
#    plot_temp(t,sol[1,:,:],title);
    plot_sol(t,sol[N_run,:,:],title)
    plot_soldy(t,sol_dy[N_run,:,:],title)
    sys.exit()
    
    #IsoThermal Model
    iso=1.0
    sol2 = np.ones((N_runs,len(t),N_state_vars))
    for j in range(0,N_runs):
        out = odeint(odefun1, y0[j,:], t,args=([0],iso,))
        sol2[j,:,:] = out
    
    # Plotting
    title = 'Isothermal Profiles'
#    plot_temp(t,sol2[1,:,:],title);
    plot_sol(t,sol2[N_run,:,:],title)
    
    #Save and store data with pandas dataframes
    df_sol = pd.DataFrame(data=sol.reshape(N_t*N_runs,N_state_vars)*1,
                          columns=['Temp','EB','Sty','H2','BeEt','ToMe'])
    df_sol['t'] = np.tile(t,N_runs)
    df_sol['Run'] = np.repeat(np.arange(N_runs),N_t)
    df_sol[['dTdt','dEBdt','dStydt','dH2dt','dBeEtdt','dToMedt']] = pd.DataFrame(
            data=sol_dy.reshape(N_t*N_runs,N_state_vars)*1,
            columns=['dTdt','dEBdt','dStydt','dH2dt','dBeEtdt','dToMedt'])
    df_sol2 = pd.DataFrame(data=sol2.reshape(N_t*N_runs,N_state_vars)*1,
                           columns=['Temp','EB','Sty','H2','BeEt','ToMe'])
    df_sol2['t'] = np.tile(t,N_runs)
    df_sol2['Run'] = np.repeat(np.arange(N_runs),N_t)
    
    #Create directory (if it doesn't exist) and add dataframes
    # CreateFolder('data')
    # df_sol.to_csv(r'data/NonIso_PFR.csv',index = None, header=True) 
    # df_sol2.to_csv(r'data/Iso_PFR.csv',index = None, header=True)  
# sys.exit()
# =============================================================================
# # This second part plots some derivative information (not for Reveiw paper)
# =============================================================================
def r2deriv(t,y,par):
    p=2137; o=0.4;  #Density of pellet:kg/m^3   #Void fraction
    P0=2.4; Fsteam=48; Ft=Fsteam+sum(y[0,1:]); #Inert steam pressure: atm, #Flow rate: mol/s
    
    Peb=y[0,1]/Ft*P0; #Pst=y[:,2]/Ft*P0;  #atm
    r2=1000*p*(1-o)*np.exp(par-25000/y[0,0])*(Peb);
    r2der = 1000*p*(1-o)*np.exp(par-25000/y[0,0])*(Peb);
    return r2, r2der
def plot_r2(A2,r2,deriv,title):
    plt.figure()
    plt.plot(A2,r2,'k',label='Rxn 2')
    
    plt.plot(A2,deriv, 'b', label='Rxn 2 Derivative')
    A2_IC = 1000*2137*(1-0.4)*np.exp(1-25000/950)*4/52*2.4
    A2_true = 1000*2137*(1-0.4)*np.exp(13.2392-25000/950)*4/52*2.4
    plt.plot(1,A2_IC,'go',label='Initial Freq Factor Estimate')
    plt.plot(13.2392,A2_true,'ro',label='True Freq Factor')
    plt.legend(loc='best')
    plt.xlabel('Frequency Factor A')
    plt.ylabel('dr/dA')
    plt.yscale('log')
    plt.savefig('{}.png'.format(title),dpi=300)
    plt.show()

#Derivatives
if __name__ == '__main__':
#    y0 = np.array([[950,4,0,0,0,0],[950,6,0.25,0,0.2,0.2]])
    A2 = np.linspace(0,15,101)
    r2, r2der = r2deriv(t,y0,A2)
    title = 'Derivative Profiles'
#    plot_r2(A2,r2,r2der,title)
    

