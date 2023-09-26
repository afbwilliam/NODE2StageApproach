# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:12:17 2020

@author: Afbwi
"""
import matplotlib.pyplot as plt
import numpy as np

import os
import imageio as io

#List of functions:
'''plot_temp, plot_pred, plot_dy
class visuals
plot_Pendy'''

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


# Plot Styrene
def plot_temp(t,pred_y,batch_y,true_y,idx,title):
    plt.figure()
#    plt.plot(t[idx],true_y[idx,0],'kx',label='True Temp')
#    plt.plot(t,MMS.transform(true_y[N_run,:,:])[:,0],'ko',label='True Temp')
    plt.plot(t,pred_y[:,0],'b',label='Temperature')
    plt.plot(t[idx],batch_y[:,0:1],'bo',label='Meas Temp')#MMS.inverse_transform(batch_y[:,:])[:,0:1],'bo')
#    plt.plot(t[idx],batch_y[N_run][:,0:1],'co')
    plt.title(title)
    plt.xlabel('t (meters)')
    plt.ylabel('Temperature (K)')
    plt.legend(loc='best')
    plt.savefig('Visuals/StyTempstate{}'.format(title),dpi=300)
    plt.show()

def plot_pred(t,pred_y,batch_y,true_y,idx,title):
    plt.figure()
#    plt.subplot(211)
    plt.plot(t,pred_y[:,1],'c',label='Ethylbenzene')
    plt.plot(t,pred_y[:,2],'r',label='Styrene')
    plt.plot(t,pred_y[:,3],'b',label='Hydrogen')
    plt.plot(t,pred_y[:,4],'y',label='Benzene/Ethylene')
    plt.plot(t,pred_y[:,5],'m',label='Toluene/Methane')
    
#    plt.plot(t,MMS.transform(true_y[N_run,:,:])[:,1],'ko',label='True Ethylbenzene')
#    plt.plot(t,true_y[N_run,:,1],'kx',label='True Ethylbenzene')
    #plt.show()
    plt.title(title)
    plt.xlabel('t (meters)')
    plt.ylabel('Molar Flow Rate (mol/s)')
    plt.plot(t[idx],batch_y[:,1:2],'co',
             t[idx],batch_y[:,2:3],'ro',
             t[idx],batch_y[:,3:4],'bo',
             t[idx],batch_y[:,4:5],'yo',
             t[idx],batch_y[:,5:6],'mo')
#    plt.plot(t[idx],MMS.inverse_transform(batch_y[N_run])[:,1:2],'go',)
#    plt.plot(t[idx],MMS.inverse_transform(batch_y[N_run])[:,2:3],'ro',)
#    plt.plot(t[idx],MMS.inverse_transform(batch_y[N_run])[:,3:4],'bo',)
#    plt.plot(t[idx],MMS.inverse_transform(batch_y[N_run])[:,4:5],'yo',)
#    plt.plot(t[idx],MMS.inverse_transform(batch_y[N_run])[:,5:6],'mo')
    plt.legend(loc='best')
#    plt.subplot(212)
    plt.savefig('Visuals/StyChemstate{}'.format(title),dpi=300)
    plt.show()

def plot_dy(t,pred_dy,true_dy,idx,title,noise):
    plt.figure()
    plt.plot(t,pred_dy[:,0],'b')
    plt.plot(t[idx],true_dy[idx,0],'bx')
    plt.xlabel('t (meters)'); plt.ylabel('dTdt')
    plt.title(title)
    plt.legend(['predict dT/dt','True dT/dt']); plt.show()
    plt.subplots_adjust(left=0.15)
    plt.savefig('Visuals/StydTdt{}N'.format(int(noise)*100),dpi=300)
    
    plt.figure()
    plt.plot(t,pred_dy[:,1],'c',label='predict dEB/dt')
    plt.plot(t,pred_dy[:,2],'r',label='predict dSty/dt')
    plt.title(title)
    plt.xlabel('t (meters)')
    plt.ylabel('dEB/dt, dSty/dt')
    plt.plot(t[idx],true_dy[idx,1],'cx',label='True dEB/dt')
    plt.plot(t[idx],true_dy[idx,2],'rx',label='True dSty/dt')
    plt.legend(loc='best')
#    plt.tight_layout()
    plt.savefig('Visuals/StydCdt{}N'.format(int(noise)*100),dpi=300)
    plt.show()

# Plot Pen, Lovo, Sty
class visuals():
    def __init__(self,args,N_vars):
        self.args = args
        self.N_vars = N_vars
        if args.mod == 'LoVo':
            self.fig = plt.figure()
#        if args.mod == 'Pen':
        else:
            self.fig, self.axes = plt.subplots(2, int((N_vars+1)/2), sharex='col')#, sharey='row')
        
    # Function for graphing LoVo only
    def visualize(self,t,true_y, true_t, pred_y, idx, itr):
        N_runs = len(pred_y[:,0,0])
#        if self.args.viz == 'ClearGraph': #For loVo, single timeline
        plt.cla()
        for N_run in range(0,N_runs):
            tp = true_t[N_run]

            plt.plot(tp,pred_y[N_run,:,0].T,'g',tp,pred_y[N_run,:,1].T,'g')
#            plt.plot(tp[:],true_y[N_run,:,0].T,'r--',tp[:],true_y[N_run,:,0].T,'ko',
#                     tp[:],true_y[N_run,:,1].T,'b--',tp[:],true_y[N_run,:,1].T,'ko')
        plt.plot(t[:],true_y[0,:,0].T,'r--',label='x'); plt.plot(t[idx],true_y[0,idx,0].T,'ko')
        plt.plot(t[:],true_y[0,:,1].T,'b--',label='y'); plt.plot(t[idx],true_y[0,idx,1].T,'ko', label='data')
        plt.legend(loc='upper right')
        plt.title('Training progression Neural ODE')
        plt.xlabel('t')
        plt.ylabel('x,y')
        plt.savefig('Visuals/LoVo{:03d}'.format(itr),dpi=300)
        plt.show()
    # Function for graphing Pen and Sty
    def final_vis(self,t,true_y,idx,dat_y,true_t,pred_y,noise): #single run, multiple IC
        if self.args.viz == 'Subplots':
            i = 0; j = 0;
            N_runs = len(pred_y[:,0,0])
#            idx = np.round(np.linspace(0, len(t) - N_Bsteps-1, N_dat)).astype(int) # 10 data points
            for N_var in range(0,self.N_vars):
                for N_run in range(0,N_runs):
#                    tp = t[idx][N_run:N_run+N_Bsteps+1]
                    self.axes[i,j].plot(true_t[N_run,:],pred_y[N_run,:,N_var],'g')
#                    self.axes[i,j].plot(tp,pred_y[N_run,:,0].T,'g')
#                    self.axes[i,j].plot(tp,pred_y[N_run,:,1].T,'g')
#                    self.axes[i,j].plot(tp,pred_y[N_run,:,2].T,'g')
                self.axes[i,j].plot(t[:],true_y[:,N_var],'r--',label='true')
                self.axes[i,j].plot(t[idx],dat_y[:,N_var].T,'ko',label='data')
#                self.axes[i,j].plot(tp[:],true_y[N_run,:,0].T,'r--',tp[idx],true_y[N_run,idx,0].T,'ko',
#                         tp[:],true_y[N_run,:,1].T,'b--',tp[idx],true_y[N_run,idx,1].T,'ko')

#                self.axes[i,j].plot(t,pred_y[:,N_var],'g',label='Neural ODE')
#                self.axes[i,j].plot(t,true_y[:,N_var],'ko',label='data')
#                plt.xlabel('t')
                
#                self.axes[i,j].set_ylabel(states[N_var])
                self.axes[i,j].legend()
                if N_var % 2 == 0:
                    i = 1
                    self.axes[i,j].set_xlabel('time (hr)')
                else:
                    i = 0
                    j = j + 1
            self.axes[0,0].set_ylabel('Biomass (g/L)'); self.axes[0,1].set_ylabel('Penicillin (g/L)');
            self.axes[1,0].set_ylabel('Substrate (g/L)'); self.axes[1,1].set_ylabel('Volume (L)')
            self.fig.suptitle('Nueral ODE Fit')
            plt.tight_layout()
        plt.tight_layout()
        plt.savefig('data/{}MultipleIC{}N'.format(self.args.mod,int(noise*100)),dpi=300)
            
    # For graphing Penicillin
    def single_graph(self,t,true_y,idx,dat_y,pred_y,noise): #single run, single IC
        if self.args.viz == 'Subplots':
            i = 0; j = 0;
            tp = t
#            idx = np.round(np.linspace(0, len(t) - N_Bsteps-1, N_dat)).astype(int) # 10 data points
            for N_var in range(0,self.N_vars):
                self.axes[i,j].plot(tp,pred_y[:,N_var],'g',label='predict')
                self.axes[i,j].plot(t[:],true_y[:,N_var],'r--',label='true')
                self.axes[i,j].plot(t[idx],dat_y[:,N_var],'ko',label='data')

                if N_var % 2 == 0:
                    i = 1            
                    self.axes[i,j].set_xlabel('time (hr)')
                else:
                    i = 0
                    j = j + 1
            self.axes[i,j-1].legend()
            self.axes[0,0].set_ylabel('Biomass (g/L)'); self.axes[0,1].set_ylabel('Penicillin (g/L)');
            self.axes[1,0].set_ylabel('Substrate (g/L)'); self.axes[1,1].set_ylabel('Volume (L)')
        self.fig.suptitle('Nueral ODE Fit')
        plt.tight_layout()
        plt.savefig('data/PenSingleIC{}N'.format(int(noise*100)),dpi=300)

def plot_Pendy(t,pred_dy,true_dy,idx,title,noise):
    plt.figure()
    plt.subplot(221)
    #plt.title('Comparison of true and mechanistic model predictions')
    st = 0; st2 = 0
#    st = 1
#    st2 = 7
    plt.plot(t[st2:],pred_dy[st2:,0],label='Predict')
    plt.plot(t[idx][st:],true_dy[st:,0],'kx',label='True')
    plt.xlabel('time (hr)')
    plt.ylabel('dB/dt (g/L/hr)')
    plt.legend()
#    plt.show()
    #plt.legend(['B','S','P','V'])
    plt.subplot(223)
    plt.plot(t[st2:],pred_dy[st2:,1])
    plt.plot(t[idx][st:],true_dy[st:,1],'kx',)
    plt.xlabel('time (hr)')
    plt.ylabel('dS/dt (g/L/hr)')
#    plt.show()
    plt.subplot(222)
    plt.plot(t[st2:],pred_dy[st2:,2])
    plt.plot(t[idx][st:],true_dy[st:,2],'kx',)
    plt.xlabel('time (hr)')
    plt.ylabel('dP/dt (g/L/hr)')
#    plt.show()
    if len(pred_dy[0,:]) > 3:
        plt.subplot(224)
        plt.plot(t[st2:],pred_dy[st2:,3])
        plt.plot(t[idx][st:],true_dy[st:,3],'kx',)
        plt.xlabel('time (hr)')
        plt.ylabel('dV/dt (L/hr)')
        plt.ylim((.05,.15))
        plt.show()
#    plt.suptitle('Nueral ODE Fit')
    plt.tight_layout()
    plt.savefig('data/Pendy{}N'.format(int(noise*100)),dpi=300)

def plot_LoVo(t,sim_y,batch_y,true_y,idx,title,noise,idxes,final=False):
    plt.figure(10)
    ls = '-'
    # if N_hnodes == 7: ls = '-'; 
    # if N_hnodes==10: ls='--'; 
    # if N_hnodes==15: ls='-.' 
    # plt.plot(t,sim_y[:,0],'b',linestyle=ls, label='x_{}nodes'.format(N_hnodes))
    # plt.plot(t,sim_y[:,1],'r',linestyle=ls, label='y_{}nodes'.format(N_hnodes))
    # Original plotting code
    if final == True:
        plt.plot(t,sim_y[:,0],'c',linewidth=3)
        plt.plot(t,sim_y[:,1],'c',linewidth=3)
    else:
        plt.plot(t,sim_y[:,0],'b',)#label='x')
        plt.plot(t,sim_y[:,1],'r',)#label='y')

    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('x,y')
    # plt.plot(t,true_y[:,0:1],'k--',
    #           t,true_y[:,1:2],'k--')   # true data 
    # plt.plot(t[idx],batch_y[:,0:1],'bo',
    #           t[idx],batch_y[:,1:2],'ro',)

    plt.plot(t[idx],batch_y[:,0:1],'bo', 
              t[idx],batch_y[:,1:2],'ro',) # sparse data
    plt.legend(['x NODE','y NODE','x data', 'y data'],loc='upper right')
    # plt.legend(['x true','y true','x data', 'y data', ])
    if t[-1] > 6: # Extrapolation
        plt.savefig('Visuals/LovoNODE{}NExtrap'.format(int(noise*100)),dpi=300)
    else:
        plt.savefig('Visuals/LovoNODE{}N'.format(int(noise*100)),dpi=300)
    plt.show()
    
def plot_LoVody(t,sim_dy,true_dy,idx,title,noise,idxes,final=False):
    plt.figure(11)
    ls = '-'
    # if N_hnodes == 7: ls = '-'; 
    # if N_hnodes==10: ls='--'; 
    # if N_hnodes==15: ls='-.' 
    # plt.plot(t,sim_dy[:,0],'b',linestyle=ls, label='x_{}nodes'.format(N_hnodes))
    # plt.plot(t,sim_dy[:,1],'r',linestyle=ls, label='y_{}nodes'.format(N_hnodes))
    # Original plotting code
    if final == True:
        plt.plot(t,sim_dy[:,0],'c',linewidth=3)
        plt.plot(t,sim_dy[:,1],'c',linewidth=3)
    else:
        plt.plot(t,sim_dy[:,0],'b',)#label='x')
        plt.plot(t,sim_dy[:,1],'r',)#label='y')
    plt.plot(t,true_dy[:,0:1],'k--',
              t,true_dy[:,1:2],'k--') # true data
    # plt.legend(['x NODE','y NODE','true derivatives'],loc='best')
    plt.xlabel('time')
    plt.ylabel('dx/dt,dy/dt')
    plt.plot(t[idx],true_dy[idx,0],'bo')
    plt.plot(t[idx],true_dy[idx,1],'ro')
    # plt.plot(t[idx][idxes],true_dy[idx,0][idxes],'bo') # sparse data
    # plt.plot(t[idx][idxes],true_dy[idx,1][idxes],'ro')
    # plt.legend(['dx NODE','dy NODE','true derivatives'],loc='best')
    plt.tight_layout()
    plt.savefig('Visuals/LovoNODE_dy{}N'.format(int(noise*100)),dpi=300)
    plt.show()
    
def plotMKM(t,pred_y,batch_y,true_y,idx,title):
    plt.figure()
    ls = '-'
    # plt.title('da MKM')
    plt.plot(t,pred_y[:,0],ls,color='c')
    plt.plot(t,pred_y[:,1],ls,color='g')
    plt.plot(t,pred_y[:,2],ls,color='y')
    
    plt.plot(t[idx],batch_y[:,0],'co')
    plt.plot(t[idx],batch_y[:,1],'go')
    plt.plot(t[idx],batch_y[:,2],'yo')    
    plt.xlabel('time')
    plt.ylabel('concentration')
    plt.legend(['A','B','C'],loc='best')
    plt.show()
    
    plt.figure()
    # plt.title('da MKM ads')
    plt.plot(t,pred_y[:,3],ls,color='c')
    plt.plot(t,pred_y[:,4],ls,color='g')
    plt.plot(t,pred_y[:,5],ls,color='y')
    plt.plot(t,pred_y[:,6],ls,color='m')
    plt.legend(['Astar','Bstar','Cstar','star'],loc='best')
    if len(pred_y[0,:]) > 6: #(i.e. we are using dcMKM)
        plt.plot(t,pred_y[:,7],ls,color='orange')
        plt.plot(t[idx],batch_y[:,7],'o',color='orange')
        plt.legend(['Astar','Bstar','Cstar','star','Dstar'],loc='best')
    
    plt.plot(t[idx],batch_y[:,3],'co',t[idx],batch_y[:,4],'go',
             t[idx],batch_y[:,5],'yo',t[idx],batch_y[:,6],'mo')
    plt.xlabel('time')
    plt.ylabel('concentration')
    
    plt.show()
    
def plot_MKMdy(t,sim_dy,true_dy,idx,title,noise):
    plt.figure()
    ls = '-'
    # idx = idx[1:] # don't plot initial values
    plt.plot(t,sim_dy[:,0],ls,color='c')
    plt.plot(t,sim_dy[:,1],ls,color='g')
    plt.plot(t,sim_dy[:,2],ls,color='y')
    plt.plot(t[idx],true_dy[idx,0],'cx',t[idx],true_dy[idx,1],'gx',
             t[idx],true_dy[idx,2],'yx')
    plt.xlabel('time')
    plt.ylabel('dx/dt')
    plt.legend(['dAdt','dBdt','dCdt'],loc='best')
    plt.show()
    
    plt.figure()
    plt.plot(t,sim_dy[:,3],ls,color='c')
    plt.plot(t,sim_dy[:,4],ls,color='g')
    plt.plot(t,sim_dy[:,5],ls,color='y')
    plt.plot(t,sim_dy[:,6],ls,color='m')
    plt.legend(['dAstardt','dBstardt','dCstardt','dstardt'],loc='best')
    if len(sim_dy[0,:]) > 6: #(i.e. we are using dcMKM)
        plt.plot(t,sim_dy[:,7],ls,color='orange',)
        plt.plot(t[idx],true_dy[idx,7],'x',color='orange')
        plt.legend(['dAstar/dt','dBstar/dt','dCstar/dt','dstar/dt','dDstar/dt'],loc='best')

    plt.plot(t[idx],true_dy[idx,3],'cx',t[idx],true_dy[idx,4],'gx',
             t[idx],true_dy[idx,5],'yx',t[idx],true_dy[idx,6],'mx')
    plt.xlabel('time')
    plt.ylabel('dx/dt')
    
    plt.show()

def plot_robert(t,pred_y,batch_y,idx,title,noise):
    plt.figure()
    # plt.title('da MKM')
    plt.plot(t,pred_y[:,0],'c-v',t,pred_y[:,1],'g-v',t,pred_y[:,2],'y-v')
    plt.plot(t[idx],batch_y[:,0],'co',t[idx],batch_y[:,1],'go',
             t[idx],batch_y[:,2],'yo')    
    plt.xlabel('time')
    plt.ylabel('concentration')
    plt.legend(['A','B','C'],loc='best')
    plt.show()

# Hasn't been modified from plotMKM yet
def plot_MAPK(t,pred_y,batch_y,idx,title,):
    plt.figure()
    ls = '-'
    plt.title('da MKM')
    N_vars = len(pred_y[0])
    for var in range(N_vars):
        plt.plot(t,pred_y[:,var])
    for var in range(N_vars):
        plt.plot(t[idx],batch_y[:,var],'ko')

    # plt.plot(t,pred_y[:,0],ls,color='c')
    # plt.plot(t,pred_y[:,1],ls,color='g')
    # plt.plot(t,pred_y[:,2],ls,color='y')
    
    # plt.plot(t[idx],batch_y[:,0],'co')
    # plt.plot(t[idx],batch_y[:,1],'go')
    # plt.plot(t[idx],batch_y[:,2],'yo')    
    
    plt.xlabel('time')
    plt.ylabel('concentration')
    plt.legend(['x0','x1','x2','x3','x4','x5','x6','x7','x8',],loc='best')
    # plt.legend(['A','B','C'],loc='best')
    plt.show()
    
    # plt.figure()
    # # plt.title('da MKM ads')
    # plt.plot(t,pred_y[:,3],ls,color='c')
    # plt.plot(t,pred_y[:,4],ls,color='g')
    # plt.plot(t,pred_y[:,5],ls,color='y')
    # plt.plot(t,pred_y[:,6],ls,color='m')
    # plt.legend(['Astar','Bstar','Cstar','star'],loc='best')
    # if len(pred_y[0,:]) > 6: #(i.e. we are using dcMKM)
    #     plt.plot(t,pred_y[:,7],ls,color='orange')
    #     plt.plot(t[idx],batch_y[:,7],'o',color='orange')
    #     plt.legend(['Astar','Bstar','Cstar','star','Dstar'],loc='best')
    
    # plt.plot(t[idx],batch_y[:,3],'co',t[idx],batch_y[:,4],'go',
    #          t[idx],batch_y[:,5],'yo',t[idx],batch_y[:,6],'mo')
    # plt.xlabel('time')
    # plt.ylabel('concentration')
    
    # plt.show()
    
def plot_MAPKdy(t,pred_dy,true_dy,idx,title):
    plt.figure()
    N_vars = len(pred_dy[0])
    for var in range(N_vars):
        plt.plot(t,pred_dy[:,var])
    for var in range(N_vars):
        plt.plot(t[idx],true_dy[idx,var],'ko')
    plt.xlabel('time')
    plt.ylabel('concentration')
    plt.legend(['dx0','dx1','dx2','dx3','dx4','dx5','dx6','dx7','dx8',],loc='right')
    plt.show()

# Plot tanh and sec^2
def simple_plot(x,y,z):
    plt.figure()
    plt.subplot(121)
    plt.plot(x,y,'c--',label='tanh(x)')
    plt.axvline(linewidth=1, color='k')
    plt.axhline(y=0,linewidth=1,color='k')
    plt.legend()

    plt.subplot(122)
    plt.plot(x,z,'g-.',label='$sec^{2}$(x)')
    plt.axvline(linewidth=1, color='k')
    plt.axhline(y=0,linewidth=1,color='k')
    plt.legend()
    plt.show()
#    '$q_p$ ($hr^{-1}$)'

# Plot nonlinear functions: tanh vs deriv(tanh)
if __name__ == '__main__':
    x = np.linspace(-2,2,50)
    y = np.tanh(x)
    z = 1 - np.tanh(x)**2
    simple_plot(x,y,z)
    
    
    #CREATE SORTED LIST OF FILENAMES
    file_names = sorted((fn for fn in os.listdir('./scratch') if fn.startswith('LoVo')))
    #USE IMAGE IO TO CREATE A GIF
    with io.get_writer('scratch/LoVo1.gif', mode='I', duration=0.2) as writer:
        #ITERATE OVER FILENAMES
        for filename in file_names:
            #READ IN FILE
            image = io.imread(f'scratch/{filename}')
            #APPEND FILE TO GIF
            writer.append_data(image)
    writer.close()