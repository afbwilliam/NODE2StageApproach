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

from AG_Funs import odeint, Store

# Configure a few plotting variables
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

# Fix seed so that Neural ODE parameter initialization is the same each time
torch.manual_seed(34)
np.random.seed(84)

# =============================================================================
# # Create dataset, format for future PyTorch functions
# =============================================================================
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=140) #changed from 2000
parser.add_argument('--test_freq', type=int, default=4) #changed from 20
args = parser.parse_args()
args.data_size = 20
args.mod = 'RxnABC'

# Create kinetic model to simulate training data for Neural ODE
# A simple reversible reaction with product B degrading to C:
    # A <--> B and 2*B --> C
# Equations: 
    # dCa/dt = -k1*Ca + k2*Cb
    # dCb/dt = k1*Ca - k2*Cb - k3*2*Cb
    # dCc/dt = k3*Cb
states = ['A','B','C']
derivs = ['dAdt','dBdt','dCdt']

# Stoichiometry
true_A = torch.tensor([[-1.0,1.0,0.0], [1.0,-1.0,-2.0], [0.0,0.0,1.0]]).T

# Define model and reaction conditions 
class Lambda(nn.Module):
    def __init__(self):
        super(Lambda, self).__init__()
        self.k = k
        
    def forward(self, N_t, x, z=0, t=0):
        Ca = x[:,0]
        Cb = x[:,1]
        terms = torch.stack((Ca,Cb,Cb),dim=1)
        return torch.mm(terms*self.k,true_A) #Simple model with 3 parametetrs 
true_y0 = torch.tensor([[2., 1.0, 0.0]]) # Initial conditions
t = np.linspace(0., 10., args.data_size)
k = torch.tensor([0.1,0.06,0.03],dtype=torch.float32,requires_grad=True) #correct k values

# Simulate model. Collect training data for NODE.
with torch.no_grad(): #plotting data and ground truth in loss func do not need gradient
    true_mod = Lambda()
    true_y = odeint(true_mod, true_y0, t, method='dopri5')
    max_y = np.amax(np.amax(true_y.numpy(),axis=1),axis=0) #for adding noise
N_t = len(t)
N_runs = len(true_y0.numpy()[:,0])
N_vars = 3
N_run = 0

# Preliminary plot of training data
#plt.figure()
#plt.plot(t,true_y[N_run][:,0],'b',t,true_y[N_run][:,1],'r')
#plt.plot(t,true_y[N_run][:,2],'c')
#plt.title('Trajectories True Function')
#plt.xlabel('t')
#plt.ylabel('Ca,Cb')
#plt.legend(['Ca','Cb','Cc'])
#plt.show()

# Add noise to all training data
noise = 0.00
idx = np.round(np.linspace(0, len(true_y[0][:,0]) - 1, 10)).astype(int) # 10 data points
batch_t = t[idx]
batch_y = true_y[:,idx,:]

batch_y[:,:,0:1] = true_y[:,idx,0:1] + np.random.normal(loc=0.0,scale=noise*max_y[0],size=(N_runs,len(idx),1))
batch_y[:,:,1:2] = true_y[:,idx,1:2] + np.random.normal(loc=0.0,scale=noise*max_y[1],size=(N_runs,len(idx),1))
batch_y[:,:,2:3] = true_y[:,idx,2:3] + np.random.normal(loc=0.0,scale=noise*max_y[2],size=(N_runs,len(idx),1))
with torch.no_grad():
    true_dy2d = true_mod(t,true_y[:,idx,:].reshape((-1,N_vars)))
#    max_y = np.amax(batch_y.numpy(),axis=0) #for scaling loss function
    max_y = np.amax(np.amax(batch_y.numpy(),axis=1),axis=0)
    min_y = np.amin(np.amin(batch_y.numpy(),axis=1),axis=0) #for scaling dy
#    min_y = np.amin(batch_y.numpy(),axis=0)

# Create var for y data after adding noise, before rearranging and scaling
torch_y = batch_y[:,:,:].detach().clone()

# Scale the data
from sklearn.preprocessing import MinMaxScaler
MMS = MinMaxScaler()
batch_y2d = torch.tensor(MMS.fit_transform(batch_y.reshape((-1,N_vars))),
                       dtype=torch.float32) # for scaling loss function
batch_y = batch_y2d.reshape((N_runs,len(idx),N_vars)).clone().detach()
true_y2d = true_y[:,idx,:].reshape((-1,N_vars)).clone().detach()
batch_y0 = torch.tensor(MMS.transform(true_y0),
                        dtype=torch.float32)
batch_dy = true_mod(t,batch_y2d) #True derivatives of noisy data, 2d

# =============================================================================
# # Set-up purely data-driven Neural ODE
# =============================================================================
class ODEFunct(nn.Module): # Nueral ODE Function
    def __init__(self):
        super(ODEFunct, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 10),
            nn.Tanh(),
            nn.Linear(10, 3),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.2)
                nn.init.constant_(m.bias, val=0)
    # Black-box Model
    def forward(self, N_t, x, z=0, t=0):
        terms = self.net(x)
        return terms
def Check_point(history,losses): #Check for stopping criteria while updating history
    tot_loss,reglos,los_raw,losdy_raw,los,losdy = losses
    stop = False
    # Stop training if 10-fold increase in reg loss (i.e. training diverges)
    if len(history.reg_loss) > 10:
        if reglos/10 > history.reg_loss[-10]:
            print('reg loss is increasing X10'); 
            stop = True #break
    # Stop training if loss does not substantially improve (i.e. training converges)
    if len(history.tot_loss) > 12:
        stag_iters = sum(history.tot_loss[-12:-2] - history.tot_loss[-11:-1] <1*10**-5)
        if stag_iters == 10:
            print('Total loss did not improve more than 0.00001 for 10 consecutive epochs')
            stop = True #break
    return stop 

def closure():
    optimizer.zero_grad()
    pred_y = odeint(funct, batch_y0, t)
#    pred_y2d = pred_y[:,idx,:].reshape((-1,N_vars))
    #    los = torch.mean((pred_y[0][idx,:] - batch_y)**2) #+ torch.mean((pred_dy-dCidt)**2)
    reglos = torch.sum(w1**2) + torch.sum(w2**2) + \
        torch.sum(w3**2) + torch.sum(w4**2)
    los = torch.mean((pred_y[:,idx,:] - batch_y)**2) +lamb*reglos#+ torch.mean((pred_ys[idx,:] - batch_y)**2)
    los.backward()
    return los

def compute_losses(model,pred_y,batch_y,true_y):
    # All losses are 2-dimensional and indexed
    # Regulaization loss
    reglos = torch.sum(w1**2) + torch.sum(w2**2) + \
        torch.sum(w3**2) + torch.sum(w4**2)
    # Loss in loss function    
    tot_loss = torch.mean(((pred_y - batch_y))**2) + lamb*reglos
    with torch.no_grad():
        # Loss between prediction and noisy batch data
        los_raw = torch.mean((pred_y - batch_y)**2)
        # Loss between Derivative prediction and noisy batch data
        pred_dy = model(t, batch_y).detach().numpy()*(max_y - min_y)
        pred_dy = torch.tensor(pred_dy,dtype=torch.float32)
        losdy_raw = torch.mean((pred_dy - batch_dy)**2) #this doesn't mean much
        
        # Loss between Derivative pred and true deriatives
        pred_dy = model(t, pred_y).detach().numpy()*(max_y - min_y) #forward func
        pred_dy = torch.tensor(pred_dy,dtype=torch.float32)
        losdy = torch.mean((pred_dy - true_dy2d)**2)
    # Loss between pred and true data
    with torch.no_grad():
        true_y = torch.tensor(MMS.transform(true_y),dtype=torch.float32)
        los = torch.mean((pred_y - true_y)**2)
    losses = [tot_loss,reglos,los_raw,losdy_raw,los,losdy]
    stop = Check_point(history,losses)
    history.l_update(losses)
    return [tot_loss, stop]


# =============================================================================
# # Training of Neural ODE
# =============================================================================
learning_rate = 0.1
lamb = 0.0001

pred_y = odeint(ODEFunct(), batch_y0, t, method='dopri5')
print(f'Prediction before training: f([[2., 1.0, 0.0]]) = {pred_y[0][-1].detach().numpy()}') # [1.379,.379,0.621,1.121]
loss = torch.mean(torch.abs(pred_y[0][idx,:] - true_y[0][idx,:]))
print('Iter {:04d} | Abs Loss {:.6f}'.format(0, loss.item()))

# Set up NODE for training
if __name__ == '__main__':
    ii = 0
    plt.figure()

    funct = ODEFunct()
    params = list(funct.parameters()) #+ list([k])
    [w1,w2,w3,w4] = funct.parameters()
    history = Store()

    # Select optimizer
#    optimizer = torch.optim.SGD(params,lr=learning_rate)
    optimizer = torch.optim.LBFGS(params, lr=learning_rate,history_size=10, max_iter=4) #AG default
#    optimizer = torch.optim.LBFGS([func.k], lr=learning_rate,history_size=10, max_iter=4) #AG default
#    optimizer = optim.RMSprop(params, lr=1e-2, weight_decay=0.5)
    end = time.time()
    tot_time = time.time()
    
    # Training loop
    for itr in range(1, args.niters + 1):
        ii += 1
        optimizer.zero_grad()
        pred_y = odeint(funct, batch_y0, t)
        pred_y2d = pred_y[:,idx,:].reshape((-1,N_vars))
        loss, stop = compute_losses(funct,pred_y2d,batch_y2d,true_y2d)
        if stop == True:
            break
        loss.backward()
        optimizer.step(closure)
#        optimizer.step()
        with torch.no_grad():
            l = torch.mean(torch.abs(pred_y[0][idx,:] - true_y[0][idx,:]))
            if itr % args.test_freq == 0:
                pred_y = MMS.inverse_transform(odeint(funct, batch_y0, t)[0])
                print('Iter {:04d} | Abs Loss {:.6f} | Tot Loss {:6f}'.format(
                        itr, l.item(), loss.item()))
                plt.plot(t,pred_y[:,0],'g',t,pred_y[:,2],'g')
            
        end = time.time() #duration of fitting
    tot_time = time.time() - tot_time
print('Sim time: ',tot_time)
with torch.no_grad():
    pred_y = MMS.inverse_transform(odeint(funct, batch_y0, t)[0]) #replacing true_y0 

# =============================================================================
# # Let's visualize those results!
# =============================================================================
plt.plot(t[:],true_y[0][:,0],'b',t[idx],true_y[0][idx,0],'ko',
         t[:],true_y[0][:,2],'b',t[idx],true_y[0][idx,2],'ko')
plt.plot(t,pred_y[:,0],'r--',t,pred_y[:,2],'r--',label='final')
plt.legend(['Predict Ca','Predict Cc'])
plt.title('Training progression Neural ODE')
plt.xlabel('t')
plt.ylabel('Concentration')
plt.show()

# dy graphs
plt.figure()
N_run = 0
dum = torch.tensor(MMS.transform(true_y[N_run]),dtype=torch.float32)
#pred_dy = MMS.inverse_transform(funct(t, dum).detach().numpy())
pred_dy = funct(t,dum).detach().numpy()
pred_dy = pred_dy*(max_y - min_y)
plt.plot(t,pred_dy[:,0],'b',t,pred_dy[:,1],'r')
plt.plot(t,pred_dy[:,2],'c')#,t,pred_dy[:,3],'g')
#plt.plot(t,pred_dy[:,4],'p')
plt.title('Trajectories NN+ODE')
plt.xlabel('t')
plt.ylabel('dCadt,dCbdt')
plt.plot(t[idx],true_dy2d[:,0],'bo',t[idx],true_dy2d[:,1],'ro',
         t[idx],true_dy2d[:,2],'co')
plt.legend(['dCadt','dCbdt','dCcdt'])#,'dCddt','dCedt','true data'])
plt.show()

# y graphs
plt.figure()
plt.plot(t,pred_y[:,0],'b',t,pred_y[:,1],'r')
plt.plot(t,pred_y[:,2],'c')#,t,pred_y[:,3],'g')
#plt.plot(t,pred_y[:,4],'p')
plt.title('Trajectories NN+ODE')
plt.xlabel('t')
plt.ylabel('Concentration')
plt.plot(t[idx],MMS.inverse_transform(batch_y2d)[:,0:1],'bo',
         t[idx],MMS.inverse_transform(batch_y2d)[:,1:2],'ro',
         t[idx],MMS.inverse_transform(batch_y2d)[:,2:3],'co')
#plt.plot(t[idx],true_y[0][idx,0],'bo',t[idx],true_y[0][idx,1],'ro',
#         t[idx],true_y[0][idx,2],'co')
plt.legend(['Ca','Cb','Cc'])#,'Cd','Ce','true data'])
plt.show()

# Change in loss function
plt.figure()
plt.plot(np.arange(1,ii+1,1),history.y_rawloss[1:])
plt.plot(np.arange(1,ii+1,1),history.dy_rawloss[1:]*10)
plt.plot(np.arange(1,ii+1,1),history.reg_loss[1:]*0.2)
plt.plot(np.arange(1,ii+1,1),history.tot_loss[1:])
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['Raw Loss','Rate Loss*10','Reg Loss*0.2','Total loss'])
plt.title('Change in loss vs epoch')
plt.show()


# =============================================================================
# # Save results in csv files
# =============================================================================
# Create df of state and deriv results
df_pred_dt = pd.DataFrame(data=pred_dy.reshape((-1,N_vars)),
                         columns=derivs)
df_pred_dt['t'] = np.tile(t[:],len(torch_y[:,0,0]))
df_pred_dt['Run'] = np.repeat(np.arange(len(torch_y[:,0,0])),len(t))

df_pred_y = pd.DataFrame(data=pred_y.reshape(N_t*len(torch_y[:,0,0]),N_vars)*1,
                      columns=states)
df_pred_y['t'] = np.tile(t,len(torch_y[:,0,0]))
df_pred_y['Run'] = np.repeat(np.arange(len(torch_y[:,0,0])),N_t)

# dum = torch_y2d #.reshape(len(idx)*N_runs,N_vars)*1
# dum2 = MMS.inverse_transform(dum).numpy()#dum*(max_y - min_y) + min_y                    

df_batch_y = pd.DataFrame(data=torch_y.numpy()[0],
                      columns=states)
df_batch_y['t'] = np.tile(t[idx],len(torch_y[:,0,0]))
df_batch_y['Run'] = np.repeat(np.arange(len(torch_y[:,0,0])),len(idx))



# Save batch, predicted dydt and predicted y data respecitvely
df_batch_y.to_csv(r'data/batch{}_{}Noise.csv'.format(args.mod,int(noise*100)),index= None, header=True)
df_pred_dt.to_csv(r'data/pred{}_dt{}Noise.csv'.format(args.mod,int(noise*100)),index = None, header=True)  #Change based on noise, # of samples, MVs
df_pred_y.to_csv(r'data/pred{}_y{}Noise.csv'.format(args.mod,int(noise*100)),index = None, header=True) 
# sys.exit()

# =============================================================================
# # 2nd stage of 2-stage approach
# =============================================================================

class MechMod(nn.Module):
    def __init__(self):
        super(MechMod, self).__init__()
        self.k = torch.tensor([1.0,1.0,1.0],requires_grad=True)


    # Black-box Model
    def forward(self, N_t, x, z=0, t=0):
        # x_std = MMS.inverse_transform(x)
        Ca = x[:,0:1];   Cb = x[:,1:2];   Cc = x[:,2:3]
        dCadt = -self.k[0]*Ca + self.k[1]*Cb
        dCbdt = self.k[0]*Ca - self.k[1]*Cb - self.k[2]*2*Cb
        dCcdt = self.k[2]*Cb
        
        dXdt = torch.cat([dCadt,dCbdt,dCcdt],dim=1)
        return dXdt

# Assign mechanistic model
HM = MechMod()
args.N_steps = 20 # step size of Euler integrator
args.viz = None   # No plotting
pred_y = torch.tensor(pred_y)  # re-convert to a tensor
pred_dy = torch.tensor(pred_dy) # re-convert to a tensor
# Initial predictions of the mechanistic model
m_pred_y = odeint(HM, batch_y0, t, N_steps=args.N_steps) #scaled inputs
m_pred_dy = HM(t,pred_y)
m_params = list([HM.k])

def closure():
    optimizer.zero_grad()
    m_pred_dy = HM(t,pred_y)

    los = torch.mean((pred_dy - m_pred_dy)**2)

    los.backward()
    return los

if __name__ == '__main__':
    ii = 0
    if args.viz != None: plt.figure()

    # print('training with mod:{}, t:{}, N_dat:{}'.format(args.mod,len(t),N_dat))
    history = Store()

    # optimizer = torch.optim.SGD(m_params,lr=learning_rate)
    # optimizer = torch.optim.LBFGS(m_params, lr=learning_rate,history_size=10, max_iter=4) #AG default
    optimizer = torch.optim.LBFGS(m_params, ) #AG default
    # optimizer = torch.optim.RMSprop(m_params, lr=1e-2, weight_decay=0.5)
    end = time.time()
    tot_time = time.time()

    for itr in range(0, args.niters + 1):
#        break
        optimizer.zero_grad()
        # personal odeint
        m_pred_dy = HM(t,pred_y)                

        MSE = torch.mean((pred_dy - m_pred_dy)**2)
        # losses = compute_losses(HM,pred_dy[:,:,:],batch_y[:,:,:],true_y[:,:,:],lamb)
        # losses[0] = losses[0]#*100
        # losses[0].backward()
        MSE.backward()
        if np.isnan(MSE.detach().numpy()) > 0:
            print("loss nans on the loose."); break

        optimizer.step(closure)
        if MSE < 1e-6 and itr > 10:
            break
        # optimizer.step()
        with torch.no_grad():
            
            l = torch.mean((pred_dy - m_pred_dy)**2)
            if itr % args.test_freq == 0:
                if args.viz != None:
                    # personal odeint
                    m_pred_dy = odeint(HM, batch_y0[0:1,:], t,) #scaled inputs
                    plt.plot(t,m_pred_y[:,0],'g',t,m_pred_y[:,2],'g')
                print('Iter {:04d} | Sqrd Loss {:.6f} | Tot Loss {:6f}'.format(
                        itr, l.item(), MSE.item()))

            ii += 1
        end = time.time() #duration of fitting
    tot_time = time.time() - tot_time
print('Sim time: ',tot_time)

# Print values of mechanistic parameters
print('True Parameter values:','\n',
      'k1 = ',k[0].detach().numpy(),'\n',
      'k2 = ',k[1].detach().numpy(),'\n',
      'k3 = ',k[2].detach().numpy(),)
print('Predicted param values:','\n',
      'k1 = ',HM.k.detach().numpy()[0],'\n',
      'k2 = ',HM.k.detach().numpy()[1],'\n',
      'k3 = ',HM.k.detach().numpy()[2],)


# =============================================================================
# #Checking Extrapolation
# =============================================================================
# y graphs
dum = torch.tensor([[10.0,5.0,0.0]])
#dum = torch.tensor([[4.0,1.0,0.25,0.5]])
#dum = torch.tensor([[1.0,0.0,2.0,1.5]])
true_yex = odeint(Lambda(), dum, t, method='dopri5')[0].detach().numpy()

plt.figure()
plt.subplot(121)
pred_y = odeint(funct, dum, t)[0].detach().numpy() #replacing true_y0 true_y[0][50:51,:]
plt.plot(t,pred_y[:,0],'b',t,pred_y[:,1],'r')
plt.plot(t,pred_y[:,2],'c')#)t,pred_y[:,3],'g')
plt.title('Fitted Neural ODE')
plt.xlabel('t')
plt.ylabel('Concentration')
plt.plot(t[idx],true_yex[idx,0],'ko',t[idx],true_yex[idx,1],'ko')
plt.plot(t[idx],true_yex[idx,2],'ko')#,t[idx],true_yex[idx,3],'go')

plt.subplot(122)
pred_y = odeint(HM, dum, t)[0].detach().numpy() #replacing true_y0 true_y[0][50:51,:]
plt.plot(t,pred_y[:,0],'b',t,pred_y[:,1],'r')
plt.plot(t,pred_y[:,2],'c')#)t,pred_y[:,3],'g')
plt.title('Fitted mechanistic ODE')
plt.xlabel('t')
# plt.ylabel('Concentration')
plt.plot(t[idx],true_yex[idx,0],'ko',t[idx],true_yex[idx,1],'ko')
plt.plot(t[idx],true_yex[idx,2],'ko')#,t[idx],true_yex[idx,3],'go')
plt.legend(['Ca','Cb','Cc','True data'])
plt.show()
plt.savefig('visuals/Extrap',dpi=300)

sys.exit()
