# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:27:53 2020

@author: Afbwi
"""
#Needed fix: Check if t is a torch var.  Make it a torch var if not
#Bugs: Euler step t1 is added to solution only if in true_t, but converting
#       tensor object into numpy object sometimes does not match original
#       numpy array; this may be a machine precision error with PyTorch
#Functions for integrating and tracking the history of ODE optimization
    #Using Autograd via pytorch

#Used by EB_PFR and RevRxn+Deg codes
import torch
import numpy as np
import torch.nn as nn

def odeint(func, y0, t, z=[], rtol=1e-7, atol=1e-9, N_steps=6, method=None, options=None, params=None):    
    solver = Eul(func, y0, z, rtol, atol, N_steps, params)
    solution = solver.integrate(t)
#    if tensor_input:
#        solution = solution[0]
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
        # return self.rk4_step_func(func, N_t, dt, y, z, t)
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
        
#    def l_update(self,lo,loss,lody,tot_loss):
    def l_update(self,losses):
        with torch.no_grad():
#        [reglos,los_raw,losdy_raw,tot_loss,los,losdy]
            self.tot_loss = torch.cat([self.tot_loss,losses[0].unsqueeze(dim=0)])
            self.reg_loss = torch.cat([self.reg_loss,losses[1].unsqueeze(dim=0)])
            self.y_rawloss = torch.cat([self.y_rawloss,losses[2].unsqueeze(dim=0)])
            self.dy_rawloss = torch.cat([self.dy_rawloss,losses[3].unsqueeze(dim=0)])
            self.y_loss = torch.cat([self.y_loss,losses[4].unsqueeze(dim=0)])
            self.dy_loss = torch.cat([self.dy_loss,losses[5].unsqueeze(dim=0)])

#        self.y_rawloss = torch.cat([self.y_rawloss,lo.unsqueeze(dim=0)])
#        self.reg_loss = torch.cat([self.reg_loss,loss.unsqueeze(dim=0)])
#        self.totdy_loss = torch.cat([self.totdy_loss,lody.unsqueeze(dim=0)])
#        self.total_loss = torch.cat([self.total_loss,tot_loss.unsqueeze(dim=0)])
    def reg_add(self,epoch):
        self.increase_reg.append(epoch)
    def tot_add(self,epoch):
        self.increase_tot.append(epoch)
        
    def update(self,p):
#        self.params.append(list(p.detach().numpy()))
#        self.grads.append(list(p.grad.detach().numpy()))
#        for p in params:
#            self.param
        self.w1.append(p[0].detach().numpy()*1); self.w2.append(p[1].detach().numpy()*1); self.w3.append(p[2].detach().numpy()*1);
        self.w4.append(p[3].detach().numpy()*1); self.k.append(p[4].detach().numpy()*1);
        self.w1g.append(p[0].grad.detach().numpy()*1); self.w2g.append(p[1].grad.detach().numpy()*1); self.w3g.append(p[2].grad.detach().numpy()*1);
        self.w4g.append(p[3].grad.detach().numpy()*1); self.kg.append(p[4].grad.detach().numpy()*1)
# =============================================================================
# # Training functions
# =============================================================================
def Check_point(history,losses): #Check for stopping criteria while updating history
    tot_loss,reglos,los_raw,losdy_raw,los,losdy = losses
    stop = False
    # Stop training if 10-fold increase in reg loss
    if len(history.reg_loss) > 10:
        if reglos/100 > history.reg_loss[-10]:
            print('WARNING: reg loss is increasing X100 --> Divergence suspected'); 
            stop = True #break
    # Track iterations where reglos increases
#    if reglos > history.reg_loss[-1] and los_raw > history.y_rawloss[-1]:
#        history.reg_add(itr)
#        if len(history.increase_reg) > 5:
#            print('reg increases > 5 times w/o decreasing raw loss'); #break
    # Track iterations where loss of data increases
#    if los_raw > history.y_rawloss[-1]:
#        history.tot_add(itr)
    if len(history.tot_loss) > 12:
        stag_iters = sum(history.tot_loss[-12:-2] - history.tot_loss[-11:-1] <1*10**-6)
        if stag_iters == 10:
            # print('Tot Loss {:6f}'.format(tot_loss.item()))
            print('Total loss did not improve more than 0.000001 for 10 consecutive epochs')
            stop = True #break
    return stop

def compute_losses(model,pred_y,batch_y,t_idx,lamb):
    # All losses are 2-dimensional and indexed
    # Regulaization loss
    reglos = 0
    # Single NN
    for param in model.parameters():
        reglos += torch.sum(param**2)
    # 1 NN for each output
#    for N_output in range(0,model.N_outputs):
#        for param in model.NNs[N_output].parameters():
#            reglos += torch.sum(param**2)#l1_crit(param)

    # Loss in loss function
    # diff = pred_y[:] - batch_y[:]
    # tot_loss = torch.mean((diff.T*t_idx)**2) + lamb*reglos    
    tot_loss = torch.mean((pred_y[:] - batch_y[:])**2) + lamb*reglos #+ \
            
#                torch.sum(nn.ReLU()(0-pred_y.view(-1,1))) #This term only appropriate if all batch_y > 0
#                    torch.mean((torch_y2d - pred_y2d)**2) #+ \
#                torch.mean((batch_y0 - batch_y[:,0,:])**2)
    # tot_loss = tot_loss*100    
# If daMKM states are missing
#    tot_loss = torch.mean((pred_y[:,0:3,:] - batch_y[:,0:3,:])**2) + lamb*reglos
    # Absolute value loss
#    tot_loss = torch.mean(abs(pred_y - batch_y)) + lamb*reglos
    with torch.no_grad():
        # Loss between prediction and noisy batch data
        los_raw = torch.mean((pred_y - batch_y)**2)
        # Loss between Derivative prediction and noisy batch data
#        pred_dy = model(t, batch_y).detach().numpy()*(max_y - min_y)
#        pred_dy = torch.tensor(pred_dy,dtype=torch.float32)
#NA        losdy_raw = torch.mean((pred_dy - batch_dy)**2) #this doesn't mean much
        losdy_raw = torch.tensor(1,dtype=torch.float32)
        
        # Loss between Derivative pred and true deriatives
#        pred_dy = model(t, pred_y).detach().numpy()*(max_y - min_y) #forward func
#        pred_dy = torch.tensor(pred_dy,dtype=torch.float32)
#NA        losdy = torch.mean((pred_dy[idx,:] - true_dy)**2)
        losdy = torch.tensor(1,dtype=torch.float32)
    # Loss between pred and true data
    with torch.no_grad():
#        true_y = torch.tensor(MMS.transform(true_y),dtype=torch.float32)
#        los = torch.mean((pred_y - true_y)**2)
        los = torch.tensor(1,dtype=torch.float32)
    losses = [tot_loss,reglos,los_raw,losdy_raw,los,losdy]
    return losses
#    stop = Check_point(history,losses)
#    history.l_update(history,losses)
#    return [tot_loss, stop]

class MMScaler(): #Similar to MinMaxScaler but for Autograd
    def __init__(self,args,max_y=[],min_y=[]):
        self.dum = 1;           self.args = args
        self.max_y = max_y;     self.min_y = min_y;
    def fit(self,data):
        if data.ndim == 2:
            self.max_y = torch.tensor(np.amax(data.numpy(),axis=0),dtype=torch.float32)
            self.min_y = torch.tensor(np.amin(data.numpy(),axis=0),dtype=torch.float32)
        if data.ndim == 3:
            self.max_y = torch.tensor(np.amax(np.amax(data.numpy(),axis=1),axis=0),dtype=torch.float32)
            self.min_y = torch.tensor(np.amin(np.amin(data.numpy(),axis=1),axis=0),dtype=torch.float32)
#        self.max_y = torch.tensor(np.amax(np.amax(data.numpy(),axis=1),axis=0))
#        self.min_y = torch.tensor(np.amin(np.amin(data.numpy(),axis=1),axis=0))
    def transform(self,data):
        if self.args.mod == 'LoVo' or self.args.mod == 'Pen' or self.args.mod == 'Sty':
            new_data = data/(self.max_y - self.min_y)
        else:
            new_data = (data - 0)/(self.max_y - self.min_y)
        return new_data
    def fit_transform(self,data):
        self.fit(data)
        return self.transform(data)
    def inverse_transform(self,data):
        if self.args.mod == 'LoVo' or self.args.mod == 'Pen' or self.args.mod == 'Sty':
            new_data = data*(self.max_y - self.min_y)
        else:
            new_data = data*(self.max_y - self.min_y)# + self.min_y
        return new_data

class MScaler(): #Similar to MinMaxScaler, but can convert torch 3D arrays to numpy 
    def __init__(self,args,max_y=[],min_y=[]):
        self.dum = 1;           self.args = args
        self.max_y = max_y;     self.min_y = min_y;
    def fit(self,data):
#        self.max_y = torch.tensor(np.amax(np.amax(data.numpy(),axis=1),axis=0))
#        self.min_y = torch.tensor(np.amin(np.amin(data.numpy(),axis=1),axis=0))
        if data.ndim == 2:
            self.max_y = np.amax(data.numpy(),axis=0)
            self.min_y = np.amin(data.numpy(),axis=0)
        if data.ndim == 3:
            self.max_y = np.amax(np.amax(data.numpy(),axis=1),axis=0)
            self.min_y = np.amin(np.amin(data.numpy(),axis=1),axis=0)
    def transform(self,data):
#        new_data = (data)/(self.max_y)
        if self.args.mod == 'Pen':
            new_data = data/(self.max_y - self.min_y)
        else:
            new_data = (data - self.min_y)/(self.max_y - self.min_y)
#        new_data = (data - self.min_y + (self.max_y - self.min_y)*0.1)/ \
#                                        (self.max_y - self.min_y)
        return new_data
    def fit_transform(self,data):
        self.fit(data)
        return self.transform(data)
    def inverse_transform(self,data):
#        new_data = data*(self.max_y)
        if self.args.mod == 'Pen':
            new_data = data*(self.max_y - self.min_y)
        else:
            new_data = data*(self.max_y - self.min_y) + self.min_y
#        new_data = data*(self.max_y - self.min_y) + self.min_y - (self.max_y - self.min_y)*0.1
        return new_data
 


# =============================================================================
# # AG version of ODEs Ethylbenzene, Penicillin, LoVo
# =============================================================================
#Multiple run, no param specs
def odefunEB(t,y,z=0,ind=0,iso=0):
    #a = Ethylbenzene, b = styrene, c = hydrogen, d = Benzene,
    #e = Ethylene, f = Toluene, g = Meth
    b1=-17.34; b2=-1.302e4; b3=5.051;
    b4=-2.314e-10; b5=1.302e-6; b6=-4.931e-3;
    p=2137; o=0.4;  #Density of pellet:kg/m^3   #Void fraction
    Cpa=299; Cpb=273; Cpc=30; Cpd=201;  #Heat Capacity: J/(mol*K)
    Cpe=90; Cpf=249; Cpg=68; Cpsteam=40;
    P0=2.4; Fsteam=48; Ft=Fsteam+torch.sum(y[:,1:],dim=1); #Inert steam pressure: atm, #Flow rate: mol/s

    # y(0)= Temp y(1)=Fa y(2)=Fb y(3)=Fc
    # y(4)=Fd=Fe y(5)=Ff=Fg
    # dydx(1)=dT/dV dydx(2)=ra=-r1st-r2-r3 
    # dydx(3)=rb=r1st dydx(4)=rc=r1st-r3
    # dydx(5)=rd=re=r2 dydx(6)=rf=rg=r3
    
    Peb=y[:,1]/Ft*P0; Pst=y[:,2]/Ft*P0;  #atm
    Ph2=y[:,3]/Ft*P0;
    Kp1=torch.exp(b1+b2/y[:,0]+b3*torch.log(y[:,0])+((b4*y[:,0]+b5)*y[:,0]+b6)*y[:,0]); #atm
    r1st=p*(1-o)*torch.exp(-0.08539-10925/y[:,0])*(Peb-Pst*Ph2/Kp1);
    r2=p*(1-o)*torch.exp(13.2392-25000/y[:,0])*(Peb);
    r3=p*(1-o)*torch.exp(0.2961-11000/y[:,0])*(Peb*Ph2);
    dhrx1=118000+(299-273-30)*(y[:,0]-273);       #kJ/kmol ethylbenzene
    dhrx2=105200+(299-201-90)*(y[:,0]-273);
    dhrx3=-53900+(299+30-249-68)*(y[:,0]-273);    #converting to degC?
    dydx1=1000*(-r1st*dhrx1-r2*dhrx2-r3*dhrx3)/((y[:,1]*Cpa+y[:,2]*Cpb+y[:,3]*Cpc+y[:,4]*Cpd+y[:,4]*Cpe+y[:,5]*Cpf+y[:,5]*Cpg+Fsteam*Cpsteam));
    dydx2=(-r1st-r2-r3)*1000;
    dydx3=r1st*1000;
    dydx4=(r1st-r3)*1000;
    dydx5=r2*1000;
    dydx6=r3*1000;
    if iso==1:
        dydx1[:]=0
    dydx=torch.stack((dydx1,dydx2,dydx3,dydx4,dydx5,dydx6),dim=1)
    return dydx

#Multiple run, spec params, MechMod
def odefunEB2(t,y,params, iso=0):
    #a = Ethylbenzene, b = styrene, c = hydrogen, d = Benzene,
    #e = Ethylene, f = Toluene, g = Meth
    b1=-17.34; b2=-1.302e4; b3=5.051;
    b4=-2.314e-10; b5=1.302e-6; b6=-4.931e-3;
    p=2137; o=0.4;  #Density of pellet:kg/m^3   #Void fraction
    Cpa=299; Cpb=273; Cpc=30; Cpd=201;  #Heat Capacity: J/(mol*K)
    Cpe=90; Cpf=249; Cpg=68; Cpsteam=40;
    P0=2.4; Fsteam=48; Ft=Fsteam+torch.sum(y[:,1:],dim=1); #Inert steam pressure: atm, #Flow rate: mol/s

    # y(0)= Temp y(1)=Fa y(2)=Fb y(3)=Fc
    # y(4)=Fd=Fe y(5)=Ff=Fg
    # dydx(1)=dT/dV dydx(2)=ra=-r1st-r2-r3 
    # dydx(3)=rb=r1st dydx(4)=rc=r1st-r3
    # dydx(5)=rd=re=r2 dydx(6)=rf=rg=r3
    if len(params) == 3:
        par = torch.tensor([10925.0,25000.0,11000.0])
        params = torch.cat((params,par))
    
    Peb=y[:,1]/Ft*P0; Pst=y[:,2]/Ft*P0;  #atm
    Ph2=y[:,3]/Ft*P0;
    Kp1=torch.exp(b1+b2/y[:,0]+b3*torch.log(y[:,0])+((b4*y[:,0]+b5)*y[:,0]+b6)*y[:,0]); #atm
    r1st=1000*p*(1-o)*torch.exp(params[0]-params[3]/y[:,0])*(Peb-Pst*Ph2/Kp1);
    r2=1000*p*(1-o)*torch.exp(params[1]-params[4]/y[:,0])*(Peb);
    r3=1000*p*(1-o)*torch.exp(params[2]-params[5]/y[:,0])*(Peb*Ph2);
    dhrx1=118000+(299-273-30)*(y[:,0]-273);       #kJ/kmol ethylbenzene
    dhrx2=105200+(299-201-90)*(y[:,0]-273);
    dhrx3=-53900+(299+30-249-68)*(y[:,0]-273);    #converting to degC?
    dydx1=(-r1st*dhrx1-r2*dhrx2-r3*dhrx3)/((y[:,1]*Cpa+y[:,2]*Cpb+y[:,3]*Cpc+y[:,4]*Cpd+y[:,4]*Cpe+y[:,5]*Cpf+y[:,5]*Cpg+Fsteam*Cpsteam));
    dydx2=(-r1st-r2-r3);
    dydx3=r1st;
    dydx4=(r1st-r3);
    dydx5=r2;
    dydx6=r3;
    if iso==1:
        dydx1[:]=0
    dydx=torch.stack((dydx1,dydx2,dydx3,dydx4,dydx5,dydx6),dim=1)
    return dydx

# For HM rate estim, rates not multiplied by 1000, HRxnMod
def odefunEB3(t,y,r, iso=0):
    #a = Ethylbenzene, b = styrene, c = hydrogen, d = Benzene,
    #e = Ethylene, f = Toluene, g = Meth
    b1=-17.34; b2=-1.302e4; b3=5.051;
    b4=-2.314e-10; b5=1.302e-6; b6=-4.931e-3;
    p=2137; o=0.4;  #Density of pellet:kg/m^3   #Void fraction
    Cpa=299; Cpb=273; Cpc=30; Cpd=201;  #Heat Capacity: J/(mol*K)
    Cpe=90; Cpf=249; Cpg=68; Cpsteam=40;
    P0=2.4; Fsteam=48; Ft=52; #Inert steam pressure: atm, #Flow rate: mol/s

    # y(0)= Temp y(1)=Fa y(2)=Fb y(3)=Fc
    # y(4)=Fd=Fe y(5)=Ff=Fg
    # dydx(1)=dT/dV dydx(2)=ra=-r1st-r2-r3 
    # dydx(3)=rb=r1st dydx(4)=rc=r1st-r3
    # dydx(5)=rd=re=r2 dydx(6)=rf=rg=r3
    
    Peb=y[:,1]/Ft*P0; Pst=y[:,2]/Ft*P0;  #atm
    Ph2=y[:,3]/Ft*P0;
    Kp1=torch.exp(b1+b2/y[:,0]+b3*torch.log(y[:,0])+((b4*y[:,0]+b5)*y[:,0]+b6)*y[:,0]); #atm
#    r1st=p*(1-o)*torch.exp(params[0]-params[1]/y[:,0])*(Peb-Pst*Ph2/Kp1);
#    r2=p*(1-o)*torch.exp(params[2]-params[3]/y[:,0])*(Peb);
#    r3=p*(1-o)*torch.exp(params[4]-params[5]/y[:,0])*(Peb*Ph2);
    r1st = r[:,0]*(Peb-Pst*Ph2/Kp1);
    r2 = r[:,1]*(Peb);
    r3 = r[:,2]*(Peb*Ph2);
    dhrx1=118000+(299-273-30)*(y[:,0]-273);       #kJ/kmol ethylbenzene
    dhrx2=105200+(299-201-90)*(y[:,0]-273);
    dhrx3=-53900+(299+30-249-68)*(y[:,0]-273);    #converting to degC?
    dydx1=(-r1st*dhrx1-r2*dhrx2-r3*dhrx3)/((y[:,1]*Cpa+y[:,2]*Cpb+y[:,3]*Cpc+y[:,4]*Cpd+y[:,4]*Cpe+y[:,5]*Cpf+y[:,5]*Cpg+Fsteam*Cpsteam));
    dydx2=(-r1st-r2-r3);
    dydx3=r1st;
    dydx4=(r1st-r3);
    dydx5=r2;
    dydx6=r3;
    if iso==1:
        dydx1[:]=0
    dydx=torch.stack((dydx1,dydx2,dydx3,dydx4,dydx5,dydx6),dim=1)
    return dydx

# For HM rate estim, will this work?
def odefunEB4(t,y,r, rates=['mech','mech','mech'],iso=0):
    #a = Ethylbenzene, b = styrene, c = hydrogen, d = Benzene,
    #e = Ethylene, f = Toluene, g = Meth
    b1=-17.34; b2=-1.302e4; b3=5.051;
    b4=-2.314e-10; b5=1.302e-6; b6=-4.931e-3;
    p=2137; o=0.4;  #Density of pellet:kg/m^3   #Void fraction
    Cpa=299; Cpb=273; Cpc=30; Cpd=201;  #Heat Capacity: J/(mol*K)
    Cpe=90; Cpf=249; Cpg=68; Cpsteam=40;
    P0=2.4; Fsteam=48; Ft=52; #Inert steam pressure: atm, #Flow rate: mol/s

    # y(0)= Temp y(1)=Fa y(2)=Fb y(3)=Fc
    # y(4)=Fd=Fe y(5)=Ff=Fg
    # dydx(1)=dT/dV dydx(2)=ra=-r1st-r2-r3 
    # dydx(3)=rb=r1st dydx(4)=rc=r1st-r3
    # dydx(5)=rd=re=r2 dydx(6)=rf=rg=r3
    
    Peb=y[:,1]/Ft*P0; Pst=y[:,2]/Ft*P0;  #atm
    Ph2=y[:,3]/Ft*P0;
    Kp1=torch.exp(b1+b2/y[:,0]+b3*torch.log(y[:,0])+((b4*y[:,0]+b5)*y[:,0]+b6)*y[:,0]); #atm
    if rates[0] == 'mech': #EB only
        r1st = 1000*p*(1-o)*torch.exp(-0.08539-10925/y[:,0])*(Peb-Pst*Ph2/Kp1);
    elif rates[0] == 'NN': # EB
        r1st = r[:,0]
    if rates[1] == 'mech': #Ben
        r2 = 1000*p*(1-o)*torch.exp(13.2392-25000/y[:,0])*(Peb);
    elif rates[1] == 'NN':
        r2 = r[:,1]
    if rates[2] =='mech':
        r3 = 1000*p*(1-o)*torch.exp(0.2961-11000/y[:,0])*(Peb*Ph2);
    elif rates[2] == 'NN':
        r3 = r[:,2]
#    r1st=p*(1-o)*torch.exp(params[0]-params[1]/y[:,0])*(Peb-Pst*Ph2/Kp1);
#    r2=p*(1-o)*torch.exp(params[2]-params[3]/y[:,0])*(Peb);
#    r3=p*(1-o)*torch.exp(params[4]-params[5]/y[:,0])*(Peb*Ph2);
    # r1st = r[:,0]*(Peb-Pst*Ph2/Kp1);
    # r2 = r[:,1]*(Peb);
    # r3 = r[:,2]*(Peb*Ph2);
    dhrx1=118000+(299-273-30)*(y[:,0]-273);       #kJ/kmol ethylbenzene
    dhrx2=105200+(299-201-90)*(y[:,0]-273);
    dhrx3=-53900+(299+30-249-68)*(y[:,0]-273);    #converting to degC?
    dydx1=(-r1st*dhrx1-r2*dhrx2-r3*dhrx3)/((y[:,1]*Cpa+y[:,2]*Cpb+y[:,3]*Cpc+y[:,4]*Cpd+y[:,4]*Cpe+y[:,5]*Cpf+y[:,5]*Cpg+Fsteam*Cpsteam));
    dydx2=(-r1st-r2-r3);
    dydx3=r1st;
    dydx4=(r1st-r3);
    dydx5=r2;
    dydx6=r3;
    if iso==1:
        dydx1[:]=0
    dydx=torch.stack((dydx1,dydx2,dydx3,dydx4,dydx5,dydx6),dim=1)
    return dydx


def odefunPen(N_t, y, z, t, params=None, extra=0):
#def FBPen_true(z,N_t,Sf,F,t,params):
#    ind = int(N_t*(len(t)-1)/(max(t))) #Set index for Sf = N_t*66/(198)
#    if ind > len(t) - 1:        #Prevent inputs from exceeding time index (66)
#        ind = len(t) - 1
##        pdb.set_trace()
#    #Set inputs
#    if ind == len(t):
#        pdb.set_trace()
    B = y[:,0]
    S = y[:,1]
    P = y[:,2]
    V = y[:,3]
#    Sf = z[:,ind,0];    F=z[:,ind,1]
    Sf = z[:,0];    F=z[:,1]
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
    cL = cLmax*B*torch.exp(-S/100.0)/(kL+B+1)
    qp = 1.5*qpm*S*B/(4*kp + S*B*(1 + S/(3*ki)))
    m_x = m_xm*B/(B + 10.0)                     #
    sigL = mu/Yxs + qp/Yps + m_x                #

    #differential equations    
    dBdt = B*(mu-D-cL) 
    dSdt = -sigL*B + (Sf - S)*D
    dPdt = qp*B - P*(D + k)
    dVdt = F
    dydx=torch.stack((dBdt,dSdt,dPdt,dVdt),dim=1)
    return dydx

#Used by EthybenPFRID.py and torchLoVoID.py
def odefunLoVo(N_t, states, z, t, params=None, extra=0):
    if params is not None:
        p1, p2, p3 = params
    else:
        p1, p2, p3 = [1.5, 2.0, 3.4]
    x = states[:,0]
    y = states[:,1]
    dxdt = p1*x - p2*x*y
    dydt = -p3*y + x*y
    dydx=torch.stack((dxdt,dydt),dim=1)
    return dydx

def odefunLoVo2(N_t, y, z, t, params=None, extra=0):
    if params is not None:
        p1, p2, p3 = params
    else:
        p1, p2, p3 = [1.5, 2.0, 3.4]
    x = y[:,0]
    y = y[:,1]
    dxdt = p1*x - p2*x*y
    dydt = -p3*y + x*y
    dydx=torch.stack((dxdt,dydt),dim=1)
    return dydx
# Not sure if this is used?
def odefunLoVo2(N_t, y, z, t, params=None, extra=0):
    if params is not None:
        p1, p2, p3 = params
    else:
        p1, p2, p3 = [1.5, 2.0, 3.4]
    x = y[:,0]
    y = y[:,1]
    dxdt = (p1*x*max_x - p2*x*y*(max_x*max_y))/max_x
    dydt = (-p3*y*max_y + x*y(max_x*max_y))/max_y
    dydx=torch.stack((dxdt,dydt),dim=1)
    return dydx

def odefunLoVolt(N_t, states, z, t, params=None, extra=0):
    if params is not None:
        p1, p2, p3, p4 = params
    else:
        p1, p2, p3, p4 = [3, 0.6, 0.5, 4.0]
    x = states[:,0]
    y = states[:,1]    
    dxdt = p1*x - p2*x*y
    dydt = p3*x*y - p4*y 
    dydx=torch.stack((dxdt,dydt),dim=1)
    return dydx

#Used by EthybenPFRID.py
def odeRevRxn(N_t, x, z, t, par=None):
    if par is not None:
        k1, k2 = par
    else:
        k1, k2 = [1.2, 0.6]            
    Ca = x[:,0];  Cb = x[:,1];  Cc = x[:,2];
    dCadt = -k1*Ca 
    dCbdt = k1*Ca - k2*Cb
    dCcdt = k2*Cb
    return [dCadt,dCbdt,dCcdt]

#Used by EthylbenPFRID.py
def odeLogNorm(N_t, x, z, t, par=None):
    if par is not None:
        Do, sigLN = par
    else:
        Do, sigLN = [9.73254364, 0.04058519]
    # lognormal distribution
    P = 1/(np.sqrt(2*np.pi)*D0*sigLN)*np.exp(-np.log(D0/Do)**2/(2*sigLN**2))
    
# Used by EthylbenPFRID.py
def odedaMKM(N_t, y, z, t, par=None):
    if par is not None:
        kd1for,kd1back,kd2for,kd2back,kd3for,kd3back,ka1for,ka1back = par
    else:
        kd1for,kd1back,kd2for,kd2back,kd3for,kd3back,ka1for,ka1back = [10,4,40,60,200,40,100,80]
    
    A = y[:,0];  B = y[:,1];  C = y[:,2]; 
    As = y[:,3]; Bs = y[:,4]; Cs = y[:,5]; st = y[:,6]
    
    dAdt = -kd1for*A*st + kd1back*As
    dAsdt = kd1for*A*st - kd1back*As - ka1for*As*Bs + ka1back*Cs*st
    
    dBdt = -kd2for*B*st + kd2back*Bs
    dBsdt = kd2for*B*st - kd2back*Bs - ka1for*As*Bs + ka1back*Cs*st
    
    dCdt = -kd3for*C*st + kd3back*Cs
    dCsdt = kd3for*C*st - kd3back*Cs + ka1for*As*Bs - ka1back*Cs*st
    
    dstdt = -kd1for*A*st - kd2for*B*st - kd3for*C*st +\
            kd1back*As + kd2back*Bs + kd3back*Cs + \
            ka1for*As*Bs - ka1back*Cs*st
    dydx=torch.stack((dAdt,dBdt,dCdt,dAsdt,dBsdt,dCsdt,dstdt),dim=1)

    return dydx

def odedcMKM(N_t, y, z, t, par=None):
    
    
    dAdt = -kdafor*A*st + kdaback*As
    dBdt = -kd2for*B*st + kd2back*Bs
    dCdt
    
    dAsdt = kd1for*A*st - kd1back*As - Kc2for*As*Ds + kc2back*Cs*st
    dBsdt = kd2for*B*st - kd2back*Bs - kc1for*Bs*st + kc1back*2*Ds
    dCsdt
    dDsdt = kc1for*Bs*st - kc1back*2*Ds - kc2for*As*Ds + kc2back*Cs*st
    
    dstdt
    
# Multiple runs
def odedcMKM2(N_t, y, z, t, par=None):
    A = y[:,0];  B = y[:,1];  C = y[:,2];
    As = y[:,3]; Bs = y[:,4]; Cs = y[:,5]; st = y[:,6]
    Ds = y[:,7]
    if par is None:
        kd1_for = 20;     kd1_back = 8
        kd2_for = 16;     kd2_back = 4
        kd3_for = 12;     kd3_back = 8
        kc1_for = 1200;   kc1_back = 400
        kc2_for = 2000;   kc2_back = 1600
    else:
        kd1_for = par[0];     kd1_back = par[1]
        kd2_for = par[2];     kd2_back = par[3]
        kd3_for = par[4];     kd3_back = par[5]
        kc1_for = par[6];     kc1_back = par[7]
        kc2_for = par[8];     kc2_back = par[9]
    
    dAdt = -kd1_for*A*st + kd1_back*As
    dAsdt = kd1_for*A*st - kd1_back*As - kc2_for*As*Ds + kc2_back*Cs*st
    
    dBdt = -kd2_for*B*st + kd2_back*Bs
    dBsdt = kd2_for*B*st - kd2_back*Bs - kc1_for*Bs*st + kc1_back*Ds**2 
    
    dCdt = -kd3_for*C*st + kd3_back*Cs
    dCsdt = kd3_for*C*st - kd3_back*Cs + kc2_for*As*Ds - kc2_back*Cs*st
    
    dDsdt = 2*kc1_for*Bs*st - 2*kc1_back*Ds**2 - kc2_for*As*Ds + kc2_back*Cs*st
    
    dstdt = -kd1_for*A*st - kd2_for*B*st - kd3_for*C*st +\
            kd1_back*As + kd2_back*Bs + kd3_back*Cs - \
            kc1_for*Bs*st + kc1_back*Ds**2 + kc2_for*As*Ds - kc2_back*Cs*st
    dydx=torch.stack((dAdt,dBdt,dCdt,dAsdt,dBsdt,dCsdt,dstdt,dDsdt),dim=1)

    return dydx
# Multiple runs

def isodaMKM2(N_t, y, z, t, par=None):
    A = y[:,0];  B = y[:,1];  C = y[:,2];
    As = y[:,3]; Bs = y[:,4]; Cs = y[:,5]; st = y[:,6]
    T = y[:,7] + 273.15
    R = 8.314
    # if par is None:
    #     kd1_for = 10;     kd1_back = 4
    #     kd2_for = 40;     kd2_back = 60
    #     kd3_for = 200;     kd3_back = 40
    #     ka1_for = 100;   ka1_back = 800
    #     Ed1_for = 4;    Ed1_back
    # else:
    #     kd1_for = par[0];     kd1_back = par[1]
    #     kd2_for = par[2];     kd2_back = par[3]
    #     kd3_for = par[4];     kd3_back = par[5]
    #     kc1_for = par[6];     kc1_back = par[7]
    #     kc2_for = par[8];     kc2_back = par[9]
    if par is None:    
        kd1_for = 10*torch.exp(-4/(R*T));      kd1_back = 4 *torch.exp(-50/(R*T))
        kd2_for = 40*torch.exp(-22/(R*T));     kd2_back = 60*torch.exp(-35/(R*T))
        kd3_for = 200*torch.exp(-15/(R*T));    kd3_back = 40*torch.exp(-18/(R*T))
        ka1_for = 100*torch.exp(-40/(R*T));    ka1_back = 80*torch.exp(-30/(R*T))
    # else:
    #     kd1_for = par[0]*torch.exp(-par[8]/(R*T));  kd1_back = par[1]*torch.exp(-par[9]/(R*T))
    #     kd2_for = par[2]*torch.exp(-par[10]/(R*T)); kd2_back = par[3]*torch.exp(-par[11]/(R*T))
    #     kd3_for = torch.exp(par[4])*torch.exp(-par[12]/(R*T)); kd3_back = par[5]*torch.exp(-par[13]/(R*T))
    #     ka1_for = par[6]*torch.exp(-par[14]/(R*T)); ka1_back = par[7]*torch.exp(-par[15]/(R*T))        
    else:
        kd1_for = torch.exp(par[0])*torch.exp(-par[8]/(R*T));  kd1_back = torch.exp(par[1])*torch.exp(-par[9]/(R*T))
        kd2_for = torch.exp(par[2])*torch.exp(-par[10]/(R*T)); kd2_back = torch.exp(par[3])*torch.exp(-par[11]/(R*T))
        kd3_for = torch.exp(par[4])*torch.exp(-par[12]/(R*T)); kd3_back = torch.exp(par[5])*torch.exp(-par[13]/(R*T))
        ka1_for = torch.exp(par[6])*torch.exp(-par[14]/(R*T)); ka1_back = torch.exp(par[7])*torch.exp(-par[15]/(R*T))        

    
    
    dAdt = -kd1_for*A*st + kd1_back*As
    dAsdt = kd1_for*A*st - kd1_back*As - ka1_for*As*Bs + ka1_back*Cs*st
    
    dBdt = -kd2_for*B*st + kd2_back*Bs
    dBsdt = kd2_for*B*st - kd2_back*Bs - ka1_for*As*Bs + ka1_back*Cs*st
    
    dCdt = -kd3_for*C*st + kd3_back*Cs
    dCsdt = kd3_for*C*st - kd3_back*Cs + ka1_for*As*Bs - ka1_back*Cs*st
    
    dstdt = -kd1_for*A*st - kd2_for*B*st - kd3_for*C*st +\
            kd1_back*As + kd2_back*Bs + kd3_back*Cs  + \
            ka1_for*As*Bs - ka1_back*Cs*st
            
    dTdt = torch.zeros(len(y[:,7]))
    dydx=torch.stack((dAdt,dBdt,dCdt,dAsdt,dBsdt,dCsdt,dstdt,dTdt),dim=1)

    return dydx






