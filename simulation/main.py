#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from generate import gen_univ,quant_univ,qloss
from generate import gen_multi,quant_multi
from model import DQRP,DQR,DQR_NC,DQR_NC2
from functions import train_multi,train_process
import torch
import numpy as np
import matplotlib.pyplot as plt



#%%
# Data Generation
model='wave';error='sinex';

SIZE=2**9;df=2;sigma=1;d=1;

data_train= gen_univ(model=model,size=SIZE,error=error,df=df,sigma=sigma); 
x_train=data_train[:][0];y_train=data_train[:][1];
data_val= gen_univ(model=model,size=10**4,error=error,df=df,sigma=sigma);

# torch.manual_seed(2022);A=-1*torch.randint(0,3,[d,1])*torch.randn([d,1]);
# torch.manual_seed(2021);B=-1*torch.randint(0,3,[d,1])*torch.randn([d,1]);

# data_train= gen_multi(A,B,size=SIZE,d=8,model=model,error=error,df=df,sigma=sigma);
# data_val= gen_multi(A,B,size=SIZE,d=8,model=model,error=error,df=df,sigma=sigma);

batch_size = int(SIZE/2);epochs=1000;

#%% View the data

x_test=torch.linspace(0,1,1000).unsqueeze(1);

taus = torch.linspace(0.05,0.95,19).unsqueeze(1);
quants = quant_univ(x_test, taus,model=model,error=error,df=df,sigma=sigma)
plt.figure(figsize=(10,8))
plt.scatter(data_train[:][0].data.numpy(), data_train[:][1].data.numpy(), color = "k", alpha=0.15,label='Data',s=15)
plt.plot(x_test.data.numpy(), quants[:,:].data.numpy(),alpha=0.9,lw=3)
plt.title('Quantile Regression')
plt.xlabel('Independent varible')
plt.ylabel('Dependent varible')


#%% Implementation of Deep Non-crossing Quantile Networks 

width=128
net_NC = DQR_NC(value_layer=[d,width,width,width,1],delta_layer=[d,width,width,width,len(taus)]);
optimizer_NC = torch.optim.Adam(net_NC.parameters(), lr=0.001);
check = qloss(mode='multiple');
net_NC = train_multi(net_NC,optimizer_NC, epochs, batch_size,80,check,data_train, data_val,taus);

#%% View the estimation of Deep Non-crossing Quantile Networks 
fig, ax = plt.subplots(figsize=(15,10))

plt.cla()
ax.set_title('Deep Quantile Regression', fontsize=35);
ax.set_xlabel('Independent variable', fontsize=24);
ax.set_ylabel('Dependent variable', fontsize=24);
ax.set_xlim(-0.05, 1.05);

taus_test = torch.Tensor([0.05,0.25,0.5,0.75,0.95]).unsqueeze(1);
x = torch.linspace(0,1,1000).unsqueeze(1);
quants = quant_univ(x, taus_test,model=model,error=error,df=df,sigma=sigma);

ax.scatter(data_train[:][0].data.numpy(), data_train[:][1].data.numpy(), color = "k", alpha=0.15,label='Data',s=10);

ax.plot(x.data.numpy(), net_NC(x)[:,0].data.numpy(), color='tab:blue',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_NC(x)[:,1].data.numpy(), color='tab:orange',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_NC(x)[:,2].data.numpy(), color='tab:green',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_NC(x)[:,3].data.numpy(), color='tab:red',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_NC(x)[:,4].data.numpy(), color='tab:purple',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,0].data.numpy(), color='tab:blue',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,1], color='tab:orange',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,2], color='tab:green',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,3], color='tab:red',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,4], color='tab:purple',linestyle='--',alpha=0.9,lw=2)

plt.savefig('figure/NC.png')

#%% Implementation of Deep Non-crossing Quantile Networks (Zhou2020 NeurIPS)

width=128
net_NC2 = DQR_NC2(logit_layer=[d,width,width,width,len(taus)],factor_layer=[d,width,width,width,2]);

optimizer_NC2 = torch.optim.Adam(net_NC2.parameters(), lr=0.001);
check = qloss(mode='multiple');
net_NC2 = train_multi(net_NC2,optimizer_NC2, epochs, batch_size,100,check,data_train, data_val,taus);

#%% View the estimation of Deep Non-crossing Quantile Networks (Zhou2020 NeurIPS)
fig, ax = plt.subplots(figsize=(15,10))

plt.cla()
ax.set_title('Deep Quantile Regression', fontsize=35);
ax.set_xlabel('Independent variable', fontsize=24);
ax.set_ylabel('Dependent variable', fontsize=24);
ax.set_xlim(-0.05, 1.05);

taus_test = torch.Tensor([0.05,0.25,0.5,0.75,0.95]).unsqueeze(1);
x = torch.linspace(0,1,1000).unsqueeze(1);
quants = quant_univ(x, taus_test,model=model,error=error,df=df,sigma=sigma);

ax.scatter(data_train[:][0].data.numpy(), data_train[:][1].data.numpy(), color = "k", alpha=0.15,label='Data',s=10);

ax.plot(x.data.numpy(), net_NC2(x)[:,0].data.numpy(), color='tab:blue',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_NC2(x)[:,1].data.numpy(), color='tab:orange',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_NC2(x)[:,2].data.numpy(), color='tab:green',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_NC2(x)[:,3].data.numpy(), color='tab:red',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_NC2(x)[:,4].data.numpy(), color='tab:purple',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,0].data.numpy(), color='tab:blue',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,1], color='tab:orange',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,2], color='tab:green',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,3], color='tab:red',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,4], color='tab:purple',linestyle='--',alpha=0.9,lw=2)
plt.savefig('figure/NC-QRDQN.png')


#%% Implementation of Deep Quantile Rergession at multiple quantiles (Padilla et al. 2020)
width= 128
net_DQR = DQR(width_vec=[d,width,width,width,len(taus)], Noncrossing=False);
optimizer_DQR = torch.optim.Adam(net_DQR.parameters(), lr=0.001);
check = qloss(mode='multiple');
net_DQR = train_multi(net_DQR,optimizer_DQR, epochs, batch_size,50,check,data_train, data_val,taus);


#%% View the estimation of the Deep Quantile Estimations at multiple quantiles (Padilla et al. 2020)
fig, ax = plt.subplots(figsize=(15,10))

plt.cla()
ax.set_title('Deep Quantile Regression', fontsize=35);
ax.set_xlabel('Independent variable', fontsize=24);
ax.set_ylabel('Dependent variable', fontsize=24);
ax.set_xlim(-0.05, 1.05);

taus_test = torch.Tensor([0.05,0.25,0.5,0.75,0.95]).unsqueeze(1);
x = torch.linspace(0,1,1000).unsqueeze(1);
quants = quant_univ(x, taus_test,model=model,error=error,df=df,sigma=sigma);

ax.scatter(data_train[:][0].data.numpy(), data_train[:][1].data.numpy(), color = "k", alpha=0.15,label='Data',s=10);

ax.plot(x.data.numpy(), net_DQR(x)[:,0].data.numpy(), color='tab:blue',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_DQR(x)[:,1].data.numpy(), color='tab:orange',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_DQR(x)[:,2].data.numpy(), color='tab:green',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_DQR(x)[:,3].data.numpy(), color='tab:red',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_DQR(x)[:,4].data.numpy(), color='tab:purple',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,0].data.numpy(), color='tab:blue',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,1], color='tab:orange',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,2], color='tab:green',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,3], color='tab:red',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,4], color='tab:purple',linestyle='--',alpha=0.9,lw=2)
plt.savefig('figure/DQR.png')

#%% Implementation of Deep Quantile Rergession at multiple quantiles with noncrossing constraint (Padilla et al. 2020)
width= 128

net_DQR_NC = DQR(width_vec=[d,width,width,width,len(taus)], Noncrossing=True);
optimizer_DQR_NC = torch.optim.Adam(net_DQR_NC.parameters(), lr=0.001);
check = qloss(mode='multiple');
net_DQR_NC = train_multi(net_DQR_NC,optimizer_DQR_NC, epochs, batch_size,50,check,data_train, data_val,taus);

#%% View the estimation of the Deep Quantile Estimations at multiple quantiles with noncrossing constraint (Padilla et al. 2020)
fig, ax = plt.subplots(figsize=(15,10))

plt.cla()
ax.set_title('Deep Quantile Regression', fontsize=35);
ax.set_xlabel('Independent variable', fontsize=24);
ax.set_ylabel('Dependent variable', fontsize=24);
ax.set_xlim(-0.05, 1.05);

taus_test = torch.Tensor([0.05,0.25,0.5,0.75,0.95]).unsqueeze(1);
x = torch.linspace(0,1,1000).unsqueeze(1);
quants = quant_univ(x, taus_test,model=model,error=error,df=df,sigma=sigma);

ax.scatter(data_train[:][0].data.numpy(), data_train[:][1].data.numpy(), color = "k", alpha=0.15,label='Data',s=10);

ax.plot(x.data.numpy(), net_DQR_NC(x)[:,0].data.numpy(), color='tab:blue',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_DQR_NC(x)[:,1].data.numpy(), color='tab:orange',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_DQR_NC(x)[:,2].data.numpy(), color='tab:green',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_DQR_NC(x)[:,3].data.numpy(), color='tab:red',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), net_DQR_NC(x)[:,4].data.numpy(), color='tab:purple',linestyle='-',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,0].data.numpy(), color='tab:blue',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,1], color='tab:orange',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,2], color='tab:green',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,3], color='tab:red',linestyle='--',alpha=0.9,lw=2)
ax.plot(x.data.numpy(), quants[:,4], color='tab:purple',linestyle='--',alpha=0.9,lw=2)
plt.savefig('figure/DQR_NC.png')


#%% Implementation of Deep Quantile  Rergession Process
# Create DNN
width=128
net_DQRP = DQRP(width_vec=[d+1,width,width,width,1],activation='ReQU')
optimizer_DQRP=torch.optim.Adam(net_DQRP.parameters(), lr=0.001)
check1=qloss(mode='process')

net_DQRP = train_process(net_DQRP,optimizer_DQRP,epochs,batch_size,50,np.log(SIZE),check1,data_train,data_val,algo=True)
#net1=train(net1,optimizer1,epochs,batch_size,80,np.log(SIZE),check1,data_train,data_val,algo=True,B=100*np.log(SIZE)**2,B_prime=100*np.log(SIZE)**2,xi=xi,alpha=alpha,beta=beta)


#%% View the estimation of the proposed Deep Quantile Regression Process (DQRP)
u = data_train[:][2];
fig, ax = plt.subplots(figsize=(15,10));
ax.scatter(torch.linspace(0,1,u.shape[0]).numpy(), u.detach().numpy(), color = "k", alpha=0.25,label='Data',s=20);

plt.cla()
ax.set_title('Deep Quantile Regression', fontsize=35);
ax.set_xlabel('Independent variable', fontsize=24);
ax.set_ylabel('Dependent variable', fontsize=24);
ax.set_xlim(-0.05, 1.05);


ax.scatter(data_train[:][0].data.numpy(), data_train[:][1].data.numpy(), color = "k", alpha=0.15,label='Data',s=10);
x=torch.linspace(0,1,1000).unsqueeze(1);
quants = quant_univ(x, taus,model=model,error=error,df=df,sigma=sigma);
ax.plot(x.data.numpy(), net_DQRP(x,0.95*torch.ones(x.shape)).data.numpy(), color='tab:blue',linestyle='-',alpha=0.9,lw=2);
ax.plot(x.data.numpy(), net_DQRP(x,0.75*torch.ones(x.shape)).data.numpy(), color='tab:orange',linestyle='-',alpha=0.9,lw=2);
ax.plot(x.data.numpy(), net_DQRP(x,0.5*torch.ones(x.shape)).data.numpy(), color='tab:green',linestyle='-',alpha=0.9,lw=2);
ax.plot(x.data.numpy(), net_DQRP(x,0.25*torch.ones(x.shape)).data.numpy(), color='tab:red',linestyle='-',alpha=0.9,lw=2);
ax.plot(x.data.numpy(), net_DQRP(x,0.05*torch.ones(x.shape)).data.numpy(), color='tab:purple',linestyle='-',alpha=0.9,lw=2);
ax.plot(x.data.numpy(), quants[:,4].data.numpy(), color='tab:blue',linestyle='--',alpha=0.9,lw=2);
ax.plot(x.data.numpy(), quants[:,3], color='tab:orange',linestyle='--',alpha=0.9,lw=2);
ax.plot(x.data.numpy(), quants[:,2], color='tab:green',linestyle='--',alpha=0.9,lw=2);
ax.plot(x.data.numpy(), quants[:,1], color='tab:red',linestyle='--',alpha=0.9,lw=2);
ax.plot(x.data.numpy(), quants[:,0], color='tab:purple',linestyle='--',alpha=0.9,lw=2);
plt.savefig('figure/DQRP.png')


#%% Briefly summarize one estimation result for uni-variate case DQPR
x_test=torch.rand([10000,1]);
quants = quant_univ(x_test, taus,model=model,error=error,df=df,sigma=sigma);

L1_DQR = torch.zeros([1,len(taus)]); L2_DQR = torch.zeros([1,len(taus)]); preds_DQR = torch.zeros([10000,len(taus)]);
L1_DQR_NC = torch.zeros([1,len(taus)]); L2_DQR_NC = torch.zeros([1,len(taus)]); preds_DQR_NC = torch.zeros([10000,len(taus)]);
L1_NC = torch.zeros([1,len(taus)]); L2_NC = torch.zeros([1,len(taus)]); preds_NC = torch.zeros([10000,len(taus)]);
L1_DQRP=torch.zeros([1,len(taus)]);L2_DQRP=torch.zeros([1,len(taus)]);preds_DQRP=torch.zeros([10000,len(taus)]);



for j in range(len(taus)):
    preds_DQR[:,j] = net_DQR(x_test)[:,j].squeeze().detach();
    preds_DQR_NC[:,j] = net_DQR_NC(x_test)[:,j].squeeze().detach();
    preds_NC[:,j] = net_NC(x_test)[:,j].squeeze().detach();
    preds_DQRP[:,j]=net_DQRP(x_test,taus[j].repeat(10000,1).float()).squeeze().detach();
    
    
L1_DQR[0,:]=torch.abs(preds_DQR-quants).mean(0);L2_DQR[0,:]=torch.pow(preds_DQR-quants,2).mean(0);
L1_DQR_NC[0,:]=torch.abs(preds_DQR_NC-quants).mean(0);L2_DQR_NC[0,:]=torch.pow(preds_DQR_NC-quants,2).mean(0);
L1_NC[0,:]=torch.abs(preds_NC-quants).mean(0);L2_NC[0,:]=torch.pow(preds_NC-quants,2).mean(0);
L1_DQRP[0,:]=torch.abs(preds_DQRP-quants).mean(0);L2_DQRP[0,:]=torch.pow(preds_DQRP-quants,2).mean(0);


print(L1_DQR);
print(L1_DQR_NC);
print(L1_DQRP);
print(L1_NC,'\n');

print(L2_DQR);
print(L2_DQR_NC);
print(L2_DQRP);
print(L2_NC,'\n');
