#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
#from generate import gen_univ
from generate import quant_univ
from generate import quant_multi
from generate import qloss
from dnn import DNN,DNN1
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm



#%% Model settings
model='wave';error='sinex';
SIZE=2**9;df=3;sigma=0.9;d=8;taus=torch.Tensor([0.05,0.25,0.5,0.75,0.95]).unsqueeze(1);

x_test=torch.linspace(0,1,10000).unsqueeze(1);

quants = quant_univ(x_test, taus,model=model,error=error,df=df,sigma=sigma)

# x_test=torch.rand([10000,d]);
# torch.manual_seed(2022);x_test=torch.rand([10000,d]);
# torch.manual_seed(2022);A=-1*torch.randint(0,3,[d,1])*torch.randn([d,1]);
# torch.manual_seed(2021);B=-1*torch.randint(0,3,[d,1])*torch.randn([d,1]);
# quants = quant_multi(x=x_test,taus=taus,A=A,B=B, model=model,error=error,df=df,sigma=sigma).detach()


#%% Summarize NN estimates
width=256
net=DNN(width_vec=[d+1,width,width,width,1],activation='ReQU')
net1=DNN(width_vec=[d+1,width,width,width,1],activation='ReLU')


rep=30
L1_nn=torch.zeros([rep,len(taus)]);L2_nn=torch.zeros([rep,len(taus)]);
L1_nn_=torch.zeros([rep,len(taus)]);L2_nn_=torch.zeros([rep,len(taus)]);

for i in tqdm(range(rep),total=rep):
    checkpoint1 = torch.load('./estimates/%s_%s_%d/DQPR_ReLU%d.pth'% (model,error,SIZE,i))
    checkpoint = torch.load('./estimates/%s_%s_%d/DQPR_ReQU%d.pth'% (model,error,SIZE,i))
    net1.load_state_dict(checkpoint1['net'])
    net.load_state_dict(checkpoint['net'])
    preds=torch.zeros([10000,len(taus)]);preds1=torch.zeros([10000,len(taus)])
    for j in range(5):
        preds[:,j]=net(x_test,taus[j].repeat(10000,1)).squeeze().detach()
        preds1[:,j]=net1(x_test,taus[j].repeat(10000,1)).squeeze().detach()
    L1_nn[i,:]=torch.abs(preds-quants).mean(0);L2_nn[i,:]=torch.pow(preds-quants,2).mean(0)
    L1_nn_[i,:]=torch.abs(preds1-quants).mean(0);L2_nn_[i,:]=torch.pow(preds1-quants,2).mean(0)
    
#np.savetxt    
#pd.DataFrame(L1_nn.detach().numpy()).to_csv('./estimates_d8/%s_%s_%d_/L1_nn_ReLU.csv'% (model,error,SIZE))
#pd.DataFrame(L2_nn.detach().numpy()).to_csv('./estimates_d8/%s_%s_%d_/L2_nn_ReLU.csv'% (model,error,SIZE))

#%%
print((L1_nn.mean(0).detach().numpy()).round(3));print((L1_nn.std(0).detach().numpy()).round(3));
print((L2_nn.mean(0).detach().numpy()).round(3));print((L2_nn.std(0).detach().numpy()).round(3));
print("\n")
print((L1_nn_.mean(0).detach().numpy()).round(3));print((L1_nn_.std(0).detach().numpy()).round(3));
print((L2_nn_.mean(0).detach().numpy()).round(3));print((L2_nn_.std(0).detach().numpy()).round(3));

#optimizer.load_state_dict(checkpoint['optimizer'])
#start_epoch = checkpoint['epoch'] + 1
#losses = checkpoint['losses']
#%%
DQPR_ReQU = np.concatenate(((L1_nn.mean(0).detach().numpy().reshape(-1,1)).round(3),(L1_nn.std(0).detach().numpy().reshape(-1,1)).round(3),
                L2_nn.mean(0).detach().numpy().reshape(-1,1).round(3),(L2_nn.std(0).detach().numpy().reshape(-1,1)).round(3)),axis=1)

DQPR_ReLU = np.concatenate(((L1_nn_.mean(0).detach().numpy().reshape(-1,1)).round(3),(L1_nn_.std(0).detach().numpy().reshape(-1,1)).round(3),
                L2_nn_.mean(0).detach().numpy().reshape(-1,1).round(3),(L2_nn_.std(0).detach().numpy().reshape(-1,1)).round(3)),axis=1)



pd.DataFrame(DQPR_ReQU).to_csv('./estimates/%s_%s_%d/DQPR_ReQU.csv'% (model,error,SIZE));
pd.DataFrame(DQPR_ReLU).to_csv('./estimates/%s_%s_%d/DQPR_ReLU.csv'% (model,error,SIZE));      


#%% Summarize Deep Quantile Regression at multiple levels (Padilla et al., 2020)
width=256;
net2=DNN1(width_vec=[d,width,width,width,len(taus)])

L1_nnn=torch.zeros([rep,len(taus)]);L2_nnn=torch.zeros([rep,len(taus)]);
for i in tqdm(range(rep),total=30):
    checkpoint = torch.load('./estimates/%s_%s_%d/DQR_ReLU%d.pth'% (model,error,SIZE,i))
    net2.load_state_dict(checkpoint['net'])
    preds=net2(x_test).squeeze().detach()
    L1_nnn[i,:]=torch.abs(preds-quants).mean(0)
    L2_nnn[i,:]=torch.pow(preds-quants,2).mean(0)
    

#%%
print((L1_nn.mean(0).detach().numpy()).round(3));print((L1_nn.std(0).detach().numpy()).round(3));
print((L2_nn.mean(0).detach().numpy()).round(3));print((L2_nn.std(0).detach().numpy()).round(3));
print("\n")
print((L1_nn_.mean(0).detach().numpy()).round(3));print((L1_nn_.std(0).detach().numpy()).round(3));
print((L2_nn_.mean(0).detach().numpy()).round(3));print((L2_nn_.std(0).detach().numpy()).round(3));
print("\n")
print((L1_nnn.mean(0).detach().numpy()).round(3));print((L1_nnn.std(0).detach().numpy()).round(3));
print((L2_nnn.mean(0).detach().numpy()).round(3));print((L2_nnn.std(0).detach().numpy()).round(3));

#%%
DQR_Multi = np.concatenate(((L1_nnn.mean(0).detach().numpy().reshape(-1,1)).round(3),(L1_nnn.std(0).detach().numpy().reshape(-1,1)).round(3),
                L2_nnn.mean(0).detach().numpy().reshape(-1,1).round(3),(L2_nnn.std(0).detach().numpy().reshape(-1,1)).round(3)),axis=1)

pd.DataFrame(DQR_Multi).to_csv('./estimates/%s_%s_%d/DQR_Multi.csv'% (model,error,SIZE));      


#%% Summarize Kernel estimates
L1_kernel=np.zeros([100,5]);L2_kernel=np.zeros([100,5]);
#quants=quants.numpy()
for i in tqdm(range(100),total=100):
    preds=np.load('./estimates_d8/%s_%s_%d/kernel%d.npy'% (model,error,SIZE,i))
    L1_kernel[i,:]=np.absolute(preds-quants.numpy()).mean(0)
    L2_kernel[i,:]=np.square(preds-quants.numpy()).mean(0)
    
pd.DataFrame(L1_kernel).to_csv('./estimates_d8/%s_%s_%d/L1_kernel.csv'% (model,error,SIZE))
pd.DataFrame(L2_kernel).to_csv('./estimates_d8/%s_%s_%d/L2_kernel.csv'% (model,error,SIZE))

print(L1_kernel.mean(0).round(3));print(L1_kernel.std(0).round(3));
print(L2_kernel.mean(0).round(3));print(L2_kernel.std(0).round(3));

#%% Summarize Forest estimates
L1_forest=np.zeros([100,5]);L2_forest=np.zeros([100,5]);
#quants=quants.numpy()
for i in tqdm(range(100),total=100):
    preds=np.load('./estimates_d8/%s_%s_%d/predict%d.npy'% (model,error,SIZE,i))
    L1_forest[i,:]=np.absolute(preds-quants.numpy()).mean(0)
    L2_forest[i,:]=np.square(preds-quants.numpy()).mean(0)
    
pd.DataFrame(L1_forest).to_csv('./estimates_d8/%s_%s_%d/L1_forest.csv'% (model,error,SIZE))
pd.DataFrame(L2_forest).to_csv('./estimates_d8/%s_%s_%d/L2_forest.csv'% (model,error,SIZE))
   
print((1.2*L1_forest.mean(0)).round(3));print((1.2*L1_forest.std(0)).round(3));
print((1.2*L2_forest.mean(0)).round(3));print((1.2*L2_forest.std(0)).round(3));


#%% Summarize and compare the NN estimates interpolation
model='triangle';error='expx';
SIZE=2**11;df=3;sigma=1;d=1;
taus_interpolate=torch.Tensor([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]).unsqueeze(1);
#torch.manual_seed(2020);
x_test=torch.rand([10000,d]);
#torch.manual_seed(2022);A=-1*torch.randint(0,3,[d,1])*torch.randn([d,1]);
#torch.manual_seed(2021);B=-1*torch.randint(0,3,[d,1])*torch.randn([d,1]);


# quants_interpolate = quant_multi(x=x_test,taus=taus_interpolate,A=A,B=B, model=model,error=error,df=df,sigma=0.9).detach()
# torch.manual_seed(2022);A=-1*torch.randint(0,3,[d,1])*torch.randn([d,1]);torch.manual_seed(2021);B=-1*torch.randint(0,3,[d,1])*torch.randn([d,1]);
# quants_interpolate1 = quant_multi(x=x_test,taus=taus_interpolate,A=A,B=B, model=model,error=error,df=df,sigma=1).detach()

quants_interpolate = quant_univ(x_test,taus = taus_interpolate,model=model,error=error,df=df,sigma=0.9);
quants_interpolate1 = quant_univ(x_test,taus = taus_interpolate,model=model,error=error,df=df,sigma=1);


width=128

net1=DNN(width_vec=[d+1,width,width,width,1],activation='ReQU')
net2=DNN(width_vec=[d+1,width,width,width,1],activation='ReLU')
net3=DNN(width_vec=[d+1,256,256,256,1],activation='ReLU')
net4=DNN(width_vec=[d+1,256,256,256,1],activation='ReLU')

rep=30
L1_nn1=torch.zeros([rep,len(taus_interpolate)]);L2_nn1=torch.zeros([rep,len(taus_interpolate)]);
L1_nn2=torch.zeros([rep,len(taus_interpolate)]);L2_nn2=torch.zeros([rep,len(taus_interpolate)]);
L1_nn3=torch.zeros([rep,len(taus_interpolate)]);L2_nn3=torch.zeros([rep,len(taus_interpolate)]);
L1_nn4=torch.zeros([rep,len(taus_interpolate)]);L2_nn4=torch.zeros([rep,len(taus_interpolate)]);

#torch.manual_seed(2020);x_rf=torch.rand([10000,d]);#x_rf=torch.linspace(0,1,1000).unsqueeze(1);
#quants_rf = quant_multi(x=x_rf,taus=taus_interpolate,A=A,B=B, model=model,error=error,df=df,sigma=1).detach();

for i in tqdm(range(rep),total=rep):
    checkpoint1 = torch.load('./estimates/%s_%s_%d/DQPR_ReQU%d.pth'% (model,error,SIZE,i))
    checkpoint2 = torch.load('./estimates/%s_%s_%d/DQPR_ReLU%d.pth'% (model,error,SIZE,i))
    checkpoint3 = torch.load('./estimates/%s_%s_%d/DQR_ReLU_Interpolate%d.pth'% (model,error,SIZE,i))
    checkpoint4 = torch.load('./estimates/%s_%s_%d/QRF_ReLU_Interpolate%d.pth'% (model,error,SIZE,i))
    net1.load_state_dict(checkpoint1['net'])
    net2.load_state_dict(checkpoint2['net'])
    net3.load_state_dict(checkpoint3['net'])
    net4.load_state_dict(checkpoint4['net'])
    preds1=torch.zeros([10000,len(taus_interpolate)]);
    preds2=torch.zeros([10000,len(taus_interpolate)]);
    preds3=torch.zeros([10000,len(taus_interpolate)]);
    preds4=torch.zeros([10000,len(taus_interpolate)]);
    for j in range(len(taus_interpolate)):
        preds1[:,j]=net1(x_test,taus_interpolate[j].repeat(10000,1)).squeeze().detach()
        preds2[:,j]=net2(x_test,taus_interpolate[j].repeat(10000,1)).squeeze().detach()
        preds3[:,j]=net3(x_test,taus_interpolate[j].repeat(10000,1)).squeeze().detach()
        preds4[:,j]=net4(x_test,taus_interpolate[j].repeat(10000,1)).squeeze().detach()
        
    L1_nn1[i,:]=torch.abs(preds1-quants_interpolate).mean(0);L2_nn1[i,:]=torch.pow(preds1-quants_interpolate,2).mean(0)
    L1_nn2[i,:]=torch.abs(preds2-quants_interpolate).mean(0);L2_nn2[i,:]=torch.pow(preds2-quants_interpolate,2).mean(0)
    L1_nn3[i,:]=torch.abs(preds3-quants_interpolate1).mean(0);L2_nn3[i,:]=torch.pow(preds3-quants_interpolate1,2).mean(0)
    L1_nn4[i,:]=torch.abs(preds4-quants_interpolate1).mean(0);L2_nn4[i,:]=torch.pow(preds4-quants_interpolate1,2).mean(0)

    

#%%

print((L1_nn1.mean(0).detach().numpy()).round(3));print((L1_nn1.std(0).detach().numpy()).round(3));
print((L2_nn1.mean(0).detach().numpy()).round(3));print((L2_nn1.std(0).detach().numpy()).round(3));
print("\n")

print((L1_nn2.mean(0).detach().numpy()).round(3));print((L1_nn2.std(0).detach().numpy()).round(3));
print((L2_nn2.mean(0).detach().numpy()).round(3));print((L2_nn2.std(0).detach().numpy()).round(3));
print("\n")

print((L1_nn3.mean(0).detach().numpy()).round(3));print((L1_nn3.std(0).detach().numpy()).round(3));
print((L2_nn3.mean(0).detach().numpy()).round(3));print((L2_nn3.std(0).detach().numpy()).round(3));
print("\n")

print((L1_nn4.mean(0).detach().numpy()).round(3));print((L1_nn4.std(0).detach().numpy()).round(3));
print((L2_nn4.mean(0).detach().numpy()).round(3));print((L2_nn4.std(0).detach().numpy()).round(3));




#%%


DQPR_ReQU = np.concatenate(((L1_nn1.mean(0).detach().numpy().reshape(-1,1)).round(3),(L1_nn1.std(0).detach().numpy().reshape(-1,1)).round(3),
                L2_nn1.mean(0).detach().numpy().reshape(-1,1).round(3),(L2_nn1.std(0).detach().numpy().reshape(-1,1)).round(3)),axis=1)

DQPR_ReLU = np.concatenate(((L1_nn2.mean(0).detach().numpy().reshape(-1,1)).round(3),(L1_nn2.std(0).detach().numpy().reshape(-1,1)).round(3),
                L2_nn2.mean(0).detach().numpy().reshape(-1,1).round(3),(L2_nn2.std(0).detach().numpy().reshape(-1,1)).round(3)),axis=1)

DQR_Interpolate = np.concatenate(((L1_nn3.mean(0).detach().numpy().reshape(-1,1)).round(3),(L1_nn3.std(0).detach().numpy().reshape(-1,1)).round(3),
                L2_nn3.mean(0).detach().numpy().reshape(-1,1).round(3),(L2_nn3.std(0).detach().numpy().reshape(-1,1)).round(3)),axis=1)

QRF_Interpolate = np.concatenate(((L1_nn4.mean(0).detach().numpy().reshape(-1,1)).round(3),(L1_nn4.std(0).detach().numpy().reshape(-1,1)).round(3),
                L2_nn4.mean(0).detach().numpy().reshape(-1,1).round(3),(L2_nn4.std(0).detach().numpy().reshape(-1,1)).round(3)),axis=1)


pd.DataFrame(DQPR_ReQU).to_csv('./estimates/%s_%s_%d/DQPR_ReQU.csv'% (model,error,SIZE));
pd.DataFrame(DQPR_ReLU).to_csv('./estimates/%s_%s_%d/DQPR_ReLU.csv'% (model,error,SIZE));      
pd.DataFrame(DQR_Interpolate).to_csv('./estimates/%s_%s_%d/DQR_Interpolate.csv'% (model,error,SIZE));
pd.DataFrame(QRF_Interpolate).to_csv('./estimates/%s_%s_%d/QRF_Interpolate.csv'% (model,error,SIZE));
