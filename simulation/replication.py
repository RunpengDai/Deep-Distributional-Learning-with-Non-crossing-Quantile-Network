#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from generate import gen_univ,quant_univ,qloss
from generate import gen_multi,quant_multi
from model import DQRP,DQR,DQR_NC,DQR_NC2
from functions import train_multi,train_process
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Data Generation

#taus = torch.Tensor([0.05,0.25,0.5,0.75,0.95]).unsqueeze(1);
taus = torch.linspace(0.05,0.95,19).unsqueeze(1)
check=qloss(mode='multiple'); check1=qloss(mode='process');

model='triangle'; error="sinex"; d=1;df=2;sigma=1;SIZE=2**9

#torch.manual_seed(2024);A= torch.randint(0,3,[d,1])*torch.randn([d,1]);
#torch.manual_seed(2025);B= torch.randint(0,3,[d,1])*torch.randn([d,1]);

R=10; width=128;width_NC=128;
epochs=1000;batch_size=int(SIZE/2)

#L1_DQR = torch.zeros([R,len(taus)]); L2_DQR = torch.zeros([R,len(taus)]); preds_DQR = torch.zeros([10000,len(taus)]);
#L1_DQR_NC = torch.zeros([R,len(taus)]); L2_DQR_NC = torch.zeros([R,len(taus)]); preds_DQR_NC = torch.zeros([10000,len(taus)]);
#L1_DQRP=torch.zeros([R,len(taus)]);L2_DQRP=torch.zeros([R,len(taus)]);preds_DQRP=torch.zeros([10000,len(taus)]);
#L1_NC = torch.zeros([R,len(taus)]); L2_NC = torch.zeros([R,len(taus)]); preds_NC = torch.zeros([10000,len(taus)]);
L1_NC2 = torch.zeros([R,len(taus)]); L2_NC2 = torch.zeros([R,len(taus)]); preds_NC2 = torch.zeros([10000,len(taus)]);    

for r in range(R):
    data_train= gen_univ(model=model,size=SIZE,error=error,df=df,sigma=sigma)
    data_val= gen_univ(model=model,size=int(SIZE/4),error=error,df=df,sigma=sigma)
    #data_train= gen_multi(A=A,B=B,model=model,size=SIZE,d=d,error=error,df=df,sigma=sigma)
    #data_val= gen_multi(A=A,B=B,model=model,size=10000,d=d,error=error,df=df,sigma=sigma)
    #net_DQR = DQR(width_vec=[d,width,width,width,len(taus)], Noncrossing=False);optimizer_DQR = torch.optim.Adam(net_DQR.parameters(), lr=0.001);
    #net_DQR_NC = DQR(width_vec=[d,width,width,width,len(taus)], Noncrossing=True);optimizer_DQR_NC = torch.optim.Adam(net_DQR_NC.parameters(), lr=0.001);
    #net_DQRP = DQRP(width_vec=[d+1,width,width,width,1],activation='ReQU');optimizer_DQRP=torch.optim.Adam(net_DQRP.parameters(), lr=0.001);
    #net_NC = DQR_NC(value_layer=[d,int(width_NC/2),int(width_NC/2),int(width_NC/2),1],delta_layer=[d,int(width_NC/2),int(width_NC/2),int(width_NC/2),len(taus)],activation='ELU');
    #optimizer_NC = torch.optim.Adam(net_NC.parameters(), lr=0.001);
    net_NC2 = DQR_NC2(logit_layer=[d,int(width_NC/2),int(width_NC/2),int(width_NC/2),len(taus)],factor_layer=[d,int(width_NC/2),int(width_NC/2),int(width_NC/2),2],activation="ReLU");
    optimizer_NC2 = torch.optim.Adam(net_NC2.parameters(), lr=0.001);
    
    #net_DQR = train_multi(net_DQR,optimizer_DQR, epochs, batch_size,80,check,data_train, data_val,taus);
    #net_DQR_NC = train_multi(net_DQR_NC,optimizer_DQR_NC, epochs, batch_size,80,check,data_train, data_val,taus);
    #net_DQRP = train_process(net_DQRP,optimizer_DQRP,epochs,batch_size,80,np.log(SIZE),check1,data_train,data_val,algo=False)
    #net_NC = train_multi(net_NC,optimizer_NC, epochs, batch_size,80,check,data_train, data_val,taus);
    net_NC2 = train_multi(net_NC2,optimizer_NC2, epochs, batch_size,100,check,data_train, data_val,taus);
    
    x_test = torch.rand([10000,d]);
    quants = quant_univ(x_test, taus,model=model,error=error,df=df,sigma=sigma);
    #quants = quant_multi(x_test,taus,A,B,model=model,error=error,df=df,sigma=sigma)

    for j in range(len(taus)):
        #preds_DQR[:,j] = net_DQR(x_test)[:,j].squeeze().detach();
        #preds_DQR_NC[:,j] = net_DQR_NC(x_test)[:,j].squeeze().detach();
        #preds_DQRP[:,j]=net_DQRP(x_test,taus[j].repeat(10000,1).float()).squeeze().detach();
        #preds_NC[:,j] = net_NC(x_test)[:,j].squeeze().detach();
        preds_NC2[:,j] = net_NC2(x_test)[:,j].squeeze().detach();
        
        
    #L1_DQR[r,:]=torch.abs(preds_DQR-quants).mean(0);L2_DQR[r,:]=torch.pow(preds_DQR-quants,2).mean(0);
    #L1_DQR_NC[r,:]=torch.abs(preds_DQR_NC-quants).mean(0);L2_DQR_NC[r,:]=torch.pow(preds_DQR_NC-quants,2).mean(0);
    #L1_DQRP[r,:]=torch.abs(preds_DQRP-quants).mean(0);L2_DQRP[r,:]=torch.pow(preds_DQRP-quants,2).mean(0);
    #L1_NC[r,:]=torch.abs(preds_NC-quants).mean(0);L2_NC[r,:]=torch.pow(preds_NC-quants,2).mean(0);
    L1_NC2[r,:]=torch.abs(preds_NC2-quants).mean(0);L2_NC2[r,:]=torch.pow(preds_NC2-quants,2).mean(0);

    #print(L1_DQR[r,:]);
    #print(L1_DQR_NC[r,:]);
    #print(L1_DQRP[r,:]);
    #print(L1_NC[r,:]);
    print(L1_NC2[r,:],);
    print('\n')
    
    #print(L2_DQR[r,:]);
    #print(L2_DQR_NC[r,:]);
    #print(L2_DQRP[r,:]);
    #print(L2_NC[r,:]);
    print(L2_NC2[r,:],'\n');
    print('\n')
    
    #state = {'net':net.state_dict(),'optimizer':optimizer.state_dict()};
    #state1 = {'net':net1.state_dict(),'optimizer':optimizer1.state_dict()};
    #torch.save(state, './estimates/%s_%s_%d/DQPR_ReQU%d.pth'% (model,error,SIZE,i))
    #torch.save(state1, './estimates/%s_%s_%d/DQPR_ReLU%d.pth'% (model,error,SIZE,i))
    
#% Summarize the replication results

#print(L1_DQR.mean(dim=0));
#print(L1_DQR_NC.mean(dim=0));
#print(L1_DQRP.mean(dim=0));
print(L1_NC2.mean(dim=0));
#print(L1_NC.mean(dim=0),'\n');

#print(L1_DQR.std(dim=0));
#print(L1_DQR_NC.std(dim=0));
#print(L1_DQRP.std(dim=0));
print(L1_NC2.std(dim=0));
#print(L1_NC.std(dim=0),'\n');

#%
#DQR = np.concatenate(((L1_DQR.mean(dim=0).detach().numpy().reshape(-1,1)).round(3),(L1_DQR.std(dim=0).detach().numpy().reshape(-1,1)).round(3),L2_DQR.mean(dim=0).detach().numpy().reshape(-1,1).round(3),(L2_DQR.std(dim=0).detach().numpy().reshape(-1,1)).round(3)),axis=1)

#DQR_NC = np.concatenate(((L1_DQR_NC.mean(dim=0).detach().numpy().reshape(-1,1)).round(3),(L1_DQR_NC.std(dim=0).detach().numpy().reshape(-1,1)).round(3),L2_DQR_NC.mean(dim=0).detach().numpy().reshape(-1,1).round(3),(L2_DQR_NC.std(dim=0).detach().numpy().reshape(-1,1)).round(3)),axis=1)

#DQRP = np.concatenate(((L1_DQRP.mean(dim=0).detach().numpy().reshape(-1,1)).round(3),(L1_DQRP.std(dim=0).detach().numpy().reshape(-1,1)).round(3),L2_DQRP.mean(dim=0).detach().numpy().reshape(-1,1).round(3),(L2_DQRP.std(dim=0).detach().numpy().reshape(-1,1)).round(3)),axis=1)
#NC = np.concatenate(((L1_NC.mean(dim=0).detach().numpy().reshape(-1,1)).round(3),(L1_NC.std(dim=0).detach().numpy().reshape(-1,1)).round(3),L2_NC.mean(dim=0).detach().numpy().reshape(-1,1).round(3),(L2_NC.std(dim=0).detach().numpy().reshape(-1,1)).round(3)),axis=1)

NC2 = np.concatenate(((L1_NC2.mean(dim=0).detach().numpy().reshape(-1,1)).round(3),(L1_NC2.std(dim=0).detach().numpy().reshape(-1,1)).round(3),L2_NC2.mean(dim=0).detach().numpy().reshape(-1,1).round(3),(L2_NC2.std(dim=0).detach().numpy().reshape(-1,1)).round(3)),axis=1)

#print(DQR)
#print(DQR_NC)
#print(DQRP)
#print(NC)
print(NC2)

#%% Save the results to local directory
#pd.DataFrame(DQR).to_csv('./%s_%s_%d/DQR.csv'% (model,error,SIZE));
#pd.DataFrame(DQR_NC).to_csv('./%s_%s_%d/DQR_NC.csv'% (model,error,SIZE));  
#pd.DataFrame(DQRP).to_csv('./%s_%s_%d/DQRP.csv'% (model,error,SIZE));
#pd.DataFrame(NC).to_csv('./%s_%s_%d/NC.csv'% (model,error,SIZE));  
pd.DataFrame(NC2).to_csv('./%s_%s_%d/NC2.csv'% (model,error,SIZE));  




