from generate import gen_univ,quant_univ,qloss
from generate import gen_multi,quant_multi
from model import DQRP,DQR,DQR_NC,DQR_NC2
from functions import train_multi,train_process
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse


def run(model):
    # Data Generation
    width=128;
    taus = torch.linspace(0.1,0.9,9).unsqueeze(1);
    check=qloss(mode='multiple'); check1=qloss(mode='process');

    error="sinex"; d=1;df=2;sigma=1;SIZE=2**9
    # Prepare dataloader
    epochs=1000;batch_size=int(SIZE/2)

    #%% View the data

    x_test=torch.linspace(0,1,1000).unsqueeze(1);

    taus=torch.Tensor([0.05,0.25,0.5,0.75,0.95]).unsqueeze(1);

    quants = quant_univ(x_test, taus,model=model,error=error,df=df,sigma=sigma)
    plt.figure(figsize=(10,8))
    plt.plot(x_test.data.numpy(), quants[:,:].data.numpy(),alpha=0.9,lw=3)
    plt.title('Quantile Regression')
    plt.xlabel('Independent varible')
    plt.ylabel('Dependent varible')
    plt.show()


    #%% Train and save the predictions for one replication


    preds_DQR = torch.zeros([1000,len(taus)]);
    preds_DQR_NC = torch.zeros([1000,len(taus)]);
    preds_NC = torch.zeros([1000,len(taus)]);
    preds_NC_ = torch.zeros([1000,len(taus)]);
    preds_NC2 = torch.zeros([1000,len(taus)]);
    preds_DQRP=torch.zeros([1000,len(taus)]);
        
    data_train= gen_univ(model=model,size=SIZE,error=error,df=df,sigma=sigma)
    data_val= gen_univ(model=model,size=int(SIZE),error=error,df=df,sigma=sigma)
    net_DQR = DQR(width_vec=[d,width,width,width,len(taus)], Noncrossing=False);optimizer_DQR = torch.optim.Adam(net_DQR.parameters(), lr=0.001);
    net_DQR_NC = DQR(width_vec=[d,width,width,width,len(taus)], Noncrossing=True);optimizer_DQR_NC = torch.optim.Adam(net_DQR_NC.parameters(), lr=0.001);
    net_DQRP = DQRP(width_vec=[d+1,width,width,width,1],activation='ReQU');optimizer_DQRP=torch.optim.Adam(net_DQRP.parameters(), lr=0.001);
    net_NC = DQR_NC(value_layer=[d,int(width/2),int(width/2),int(width/2),1],delta_layer=[d,int(width/2),int(width/2),int(width/2),len(taus)]); optimizer_NC = torch.optim.Adam(net_NC.parameters(), lr=0.001);
    net_NC_ = DQR_NC(value_layer=[d,int(width/2),int(width/2),int(width/2),1],delta_layer=[d,int(width/2),int(width/2),int(width/2),len(taus)],activation='ReLU'); optimizer_NC_ = torch.optim.Adam(net_NC_.parameters(), lr=0.001);
    net_NC2 = DQR_NC2([d,int(width/2),int(width/2),int(width/2),len(taus)],[d,int(width/2),int(width/2),int(width/2),2]); optimizer_NC2 = torch.optim.Adam(net_NC2.parameters(), lr=0.001);
        


    net_DQR = train_multi(net_DQR,optimizer_DQR, epochs, batch_size,100,check,data_train, data_val,taus);
    net_DQR_NC = train_multi(net_DQR_NC,optimizer_DQR_NC, epochs, batch_size,100,check,data_train, data_val,taus);
    net_DQRP = train_process(net_DQRP,optimizer_DQRP,epochs,batch_size,100,np.log(SIZE),check1,data_train,data_val,algo=True)
    net_NC = train_multi(net_NC,optimizer_NC, epochs, batch_size,100,check,data_train, data_val,taus);
    net_NC2 = train_multi(net_NC2,optimizer_NC2, epochs, batch_size,100,check,data_train, data_val,taus);
    net_NC_ = train_multi(net_NC_,optimizer_NC_, epochs, batch_size,100,check,data_train, data_val,taus);


    
    x_test=torch.linspace(-0.1,1.1,1000).unsqueeze(1)
    quants = quant_univ(x_test, taus,model=model,error=error,df=df,sigma=sigma)

    for j in range(len(taus)):
        preds_DQR[:,j] = net_DQR(x_test)[:,j].squeeze().detach();
        preds_DQR_NC[:,j] = net_DQR_NC(x_test)[:,j].squeeze().detach();
        preds_NC[:,j] = net_NC(x_test)[:,j].squeeze().detach();
        preds_NC2[:,j] = net_NC2(x_test)[:,j].squeeze().detach();
        preds_DQRP[:,j]=net_DQRP(x_test,taus[j].repeat(1000,1).float()).squeeze().detach();
        preds_NC_[:,j] = net_NC_(x_test)[:,j].squeeze().detach();
        
    preds=[preds_NC,preds_DQR,preds_DQR_NC,preds_NC_,preds_NC2,preds_DQRP]
    
        
    #%% Plot and show the estimations for crossing and non-crossing methods
    colors=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:cyan']
    #methods=['With Crossing','Without Crossing']
    methods=['NQ-Net','DQR','DQR*','NQ-Net*','NC-QR-DQN','DQRP']

    names=(r'$\tau=0.05$',r'$\tau=0.25$',r'$\tau=0.50$',r'$\tau=0.75$',r'$\tau=0.95$','Data')

    x_test=torch.linspace(-0.1,1.1,1000).unsqueeze(1)
    quants = quant_univ(x_test, taus,model=model,error=error,df=df,sigma=sigma)


    figs, axs = plt.subplots(2,3,figsize=(60,34))
    #ticksize=15;titlesize=32;llw=3;dlw=3;
    for m,method in enumerate(methods):
        axs[m//3,m%3].tick_params(axis='both', which='major', labelsize=35)
        axs[m//3,m%3].set_title('%s'% (methods[m]),fontdict={'family':'Times New Roman','size':55})
        axs[m//3,m%3].set_xlabel(r'$X$', fontdict={'family': 'Times New Roman', 'size': 40})
        axs[m//3,m%3].set_ylabel(r'$Y$', fontdict={'family': 'Times New Roman', 'size': 40})
        axs[m//3,m%3].set_xlim(-0.05, 1.05)
        axs[m//3,m%3].set_ylim([-6,6])
        for j in range(len(taus)):
            axs[m//3,m%3].plot(x_test, preds[m][:,j], color=colors[j],
                        linestyle='-',alpha=0.9,lw=5)
            
        axs[m//3,m%3].scatter(data_train[:][0].data.numpy(), data_train[:][1].data.numpy(), color = "k", alpha=0.25,label='Data',s=50)
        axs[m//3,m%3].legend(names,loc='upper left',fontsize=32,ncol=1)
        axs[m//3,m%3].plot(x_test, quants, alpha=0.8,lw=4,linestyle='--')

    plt.savefig(f'figure/{model}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="wave")    
    args = parser.parse_args()
    run(args.model)





