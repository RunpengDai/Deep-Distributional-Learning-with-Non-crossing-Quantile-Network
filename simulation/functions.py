#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as Data
from torch.autograd import Variable,grad
from tqdm import tqdm


#%% Define functions for training Deep Quantile Networks at multiple levels

#Validation
def validation_multi(model,taus,data_val,loss_function):
    model.eval()
    with torch.no_grad():
        x_val = data_val[:][0]
        y_val = data_val[:][1]
        output=model(x_val);
        loss_val=0;
        for j in range(taus.shape[0]):
            loss_val = loss_val+loss_function(output[:,j].unsqueeze(1),y_val,taus[j]).mean()
      
    return loss_val

# Train
def train_multi(model, optimizer, epochs, batch_size,patience,loss_function,data_train, data_val,taus):
    train_loader = Data.DataLoader(dataset=data_train,batch_size=batch_size,shuffle=True, num_workers=0,);
    loss_list=10*torch.ones([1,1]);
    trigger_list=torch.zeros([1,1]);
    trigger_times = 0
    for epoch in tqdm(range(epochs),total=epochs):
        model.train()
        for step, (x,y,_,_) in enumerate(train_loader):
            x, y = Variable(x.float(),requires_grad=True), Variable(y.float());
            output=model(x);
            loss=0;
            for j in range(taus.shape[0]):
                loss= loss+loss_function(output[:,j].unsqueeze(1),y,taus[j]).mean();
            optimizer.zero_grad();   # clear gradients for next train
            loss.backward();   # backpropagation, compute gradients
            optimizer.step();        # apply gradients
            del x,y,output,loss
        model.eval()
        current_loss = validation_multi(model=model,taus=taus, data_val=data_val,loss_function=loss_function);
        loss_list=torch.cat((loss_list,current_loss.unsqueeze(0).unsqueeze(1)),0);
        loss_min=loss_list.min();
        if epoch>patience:
            if current_loss >loss_min:
                trigger_list = torch.cat((trigger_list,torch.ones([1,1])),0);
                trigger_times = trigger_list[-patience:].sum();
                if trigger_times >= patience:
                    print('Early stopping!')
                    break
                    return model
            else:
                trigger_list=torch.cat((trigger_list,torch.zeros([1,1])),0);
                trigger_times = trigger_list[-patience:].sum();
    return model

#%% Define functions for training Deep Quantile Rergession Process 

#Validation
def validation_process(model,data_val,loss_function):
    # Settings
    model.eval()
    # Test validation data
    with torch.no_grad():
        x_val = data_val[:][0]
        y_val = data_val[:][1]
        u_val = data_val[:][2]
        y_hat_val = model(x_val,u_val)
        loss_val = loss_function(y_hat_val,y_val,u_val).mean()
    return loss_val

# Train
def train_process(model, optimizer, epochs, batch_size, patience,penalty,loss_function,data_train, data_val, algo=False,B=1e10,B_prime=1e10,xi='uniform',alpha=0.5,beta=0.5):
    train_loader = Data.DataLoader(dataset=data_train,batch_size=batch_size,shuffle=True, num_workers=0,)
    # Early stopping
    loss_list=10*torch.ones([1,1])
    trigger_list=torch.zeros([1,1])
    trigger_times = 0
    for epoch in tqdm(range(epochs),total=epochs):
        model.train()
        for step, (x,y,u,_) in enumerate(train_loader):
            x, y ,u= Variable(x.float(),requires_grad=True), Variable(y.float()),Variable(u.float(),requires_grad=True)
            if xi=='uniform':
                uu=torch.rand([x.size()[0],1]);uu=Variable(uu,requires_grad=True);
            if xi=='beta':
                torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta])).sample([x.size()[0]]);uu=Variable(uu,requires_grad=True);
            if algo==True:
                pred= model(x,uu)    # input x and predict based on x
                grads = grad(outputs=pred,inputs=uu,grad_outputs=torch.ones_like(pred),
                         retain_graph=True, create_graph=True, only_inputs=True)[0]
                ratio_B_prime = min(B_prime/torch.max(torch.abs(grads.detach())),1)
                ratio_B = min(B/torch.max(torch.abs(pred.detach())),1)
                last_para = list(list(model.children())[-1][-1].parameters())[0]
                last_para.data = last_para.data*min(ratio_B,ratio_B_prime)
                pred= model(x,uu)    # input x and predict based on x
                loss1 = loss_function(pred,y,uu)     # must be (1. nn output, 2. target)
                grads = grad(outputs=pred,inputs=uu,grad_outputs=torch.ones_like(pred),
                         retain_graph=True, create_graph=True, only_inputs=True)[0]
            if algo==False:
                pred= model(x,u)    # input x and predict based on x
                grads = grad(outputs=pred,inputs=u,grad_outputs=torch.ones_like(pred),
                         retain_graph=True, create_graph=True, only_inputs=True)[0]
                ratio_B_prime = min(B_prime/torch.max(torch.abs(grads.detach())),1)
                ratio_B = min(B/torch.max(torch.abs(pred.detach())),1)
                last_para = list(list(model.children())[-1][-1].parameters())[0]
                last_para.data = last_para.data*min(ratio_B,ratio_B_prime)
                pred= model(x,u)    # input x and predict based on x
                loss1 = loss_function(pred,y,u)     # must be (1. nn output, 2. target)
                grads = grad(outputs=pred,inputs=u,grad_outputs=torch.ones_like(pred),
                         retain_graph=True, create_graph=True, only_inputs=True)[0]
            grads = grads.view(grads.size(0),  -1)
            loss2= penalty*torch.max(-grads,0*torch.ones(grads.shape)).mean()
            loss=loss1+loss2
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            del x,y,u,uu,pred,loss1,loss2,loss,grads
        model.eval()
            
        # Early stopping
        current_loss = validation_process(model, data_val, loss_function)
        loss_list=torch.cat((loss_list,current_loss.unsqueeze(0).unsqueeze(1)),0)
        loss_min=loss_list.min()
        if epoch>patience:
            if current_loss >loss_min:
                trigger_list = torch.cat((trigger_list,torch.ones([1,1])),0)
                trigger_times = trigger_list[-patience:].sum()
                if trigger_times >= patience:
                    print('Early stopping!')
                    break
                    return model
            else:
                trigger_list=torch.cat((trigger_list,torch.zeros([1,1])),0)
                trigger_times = trigger_list[-patience:].sum()
    return model
