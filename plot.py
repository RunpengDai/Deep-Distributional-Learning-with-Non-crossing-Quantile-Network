import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
from util import split_list

model = ["DEnet", "ncQRDQN"]
envs = os.listdir("logs/")
env_list = split_list(envs, 3)
height = len(envs) // 3 + 1

fig = plt.figure(figsize=(20, 5*height))

def get_CI(data):
    data = [sub for sub in data if sub]
    lenth = min([len(d) for d in data])
    data  = np.array([d[:lenth] for d in data])
    mean = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    return mean, std

def process_line(env,model):
    base_dir  = "logs/" + env
    data = []
    dirs = [base_dir + "/" + dir for dir in os.listdir(base_dir) if model in dir]
    #print(dirs)
    for dir in dirs:
        try:
            summary = pickle.load(open(dir+'/summary/return.pkl', 'rb'))
        except:
            print(dir)
            continue
        data.append(summary[1])
    #print(data)
    mean, std = get_CI(data)
    return mean , std
    
for col, env_l in enumerate(env_list):
    for row, env in enumerate(env_l):
        ax = fig.add_subplot(height, 3, row*3+col+1)
        mean_base, std_base = process_line(env, model[1])
        mean_deep, std_deep = process_line(env, model[0])
        x_base = np.arange(len(mean_base))
        x_deep = np.arange(len(mean_deep))
        #print(mean_base, mean_deep)
        ax.plot(x_base, mean_base, color='magenta', markerfacecolor='none', marker='o', markersize =5,label = "ncQRDQN")
        ax.fill_between(x_base, mean_base - std_base, mean_base + std_base, color='magenta', alpha=0.2)

        ax.plot(x_deep, mean_deep, color='limegreen', markerfacecolor='none', marker='o', markersize =5,label = "DEnet")
        ax.fill_between(x_deep, mean_deep - std_deep, mean_deep + std_deep, color='limegreen', alpha=0.2)
        ax.set_title(env)

fig.legend()
fig.savefig("results.png")


# for num,l in enumerate(["l5", "l10", "l15"]):
#     dirs = ["results/"+dir for dir in os.listdir("results/") if l in dir]
#     if len(dirs) == 0:
#         continue
#     ax = fig.add_subplot(1,3,num+1)
#     false,true = [],[]
#     for dir in dirs:
#         with open(dir,'rb') as f:
#             b = pickle.load(f)
#         false.append(b) if "False" in dir else true.append(b)
#     false,true = np.array(false).reshape(len(false), 6), np.array(true).reshape(len(true), 6)
#     false = false[false[:,0].argsort()]
#     true = true[true[:,0].argsort()]

#     xt, xf = true[:,0], false[:,0]
#     ax.plot(xf,false[:,1],color='darkred', markerfacecolor='none', marker='o', markersize =5,label = "Mean DR")
#     ax.fill_between(xf, false[:,1] - false[:,3], false[:,1] + false[:,3], color='darkred', alpha=0.2)
    
#     ax.plot(xf,false[:,2],color='darkgreen', markerfacecolor='none', marker='o', markersize =5,label = "Mean PLG")
#     ax.fill_between(xf, false[:,2] - false[:,4], false[:,2] + false[:,4], color='darkgreen', alpha=0.2)

#     ax.plot(xt,true[:,1],color='magenta', markerfacecolor='none', marker='o', markersize =5,label = "Deep DR")
#     ax.fill_between(xt, true[:,1] - true[:,3], true[:,1] + true[:,3], color='magenta', alpha=0.2)
    
#     ax.plot(xt,true[:,2],color='limegreen', markerfacecolor='none', marker='o', markersize =5,label = "Deep PLG")
#     ax.fill_between(xt, true[:,2] - true[:,4], true[:,2] + true[:,4], color='limegreen', alpha=0.2)
# fig.legend()
# fig.savefig("results.png")