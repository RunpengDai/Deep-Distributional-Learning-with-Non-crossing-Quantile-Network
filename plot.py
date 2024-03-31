import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
from util import split_list
#, "DEnet1000new", "DEnet1000old", "DEnet10000old",
model = ["ncQRDQN", "DEnet1000old", "DEnet1000new", "DEnet10000new", "DEnet10000old"]
colors = ["darkgreen", "magenta", "limegreen", "blue", "orange"]
envs = ["YarsRevengeNoFrameskip-v4", "JamesbondNoFrameskip-v4"] #os.listdir("logs/")
env_list = split_list(envs, 3)
height = len(envs) // 3 + 1

fig = plt.figure(figsize=(20, 5*height))

def get_CI(data):
    data = [sub for sub in data if sub]
    assert len(data) > 0
    lenth = min([len(d) for d in data])
    data  = np.array([d[:lenth] for d in data])
    mean = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    return mean, std

def process_line(env,model):
    base_dir  = "logs/" + env
    data = []
    dirs = [base_dir + "/" + dir for dir in os.listdir(base_dir) if model+"-" in dir]
    if len(dirs) == 0:
        return None, None
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
        for idx, mode in enumerate(model):
            mean, std = process_line(env, mode)
            print(env, mode)

            if mean is None:
                continue
            x = np.arange(len(mean))
            ax.plot(x, mean, color = colors[idx], markerfacecolor='none', markersize =5,marker = "o", label = mode)
            ax.fill_between(x, mean - std, mean + std, color = colors[idx], alpha=0.2)
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