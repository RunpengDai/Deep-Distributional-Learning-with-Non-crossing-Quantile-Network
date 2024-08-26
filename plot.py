import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
import re
from util import split_list
model = ["ncQRDQN-200-5e-05-10000",  "DEnet-200-5e-05-10000-l*True"] 
colors = ["blue", "blue", "red"]
names = ["NC-QR-DQN", r"$NQ-Net^*$"]
envs = ["TennisNoFrameskip-v4", "KangarooNoFrameskip-v4", "JamesbondNoFrameskip-v4"] 

env_list = split_list(envs, 3)
height = len(envs) // 3 + 1
fig = plt.figure(figsize=(20, 5*height), dpi = 500)
window = 5

def moving_average(data, window_size):
    num = np.mean(data[-window_size:])
    arr = np.append(data, (window_size-1)*[num])
    weights = np.ones(window_size) / window_size
    return np.convolve(arr, weights, mode='valid').tolist()

def calculate_mean_variance(arrays):
    max_length = max(len(arr) for arr in arrays)
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan) for arr in arrays]
    stacked_arrays = np.vstack(padded_arrays)
    mean = np.nanmean(stacked_arrays, axis=0)
    variance = np.nanstd(stacked_arrays, axis=0)
    return mean, variance



def get_CI(data):
    data = [sub for sub in data if sub]
    mean, std = calculate_mean_variance(data)
    # assert len(data) > 0
    # lenth = min([len(d) for d in data])
    # data  = np.array([d[:lenth] for d in data])
    # mean = np.mean(data, axis=0)
    # std  = np.std(data, axis=0)
    return mean, std


def process_line(env,model):
    base_dir  = "logs/" + env
    data = []
    dirs = [base_dir + "/" + dir for dir in os.listdir(base_dir) if model in dir and len(dir)==(len(model)+2)][:3] # find all the dir with model-seed  and int(dir[-1])!=0
    
    if len(dirs) == 0:
        return None, None
    for dir in dirs:
        try:
            summary = pickle.load(open(dir+'/summary/return.pkl', 'rb'))
        except:
            print(dir)
            continue
        data.append(moving_average(summary[1], window))
    mean, std = get_CI(data)
    return mean , std
    

# To store handles and labels for the legend
handles = []
labels = []
for col, env_l in enumerate(env_list):
    for row, env in enumerate(env_l):
        ax = fig.add_subplot(height, 3, row*3+col+1)
        name = re.search(r"(.*?)(No)", env).group(1).strip()

        for idx, mode in enumerate(model):
            print(env, mode)
            mean, std = process_line(env, mode)

            if mean is None:
                continue
            x = np.arange(len(mean))
            line, =ax.plot(x, mean, color = colors[idx], markerfacecolor='none', markersize =5, label = names[model.index(mode)])
            ax.fill_between(x, mean - std, mean + std, color = colors[idx], alpha=0.2)
            # fig.legend()
            if row == 0 and col == 0:  # Collect handles and labels only from the first subplot
                handles.append(line)
                labels.append(names[model.index(mode)])
        ax.set_title(name, fontsize=15)

# fig.legend(handles, labels, loc=' center')
fig.legend(handles, labels, loc='lower center',  fontsize=14, ncol=2)

# 
fig.savefig("results.png")
