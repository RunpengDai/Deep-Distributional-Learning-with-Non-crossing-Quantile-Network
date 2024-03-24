import os 
import numpy as np
import pickle
import argparse
def check(log_dir):
    dir_name = os.path.dirname(log_dir)
    model_name = os.path.basename(log_dir)
    model_name = model_name.split('QN')[0]+'QN'
    mdl_len = len(model_name)

    name_list = os.listdir(dir_name)
    cdd_list = []

    for pth in name_list:
        if pth[:mdl_len] == model_name:
            cdd_list.append(pth)
    rd_cdd = [os.path.join(dir_name,x) for x in cdd_list]
    test_n = []
    train_n = []


    for pth in rd_cdd:
        pth = os.path.join(pth,'summary/return.pkl')
        if not os.path.exists(pth):
            #print('pass')
            continue
        f = open(pth,'rb')
        data = pickle.load(f)
        f.close()
        test_n.append(len(data[1]))
        train_n.append(len(data[0]))
    if len(test_n)==0:
        return [0,0,0]
    idx = np.argmax(test_n)
    chosen_pth = rd_cdd[idx]
    

    log_dir = chosen_pth

    summary_dir = os.path.join(log_dir, 'summary')

    r_path = os.path.join(summary_dir, 'return.pkl')
        
    f = open(r_path,'rb')
    data = pickle.load(f)
    f.close()

    eval_r = data[1]
    steps = test_n[idx] * 250000
    print(steps*4)
    return [4*steps,np.max(eval_r),np.mean(eval_r)]

log_dir = os.listdir('logs') 
print(log_dir)
result = {}
for pth in log_dir:
#    print(os.path.join(pth,'ncEXQRDQN-0'))
    sol = check(os.path.join('logs',pth,'QRDQN-0'))
    if sol[0]==0:
        continue
    result[pth] = check(os.path.join('logs',pth,'QRDQN-0'))


print(result)
