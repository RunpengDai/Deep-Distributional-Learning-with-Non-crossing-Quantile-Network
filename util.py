import os
import pickle

def split_list(input_list, n):
    length = len(input_list)
    return [input_list[i*length // n: (i+1)*length // n] for i in range(n)]

def load_breakpoint(agent):
    try:
        agent.load_models(os.path.join(agent.model_dir, "final"))
    except:
        print("No final model found")
    try:
        pkl_file = open(agent.summary_dir, 'rb')
        summary = pickle.load(pkl_file)
        agent.r = summary[0]
        agent.eval_r = summary[1]
        agent.episodes = len(agent.r)
        pkl_file.close()
    except:
        print("No summary found")