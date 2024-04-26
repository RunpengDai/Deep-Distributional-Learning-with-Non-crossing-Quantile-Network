import os
from util import split_list


model = ["DEnet", "ncQRDQN"]
quantile = [200]
seed = [i for i in range(5)]
env = ["KangarooNoFrameskip-v4", "MsPacmanNoFrameskip-v4"]
NUM_GPU = 4

# "JamesbondNoFrameskip-v4", "TennisNoFrameskip-v4", 
# "VentureNoFrameskip-v4"
### Empty the run folder
folder_path = "vrun"  # 你的文件夹路径

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and (file_path.endswith(".sh") or file_path.endswith(".job")):
        os.remove(file_path)

### Create the single job file
command_names = []
for agent in model:
    for quant in quantile:
        for env_id in env:
            for rand in seed:
                command_name = f"nohup python -u train.py --cuda --model {agent} --env_id {env_id} --seed {rand} > {env_id+agent+rand}.log 2>&1 &"
                command_names.append(command_name)     

GPU_lists = split_list(command_names, NUM_GPU)                

with open("vrun/run.sh", "a") as f:
        f.write(f"cd ..\n")

for i in range(0, NUM_GPU):
    with open("vrun/run.sh", "a") as f:
        f.write(f"export export CUDA_VISIBLE_DEVICES={i}\n")
    for task in GPU_lists[i]:
        with open(f"vrun/run.sh", "a") as f:
            f.write(f"{task}\n")
            
os.chmod('vrun/run.sh', 0o775)        