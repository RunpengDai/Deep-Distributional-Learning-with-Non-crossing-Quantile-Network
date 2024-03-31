import os
import shutil
from util import split_list
NUM_GPU = 3


model = ["ncQRDQN", "DEnet"]
seed = [i for i in range(5)]
env = ["YarsRevengeNoFrameskip-v4"]

# "JamesbondNoFrameskip-v4", "TennisNoFrameskip-v4", 

### Empty the run folder
shutil.rmtree("run/")
os.mkdir("run/")

### Create the single job file
command_names = []
for agent in model:
    if agent == "DEnet":
        network = ["old", "new"]
        interval = [10000, 1000]
        for net in network:
            for inte in interval:
                for env_id in env:
                    for rand in seed:
                        out_name = f"{agent}-{env_id}-{net}-{rand}.out"
                        command_name = "python -u train.py --cuda --model {} --env_id {} --interval {} --network {} --seed {} > run/out/{}".format(agent, env_id, inte, net, rand, out_name)
                        command_names.append(command_name)
    else:
        for env_id in env:
            for rand in seed:
                out_name = f"{agent}-{env_id}-{rand}.out"
                command_name = "python -u train.py --cuda --model {} --env_id {} --seed {} > run/out/{}".format(agent, env_id, rand, out_name)
                command_names.append(command_name)     

### Create the sbatch files
GPU_lists = split_list(command_names, NUM_GPU)                

for i in range(0, NUM_GPU):
    num_tasks = len(GPU_lists[i])
    with open("run/run.sh".format(i), "a") as f:
        f.write(f"sbatch GPU{i}.job\n")
    with open("run/GPU{}.job".format(i), "a") as f:
        f.write(f"""#!/bin/bash
#SBATCH -n {num_tasks}
#SBATCH -t 11-00:00:00
#SBATCH -p htzhulab
#SBATCH --mem 160g
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH -o out/GPU{i}.out

module rm python
module load cuda/11.8
module load anaconda
conda activate base
conda activate distRL
                   
bash GPU{i}.sh
""")
    with open(f"run/GPU{i}.sh", "a") as f:
        f.write("""#!/bin/sh
cd ..
""")
    # write the python command in sh file
    for file in GPU_lists[i]:
        with open(f"run/GPU{i}.sh", "a") as f:
            f.write(file + " &\n")
        
    # finish the sh file
    with open(f"run/GPU{i}.sh", "a") as f:
        f.write("""delay=600  # 10 minutes

while true; do
    nvidia-smi
    sleep $delay
done
wait""")
    os.chmod(f'run/GPU{i}.sh', 0o775)
            
os.chmod('run/run.sh', 0o775)        