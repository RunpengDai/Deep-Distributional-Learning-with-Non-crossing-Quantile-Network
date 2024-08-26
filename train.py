import os
import yaml
import argparse
from agents.env import make_pytorch_env
from agents.agent import DEnetAgent
from agents.agent import ncQRDQNAgent
from agents.agent import QRDQNAgent

# Mapping the names and agent classes.
agent_dict = {"QRDQN": "QRDQNAgent", "ncQRDQN": "ncQRDQNAgent", "DEnet": "DEnetAgent"}

def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Modify the hyperparameters.
    config["target_update_interval"] = args.interval
    config["N"] = args.quantile
    config["lr"] = args.lr
    

    # Create environments, standard approach as stable-baselines3. 
    # https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html
    env = make_pytorch_env(args.env_id)
    test_env = make_pytorch_env(args.env_id, episode_life=False, clip_rewards=False)

    # Create directories for logging reward and model checkpoints.
    # May modify if more hyperparameters tuning is needed.
    log_name = f'{args.model}-{args.quantile}-{args.lr}-{args.interval}-{args.seed}'
    if args.model == "DEnet":
        log_name = f'{args.model}-{args.quantile}-{args.lr}-{args.interval}-l*{args.star}-{args.seed}'

    log_dir = os.path.join('logs', args.env_id, log_name)
    

    # Create the agent and run.
    agent = eval(agent_dict[args.model])(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, star = args.star, tensorboard = args.tensorboard, **config)
    
    # Load the model if stopped in the middle of training.
    if args.load:  
        agent.load_checkpoint()

    agent.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('config', 'distrl.yaml'))
    parser.add_argument("--model", type=str, default="DEnet")
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--interval', type=int, default=10000, help="Target update interval")
    parser.add_argument('--quantile', type=int, default=200, help="Number of quantiles")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--load', action='store_true', default=False, help="Load the model checkpoint")
    parser.add_argument('--star', action='store_true', default=True, help="Use the star network, special for DEnet")
    parser.add_argument('--tensorboard', action='store_true', default=False, help="Use tensorboard")
    
    args = parser.parse_args()
    run(args)
