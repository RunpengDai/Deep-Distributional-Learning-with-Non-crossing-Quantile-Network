import os
import yaml
import argparse
from datetime import datetime

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import DEnetAgent
from fqf_iqn_qrdqn.agent import ncQRDQNAgent
from fqf_iqn_qrdqn.agent import QRDQNAgent
import tensorboard
agent_dict = {"QRDQN": "QRDQNAgent", "ncQRDQN": "ncQRDQNAgent", "DEnet": "DEnetAgent"}

def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config["target_update_interval"] = args.interval
    config["N"] = args.quantile
    config["lr"] = args.lr
    # Create environments.
    env = make_pytorch_env(args.env_id)
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False)

    # Specify the directory to log.
    log_name = f'{args.model}-{args.quantile}-{args.lr}-{args.interval}-{args.seed}'
    if args.other is not None:
        log_name = f'{args.model}-{args.quantile}-{args.lr}-{args.interval}-{args.other}-{args.seed}'
    if args.specify is not None:
        log_name = f'{args.specify}-{args.seed}'
    log_dir = os.path.join(
            'logs', args.env_id, log_name)
# Dist-DEnet-ncqr/logs/YarsRevengeNoFrameskip-v4/DEnet-200-1e-05-10000-0
    # Create the agent and run.
    agent = eval(agent_dict[args.model])(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    if args.load:  
        agent.load_checkpoint()
    agent.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'qrdqn.yaml'))
    parser.add_argument("--model", type=str, default="DEnet")
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--interval', type=int, default=10000)
    parser.add_argument('--network', type=str, default="old")
    parser.add_argument('--quantile', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--other', type=str, default = None)
    parser.add_argument('--load', action='store_true')
    parser.add_argument("--specify", type=str, default=None)
    
    args = parser.parse_args()
    print(args)
    run(args)
