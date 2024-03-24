import os
import yaml
import argparse
from datetime import datetime

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import DEnetAgent
from fqf_iqn_qrdqn.agent import ncQRDQNAgent
import tensorboard
agent_dict = {"QRDQN": "QRDQNAgent", "ncQRDQN": "ncQRDQNAgent", "DEnet": "DEnetAgent"}

def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = make_pytorch_env(args.env_id)
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False)

    # Specify the directory to log.
    log_dir = os.path.join(
        'logs', args.env_id, f'{args.model}-{args.seed}')

    # Create the agent and run.
    agent = eval(agent_dict[args.model])(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'qrdqn.yaml'))
    parser.add_argument("--model", type=str, default="DEnet")
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(args)
    run(args)
