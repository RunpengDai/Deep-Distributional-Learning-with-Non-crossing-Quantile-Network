#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1

export CUDA_VISIBLE_DEVICES=0
python -u train.py --cuda --model DEnet --env_id YarsRevengeNoFrameskip-v4 --interval 10000 --seed 0 --star &
# python -u train.py --cuda --model DEnet --env_id YarsRevengeNoFrameskip-v4 --interval 10000 --seed 3 --star &
 
export CUDA_VISIBLE_DEVICES=1
# python -u train.py --cuda --model DEnet --env_id TennisNoFrameskip-v4 --interval 10000 --seed 0 --star &
python -u train.py --cuda --model DEnet --env_id TennisNoFrameskip-v4 --interval 10000 --seed 3 --star &
