#!/bin/sh
cd ..
python -u train.py --cuda --model DEnet --env_id JamesbondNoFrameskip-v4 --seed 2 > run/out/DEnet-JamesbondNoFrameskip-v4-2.out &
python -u train.py --cuda --model DEnet --env_id JamesbondNoFrameskip-v4 --seed 3 > run/out/DEnet-JamesbondNoFrameskip-v4-3.out &
python -u train.py --cuda --model DEnet --env_id TennisNoFrameskip-v4 --seed 0 > run/out/DEnet-TennisNoFrameskip-v4-0.out &
python -u train.py --cuda --model DEnet --env_id TennisNoFrameskip-v4 --seed 1 > run/out/DEnet-TennisNoFrameskip-v4-1.out &
python -u train.py --cuda --model DEnet --env_id TennisNoFrameskip-v4 --seed 2 > run/out/DEnet-TennisNoFrameskip-v4-2.out &
python -u train.py --cuda --model DEnet --env_id TennisNoFrameskip-v4 --seed 3 > run/out/DEnet-TennisNoFrameskip-v4-3.out &
delay=600  # 10 minutes

while true; do
    nvidia-smi
    sleep $delay
done
wait