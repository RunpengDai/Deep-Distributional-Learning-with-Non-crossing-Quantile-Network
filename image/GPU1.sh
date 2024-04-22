#!/bin/sh
cd ..
python -u train.py --cuda --model ncQRDQN --env_id VentureNoFrameskip-v4 --seed 0 --quantile 200 > run/out/ncQRDQN-VentureNoFrameskip-v4-0-200.out &
python -u train.py --cuda --model ncQRDQN --env_id VentureNoFrameskip-v4 --seed 1 --quantile 200 > run/out/ncQRDQN-VentureNoFrameskip-v4-1-200.out &
python -u train.py --cuda --model ncQRDQN --env_id VentureNoFrameskip-v4 --seed 2 --quantile 200 > run/out/ncQRDQN-VentureNoFrameskip-v4-2-200.out &
python -u train.py --cuda --model ncQRDQN --env_id VentureNoFrameskip-v4 --seed 3 --quantile 200 > run/out/ncQRDQN-VentureNoFrameskip-v4-3-200.out &
python -u train.py --cuda --model ncQRDQN --env_id VentureNoFrameskip-v4 --seed 4 --quantile 200 > run/out/ncQRDQN-VentureNoFrameskip-v4-4-200.out &
python -u train.py --cuda --model DEnet --env_id VentureNoFrameskip-v4 --interval 10000 --seed 0 --quantile 200 > run/out/DEnet-VentureNoFrameskip-v4-0-200.out &
python -u train.py --cuda --model DEnet --env_id VentureNoFrameskip-v4 --interval 10000 --seed 1 --quantile 200 > run/out/DEnet-VentureNoFrameskip-v4-1-200.out &
python -u train.py --cuda --model DEnet --env_id VentureNoFrameskip-v4 --interval 10000 --seed 2 --quantile 200 > run/out/DEnet-VentureNoFrameskip-v4-2-200.out &
python -u train.py --cuda --model DEnet --env_id VentureNoFrameskip-v4 --interval 10000 --seed 3 --quantile 200 > run/out/DEnet-VentureNoFrameskip-v4-3-200.out &
python -u train.py --cuda --model DEnet --env_id VentureNoFrameskip-v4 --interval 10000 --seed 4 --quantile 200 > run/out/DEnet-VentureNoFrameskip-v4-4-200.out &
delay=600  # 10 minutes

while true; do
    nvidia-smi
    sleep $delay
done
wait