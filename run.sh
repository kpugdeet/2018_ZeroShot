#!/usr/bin/env bash
for trial in {0..9}
do
    rand=$((1 + $RANDOM % 10000))
    CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_$rand --TA=1 --SELATT=0 --OPT=1 --maxSteps=100 --HEADER=1 --numClass=32 --numAtt=64 --lr=1e-6 --SEED=$rand
    CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_$rand --TC=1 --SELATT=0 --OPT=2 --maxSteps=50000 --numClass=32 --numAtt=64 --lr=1e-4 --SEED=$rand
    CUDA_VISIBLE_DEVICES=0 python3 main.py --KEY=APY --DIR=APY_$rand --SELATT=0 --OPT=3 --numClass=32 --numAtt=64 --SEED=$rand
done

