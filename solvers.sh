#!/bin/sh
{
    python main.py --dataset "$DATASET" --name euler --solver euler;
    python main.py --dataset "$DATASET" --name euler_fast --solver euler --fast;
    python main.py --dataset "$DATASET" --name euler_guide --solver euler --guide;
    python main.py --dataset "$DATASET" --name heun3 --solver heun3;
    python main.py --dataset "$DATASET" --name heun3_fast --solver heun3 --fast;
    python main.py --dataset "$DATASET" --name heun3_guide --solver heun3 --guide;
} >> output/results/"$DATASET".txt
