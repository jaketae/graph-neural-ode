#!/bin/sh
{
    python main.py --dataset "$DATASET" --name baseline;
    python main.py --dataset "$DATASET" --name guide;
    python main.py --dataset "$DATASET" --name fast;
} > output/results/"$DATASET".txt
