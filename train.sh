#!/bin/sh
{
    python main.py --dataset "$DATASET" --name baseline;
    python main.py --dataset "$DATASET" --name guide --guide;
    python main.py --dataset "$DATASET" --name fast --fast;
} >> output/results/"$DATASET".txt
