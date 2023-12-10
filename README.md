## Graph Neural Differential Equations

This repository contains code for [Graph Neural ODE++](https://drive.google.com/file/d/1Rjn1X87AvP62rmBsImnTcj74V1Mc6IS_/view?usp=sharing). This work was completed as part of CPSC 483: Deep Learning on Graph-Structured Data. 

## Abstract

> We propose Graph Neural ODE++, an improved paradigm for Graph Neural Ordinary Differential Equations (GDEs). Inspired by recent literature in score-based generative models, we explore two different heuristics for training GDEs: linear simplex refinement and consistency modeling. We observe that both methods improve model performance on standard transductive node classification datasets, albeit marginally. Furthermore, we show that there is a direct relationship between training methodology and the behavior of the model at different time steps within the integration window of the ODE. 

## Quickstart

1. Clone this repository.

```
$ git clone https://github.com/jaketae/graph-neural-ode/
```

2. Create a Python virtual enviroment and install package requirements.

```
$ cd graph-neural-ode
$ python -m venv venv
$ pip install -U pip wheel # update pip
$ pip install -r requirements.txt
```

3. Train all 3 models (GDE, GDE++ LSR, GDE++ CM) on the Cora dataset.

```
CUDA_VISIBLE_DEVICES=0 DATASET=cora sh train.sh
```

## Training

To train a model, run [`main.py`](main.py). The full list of supported arguments are shown below.

```
$ python main.py --help
usage: main.py [-h] [--name NAME] [--dataset [{cora,citeseer,pubmed}]] [--repeat REPEAT] [--hidden_channels HIDDEN_CHANNELS]
               [--steps STEPS] [--dropout DROPOUT] [--atol ATOL] [--rtol RTOL] [--verbose VERBOSE] [--guide | --no-guide]
               [--fast | --no-fast] [--adjoint | --no-adjoint] [--seed SEED]
               [--solver [{dopri8,dopri5,bosh3,fehlberg2,adaptive_heun,euler,midpoint,heun3,rk4,explicit_adams,implicit_adams,fixed_adams,scipy_solver}]]

options:
  -h, --help            show this help message and exit
  --name NAME
  --dataset [{cora,citeseer,pubmed}]
  --repeat REPEAT
  --hidden_channels HIDDEN_CHANNELS
  --steps STEPS
  --dropout DROPOUT
  --atol ATOL
  --rtol RTOL
  --verbose VERBOSE
  --guide, --no-guide
  --fast, --no-fast
  --adjoint, --no-adjoint
  --seed SEED
  --solver [{dopri8,dopri5,bosh3,fehlberg2,adaptive_heun,euler,midpoint,heun3,rk4,explicit_adams,implicit_adams,fixed_adams,scipy_solver}]
```

The script will report the mean and standard deviation of the test accuracy under `output/results`. The best model checkpoint measured by validation accuracy will be saved under `output/checkpoints`.

## Inference

To evaluate a model checkpoint, run [`inference.py`](inference.py). The full list of supported arguments are shown below.

```
$ python inference.py --help
usage: inference.py [-h] [--name NAME] [--dataset [{cora,citeseer,pubmed}]] [--repeat REPEAT] [--hidden_channels HIDDEN_CHANNELS]
                    [--steps STEPS] [--dropout DROPOUT] [--atol ATOL] [--rtol RTOL] [--verbose VERBOSE] [--guide | --no-guide]
                    [--fast | --no-fast] [--adjoint | --no-adjoint] [--seed SEED]
                    [--solver [{dopri8,dopri5,bosh3,fehlberg2,adaptive_heun,euler,midpoint,heun3,rk4,explicit_adams,implicit_adams,fixed_adams,scipy_solver}]]

options:
  -h, --help            show this help message and exit
  --name NAME
  --dataset [{cora,citeseer,pubmed}]
  --repeat REPEAT
  --hidden_channels HIDDEN_CHANNELS
  --steps STEPS
  --dropout DROPOUT
  --atol ATOL
  --rtol RTOL
  --verbose VERBOSE
  --guide, --no-guide
  --fast, --no-fast
  --adjoint, --no-adjoint
  --seed SEED
  --solver [{dopri8,dopri5,bosh3,fehlberg2,adaptive_heun,euler,midpoint,heun3,rk4,explicit_adams,implicit_adams,fixed_adams,scipy_solver}]
```

The script will automatically locate the checkpoint file based on the `name` and `dataset` arguments.

## License

Released under the [MIT License](LICENSE).
