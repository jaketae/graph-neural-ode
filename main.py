import argparse
import os
import random

import dgl
import lightning as L
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

from torchgde import GCNLayer, GDEFunc, ODEBlock, accuracy
from torchgde.models.odeblock import ODESolvers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument(
        "--dataset",
        default="cora",
        const="cora",
        nargs="?",
        choices=("cora", "citeseer", "pubmed"),
    )
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dropout", type=float, default=0.9)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--verbose", type=int, default=-1)
    parser.add_argument("--guide", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction)
    parser.add_argument("--adjoint", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--solver",
        default="rk4",
        const="rk4",
        nargs="?",
        choices=tuple(e.value for e in ODESolvers),
    )
    args = parser.parse_args()
    return args


def build_model(graph, args, in_features, num_classes):
    dropout = args.dropout
    hidden_channels = args.hidden_channels
    gnn = nn.Sequential(
        GCNLayer(
            g=graph,
            in_feats=hidden_channels,
            out_feats=hidden_channels,
            activation=nn.Softplus(),
            dropout=dropout,
        ),
        GCNLayer(
            g=graph,
            in_feats=hidden_channels,
            out_feats=hidden_channels,
            activation=None,
            dropout=dropout,
        ),
    )
    gdefunc = GDEFunc(gnn)
    gde = ODEBlock(
        odefunc=gdefunc,
        method=args.solver,
        atol=args.atol,
        rtol=args.rtol,
        use_adjoint=args.adjoint,
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ode = gde
            self.embed = GCNLayer(
                g=graph,
                in_feats=in_features,
                out_feats=hidden_channels,
                activation=F.relu,
                dropout=0.4,
            )
            self.out = GCNLayer(
                g=graph,
                in_feats=hidden_channels,
                out_feats=num_classes,
                activation=None,
                dropout=0.0,
            )

        def forward(self, x: torch.Tensor, t: int = 1):
            x = self.embed(x)
            x = self.ode(x, t)
            x = self.out(x)
            return x

    model = Model()
    return model


def get_data(args, device):
    if args.dataset == "cora":
        dataset_cls = dgl.data.CoraGraphDataset
    elif args.dataset == "citeseer":
        dataset_cls = dgl.data.CiteseerGraphDataset
    else:
        dataset_cls = dgl.data.PubmedGraphDataset
    data = dataset_cls(verbose=False)
    X = data[0].ndata["feat"].to(device)
    Y = data[0].ndata["label"].to(device)
    train_mask = torch.BoolTensor(data[0].ndata["train_mask"])
    val_mask = torch.BoolTensor(data[0].ndata["val_mask"])
    test_mask = torch.BoolTensor(data[0].ndata["test_mask"])
    num_features = X.shape[1]
    num_classes = data.num_classes
    g = data[0]
    g = dgl.add_self_loop(g)
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata["norm"] = norm.unsqueeze(1)
    g = g.to(device)
    return g, X, Y, train_mask, val_mask, test_mask, num_features, num_classes


def main(args):
    if args.seed != -1:
        L.seed_everything(args.seed)
    best_val = 0
    device = torch.device("cuda")
    name = f"{args.dataset}_{args.name}"
    checkpoint_path = os.path.join("output", "checkpoints", f"{name}.pt")
    g, X, Y, train_mask, val_mask, test_mask, num_feats, n_classes = get_data(args, device)
    model = build_model(g, args, num_feats, n_classes).to(device)
    lr = 1e-2 if args.dataset == "pubmed" else 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for i in tqdm(range(args.steps), leave=False):
        model.train()
        optimizer.zero_grad()
        if not (args.guide or args.fast):
            t = 1
        else:
            t = random.uniform(0, 1)
        outputs = model(X, t)
        nfe = model.ode.odefunc.nfe
        y_pred = outputs
        if args.fast:
            t = 1
        elif args.guide:
            t = 2
        loss = t * F.cross_entropy(y_pred[train_mask], Y[train_mask])
        loss.backward()
        optimizer.step()
        with torch.inference_mode():
            model.eval()
            y_pred = model(X)
            model.ode.odefunc.nfe = 0
            train_loss = loss.item()
            train_acc = accuracy(y_pred[train_mask], Y[train_mask]).item()
            val_acc = accuracy(y_pred[val_mask], Y[val_mask]).item()
            if args.verbose != -1 and i % args.verbose == 0:
                print(
                    "[{}], loss: {:3.3f}, train acc: {:3.3f}, val acc: {:3.3f}, nfe: {}".format(
                        i, train_loss, train_acc, val_acc, nfe
                    )
                )
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    y_pred = model(X)
    test_acc = accuracy(y_pred[test_mask], Y[test_mask]).item()
    return test_acc


if __name__ == "__main__":
    args = parse_args()
    result = [main(args) for _ in tqdm(range(args.repeat))]
    mean = np.mean(result)
    stdev = np.std(result)
    print(args.name, mean, stdev)
