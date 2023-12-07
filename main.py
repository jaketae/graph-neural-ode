import argparse
import os
import random
import time

import dgl
import dgl.data
import lightning as L
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F

from torchgde import GCNLayer, GDEFunc, ODEBlock, PerformanceContainer, accuracy
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
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--dropout", type=float, default=0.9)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--guide", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction)
    parser.add_argument("--adjoint", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--solver",
        default="rk4",
        const="rk4",
        nargs="?",
        choices=tuple(e.value for e in ODESolvers),
    )
    args = parser.parse_args()
    return args


def build_model(g, args, in_features, num_classes):
    dropout = args.dropout
    hidden_channels = args.hidden_channels
    gnn = nn.Sequential(
        GCNLayer(
            g=g,
            in_feats=hidden_channels,
            out_feats=hidden_channels,
            activation=nn.Softplus(),
            dropout=dropout,
        ),
        GCNLayer(
            g=g,
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
                g=g,
                in_feats=in_features,
                out_feats=hidden_channels,
                activation=F.relu,
                dropout=0.4,
            )
            self.out = GCNLayer(
                g=g, in_feats=hidden_channels, out_feats=num_classes, activation=None, dropout=0.0
            )

        def forward(self, x: torch.Tensor, t: int = 1):
            x = self.embed(x)
            x = self.ode(x, t)
            x = self.out(x)
            return x

    model = Model()
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_data(args):
    if args.dataset == "cora":
        return dgl.data.CoraGraphDataset()
    elif args.dataset == "citeseer":
        return dgl.data.CiteseerGraphDataset()
    return dgl.data.PubmedGraphDataset()


def plot(logger, path):
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1)
    plt.plot(logger.data["train_loss"])
    plt.plot(logger.data["val_loss"])
    plt.legend(["Train loss", "Validation loss"])
    plt.subplot(2, 2, 2)
    plt.plot(logger.data["train_accuracy"])
    plt.plot(logger.data["val_accuracy"])
    plt.legend(["Train accuracy", "Validation accuracy"])
    plt.subplot(2, 2, 3)
    plt.plot(logger.data["forward_time"])
    plt.plot(logger.data["backward_time"])
    plt.legend(["Forward time", "Backward time"])
    plt.subplot(2, 2, 4)
    plt.plot(logger.data["nfe"], marker="o", linewidth=0.1, markersize=1)
    plt.legend(["NFE"])
    plt.savefig(path)


def main(args):
    L.seed_everything(args.seed)
    data = get_data(args)
    device = torch.device("cuda")

    X = data[0].ndata["feat"].to(device)
    Y = data[0].ndata["label"].to(device)
    train_mask = torch.BoolTensor(data[0].ndata["train_mask"])
    val_mask = torch.BoolTensor(data[0].ndata["val_mask"])
    test_mask = torch.BoolTensor(data[0].ndata["test_mask"])
    num_feats = X.shape[1]
    n_classes = data.num_classes
    g = data[0]
    g = dgl.add_self_loop(g)
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata["norm"] = norm.unsqueeze(1)
    g = g.to(device)

    model = build_model(g, args, num_feats, n_classes).to(device)
    print(f"parameters: {count_parameters(model)}")
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    logger = PerformanceContainer(
        data={
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "forward_time": [],
            "backward_time": [],
            "nfe": [],
        }
    )

    best_val = 0
    checkpoint_path = os.path.join("output", "checkpoints", f"{args.name}.pt")
    for i in range(args.steps):
        model.train()
        optimizer.zero_grad()
        if not (args.guide or args.fast):
            t = 1
        else:
            t = random.uniform(0, 1)
        start_time = time.time()
        outputs = model(X, t)
        f_time = time.time() - start_time
        nfe = model.ode.odefunc.nfe
        y_pred = outputs
        if args.fast:
            t = 1
        loss = t * F.cross_entropy(y_pred[train_mask], Y[train_mask])
        start_time = time.time()
        loss.backward()
        b_time = time.time() - start_time
        optimizer.step()
        with torch.inference_mode():
            model.eval()
            y_pred = model(X)
            model.ode.odefunc.nfe = 0
            train_loss = loss.item()
            train_acc = accuracy(y_pred[train_mask], Y[train_mask]).item()
            val_acc = accuracy(y_pred[val_mask], Y[val_mask]).item()
            val_loss = F.cross_entropy(y_pred[val_mask], Y[val_mask]).item()
            logger.deep_update(
                logger.data,
                dict(
                    train_loss=[train_loss],
                    train_accuracy=[train_acc],
                    val_loss=[val_loss],
                    val_accuracy=[val_acc],
                    nfe=[nfe],
                    forward_time=[f_time],
                    backward_time=[b_time],
                ),
            )
            if i % args.verbose == 0:
                print(
                    "[{}], loss: {:3.3f}, train acc: {:3.3f}, val acc: {:3.3f}, nfe: {}".format(
                        i, train_loss, train_acc, val_acc, nfe
                    )
                )
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), checkpoint_path)

    model.eval()
    y_pred = model(X)
    test_acc = accuracy(y_pred[test_mask], Y[test_mask]).item()
    print(f"last test: {test_acc}")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    y_pred = model(X)
    test_acc = accuracy(y_pred[test_mask], Y[test_mask]).item()
    print(f"best test: {test_acc}")
    plot(logger, checkpoint_path=os.path.join("output", "figs", f"{args.name}.png"))


if __name__ == "__main__":
    main(parse_args())
