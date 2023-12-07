import argparse
import time

import dgl
import dgl.data
import lightning as L
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from torchgde import GCNLayer, GDEFunc, ODEBlock, PerformanceContainer, accuracy
from torchgde.models.odeblock import ODESolvers

# import os
# import random


def parse_args():
    parser = argparse.ArgumentParser()
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
    hidden_channels = args.hidden_channels
    gnn = nn.Sequential(
        GCNLayer(
            g=g,
            in_feats=hidden_channels,
            out_feats=hidden_channels,
            activation=nn.Softplus(),
            dropout=0.9,
        ),
        GCNLayer(
            g=g, in_feats=hidden_channels, out_feats=hidden_channels, activation=None, dropout=0.9
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


def main():
    device = "cpu"
    args = parse_args()
    data = get_data(args)
    L.seed_everything(args.seed)

    # Cora is a node-classification datasets with 2708 nodes
    X = torch.FloatTensor(data[0].ndata["feat"]).to(device)
    Y = torch.LongTensor(data[0].ndata["label"]).to(device)

    # In transductive semi-supervised node classification tasks on graphs, the model has access to all
    # node features but only a masked subset of the labels
    train_mask = torch.BoolTensor(data[0].ndata["train_mask"])
    val_mask = torch.BoolTensor(data[0].ndata["val_mask"])
    test_mask = torch.BoolTensor(data[0].ndata["test_mask"])

    num_feats = X.shape[1]
    n_classes = data.num_classes

    # 140 training samples, 300 validation, 1000 test
    n_classes, train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item()

    # # add self-edge for each node
    # g = data[0].graph
    # g.remove_edges_from(nx.selfloop_edges(g))
    # g.add_edges_from(zip(g.nodes(), g.nodes()))
    # g = dgl.DGLGraph(g)
    # edges = g.edges()
    # n_edges = g.number_of_edges()

    # n_edges
    g = data[0]

    # compute diagonal of normalization matrix D according to standard formula
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    # add to dgl.Graph in order for the norm to be accessible at training time
    g.ndata["norm"] = norm.unsqueeze(1)  # .to(device)

    g = g.to(device)
    """# Graph Neural Differential Equations (GDEs)

    As Neural ODEs, GDEs require specification of an ODE function (`ODEFunc`), representing the set of layers that will be called repeatedly by the ODE solver, as well as an ODE block (`ODEBlock`), tasked with calling the ODE solver on the ODE function. The ODEFunc is passed to the ODEBlock at initialization.

    We introduce the convolutional variant of GDEs, `GCDEs`. The only difference resides in the type of GNN layer utilized in the ODEFunc.

    For adaptive step GDEs (dopri5) we increase the hidden dimension to 64 to reduce the stiffness of the ODE and therefore the number of ODEFunc evaluations (`NFE`: Number Function Evaluation)
    """

    # for GCDEs, the ODEFunc is specified by two GCN layers. Softplus is used as activation. Smoother activations
    # have been observed to help avoid numerical instability and reduce stiffness of the ODE described
    # by a repeated call to the ODEFunc. High dropout improves performance on transductive node classification
    # tasks due to their small training sets. GDEs can take advantage of this property due to their 'deeper'
    # computational graph. NOTE: too much dropout increases stiffness and therefore NFEs

    m = build_model(g, args, num_feats, n_classes)
    """### Training loop

    We use standard hyperparameters for GCNs, namely `1e-2` learning rate and `5e-4` weight decay.
    For a fair comparison, GDE-dpr5 should be evaluated against deeper GCN models (GCNs with 4+
    layers). This is because datasets such as Cora penalize deeper models due to small training
    sets and thus need for very strong regularizers. GDE-rk4, whose ODEFunc is evaluated only 4
    times, should be compared with shallower GCN models.
    """

    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    logger = PerformanceContainer(
        data={
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
            "forward_time": [],
            "backward_time": [],
            "nfe": [],
        }
    )
    steps = 3000
    verbose_step = 1
    num_grad_steps = 0

    for i in range(steps):  # looping over epochs
        m.train()
        start_time = time.time()

        outputs = m(X)
        f_time = time.time() - start_time

        nfe = m.ode.odefunc.nfe

        y_pred = outputs

        loss = criterion(y_pred[train_mask], Y[train_mask])
        opt.zero_grad()

        start_time = time.time()
        loss.backward()
        b_time = time.time() - start_time

        opt.step()
        num_grad_steps += 1

        with torch.inference_mode():
            m.eval()

            # calculating outputs again with zeroed dropout
            y_pred = m(X)
            m.ode.odefunc.nfe = 0

            train_loss = loss.item()
            train_acc = accuracy(y_pred[train_mask], Y[train_mask]).item()
            test_acc = accuracy(y_pred[test_mask], Y[test_mask]).item()
            test_loss = criterion(y_pred[test_mask], Y[test_mask]).item()
            logger.deep_update(
                logger.data,
                dict(
                    train_loss=[train_loss],
                    train_accuracy=[train_acc],
                    test_loss=[test_loss],
                    test_accuracy=[test_acc],
                    nfe=[nfe],
                    forward_time=[f_time],
                    backward_time=[b_time],
                ),
            )

        if num_grad_steps % verbose_step == 0:
            print(
                "[{}], Loss: {:3.3f}, Train Accuracy: {:3.3f}, Test Accuracy: {:3.3f}, NFE: {}".format(
                    num_grad_steps, train_loss, train_acc, test_acc, nfe
                )
            )

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1)
    plt.plot(logger.data["train_loss"])
    plt.plot(logger.data["test_loss"])
    plt.legend(["Train loss", "Test loss"])
    plt.subplot(2, 2, 2)
    plt.plot(logger.data["train_accuracy"])
    plt.plot(logger.data["test_accuracy"])
    plt.legend(["Train accuracy", "Test accuracy"])
    plt.subplot(2, 2, 3)
    plt.plot(logger.data["forward_time"])
    plt.plot(logger.data["backward_time"])
    plt.legend(["Forward time", "Backward time"])
    plt.subplot(2, 2, 4)
    plt.plot(logger.data["nfe"], marker="o", linewidth=0.1, markersize=1)
    plt.legend(["NFE"])
