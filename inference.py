import os

import torch
from tqdm import tqdm

from main import build_model, get_data, parse_args
from torchgde import accuracy


@torch.inference_mode()
def main(args):
    device = torch.device("cuda")
    name = f"{args.dataset}_{args.name}"
    checkpoint_path = os.path.join("output", "checkpoints", f"{name}.pt")
    graph, X, Y, _, _, test_mask, num_feats, n_classes = get_data(args, device)
    model = build_model(graph, args, num_feats, n_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    for t in tqdm(torch.linspace(0.1, 1, 10)):
        model.ode.odefunc.nfe = 0
        y_pred = model(X, t)
        test_acc = accuracy(y_pred[test_mask], Y[test_mask]).item()
        print(t.item(), test_acc)


if __name__ == "__main__":
    main(parse_args())
