from enum import Enum
from functools import partial

import torch
import torchdiffeq
from torch import nn
from torchdiffeq import odeint, odeint_adjoint


class ODESolvers(str, Enum):
    DOPRI8 = "dopri8"
    DOPRI5 = "dopri5"
    BOSH3 = "bosh3"
    FELBERG2 = "fehlberg2"
    ADAPTIVE_HEUN = "adaptive_heun"
    EULER = "euler"
    MIDPOINT = "midpoint"
    HEUN3 = "heun3"
    RK4 = "rk4"
    EXPLICIT_ADAMS = "explicit_adams"
    IMPLICIT_ADAMS = "implicit_adams"
    FIXED_ADAMS = "fixed_adams"
    SCIPY_SOLVER = "scipy_solver"


class ODEBlock(nn.Module):
    def __init__(
        self,
        odefunc: nn.Module,
        method: ODESolvers = ODESolvers.DOPRI5,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        use_adjoint: bool = True,
    ):
        super().__init__()
        self.odefunc = odefunc
        self.odeint = partial(
            odeint_adjoint if use_adjoint else odeint,
            func=odefunc,
            atol=atol,
            rtol=rtol,
            method=method,
        )

    def forward(self, x: torch.Tensor, T: int = 1):
        integration_time = torch.tensor([0.0, T], device=x.device)
        out = self.odeint(y0=x, t=integration_time)
        return out[1]

    def forward_batched(self, x: torch.Tensor, nn: int, indices: list, timestamps: set):
        """Modified forward for ODE batches with different integration times."""
        timestamps = torch.Tensor(list(timestamps))
        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(
                self.odefunc, x, timestamps, rtol=self.rtol, atol=self.atol, method=self.method
            )
        else:
            out = torchdiffeq.odeint(
                self.odefunc, x, timestamps, rtol=self.rtol, atol=self.atol, method=self.method
            )

        out = self._build_batch(out, nn, indices).reshape(x.shape)
        return out

    def _build_batch(self, odeout, nn, indices):
        b_out = []
        for i in range(len(indices)):
            b_out.append(odeout[indices[i], i * nn : (i + 1) * nn])
        return torch.cat(b_out).to(odeout.device)

    def trajectory(self, x: torch.Tensor, T: int, num_points: int):
        integration_time = torch.linspace(0, T, num_points)
        integration_time = integration_time.type_as(x)
        out = self.odeint(y0=x, t=integration_time)
        return out
