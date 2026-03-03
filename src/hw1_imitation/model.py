"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn

class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, action_dim * chunk_size)) 
        self.net = nn.Sequential(*layers)


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.net(state).view(-1, self.chunk_size, self.action_dim)
        return nn.MSELoss()(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.net(state).view(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        in_dim = state_dim + action_dim * chunk_size + 1 
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, action_dim * chunk_size)) 
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        A_t0 = torch.randn_like(action_chunk)
        tau = torch.rand(action_chunk.shape[0], 1, 1, device=action_chunk.device)
        A_ttau = tau * action_chunk + (1 - tau) * A_t0 
        target = action_chunk - A_t0                     

        A_ttau_flat = A_ttau.view(-1, self.chunk_size * self.action_dim)
        tau_flat = tau.view(-1, 1)
        net_input = torch.cat([state, A_ttau_flat, tau_flat], dim=-1)
        velocity = self.net(net_input).view(-1, self.chunk_size, self.action_dim)

        return nn.MSELoss()(velocity, target)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        A_t = torch.randn(state.shape[0], self.chunk_size, self.action_dim, device=state.device)
        for step in range(num_steps):
            tau = torch.full((state.shape[0], 1, 1), step / num_steps, device=state.device)
            A_t_flat = A_t.view(-1, self.chunk_size * self.action_dim)
            tau_flat = tau.view(-1, 1)
            net_input = torch.cat([state, A_t_flat, tau_flat], dim=-1)
            velocity = self.net(net_input).view(-1, self.chunk_size, self.action_dim)
            A_t = A_t + velocity / num_steps

        return A_t


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "MSE":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "Flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
