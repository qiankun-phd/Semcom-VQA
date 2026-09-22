from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - depends on training env
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError("HybridVQAModel requires torch in the DI-engine environment.") from _TORCH_IMPORT_ERROR


class _LogSigmaView:
    def __init__(self, log_sigma_param: Any) -> None:
        self.log_sigma_param = log_sigma_param


class HybridVQAModel(nn.Module if nn is not None else object):
    """HPPO-ready actor-critic with task-level discrete heads and resource heads.

    The output follows DI-engine's hybrid PPO convention:
    ``logit.action_type`` is a list of categorical logits, and
    ``logit.action_args`` contains Gaussian ``mu`` and ``sigma`` tensors.
    """

    mode = ["compute_actor", "compute_critic", "compute_actor_critic"]

    def __init__(
        self,
        obs_shape: int,
        max_tasks: int,
        num_uavs: int,
        num_service_levels: int = 4,
        hidden_size: int = 128,
        fixed_sigma_value: float = 0.25,
    ) -> None:
        _require_torch()
        super().__init__()
        self.obs_shape = int(obs_shape)
        self.max_tasks = int(max_tasks)
        self.num_uavs = int(num_uavs)
        self.num_service_levels = int(num_service_levels)
        self.action_args_shape = self.max_tasks * 4
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.assigned_uav_heads = nn.ModuleList([nn.Linear(hidden_size, self.num_uavs) for _ in range(self.max_tasks)])
        self.sensing_heads = nn.ModuleList([nn.Linear(hidden_size, 3) for _ in range(self.max_tasks)])
        self.service_heads = nn.ModuleList([nn.Linear(hidden_size, self.num_service_levels) for _ in range(self.max_tasks)])
        self.resource_mu = nn.Linear(hidden_size, self.action_args_shape)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_sigma = nn.Parameter(torch.full((self.action_args_shape,), float(fixed_sigma_value)).log())
        self.actor = nn.ModuleDict(
            {
                "assigned_uav_heads": self.assigned_uav_heads,
                "sensing_heads": self.sensing_heads,
                "service_heads": self.service_heads,
                "resource_mu": self.resource_mu,
            }
        )
        self.actor_head = [self.actor, _LogSigmaView(self.log_sigma)]

    def forward(self, x: Any, mode: str) -> dict[str, Any]:
        if mode not in self.mode:
            raise ValueError(f"unsupported mode: {mode}")
        return getattr(self, mode)(x)

    def compute_actor(self, x: Any) -> dict[str, Any]:
        embedding, mask = self._encode(x)
        return {"logit": self._actor_logit(embedding, mask)}

    def compute_critic(self, x: Any) -> dict[str, Any]:
        embedding, _mask = self._encode(x)
        return {"value": self.critic(embedding).squeeze(-1)}

    def compute_actor_critic(self, x: Any) -> dict[str, Any]:
        embedding, mask = self._encode(x)
        return {
            "logit": self._actor_logit(embedding, mask),
            "value": self.critic(embedding).squeeze(-1),
        }

    def _encode(self, x: Any) -> tuple[Any, dict[str, Any] | None]:
        mask = None
        if isinstance(x, dict):
            mask = x.get("action_mask")
            state = x.get("observation", x.get("state"))
        else:
            state = x
        if state is None:
            raise KeyError("HybridVQAModel input needs `state` or `observation`.")
        return self.encoder(state.float()), mask

    def _actor_logit(self, embedding: Any, mask: dict[str, Any] | None) -> dict[str, Any]:
        action_type = []
        for idx in range(self.max_tasks):
            action_type.append(self._apply_mask(self.assigned_uav_heads[idx](embedding), mask, "assigned_uav_mask", idx))
            action_type.append(self._apply_mask(self.sensing_heads[idx](embedding), mask, "sensing_mask", idx))
            action_type.append(self._apply_mask(self.service_heads[idx](embedding), mask, "service_level_mask", idx))
        mu = torch.tanh(self.resource_mu(embedding))
        sigma = self.log_sigma.exp().unsqueeze(0).expand_as(mu)
        return {"action_type": action_type, "action_args": {"mu": mu, "sigma": sigma}}

    @staticmethod
    def _apply_mask(logit: Any, mask: dict[str, Any] | None, key: str, idx: int) -> Any:
        if not mask or key not in mask:
            return logit
        row = mask[key]
        if not torch.is_tensor(row):
            row = torch.as_tensor(row, dtype=logit.dtype, device=logit.device)
        row = row.to(dtype=logit.dtype, device=logit.device)
        if row.dim() == 2:
            row = row.unsqueeze(0).expand(logit.shape[0], -1, -1)
        if row.dim() == 3:
            row = row[:, idx, :]
        return logit.masked_fill(row <= 0.0, -1e8)
