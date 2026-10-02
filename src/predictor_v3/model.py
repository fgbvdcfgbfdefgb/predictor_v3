"""Large residual actor-critic sized to approximately 450M trainable parameters."""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, expansion: int = 2):
        super().__init__()
        inner = width * expansion
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, inner)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(inner, width)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))

class ActorCritic(nn.Module):
    """Residual MLP policy/value network.

    With 16 input features, width=2941, blocks=13 and 7 actions this has
    450,043,592 trainable parameters (approximately 450M). The parameter
    count is written to the run metadata by the trainer rather than assumed.
    """
    def __init__(self, n, hidden=2941, actions=7, blocks=13,
                 gradient_checkpointing=True):
        super().__init__()
        self.input = nn.Linear(n, hidden)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden, expansion=2) for _ in range(blocks)
        )
        self.final_norm = nn.LayerNorm(hidden)
        self.actor = nn.Linear(hidden, actions)
        self.critic = nn.Linear(hidden, 1)
        self.gradient_checkpointing = gradient_checkpointing
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        h = self.input(x)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and h.requires_grad:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        h = self.final_norm(h)
        return self.actor(h), self.critic(h).squeeze(-1)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
