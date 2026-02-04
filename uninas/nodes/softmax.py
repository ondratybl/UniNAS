import torch.nn as nn
from .base import BaseNode


class Softmax(BaseNode):  # simple
    def __init__(self, shape: tuple):
        super().__init__(shape, shape[0]*shape[1]*(3*shape[2] - 1), 0)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.act(x)