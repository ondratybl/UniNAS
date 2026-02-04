import torch.nn as nn
from .base import BaseNode


class Sigmoid(BaseNode):  # simple
    def __init__(self, shape: tuple):
        super().__init__(shape, 3*shape[0]*shape[1]*shape[2], 0)  # sigmoid ~ 3 FLOPs per entry
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(x)