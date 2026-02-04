from torch import zeros_like
from .base import BaseNode


class Zero(BaseNode):  # fork & merge
    def __init__(self, shape: tuple):
        super().__init__(shape)

    def forward(self, x):
        return zeros_like(x)