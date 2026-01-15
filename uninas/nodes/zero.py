from torch import zeros_like
from .base import BaseNode


class Zero(BaseNode):  # fork & merge
    def __init__(self, shape: tuple, root_shape: tuple):
        super().__init__(shape, root_shape)

    def forward(self, x):
        return zeros_like(x)