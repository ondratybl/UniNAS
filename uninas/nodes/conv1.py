import torch.nn as nn
from .base import BaseNode
from .utils import _init_conv_in_graph


class Conv1(BaseNode):  # simple
    def __init__(self, shape: tuple, root_shape: tuple, scheme='kaiming_normal'):
        super().__init__(shape, root_shape, shape[0]*shape[1]*shape[2]*2*shape[0], shape[0]*(shape[0]*1*1 + 1))
        self.conv = nn.Conv2d(shape[0], shape[0], kernel_size=1, padding=0)
        _init_conv_in_graph(self.conv, scheme)

    def forward(self, x):
        return self.conv(x)