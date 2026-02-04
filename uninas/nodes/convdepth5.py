import torch.nn as nn
from .base import BaseNode
from .utils import _init_conv_in_graph


class ConvDepth5(BaseNode):  # simple
    def __init__(self, shape: tuple, scheme='kaiming_normal'):
        super().__init__(shape, 2*5*5*shape[0]*shape[1]*shape[2], 5*5*shape[0])
        self.conv = nn.Conv2d(shape[0], shape[0], kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=shape[0], bias=False)
        _init_conv_in_graph(self.conv, scheme)

    def forward(self, x):
        return self.conv(x)