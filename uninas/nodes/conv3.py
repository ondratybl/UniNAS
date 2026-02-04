import torch.nn as nn
from .base import BaseNode
from .utils import _init_conv_in_graph


class Conv3(BaseNode):  # simple
    def __init__(self, shape: tuple, scheme='kaiming_normal'):
        super().__init__(shape, shape[0]*shape[1]*shape[2]*18*shape[0], shape[0]*(shape[0]*3*3 + 1))
        self.conv = nn.Conv2d(shape[0], shape[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        _init_conv_in_graph(self.conv, scheme)

    def forward(self, x):
        return self.conv(x)