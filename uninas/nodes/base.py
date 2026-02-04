from abc import ABC, abstractmethod
from typing import Tuple

import torch.nn as nn
import json


class BaseNode(nn.Module, ABC):
    """
    Base class for all nodes.

    All nodes:
    - are nn.Modules
    - have a defined output shape
    - know the root (input) shape
    - track FLOPs and parameter count
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        flops: int = 0,
        num_params: int = 0,
    ):
        super().__init__()
        self.shape = shape
        self.flops = flops
        self.num_params = num_params

    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the node.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def extra_repr(self) -> str:
        """
        Makes `print(module)` more informative.
        """
        return (
            f"shape={self.shape}, "
            f"flops={self.flops}, "
            f"num_params={self.num_params}"
        )

    def to_dict(self) -> dict:
        """
        Serialize the node to a JSON string containing:
        - class name
        - constructor parameters
        - shape
        """

        data = {
            'class': self.__class__.__name__,
            'shape': self.shape,
            'params': self._get_constructor_params()
        }
        return data

    def _get_constructor_params(self) -> dict:
        """
        Extract constructor parameters needed to reconstruct this node.
        Override this in child classes if extra params exist.
        """
        return {}  # default: no extra parameters