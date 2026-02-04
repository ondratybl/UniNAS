from .base import BaseNode


class Identity(BaseNode):  # fork & merge
    def __init__(self, shape: tuple):
        super().__init__(shape)

    def forward(self, x):
        return x