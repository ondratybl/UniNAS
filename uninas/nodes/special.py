import torch.nn as nn
from torch import Tensor, cat, chunk, sqrt, tensor
from .base import BaseNode
from .utils import _init_conv_in_graph
from .identity import Identity
import json

fork_merge_pair_tuples = (
    ('chunk', 'concat'),
    ('copy', 'add'),
    ('copy', 'multiply'),
    ('chunk', 'matmul'),
    ('convexp3', 'concat'),
    ('', '')
)


class ForkMerge(BaseNode):  # composite
    def __init__(self, shape: tuple, branches_count: int, fork_merge_tuple: tuple[str, str]):
        super().__init__(shape)
        func_fork_str, func_merge_str = fork_merge_tuple
        assert fork_merge_tuple in fork_merge_pair_tuples, 'Unsupported fork_merge_tuple'
        assert branches_count in (1, 2, 3), 'Up to three branches allowed.'
        assert branches_count == 3 if func_merge_str in ('convexp3', 'matmul') else True, 'Can be only combined with three branches.'
        assert branches_count in (2, 3) if func_merge_str == 'multiply' else True, 'Can only multiply 2 or 3 branches.'
        assert shape[0] % branches_count == 0 if func_fork_str == 'chunk' else True, 'Must be divisible for chunk.'
        self.inner_shape = (shape[0] // branches_count, shape[1], shape[2]) if func_fork_str == 'chunk' else shape
        self.func_merge = MergeModule(shape, func_merge_str)
        self.func_fork = ForkModule(shape, branches_count, func_fork_str)
        self.fork_merge_tuple = fork_merge_tuple
        self.branches = nn.ModuleList([SequentialModule(self.inner_shape) for _ in range(branches_count)])  # branch keeps dimension

    def forward(self, x):
        x = self.func_fork(x)
        if isinstance(x, Tensor):
            x = (x,)
        x = tuple([module(x[i]) for i, module in enumerate(self.branches)])
        x = self.func_merge(x)
        return x

    def _get_constructor_params(self):
        return {
            'branches_count': len(self.branches),
            'fork_merge_tuple': self.fork_merge_tuple,
            'branches': [b.to_string() for b in self.branches]
        }


class ForkMergeAttention(BaseNode):  # very special
    def __init__(self, shape: tuple, head_dim: int = 32):
        super().__init__(shape)
        assert shape[0] % head_dim == 0, 'Number of channels must be divisible by channel size.'
        self.num_heads = shape[0] // head_dim
        assert self.num_heads in (3, 6, 12, 24)
        self.qkv = ConvExp3(shape, scheme='xavier_uniform')
        self.chunk = Chunk(3*self.num_heads)
        self.branches = nn.ModuleList([SequentialModule((head_dim, shape[1], shape[2])) for _ in range(3*self.num_heads)])
        self.matmullefts = nn.ModuleList([MatmulLeft((head_dim, shape[1], shape[2])) for _ in range(self.num_heads)])
        self.matmulrights = nn.ModuleList([MatmulRight((head_dim, shape[1], shape[2])) for _ in range(self.num_heads)])
        self.connections = nn.ModuleList([SequentialModule((1, shape[1]*shape[2], shape[1]*shape[2])) for _ in range(self.num_heads)])
        self.post_branches = nn.ModuleList([SequentialModule((head_dim, shape[1], shape[2])) for _ in range(self.num_heads)])
        self.concat = Concat()

    def forward(self, x):
        x = self.qkv(x)
        x = self.chunk(x)
        x = tuple([self.branches[i](x[i]) for i in range(3*self.num_heads)])
        v = tuple(x[3*i + 2] for i in range(self.num_heads))
        x = tuple(self.matmullefts[i](x[3*i:3*i+2]) for i in range(self.num_heads))
        x = tuple(self.connections[i](x[i]) for i in range(self.num_heads))
        x = tuple([self.matmulrights[i](tuple([x[i], v[i]])) for i in range(self.num_heads)])
        x = tuple([self.post_branches[i](x[i]) for i in range(self.num_heads)])
        x = self.concat(x)
        return x

    def _get_constructor_params(self):
        return {
            'branches': [b.to_string() for b in self.branches],
            'matmullefts': [m.to_string() for m in self.matmullefts],
            'matmulrights': [m.to_string() for m in self.matmulrights],
            'connections': [c.to_string() for c in self.connections],
            'post_branches': [p.to_string() for p in self.post_branches]
        }


class SequentialModule(BaseNode):  # special
    def __init__(self, shape: tuple, sequential: nn.Sequential = None):
        super().__init__(shape)
        self.sequential = sequential if sequential is not None else nn.Sequential()

    def forward(self, x):
        return self.sequential(x)

    def add_to_position(self, node: nn.Module, idx: int = 0):
        modules = list(self.sequential.children())  # Extract modules without names

        if not modules:
            # Sequential is empty, ignore idx and add the node directly
            self.sequential = nn.Sequential(node)
        else:
            if not (0 <= idx <= len(modules)):
                raise IndexError(f"Index {idx} out of range for modules of length {len(modules)}")
            modules.insert(idx, node)
            self.sequential = nn.Sequential(*modules)

    def _get_constructor_params(self):
        return {
            'modules': [m.to_string() for m in self.sequential]
        }


class MergeModule(BaseNode):  # special
    def __init__(self, shape: tuple, func_merge_str):
        super().__init__(shape)
        if func_merge_str in ('', 'concat'):  # covers also single branch
            self.merge = Concat()
        elif func_merge_str == 'add':
            self.merge = Add(shape)
        elif func_merge_str == 'multiply':
            self.merge = Multiply(shape)
        elif func_merge_str == 'matmul':
            self.merge = MatmulLeft(shape)
            #self.process = ForkMerge((self.merge.num_heads, shape[1]**2, shape[2]**2), 1, ('', ''))
            self.process = SequentialModule((self.merge.num_heads, shape[1]**2, shape[2]**2))
            self.merge2 = MatmulRight(shape)
        else:
            raise NotImplementedError('Unknown merge function.')

    def forward(self, x):
        if hasattr(self, 'process'):
            return self.merge2((self.process(self.merge(x[0:2])), x[2]))
        else:
            return self.merge(x)

    def _get_constructor_params(self):
        params = {'func_merge_str': getattr(self.merge, '__class__', type(self.merge)).__name__.lower()}
        if hasattr(self, 'process'):
            params['process'] = self.process.to_string()
        return params


class ForkModule(BaseNode):  # special
    def __init__(self, shape: tuple, branches_count, func_fork_str):
        super().__init__(shape)
        if func_fork_str == '':
            self.fork = Identity()
        elif func_fork_str == 'chunk':
            self.fork = Chunk(branches_count)
        elif func_fork_str == 'copy':
            self.fork = Copy(branches_count)
        elif func_fork_str == 'convexp3':
            self.fork = SequentialModule(shape, nn.Sequential(ConvExp3(shape, scheme='xavier_uniform'), Chunk(branches_count)))
        else:
            raise NotImplementedError('Unknown fork function.')

    def forward(self, x):
        return self.fork(x)

    def _get_constructor_params(self):
        fork_type = type(self.fork).__name__.lower()
        params = {'func_fork_str': fork_type}
        # if fork is SequentialModule (like convexp3 + chunk), serialize inner sequential
        if isinstance(self.fork, SequentialModule):
            params['sequential'] = self.fork.to_string()
        return params


class Concat(BaseNode):  # merge
    def __init__(self):
        super().__init__(())

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError("Input must be a list or tuple of tensors.")
        return cat(x, dim=1)

    def _get_constructor_params(self):
        return {}


class Add(BaseNode):  # merge
    def __init__(self, shape: tuple):
        super().__init__(shape, shape[0]*shape[1]*shape[2], 0)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) not in (1, 2, 3):
            raise TypeError("Input must be a list or tuple of 2 tensors.")
        return sum(x)

    def _get_constructor_params(self):
        return {}


class Multiply(BaseNode):  # merge
    def __init__(self, shape: tuple):
        super().__init__(shape, 4*shape[0]*shape[1]*shape[2], 0)  # sigmoid ~ 3 FLOPs per entry
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if not (isinstance(x, (list, tuple)) and len(x) == 2):
            raise TypeError("Input must be a list or tuple of 2 tensors.")
        return x[0] * self.sigmoid(x[1])

    def _get_constructor_params(self):
        return {}


class Chunk(BaseNode):  # fork
    def __init__(self, branches_count: int = 2):
        super().__init__(())
        self.branches_count = branches_count

    def forward(self, x):
        return chunk(x, chunks=self.branches_count, dim=1)

    def _get_constructor_params(self):
        return {'branches_count': self.branches_count}


class Copy(BaseNode):  # fork
    def __init__(self, branches_count: int = 2):
        super().__init__(())
        self.branches_count = branches_count

    def forward(self, x):
        return self.branches_count*(x,)

    def _get_constructor_params(self):
        return {'branches_count': self.branches_count}


class MatmulRight(BaseNode):  # fork
    def __init__(self, shape: tuple):
        super().__init__(shape, 2*32*(shape[1]**2*shape[2]**2), 0)  # FIXME: assumes dim_head=32
        self.num_heads = 1
        self.shape = shape  # the shape of V should be passed

    def forward(self, x):
        # returns (\*, C, H, W), x[0] should be from MatmulLeft, x[1] should be (\*, C, H, W)
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("Input must be a list or tuple of 2 tensors.")
        B, C, H, W = x[1].shape[0], x[1].shape[1], x[1].shape[2], x[1].shape[3]
        return ((x[0] @ x[1].flatten(2).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(-2,
                                                                                                       -1))).transpose(-2, -1).view(B, C, H, W)

    def _get_constructor_params(self):
        return {}


class MatmulLeft(BaseNode):  # fork
    def __init__(self, shape: tuple):
        super().__init__(shape, 2*32*(shape[1]**2*shape[2]**2), 0)  # FIXME: assumes dim_head=32
        self.num_heads = 1
        self.shape = (self.num_heads, shape[1] * shape[1], shape[2] * shape[2])

    def forward(self, x):
        # returns (\*, num_heads, H\*H, W\*W)
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("Input must be a list or tuple of 2 tensors.")
        B, C = x[1].shape[0], x[1].shape[1]
        return (x[0].flatten(2).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(-2, -1) @ x[1].flatten(
            2).view(B, self.num_heads, int(C / self.num_heads), -1)) / sqrt(tensor(C / self.num_heads))

    def _get_constructor_params(self):
        return {}


class ConvExp3(BaseNode):  # fork
    def __init__(self, shape: tuple, scheme='kaiming_normal'):
        super().__init__(shape, 2*shape[0]*3*shape[0]*shape[1]*shape[2], shape[0]*3*shape[0]*1*1+3*shape[0])
        self.conv = nn.Conv2d(shape[0], 3*shape[0], kernel_size=1, padding=0)
        _init_conv_in_graph(self.conv, scheme)

    def forward(self, x):
        return self.conv(x)

    def _get_constructor_params(self):
        return {}


class ExpandAndReduce(BaseNode):  # composite
    def __init__(self, shape: tuple, sequential: nn.Sequential or None = None, factor: int = 4, scheme='kaiming_normal'):
        super().__init__(shape, 2*2*shape[0]*4*shape[0]*shape[1]*shape[2], (8*shape[0] + 5) * shape[0])
        self.resize1 = nn.Conv2d(shape[0], factor*shape[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.sequential = SequentialModule((factor*shape[0], shape[1], shape[2]), sequential)
        self.resize2 = nn.Conv2d(factor*shape[0], shape[0], kernel_size=1, stride=1, padding=0, bias=True)
        _init_conv_in_graph(self.resize1, scheme=scheme)
        _init_conv_in_graph(self.resize2, scheme=scheme)

    def forward(self, x):
        return self.resize2(self.sequential(self.resize1(x)))

    def _get_constructor_params(self):
        return {
            'sequential': self.sequential.to_string(),
            'factor': self.resize1.out_channels // self.resize1.in_channels,
            'scheme': 'kaiming_normal'
        }


class ReduceAndExpand(BaseNode):  # composite
    def __init__(self, shape: tuple, sequential: nn.Sequential or None = None, factor: int = 4, scheme='kaiming_normal'):
        super().__init__(shape, 4 * shape[0] * (shape[0] // factor) * shape[1] * shape[2],
                         shape[0] // factor * (2 * shape[0] + factor + 1))
        assert shape[0] % factor == 0, 'Channels have to be divisible by reduce factor.'
        self.resize1 = nn.Conv2d(shape[0], shape[0] // factor, kernel_size=1, stride=1, padding=0, bias=True)  # TODO: change to node op
        self.sequential = SequentialModule((shape[0] // factor, shape[1], shape[2]), sequential)
        self.resize2 = nn.Conv2d(shape[0] // factor, shape[0], kernel_size=1, stride=1, padding=0, bias=True)  # TODO: change to node op
        _init_conv_in_graph(self.resize1, scheme=scheme)
        _init_conv_in_graph(self.resize2, scheme=scheme)

    def forward(self, x):
        return self.resize2(self.sequential(self.resize1(x)))

    def _get_constructor_params(self):
        return {
            'sequential': self.sequential.to_string(),
            'factor': self.resize1.in_channels // self.resize1.out_channels,
            'scheme': 'kaiming_normal'
        }


class AvgAndUpsample(BaseNode):  # composite
    def __init__(self, shape: tuple, sequential: nn.Sequential or None = None):
        super().__init__(shape, shape[0]*shape[1]*shape[2], 0)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.sequential = SequentialModule((shape[0], 1, 1), sequential)
        self.upsample = nn.Upsample(size=(shape[1], shape[2]))

    def forward(self, x):
        return self.upsample(self.sequential(self.avg(x)))

    def _get_constructor_params(self):
        return {
            'sequential': self.sequential.to_string()
        }