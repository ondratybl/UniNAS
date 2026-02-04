import torch
import torch.nn as nn
from dataclasses import dataclass
from collections import OrderedDict
from functools import partial
from typing import Union, Tuple, Callable
import json

from timm.models.maxxvit import Downsample2d, Stem, ClassifierHead, _init_conv, _init_transformer
from .nodes import (
    Add,
    AvgAndUpsample,
    BaseNode,
    BatchNorm,
    Chunk,
    Concat,
    Copy,
    Conv1,
    Conv3,
    ConvDepth3,
    ConvDepth5,
    ConvExp3,
    Dropout,
    ExpandAndReduce,
    ForkMerge,
    ForkMergeAttention,
    ForkModule,
    GELU,
    Identity,
    LayerNorm,
    LayerNorm2d,
    MatmulLeft,
    MatmulRight,
    Mask,
    MaxPool,
    MergeModule,
    Multiply,
    ReduceAndExpand,
    RelPosBias,
    SequentialModule,
    Sigmoid,
    Softmax,
    Zero
)
from .nodes.utils import _init_conv_in_graph

@dataclass
class UNIModelCfg:
    img_size: int = 224
    num_classes: int = 1000
    drop_rate: float = 0.
    embed_dim: Tuple[int, ...] = (96, 192, 384, 768)
    depths: Tuple[int, ...] = (2, 3, 5, 2)
    model_str: str = '[["E", "E"], ["E", "R", "R"], ["T", "T", "T", "T", "T"], ["E", "R"]]'
    stem_width: Union[int, Tuple[int, int]] = (32, 64)

    def to_dict(self):
        return {
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'drop_rate': self.drop_rate,
            'embed_dim': self.embed_dim,
            'depths': self.depths,
            'model_str': self.model_str,
            'stem_width': self.stem_width
        }

    def to_string(self):
        return json.dumps(self.to_dict())#, indent=2)

    @classmethod
    def from_string(cls, s: str):
        data = json.loads(s)

        return cls(
            img_size=data.get("img_size", 224),
            num_classes=data.get("num_classes", 1000),
            drop_rate=data.get("drop_rate", 0.0),
            embed_dim=tuple(data.get("embed_dim", (96, 192, 384, 768))),
            depths=tuple(data.get("depths", (2, 3, 5, 2))),
            model_str=data.get("model_str", '[["E", "E"], ["E", "R", "R"], ["T", "T", "T", "T", "T"], ["E", "R"]]'),
            stem_width=data.get("stem_width", (32, 64))
        )


def named_apply(
        fn: Callable,
        module: nn.Module, name='',
        depth_first: bool = True,
        include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def within_5_percent(a: float, b: float) -> bool:
    if a == b:
        return True
    if a == 0 or b == 0:
        return False  # Relative error not meaningful with zero
    relative_diff = abs(a - b) / max(abs(a), abs(b))
    return relative_diff <= 0.05


def get_flops(module: nn.Module, shape: Tuple[int]) -> int:
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_flops=True,
            profile_memory=False
    ) as prof:
        with torch.no_grad():
            module(torch.rand((1, ) + shape))
    return int(sum([e.flops for e in prof.key_averages() if e.flops is not None]))


def create_squeeze_and_excitation(shape: tuple, factor: int = 4) -> ForkMerge:
    assert shape[0] % factor == 0, 'Number of channels must be divisible by factor.'
    se = ForkMerge(shape, branches_count=2, fork_merge_tuple=('copy', 'multiply'))
    se.branches[1].add_to_position(AvgAndUpsample(shape))  # TODO: check
    se.branches[1].sequential[0].sequential.add_to_position(ReduceAndExpand(se.branches[1].sequential[0].sequential.shape, factor=factor))
    se.branches[1].sequential[0].sequential.sequential[0].sequential.add_to_position(GELU(se.branches[1].sequential[0].sequential.shape))
    return se


def create_depthwise_separable(shape: tuple, se_factor: int = 4, exp_factor: int = 4) -> SequentialModule:
    '''
    x = conv_pw(x)
    x = bn1(x)
    x = relu1(x)
    x = conv_dw(x)
    x = bn2(x)
    x = relu2(x)
    x = se(x)
    x = conv_pwl(x)
    x = bn3(x)
    '''
    inner_shape = (shape[0] * exp_factor, shape[1], shape[2])
    inner_branch = nn.Sequential(
        BatchNorm(inner_shape),
        GELU(inner_shape),
        ConvDepth5(inner_shape),
        BatchNorm(inner_shape), GELU(inner_shape),
        create_squeeze_and_excitation(inner_shape, se_factor)
    )
    expand_and_reduce = ExpandAndReduce(shape, inner_branch, exp_factor)
    return SequentialModule(shape, nn.Sequential(expand_and_reduce, BatchNorm(shape), ))


def create_resnet(shape: tuple) -> SequentialModule:
    return SequentialModule(shape,
                            nn.Sequential(Conv3(shape), BatchNorm(shape), GELU(shape), Conv3(shape), BatchNorm(shape), GELU(shape)))


class UNIStage(nn.Module):
    def __init__(self, stage_desc: str, in_shape: Tuple[int], channels, depth):
        super().__init__()
        #self.stage_str = stage_str
        self.in_shape = in_shape
        self.channels = channels
        self.depth = depth

        stride = 2
        blocks = []
        assert len(stage_desc) == self.depth, "Inconsistent stage specification. Number of block differs."
        for block_str in stage_desc:
            if stride == 2:
                blocks.append(UNIBlock(block_str, in_shape, channels, stride))
            else:
                blocks.append(UNIBlock(block_str, (channels, in_shape[1] // 2, in_shape[2] // 2), channels, stride))
            stride = 1
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class UNIBlock(nn.Module):
    def __init__(self, block_str: str, in_shape: Tuple[int], channels: int, stride: int = 2):
        super().__init__()
        #self.block_str = block_str
        self.in_channels = in_shape[0]
        self.channels = channels
        self.in_shape = in_shape
        self.stride = stride
        self.shape_temp = shape_temp = (channels, in_shape[1] // stride, in_shape[2] // stride)
        if block_str == 'T':
            self.subblock1 = SequentialModule(shape_temp, nn.Sequential(
                ForkMergeAttention(shape_temp), Conv1(shape_temp)
            ))
            for con in self.subblock1.sequential[0].connections:
                con.add_to_position(Softmax(con.shape))
                con.add_to_position(RelPosBias(con.shape))
            self.subblock2 = SequentialModule(shape_temp, nn.Sequential(
                ExpandAndReduce(shape_temp, nn.Sequential(
                    LayerNorm((4 * shape_temp[0], shape_temp[1], shape_temp[2])),
                    GELU(shape_temp)), scheme='kaiming_normal_mlp')))
        elif block_str == 'E':
            self.subblock1 = create_depthwise_separable(shape_temp)
            self.subblock2 = SequentialModule(shape_temp, nn.Sequential(Zero(shape_temp)))
        elif block_str == 'R':
            self.subblock1 = create_resnet(shape_temp)
            self.subblock2 = SequentialModule(shape_temp, nn.Sequential(Zero(shape_temp)))
        elif ('subblock1' in block_str) and ('subblock2' in block_str):
            self.subblock1 = node_from_string(block_str['subblock1'])
            self.subblock2 = node_from_string(block_str['subblock2'])
        else:
            NotImplementedError("Unknown block string.")

        if stride == 2:
            self.shortcut = Downsample2d(self.in_channels, self.channels, pool_type='avg2', bias=True)
            self.norm1 = nn.Sequential(OrderedDict([
                ('norm', LayerNorm2d(self.in_channels)),
                ('down', Downsample2d(self.in_channels, self.channels, pool_type='avg2')),  # TODO: changed from in_channels
                #('relu', nn.ReLU()),  # TODO: added
            ]))
        else:
            assert self.in_channels == self.channels
            self.shortcut = Identity(shape_temp)  # FIXME: check if shapes are correct
            self.norm1 = LayerNorm2d(self.in_channels)
        self.norm2 = LayerNorm2d(self.channels)

        named_apply(partial(_init_conv, scheme='kaiming_normal'), self.shortcut)
        named_apply(partial(_init_conv, scheme='kaiming_normal'), self.norm1)
        named_apply(partial(_init_conv, scheme='kaiming_normal'), self.norm2)

        self.num_params, self.flops = self.count_flops_params(in_shape)

    def forward(self, x):
        x = self.shortcut(x) + self.subblock1(self.norm1(x))  # different to original, where channels are changed within qkv computation
        return x + self.subblock2(self.norm2(x))

    def init(self):
        if self.stride == 2 and len(self.subblock1.sequential) > 0:
            first_module = self.subblock1.sequential[0]
            if isinstance(first_module, Conv1):
                self.subblock1.sequential[0].conv = nn.Conv2d(self.in_channels, first_module.conv.out_channels,
                                                           kernel_size=1, padding=0)
                _init_conv_in_graph(self.subblock1.sequential[0].conv, 'kaiming_normal')
            elif isinstance(first_module, Conv3):
                self.subblock1.sequential[0].conv = nn.Conv2d(self.in_channels, first_module.conv.out_channels, kernel_size=(3, 3), stride=(1, 1),
                                                           padding=(1, 1))
                _init_conv_in_graph(self.subblock1.sequential[0].conv, 'kaiming_normal')
            elif isinstance(first_module, ExpandAndReduce):
                self.subblock1.sequential[0].resize1 = nn.Conv2d(self.in_channels, first_module.resize1.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
                _init_conv_in_graph(self.subblock1.sequential[0].resize1, 'kaiming_normal')
            elif isinstance(first_module, ReduceAndExpand):
                self.subblock1.sequential[0].resize1 = nn.Conv2d(self.in_channels, first_module.resize1.out_channels, kernel_size=1, stride=1, padding=0,
                          bias=True)
                _init_conv_in_graph(self.subblock1.sequential[0].resize1, 'kaiming_normal')
            elif isinstance(first_module, ForkMergeAttention):
                self.subblock1.sequential[0].qkv.conv = nn.Conv2d(self.in_channels, self.subblock1.sequential[0].qkv.conv.out_channels, kernel_size=1, padding=0)
                _init_conv_in_graph(self.subblock1.sequential[0].qkv.conv, 'kaiming_normal')
            # Do nothing for separable convolution as the number of groups has to divide both in and out channels
            else:
                return 0
            self.norm1.down.expand = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1)
            print(f'Nodes connected while in_channels={self.in_channels}')
            return 1


    def count_flops_params(self, shape: Tuple[int]):
        num_params = {
            'shortcut': sum(p.numel() for p in self.shortcut.parameters()),
            'norm1': sum(p.numel() for p in self.norm1.parameters()),
            'subblock1': sum(p.numel() for p in self.subblock1.parameters()),
            'norm2': sum(p.numel() for p in self.norm2.parameters()),
            'subblock2': sum(p.numel() for p in self.subblock2.parameters()),
        }
        flops = {  # FIXME: we mix FLOPs from native and our implementation
            'shortcut': get_flops(self.shortcut, shape),
            'norm1': get_flops(self.norm1, shape),
            'subblock1': sum(p.flops for p in self.subblock1.modules() if isinstance(p, BaseNode)),
            'norm2': get_flops(self.norm2, (self.channels, int(shape[1] // self.stride), int(shape[2] // self.stride))),
            'subblock2': sum(p.flops for p in self.subblock2.modules() if isinstance(p, BaseNode)),
        }
        assert within_5_percent(flops['subblock1'], sum(p.flops for p in self.subblock1.modules() if isinstance(p, BaseNode)))
        assert within_5_percent(flops['subblock2'], sum(p.flops for p in self.subblock2.modules() if isinstance(p, BaseNode)))
        return num_params, flops

    def init_weights(self, scheme='', type='nonT'):
        if type == 'T':
            named_apply(partial(_init_transformer, scheme=scheme), self)
        else:
            named_apply(partial(_init_conv, scheme=scheme), self)


class UNIModel(nn.Module):
    def __init__(self, model_cfg: UNIModelCfg, **kwargs):
        super().__init__()
        if isinstance(model_cfg.img_size, int):
            img_size = (model_cfg.img_size, model_cfg.img_size)
        else:
            img_size = model_cfg.img_size
        self.img_size = img_size
        self.num_classes = model_cfg.num_classes
        self.num_features = model_cfg.embed_dim[-1]
        self.stem_width = model_cfg.stem_width

        self.feature_info = []

        self.stem = Stem(in_chs=3, out_chs=self.stem_width)
        self.feature_info += [dict(num_chs=self.stem.out_chs, reduction=2, module='stem')]
        feat_size = tuple([i // s for i, s in zip(img_size, (2, 2))])

        in_channels = self.stem.out_chs
        stages = []
        model_desc = json.loads(model_cfg.model_str)
        for i in range(len(model_cfg.embed_dim)):
            stages += [UNIStage(model_desc[i], (in_channels, feat_size[0], feat_size[1]), model_cfg.embed_dim[i], model_cfg.depths[i])]
            feat_size = tuple([(r - 1) // 2 + 1 for r in feat_size])
            in_channels = model_cfg.embed_dim[i]
            self.feature_info += [dict(num_chs=model_cfg.embed_dim[i], reduction=2 ** (i + 2), module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        self.norm = LayerNorm2d(self.num_features)
        self.head = ClassifierHead(self.num_features, model_cfg.num_classes, pool_type='avg', drop_rate=model_cfg.drop_rate)

        named_apply(partial(_init_conv, scheme='kaiming_normal'), self.stem)
        named_apply(partial(_init_conv, scheme='kaiming_normal'), self.head)

        self.freeze_unused_params()

        self.num_params, self.flops = self.count_flops_params()

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        x = self.head(x)
        return x

    def to_config(self) -> UNIModelCfg:
        model_desc = [[{'subblock1': block.subblock1.to_dict(), 'subblock2': block.subblock2.to_dict()} for block in stage.blocks] for stage in self.stages]

        return UNIModelCfg(
            img_size=self.img_size[0],
            num_classes=self.num_classes,
            drop_rate=getattr(self.head, "drop_rate", 0.0),
            embed_dim=tuple(stage.channels for stage in self.stages),
            depths=tuple(len(stage.blocks) for stage in self.stages),
            model_str=json.dumps(model_desc),
            stem_width=self.stem_width,
        )

    def to_string(self) -> str:
        return self.to_config().to_string()

    def count_flops_params(self):
        flops = {
            'stem': get_flops(self.stem, (self.stem.conv1.in_channels, ) + self.img_size),
            'norm': get_flops(self.norm, (len(self.norm.weight), self.img_size[0] // self.feature_info[-1]['reduction'], self.img_size[1] // self.feature_info[-1]['reduction'])),
            'head': get_flops(self.head, (len(self.norm.weight), self.img_size[0] // self.feature_info[-1]['reduction'], self.img_size[1] // self.feature_info[-1]['reduction'])),
        }
        for i, stage in enumerate(self.stages):
            for j, block in enumerate(stage.blocks):
                for key, value in block.flops.items():
                    flops[f'block{i}_{j}_{key}'] = value

        num_params = {
            'stem': sum(p.numel() for p in self.stem.parameters()),
            'norm': sum(p.numel() for p in self.norm.parameters()),
            'head': sum(p.numel() for p in self.head.parameters()),
        }
        for i, stage in enumerate(self.stages):
            for j, block in enumerate(stage.blocks):
                for key, value in block.num_params.items():
                    num_params[f'block{i}_{j}_{key}'] = value
        return num_params, flops

    def freeze_unused_params(self):

        def register_hooks():
            param_usage = {}

            for name, param in self.named_parameters():
                param_usage[name] = False

                if param.requires_grad:
                    def make_hook(name=name):
                        return lambda grad: param_usage.__setitem__(name, True)

                    param.register_hook(make_hook(name))

            return param_usage

        # Get unused params
        self.train()
        param_usage = register_hooks()
        output = self(torch.rand(1, 3, self.img_size[0], self.img_size[1]))
        loss = nn.CrossEntropyLoss()(output, torch.rand(1, output.shape[1]))
        loss.backward()
        unused_params = [name for name, used in param_usage.items() if not used]
        print(f"Unused parameters will be freezed: {unused_params}")

        # Freeze unused params
        for name, param in self.named_parameters():
            if name in unused_params:
                param.requires_grad = False

# Registry of all node classes
NODE_CLASSES = {
    'BatchNorm': BatchNorm,
    'ForkMerge': ForkMerge,
    'ForkMergeAttention': ForkMergeAttention,
    'SequentialModule': SequentialModule,
    'MergeModule': MergeModule,
    'ForkModule': ForkModule,
    'Concat': Concat,
    'Add': Add,
    'Multiply': Multiply,
    'Chunk': Chunk,
    'Conv1': Conv1,
    'Conv3': Conv3,
    'ConvDepth3': ConvDepth3,
    'ConvDepth5': ConvDepth5,
    'Copy': Copy,
    'GELU': GELU,
    'LayerNorm': LayerNorm,
    'MatmulLeft': MatmulLeft,
    'MatmulRight': MatmulRight,
    'MaxPool': MaxPool,
    'ConvExp3': ConvExp3,
    'ExpandAndReduce': ExpandAndReduce,
    'ReduceAndExpand': ReduceAndExpand,
    'AvgAndUpsample': AvgAndUpsample,
    'RelPosBias': RelPosBias,
    'Softmax': Softmax,
    'Zero': Zero
}


def node_from_string(data: dict):
    """
    Reconstruct any BaseNode-derived node from a JSON string.
    Works recursively for nested modules.
    """
    class_name = data['class']
    shape = tuple(data['shape'])
    params = data.get('params', {})

    node_class = NODE_CLASSES.get(class_name)
    if node_class is None:
        raise ValueError(f"Unknown node class {class_name}")

    # ---------------------------
    # Handle special node classes
    # ---------------------------
    if class_name == 'ForkMerge':
        branches = [node_from_string(b_str) for b_str in params['branches']]
        node = node_class(shape, params['branches_count'], tuple(params['fork_merge_tuple']))
        node.branches = nn.ModuleList(branches)
        return node

    elif class_name == 'ForkMergeAttention':
        node = node_class(shape)
        node.branches = nn.ModuleList([node_from_string(b) for b in params['branches']])
        node.matmullefts = nn.ModuleList([node_from_string(m) for m in params['matmullefts']])
        node.matmulrights = nn.ModuleList([node_from_string(m) for m in params['matmulrights']])
        node.connections = nn.ModuleList([node_from_string(c) for c in params['connections']])
        node.post_branches = nn.ModuleList([node_from_string(p) for p in params['post_branches']])
        return node

    elif class_name == 'SequentialModule':
        modules = [node_from_string(m_str) for m_str in params.get('modules', [])]
        seq = nn.Sequential(*modules)
        return node_class(shape, seq)

    elif class_name == 'MergeModule':
        func_merge_str = params['func_merge_str']
        node = node_class(shape, func_merge_str)
        if 'process' in params:
            node.process = node_from_string(params['process'])
        return node

    elif class_name == 'ForkModule':
        func_fork_str = params['func_fork_str']
        if 'sequential' in params:
            sequential = node_from_string(params['sequential'])
            node = node_class(shape, len(sequential.sequential), func_fork_str)
            node.fork = sequential
        else:
            node = node_class(shape, 2, func_fork_str)
        return node

    elif class_name in ('Concat', 'Add'):
        return node_class()

    elif class_name == 'Multiply':
        return node_class(shape)

    elif class_name in ('Chunk', 'Copy'):
        return node_class(params['branches_count'])

    elif class_name in ('MatmulLeft', 'MatmulRight'):
        return node_class(shape)

    elif class_name == 'ConvExp3':
        return node_class(shape, params.get('scheme', 'kaiming_normal'))

    elif class_name in ('ExpandAndReduce', 'ReduceAndExpand', 'AvgAndUpsample'):
        sequential = node_from_string(params['sequential'])
        factor = params.get('factor', 4)
        scheme = params.get('scheme', 'kaiming_normal')
        if class_name == 'ExpandAndReduce':
            return node_class(shape, sequential.sequential, factor=factor, scheme=scheme)
        elif class_name == 'ReduceAndExpand':
            return node_class(shape, sequential.sequential, factor=factor, scheme=scheme)
        else:  # AvgAndUpsample
            return node_class(shape, sequential.sequential)

    else:
        return node_class(shape)

