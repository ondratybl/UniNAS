import torch
import torch.nn as nn
from dataclasses import dataclass
from collections import OrderedDict
from functools import partial
from typing import Union, Tuple, Callable

from timm.models.maxxvit import Downsample2d, Stem, ClassifierHead, _init_conv, _init_transformer
from .nodes import (
    AvgAndUpsample,
    BaseNode,
    BatchNorm,
    Conv1,
    Conv3,
    ConvDepth3,
    ConvDepth5,
    Dropout,
    ExpandAndReduce,
    ForkMerge,
    ForkMergeAttention,
    GELU,
    Identity,
    LayerNorm,
    LayerNorm2d,
    Mask,
    MaxPool,
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
    model_str: str = 'T-T/T-T-T/T-T-T-T-T/T-T'
    stem_width: Union[int, Tuple[int, int]] = (32, 64)
    weight_init: str = 'vit_eff'


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


def create_attention(shape: tuple, root_shape: tuple, num_heads: int) -> ForkMerge:
    assert shape[0] % num_heads == 0, 'Number of channels must be divisible by number of heads.'
    assert num_heads in (3, 6, 12, 24), 'Number of heads must be either 3 or 6 or 12 or 24.'

    def create_tree_for_attention(current_node, current_depth, max_depth):
        if current_depth == max_depth:
            # Leaf node, assign a value
            current_node.add_to_position(ForkMerge(current_node.shape, root_shape, branches_count=3, fork_merge_tuple=('chunk', 'matmul')))
            current_node.sequential[0].func_merge.process.add_to_position(Softmax((1, root_shape[1]**2, root_shape[2]**2), root_shape))
            current_node.sequential[0].func_merge.process.add_to_position(RelPosBias(current_node.sequential[0].func_merge.process.shape, root_shape))
        else:
            current_node.add_to_position(
                ForkMerge(current_node.shape, root_shape, branches_count=2, fork_merge_tuple=('chunk', 'concat'))
            )
            for child_node in current_node.sequential[0].branches:
                create_tree_for_attention(child_node, current_depth + 1, max_depth)

    tree_depth = int(torch.log2(torch.tensor(num_heads // 3)).item())
    attn = ForkMerge(shape, root_shape, branches_count=3, fork_merge_tuple=('convexp3', 'concat'))
    for node in attn.branches:
        create_tree_for_attention(node, 0, tree_depth)

    return attn


def create_squeeze_and_excitation(shape: tuple, root_shape: tuple, factor: int = 4) -> ForkMerge:
    assert shape[0] % factor == 0, 'Number of channels must be divisible by factor.'
    se = ForkMerge(shape, root_shape, branches_count=2, fork_merge_tuple=('copy', 'multiply'))
    se.branches[1].add_to_position(AvgAndUpsample(shape, root_shape))  # TODO: check
    se.branches[1].sequential[0].sequential.add_to_position(ReduceAndExpand(se.branches[1].sequential[0].sequential.shape, root_shape, factor=factor))
    se.branches[1].sequential[0].sequential.sequential[0].sequential.add_to_position(GELU(se.branches[1].sequential[0].sequential.shape, root_shape))
    return se


def create_depthwise_separable(shape: tuple, root_shape: tuple, se_factor: int = 4, exp_factor: int = 4) -> SequentialModule:
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
        BatchNorm(inner_shape, root_shape),
        GELU(inner_shape, root_shape),
        ConvDepth5(inner_shape, root_shape),
        BatchNorm(inner_shape, root_shape), GELU(inner_shape, root_shape),
        create_squeeze_and_excitation(inner_shape, root_shape, se_factor)
    )
    expand_and_reduce = ExpandAndReduce(shape, root_shape, inner_branch, exp_factor)
    return SequentialModule(shape, root_shape, nn.Sequential(expand_and_reduce, BatchNorm(shape, root_shape), ))


def create_resnet(shape: tuple, root_shape: tuple) -> SequentialModule:
    return SequentialModule(shape, root_shape,
                            nn.Sequential(Conv3(shape, root_shape), BatchNorm(shape, root_shape), GELU(shape, root_shape), Conv3(shape, root_shape), BatchNorm(shape, root_shape), GELU(shape, root_shape)))


class UNIStage(nn.Module):
    def __init__(self, stage_str: str, in_shape: Tuple[int], channels, depth):
        super().__init__()
        self.stage_str = stage_str
        self.in_shape = in_shape
        self.channels = channels
        self.depth = depth

        stride = 2
        blocks = []
        block_str_tuple = tuple(self.stage_str.split("-"))
        assert len(block_str_tuple) == self.depth, "Inconsistent stage specification. Number of block differs."
        for block_str in block_str_tuple:
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
        self.block_str = block_str
        self.in_channels = in_shape[0]
        self.channels = channels
        self.in_shape = in_shape
        self.stride = stride
        self.shape_temp = shape_temp = (channels, in_shape[1] // stride, in_shape[2] // stride)
        if self.block_str == 'T':
            self.block1 = SequentialModule(shape_temp, shape_temp, nn.Sequential(
                ForkMergeAttention(shape_temp, shape_temp), Conv1(shape_temp, shape_temp)
            ))
            for con in self.block1.sequential[0].connections:
                con.add_to_position(Softmax(con.shape, shape_temp))
                con.add_to_position(RelPosBias(con.shape, shape_temp))
            self.block2 = SequentialModule(shape_temp, shape_temp, nn.Sequential(
                ExpandAndReduce(shape_temp, shape_temp, nn.Sequential(
                    LayerNorm((4 * shape_temp[0], shape_temp[1], shape_temp[2]), shape_temp),
                    GELU(shape_temp, shape_temp)), scheme='kaiming_normal_mlp')))
        elif self.block_str == 'E':
            self.block1 = create_depthwise_separable(shape_temp, shape_temp)
            self.block2 = SequentialModule(shape_temp, shape_temp, nn.Sequential(Zero(shape_temp, shape_temp)))
        elif self.block_str == 'R':
            self.block1 = create_resnet(shape_temp, shape_temp)
            self.block2 = SequentialModule(shape_temp, shape_temp, nn.Sequential(Zero(shape_temp, shape_temp)))
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
            self.shortcut = Identity(shape_temp, shape_temp)  # FIXME: check if shapes are correct
            self.norm1 = LayerNorm2d(self.in_channels)
        self.norm2 = LayerNorm2d(self.channels)

        named_apply(partial(_init_conv, scheme='kaiming_normal'), self.shortcut)
        named_apply(partial(_init_conv, scheme='kaiming_normal'), self.norm1)
        named_apply(partial(_init_conv, scheme='kaiming_normal'), self.norm2)

        self.num_params, self.flops = self.count_flops_params(in_shape)

    def forward(self, x):
        x = self.shortcut(x) + self.block1(self.norm1(x))  # different to original, where channels are changed within qkv computation
        return x + self.block2(self.norm2(x))

    def init(self):
        if self.stride == 2 and len(self.block1.sequential) > 0:
            first_module = self.block1.sequential[0]
            if isinstance(first_module, Conv1):
                self.block1.sequential[0].conv = nn.Conv2d(self.in_channels, first_module.conv.out_channels,
                                                           kernel_size=1, padding=0)
                _init_conv_in_graph(self.block1.sequential[0].conv, 'kaiming_normal')
            elif isinstance(first_module, Conv3):
                self.block1.sequential[0].conv = nn.Conv2d(self.in_channels, first_module.conv.out_channels, kernel_size=(3, 3), stride=(1, 1),
                                                           padding=(1, 1))
                _init_conv_in_graph(self.block1.sequential[0].conv, 'kaiming_normal')
            elif isinstance(first_module, ExpandAndReduce):
                self.block1.sequential[0].resize1 = nn.Conv2d(self.in_channels, first_module.resize1.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
                _init_conv_in_graph(self.block1.sequential[0].resize1, 'kaiming_normal')
            elif isinstance(first_module, ReduceAndExpand):
                self.block1.sequential[0].resize1 = nn.Conv2d(self.in_channels, first_module.resize1.out_channels, kernel_size=1, stride=1, padding=0,
                          bias=True)
                _init_conv_in_graph(self.block1.sequential[0].resize1, 'kaiming_normal')
            elif isinstance(first_module, ForkMergeAttention):
                self.block1.sequential[0].qkv.conv = nn.Conv2d(self.in_channels, self.block1.sequential[0].qkv.conv.out_channels, kernel_size=1, padding=0)
                _init_conv_in_graph(self.block1.sequential[0].qkv.conv, 'kaiming_normal')
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
            'block1': sum(p.numel() for p in self.block1.parameters()),
            'norm2': sum(p.numel() for p in self.norm2.parameters()),
            'block2': sum(p.numel() for p in self.block2.parameters()),
        }
        flops = {  # FIXME: we mix FLOPs from native and our implementation
            'shortcut': get_flops(self.shortcut, shape),
            'norm1': get_flops(self.norm1, shape),
            'block1': sum(p.flops for p in self.block1.modules() if isinstance(p, BaseNode)),
            'norm2': get_flops(self.norm2, (self.channels, int(shape[1] // self.stride), int(shape[2] // self.stride))),
            'block2': sum(p.flops for p in self.block2.modules() if isinstance(p, BaseNode)),
        }
        assert within_5_percent(flops['block1'], sum(p.flops for p in self.block1.modules() if isinstance(p, BaseNode)))
        assert within_5_percent(flops['block2'], sum(p.flops for p in self.block2.modules() if isinstance(p, BaseNode)))
        return num_params, flops

    def init_weights(self, scheme=''):
        if self.block_str == 'T':
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
        self.num_features = self.embed_dim = model_cfg.embed_dim[-1]

        self.feature_info = []

        self.stem = Stem(in_chs=3, out_chs=model_cfg.stem_width)
        self.feature_info += [dict(num_chs=self.stem.out_chs, reduction=2, module='stem')]
        feat_size = tuple([i // s for i, s in zip(img_size, (2, 2))])

        in_channels = self.stem.out_chs
        stages = []
        model_str_tuple = tuple(model_cfg.model_str.split('/'))
        for i in range(len(model_cfg.embed_dim)):
            stages += [UNIStage(model_str_tuple[i], (in_channels, feat_size[0], feat_size[1]), model_cfg.embed_dim[i], model_cfg.depths[i])]
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