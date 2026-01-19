from __future__ import annotations

from .model import UNIModel
import torch
import torch.nn as nn
import copy
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

all_nodes = [
    (ForkMerge, {'branches_count': 2, 'fork_merge_tuple': ('chunk', 'concat')}),
    (ForkMerge, {'branches_count': 2, 'fork_merge_tuple': ('copy', 'add')}),
    (ForkMerge, {'branches_count': 2, 'fork_merge_tuple': ('copy', 'multiply')}),
    (ForkMergeAttention, {}),
    (ExpandAndReduce, {}),
    (ReduceAndExpand, {}),
    (AvgAndUpsample, {}),
    (Softmax, {}),
    (Sigmoid, {}),
    (GELU, {}),
    (Dropout, {}),
    (BatchNorm, {}),
    (LayerNorm, {}),
    (MaxPool, {}),
    (Conv1, {}),
    (Conv3, {}),
    (ConvDepth3, {}),
    (ConvDepth5, {}),
    (Mask, {}),
    (RelPosBias, {}),
]


def create_new_model(
    model_old: UNIModel,
    flops_min: int = None,
    flops_max: int = None,
    params_min: int = None,
    params_max: int = None,
    n_patience: int = 5,
    n_changes: int = 1,
    p_eliminate: float = 0.3,
) -> UNIModel | None:
    """
    Create a mutated copy of `model_old` by randomly eliminating or adding nodes,
    subject to FLOPs and parameter constraints.

    Returns:
        A new UNIModel if successful, otherwise None.
    """

    assert 0.0 <= p_eliminate <= 1.0

    cur_flops = sum(model_old.flops.values())
    cur_params = sum(model_old.num_params.values())

    # Set default bounds
    flops_min = 0 if flops_min is None else flops_min
    flops_max = float("inf") if flops_max is None else flops_max
    params_min = 0 if params_min is None else params_min
    params_max = float("inf") if params_max is None else params_max

    FLOPS_FLOOR = max(0, cur_flops - flops_min)
    FLOPS_TOP = max(0, flops_max - cur_flops)
    PARAMS_FLOOR = max(0, cur_params - params_min)
    PARAMS_TOP = max(0, params_max - cur_params)

    # Deep copy the model to mutate
    model_temp = copy.deepcopy(model_old)

    successful_changes = 0
    attempts = 0

    while successful_changes < n_changes:
        n_stages = len(model_temp.stages)
        stage_idx = torch.randint(0, n_stages, ()).item()

        blocks = model_temp.stages[stage_idx].blocks
        block_idx = torch.randint(0, len(blocks), ()).item()

        # Choose subblock
        if torch.rand(()) < 0.5:
            subblock_idx = 1
            subblock = blocks[block_idx].block1
        else:
            subblock_idx = 2
            subblock = blocks[block_idx].block2

        # Choose operation
        op_idx = torch.multinomial(
            torch.tensor([p_eliminate, 1.0 - p_eliminate]),
            num_samples=1,
        ).item()
        operation = "eliminate" if op_idx == 0 else "add"

        if operation == "eliminate":
            flops_diff, params_diff = eliminate_node(
                subblock, FLOPS_FLOOR, PARAMS_FLOOR
            )
        else:
            flops_diff, params_diff = add_node(
                subblock, FLOPS_TOP, PARAMS_TOP
            )

        # Apply successful change
        if flops_diff != 0 or params_diff != 0:
            FLOPS_FLOOR += flops_diff
            FLOPS_TOP -= flops_diff
            PARAMS_FLOOR += params_diff
            PARAMS_TOP -= params_diff

            key = f"block{stage_idx}_{block_idx}_block{subblock_idx}"
            model_temp.flops[key] += flops_diff
            model_temp.num_params[key] += params_diff

            successful_changes += 1

        attempts += 1
        if attempts > n_patience:
            return None

    return model_temp


def eliminate_node(
    seq: SequentialModule,
    flops_floor: int,
    params_floor: int,
):
    """
    Randomly remove a node from a SequentialModule while respecting constraints.
    """
    if seq.sequential is None or len(seq.sequential) == 0:
        return 0, 0

    # Collect all eligible SequentialModules
    all_seq = [
        s for s in seq.sequential.modules()
        if isinstance(s, SequentialModule)
        and s.sequential is not None
        and len(s.sequential) > 0
    ]
    all_seq.append(seq)

    seq_idx = torch.randint(0, len(all_seq), ()).item()
    target_seq = all_seq[seq_idx]

    node_idx = torch.randint(0, len(target_seq.sequential), ()).item()
    node = target_seq.sequential[node_idx]

    flops_diff = -sum(
        m.flops for m in node.modules() if hasattr(m, "flops")
    )
    params_diff = -sum(
        m.num_params for m in node.modules() if hasattr(m, "num_params")
    )

    if flops_floor + flops_diff < 0 or params_floor + params_diff < 0:
        return 0, 0

    # Remove node
    target_seq.sequential = nn.Sequential(
        *[m for i, m in enumerate(target_seq.sequential) if i != node_idx]
    )

    return flops_diff, params_diff


def add_node(
    seq: SequentialModule,
    flops_top: int,
    params_top: int,
):
    """
    Randomly insert a node into a SequentialModule while respecting constraints.
    """
    if seq.sequential is None:
        seq.sequential = nn.Sequential()
        return 0, 0

    all_seq = [
        s for s in seq.sequential.modules()
        if isinstance(s, SequentialModule)
    ]
    all_seq.append(seq)

    seq_idx = torch.randint(0, len(all_seq), ()).item()
    target_seq = all_seq[seq_idx]

    node_idx = torch.randint(0, len(target_seq.sequential) + 1, ()).item()

    params_common = {
        "shape": target_seq.shape,
        "root_shape": target_seq.root_shape,
    }

    valid_nodes = []
    for cls, params in all_nodes:
        try:
            node = cls(**params, **params_common)
        except AssertionError:
            continue

        node_flops = sum(
            m.flops for m in node.modules() if hasattr(m, "flops")
        )
        node_params = sum(
            m.num_params for m in node.modules() if hasattr(m, "num_params")
        )

        if node_flops <= flops_top and node_params <= params_top:
            valid_nodes.append((node, node_flops, node_params))

    if not valid_nodes:
        return 0, 0

    choice_idx = torch.randint(0, len(valid_nodes), ()).item()
    node_new, flops_diff, params_diff = valid_nodes[choice_idx]

    target_seq.add_to_position(node_new, node_idx)
    return flops_diff, params_diff
