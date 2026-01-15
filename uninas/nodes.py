import torch
import torch.nn as nn
import math
import warnings

fork_merge_pair_tuples = (
    ('chunk', 'concat'),
    ('copy', 'add'),
    ('copy', 'multiply'),
    ('chunk', 'matmul'),
    ('convexp3', 'concat'),
    ('', '')
)

'''
Composite:
    - ForkMerge
    - ForkMergeAttention
    - ExpandAndReduce
    - ReduceAndExpand
    - AvgAndUpsample
Simple:
    - Softmax
    - Sigmoid
    - GELU
    - Dropout
    - BatchNorm
    - LayerNorm
    - MaxPool
    - Conv1
    - Conv3
    - ConvDepth3
    - ConvDepth5
    - Mask
    - RelPosBias
'''




