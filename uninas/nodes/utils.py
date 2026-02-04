import torch.nn as nn


def _init_conv_in_graph(module, scheme=''):
    assert isinstance(module, nn.Conv2d), 'Only Conv2d is supported.'
    if scheme == 'kaiming_normal':
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif scheme == 'xavier_uniform':
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif scheme == 'kaiming_normal_mlp':
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.normal_(module.bias, std=1e-6)