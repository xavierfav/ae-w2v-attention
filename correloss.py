#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import Tensor, zeros
from torch.nn import functional

__author__ = 'Konstantinos Drossos - TAU, Stylianos I. Mimilakis - Fraunhofer IDMT'
__docformat__ = 'reStructuredText'
__all__ = ['corre_loss']


def corre_loss(x: Tensor,
               y: Tensor) \
        -> Tensor:
    """Correlation loss.

    :param x: First modality (e.g. audio). Shape must be
              batch x time1 x features. 
    :type x: torch.Tensor
    :param y: Second modality (e.g. text). Shape must be
              batch x time2 x features.
    :type y: torch.Tensor
    :returns: Correlation loss. Return shape is
              batch x batch.
    :rtype: torch.Tensor
    """
    b_size = x.size()[0]
    loss = zeros(b_size, b_size)

    if x.size()[1] < y.size()[1]:
        t_dif = y.size()[1] - x.size()[1]
       
        pad = [t_dif//2, t_dif//2]
        if divmod(t_dif, 2)[-1] != 0:
            pad[0] += 1

        _x = functional.pad(x, (0, 0, ) + tuple(pad), 'constant', 0)

    else:
        _x = x

    _x_time = _x.size()[1]
    _x_feats = _x.size()[-1]
    
    for b_row in range(b_size):
        x_ = _x[b_row:b_row + 1].permute(2, 0, 1)
        for b_col in range(b_size):
            y_ = y[b_col:b_col + 1].permute(2, 0, 1)
            tmp = functional.conv1d(x_, y_).sum(-1)
            loss[b_row, b_col] = tmp.sum(1).div(_x_time * _x_feats).neg().exp().sum()

    return loss

# EOF

