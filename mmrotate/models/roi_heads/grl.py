# Copyright (c) OpenMMLab. All rights reserved.
"""Gradient Reversal Layer (DANN, Ganin & Lempitsky 2015).

Forward is identity; backward multiplies the gradient by ``-alpha``.  Used to
make an embedding *fail* an auxiliary prediction task: an adversary tries to
predict a factor (e.g. corruption id from F_disc), and the GRL pushes the
embedding to remove that factor.
"""
import math

from torch.autograd import Function


class _GradientReversal(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    """Apply gradient reversal with strength ``alpha``."""
    return _GradientReversal.apply(x, alpha)


def dann_alpha(step, warmup_iters, alpha_max=1.0, gamma=10.0):
    """DANN schedule: ramp alpha from 0 to ``alpha_max`` over ``warmup_iters``.

    alpha(p) = alpha_max * (2 / (1 + exp(-gamma * p)) - 1),  p = min(1, step/W)
    """
    if warmup_iters is None or warmup_iters <= 0:
        return alpha_max
    p = min(1.0, float(step) / float(warmup_iters))
    return alpha_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
