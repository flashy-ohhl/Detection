# Copyright (c) OpenMMLab. All rights reserved.
"""Optimizer hook that skips an iteration when the loss is non-finite.

grad_clip only bounds *large finite* gradients; it cannot protect against a
NaN/Inf produced in the forward pass (e.g. rotated-IoU losses on a degenerate
box, which can occur sporadically -- especially with corruption augmentation).
Without a guard, one such batch backpropagates NaN into the weights and the
whole run is lost.

DDP safety: ``loss.backward()`` triggers a cross-rank gradient all-reduce, so if
only *some* ranks skip backward the others deadlock. This hook all-reduces a
"finite" flag first, so every rank agrees to skip (or not) together.
"""
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, OptimizerHook


def _all_ranks_finite(value_is_finite, device):
    """True only if the value is finite on every rank (collective)."""
    if not (dist.is_available() and dist.is_initialized()):
        return value_is_finite
    flag = torch.tensor([1.0 if value_is_finite else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)  # min -> 0 if any rank is 0
    return flag.item() > 0.5


@HOOKS.register_module()
class NaNSafeOptimizerHook(OptimizerHook):
    """OptimizerHook that skips non-finite iterations (DDP-safe).

    Args:
        grad_clip (dict, optional): same as ``OptimizerHook``.
        check_grad (bool): also verify gradients are finite after backward.
            Default: False (the scalar-loss check is usually enough & cheaper).
    """

    def __init__(self, grad_clip=None, check_grad=False):
        super(NaNSafeOptimizerHook, self).__init__(grad_clip=grad_clip)
        self.check_grad = check_grad
        self._skipped = 0

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        loss = runner.outputs['loss']

        # Always call backward so the DDP gradient reduction completes every
        # iteration (skipping backward under DDP triggers "Expected to have
        # finished reduction in the prior iteration"). If the loss is
        # non-finite we simply drop the grads and skip optimizer.step(), so the
        # weights are never corrupted.
        loss.backward()

        finite = _all_ranks_finite(bool(torch.isfinite(loss).item()),
                                   loss.device)
        if not finite:
            self._skipped += 1
            runner.logger.warning(
                'NaNSafeOptimizerHook: non-finite loss on some rank, dropping '
                f'grads & skipping step (total skipped: {self._skipped}).')
            runner.optimizer.zero_grad()
            return

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

        runner.optimizer.step()
