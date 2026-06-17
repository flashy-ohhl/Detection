# Copyright (c) OpenMMLab. All rights reserved.
"""Optimizer hook that skips an iteration when the loss/grad is non-finite.

grad_clip only bounds *large finite* gradients; it cannot protect against a
NaN/Inf produced in the forward pass (e.g. rotated-IoU losses on a degenerate
box, which can occur sporadically -- especially with corruption augmentation).
Without a guard, one such batch backpropagates NaN into the weights and the
whole run is lost.

This hook checks the loss before backward and (optionally) the grads after; if
either is non-finite it zeroes the grads and skips ``optimizer.step()`` for that
iteration, so a single bad batch no longer kills training.
"""
import torch
from mmcv.runner import HOOKS, OptimizerHook


@HOOKS.register_module()
class NaNSafeOptimizerHook(OptimizerHook):
    """OptimizerHook that skips non-finite iterations.

    Args:
        grad_clip (dict, optional): same as ``OptimizerHook``.
        check_grad (bool): also verify gradients are finite after backward.
            Default: False (checking the scalar loss is usually enough and
            cheaper).
    """

    def __init__(self, grad_clip=None, check_grad=False):
        super(NaNSafeOptimizerHook, self).__init__(grad_clip=grad_clip)
        self.check_grad = check_grad
        self._skipped = 0

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        loss = runner.outputs['loss']

        if not torch.isfinite(loss):
            self._skipped += 1
            runner.logger.warning(
                f'NaNSafeOptimizerHook: non-finite loss ({loss.item()}), '
                f'skipping iter (total skipped: {self._skipped}).')
            runner.optimizer.zero_grad()
            return

        loss.backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

        if self.check_grad:
            finite = all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in runner.model.parameters())
            if not finite:
                self._skipped += 1
                runner.logger.warning(
                    'NaNSafeOptimizerHook: non-finite grad, skipping iter '
                    f'(total skipped: {self._skipped}).')
                runner.optimizer.zero_grad()
                return

        runner.optimizer.step()
