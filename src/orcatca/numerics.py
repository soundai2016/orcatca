"""FP32 model parity for the active backend and tensor-format policy."""
from __future__ import annotations

import copy
import math
import torch
from .model import AcousticNet


def relative_l2(actual, expected):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))


def model_parity(rt, shape, width):
    """Use synthetic full-size features; do not read training or target recordings."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6741)
        ref = AcousticNet(width, 0.0).float().eval()
        x = torch.randn(min(2, shape[0]), *shape[1:])
        target = torch.arange(len(x)) % 2
        model = copy.deepcopy(ref).to(rt.device)
        expected = ref(x)
        ref_loss = torch.nn.functional.cross_entropy(expected[:, :2], target)
        ref_loss.backward()
        actual = model(x.to(rt.device))
        loss = torch.nn.functional.cross_entropy(actual[:, :2], target.to(rt.device))
        loss.backward()
        rt.sync()
        expected_grad = torch.cat([p.grad.detach().reshape(-1) for p in ref.parameters()])
        actual_grad = torch.cat([p.grad.detach().cpu().reshape(-1) for p in model.parameters()])
        values = dict(logits_relative_l2=relative_l2(actual, expected),
                      gradients_relative_l2=relative_l2(actual_grad, expected_grad),
                      loss_relative_error=abs(float(loss.detach().cpu()) - float(ref_loss.detach())) / max(abs(float(ref_loss.detach())), 1e-12),
                      logits_max_abs=float((actual.detach().cpu() - expected.detach()).abs().max()))
        limits = dict(logits_relative_l2=0.02, gradients_relative_l2=0.03, loss_relative_error=0.01,
                      logits_max_abs=0.005 + 0.01 * float(expected.detach().abs().max()))
        passed = all(math.isfinite(v) and v <= limits[k] for k, v in values.items())
        result = dict(status='passed' if passed else 'failed', reference='CPU FP32',
                      input_shape=list(x.shape), width=width, measurements=values, limits=limits,
                      scope='Synthetic forward/backward check; not full model or dataset certification')
        del model, ref
    return result
