"""Explicit single-node HCCL/Gloo. NPU failure never falls back to CPU."""
from __future__ import annotations
from contextlib import nullcontext
from datetime import timedelta
import os
import random
import numpy as np
import torch
import torch.distributed as dist

class Runtime:

    def __init__(self, cfg):
        self.cfg = cfg
        self.rank = int(os.getenv('RANK', '0'))
        self.world = int(os.getenv('WORLD_SIZE', '1'))
        self.local = int(os.getenv('LOCAL_RANK', '0'))
        torch.set_num_threads(cfg['threads'])
        self.kind = cfg['device']
        if self.kind == 'npu':
            from .ascend_policy import communication_policy, import_backend, configure_formats
            from .logging import event
            self.communication = communication_policy(cfg)
            event('hccl_timeout_policy', **self.communication)
            torch_npu, self.installation = import_backend()
            if not torch.npu.is_available():
                raise RuntimeError('Requested Ascend NPU is unavailable')
            if self.local >= torch.npu.device_count():
                raise RuntimeError('LOCAL_RANK exceeds visible NPU count')
            torch.npu.set_device(self.local)
            self.formats = configure_formats(torch, cfg)
            event('npu_format_policy', **self.formats)
            self.device = torch.device('npu', self.local)
            self.npu = torch_npu
        else:
            self.device = torch.device('cpu')
        torch.use_deterministic_algorithms(cfg.get('deterministic', True))
        if self.world > 1:
            dist.init_process_group('hccl' if self.kind == 'npu' else 'gloo', timeout=timedelta(seconds=cfg['timeout_s']))

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed % 2 ** 32)
        torch.manual_seed(seed)
        if self.kind == 'npu':
            torch.npu.manual_seed_all(seed)

    def sync(self):
        if self.kind == 'npu':
            torch.npu.synchronize()

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def autocast(self):
        if self.cfg['precision'] == 'fp32':
            return nullcontext()
        return torch.autocast(device_type='npu', dtype=torch.bfloat16)

    def sum(self, value):
        t = torch.tensor(value, device=self.device, dtype=torch.float32)
        if self.world > 1:
            dist.all_reduce(t)
        return t.cpu().tolist()

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def metadata(self):
        return dict(torch=torch.__version__, torch_npu=getattr(getattr(self, 'npu', None), '__version__', None), backend='hccl' if self.kind == 'npu' else 'gloo', world=self.world, device=str(self.device), precision=self.cfg['precision'], visible_npu=torch.npu.device_count() if self.kind == 'npu' else 0, cann=os.getenv('ASCEND_HOME_PATH'), visible_setting=os.getenv('ASCEND_RT_VISIBLE_DEVICES'), communication={k: v for k, v in getattr(self, 'communication', {}).items() if k != 'overridden_environment'}, formats=getattr(self, 'formats', {}))
