"""Runtime provenance without credentials or full process environments."""
from __future__ import annotations
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from .utils import atomic_json, code_digest, sha256
SAFE_ENV = ('ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME', 'ASCEND_OPP_PATH', 'ASCEND_RT_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONNOUSERSITE', 'HCCL_DETERMINISTIC', 'HCCL_CONNECT_TIMEOUT', 'HCCL_EXEC_TIMEOUT', 'HCCL_ASYNC_ERROR_HANDLING', 'TORCH_DEVICE_BACKEND_AUTOLOAD', 'ASCEND_LAUNCH_BLOCKING')

def command(args, timeout=15):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return dict(returncode=p.returncode, stdout=p.stdout[:32768], stderr=p.stderr[:8192])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=str(exc))

def memory_limits():
    info = {}
    meminfo = Path('/proc/meminfo')
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable', 'Cached'):
                info[key + '_bytes'] = int(value.split()[0]) * 1024
    for name in ('memory.max', 'memory.current', 'memory.events'):
        path = Path('/sys/fs/cgroup') / name
        if path.is_file():
            info['cgroup_' + name] = path.read_text().strip()
    return info

def capture(root, path, runtime=None, config_hash=None):
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get('Name')
        if name:
            packages[name] = dist.version
    report = dict(python=sys.version, executable=sys.executable, prefix=sys.prefix, platform=platform.platform(), architecture=platform.machine(), libc=platform.libc_ver(), cpu_count=os.cpu_count(), cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None, packages=dict(sorted(packages.items())), environment={k: os.getenv(k) for k in SAFE_ENV if os.getenv(k) is not None}, code_sha256=code_digest(root), config_sha256=config_hash, raw_corpus_modified=False, disk_free_gib=shutil.disk_usage(root).free / 1024 ** 3, npu_smi=command(['npu-smi', 'info']), timestamp=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    version_files = {}
    base = os.getenv('ASCEND_HOME_PATH')
    candidates = [Path('/usr/local/Ascend/driver/version.info')]
    if base:
        candidates.extend([Path(base) / 'opp/version.info', Path(base) / 'aarch64-linux/ascend_toolkit_install.info'])
    for p in candidates:
        if p.is_file():
            version_files[str(p)] = dict(sha256=sha256(p), text=p.read_text(errors='replace')[:8192])
    report['cann_files'] = version_files
    report['memory'] = memory_limits()
    if runtime:
        import torch
        report['runtime'] = runtime.metadata()
        report['cann_path_audit'] = getattr(runtime, 'installation', None)
        report['torch_build'] = torch.__config__.show()
        report['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
    atomic_json(path, report)
    return report
