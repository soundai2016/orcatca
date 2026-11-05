"""Explicit Ascend timeout, format and installation-trust policies."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import re
import stat
import warnings

OWNER_MESSAGE = re.compile(r'^Warning: The (.+) owner does not match the current owner\.$')


def communication_policy(cfg: dict, environ=None) -> dict:
    """Apply finite timeouts before importing TorchNPU or creating a process group."""
    env = os.environ if environ is None else environ
    execute = cfg.get('hccl_exec_timeout_s', 1800)
    connect = cfg.get('hccl_connect_timeout_s', 600)
    margin = cfg.get('timeout_margin_s', 300)
    watchdog = cfg['timeout_s']
    for key, value in [('exec', execute), ('connect', connect), ('margin', margin), ('watchdog', watchdog)]:
        if type(value) is not int or value <= 0:
            raise ValueError(f'{key} timeout must be a positive integer in seconds')
    if watchdog < max(execute, connect) + margin:
        raise ValueError('timeout_s must be at least max(hccl_exec_timeout_s, hccl_connect_timeout_s) + timeout_margin_s')
    values = {'HCCL_EXEC_TIMEOUT': str(execute), 'HCCL_CONNECT_TIMEOUT': str(connect),
              'HCCL_ASYNC_ERROR_HANDLING': '1', 'TORCH_DEVICE_BACKEND_AUTOLOAD': '0'}
    changed = {k: {'previous': env[k], 'configured': v} for k, v in values.items() if k in env and env[k] != v}
    env.update(values)
    return {'watchdog_timeout_s': watchdog, 'hccl_exec_timeout_s': execute,
            'hccl_connect_timeout_s': connect, 'margin_s': margin,
            'async_error_handling': True, 'overridden_environment': changed}


def _inspect(path: Path, uid: int) -> dict:
    st = path.stat()
    mode = stat.S_IMODE(st.st_mode)
    # A root-owned sticky ancestor (e.g. /tmp) cannot replace another owner's entry.
    sticky_parent = path.is_dir() and st.st_uid == 0 and bool(mode & stat.S_ISVTX)
    unsafe_write = bool(mode & 0o022) and not sticky_parent
    readable = os.access(path, os.R_OK | (os.X_OK if path.is_dir() else 0))
    return {'path': str(path), 'uid': st.st_uid, 'gid': st.st_gid,
            'mode': oct(mode), 'readable': readable,
            'trusted': st.st_uid in (0, uid) and not unsafe_write and readable}


def inspect_installation(path: str | Path, root: str | Path | None = None) -> dict:
    """Check the referenced path and ancestors, not every toolkit payload file."""
    uid = os.geteuid()
    requested = Path(path).absolute()
    try:
        resolved = requested.resolve(strict=True)
        if root is not None and not resolved.is_relative_to(Path(root).resolve(strict=True)):
            raise ValueError('Path resolves outside the selected toolkit')
        nodes = set((requested, resolved))
        nodes.update(requested.parents)
        nodes.update(resolved.parents)
        checks = [_inspect(p, uid) for p in sorted(nodes)]
        return {'requested': str(requested), 'resolved': str(resolved), 'current_uid': uid,
                'trusted': all(p['trusted'] for p in checks), 'paths': checks,
                'scope': 'Referenced path and ancestors; not a recursive payload integrity certificate'}
    except (OSError, ValueError, RuntimeError) as exc:
        return {'requested': str(requested), 'current_uid': uid, 'trusted': False, 'error': str(exc)}


def trusted_owner_notice(message, filename: str) -> dict | None:
    """Downgrade only the exact TorchNPU notice for a checked toolkit path."""
    match = OWNER_MESSAGE.fullmatch(str(message))
    source = filename.replace('\\', '/')
    home = os.getenv('ASCEND_HOME_PATH')
    if not match or not source.endswith('/torch_npu/utils/collect_env.py') or not home:
        return None
    result = inspect_installation(match.group(1), home)
    if not result['trusted']:
        return None
    return {'path': result['resolved'], 'owner_uid': Path(result['resolved']).stat().st_uid,
            'current_uid': result['current_uid'], 'permissions_checked': True,
            'original_message': str(message), 'action': 'record_as_info; no ownership or permission changes'}


def import_backend():
    from .logging import _warning, event
    home = os.getenv('ASCEND_HOME_PATH')
    if not home:
        raise RuntimeError('ASCEND_HOME_PATH is unset; source the installed CANN set_env.sh')
    trust = inspect_installation(home)
    event('cann_installation_audit', **trust)
    if not trust['trusted']:
        raise PermissionError('CANN path is unreadable or has unexpected ownership/write access; inspect the installation audit')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try:
            backend = importlib.import_module('torch_npu')
        finally:
            for item in captured:
                _warning(item.message, item.category, item.filename, item.lineno)
    return backend, trust


def _switch_value(value, name: str) -> bool | None:
    """Decode a query result without treating nonempty false strings as true."""
    if value is None:
        return None
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, bytes):
        value = value.decode('ascii')
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('enable', 'true', '1'):
            return True
        if normalized in ('disable', 'false', '0'):
            return False
    raise RuntimeError(f'Unexpected {name} query value: {value!r}')


def configure_formats(torch, cfg: dict) -> dict:
    """Set public options; an absent getter is not a rejected assignment."""
    allow = cfg.get('allow_internal_format', True)
    jit = cfg.get('jit_compile', False)
    if type(allow) is not bool or type(jit) is not bool:
        raise ValueError('allow_internal_format and jit_compile must be booleans')
    npu = torch.npu
    try:
        config = npu.config
        config.allow_internal_format = allow
    except Exception as exc:
        raise RuntimeError('Failed to set torch.npu.config.allow_internal_format') from exc

    # TorchNPU's _npuConfig can dispatch writes without storing a readable field.
    # Use the public setter, never inject a Python shadow or rely on private APIs.
    try:
        actual = _switch_value(config.allow_internal_format, 'allow_internal_format')
    except AttributeError:
        actual = None
    if actual is not None and actual != allow:
        raise RuntimeError('TorchNPU allow_internal_format readback differs from the request')

    try:
        npu.set_compile_mode(jit_compile=jit)
    except Exception as exc:
        raise RuntimeError('Failed to set torch.npu.set_compile_mode') from exc
    query = getattr(npu, 'is_jit_compile_false', None)
    actual_jit = None
    if callable(query):
        disabled = _switch_value(query(), 'is_jit_compile_false')
        actual_jit = None if disabled is None else not disabled
    if actual_jit is not None and actual_jit != jit:
        raise RuntimeError('TorchNPU jit_compile readback differs from the request')
    return {
        'allow_internal_format_requested': allow,
        'allow_internal_format': actual,
        'allow_internal_format_setter_completed': True,
        'allow_internal_format_verification': 'public_readback' if actual is not None else 'setter_only',
        'config_type': f'{type(config).__module__}.{type(config).__qualname__}',
        'jit_compile_requested': jit,
        'jit_compile': actual_jit,
        'jit_compile_setter_completed': True,
        'jit_compile_verification': 'public_readback' if actual_jit is not None else 'setter_only',
    }
