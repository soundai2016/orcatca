"""Per-process logs and deduplicated Python warnings."""
from __future__ import annotations
from collections import Counter
import os
from pathlib import Path
import sys
import warnings
from loguru import logger
_counts = Counter()
_original = warnings.showwarning

def _warning(message, category, filename, lineno, file=None, line=None):
    key = (category.__name__, str(message), filename, lineno)
    _counts[key] += 1
    if _counts[key] == 1:
        from .ascend_policy import trusted_owner_notice
        notice = trusted_owner_notice(message, filename)
        if notice is not None:
            event('cann_owner_notice', **notice)
        else:
            event('python_warning', level='WARNING', category=key[0], message=key[1], file=filename, line=lineno)

def configure(directory: Path, role='main', rank=0, console=True):
    directory.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.configure(extra={'role': role, 'rank': rank})
    stem = f'{role}.r{rank}.p{os.getpid()}'
    opts = dict(enqueue=True, backtrace=True, diagnose=False, encoding='utf-8')
    fmt = '{time:YYYY-MM-DD HH:mm:ss.SSS!UTC} | {level:<8} | r{extra[rank]} p{process.id} | {message} | {extra}'
    logger.add(directory / (stem + '.log'), format=fmt, rotation='20 MB', retention=5, compression='gz', **opts)
    logger.add(directory / (stem + '.jsonl'), serialize=True, rotation='20 MB', retention=5, compression='gz', **opts)
    if console:
        logger.add(sys.stderr, format=fmt, diagnose=False, colorize=False, enqueue=True)
    _counts.clear()
    warnings.showwarning = _warning
    return logger

def event(name, level='INFO', **values):
    logger.bind(event=name, **values).log(level, '{}', name)

def close():
    for (category, message, filename, lineno), count in _counts.items():
        if count > 1:
            event('repeated_warning', category=category, message=message, file=filename, line=lineno, count=count)
    warnings.showwarning = _original
    logger.complete()
    logger.remove()
