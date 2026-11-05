"""Bounded structural checks. Read headers/chunk tables, never repair audio.

For RIFF/WAVE, the enclosing length and data geometry must describe the complete
physical file. A decoder accepting only a prefix is not sufficient evidence.
This does not verify the entire decoded waveform or the remote source object.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import struct
VALIDATION_REVISION = 200
W64_RIFF = bytes.fromhex('726966662e91cf11a5d628db04c10000')

class AudioValidationError(ValueError):

    def __init__(self, code: str, evidence: dict):
        self.code = code
        self.evidence = dict(evidence, status='rejected', reason_code=code)
        super().__init__(f"{code}: declared={evidence.get('declared_total_bytes')}, physical={evidence.get('physical_bytes')}")

def _reject(code: str, result: dict):
    raise AudioValidationError(code, result)

def inspect_container(path: str | Path, max_chunks: int=65536) -> dict:
    """Seek over payloads, check container boundaries and uncompressed PCM sizes."""
    path = Path(path)
    before = path.stat()
    total = before.st_size
    result = dict(revision=VALIDATION_REVISION, physical_bytes=total, status='header_only', waveform_fully_decoded=False, raw_file_modified=False)
    with path.open('rb') as f:
        head = f.read(40)
        if len(head) < 12:
            _reject('audio_prefix_too_short', result)
        magic = head[:4]
        if magic in (b'RIFF', b'RIFX', b'RF64', b'BW64'):
            if head[8:12] != b'WAVE':
                _reject('riff_not_wave', result)
            endian = '>' if magic == b'RIFX' else '<'
            result['container'] = magic.decode() + '_WAVE'
            declared = struct.unpack_from(endian + 'I', head, 4)[0] + 8
            data64 = samples64 = None
            size_table = {}
            if magic in (b'RF64', b'BW64'):
                if declared != 4294967295 + 8 or head[12:16] != b'ds64':
                    _reject('rf64_missing_size_extension', result)
                ds_size = struct.unpack_from('<I', head, 16)[0]
                if not 28 <= ds_size <= 1024 * 1024 or 20 + ds_size > total:
                    _reject('rf64_invalid_size_extension', result)
                f.seek(20)
                data = f.read(ds_size)
                riff64, data64, samples64, nt = struct.unpack_from('<QQQI', data)
                if ds_size != 28 + 12 * nt or nt > 65536:
                    _reject('rf64_invalid_size_table', result)
                for i in range(nt):
                    key, value = struct.unpack_from('<4sQ', data, 28 + 12 * i)
                    if key in size_table:
                        _reject('rf64_ambiguous_size_table', result)
                    size_table[key] = value
                declared = riff64 + 8
            result['declared_total_bytes'] = declared
            if declared != total:
                if declared > 0:
                    q, rem = divmod(total, declared)
                    result.update(physical_size_multiple=q, physical_size_remainder=rem, repeated_container_size_hint=q > 1 and rem == 0)
                _reject('container_length_mismatch', result)
            offset = 12
            fmt = None
            data_chunks = []
            count = 0
            while offset < total:
                count += 1
                if count > max_chunks:
                    _reject('container_chunk_limit_exceeded', result)
                f.seek(offset)
                chunk = f.read(8)
                if len(chunk) != 8:
                    _reject('truncated_chunk_header', result)
                cid, size = struct.unpack(endian + '4sI', chunk)
                if size == 4294967295 and magic in (b'RF64', b'BW64'):
                    size = data64 if cid == b'data' and (not data_chunks) else size_table.pop(cid, None)
                    if size is None:
                        _reject('rf64_unknown_chunk_length', result)
                end = offset + 8 + size
                padded_end = end + size % 2
                if end > total or padded_end > total:
                    _reject('chunk_exceeds_container', result)
                if cid == b'fmt ':
                    if fmt is not None or size < 16:
                        _reject('invalid_or_multiple_fmt_chunk', result)
                    fmt = f.read(min(size, 64))
                elif cid == b'data':
                    data_chunks.append(dict(offset=offset + 8, bytes=size))
                offset = padded_end
            if fmt is None or len(data_chunks) != 1:
                _reject('missing_or_multiple_audio_data_chunks', result)
            tag, channels, rate, byte_rate, align, bits = struct.unpack_from(endian + 'HHIIHH', fmt)
            if tag == 65534:
                if len(fmt) < 40 or struct.unpack_from(endian + 'H', fmt, 16)[0] < 22:
                    _reject('invalid_extensible_fmt', result)
                if endian != '<' or fmt[28:40] != bytes.fromhex('00001000800000aa00389b71'):
                    _reject('unsupported_extensible_subformat', result)
                tag = struct.unpack_from('<I', fmt, 24)[0]
            result.update(chunk_count=count, data_bytes=data_chunks[0]['bytes'], data_offset=data_chunks[0]['offset'], channels=channels, samplerate=rate)
            if tag in (1, 3):
                if bits == 0 or bits % 8 or channels < 1 or (rate < 1) or (align != channels * (bits // 8)) or (byte_rate != rate * align):
                    _reject('inconsistent_pcm_geometry', result)
                if tag == 3 and bits not in (32, 64):
                    _reject('invalid_float_sample_width', result)
                if not result['data_bytes'] or result['data_bytes'] % align:
                    _reject('unaligned_or_empty_pcm_payload', result)
                result.update(frames=result['data_bytes'] // align, bits_per_sample=bits, status='container_and_pcm_geometry_checked')
                if samples64 not in (None, 0, result['frames']):
                    _reject('rf64_sample_count_mismatch', result)
            else:
                result['status'] = 'container_boundaries_checked'
        elif magic == b'FORM':
            result['container'] = 'AIFF'
            result['declared_total_bytes'] = struct.unpack_from('>I', head, 4)[0] + 8
            if head[8:12] not in (b'AIFF', b'AIFC') or result['declared_total_bytes'] != total:
                _reject('aiff_container_length_mismatch', result)
            result['status'] = 'container_length_checked'
        elif head[:16] == W64_RIFF:
            result['container'] = 'WAVE64'
            result['declared_total_bytes'] = struct.unpack_from('<Q', head, 16)[0]
            if result['declared_total_bytes'] != total:
                _reject('wave64_container_length_mismatch', result)
            result['status'] = 'container_length_checked'
        elif magic == b'fLaC':
            result['container'] = 'FLAC'
        else:
            result['container'] = 'other_decoder_supported_format'
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        _reject('file_changed_during_container_check', result)
    return result

def compare_decoder_geometry(container: dict, info: dict) -> None:
    if 'frames' in container:
        for key in ('frames', 'samplerate', 'channels'):
            if int(info[key]) != int(container[key]):
                _reject('decoder_container_geometry_mismatch', dict(container, field=key, decoder_value=info[key], container_value=container[key]))

@lru_cache(maxsize=32768)
def checked_container(path: str, size: int, mtime_ns: int) -> dict:
    """Cache by recorded file identity. Callers must stat before each audio read."""
    st = Path(path).stat()
    if (st.st_size, st.st_mtime_ns) != (size, mtime_ns):
        raise ValueError('Audio changed before container check: ' + path)
    return inspect_container(path)
