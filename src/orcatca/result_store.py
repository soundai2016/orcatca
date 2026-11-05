"""Read frozen prediction results without extracting archives or loading weights."""
from __future__ import annotations
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile
import numpy as np
import yaml
from .utils import sha256
from .references import protocol_directory


class ReplayTrace:
    """Verify the same study inputs across report ZIPs and relocated suite trees."""
    def __init__(self, path, methods, repeats, crossfit_repeats, seed):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()

        def unique(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError('Duplicate expected trace key: ' + key)
                result[key] = value
            return result

        trace = json.loads(raw, object_pairs_hook=unique)
        settings = dict(status='completed', methods=methods, analysis_seed=seed,
                        bootstrap_repeats=repeats, crossfit_repeats=crossfit_repeats)
        for key, value in settings.items():
            if trace.get(key) != value:
                raise ValueError('Expected analysis trace has different ' + key)
        self.seeds = trace['seeds']
        self.roles = {}
        for key, sha in trace['used_file_sha256'].items():
            role, _, member = key.partition(':')
            path = PurePosixPath(member)
            if (role not in {'primary','reference'} or path.as_posix() != member or
                    '..' in path.parts or not re.fullmatch('[0-9a-f]{64}', sha)):
                raise ValueError('Invalid expected study input: ' + key)
            if member.startswith('result/runs/'):
                member = member[len('result/'):]
            if not member.startswith(('runs/', 'state/protocol/')):
                raise ValueError('Unrecognized expected study input: ' + key)
            members = self.roles.setdefault(role, {})
            if member in members:
                raise ValueError('Duplicate expected study member after normalizing prefixes: ' + key)
            members[member] = sha
        if 'primary' not in self.roles:
            raise ValueError('Expected trace contains no primary input hashes')


class ResultStore:
    def __init__(self, root: Path, source: str, expected=None):
        self.root = root.resolve()
        self.path = (self.root / source).resolve()
        raw = (self.root / 'dclde_2027_killer_whales').resolve()
        if self.path.is_relative_to(raw):
            raise ValueError('A result source cannot be inside the raw corpus')
        self.used = {}
        self.expected = expected
        self.checked_members = set()
        self.protocol_dir = None
        self.archive = None
        self.checksums = {}
        if self.path.is_file():
            self.archive = zipfile.ZipFile(self.path)
            members = [i.filename for i in self.archive.infolist() if not i.is_dir()]
            if len(members) != len(set(members)):
                raise ValueError('Duplicate archive members')
            for name in members:
                rel = PurePosixPath(name)
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('Unsafe archive member')
            prefixes = [x for x in ('runs/', 'result/runs/') if x+'config.yaml' in members]
            if len(prefixes) != 1:
                self.archive.close()
                if 'manifest.json' in members and any(s.startswith('objects/') for s in members):
                    raise ValueError('Restore the prediction companion with scripts/study_reports.py restore first')
                raise ValueError('Expected one frozen run configuration in report')
            self.prefix = prefixes[0]
            if 'REPORT_SHA256.json' not in members:
                raise ValueError('Report lacks integrity manifest')
            self.checksums = json.loads(self.archive.read('REPORT_SHA256.json'))
            self.source_hash = sha256(self.path)
        elif self.path.is_dir():
            if not self.path.is_relative_to(self.root):
                raise ValueError('A loose suite must be inside the project; external report ZIPs are supported')
            if not (self.path / 'config.yaml').is_file():
                raise ValueError(f'{source} is not a frozen run: expected config.yaml, config.json and per-fit '
                                 'calibration.jsonl.gz and holdout.jsonl.gz. Summary holdout.json files '
                                 'and derived tables cannot replace window scores.')
            self.prefix = ''
        else:
            raise FileNotFoundError(f'No result suite or report at {source}; use the available runs/<suite>/ '
                                    'or a matching restored prediction report.')
        config = self.bytes('config.yaml')
        self.config_sha256 = hashlib.sha256(config).hexdigest()
        if not self.archive:
            self.source_hash = self.config_sha256
        self.config = yaml.safe_load(config.decode())
        frozen = self.json('config.json')
        if frozen.get('config') != self.config:
            raise ValueError('Frozen configuration mismatch')
        self.name = self.config['name']

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def bytes(self, relative: str, protocol: bool = False) -> bytes:
        rel = PurePosixPath(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe result path')
        if self.archive:
            member = ('state/protocol/' if protocol else self.prefix) + relative
            info = self.archive.getinfo(member)
            if info.file_size > 1024**3:
                raise ValueError('Unexpectedly large result member')
            value = self.archive.read(member)
            actual = hashlib.sha256(value).hexdigest()
            if self.checksums.get(member) != actual:
                raise ValueError('Missing or incorrect result checksum: '+member)
            self.used[member] = actual
        else:
            if protocol:
                if self.protocol_dir is None:
                    self.protocol_dir = protocol_directory(self.root, self.config, allow_relocated=True)
                path = self.protocol_dir / relative
            else:
                path = self.path / relative
            resolved = path.resolve()
            if (not resolved.is_relative_to(self.root) or
                    resolved.is_relative_to((self.root/'dclde_2027_killer_whales').resolve()) or path.is_symlink()):
                raise ValueError('Result input escapes the project')
            value = path.read_bytes()
            actual = hashlib.sha256(value).hexdigest()
            # Use archive-style keys so this trace can itself check a later replay.
            self.used[('state/protocol/' if protocol else 'runs/') + relative] = actual
        member = ('state/protocol/' if protocol else 'runs/') + relative
        if protocol and relative == 'complete.json':
            binding = self.config.get('input_reference')
            if binding and binding['protocol_sha256'] != actual:
                raise ValueError('Protocol completion record differs from frozen configuration: ' + member)
        if self.expected is not None:
            if self.expected.get(member) != actual:
                raise ValueError('Missing or mismatched expected study input: ' + member)
            self.checked_members.add(member)
        return value

    def provenance(self):
        return dict(path=str(self.path), kind='report' if self.archive else 'suite_directory',
                    source_hash_kind='archive_bytes' if self.archive else 'config.yaml',
                    protocol='state/protocol/ (archive)' if self.archive else str(self.protocol_dir),
                    config_sha256=self.config_sha256)

    def json(self, relative, protocol=False):
        return json.loads(self.bytes(relative, protocol))

    def rows(self, relative, protocol=False):
        data = self.bytes(relative, protocol)
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def protocol(self, fold):
        complete = self.json('complete.json', True)
        name = complete['folds'][fold]['manifest']
        raw = self.bytes(name, True)
        if hashlib.sha256(raw).hexdigest() != complete['files'][name]:
            raise ValueError('Protocol manifest mismatch')
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def predictions(self, fold, methods, seeds, role):
        reference = None
        matrices = []
        calibrations = []
        for method in methods:
            values, cals = [], []
            for seed in seeds:
                prefix = f'fold{fold:02d}/{method}_s{seed}/'
                status = self.json(prefix+'status.json')
                if status.get('status') != 'completed':
                    raise ValueError('Requested fit is incomplete: '+prefix)
                rows = self.rows(prefix+role+'.jsonl.gz')
                indexed = {r['id']: r for r in rows}
                if len(rows) != len(indexed) or not rows:
                    raise ValueError('Empty or duplicate predictions')
                if reference is None:
                    reference = [{k: v for k, v in r.items() if k != 'p'} for r in sorted(rows, key=lambda r:r['id'])]
                if set(indexed) != {r['id'] for r in reference}:
                    raise ValueError('Prediction IDs differ')
                for r in reference:
                    match = indexed[r['id']]
                    if any(r.get(k) != match.get(k) for k in ('label','raw_class','site','group','recording','block')):
                        raise ValueError('Prediction metadata differ')
                    if r['raw_class'] not in {'KW','HW','AB'} or r['label'] != int(r['raw_class']=='KW'):
                        raise ValueError('Inconsistent labels')
                p = np.asarray([indexed[r['id']]['p'] for r in reference], float)
                if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
                    raise ValueError('Invalid probability')
                values.append(p)
                cals.append(self.json(prefix+'calibration.json'))
            matrices.append(values)
            calibrations.append(cals)
        return reference, np.asarray(matrices), calibrations


class CombinedStore:
    """Pair new controls with frozen reference predictions under identical inputs."""
    def __init__(self, root, source, reference=None, trace=None):
        sources = {'primary','reference'} if reference else {'primary'}
        if trace and set(trace.roles) != sources:
            raise ValueError('Expected trace and supplied inputs differ: pass the required --source and --reference')
        self.primary=ResultStore(root,source,trace.roles['primary'] if trace else None)
        try:
            self.reference=ResultStore(root,reference,trace.roles['reference'] if trace else None) if reference else None
        except Exception:
            self.primary.close()
            raise
        self.path=self.primary.path
        self.name=self.primary.name
        import copy
        self.config=copy.deepcopy(self.primary.config)
        self.source_hash=self.primary.source_hash
        self._checked=set()
        if self.reference:
            a=self.primary.config;b=self.reference.config
            bound_config=(a.get('input_reference') or {}).get('source_config_sha256')
            if bound_config is not None and bound_config != self.reference.config_sha256:
                self.close();raise ValueError('Reference configuration differs from the primary suite binding')
            for key in ('data','cache','augmentation','runtime','evaluation'):
                if a[key]!=b[key]:
                    self.close();raise ValueError('Reference comparison changed '+key)
            for key,value in b['experiment'].items():
                if key=='seeds' and set(a['experiment'][key]) <= set(value):continue
                if key=='folds' and a['experiment'][key] <= value:continue
                if key not in {'methods'} and a['experiment'].get(key)!=value:
                    self.close();raise ValueError('Reference experiment differs: '+key)
            self.config['experiment']['methods']=list(dict.fromkeys(a['experiment']['methods']+b['experiment']['methods']))
            self.source_hash=hashlib.sha256((self.primary.source_hash+self.reference.source_hash).encode()).hexdigest()

    def verified_trace(self, trace):
        counts = {}
        for role, expected in trace.roles.items():
            checked = getattr(self, role).checked_members
            missing = sorted(expected.keys() - checked)
            if missing:
                raise ValueError(f'Expected {role} study inputs were not consumed: ' + ', '.join(missing[:3]))
            counts[role] = len(checked)
        return dict(path=str(trace.path), sha256=trace.sha256, checked_members=counts,
                    status='all_expected_inputs_match')

    @property
    def used(self):
        d={'primary:'+k:v for k,v in self.primary.used.items()}
        if self.reference:d.update({'reference:'+k:v for k,v in self.reference.used.items()})
        return d

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def close(self):
        self.primary.close()
        if self.reference:self.reference.close()

    def json(self,name,protocol=False):return self.primary.json(name,protocol)

    def protocol(self,fold):
        a=self.primary.protocol(fold)
        if self.reference and fold not in self._checked:
            b=self.reference.protocol(fold)
            from .utils import digest
            if digest(a)!=digest(b):raise ValueError('Reference uses different samples or source roles')
            self._checked.add(fold)
        return a

    def predictions(self,fold,methods,seeds,role):
        self.protocol(fold)
        rows=None;matrices=[];calibrations=[]
        for method in methods:
            store=self.primary if method in self.primary.config['experiment']['methods'] else self.reference
            if store is None or method not in store.config['experiment']['methods']:
                raise ValueError('Method not found: '+method)
            r,p,c=store.predictions(fold,[method],seeds,role)
            if rows is None:rows=r
            elif rows!=r:raise ValueError('Cross-suite prediction metadata differ')
            matrices.append(p[0]);calibrations.append(c[0])
        return rows,np.asarray(matrices),calibrations
