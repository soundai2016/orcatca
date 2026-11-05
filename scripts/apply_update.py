"""Install the code archive in an existing OrcaTCA project. No data migration."""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from zipfile import ZipFile

TOP_FILES = {'.gitignore', 'LICENSE', 'README.md', 'pyproject.toml', 'requirements.txt', 'run.sh', 'setup.sh'}
TOP_DIRS = {'assets', 'configs', 'scripts', 'src', 'tests'}
# Pairs of path and content digests identify retired, unmodified distribution files.
RETIRED = {
    '1f7af29381cf98baf40d1a655dee3b47f197aec0d84327fa9ec455c3c8a8bc28': '24972d6c6f1f3bd4237ea330a9556d7fd7b418e3ff90980061c52835878221c0',
    '09616846d539ea3789ae1580b8ca470cbba4aa38f686d299ce494d026b7fbd9c': '872706436ab98a8099f8fdbde126d5b3c2676c3d03c4abfafe9e45c4c61db0cf',
    '0bfced6db18b3fce14f8856db302efe1a1e64ca21fae46a02989e4ff4145834e': 'dab9bdf1e8fc4a0bdd02ab208ff69045b2b0a7fc49c0ecf36a006fd4f64d6799',
}

def apply(archive, destination):
    root = destination.resolve(strict=True)
    cfg = root / 'pyproject.toml'
    if not cfg.is_file() or 'orcatca' not in cfg.read_text().lower():
        raise ValueError('Expected an existing OrcaTCA project root')
    def checked(name):
        rel = PurePosixPath(name)
        dest = root.joinpath(*rel.parts)
        if (rel.is_absolute() or '..' in rel.parts or rel.as_posix() != name or
            not (name in TOP_FILES or (len(rel.parts)>1 and rel.parts[0] in TOP_DIRS)) or
            dest.resolve() != dest or (dest.exists() and not dest.is_file())):
            raise ValueError('Unexpected or redirected destination: '+name)
        return dest
    def sha(data): return hashlib.sha256(data).hexdigest()
    retired = []; preserved = []; changes = []
    with ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or not {'pyproject.toml','scripts/plot.py','__main__.py'} <= set(names):
            raise ValueError('Expected the code update archive')
        for item in z.infolist():
            if item.filename == '__main__.py': continue
            dest = checked(item.filename); data = z.read(item)
            if not dest.exists() or dest.read_bytes() != data:
                changes.append((dest,data,0o755 if item.external_attr >> 16 & 0o111 else 0o644))
        for folder in ('scripts','tests'):
            for file in (root/folder).glob('*.py'):
                name = file.relative_to(root).as_posix()
                expected = RETIRED.get(sha(name.encode()))
                if expected is None or name in names: continue
                checked(name)
                if sha(file.read_bytes()) != expected:
                    preserved.append(name); continue
                retired.append(file)
                for cache in (file.parent/'__pycache__').glob(file.stem+'.*.pyc'):
                    retired.append(checked(cache.relative_to(root).as_posix()))
    old = [p for p,_,_ in changes if p.exists()] + retired
    backup = Path(tempfile.mkdtemp(prefix='orcatca-code-')) if old else None
    for file in old:
        copy = backup/file.relative_to(root); copy.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,copy)
    for dest,data,mode in changes:
        dest.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False) as buf:
            buf.write(data); temp = Path(buf.name)
        try:
            temp.chmod(mode); os.replace(temp,dest)
        finally:
            temp.unlink(missing_ok=True)
    for file in retired: file.unlink()
    print('Project root:',root)
    print(f'Updated {len(changes)} files; retired {len(retired)} unchanged files or bytecode caches.')
    if backup: print('Previous code:',backup)
    for name in preserved: print('Kept locally modified retired file:',name)
    print('Renderer:',root/'scripts/plot.py')
    print('Next: python "'+str(root/'scripts/plot.py')+'"')
    print('Check: python "'+str(root/'scripts/check_figures.py')+'"')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=Path,required=True,help='Existing project directory')
    p.add_argument('--bundle',type=Path,default=Path(sys.argv[0]),help='Code update ZIP; defaults to this executable ZIP')
    args=p.parse_args();apply(args.bundle,args.target)

if __name__=='__main__': main()
