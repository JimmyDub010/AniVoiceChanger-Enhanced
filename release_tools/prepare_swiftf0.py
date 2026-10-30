"""Verify official PyPI wheel hashes and safely unpack the reviewed private bundle.

Developer build tool only; never invoked by the application. No pip installation.
"""
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent / 'vendor/swiftf0'


def main():
    destination = ROOT / 'packages'
    destination.mkdir(exist_ok=False)
    wheels, files = [], {}
    for wheel in sorted((ROOT / 'wheels').glob('*.whl')):
        with ZipFile(wheel) as archive:
            metadata = BytesParser().parsebytes(archive.read(next(n for n in archive.namelist() if n.endswith('/METADATA'))))
            name, version = metadata['Name'], metadata['Version']
            with urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=30) as response:
                official = json.load(response)
            release = next(item for item in official['urls'] if item['filename'] == wheel.name)
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            if digest != release['digests']['sha256']: raise ValueError('PyPI hash mismatch: ' + wheel.name)
            wheels.append(dict(name=name, version=version, filename=wheel.name, sha256=digest, url=release['url'],
                               license=metadata.get('License-Expression', metadata.get('License', 'See bundled licenses'))))
            for item in archive.infolist():
                path = PurePosixPath(item.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in item.filename or ':' in item.filename:
                    raise ValueError('Unsafe wheel path')
                if stat.S_ISLNK(item.external_attr >> 16): raise ValueError('Wheel symlink')
                if item.is_dir(): continue
                if item.file_size > 100 * 1024**2: raise ValueError('Oversized wheel entry')
                # .data contains developer console scripts; no scripts/startup hooks are installed.
                if any(part.endswith('.data') for part in path.parts): continue
                target = destination.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream: stream.write(archive.read(item))
                files[path.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (ROOT / 'BUNDLE.json').write_text(json.dumps({'schema': 1, 'python': 'CPython 3.9 Windows x64',
        'numpy': 'Uses unchanged RVC NumPy >=1.21.6 (tested 1.23.5); no NumPy shadow copy',
        'wheels': wheels, 'files': files}, indent=2), encoding='utf-8')
    print('Verified and unpacked', len(wheels), 'official wheels;', len(files), 'files.')


if __name__ == '__main__': main()
