"""Build relocatable app/source candidates, with only reviewed SwiftF0 assets."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from supplemental_assets import copy_assets

APP_FILES = ['activity.py', 'audio_effects.py', 'calibration_dialog.py', 'desktop.py',
             'diagnose.py', 'engine_client.py', 'engine_worker.py', 'enhanced.py',
             'global_hotkeys.py', 'pitch_calibration.py', 'routing.py', 'safe_checkpoint.py',
             'voice_dialog.py', 'voice_library.py', 'pitch_lab.py', 'pitch_lab_dialog.py',
             'pitch_methods.py', 'swiftf0_adapter.py']
MODEL_EXTENSIONS = {'.pt', '.pth', '.index', '.onnx', '.safetensors', '.ckpt', '.h5'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}


def allowed_runtime(path):
    if any(p.lower() in {'__pycache__', 'test', 'tests', 'idle_test', '.git'} for p in path.parts): return False
    if path.name in {'_virtualenv.pth', '_virtualenv.py', 'direct_url.json'}: return False
    if path.suffix.lower() in {'.pyc', '.whl'} | AUDIO_EXTENSIONS: return False
    # pywin32.pth is a text Python startup configuration, not trained weights.
    if path.suffix.lower() in MODEL_EXTENSIONS and path.name != 'pywin32.pth': return False
    return True


def copy_runtime(source, destination):
    for file in source.rglob('*'):
        relative = file.relative_to(source)
        if file.is_file() and allowed_runtime(relative):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)


def add_manifest(folder):
    files = {}
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.name != 'SHA256-MANIFEST.json':
            files[path.relative_to(folder).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / 'SHA256-MANIFEST.json').write_text(json.dumps(files, indent=2), encoding='utf-8')


def archive(folder):
    target = Path(str(folder) + '.zip')
    with ZipFile(target, 'x', ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(folder.rglob('*')):
            if path.is_file(): output.write(path, path.relative_to(folder.parent).as_posix())
    with ZipFile(target) as check:
        assert check.testzip() is None
        names = check.namelist()
        bundle = folder / ('app/vendor/swiftf0' if (folder / 'app').exists() else 'vendor/swiftf0')
        reviewed = json.loads((bundle / 'BUNDLE.json').read_text(encoding='utf-8'))['files']
        permitted = {(bundle / 'packages' / name).relative_to(folder.parent).as_posix() for name in reviewed}
        app = folder / 'app' if (folder / 'app').exists() else folder
        supplemental = json.loads((app / 'PITCH-ASSET-MANIFEST.json').read_text(encoding='utf-8'))['files']
        permitted.update((app / name).relative_to(folder.parent).as_posix() for name in supplemental)
        assert not any(Path(n).suffix.lower() in MODEL_EXTENSIONS and not n.endswith('/pywin32.pth') and n not in permitted for n in names)
        assert not any(Path(n).name in {'settings.json', '.env', 'voices.json'} for n in names)
    return dict(file=target.name, bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest(), entries=len(names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', default='2026.10.26-enhanced')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    portable = output / ('AniVoiceChanger-Enhanced-' + args.version + '-Windows-x64')
    code = output / ('AniVoiceChanger-Enhanced-' + args.version + '-Source')
    portable.mkdir(); code.mkdir()
    for folder in (portable / 'app', code):
        folder.mkdir(exist_ok=True)
        for name in APP_FILES: shutil.copyfile(source / name, folder / name)
        shutil.copyfile(source / 'LICENSE', folder / 'LICENSE')
        requirements = source / 'desktop-requirements.txt'
        if not requirements.exists(): requirements = source / 'requirements.txt'
        shutil.copyfile(requirements, folder / 'requirements.txt')
        # SwiftF0 is offline-ready, including the MIT-licensed bundled ONNX model,
        # private CPython 3.9 ONNX Runtime DLLs, transitive dependencies and notices.
        bundle = source / 'vendor/swiftf0'
        manifest = json.loads((bundle / 'BUNDLE.json').read_text(encoding='utf-8'))
        for name, expected in manifest['files'].items():
            original = bundle / 'packages' / name
            assert hashlib.sha256(original.read_bytes()).hexdigest() == expected, 'Changed SwiftF0 asset: ' + name
            target = folder / 'vendor/swiftf0/packages' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        shutil.copyfile(bundle / 'BUNDLE.json', folder / 'vendor/swiftf0/BUNDLE.json')
        copy_assets(source, folder)
        for document in ('SWIFTF0.md', 'ACKNOWLEDGMENTS.md', 'RELEASE-NOTES-2026.10.30.md', 'EXPERIMENTAL-PITCH-LAB.md', 'PITCH-DEVICE-SUPPORT.md'):
            shutil.copyfile(source / document, folder / document)
        shutil.copyfile(source / 'release_tools/PUBLIC-PITCH-LAB.md', folder / 'EXPERIMENTAL-PITCH-LAB.md')
    readme = (source / 'release_tools' / 'RELEASE-README.md').read_text(encoding='utf-8')
    for folder in (portable, code):
        (folder / 'README.md').write_text(readme, encoding='utf-8')
    (portable / 'Launch-Enhanced.cmd').write_text('@echo off\nsetlocal\ncd /d "%~dp0"\n"runtime\\python.exe" -B -s app\\enhanced.py\nif errorlevel 1 pause\n', encoding='utf-8')
    (code / 'Launch-Enhanced.cmd').write_text('@echo off\nsetlocal\ncd /d "%~dp0"\nif not exist ".venv\\Scripts\\python.exe" (\n echo Run Setup.cmd first.\n pause\n exit /b 1\n)\n".venv\\Scripts\\python.exe" -B enhanced.py\nif errorlevel 1 pause\n', encoding='utf-8')
    (code / 'Setup.cmd').write_text('@echo off\nsetlocal\ncd /d "%~dp0"\npy -3.10 -m venv .venv\nif errorlevel 1 goto failed\n".venv\\Scripts\\python.exe" -m pip install -r requirements.txt\nif errorlevel 1 goto failed\necho Setup complete. Run Launch-Enhanced.cmd.\npause\nexit /b 0\n:failed\necho Setup failed. Python 3.10 x64 and internet access are required.\npause\nexit /b 1\n', encoding='utf-8')
    (code / 'tests').mkdir()
    for test in (source / 'tests').glob('test_*.py'): shutil.copyfile(test, code / 'tests' / test.name)
    (code / 'test-results').mkdir()
    (code / 'test-results' / '.gitkeep').touch()
    shutil.copytree(source / 'release_tools', code / 'release_tools', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    print('Copying desktop Python runtime and dependencies...', flush=True)
    copy_runtime(Path(sys.base_prefix), portable / 'runtime')
    copy_runtime(Path(sys.prefix) / 'Lib' / 'site-packages', portable / 'runtime' / 'Lib' / 'site-packages')
    notices = ['Bundled desktop dependencies; original license files are retained in runtime.\n']
    for dist in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
        notices.append(f"{dist.metadata['Name']} {dist.version}: {dist.metadata.get('License', 'See package license files')}\n")
    (portable / 'THIRD-PARTY-NOTICES.txt').write_text('\n'.join(notices), encoding='utf-8')
    reports = []
    for folder in (portable, code):
        add_manifest(folder)
        print('Creating ' + folder.name + '.zip', flush=True)
        reports.append(archive(folder))
    (output / 'release-checksums.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == '__main__': main()
