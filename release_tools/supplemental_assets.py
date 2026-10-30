"""Bundle only the exact previously tested, license-reviewed optional pitch assets."""
import hashlib
import json
from pathlib import Path
import shutil


def copy_assets(source, destination):
    expected = {'experiments/rmvpe-2026.pt': '6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193',
                'experiments/fcpe/packages/torchfcpe/assets/fcpe_c_v001.pt': 'b9aeaeb673436eeda50ceafd632aa681aa63417e52eae4207503d180c9b10015'}
    originals = [source / 'experiments/rmvpe-2026.pt']
    for directory in ('packages', 'licenses'):
        originals += [p for p in (source / 'experiments/fcpe' / directory).rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    originals += list((source / 'release_tools/asset-notices').glob('*'))
    files = {}
    for original in originals:
        relative = original.relative_to(source).as_posix()
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if relative in expected and digest != expected[relative]: raise ValueError('Changed pitch checkpoint: ' + relative)
        if original.suffix.lower() in ('.pt', '.pth', '.onnx', '.index') and relative not in expected:
            raise ValueError('Unreviewed pitch weight: ' + relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        files[relative] = digest
    if not all(name in files for name in expected): raise ValueError('Missing reviewed pitch assets')
    manifest = {'schema': 1, 'files': files, 'provenance': {
        'rmvpe-new': {'url': 'https://huggingface.co/lj1995/VoiceConversionWebUI/tree/e6d0c1a17da07c33557852f9dfa2bd44cc75737d',
                      'license': 'MIT model-card metadata', 'sha256': expected['experiments/rmvpe-2026.pt']},
        'fcpe': {'url': 'https://github.com/CNChTu/FCPE/blob/main/torchfcpe/assets/fcpe_c_v001.pt',
                 'git_blob_sha1': '1741230325796805def4b15aebc12c56c70da974', 'license': 'MIT, copyright 2023 CN_ChiTu'},
        'fcpe-dependencies': 'Bundled einops/local-attention retain their MIT notices. local_attention/__init__.py is a minimal compatibility import shim.'}}
    (destination / 'PITCH-ASSET-MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    import sys
    copy_assets(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
