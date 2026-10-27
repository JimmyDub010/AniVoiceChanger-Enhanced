"""Read-only restoration checks; never import or unpickle voice models."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from routing import endpoints, resolve


ROOT = Path(__file__).resolve().parent


def inspect_engine(folder):
    root = Path(folder).expanduser() if folder else ROOT.parent
    model_module = ('lib/infer_pack/models.py' if (root / 'lib/infer_pack/models.py').is_file()
                    else 'infer_pack/models.py')
    required = ('vc_infer_pipeline.py', model_module, 'my_utils.py', 'hubert_base.pt', 'runtime/python.exe')
    missing = [name for name in required if not (root / name).is_file()]
    return {'folder': str(root), 'missing': missing,
            'ready_files': not missing,
            'note': 'File presence alone does not establish compatible dependencies or inference.'}


def report(folder=''):
    result = {'python': sys.version, 'executable': sys.executable,
              'engine': inspect_engine(folder), 'devices': []}
    for endpoint in endpoints():
        item = vars(endpoint).copy()
        try:
            _, device = resolve(endpoint.id, endpoint.direction)
            item['sample_rate'] = device['default_samplerate']
            item['routing'] = 'resolvable'
        except Exception as error:
            item['routing'] = str(error)
        result['devices'].append(item)
    if importlib.util.find_spec('torch'):
        import torch
        result['torch'] = {'version': torch.__version__, 'cuda': torch.cuda.is_available()}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result['torch'].update(gpu=torch.cuda.get_device_name(), free_bytes=free, total_bytes=total)
    else:
        result['torch'] = 'Not installed in isolated environment; engine dependency restoration pending.'
    runtime = Path(result['engine']['folder']) / 'runtime' / 'python.exe'
    if runtime.is_file():
        try:
            check = subprocess.run(
                [str(runtime), '-B', '-c',
                 'import json,torch; print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,'
                 'cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) '
                 'if torch.cuda.is_available() else None)))'],
                cwd=runtime.parent.parent, capture_output=True, text=True, timeout=30)
            result['rvc_runtime'] = json.loads(check.stdout) if check.returncode == 0 else check.stderr
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            result['rvc_runtime'] = str(error)
    try:
        result['nvidia_smi'] = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv'],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        result['nvidia_smi'] = str(error)
    return result


if __name__ == '__main__':
    print(json.dumps(report(sys.argv[1] if len(sys.argv) > 1 else ''), indent=2))
