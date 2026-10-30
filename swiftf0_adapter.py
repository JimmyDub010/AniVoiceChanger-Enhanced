"""SwiftF0 0.3.0 adapter; lazy private dependencies, native RVC time grid."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

BUNDLE = Path(__file__).resolve().parent / 'vendor/swiftf0'


def available(root=BUNDLE):
    root = Path(root)
    return all((root / name).is_file() for name in (
        'BUNDLE.json', 'packages/swift_f0/core.py', 'packages/swift_f0/model.onnx',
        'packages/onnxruntime/capi/onnxruntime_pybind11_state.pyd', 'packages/onnxruntime/capi/onnxruntime.dll'))


def detector():
    if sys.version_info[:2] != (3, 9) or sys.platform != 'win32':
        raise ValueError('Bundled SwiftF0 requires the supported Windows x64 Python 3.9 RVC engine. No packages were changed.')
    if not available(): raise ValueError('SwiftF0 bundled dependencies or model are missing. Restore the complete Enhanced package; no fallback was used.')
    packages = BUNDLE / 'packages'
    manifest = json.loads((BUNDLE / 'BUNDLE.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = (packages / relative).resolve()
        if not path.is_relative_to(packages.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('SwiftF0 bundle integrity check failed: ' + relative)
    previous = sys.modules.get('onnxruntime')
    if previous and not Path(previous.__file__).resolve().is_relative_to(packages.resolve()):
        raise ValueError('A conflicting ONNX Runtime was already loaded. Reload the voice worker before selecting SwiftF0.')
    sys.path.insert(0, str(packages))
    try:
        from swift_f0 import SwiftF0, __version__
        import onnxruntime
        if __version__ != '0.3.0' or onnxruntime.__version__ != '1.19.2':
            raise ValueError('Unexpected private SwiftF0 dependency version.')
        return SwiftF0(threads=2, spin=False)
    except Exception as error:
        raise ValueError('SwiftF0 could not initialize its bundled CPU extractor: ' + str(error)) from error


def align_result(result, length, shift=0):
    times = np.asarray(result.timestamps, dtype='float64')
    pitch = np.asarray(result.pitch_hz, dtype='float64')
    confidence = np.asarray(result.confidence, dtype='float64')
    if times.ndim != 1 or pitch.shape != times.shape or confidence.shape != times.shape or not len(times):
        raise ValueError('SwiftF0 returned an empty or inconsistent pitch contour.')
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('SwiftF0 returned invalid timestamps.')
    voiced = np.isfinite(pitch) & np.isfinite(confidence) & (confidence >= .5) & (pitch >= 50) & (pitch <= 1100)
    target = np.arange(length, dtype='float64') * .01
    right = np.clip(np.searchsorted(times, target), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(np.abs(target - times[left]) <= np.abs(target - times[right]), left, right)
    frequency = np.where(voiced[nearest], pitch[nearest], 0.)
    # Interpolate only within voiced runs. Never smooth a voiced/unvoiced boundary.
    both = voiced[left] & voiced[right] & (right != left) & (target >= times[left]) & (target <= times[right])
    alpha = (target[both] - times[left[both]]) / (times[right[both]] - times[left[both]])
    frequency[both] = np.exp((1 - alpha) * np.log(pitch[left[both]]) + alpha * np.log(pitch[right[both]]))
    frequency[(target < times[0] - .008) | (target >= times[-1] + .016)] = 0
    frequency *= 2 ** (shift / 12)
    from pitch_methods import coarse
    return coarse(frequency), frequency.astype('float32')


def adapter(engine):
    if not hasattr(engine, 'swiftf0'): engine.swiftf0 = detector()
    def extract(path, waveform, length, shift, method, filter_radius, inp_f0=None):
        # The existing RVC pipeline has already mixed/resampled/padded audio to mono 16 kHz.
        result = engine.swiftf0.detect(np.asarray(waveform, dtype='float32'), 16000, fmin=50, fmax=1100)
        return align_result(result, length, shift)
    return extract
