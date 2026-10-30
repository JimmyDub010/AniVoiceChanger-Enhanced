"""Explicit method capabilities and opt-in experimental F0 adapters."""
from pathlib import Path
import numpy as np

NATIVE = {'rmvpe': 'RMVPE current (GPU only)', 'crepe': 'CREPE Full (GPU only)',
          'rmvpe-new': 'RMVPE New (GPU only)', 'fcpe-experimental': 'FCPE Experimental (GPU only)',
          'pm': 'Praat autocorrelation (CPU only)', 'harvest': 'Harvest (CPU only)',
          'swiftf0': 'SwiftF0 0.3.0 (CPU only)'}
EXPERIMENTAL = {'mangio-crepe-80': 'Mangio-style CREPE Full hop 80 Experimental (GPU only)',
                'rmvpe-range': 'RMVPE constrained range Experimental (GPU only)',
                'hybrid-rmvpe-fcpe': 'RMVPE + FCPE agreement hybrid Experimental (GPU only)'}
LABELS = dict(NATIVE, **EXPERIMENTAL)
# Actual Enhanced execution paths, not theoretical library capabilities.
EXTRACTION_DEVICES = {key: 'cpu' if key in ('pm', 'harvest', 'swiftf0') else 'cuda' for key in LABELS}
DEVICE_NOTE = ('Device labels describe pitch extraction in Enhanced. Full voice conversion requires '
               'NVIDIA CUDA, including when a CPU-only extractor is selected. CPU-only conversion is not supported.')


def capabilities(engine, app=None, f0=True):
    engine = Path(engine)
    app = Path(app) if app else Path(__file__).resolve().parent
    packages = engine / 'runtime/Lib/site-packages'
    checks = {
        'rmvpe': (engine / 'rmvpe.pt').is_file(),
        'crepe': (packages / 'torchcrepe/assets/full.pth').is_file(),
        'rmvpe-new': (app / 'experiments/rmvpe-2026.pt').is_file(),
        'fcpe-experimental': (app / 'experiments/fcpe/packages/torchfcpe/assets/fcpe_c_v001.pt').is_file(),
        'pm': bool(list(packages.glob('parselmouth*'))),
        'harvest': (packages / 'pyworld').is_dir(),
    }
    checks['mangio-crepe-80'] = checks['crepe']
    checks['rmvpe-range'] = checks['rmvpe']
    checks['hybrid-rmvpe-fcpe'] = checks['rmvpe'] and checks['fcpe-experimental']
    from swiftf0_adapter import available
    checks['swiftf0'] = available(app / 'vendor/swiftf0') and (engine / 'runtime/python39.dll').is_file()
    return [dict(key=key, label=label, available=bool(checks[key] and f0),
                 reason='' if checks[key] and f0 else 'Missing installed weights/dependency, or the voice is non-F0.',
                 native=key in ('rmvpe', 'crepe', 'pm', 'harvest')) for key, label in LABELS.items()]


def coarse(frequency):
    mel = 1127 * np.log(1 + frequency / 700)
    low, high = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
    voiced = mel > 0
    mel[voiced] = (mel[voiced] - low) * 254 / (high - low) + 1
    return np.rint(np.clip(mel, 1, 255)).astype(np.int64)


def align(frequency, length):
    frequency = np.asarray(frequency, dtype='float32').reshape(-1)
    if len(frequency) == length: return frequency.copy()
    if not len(frequency): return np.zeros(length, dtype='float32')
    # Align frames without interpolating across unvoiced gaps.
    result = np.interp(np.linspace(0, len(frequency) - 1, length), np.arange(len(frequency)), frequency)
    nearest = np.rint(np.linspace(0, len(frequency) - 1, length)).astype(int)
    result[frequency[nearest] <= 0] = 0
    return result.astype('float32')


def hybrid(first, second, tolerance=3):
    first, second = np.asarray(first), np.asarray(second)
    both = (first > 0) & (second > 0)
    agree = np.zeros(first.shape, dtype=bool)
    agree[both] = np.abs(12 * np.log2(first[both] / second[both])) <= tolerance
    result = first.copy()  # RMVPE is authoritative for voicing and disagreements.
    result[agree] = np.sqrt(first[agree] * second[agree])
    return result, {'agreement_frames': int(agree.sum()), 'disagreement_frames': int((both & ~agree).sum())}


def adapter(engine, key, base, options, fcpe_function=None):
    """Return a get_f0-compatible function. The untouched native path stays default."""
    import torch
    low = float(options.get('f0_min', 50))
    high = float(options.get('f0_max', 1100))
    if not 50 <= low < high <= 1100: raise ValueError('Experimental pitch range must satisfy 50 <= minimum < maximum <= 1100 Hz.')
    details = {'adapter': key, 'f0_min': low, 'f0_max': high}

    def extract(path, waveform, length, shift, method, filter_radius, inp_f0=None):
        if key == 'mangio-crepe-80':
            import torchcrepe
            scale = max(float(np.quantile(np.abs(waveform), .999)), 1e-6)
            tensor = torch.from_numpy((waveform / scale).astype('float32'))[None, :]
            predicted = torchcrepe.predict(tensor, 16000, 80, low, high, 'full',
                                           batch_size=160, device=engine.config.device, pad=True)
            frequency = align(predicted[0].detach().cpu().numpy(), length)
            details['hop_samples'] = 80
        elif key == 'rmvpe-range':
            if not hasattr(engine.vc, 'model_rmvpe'):
                from rmvpe import RMVPE
                engine.vc.model_rmvpe = RMVPE('rmvpe.pt', is_half=engine.half, device=engine.config.device)
                engine.pitch_checkpoint = 'classic'
            predictor = engine.vc.model_rmvpe
            tensor = torch.from_numpy(waveform).float().to(engine.config.device)[None, :]
            salience = predictor.mel2hidden(predictor.mel_extractor(tensor, center=True)).squeeze(0).float().cpu().numpy()
            bins = 10 * 2 ** ((20 * np.arange(salience.shape[1]) + 1997.3794084376191) / 1200)
            salience[:, (bins < low) | (bins > high)] = 0
            valid = salience.max(axis=1) > .03
            # decode can divide by zero on fully masked/unvoiced frames; explicitly preserve zero F0.
            frequency = np.zeros(len(salience), dtype='float32')
            if valid.any(): frequency[valid] = predictor.decode(salience[valid], thred=.03)
            frequency = align(frequency, length)
        elif key == 'hybrid-rmvpe-fcpe':
            if fcpe_function is None: raise ValueError('FCPE is unavailable for the hybrid adapter.')
            _, one = base(path, waveform, length, 0, 'rmvpe', filter_radius, inp_f0)
            _, two = fcpe_function(path, waveform, length, 0, 'fcpe-experimental', filter_radius, inp_f0)
            frequency, agreement = hybrid(align(one, length), align(two, length))
            details.update(agreement)
        else:
            raise ValueError('Unknown experimental pitch adapter: ' + key)
        frequency = np.nan_to_num(frequency, nan=0., posinf=0., neginf=0.) * (2 ** (shift / 12))
        return coarse(frequency), frequency
    return extract, details
