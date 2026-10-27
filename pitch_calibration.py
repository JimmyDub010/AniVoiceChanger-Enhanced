"""Measured pitch comparisons, not a naturalness predictor or gender classifier."""
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(audio, rate, frequency):
    audio = np.asarray(audio, dtype='float32')
    frequency = np.asarray(frequency, dtype='float64')
    voiced = frequency[np.isfinite(frequency) & (frequency >= 50) & (frequency <= 1000)]
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Audio is empty or contains invalid samples.')
    result = {'seconds': len(audio) / rate, 'frames': len(frequency),
              'voiced_frames': len(voiced), 'voiced_fraction': len(voiced) / max(1, len(frequency)),
              'peak': float(np.max(np.abs(audio))),
              'clipped_fraction': float(np.mean(np.abs(audio) >= .999)),
              'rms': float(np.sqrt(np.mean(audio ** 2))),
              'analyzer': 'Praat autocorrelation, 10 ms, 50–1000 Hz; estimates can contain octave errors'}
    if len(voiced):
        result.update(zip(('p10_hz', 'median_hz', 'p90_hz'), map(float, np.percentile(voiced, [10, 50, 90]))))
        result['range_semitones'] = float(12 * np.log2(result['p90_hz'] / result['p10_hz']))
    return result


def analyze_file(path):
    # Run in the existing RVC runtime, which already includes Praat/parselmouth.
    import soundfile as sf
    import parselmouth
    audio, rate = sf.read(str(path), dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if not 1 <= len(audio) / rate <= 15:
        raise ValueError('Use a sample between 1 and 15 seconds.')
    if not np.isfinite(audio).all(): raise ValueError('Audio contains invalid samples.')
    pitch = parselmouth.Sound(audio, sampling_frequency=rate).to_pitch_ac(
        time_step=.01, voicing_threshold=.6, pitch_floor=50, pitch_ceiling=1000)
    return summarize(audio, rate, pitch.selected_array['frequency'])


def require_speech(analysis):
    if analysis['voiced_frames'] < 30 or analysis['voiced_fraction'] < .15:
        raise ValueError('Too little voiced speech to estimate pitch. Record several clear sentences without whispering.')
    if analysis['clipped_fraction'] > .02:
        raise ValueError('The recording clips heavily. Lower microphone gain and record again.')


def candidate_pitches(source, current=0, reference=None, supports_f0=True):
    require_speech(source)
    if not supports_f0: return [0]
    center = 0
    if reference:
        require_speech(reference)
        center = round(12 * np.log2(reference['median_hz'] / source['median_hz']))
    values = {0, int(current)} | {center + delta for delta in (-6, -3, 0, 3, 6)}
    return sorted({max(-24, min(24, value)) for value in values})


def rank_candidates(source, candidates, reference=None):
    """Rank measurable defects and pitch tracking; never label a score naturalness."""
    rows = []
    for item in candidates:
        measurement = item['analysis']
        expected = source['median_hz'] * 2 ** (item['pitch'] / 12)
        median = measurement.get('median_hz')
        tracking = abs(12 * np.log2(median / expected)) if median else 99.
        loss = max(0., source['voiced_fraction'] - measurement['voiced_fraction'])
        mismatch = abs(12 * np.log2(median / reference['median_hz'])) if reference and median else 0.
        # Lexicographic measurements keep the reasons visible; no invented AI confidence.
        order = (measurement['voiced_frames'] < 30,
                 measurement['clipped_fraction'] > .005,
                 round(loss, 1), round(mismatch, 1), round(tracking, 1), abs(item['pitch']))
        rows.append(dict(item, tracking_error_semitones=float(tracking),
                         voiced_fraction_loss=float(loss), reference_distance_semitones=float(mismatch),
                         rank_values=order))
    return sorted(rows, key=lambda row: row['rank_values'])


def cache_key(sample, model_hash, index_hash, settings, pitches, reference=None):
    digest = hashlib.sha256(Path(sample).read_bytes()).hexdigest()
    value = [1, digest, model_hash, index_hash, settings, pitches, reference]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def convert_candidates(client, sample, folder, settings, pitches, source, cancel, progress=lambda text: None, reference=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, pitch in enumerate(pitches, 1):
        if cancel.is_set(): return []
        progress(f'Comparing pitch {pitch:+d} semitones, {number} of {len(pitches)}.')
        target = folder / f'pitch-{pitch:+d}.wav'
        metadata = target.with_suffix('.json')
        if target.exists() and not metadata.exists():
            # Preserve interrupted output and create a fresh, exclusive conversion.
            from uuid import uuid4
            target = folder / f'pitch-{pitch:+d}-{uuid4().hex[:8]}.wav'
            metadata = target.with_suffix('.json')
        if target.exists() and metadata.exists():
            measured = json.loads(metadata.read_text())
        else:
            params = dict(settings, pitch=pitch, seed=1729)
            client.request('convert', input=str(sample), output=str(target), **params)
            if cancel.is_set(): return []
            measured = client.request('analyze', input=str(target))
            from voice_library import atomic_json
            atomic_json(metadata, measured)
        rows.append({'pitch': pitch, 'path': str(target), 'analysis': measured})
    return [] if cancel.is_set() else rank_candidates(source, rows, reference)
