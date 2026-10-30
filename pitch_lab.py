"""Persistent same-input comparison runs. No microphone/GPU work on import."""
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4

import numpy as np
import soundfile as sf
from voice_library import atomic_json, file_hash


class LabCancelled(Exception):
    pass


def snapshot_audio(source, run):
    source, run = Path(source).resolve(), Path(run).resolve()
    info = sf.info(source)
    if not 1 <= info.frames / info.samplerate <= 60:
        raise ValueError('Use an audio sample between one and sixty seconds.')
    if info.channels > 8: raise ValueError('The comparison input has too many channels.')
    target = run / ('input' + source.suffix.lower())
    if target == source or target.exists(): raise ValueError('Comparison input requires a fresh destination.')
    with source.open('rb') as reader, target.open('xb') as writer: shutil.copyfileobj(reader, writer)
    if file_hash(source) != file_hash(target): raise ValueError('Input changed while it was being copied. Try again.')
    audio, rate = sf.read(target, dtype='float32', always_2d=True)
    if not np.isfinite(audio).all(): raise ValueError('Input audio contains non-finite samples.')
    return target


def make_run(root):
    folder = Path(root) / ('comparison-' + datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def compare(client, source, run, voice, parameters, methods, cancel, progress=lambda text: None, options=None):
    run, source = Path(run), Path(source)
    if not methods: raise ValueError('Select at least one installed compatible method.')
    if len(set(methods)) != len(methods): raise ValueError('Duplicate extraction methods selected.')
    fixed = copy.deepcopy(parameters)
    fixed.pop('_generation', None)
    fixed.pop('method', None)
    fixed['seed'] = 114514
    fixed['measure_pitch'] = True
    report = {'schema': 1, 'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'voice_id': voice['id'], 'voice_name': voice['name'], 'model_hash': voice.get('model_hash'),
              'index_hash': voice.get('index_hash'), 'input': source.name, 'input_sha256': file_hash(source),
              'parameters': fixed, 'options': options or {}, 'methods': list(methods), 'results': [],
              'quality_note': 'Measurements are not a naturalness score. Audition the saved audio.'}
    atomic_json(run / 'results.json', report)
    for number, method in enumerate(methods, 1):
        if cancel.is_set(): break
        progress(f'Comparing {method}, {number} of {len(methods)}. No automatic playback.')
        target = run / (f'{number:02d}-' + method + '.wav')
        if target.exists(): raise ValueError('Comparison output already exists; create a new run.')
        started = time.perf_counter()
        try:
            result = client.request('convert', input=str(source), output=str(target), method=method,
                                    pitch_lab_options=options or {}, **fixed)
            row = {'method': method, 'status': 'complete', 'path': target.name,
                   'wall_seconds': time.perf_counter() - started, 'conversion': result,
                   'output_sha256': file_hash(target)}
            # Independent acoustic measurement; may report too little voicing rather than invent a score.
            try: row['analysis'] = client.request('analyze', input=str(target))
            except Exception as error: row['analysis_error'] = str(error)
        except Exception as error:
            if cancel.is_set(): break
            row = {'method': method, 'status': 'failed', 'error': str(error),
                   'wall_seconds': time.perf_counter() - started}
        report['results'].append(row)
        atomic_json(run / 'results.json', report)
        progress(f'{method}: ' + ('saved.' if row['status'] == 'complete' else 'failed: ' + row['error']))
    report['status'] = 'cancelled' if cancel.is_set() else 'finished'
    atomic_json(run / 'results.json', report)
    return report


def describe_result(row, parameters):
    if row['status'] != 'complete': return row['method'] + ' failed: ' + row.get('error', 'unknown error')
    conversion = row['conversion']
    metrics = conversion.get('pitch_metrics', {})
    allocation = conversion.get('peak_vram_bytes', 0) / 1024 ** 2
    return (f"{row['method']}. Total worker call {row['wall_seconds']:.3f} seconds; "
            f"pitch extraction {metrics.get('seconds', conversion.get('pipeline_stage_seconds', [0, 0])[1]):.3f} seconds; "
            f"peak allocated GPU memory {allocation:.0f} MiB. Pitch {parameters.get('pitch', 0):+d}; "
            f"index influence {parameters.get('index_rate', 0):.0%}; sample rate {conversion['sample_rate']} Hz. "
            f"Voiced pitch fraction {metrics.get('voiced_fraction', 0):.0%}. "
            'These measurements do not establish naturalness.')


def save_favorite(library, voice_id, method, available, options=None):
    if method not in available: raise ValueError('This method is not currently available; no preference was changed.')
    voice = library.get(voice_id)
    preferences = copy.deepcopy(voice['preferences'])
    preferences['method'] = method
    if method in ('mangio-crepe-80', 'rmvpe-range'):
        preferences['method_options'] = dict(method=method, **(options or {}))
    library.edit(voice_id, preferences=preferences)
