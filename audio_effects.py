"""Opt-in batch speech effects. Empty settings preserve the signal exactly."""
import numpy as np
from scipy import signal

DEFAULTS = {'input_gain_db': 0.0, 'noise_reduction': False, 'output_gain_db': 0.0,
            'normalize': False, 'compression': False, 'limiter': False, 'reverb': 'Off',
            'singing_mode': False, 'pitch_correction': False, 'correction_strength': 0.0,
            'musical_key': 'C', 'scale': 'Chromatic'}
KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SCALES = {'Chromatic': list(range(12)), 'Major': [0, 2, 4, 5, 7, 9, 11],
          'Minor': [0, 2, 3, 5, 7, 8, 10]}


def settings(value=None):
    if value is not None and not isinstance(value, dict): raise ValueError('Invalid effects settings.')
    result = dict(DEFAULTS, **(value or {}))
    for key in ('input_gain_db', 'output_gain_db'):
        if type(result[key]) not in (int, float) or not -24 <= result[key] <= 12:
            raise ValueError('Audio gain must be between -24 and +12 dB.')
    for key in ('noise_reduction', 'normalize', 'compression', 'limiter', 'singing_mode', 'pitch_correction'):
        if type(result[key]) is not bool: raise ValueError('Invalid effects switch.')
    if (result['reverb'] not in ('Off', 'Small room', 'Wrestling arena')
            or result['musical_key'] not in KEYS or result['scale'] not in SCALES
            or type(result['correction_strength']) not in (int, float) or not 0 <= result['correction_strength'] <= 1):
        raise ValueError('Invalid reverb or musical pitch settings.')
    return result


def input_audio(audio, rate, preferences):
    prefs = settings(preferences)
    data = np.asarray(audio, dtype='float32')
    warnings = []
    if np.max(np.abs(data)) >= .999:
        warnings.append('Input clipping detected; reduce microphone level or input gain.')
    if prefs['input_gain_db']:
        data = data * np.float32(10 ** (prefs['input_gain_db'] / 20))
    if prefs['noise_reduction'] and len(data) > 1024:
        _, _, spectrum = signal.stft(data, rate, nperseg=512, noverlap=256)
        magnitude = np.abs(spectrum)
        floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        gain = np.clip(1 - .25 * floor / np.maximum(magnitude, 1e-8), .65, 1)
        _, filtered = signal.istft(spectrum * gain, rate, nperseg=512, noverlap=256)
        data = filtered[:len(data)].astype('float32')
    if np.max(np.abs(data)) > 1: warnings.append('Input gain exceeds full scale; the RVC input normalizer will reduce its peak.')
    return data, warnings


def output_audio(audio, rate, preferences):
    prefs = settings(preferences)
    data = np.asarray(audio, dtype='float32')
    if not any((prefs['output_gain_db'], prefs['normalize'], prefs['compression'], prefs['limiter'], prefs['reverb'] != 'Off')):
        return data, []
    if prefs['output_gain_db']: data = data * np.float32(10 ** (prefs['output_gain_db'] / 20))
    if prefs['normalize']:
        rms = float(np.sqrt(np.mean(data.astype('float64') ** 2)))
        if rms > .001: data = data * np.float32(min(4, .1 / rms))
    if prefs['compression']:
        # Gentle static 2:1 curve, followed by a 5 ms control smoother.
        envelope = np.maximum(np.abs(data), 1e-8)
        target = np.minimum(1, np.sqrt(.126 / envelope))
        alpha = np.exp(-1 / (rate * .005))
        gain = signal.lfilter([1 - alpha], [1, -alpha], target, zi=[target[0] * alpha])[0]
        data = (data * gain).astype('float32')
    if prefs['reverb'] != 'Off':
        taps = [(.023, .22), (.041, .14), (.073, .08)] if prefs['reverb'] == 'Small room' else [
            (.045, .16), (.083, -.13), (.121, .11), (.173, .08), (.229, -.06), (.311, .04), (.43, .02)]
        wet = np.zeros(len(data) + int(rate * taps[-1][0]), dtype='float32')
        wet[:len(data)] = data
        for delay, gain in taps:
            offset = int(rate * delay)
            wet[offset:offset + len(data)] += data * np.float32(gain)
        data = wet
    warnings = []
    peak = float(np.max(np.abs(data)))
    if peak > .98:
        if prefs['limiter']:
            # Transparent global attenuation for batch output; no lookahead or truncation.
            data = data * np.float32(.98 / peak)
        elif peak > 1:
            warnings.append('Output would clip. Reduce output gain or enable the sample-peak limiter.')
    return data, warnings


def correct_pitch(frequency, preferences):
    prefs = settings(preferences)
    if not prefs['singing_mode'] or not prefs['pitch_correction'] or not prefs['correction_strength']:
        return frequency
    result = frequency.copy()
    voiced = result > 0
    notes = 69 + 12 * np.log2(result[voiced] / 440)
    allowed = [octave * 12 + (KEYS.index(prefs['musical_key']) + step) % 12
               for octave in range(-2, 13) for step in SCALES[prefs['scale']]]
    allowed = np.asarray(allowed)
    nearest = allowed[np.argmin(np.abs(notes[:, None] - allowed[None, :]), axis=1)]
    corrected = notes + prefs['correction_strength'] * (nearest - notes)
    result[voiced] = 440 * 2 ** ((corrected - 69) / 12)
    return result
