"""Resolve persisted Windows endpoint IDs without persisting PortAudio numbers."""
from dataclasses import dataclass
from math import gcd

import numpy as np
import sounddevice as sd
from pycaw.pycaw import AudioUtilities
from scipy.signal import resample_poly
from threading import Lock


@dataclass(frozen=True)
class Endpoint:
    id: str
    name: str
    direction: str


def endpoints():
    return [
        Endpoint(d.id, d.FriendlyName,
                 "input" if d.id.startswith("{0.0.1.") else "output")
        for d in AudioUtilities.GetAllDevices(device_state=1)
    ]


def resolve(endpoint_id, direction):
    matches = [d for d in endpoints() if d.id == endpoint_id and d.direction == direction]
    if len(matches) != 1:
        raise ValueError("Selected device is disconnected or unavailable. Refresh devices.")
    endpoint = matches[0]
    hosts = sd.query_hostapis()
    devices = [d for d in sd.query_devices()
               if hosts[d['hostapi']]['name'] == 'Windows WASAPI'
               and d['name'] == endpoint.name and d[f'max_{direction}_channels'] > 0]
    same_names = [d for d in endpoints() if d.name == endpoint.name and d.direction == direction]
    if len(devices) != 1 or len(same_names) != 1:
        raise ValueError("Device name cannot be mapped uniquely to WASAPI. No audio was routed.")
    return endpoint, devices[0]


def resample(audio, source_rate, target_rate):
    if source_rate == target_rate:
        return audio
    divisor = gcd(int(source_rate), int(target_rate))
    return resample_poly(audio, int(target_rate) // divisor,
                         int(source_rate) // divisor, axis=0).astype('float32')


def playback(endpoint_id, audio, rate, cancel=None):
    if cancel is not None and cancel.is_set(): return
    _, device = resolve(endpoint_id, 'output')
    target_rate = int(device['default_samplerate'])
    data = resample(np.asarray(audio, dtype='float32'), rate, target_rate)
    if data.ndim == 1:
        data = data[:, None]
    channels = min(2, device['max_output_channels'])
    if data.shape[1] == 1 and channels == 2:
        data = np.repeat(data, 2, axis=1)
    with sd.OutputStream(device=device['index'], samplerate=target_rate,
                         channels=data.shape[1], dtype='float32') as stream:
        # Bounded writes let a voice change discard buffered playback promptly.
        chunk = max(1, target_rate // 50)
        for start in range(0, len(data), chunk):
            if cancel is not None and cancel.is_set():
                stream.abort()
                return
            stream.write(data[start:start + chunk])
        if cancel is not None and cancel.is_set(): stream.abort()


def microphone_test(endpoint_id, seconds=3):
    endpoint, device = resolve(endpoint_id, 'input')
    if any(word in endpoint.name.casefold() for word in ('cable', 'stereo mix', 'line 1')):
        raise ValueError("Select a physical microphone to prevent a routing loop.")
    rate = int(device['default_samplerate'])
    with sd.InputStream(device=device['index'], samplerate=rate,
                        channels=1, dtype='float32') as stream:
        audio, overflow = stream.read(int(rate * seconds))
    if overflow:
        raise RuntimeError("Microphone overflow during test. Try again.")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return audio, rate, peak, rms


class Recorder:
    """Bounded mono capture at the microphone's native shared-mode rate."""
    def __init__(self, endpoint_id, maximum_seconds=60):
        endpoint, device = resolve(endpoint_id, 'input')
        if any(word in endpoint.name.casefold() for word in ('cable', 'stereo mix', 'line 1')):
            raise ValueError('Choose a physical microphone to prevent a routing loop.')
        self.rate = int(device['default_samplerate'])
        self.maximum_frames = self.rate * maximum_seconds
        self.frames = 0
        self.blocks = []
        self.errors = []
        self.lock = Lock()
        self.stream = sd.InputStream(device=device['index'], samplerate=self.rate,
                                     channels=1, dtype='float32', callback=self.callback)

    def callback(self, data, frames, timing, status):
        with self.lock:
            if status:
                self.errors.append(str(status))
                raise sd.CallbackAbort
            if self.frames + frames > self.maximum_frames:
                self.errors.append('Recording exceeded the 60-second limit. Record a shorter utterance.')
                raise sd.CallbackAbort
            self.blocks.append(data.copy())
            self.frames += frames

    def start(self):
        try:
            self.stream.start()
        except Exception:
            self.stream.close()
            raise

    def stop(self):
        try:
            self.stream.stop()
        finally:
            self.stream.close()
        if self.errors:
            raise RuntimeError('Microphone recording failed: ' + '; '.join(self.errors[:3]))
        if not self.blocks:
            raise ValueError('No microphone audio was captured.')
        audio = np.concatenate(self.blocks)
        if len(audio) < self.rate / 10:
            raise ValueError('Recording is too short. Hold the key longer.')
        if np.max(np.abs(audio)) < 0.0001:
            raise ValueError('No microphone signal detected. Check mute and input selection.')
        return audio, self.rate


def playback_many(endpoint_ids, audio, rate, cancel=None):
    from concurrent.futures import ThreadPoolExecutor
    import comtypes
    def play(endpoint_id):
        comtypes.CoInitialize()
        try:
            playback(endpoint_id, audio, rate, cancel)
        finally:
            comtypes.CoUninitialize()
    unique = list(dict.fromkeys(endpoint_ids))
    with ThreadPoolExecutor(max_workers=max(1, len(unique))) as pool:
        results = [pool.submit(play, item) for item in unique]
        errors = []
        for result in results:
            try:
                result.result()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('Playback routing failed: ' + '; '.join(errors))
