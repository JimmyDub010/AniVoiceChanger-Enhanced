"""JSON-lines worker for the downloaded legacy RVC runtime. No upstream edits."""
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


def emit(value):
    sys.__stdout__.write(json.dumps(value) + '\n')
    sys.__stdout__.flush()


class Engine:
    def load(self, request):
        import torch
        from fairseq import checkpoint_utils
        from vc_infer_pipeline import VC
        try:
            from lib.infer_pack import models
        except ModuleNotFoundError:
            from infer_pack import models
        model_path = Path(request['model']).resolve()
        if model_path.suffix.lower() != '.pth' or not model_path.is_file():
            raise ValueError('Choose an existing RVC .pth model.')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in the RVC runtime. CPU conversion has not been validated.')
        device = 'cuda:0'
        self.half = request.get('half', True)
        self.config = SimpleNamespace(device=device, is_half=self.half,
                                      x_pad=3 if self.half else 1, x_query=10 if self.half else 6,
                                      x_center=60 if self.half else 38, x_max=65 if self.half else 41)
        # Imported/downloaded voice checkpoints never enter a Python unpickler.
        from safe_checkpoint import SafeCheckpoint
        checkpoint = SafeCheckpoint(model_path).materialize()
        if not isinstance(checkpoint, dict) or not {'config', 'weight'} <= checkpoint.keys():
            raise ValueError('This is not an RVC inference checkpoint (config and weight are required).')
        self.version = checkpoint.get('version', 'v1')
        self.f0 = checkpoint.get('f0', 1)
        if self.version not in ('v1', 'v2') or self.f0 not in (0, 1):
            raise ValueError('Only legacy RVC v1/v2 F0 and non-F0 architectures are supported by this adapter.')
        config = list(checkpoint['config'])
        self.rate = int(config[-1])
        if self.rate not in (32000, 40000, 48000):
            raise ValueError('Unexpected model sample rate: ' + str(self.rate))
        config[-3] = checkpoint['weight']['emb_g.weight'].shape[0]
        feature_size = 256 if self.version == 'v1' else 768
        name = f'SynthesizerTrnMs{feature_size}NSFsid' + ('' if self.f0 else '_nono')
        constructor = getattr(models, name)
        self.net = constructor(*config, **({'is_half': self.half} if self.f0 else {}))
        del self.net.enc_q
        incompatibility = self.net.load_state_dict(checkpoint['weight'], strict=False)
        if incompatibility.missing_keys or incompatibility.unexpected_keys:
            raise ValueError('Checkpoint does not match the selected RVC architecture: ' + str(incompatibility))
        self.net.eval().to(device)
        self.net = self.net.half() if self.half else self.net.float()
        ensemble, _, _ = checkpoint_utils.load_model_ensemble_and_task(['hubert_base.pt'], suffix='')
        self.hubert = ensemble[0].to(device).eval()
        self.hubert = self.hubert.half() if self.half else self.hubert.float()
        self.vc = VC(self.rate, self.config)
        self.model_path = model_path
        self.index_cache = None
        del checkpoint, ensemble
        torch.cuda.synchronize()
        return {'model': str(model_path), 'version': self.version, 'f0': self.f0,
                'sample_rate': self.rate, 'feature_dimensions': feature_size,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(), 'half': self.half}

    def convert(self, request):
        import faiss
        import librosa
        import numpy as np
        import soundfile as sf
        import torch
        call_started = time.perf_counter()
        cpu_started = time.process_time()
        if not hasattr(self, 'vc'):
            raise ValueError('Load a model first.')
        if 'seed' in request:
            import random
            seed = request['seed']
            if type(seed) is not int or not 0 <= seed < 2 ** 32: raise ValueError('Invalid comparison seed.')
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        source = Path(request['input']).resolve()
        target = Path(request['output']).resolve()
        if source == target or target.exists():
            raise ValueError('Conversion requires a new output file; existing recordings are never overwritten.')
        pitch = int(request.get('pitch', 12))
        if not -24 <= pitch <= 24:
            raise ValueError('Pitch must be between -24 and +24 semitones.')
        if pitch and not self.f0:
            raise ValueError('This non-F0 model does not support pitch shifting. Set pitch to zero.')
        method = request.get('method', 'crepe')
        requested_method = method
        experimental = ('mangio-crepe-80', 'rmvpe-range', 'hybrid-rmvpe-fcpe')
        if method not in ('crepe', 'pm', 'harvest', 'rmvpe', 'rmvpe-new', 'fcpe-experimental', 'swiftf0') + experimental:
            raise ValueError('Unsupported pitch extraction method.')
        if method in ('rmvpe-range', 'hybrid-rmvpe-fcpe'):
            # Both adapters are explicitly based on the current RMVPE checkpoint.
            if getattr(self, 'pitch_checkpoint', None) != 'classic' and hasattr(self.vc, 'model_rmvpe'):
                del self.vc.model_rmvpe
            self.pitch_checkpoint = 'classic'
        if method in ('rmvpe', 'rmvpe-new'):
            selected = 'new' if method == 'rmvpe-new' else 'classic'
            if getattr(self, 'pitch_checkpoint', None) != selected:
                if hasattr(self.vc, 'model_rmvpe'): del self.vc.model_rmvpe
                if selected == 'new':
                    import hashlib
                    from rmvpe import RMVPE
                    path = Path(__file__).resolve().parent / 'experiments/rmvpe-2026.pt'
                    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != '6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193':
                        raise ValueError('The tested RMVPE New checkpoint is missing or has changed.')
                    self.vc.model_rmvpe = RMVPE(str(path), is_half=self.half, device='cuda:0')
                self.pitch_checkpoint = selected
            method = 'rmvpe'
        if method == 'rmvpe' and not Path('rmvpe.pt').is_file():
            raise ValueError('rmvpe.pt is missing from this RVC installation.')
        influence = float(request.get('index_rate', 0))
        rms = float(request.get('rms_mix', 0.25))
        if not 0 <= influence <= 1 or not 0 <= rms <= 1:
            raise ValueError('Index influence and volume envelope must be between zero and one.')
        index_path = request.get('index', '')
        original_read = faiss.read_index
        stats = {'search_calls': 0, 'searched_frames': 0, 'enabled': influence > 0,
                 'cache_hit': False, 'cached_vector_bytes': 0}
        audited_index = None
        if influence:
            if not index_path or not Path(index_path).is_file():
                raise ValueError('Index influence is enabled but no matching .index file is available.')
            prepare_start = time.perf_counter()
            path = Path(index_path).resolve()
            metadata = path.stat()
            key = (str(path), metadata.st_size, metadata.st_mtime_ns)
            cached = getattr(self, 'index_cache', None)
            if cached is not None and cached['key'] == key:
                index, vectors = cached['index'], cached['vectors']
                stats['cache_hit'] = True
            else:
                self.index_cache = None
                index = original_read(str(path))
                vectors = None
            if index.d != (256 if self.version == 'v1' else 768) or index.ntotal < 8:
                raise ValueError('Index dimensions or vector count are incompatible with this model.')
            if vectors is None:
                vectors = index.reconstruct_n(0, index.ntotal)
                vectors.setflags(write=False)
                # One entry, at most 256 MiB of reconstructed host vectors.
                # The FAISS object itself uses additional host RAM, measured separately.
                if vectors.nbytes <= 256 * 1024 * 1024:
                    self.index_cache = {'key': key, 'index': index, 'vectors': vectors}
            stats['cached_vector_bytes'] = vectors.nbytes if self.index_cache is not None else 0
            stats['prepare_seconds'] = time.perf_counter() - prepare_start
            stats.update(dimensions=index.d, vectors=index.ntotal, path=str(index_path))
            class AuditedIndex:
                def __getattr__(self, name):
                    return getattr(index, name)

                def search(self, features, k):
                    stats['search_calls'] += 1
                    stats['searched_frames'] += len(features)
                    return index.search(features, k)

                def reconstruct_n(self, start, count):
                    if start == 0 and count == index.ntotal:
                        return vectors
                    return index.reconstruct_n(start, count)
            audited_index = AuditedIndex()
        else:
            self.index_cache = None
        start = time.perf_counter()
        audio, input_rate = sf.read(str(source), dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        if len(audio) < input_rate / 10:
            raise ValueError('Recording is too short. Speak for at least a tenth of a second.')
        if not np.isfinite(audio).all():
            raise ValueError('Input contains non-finite samples.')
        from audio_effects import input_audio, output_audio, correct_pitch, settings as effect_settings
        effects = effect_settings(request.get('effects'))
        audio, audio_warnings = input_audio(audio, input_rate, effects)
        if input_rate != 16000:
            audio = librosa.resample(audio, orig_sr=input_rate, target_sr=16000)
        peak = np.max(np.abs(audio)) / 0.95
        if peak > 1:
            audio /= peak
        timings = [0, 0, 0]
        torch.cuda.reset_peak_memory_stats()
        original_f0 = self.vc.get_f0
        pitch_function = original_f0
        def coarse_pitch(frequency):
            mel = 1127 * np.log(1 + frequency / 700)
            low, high = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
            voiced = mel > 0
            mel[voiced] = (mel[voiced] - low) * 254 / (high - low) + 1
            return np.rint(np.clip(mel, 1, 255)).astype(np.int64)
        fcpe_pitch = None
        if requested_method in ('fcpe-experimental', 'hybrid-rmvpe-fcpe') and self.f0:
            if not hasattr(self, 'fcpe'):
                import hashlib
                package_root = Path(__file__).resolve().parent / 'experiments/fcpe/packages'
                checkpoint = package_root / 'torchfcpe/assets/fcpe_c_v001.pt'
                if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != 'b9aeaeb673436eeda50ceafd632aa681aa63417e52eae4207503d180c9b10015':
                    raise ValueError('The isolated experimental FCPE checkpoint is missing or changed.')
                sys.path.insert(0, str(package_root))
                from torchfcpe import spawn_bundled_infer_model
                self.fcpe = spawn_bundled_infer_model('cuda:0')
            def fcpe_pitch(input_path, waveform, length, shift, algorithm, filter_radius, inp_f0=None):
                frequency = self.fcpe.infer(torch.from_numpy(waveform).float()[None, :], sr=16000,
                                          interp_uv=False, output_interp_target_length=length).reshape(-1).detach().cpu().numpy()
                frequency = frequency * (2 ** (shift / 12))
                return coarse_pitch(frequency), frequency
            if requested_method == 'fcpe-experimental':
                pitch_function = fcpe_pitch
                self.vc.get_f0 = pitch_function
        adapter_details = {}
        if requested_method == 'swiftf0' and self.f0:
            from swiftf0_adapter import adapter as swift_adapter
            pitch_function = swift_adapter(self)
            self.vc.get_f0 = pitch_function
            adapter_details = {'adapter': 'swiftf0', 'version': '0.3.0', 'provider': 'CPUExecutionProvider',
                               'confidence_threshold': .5, 'input_rate': 16000, 'native_hop_seconds': .016,
                               'rvc_hop_seconds': .01, 'f0_min': 50, 'f0_max': 1100}
        if requested_method in experimental:
            from pitch_methods import adapter
            pitch_function, adapter_details = adapter(self, requested_method, original_f0,
                                                       request.get('pitch_lab_options', {}), fcpe_pitch)
            self.vc.get_f0 = pitch_function
        if effects['singing_mode'] and effects['pitch_correction'] and effects['correction_strength']:
            def corrected_f0(*args, **kwargs):
                coarse, frequency = pitch_function(*args, **kwargs)
                frequency = correct_pitch(frequency, effects)
                return coarse_pitch(frequency), frequency
            self.vc.get_f0 = corrected_f0
        pitch_metrics = {'seconds': 0., 'calls': 0}
        measured_pitch = self.vc.get_f0
        def measure_pitch(*args, **kwargs):
            torch.cuda.synchronize()
            began = time.perf_counter()
            value = measured_pitch(*args, **kwargs)
            torch.cuda.synchronize()
            pitch_metrics['seconds'] += time.perf_counter() - began
            pitch_metrics['calls'] += 1
            frequency = np.asarray(value[1])
            pad = self.vc.x_pad * 100
            central = frequency[pad:-pad] if pad and len(frequency) > pad * 2 else frequency
            valid = np.isfinite(central) & (central > 0)
            voiced = central[valid]
            pitch_metrics.update(frames=int(len(central)), voiced_fraction=float(valid.mean()) if len(central) else 0.,
                                 invalid_frames=int((~np.isfinite(central)).sum()),
                                 median_hz=float(np.median(voiced)) if len(voiced) else None,
                                 abrupt_octave_transitions=int(np.sum(np.abs(np.diff(np.log2(np.maximum(central, 1e-6))))[valid[1:] & valid[:-1]] > 1)) if len(central) > 1 else 0,
                                 scope='F0 time includes reflected inference padding; statistics exclude that padding. Voicing transitions are not necessarily errors.')
            return value
        if request.get('measure_pitch', False): self.vc.get_f0 = measure_pitch
        try:
            if audited_index is not None:
                faiss.read_index = lambda path: audited_index
            with torch.inference_mode():
                result = self.vc.pipeline(self.hubert, self.net, 0, audio, str(source), timings,
                                          pitch, method, index_path, influence, self.f0, 3,
                                          self.rate, 0, rms, self.version, 0.33, f0_file=None)
            torch.cuda.synchronize()
        finally:
            self.vc.get_f0 = original_f0
            faiss.read_index = original_read
            # Harvest caches otherwise retain every PTT utterance for the worker's lifetime.
            from vc_infer_pipeline import input_audio_path2wav, cache_harvest_f0
            input_audio_path2wav.clear()
            cache_harvest_f0.cache_clear()
        if influence and not stats['search_calls']:
            raise RuntimeError('Index was enabled but no retrieval search participated in conversion.')
        if not len(result) or not np.isfinite(result).all():
            raise RuntimeError('RVC returned empty or invalid audio.')
        # Preserve original PCM exactly when output effects are all disabled.
        if any((effects['output_gain_db'], effects['normalize'], effects['compression'], effects['limiter'], effects['reverb'] != 'Off')):
            processed, warnings = output_audio(result.astype('float32') / 32768, self.rate, effects)
            audio_warnings.extend(warnings)
            result = np.rint(np.clip(processed, -1, 32767 / 32768) * 32768).astype(np.int16)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects both existing output and input files.
        with target.open('xb') as file:
            sf.write(file, result, self.rate, format='WAV', subtype='PCM_16')
        return {'output': str(target), 'sample_rate': self.rate, 'pitch': pitch,
                'method': requested_method, 'effects': effects, 'warnings': audio_warnings,
                'index': stats, 'input_seconds': len(audio) / 16000,
                'output_seconds': len(result) / self.rate, 'processing_seconds': time.perf_counter() - start,
                'pipeline_stage_seconds': timings,
                'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_vram_bytes': torch.cuda.max_memory_reserved(),
                'total_conversion_seconds': time.perf_counter() - call_started,
                'cpu_seconds': time.process_time() - cpu_started,
                'pitch_metrics': pitch_metrics, 'adapter_details': adapter_details,
                'latency_note': 'File processing measurement; not microphone-to-output latency.'}


def main():
    engine_dir = Path(sys.argv[1]).resolve()
    os.chdir(engine_dir)
    sys.path.insert(0, str(engine_dir))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    engine = Engine()
    emit({'ready': True})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('command') == 'quit':
                break
            with contextlib.redirect_stdout(sys.stderr):
                if request.get('command') == 'load':
                    engine = Engine()
                    result = engine.load(request)
                elif request.get('command') == 'convert':
                    result = engine.convert(request)
                elif request.get('command') == 'analyze':
                    from pitch_calibration import analyze_file
                    result = analyze_file(request['input'])
                else:
                    raise ValueError('Unknown worker command.')
            emit({'ok': True, 'result': result})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit({'ok': False, 'error': str(error)})


if __name__ == '__main__':
    main()
