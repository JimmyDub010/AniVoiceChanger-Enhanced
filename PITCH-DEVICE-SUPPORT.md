# Pitch extractor device labels — 30 October 2026

These labels describe the installed **Enhanced extraction implementation**, not
every capability of an upstream library. All selectors use one label mapping
from pitch_methods.py: the main dropdown, Edit Voice Settings and Pitch Lab
checkboxes/results. Internal method keys and saved preferences are unchanged.
Legacy display names remain accepted when restoring older settings.

**The full conversion engine currently requires NVIDIA CUDA.** Selecting a
CPU-only pitch extractor does not enable CPU-only voice conversion. The same
explanation appears in the main routing page, per-voice dialog and Pitch Lab.
No device selector or CPU inference backend was added.

| Display name | Internal key | Installed execution evidence |
| --- | --- | --- |
| RMVPE current (GPU only) | rmvpe | Engine.load fixes config.device to cuda:0; native VC.get_f0 passes that device and precision into RMVPE. |
| CREPE Full (GPU only) | crepe | Native VC.get_f0 passes self.device, inherited cuda:0, to torchcrepe.predict. |
| RMVPE New (GPU only) | rmvpe-new | engine_worker explicitly constructs the alternate RMVPE checkpoint with device='cuda:0'. |
| Praat autocorrelation (CPU only) | pm | Native parselmouth.Sound(...).to_pitch_ac and NumPy padding, without a CUDA pitch operation. |
| Harvest (CPU only) | harvest | Native pyworld.harvest/stonemask and SciPy median filtering, without a CUDA pitch operation. |
| FCPE Experimental (GPU only) | fcpe-experimental | Adapter explicitly spawns the installed FCPE predictor on cuda:0. |
| Mangio-style CREPE Full hop 80 Experimental (GPU only) | mangio-crepe-80 | Adapter passes engine.config.device (cuda:0) to installed torchcrepe. |
| RMVPE constrained range Experimental (GPU only) | rmvpe-range | Adapter constructs RMVPE on engine.config.device and moves its tensor there before salience masking. |
| RMVPE + FCPE agreement hybrid Experimental (GPU only) | hybrid-rmvpe-fcpe | Both constituent predictors take their existing CUDA paths; NumPy blending does not make this a CPU extractor. |
| SwiftF0 0.3.0 (CPU only) | swiftf0 | Bundled official implementation explicitly creates an ONNX Runtime session with CPUExecutionProvider. |

No installed Enhanced method currently offers a selectable CPU/GPU extraction
path, so no label claims “CPU / GPU”. Some libraries, particularly RMVPE and
torchcrepe, accept other devices, but Enhanced does not currently expose or use
those paths. GPU only here means the supported NVIDIA CUDA integration; it does
not assert support for every GPU vendor or deny theoretical library CPU support.

## Verification

- CUDA-disabled subprocess: CUDA_VISIBLE_DEVICES=-1, torch.cuda.is_available()
  false throughout. Praat, Harvest and SwiftF0 executed the installed extraction
  functions on generated speech and a reference tone; CUDA remained uninitialized.
- A real EngineClient in that subprocess rejected model loading with its existing
  CUDA-unavailable error. CPU-only full conversion is not silently advertised.
- These are execution/device tests, not pitch-accuracy benchmarks. Measurements
  retain Harvest's octave-low estimate on the pure 220 Hz tone rather than
  treating it as a quality improvement or changing the algorithm.
- All 30 complete labels were read back through Windows MSAA
  AccessibleObjectFromWindow: accValue for both native dropdowns and accName for
  Lab checkboxes. No text was truncated. This verifies the standard API text
  NVDA can consume, not an auditory NVDA session.
- Unit tests cover current/legacy label-to-key mappings and settings restoration.
  Real v1/v2 conversions exercise every displayed method; main-window method
  switching/restart and Lab focus/cancellation are checked separately.

Evidence: test-results/latest-cpu-extractors.txt,
test-results/device-accessibility-61279aa6/results.json,
test-results/latest-device-labels.txt and the latest methods-* regression report.
No inference algorithm, precision, package version, model/index or user default
was changed for this update. Classic and sealed fallbacks remain untouched.
