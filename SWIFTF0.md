# SwiftF0 0.3.0

Select **SwiftF0 0.3.0 (CPU only)** in the normal Pitch extraction method control,
Edit Voice Settings, or Alt+L Experimental Pitch Lab. Normal selection saves the
method for the selected voice; comparisons change no preference unless you
explicitly save a favorite. RMVPE New, FCPE and the other adapters remain available
in the existing Enhanced setup. No defaults are changed.

The official 0.3.0 API is SwiftF0(threads=2, spin=False).detect(audio, 16000,
fmin=50, fmax=1100). The detector is cached lazily in the voice worker and uses
CPUExecutionProvider; RVC synthesis remains on CUDA. The application already
mixes input to mono and resamples to 16 kHz using its existing audio path, so no
extra SwiftF0 resampling dependency is needed. Extraction operates on the same
reflected audio padding as the other methods.

The adapter aligns the returned 16 ms timestamps to RVC's 10 ms frame positions;
it does not stretch the contour to the recording length. Confidence >=0.5 is
voiced, with finite pitch in the native RVC 50–1100 Hz extraction range. Voiced
runs use log-frequency interpolation; unvoiced transitions use the nearest frame
without bridging silence. Official silence detection remains active. Pitch shift
is applied afterward, then continuous F0 and clipped 1–255 coarse bins are passed
to the original generator. It does not add note quantization, smoothing, gain
normalization or pitch correction. Existing optional effects retain their settings.

SwiftF0 is included offline in both prepared ZIP formats. The bundle targets
the supported RVC Python 3.9 Windows x64 runtime and includes ONNX Runtime 1.19.2
without upgrading the engine's installed 1.14.1 package. No background inference,
microphone use, model downloads or service requests occur at startup. Bundle
integrity is checked before first use. Missing files, a conflicting already-loaded
ONNX Runtime or an incompatible engine produce an error, never a silent fallback.
Restore the complete package if files are missing; ordinary users do not run pip.

Official references: [API/source](https://github.com/lars76/swift-f0/tree/v0.3.0),
[dependencies](https://github.com/lars76/swift-f0/blob/v0.3.0/pyproject.toml),
[license](https://github.com/lars76/swift-f0/blob/v0.3.0/LICENSE),
[shipped-model provenance](https://github.com/lars76/swift-f0-training).
See ACKNOWLEDGMENTS.md for preserved licenses and BUNDLE.json for hashes.

Listening comparisons and validation status are recorded in RELEASE-NOTES-2026.10.30.md.
No claim that SwiftF0 sounds better than RMVPE is made.
