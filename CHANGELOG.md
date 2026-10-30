# Changelog

## 2026.10.30

- SwiftF0 0.3.0, Experimental Pitch Lab, and consistent CPU/GPU extraction labels.
- Reviewed pitch asset bundles and retained attribution; no personal voice assets.
- Release-blocking page-name, atomic-save and global hook cleanup fixes.
- 74 regression tests, v1/v2 real inference, MSAA control checks and direct F8 verification.
- Full synthesis still requires NVIDIA CUDA; NVDA auditory verification remains manual.

## 2026.10.27

Initial public AniVoiceChanger Enhanced release, based on AniVoiceChanger.

- NVDA status announcements and named native Windows controls.
- Global hold-to-talk F8, release recovery, time limits and safe cancellation.
- Real microphone capture, selectable output and optional headphone monitoring.
- Persistent voice library with ZIP/extracted imports, settings and relinking.
- Voice-switch generation checks, serialized worker replacement and cancellable playback.
- Pitch extraction methods saved per voice and restored after restart; missing
  methods inherit the current selection, with no silent engine fallback.
- Focus preservation around recording and processing.
- Optional experimental pitch analysis and listening-based calibration.
- Model-free portable Windows and source distributions.
