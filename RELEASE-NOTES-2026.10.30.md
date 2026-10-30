# AniVoiceChanger Enhanced v2026.10.30

Windows record-and-convert RVC application with native keyboard controls, a
persistent voice library and global F8 push-to-talk. This release preserves the
legacy inference architecture and adds the completed Experimental Pitch Lab,
SwiftF0 0.3.0 and consistent CPU/GPU extraction labels.

## Install

Download AniVoiceChanger-Enhanced-2026.10.30-Windows-x64.zip, extract the entire
archive to a short writable folder such as C:\AniVoiceChanger, then run
Launch-Enhanced.cmd. Desktop Python, SwiftF0, RMVPE New and FCPE pitch support
assets/dependencies are bundled with notices. No separate SwiftF0 installation.

**Supply a compatible legacy RVC-beta0717 engine and your own voice models
separately. Full conversion requires NVIDIA CUDA.** CPU-only pitch extraction
(Praat, Harvest, SwiftF0) does not provide CPU-only RVC synthesis. See README
for the pinned engine download/setup instructions and complete device table.

## Included

- ZIP/extracted .pth imports, optional paired .index, safe library copies and
  persistent per-voice pitch/method/index/effects settings.
- F8 hold/release conversion, optional toggle recording, cancellation and
  release-event recovery; independent monitoring and optional VB-CABLE routing.
- Alt+L Pitch Lab: identical recording/settings across selected methods,
  saved labeled outputs/measurements, preview, cancellation and optional favorite.
- SwiftF0, existing RMVPE/New, CREPE Full, FCPE, Mangio-style CREPE, constrained
  RMVPE range, hybrid, Praat and Harvest. Device labels reflect the installed
  execution paths. No defaults, inference algorithm or dependency versions
  were changed merely to add device labels.
- Native accessible controls and NVDA status bridge; Ctrl+Tab page navigation,
  Alt+P calibration and stable focus during asynchronous work.

## Verification and limitations

74 unit regression tests passed; all ten methods completed actual v1/v2 audio
conversions. SwiftF0/RMVPE New were compared at 0/-12/+12. CUDA-hidden extraction
tests passed for the three CPU extractors, while full synthesis correctly
reported its CUDA requirement. Model/index loading, real FAISS participation,
switching/persistence, cancellation and worker recovery have automated and real
inference coverage. Full extraction labels were read using Windows MSAA.

Final release checks also named the application-page selector and added bounded
atomic settings-save retries/error reporting for an observed transient Windows
file lock. A single down/up hook registration fixes keyboard-library cleanup
errors on disable/close. These fixes do not change the inference engine.

The final F8 press/release was performed by the user and reached the real
Windows hook, producing actual CUDA conversion with generated test input and
no playback/chat transmission. All 165 audited actionable controls have names;
30 full method labels were read through MSAA. The sealed Classic fallback's
61,760 files match its manifest, and its unchanged sources/environment completed
GUI/model/index/conversion checks with writes redirected outside the fallback.

Release packaging is tested from a short clean extraction path, with SHA-256
manifests/checksums and a scan excluding personal voices, indexes, recordings,
settings and credentials. Tests use private/generated samples which are not
distributed. No claim of superior SwiftF0 sound quality is made.

Actual NVDA speech verification remains manual when NVDA is unavailable. Please
check tab order, full labels, import dialogs, Ready/error announcements, F8 and
Lab cancellation/playback. Second-PC and all-game compatibility are not claimed.
Global hooks may be restricted by games or elevated applications; no anti-cheat
workarounds are attempted. Hardware microphone/game audio verification is
distinct from tests using generated input.

This is batch push-to-talk, **not continuous real-time conversion**. Isolated
streaming prototypes are not included or promoted. Preserve your Classic
installation/fallback separately: this release does not migrate or overwrite it.

Voice models/indexes must be obtained separately with suitable permissions.
Pitch support assets retain their own verified licenses; see ACKNOWLEDGMENTS.md
and PITCH-ASSET-MANIFEST.json. Verify downloads using SHA256SUMS.txt.
