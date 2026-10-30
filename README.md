# AniVoiceChanger Enhanced

Native Windows RVC record-and-convert voice changer with a saved voice library,
global F8 push-to-talk, named controls, NVDA status announcements, and an
Experimental Pitch Lab. Based on [AniVoiceChanger](https://github.com/SociallyIneptWeeb/AniVoiceChanger).

## Download and install

Download **AniVoiceChanger-Enhanced-2026.10.30-Windows-x64.zip** from
[GitHub Releases](https://github.com/JimmyDub010/AniVoiceChanger-Enhanced/releases/tag/v2026.10.30).
Extract the entire ZIP to a short, writable path such as `C:\AniVoiceChanger`.
Run **Launch-Enhanced.cmd**. Keep the `app` and `runtime` folders beside it.
Desktop Python is included: no system Python, pip, paid service or developer
tools are needed to run the portable desktop application.

**A compatible legacy RVC engine and your own voice models are required separately.**
The ZIP includes the desktop application and pitch support assets, not an RVC
engine installer. Full RVC conversion requires an NVIDIA CUDA-capable GPU and
compatible driver. Tested on Windows 11, RTX 4070 SUPER 12 GB, legacy Python
3.9.13 / PyTorch 2.0.0+cu118 and RVC v1/v2 F0 models.

1. Obtain the [legacy RVC-beta engine archive](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/e6d0c1a17da07c33557852f9dfa2bd44cc75737d/RVC-beta.7z).
   Extract it separately with 7-Zip. Select its folder containing
   `runtime/python.exe` in **RVC installation folder**. It must contain legacy
   `vc_infer_pipeline.py`, inference architecture modules, HuBERT and native
   RMVPE/CREPE assets and dependencies. The Python 3.12 RVC20260718 engine is not
   a drop-in replacement.
2. In Voice Library, choose **Import Voice from ZIP** or **Import Extracted Voice**.
   Select your trusted `.pth` and optional matching `.index`. Nested ZIP folders
   are searched; ambiguous pairings require selection. Imported copies become
   `model.pth` / `model.index` in unique library directories. Original downloads
   are not renamed, overwritten or deleted. After successful import, the source
   ZIP is no longer required.
3. Choose a Saved Voice, wait for Ready, and select your named physical microphone
   and playback device. Pitch, index influence and extraction settings are saved
   per voice and survive switching/restarts.
4. Enable the global hotkey. Hold **F8**, speak, then release to convert and play.
   The configurable key works outside the app subject to Windows/application hook
   restrictions. Do not run multiple copies with competing hotkeys.

Voice models are not supplied. Obtain appropriate rights/permissions for models
and recordings. Removing a library entry does not delete imported files; file
deletion is a separate confirmed action.

## Pitch extractors and devices

| Method | Installed extraction device |
| --- | --- |
| RMVPE current | GPU only |
| RMVPE New | GPU only |
| CREPE Full | GPU only |
| FCPE Experimental | GPU only |
| Mangio-style CREPE Full hop 80 Experimental | GPU only |
| RMVPE constrained range Experimental | GPU only |
| RMVPE + FCPE agreement hybrid Experimental | GPU only |
| Praat autocorrelation | CPU only |
| Harvest | CPU only |
| SwiftF0 0.3.0 | CPU only |

**CPU pitch extraction is not CPU voice conversion. Full conversion still
requires NVIDIA CUDA, including with Praat, Harvest or SwiftF0.** Labels describe
Enhanced's actual integration, not theoretical upstream library capabilities.
AMD/Intel GPU synthesis and CPU-only synthesis are not supported by this adapter.

The ZIP bundles SwiftF0 0.3.0's licensed ONNX model/private CPU runtime, the tested
RMVPE New checkpoint, and FCPE's tested checkpoint/private packages. Native
methods depend on the separately supplied supported RVC engine's assets.
Existing PyTorch/NumPy/ONNX Runtime installations are not upgraded by the app.
See PITCH-DEVICE-SUPPORT.md in the portable ZIP's app folder or source root.

## Keyboard operation and screen readers

- Tab / Shift+Tab navigate native controls; arrows operate dropdowns.
- Ctrl+Tab / Ctrl+Shift+Tab switch the library and routing pages.
- F8 holds recording and releases to convert when enabled. Toggle recording is
  optional. Escape cancels recording while Enhanced has focus. A safety timeout
  bounds capture.
- Alt+L opens **Experimental Pitch Lab**: choose a saved voice and one recording
  or audio file, select methods and run identical-setting comparisons. Results
  are saved separately and can be reopened, read and previewed. Cancel stops
  the private comparison worker. Saving a favorite is explicit; experimenting
  does not change preferences. F8 is paused while the Lab is open.
- Alt+P opens optional pitch calibration. Recommendations are experimental;
  select by listening rather than assuming a numerical ranking is best.

Controls have native accessible names. Status uses accessible_output2 for NVDA
speech/braille when available. Automated checks cover full extraction labels
through Windows MSAA, actionable control names, native navigation, focus and
cancellation. **Actual NVDA speech verification remains manual when NVDA is
unavailable; API checks alone do not prove full usability.** Check tab order,
full method/device names, Ready/error announcements, import dialogs, Lab
progress/cancel and playback with your NVDA configuration.

## Discord and game routing

VB-CABLE is optional and installed separately. Choose **CABLE Input** as Enhanced's
converted playback destination and **CABLE Output** as Discord/PlayAural/game's
microphone. Select physical headphones for independent monitoring. Windows
endpoint IDs are saved instead of temporary device numbers. Speaker playback
is not a microphone endpoint. Enhanced does not install drivers or change
Windows default devices. Local playback works without VB-CABLE; Lab previews
are restricted to physical monitoring rather than chat cable.

## Limitations and verification

- Batch push-to-talk is supported. **Continuous real-time microphone conversion
  is not a released feature.** Isolated prototypes are excluded; inference time
  is not end-to-end microphone latency.
- Actual v1/v2 conversions, index participation, pitch 0/-12/+12, persistence,
  cancellation/recovery and clean-folder portable launch/conversion were tested.
  All ten methods were exercised. Generated speech does not establish a quality
  winner; SwiftF0 is not advertised as better than RMVPE.
- A second Windows computer, every model, physical game/anti-cheat behavior and
  full NVDA auditory acceptance are not universally verified.
- Non-F0 voices cannot pitch-shift. Unsupported architectures/indexes report
  errors. Newest upstream RVC engines are not automatically compatible.
- Use a short writable path: deep nesting can exceed Windows DLL limits. App data
  is created beside the app's Python files. Keep old installations/backups
  separately; this release does not migrate or replace Classic or its fallback.

## Source, licenses and checksums

The optional **Source.zip** contains application source and reviewed pitch
bundles; Setup.cmd requires Python 3.10 and internet for desktop dependencies.
GitHub's automatic source archives contain tracked source, not prebuilt native
dependencies/model bundles. Use the Windows ZIP to run.

Original AniVoiceChanger MIT attribution is retained. Dependencies retain their
licenses/notices; see LICENSE, THIRD-PARTY-NOTICES.txt and ACKNOWLEDGMENTS.md.
The external RVC engine has its own notices. Personal voices, indexes,
recordings, settings, credentials and development snapshots are excluded.
Reviewed pitch support weights are not personal voice models.

Verify the ZIP using **SHA256SUMS.txt** or **release-checksums.json**.
Each ZIP also contains SHA256-MANIFEST.json for its contents.
