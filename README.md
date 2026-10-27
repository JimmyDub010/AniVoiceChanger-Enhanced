# AniVoiceChanger Enhanced

An NVDA-accessible Windows application for recording your microphone, converting
speech with an RVC voice model, and playing the result through headphones or a
virtual audio cable. This is **record, then convert, then play** push-to-talk;
it is not a continuous real-time voice filter.

Based on [AniVoiceChanger by SociallyIneptWeeb](https://github.com/SociallyIneptWeeb/AniVoiceChanger).
Enhanced adds a persistent voice library, resilient global push-to-talk,
per-voice settings, cancellable playback, and optional pitch calibration.

## What you need

- Windows x64 and a physical microphone; NVDA for spoken accessibility feedback.
- **Your own compatible legacy RVC beta0717 engine installation.** The latest
  RVC engine has not been validated with this adapter. RVC v1 and v2 voice models
  are supported; model version and engine release are different things.
- **Your own RVC voice models** (`.pth`) and optional matching `.index` files.
- An NVIDIA GPU and compatible drivers for the tested CUDA inference path.
  Development testing used an RTX 4070 SUPER with CUDA PyTorch 2.0.0+cu118.
  CPU-only inference has not been validated.
- Optional VB-CABLE, installed separately, for Discord or game voice chat.

**No RVC engine, voice models, retrieval indexes, HuBERT/RMVPE checkpoints, or
experimental pretrained pitch models are included in this repository or release.**
The portable ZIP bundles only the desktop Python runtime and its dependencies.

### Download the RVC engine

Get the separate engine archive here:
**[Download RVC-beta.7z from Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/RVC-beta.7z)**
(approximately 5.2 GB).

Extract the `.7z` archive with 7-Zip to its own folder. In Enhanced's
**Conversion and Audio Routing** page, set **RVC installation folder** to the
extracted engine folder containing `runtime/python.exe`. Voice models are still
supplied separately; import your own `.pth` and optional `.index` files into
Enhanced's Voice Library.

The external RVC folder must contain `runtime/python.exe`, its inference
dependencies (including PyTorch/CUDA, FAISS, fairseq, librosa and parselmouth),
`vc_infer_pipeline.py`, `my_utils.py`, `infer_pack/models.py` or
`lib/infer_pack/models.py`, and the engine's required `hubert_base.pt` and
`rmvpe.pt` assets. Obtain these separately under their respective licenses.
Downloaded experimental RMVPE New and FCPE assets are excluded. Use standard
RMVPE unless you separately supply the exact supported experimental assets.

## Install the portable app

1. Download the Windows x64 portable ZIP from [Releases](https://github.com/JimmyDub010/AniVoiceChanger-Enhanced/releases/latest).
2. Extract the **entire ZIP** to a writable folder.
3. Run `Launch-Enhanced.cmd`. Keep the `runtime` and `app` folders beside it.
   You do not need a separate desktop Python installation.
4. In **Conversion and Audio Routing**, browse to your compatible RVC engine.
5. Choose your physical microphone, playback destination, and optional headphone
   monitoring destination. Device selections persist by Windows endpoint ID.
6. Import and select a voice as described below, then enable global F8.

This is a portable Python application, not an installer or a standalone EXE.
Do not install it inside a protected read-only directory. Its library, settings,
cache and converted outputs are created inside the `app` folder. Keep backups
of your own voice library before replacing an installation.

## Run from source

Install Python **3.10 x64**, clone or download this repository, and run
`Setup.cmd`. Setup creates `.venv` and installs `requirements.txt`; it needs
internet access. Then run `Launch-Enhanced.cmd`.

Alternatively, from PowerShell in the repository:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B enhanced.py
```

The external RVC engine remains required for conversion. Source installation
does not install or download an engine or any voice models.

## NVDA and keyboard controls

Native Windows controls expose accessible names. Status, recording, processing,
success and error announcements use `accessible_output2` with NVDA.

| Key or action | Behavior |
| --- | --- |
| Tab / Shift+Tab | Navigate controls in the current page or dialog |
| Ctrl+Tab / Ctrl+Shift+Tab | Switch Voice Library and Conversion and Audio Routing |
| Up / Down in Saved Voices | Select a voice; the latest request wins during rapid navigation |
| Hold F8, then release | Record the real microphone, then convert and play, after enabling the global hotkey |
| Escape in Enhanced | Cancel an active recording without conversion or playback |
| Alt+P in the main window | Open Auto Pitch Calibration for the loaded saved voice |

The **Active voice** label confirms the successfully loaded model, separately
from the highlighted Saved Voices choice. Loading failures leave no active voice.
Pitch and index controls remain available during capture and processing; changes
apply to the next utterance. Completion restores the original control rather than
moving focus to status. Background completion does not take focus from another app.

## F8 recording and audio routing

After the model loads, select **Enable push-to-talk hotkey**. Global F8 works
outside Enhanced, including while minimized. Hold to record your microphone;
release to stop, convert and play to your selected output. Key-repeat suppression,
missed-release recovery and recording limits protect against stuck recording.
The hotkey also reaches the foreground application. Close or disable hotkeys in
other AniVoiceChanger instances to avoid duplicate capture and playback.

For local listening, choose physical headphones as playback. For voice chat,
choose **CABLE Input** as Enhanced's playback destination, optionally choose
physical headphones for monitoring, and select **CABLE Output** as the microphone
inside Discord or your game. Monitoring plays the converted result.

Unavailable devices are reported explicitly. Microphone failure or silence is
reported as an error; production never substitutes synthetic speech. Output files
are saved in `outputs`. Engine diagnostics are written locally to `.cache`.

## Import and manage voices

On **Voice Library**, use **Import Voice from ZIP** or **Import Extracted Voice**.
Imports are copied into the selected library; original downloads are retained.
ZIP importing pairs compatible `.pth` and `.index` files automatically where the
choice is unambiguous and asks you when multiple candidates need selection.
Extracted imports accept a model and optional index through file dialogs.

Saved Voices supports rename, Edit Voice Settings, relinking missing files,
removing entries while keeping imported files, and explicit confirmed deletion
of imported library copies. Pitch, extraction method, index influence and effects
are stored per voice. Selecting FCPE for one voice does not replace another
voice's saved RMVPE choice. Switching back or restarting restores the saved method;
older entries without a method inherit the current selection on first load.

Voice changes invalidate stale conversion/playback generations and wait for
in-use workers before replacement. Saved models are not mixed together.

## Optional Auto Pitch Calibration

Load a saved voice and select a physical headphone monitoring destination first.
Open **Auto Pitch Calibration** on Voice Library or press Alt+P.

1. **Record Calibration Sample** (Alt+R): speak naturally for six seconds into
   your actual microphone.
2. **Analyze Voice** (Alt+A): inspect median fundamental frequency, central pitch
   range, voiced-frame fraction and clipping. Insufficient voiced speech produces
   an error so you can record a clearer sample.
3. Optionally choose a known target-speaker reference recording, 1–15 seconds,
   with Alt+F; clear it with Alt+L. No target pitch is inferred from gender, model
   filenames or checkpoint tensors.
4. **Find Recommended Pitches** (Alt+P in this dialog): convert the same sample
   at several shifts, including zero. Comparisons use dry conversion with musical
   pitch correction off. Repeated comparisons reuse the session cache.
5. Select a candidate with arrows and **Preview Selected Candidate** (Alt+V).
   Previews use only the chosen physical monitoring device, never a virtual cable
   or the chat output. Playback cannot overlap.
6. **Save Pitch for This Voice** (Alt+S) to persist your listening choice without
   replacing the voice's other preferences.
7. **Cancel Calibration** or Escape closes the workflow. In-flight work finishes
   safely before resources and temporary audio are released. Global F8 is paused
   during calibration and available afterward if previously enabled.

Pitch ranking is **experimental**. Praat/parselmouth pitch estimates can have
octave errors. Clipping, voiced-speech retention, pitch tracking and optional
reference matching are technical measurements, not a guarantee of naturalness or
an AI confidence score. Your listening assessment determines the best shift.
There is no voice-model retraining. Saved calibration stores numerical analysis;
temporary raw recordings and candidate audio are removed when the dialog closes.

## Tests and building releases

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
.\.venv\Scripts\python.exe -B release_tools\build_release.py --output dist --version YOUR_VERSION
```

The current release passed 52 automated regression tests, real FCPE/RMVPE
conversion and method-switching tests before and after restart, rapid-switch
protection, portable startup, focus checks and archive integrity checks on the
development Windows system. Unit tests create small synthetic validation fixtures
locally; no personal or trained models are committed. Clean-machine compatibility,
recipient-supplied engines, hardware and subjective listening still need testing.

The build helper produces a model-free Windows portable ZIP and a separate source
ZIP, each with a SHA-256 file manifest. Release assets include checksums. Generated
files, local settings, models, recordings and test output are ignored by Git.

## License and attribution

Licensed under the **MIT License**, matching the original AniVoiceChanger project.
The original `Copyright (c) 2023 SociallyIneptWeeb` notice is retained in
[LICENSE](LICENSE). Bundled desktop dependencies retain their own licenses and
notices in the portable release. External RVC engines and voice models have their
own licenses and are not redistributed here.
