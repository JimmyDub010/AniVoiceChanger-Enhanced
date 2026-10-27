# AniVoiceChanger Enhanced — Windows x64

Accessible push-to-talk RVC voice conversion with a saved voice library, global
F8, audio routing, and optional experimental pitch calibration.

## Portable app ZIP

Extract the entire ZIP to a writable folder and run Launch-Enhanced.cmd.
Desktop Python 3.10 and desktop dependencies are bundled. No system Python
installation is required. Keep the runtime and app folders beside the launcher.
This is a portable Python application, not an installer or a standalone EXE.

## Source ZIP

Install Python 3.10 x64, then run Setup.cmd to create a local environment and
install requirements. Setup needs internet access. Run Launch-Enhanced.cmd.
Run tests with `.venv\Scripts\python.exe -m unittest discover -s tests`.
To rebuild the portable and source ZIPs, run the release_tools/build_release.py
script from that environment with `--output PATH --version YOUR_VERSION`.

## Required conversion engine and models — supplied separately

Download the separate RVC engine archive:
[RVC-beta.7z on Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/RVC-beta.7z)
(approximately 5.2 GB). Extract it with 7-Zip and select the extracted folder
containing runtime/python.exe as Enhanced's RVC installation folder. Voice models
must still be supplied separately.

Neither ZIP includes voice models, retrieval indexes, neural network checkpoints,
RVC engine/runtime, recordings, personal settings, experimental downloaded pitch
models, or VB-CABLE. The bundled desktop runtime does not perform RVC inference.

Provide a compatible legacy RVC beta0717 installation with its own
runtime/python.exe and dependencies (including CUDA PyTorch, FAISS,
parselmouth, librosa and fairseq), vc_infer_pipeline.py, my_utils.py,
infer_pack/models.py or lib/infer_pack/models.py, hubert_base.pt, and rmvpe.pt
for standard RMVPE pitch extraction. This adapter has been tested with RVC v1/v2
and CUDA PyTorch 2.0.0+cu118, using an NVIDIA RTX 4070 SUPER. CPU-only conversion
has not been validated. Newer RVC installations are not automatically compatible.

In Conversion and Audio Routing, browse to that RVC installation. In Voice
Library, import your own trusted model ZIP or extracted .pth and optional .index.
Downloaded experimental RMVPE New and FCPE assets are excluded; use standard
RMVPE unless you separately provide those exact supported assets.

## Controls and accessibility

NVDA announcements use accessible_output2. Ctrl+Tab and Ctrl+Shift+Tab switch
Voice Library and Conversion and Audio Routing. Saved Voices loads the selected
voice; the Active voice label confirms which model has successfully loaded.

Pitch extraction is saved separately for each voice. Selecting FCPE Experimental
for one voice does not replace another voice's saved RMVPE choice. Switching back
restores the first voice's FCPE choice, including after restarting the app. Older
voices with no saved method inherit the currently selected method on first load.
The display labels map explicitly to the engine's method identifiers; unsupported
values raise an error instead of silently choosing a fallback.

Choose your physical microphone and playback destination, load a voice, and
enable global F8. Hold F8 to record; release to convert and play. A safety timeout
bounds capture. Escape cancels recording while Enhanced has focus.
Pitch and index influence remain available during recording and processing;
changes apply to the next utterance. Recording and completion do not move focus
to status. NVDA still announces status changes.

Optional monitoring sends converted audio to physical headphones. For Discord
or game chat, install VB-CABLE separately, choose CABLE Input as playback in
Enhanced, and CABLE Output as the microphone in the other application.
Keep other AniVoiceChanger instances' global hotkeys disabled.

Alt+P opens optional Auto Pitch Calibration for the loaded saved voice. Record
six seconds of natural speech, Analyze, Find Recommended Pitches, preview through
the selected physical monitoring endpoint, and save your listening choice.
Optional target reference audio informs the pitch search. Technical measurements
are experimental recommendations, not proof of naturalness. Pitch correction
and effects are disabled for comparisons. Cancel safely closes calibration and
cleans temporary audio. F8 is paused while calibration is open.

Settings, VoiceLibrary, cache, and converted outputs are created inside the app
folder for the portable edition, or beside enhanced.py for the source edition.
Never place the application in a protected read-only directory. User models and
settings are intentionally absent from these archives.

## License and validation

Original AniVoiceChanger MIT license is retained. Bundled dependencies retain
their individual licenses and notices. Obtain and use external models and RVC
engine components under their respective licenses.

Archives contain a SHA256-MANIFEST.json for their packaged files. Desktop
startup, NVDA availability, regression tests and isolated relocation checks were
tested on the development Windows system. A clean second Windows computer,
recipient-supplied RVC environment, physical hardware, and subjective listening
still require release testing.
