# Experimental Pitch Lab

Press Alt+L or choose Experimental Pitch Lab from Voice Library.

1. Choose a saved voice. This comparison selection does not switch the main voice.
2. Choose a WAV, FLAC or OGG sample (1–60 seconds), or explicitly record your
   microphone. Stop Sample Recording ends capture; it stops automatically after
   thirty seconds. No microphone or GPU work starts merely by opening the Lab.
3. Select installed compatible extraction methods. All controls identify CPU-only
   or GPU-only extraction. Full conversion still requires NVIDIA CUDA.
4. Set the comparison pitch/index influence. Dry comparison initially bypasses
   added effects and musical pitch correction for every method equally.
5. Run Comparisons. A private worker converts the same copied input, voice,
   index, settings and fixed seed for each method. GPU kernels are not guaranteed
   bitwise deterministic. No source file or model is overwritten.
6. Navigate the saved results using the keyboard. Read method, processing/F0
   time, peak allocated GPU memory, pitch, index and sample rate. Choose Play
   Selected Result to use physical monitoring; cable previews are rejected.
7. Optionally Save Selected Method as Favorite. This changes the method and
   necessary adapter range options only. Running comparisons alone does not
   change preferences, routing or the normal selected voice.

F8 and ordinary switching are paused while this modal workflow is open. Cancel
stops the isolated worker/capture/preview and retains completed results. Close
during work requests cancellation and waits for cleanup. Progress/status updates
do not intentionally move keyboard focus. Native Windows controls and named
status fields expose information to NVDA; auditory acceptance remains manual.

Each run gets a unique comparison folder inside PitchLab, beside VoiceLibrary.
It contains the exact input copy, numbered method WAV outputs and results.json
with settings, hashes and measurements. Open Previous Comparison restores those
results after restart without inference. Keep each complete folder when moving it.
No user comparison recordings are included in the release ZIP.

F0 statistics are diagnostics, not a naturalness score. Tracking statistics omit
reflected padding, while F0 timing includes the extractor's padded input. Worker
time includes file/IPC overhead and excludes hardware playback. GPU peaks include
cached RVC models; they are not the extractor's isolated footprint. CPU utilization
percentages and microphone-to-game latency are not measured. Independent acoustic
analysis is limited to fifteen-second outputs; longer conversions can still finish
with that analysis limitation recorded.

RMVPE Range and the agreement hybrid are optional custom adapters, not universal
implementations of every other app's RMVPE+ or hybrid. Narrow ranges can suppress
expressive pitches. Mangio-style CREPE differs in voicing behavior from native
CREPE Full. SwiftF0 is CPU extraction followed by CUDA RVC synthesis. Missing
assets/dependencies produce disabled choices or explicit errors, not silent
algorithm substitution. See PITCH-DEVICE-SUPPORT.md and SWIFTF0.md.

Listen to identical-setting outputs and the input before selecting a favorite.
Neither faster processing nor greater GPU use establishes better sound quality.
