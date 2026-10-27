"""Optional, modal listening workflow. All inference uses the active real worker."""
import shutil
import tempfile
import time
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4

import comtypes
import soundfile as sf
import wx
from desktop import NamedAccessible
from pitch_calibration import candidate_pitches, cache_key, convert_candidates, require_speech
from routing import Recorder, endpoints, playback_many, resolve


class CalibrationDialog(wx.Dialog):
    def __init__(self, parent):
        self.owner = parent
        self.voice_id = parent.active_voice
        self.voice = parent.library.get(self.voice_id)
        super().__init__(parent, title='Auto Pitch Calibration — ' + self.voice['name'], size=(740, 670),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.busy = False
        self.closing = False
        self.cancel = Event()
        self.source = None
        self.analysis = None
        self.reference = None
        self.rows = []
        self.cache_root = Path(__file__).resolve().parent / '.cache' / 'calibration'
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='session-', dir=self.cache_root))
        self.client = parent.client
        self.parameters = parent.conversion_settings()
        self.parameters.pop('_generation')
        # These comparisons isolate semitone shifting from effects/pitch correction.
        self.parameters['effects'] = {}
        layout = wx.BoxSizer(wx.VERTICAL)
        intro = ('Record six seconds of natural speech with the selected physical microphone.\n'
                 'Comparisons use dry conversion with musical pitch correction off. F8 is paused while this dialog is open.\n'
                 'No reference pitch is inferred from a model name, gender, or checkpoint.\n'
                 'Measurements rank technical defects and pitch tracking, not naturalness. Listen before saving.\n'
                 'Previews use only your selected physical headphone monitoring device, never the chat output.')
        layout.Add(wx.StaticText(self, label=intro), 0, wx.ALL, 10)
        self.buttons = []
        for label, handler in [('&Record Calibration Sample', self.record), ('&Analyze Voice', self.analyze),
                               ('Choose Optional Target Re&ference Audio', self.choose_reference),
                               ('C&lear Target Reference', self.clear_reference),
                               ('Find Recommended &Pitches', self.find), ('Pre&view Selected Candidate', self.preview),
                               ('&Save Pitch for This Voice', self.save_pitch)]:
            button = wx.Button(self, label=label)
            button.Bind(wx.EVT_BUTTON, handler)
            layout.Add(button, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
            self.buttons.append(button)
        layout.Add(wx.StaticText(self, label='Pitch candidates — ranked by measured defects, not naturalness'), 0, wx.LEFT, 8)
        self.candidates = wx.Choice(self)
        self.candidates.SetName('Pitch calibration candidates')
        self.candidates.SetAccessible(NamedAccessible(self.candidates, 'Pitch calibration candidates'))
        self.candidates.Bind(wx.EVT_CHOICE, self.selected)
        layout.Add(self.candidates, 0, wx.EXPAND | wx.ALL, 8)
        self.status = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.status.SetAccessible(NamedAccessible(self.status, 'Calibration analysis and status'))
        layout.Add(self.status, 1, wx.EXPAND | wx.ALL, 8)
        cancel = wx.Button(self, wx.ID_CANCEL, label='Cancel Calibration')
        cancel.Bind(wx.EVT_BUTTON, self.cancel_calibration)
        layout.Add(cancel, 0, wx.ALL, 8)
        self.SetSizer(layout)
        self.Bind(wx.EVT_CLOSE, self.cancel_calibration)
        self.say('Calibration is optional. Record a natural speaking sample to begin. Existing pitch is unchanged until Save.')
        wx.CallAfter(self.buttons[0].SetFocus)

    def say(self, message):
        if self.closing: return
        self.status.ChangeValue(message)
        self.owner.announce(message)

    def work(self, message, function, completed):
        if self.busy: return
        self.busy = True
        for button in self.buttons: button.Disable()
        self.candidates.Disable()
        self.say(message)
        def worker():
            comtypes.CoInitialize()
            try: value = function()
            except Exception as error: value = error
            finally: comtypes.CoUninitialize()
            wx.CallAfter(finish, value)
        def finish(value):
            self.busy = False
            if self.closing:
                self.finish_close()
                return
            for button in self.buttons: button.Enable()
            self.candidates.Enable()
            if isinstance(value, Exception): self.say('Calibration error: ' + str(value))
            else: completed(value)
        Thread(target=worker, daemon=True).start()

    def record(self, event=None):
        try: endpoint = self.owner.selected('microphone')
        except Exception as error: self.say(str(error)); return
        self.source = self.analysis = None
        self.rows = []
        self.candidates.Clear()
        source = self.directory / (uuid4().hex + '.wav')
        def capture():
            recorder = Recorder(endpoint, maximum_seconds=8)
            try:
                recorder.start()
                deadline = time.monotonic() + 6
                while time.monotonic() < deadline:
                    if self.cancel.wait(.05): return None
                    if recorder.errors or not recorder.stream.active: break
                audio, rate = recorder.stop()
                sf.write(source, audio, rate, subtype='PCM_16')
                return source
            finally:
                if not recorder.stream.closed:
                    recorder.stream.abort()
                    recorder.stream.close()
        def completed(value):
            self.source = value
            self.say('Real microphone sample recorded. Choose Analyze Voice.')
        self.work('Recording six seconds. Speak in your natural voice now.', capture, completed)

    def analyze(self, event=None):
        if not self.source:
            self.say('Record a calibration sample first.'); return
        def measure():
            value = self.client.request('analyze', input=str(self.source))
            require_speech(value)
            return value
        def completed(value):
            self.analysis = value
            self.say(self.describe(value) + ' Choose Find Recommended Pitches. No target reference is assumed.'
                     if not self.reference else self.describe(value) + ' Target reference is available.')
        self.work('Analyzing fundamental frequency and voiced speech.', measure, completed)

    @staticmethod
    def describe(value):
        return (f"Median F0 {value['median_hz']:.0f} Hz. Usable central pitch range "
                f"{value['p10_hz']:.0f} to {value['p90_hz']:.0f} Hz. "
                f"{value['voiced_frames']} voiced frames, {value['voiced_fraction']:.0%} voiced. "
                f"Clipping {value['clipped_fraction']:.1%}. Pitch estimates may contain octave errors.")

    def choose_reference(self, event):
        with wx.FileDialog(self, 'Choose known target speaker reference, 1 to 15 seconds',
                           wildcard='Audio (*.wav;*.flac)|*.wav;*.flac', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() != wx.ID_OK: return
            path = dialog.GetPath()
        def measure():
            value = self.client.request('analyze', input=path)
            require_speech(value)
            return value
        def completed(value):
            self.reference = value
            self.invalidate_rows()
            self.say('Target reference supplied by you. ' + self.describe(value))
        self.work('Measuring optional target reference audio.', measure, completed)

    def clear_reference(self, event):
        self.reference = None
        self.invalidate_rows()
        self.say('Target reference cleared. Candidates will be listening comparisons without a known target range.')

    def invalidate_rows(self):
        self.rows = []
        self.candidates.Clear()

    def find(self, event=None):
        try:
            if not self.analysis: raise ValueError('Record and analyze a sample first.')
            pitches = candidate_pitches(self.analysis, self.parameters['pitch'], self.reference, bool(self.voice['rvc']['f0']))
            key = cache_key(self.source, self.voice['model_hash'], self.voice.get('index_hash'), self.parameters, pitches, self.reference)
        except Exception as error: self.say(str(error)); return
        self.invalidate_rows()
        def progress(message): wx.CallAfter(self.say, message)
        def compare():
            return convert_candidates(self.client, self.source, self.directory / key, self.parameters,
                                      pitches, self.analysis, self.cancel, progress, self.reference)
        def completed(rows):
            self.rows = rows
            labels = []
            for rank, row in enumerate(rows, 1):
                measured = row['analysis']
                labels.append(f"{rank}. {row['pitch']:+d} semitones; F0 {measured.get('median_hz', 0):.0f} Hz; "
                              f"voiced {measured['voiced_fraction']:.0%}; clipping {measured['clipped_fraction']:.1%}; "
                              f"tracking difference {row['tracking_error_semitones']:.1f} semitones")
            self.candidates.SetItems(labels)
            if rows:
                self.candidates.SetSelection(0)
                self.candidates.SetFocus()
                self.say(f"{len(rows)} candidates compared. First to audition: {rows[0]['pitch']:+d} semitones. "
                         'This ordering does not establish naturalness. Preview candidates and choose by listening.')
            else: self.say('No candidates were produced.')
        self.work('Comparing dry RVC pitch shifts. No audio will play automatically.', compare, completed)

    def selected(self, event):
        row = self.selected_row()
        self.say(f"Selected {row['pitch']:+d} semitones. Preview to assess the sound.")

    def selected_row(self):
        index = self.candidates.GetSelection()
        if index < 0 or index >= len(self.rows): raise ValueError('Find recommended pitches and select a candidate first.')
        return self.rows[index]

    def preview(self, event=None):
        try:
            row = self.selected_row()
            selection = self.owner.choices['monitor'].GetSelection()
            if selection < 1: raise ValueError('Select a physical headphone monitoring device on Audio Routing before opening calibration.')
            endpoint = self.owner.device_lists['monitor'][selection]
            device, _ = resolve(endpoint, 'output')
            if any(word in device.name.casefold() for word in ('cable', 'line 1')):
                raise ValueError('Calibration previews require physical headphones, not a virtual cable.')
        except Exception as error: self.say(str(error)); return
        def play():
            audio, rate = sf.read(row['path'], dtype='float32')
            playback_many([endpoint], audio, rate, self.cancel)
        self.work(f"Previewing {row['pitch']:+d} semitones through {device.name}.", play,
                  lambda value: self.say('Preview finished. Select another candidate or save your preferred pitch.'))

    def save_pitch(self, event=None):
        try:
            row = self.selected_row()
            if self.owner.active_voice != self.voice_id: raise ValueError('The active voice changed; calibration was not saved.')
            self.owner.library.save_calibration(self.voice_id, row['pitch'], self.analysis, self.reference)
            self.owner.pitch.SetValue(row['pitch'])
            self.owner.save()
            self.say(f"Saved {row['pitch']:+d} semitones for {self.voice['name']}. Your other voice settings are preserved.")
        except Exception as error: self.say('Calibration could not be saved: ' + str(error))

    def cancel_calibration(self, event=None):
        self.cancel.set()
        self.closing = True
        if self.busy:
            self.status.ChangeValue('Cancel requested. Waiting for the current worker operation to finish safely.')
            self.owner.announce('Cancel requested. Waiting for the current operation to finish safely.')
            if isinstance(event, wx.CloseEvent): event.Veto()
        else: self.finish_close()

    def finish_close(self):
        try:
            target = self.directory.resolve()
            if not target.is_relative_to(self.cache_root.resolve()) or not target.name.startswith('session-'):
                raise OSError('Refusing cleanup outside this calibration session.')
            shutil.rmtree(target)
        except OSError:
            self.owner.announce('Calibration closed. Some temporary audio could not be removed from ' + str(self.directory))
        else: self.owner.announce('Calibration closed. Temporary comparison audio removed. Global F8 is available again if enabled.')
        self.EndModal(wx.ID_CANCEL)
