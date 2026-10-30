"""Native, modal same-recording comparisons; isolated worker and durable results."""
import copy
import json
from pathlib import Path
from threading import Event, Thread, Lock
import time
from uuid import uuid4

import comtypes
import soundfile as sf
import wx
from wx.lib.scrolledpanel import ScrolledPanel

from desktop import NamedAccessible
from engine_client import EngineClient
from pitch_lab import make_run, snapshot_audio, compare, describe_result, save_favorite
from pitch_methods import capabilities, LABELS, DEVICE_NOTE
from routing import Recorder, playback_many, resolve
from voice_dialog import method_label


class PitchLabDialog(wx.Dialog):
    def __init__(self, owner):
        super().__init__(owner, title='Experimental Pitch Lab', size=(800, 790), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.owner = owner
        self.busy = False
        self.closing = False
        self.cancel = Event()
        self.stop_capture = Event()
        self.client = None
        self.client_lock = Lock()
        self.source = None
        self.report = None
        self.directory = None
        self.root = owner.library.root.parent / 'PitchLab'
        self.root.mkdir(exist_ok=True)
        self.voices = owner.library.voices()
        self.panel = ScrolledPanel(self)
        self.fields = wx.BoxSizer(wx.VERTICAL)
        self.panel.SetSizer(self.fields)
        note = wx.StaticText(self.panel, label=DEVICE_NOTE)
        note.Wrap(710)
        self.fields.Add(note, 0, wx.EXPAND | wx.ALL, 8)
        self.fields.Add(wx.StaticText(self.panel, label='Compare the exact same sample. F8 and ordinary voice switching are paused while this dialog is open.\n'
            'Files are retained in the PitchLab folder, including cancelled runs. No audio plays automatically.\n'
            'Measurements describe processing and pitch tracking, not naturalness. Only Save Favorite changes a voice preference.'), 0, wx.ALL, 8)
        self.voices_box = self.field('Saved voice for comparison', lambda: wx.Choice(self.panel, choices=[v['name'] + ' — ' + v['id'][:6] for v in self.voices]))
        selected = next((i for i, v in enumerate(self.voices) if v['id'] == owner.active_voice), 0)
        if self.voices: self.voices_box.SetSelection(selected)
        self.voices_box.Bind(wx.EVT_CHOICE, self.voice_changed)
        self.input_label = self.field('Comparison input file', lambda: wx.TextCtrl(self.panel, style=wx.TE_READONLY))
        self.pitch = self.field('Fixed pitch shift in semitones', lambda: wx.SpinCtrl(self.panel, min=-24, max=24, initial=0))
        self.index = self.field('Fixed index influence percent', lambda: wx.SpinCtrl(self.panel, min=0, max=100, initial=0))
        self.dry = wx.CheckBox(self.panel, label='Dry comparison — bypass added effects and musical pitch correction')
        self.dry.SetValue(True)
        self.fields.Add(self.dry, 0, wx.ALL, 8)
        self.minimum = self.field('Experimental range minimum Hz (RMVPE Range and Mangio-style only)', lambda: wx.SpinCtrl(self.panel, min=50, max=1099, initial=50))
        self.maximum = self.field('Experimental range maximum Hz', lambda: wx.SpinCtrl(self.panel, min=51, max=1100, initial=1100))
        self.method_boxes = {}
        self.fields.Add(wx.StaticText(self.panel, label='Installed compatible extraction methods'), 0, wx.ALL, 8)
        for key, label in LABELS.items():
            box = wx.CheckBox(self.panel, label=label)
            self.fields.Add(box, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
            self.method_boxes[key] = box
        self.fields.Add(wx.StaticText(self.panel, label='Unavailable methods and compatibility notes'), 0, wx.LEFT, 8)
        self.unsupported = wx.TextCtrl(self.panel, value='Method availability depends on the selected voice and bundled dependencies.',
                                      style=wx.TE_READONLY | wx.TE_MULTILINE, size=(-1, 65))
        self.unsupported.SetAccessible(NamedAccessible(self.unsupported, 'Unavailable methods and compatibility notes'))
        self.fields.Add(self.unsupported, 0, wx.ALL, 8)
        self.actions = []
        for label, handler in [('&Choose Audio File', self.choose_file), ('&Record Microphone Sample', self.record),
                               ('Stop Sample &Recording', self.stop_recording), ('&Run Comparisons', self.run_comparisons),
                               ('&Open Previous Comparison', self.open_previous), ('&Play Selected Result', self.preview),
                               ('Save Selected Method as &Favorite', self.favorite), ('Open Results &Folder', self.open_folder)]:
            button = wx.Button(self.panel, label=label)
            button.Bind(wx.EVT_BUTTON, handler)
            self.fields.Add(button, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
            self.actions.append(button)
        self.rows_box = self.field('Comparison results', lambda: wx.Choice(self.panel))
        self.rows_box.Bind(wx.EVT_CHOICE, self.result_selected)
        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(self.panel, 1, wx.EXPAND)
        layout.Add(wx.StaticText(self, label='Pitch Lab progress and result details'), 0, wx.LEFT, 8)
        self.status = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 100))
        self.status.SetAccessible(NamedAccessible(self.status, 'Pitch Lab progress and result details'))
        layout.Add(self.status, 0, wx.EXPAND | wx.ALL, 8)
        buttons = wx.BoxSizer(wx.HORIZONTAL)
        cancel = wx.Button(self, label='&Cancel Current Operation')
        cancel.Bind(wx.EVT_BUTTON, self.cancel_operation)
        close = wx.Button(self, wx.ID_CLOSE, label='&Close Pitch Lab')
        close.Bind(wx.EVT_BUTTON, self.close_lab)
        buttons.Add(cancel, 1, wx.ALL, 8); buttons.Add(close, 1, wx.ALL, 8)
        layout.Add(buttons, 0, wx.EXPAND)
        self.SetSizer(layout)
        self.panel.SetupScrolling(scroll_x=False)
        self.Bind(wx.EVT_CLOSE, self.close_lab)
        self.voice_changed()
        self.say('Select methods and choose an audio file or record a sample. Existing voice settings are unchanged.')
        wx.CallAfter(self.voices_box.SetFocus)

    def field(self, label, factory):
        self.fields.Add(wx.StaticText(self.panel, label=label), 0, wx.LEFT | wx.RIGHT, 8)
        control = factory()
        control.SetName(label); control.SetAccessible(NamedAccessible(control, label))
        self.fields.Add(control, 0, wx.EXPAND | wx.ALL, 8)
        return control

    def say(self, message):
        if self.closing: return
        self.status.ChangeValue(message)
        self.owner.announce(message)

    def voice(self):
        i = self.voices_box.GetSelection()
        if i < 0: raise ValueError('Import or choose a saved voice first.')
        return self.voices[i]

    def voice_changed(self, event=None):
        if not self.voices: return
        voice = self.voice()
        self.pitch.SetValue(voice['preferences'].get('pitch', 0))
        self.index.SetValue(round(voice['preferences'].get('index_rate', 0) * 100))
        self.available = capabilities(self.owner.engine.GetValue(), f0=bool(voice['rvc']['f0']))
        self.available_keys = {v['key'] for v in self.available if v['available']}
        for item in self.available:
            box = self.method_boxes[item['key']]
            box.Enable(item['available'])
            box.SetValue(item['available'] and item['key'] in ('rmvpe', 'crepe', 'fcpe-experimental'))
            box.SetToolTip(item['reason'] or ('Native engine method' if item['native'] else 'Optional adapter, not a quality guarantee'))
        missing = [item['label'] + ': ' + item['reason'] for item in self.available if not item['available']]
        self.unsupported.ChangeValue('\n'.join(missing + ['SwiftF0 uses its bundled CPU extractor; RVC synthesis still uses CUDA.',
            'Experimental adapters are not interchangeable with a full alternate backend.']))
        if event: self.say('Comparison voice: ' + voice['name'] + '. Saved preferences restored for this comparison only.')

    def work(self, message, function, completed):
        if self.busy: return
        self.busy = True
        self.cancel = Event(); self.stop_capture = Event()
        # Keep action buttons focusable. Their handlers ignore repeated activation.
        for control in [self.voices_box, self.pitch, self.index, self.minimum, self.maximum, self.dry] + list(self.method_boxes.values()):
            control.Disable()
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
                self.EndModal(wx.ID_CLOSE)
                return
            for control in [self.voices_box, self.pitch, self.index, self.minimum, self.maximum, self.dry]: control.Enable()
            for key, box in self.method_boxes.items(): box.Enable(key in self.available_keys)
            if isinstance(value, Exception): self.say(('Cancelled. ' if self.cancel.is_set() else 'Pitch Lab error: ') + str(value))
            else: completed(value)
        Thread(target=worker, daemon=True).start()

    def choose_file(self, event=None):
        if self.busy: return
        with wx.FileDialog(self, 'Choose one audio sample, 1 to 60 seconds', wildcard='Audio (*.wav;*.flac;*.ogg)|*.wav;*.flac;*.ogg', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() != wx.ID_OK: return
            self.source = Path(dialog.GetPath())
        self.input_label.ChangeValue(str(self.source))
        self.say('Input selected. It will be copied without modifying the original when comparisons start.')

    def record(self, event=None):
        if self.busy: return
        try: endpoint = self.owner.selected('microphone')
        except Exception as error: self.say(str(error)); return
        folder = make_run(self.root)
        target = folder / 'recorded-sample.wav'
        def capture():
            recorder = Recorder(endpoint, maximum_seconds=31)
            try:
                recorder.start()
                deadline = time.monotonic() + 30
                while not self.stop_capture.wait(.05) and time.monotonic() < deadline:
                    if self.cancel.is_set(): return None
                    if recorder.errors or not recorder.stream.active: break
                if self.cancel.is_set(): return None
                audio, rate = recorder.stop()
                if len(audio) < rate: raise ValueError('Record at least one second of audio.')
                with target.open('xb') as file: sf.write(file, audio, rate, format='WAV', subtype='PCM_16')
                return target
            finally:
                if not recorder.stream.closed: recorder.stream.abort(); recorder.stream.close()
        def finished(value):
            if value:
                self.source = value
                self.input_label.ChangeValue(str(value))
                self.say('Microphone sample saved. Run comparisons to use this identical recording for each method.')
            else: self.say('Microphone recording cancelled. No input was replaced.')
        self.work('Recording microphone sample. Speak now; choose Stop Sample Recording when finished. Maximum thirty seconds.', capture, finished)

    def stop_recording(self, event=None):
        self.stop_capture.set()

    def run_comparisons(self, event=None):
        if self.busy: return
        try:
            if not self.source: raise ValueError('Choose an audio file or record a sample first.')
            voice = copy.deepcopy(self.voice())
            methods = [key for key, box in self.method_boxes.items() if box.GetValue() and key in self.available_keys]
            if not methods: raise ValueError('Select at least one installed compatible method.')
            options = {'f0_min': self.minimum.GetValue(), 'f0_max': self.maximum.GetValue()}
            if options['f0_min'] >= options['f0_max']: raise ValueError('Minimum pitch must be below maximum pitch.')
            parameters = {'pitch': self.pitch.GetValue(), 'index_rate': self.index.GetValue() / 100 if voice.get('index') else 0,
                          'index': str(self.owner.library.path_for(voice, 'index') or ''),
                          'effects': {} if self.dry.GetValue() else copy.deepcopy(voice['preferences'].get('effects', {}))}
            model = self.owner.library.path_for(voice, 'model')
            if not model.is_file(): raise ValueError('The saved model is missing; relink it in Voice Library first.')
            source = self.source
            engine = self.owner.engine.GetValue()
            directory = make_run(self.root)
        except Exception as error: self.say(str(error)); return
        self.directory = directory
        def progress(message): wx.CallAfter(self.say, message)
        def execute():
            sample = snapshot_audio(source, directory)
            if self.cancel.is_set(): return None
            client = EngineClient(engine)
            with self.client_lock: self.client = client
            try:
                if self.cancel.is_set(): return None
                client.request('load', model=str(model))
                return compare(client, sample, directory, voice, parameters, methods, self.cancel, progress, options)
            finally:
                client.close()
                with self.client_lock: self.client = None
        def finished(report):
            if report is None: self.say('Comparison cancelled. Existing audio and preferences were not changed.'); return
            self.show_report(report, directory)
            self.say('Comparison ' + report['status'] + '. Results are saved. Select a result, then Play or read its details.')
        self.work('Loading an isolated comparison worker. F8 is paused; no automatic playback.', execute, finished)

    def show_report(self, report, directory):
        if report.get('schema') != 1 or not isinstance(report.get('results'), list): raise ValueError('Unsupported comparison report.')
        self.report, self.directory = report, Path(directory).resolve()
        self.rows_box.SetItems([LABELS.get(r['method'], r['method']) + ' — ' + r['status'] for r in report['results']])
        if report['results']: self.rows_box.SetSelection(0)

    def row(self):
        i = self.rows_box.GetSelection()
        if not self.report or i < 0: raise ValueError('Run or open a comparison and select a result first.')
        return self.report['results'][i]

    def result_selected(self, event=None):
        try: self.say(describe_result(self.row(), self.report['parameters']))
        except Exception as error: self.say(str(error))

    def preview(self, event=None):
        if self.busy: return
        try:
            row = self.row()
            if row['status'] != 'complete': raise ValueError('This method failed; there is no completed audio to play.')
            name = row['path']
            path = (self.directory / name).resolve()
            if Path(name).name != name or not path.is_relative_to(self.directory): raise ValueError('Unsafe result audio path.')
            # Explicit physical playback choice; never infer a chat endpoint as headphones.
            selection = self.owner.choices['monitor'].GetSelection()
            endpoint = self.owner.device_lists['monitor'][selection] if selection > 0 else self.owner.selected('output')
            device, _ = resolve(endpoint, 'output')
            if any(word in device.name.lower() for word in ('cable', 'line 1')):
                raise ValueError('Select physical headphone monitoring for Lab previews. Audio is never previewed into the chat cable.')
        except Exception as error: self.say(str(error)); return
        def play():
            audio, rate = sf.read(path, dtype='float32')
            playback_many([endpoint], audio, rate, self.cancel)
        self.work('Playing selected comparison through ' + device.name + '.', play,
                  lambda value: self.say('Preview finished. ' + describe_result(row, self.report['parameters'])))

    def favorite(self, event=None):
        if self.busy: return
        try:
            row = self.row()
            if row['status'] != 'complete': raise ValueError('A failed method cannot be saved as a favorite.')
            voice = self.owner.library.get(self.report['voice_id'])
            if voice.get('model_hash') != self.report.get('model_hash'): raise ValueError('This result uses a different model; no preference was changed.')
            available = {c['key'] for c in capabilities(self.owner.engine.GetValue()) if c['available']}
            save_favorite(self.owner.library, voice['id'], row['method'], available, self.report.get('options', {}))
            if self.owner.active_voice == voice['id']:
                self.owner.applying_voice = True
                self.owner.method.SetStringSelection(method_label(row['method']))
                self.owner.applying_voice = False
                self.owner.save()
            self.say('Saved ' + row['method'] + ' for ' + voice['name'] + '. Pitch, index, effects and audio routing were preserved.')
        except Exception as error: self.say('Favorite was not saved: ' + str(error))

    def open_previous(self, event=None):
        if self.busy: return
        with wx.FileDialog(self, 'Open a saved Pitch Lab results.json', defaultDir=str(self.root), wildcard='Pitch Lab reports (*.json)|*.json', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() != wx.ID_OK: return
            path = Path(dialog.GetPath())
        try:
            if path.stat().st_size > 8 * 1024 ** 2: raise ValueError('Comparison report is too large.')
            self.show_report(json.loads(path.read_text(encoding='utf-8')), path.parent)
            self.say('Saved comparison opened. Results can be played without running inference again.')
        except Exception as error: self.say('Cannot open comparison: ' + str(error))

    def open_folder(self, event=None):
        import os
        os.startfile(self.directory or self.root)

    def cancel_operation(self, event=None):
        self.cancel.set(); self.stop_capture.set()
        def stop_client():
            with self.client_lock: client = self.client
            if client: client.close()
        Thread(target=stop_client, daemon=True).start()
        self.say('Cancellation requested. Only the isolated comparison worker is stopped; completed results are kept.')

    def close_lab(self, event=None):
        if self.busy:
            self.cancel_operation()
            self.closing = True
            if isinstance(event, wx.CloseEvent): event.Veto()
            return
        self.EndModal(wx.ID_CLOSE)
