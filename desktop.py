"""Accessible legacy RVC push-to-talk desktop application."""
import json
from pathlib import Path
from threading import Thread
from uuid import uuid4
from datetime import datetime

import comtypes
import numpy as np
import wx
import keyboard
import soundfile as sf
from wx.lib.scrolledpanel import ScrolledPanel
from accessible_output2.outputs.nvda import NVDA

from diagnose import inspect_engine, report
from routing import endpoints, microphone_test, playback, playback_many, Recorder
from engine_client import EngineClient

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / 'settings.json'


class NamedAccessible(wx.Accessible):
    def __init__(self, window, name):
        super().__init__(window)
        self.name = name

    def GetName(self, child_id):
        if child_id == 0:
            return wx.ACC_OK, self.name
        return wx.ACC_NOT_IMPLEMENTED, ''


class Workbench(wx.Frame):
    def __init__(self):
        super().__init__(None, title='AniVoiceChanger — Development: index cache', size=(800, 720))
        self.speech = NVDA()
        self.busy = False
        self.client = None
        self.loaded = None
        self.recorder = None
        self.hotkey_hooks = []
        try:
            self.settings = json.loads(SETTINGS.read_text(encoding='utf-8'))
        except FileNotFoundError:
            self.settings = {}
        except (ValueError, OSError):
            self.settings = {}
        if not self.settings.get('engine') and (ROOT / 'RVC-beta0717/runtime/python.exe').is_file():
            self.settings['engine'] = str(ROOT / 'RVC-beta0717')
        panel = ScrolledPanel(self)
        layout = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(layout)
        self.panel = panel
        self.layout = layout
        intro = ('Original push-to-talk conversion. Load a trusted model, then record and convert.\n'
                 'For game chat, send playback to CABLE Input; choose CABLE Output as the microphone in your game or Discord.\n'
                 'Monitoring is optional. Use headphones to avoid acoustic feedback.')
        layout.Add(wx.StaticText(panel, label=intro), 0, wx.ALL, 10)
        self.engine = self.path_field('RVC installation folder', 'engine', directory=True)
        self.model = self.path_field('Voice model (.pth)', 'model', wildcard='RVC models (*.pth)|*.pth')
        self.index = self.path_field('Optional index (.index)', 'index', wildcard='RVC index (*.index)|*.index')
        self.index_rate = self.field('Index influence (0 to 100 percent)', lambda: wx.SpinCtrl(
            panel, min=0, max=100, initial=self.settings.get('index_rate', 0)))
        self.index_rate.Bind(wx.EVT_SPINCTRL, lambda event: self.save())
        self.pitch = self.field('Pitch shift in semitones', lambda: wx.SpinCtrl(panel, min=-24, max=24,
                                                                     initial=self.settings.get('pitch', 12)))
        self.pitch.Bind(wx.EVT_SPINCTRL, lambda event: self.save())
        self.method = self.field('Pitch extraction method', lambda: wx.Choice(panel, choices=['crepe', 'pm', 'harvest', 'rmvpe']))
        saved_method = self.settings.get('method', 'crepe')
        self.method.SetStringSelection(saved_method if saved_method in self.method.GetStrings() else 'crepe')
        self.method.Bind(wx.EVT_CHOICE, lambda event: self.save())
        self.hotkey = self.field('Push-to-talk key (for example f8)', lambda: wx.TextCtrl(panel, value=self.settings.get('hotkey', 'f8')))
        self.hotkey.Bind(wx.EVT_TEXT, lambda event: self.save())
        self.choices = {}
        self.device_lists = {}
        for key, label in [('microphone', 'Physical microphone'), ('output', 'Converted audio playback destination'),
                           ('monitor', 'Optional headphone monitoring destination')]:
            control = self.field(label, lambda: wx.Choice(panel))
            self.choices[key] = control
            control.Bind(wx.EVT_CHOICE, lambda event: self.save())
        self.buttons = []
        for label, handler in [('&Refresh devices', self.refresh), ('&Check RVC files', self.check),
                               ('Test &microphone level (3 seconds)', self.test_microphone),
                               ('Test &output tone', self.test_output), ('Test &headphone tone', self.test_monitor),
                               ('Save &diagnostic report', self.diagnostics),
                               ('&Load voice model', self.load_model), ('Convert an audio &file', self.convert_file),
                               ('Start &recording', self.begin_recording), ('&Enable push-to-talk hotkey', self.toggle_hotkey)]:
            button = wx.Button(panel, label=label)
            button.Bind(wx.EVT_BUTTON, handler)
            layout.Add(button, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
            self.buttons.append(button)
            if handler == self.toggle_hotkey:
                self.hotkey_button = button
        self.stop_button = wx.Button(panel, label='Stop recording and &convert')
        self.stop_button.Bind(wx.EVT_BUTTON, self.end_recording)
        self.stop_button.Disable()
        layout.Add(self.stop_button, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        self.status = self.field('Status', lambda: wx.TextCtrl(panel, style=wx.TE_READONLY | wx.TE_MULTILINE))
        self.status.SetMinSize((-1, 100))
        self.Bind(wx.EVT_CLOSE, self.close)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.check_recording, self.timer)
        self.timer.Start(1000)
        self.refresh()
        panel.SetupScrolling(scroll_x=False)
        wx.CallAfter(self.engine.SetFocus)

    def field(self, label, factory):
        label_control = wx.StaticText(self.panel, label=label)
        self.layout.Add(label_control, 0, wx.LEFT | wx.RIGHT, 8)
        control = factory()
        control.SetName(label)
        control.SetAccessible(NamedAccessible(control, label))
        self.layout.Add(control, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        return control

    def path_field(self, label, key, directory=False, wildcard='*.*'):
        control = self.field(label, lambda: wx.TextCtrl(self.panel, value=self.settings.get(key, '')))
        control.Bind(wx.EVT_TEXT, lambda event: self.save())
        button = wx.Button(self.panel, label='Browse ' + label)
        self.layout.Add(button, 0, wx.LEFT | wx.BOTTOM, 8)
        def browse(event):
            default_dir = (str(Path(self.engine.GetValue()) / 'models')
                           if hasattr(self, 'engine') and key != 'engine' else str(ROOT))
            dialog = (wx.DirDialog(self, label) if directory else
                      wx.FileDialog(self, label, defaultDir=default_dir,
                                    wildcard=wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST))
            with dialog:
                if dialog.ShowModal() == wx.ID_OK:
                    control.SetValue(dialog.GetPath())
        button.Bind(wx.EVT_BUTTON, browse)
        if key == 'model':
            control.Bind(wx.EVT_KILL_FOCUS, self.discover_index)
        return control

    def announce(self, message):
        self.status.SetValue(message)
        try:
            if self.speech.is_active():
                self.speech.speak(message)
                self.speech.braille(message)
        except Exception:
            pass  # Visible status remains available to the screen reader.

    def save(self):
        if not hasattr(self, 'status'):
            return
        data = {key: getattr(self, key).GetValue() for key in ('engine', 'model', 'index', 'pitch', 'index_rate', 'hotkey')}
        data['method'] = self.method.GetStringSelection()
        for key, control in self.choices.items():
            selection = control.GetSelection()
            if selection >= 0:
                data[key] = self.device_lists[key][selection]
            else:
                data[key] = self.settings.get(key, '')
        try:
            temporary = SETTINGS.with_suffix('.tmp')
            temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
            temporary.replace(SETTINGS)
            self.settings = data
        except OSError as error:
            self.announce('Could not save settings: ' + str(error))

    def refresh(self, event=None):
        try:
            import sounddevice as sd
            sd._terminate()
            sd._initialize()
            available = endpoints()
            unavailable = []
            for key, control in self.choices.items():
                direction = 'input' if key == 'microphone' else 'output'
                items = [d for d in available if d.direction == direction]
                ids = [''] + [d.id for d in items]
                names = ['None selected'] + [d.name for d in items]
                saved = self.settings.get(key, '')
                if saved and saved not in ids:
                    ids.append(saved)
                    names.append('Unavailable — previously selected device')
                    unavailable.append(key)
                control.SetItems(names)
                control.SetSelection(ids.index(saved) if saved in ids else 0)
                self.device_lists[key] = ids
            cable = any(d.name == 'CABLE Input (VB-Audio Virtual Cable)' for d in available)
            self.announce(('VB-CABLE detected. ' if cable else 'VB-CABLE not detected; local playback is available. ')
                          + ('Unavailable selections: ' + ', '.join(unavailable) if unavailable else 'Device lists refreshed.'))
        except Exception as error:
            self.announce('Device discovery failed: ' + str(error))

    def selected(self, key):
        selection = self.choices[key].GetSelection()
        if selection < 1:
            raise ValueError('Choose a device for ' + key + ' first.')
        return self.device_lists[key][selection]

    def run(self, message, function):
        if self.busy:
            return
        self.return_focus = (getattr(self, 'recording_focus', None) or wx.Window.FindFocus()) if self.IsActive() else None
        self.recording_focus = None
        self.busy = True
        self.panel_controls(False)
        for button in self.buttons:
            button.Disable()
        self.announce(message)
        def worker():
            comtypes.CoInitialize()
            try:
                result = function()
            except Exception as error:
                result = 'Error: ' + str(error)
            finally:
                comtypes.CoUninitialize()
            wx.CallAfter(self.done, result)
        Thread(target=worker, daemon=True).start()

    def done(self, message):
        self.busy = False
        self.panel_controls(True)
        for button in self.buttons:
            button.Enable()
        if self.IsActive() and self.return_focus and self.return_focus.IsEnabled():
            self.return_focus.SetFocus()
        self.announce(message)

    def panel_controls(self, enabled):
        for child in self.panel.GetChildren():
            if isinstance(child, (wx.TextCtrl, wx.Choice, wx.SpinCtrl, wx.Button)) and child != self.status:
                if getattr(self, 'preserve_recording_focus', False) and child in (self.pitch, self.index_rate):
                    continue
                child.Enable(enabled)
        self.stop_button.Enable(self.recorder is not None)
        if self.hotkey_hooks:
            self.hotkey.Disable()

    def discover_index(self, event=None):
        model = Path(self.model.GetValue())
        candidate = model.with_suffix('.index')
        if model.is_file() and not self.index.GetValue() and candidate.is_file():
            self.index.SetValue(str(candidate))
            self.announce('Matching index found. Set index influence above zero to enable retrieval.')
        if event:
            event.Skip()

    def load_model(self, event=None):
        if self.recorder:
            return
        self.discover_index()
        folder = self.engine.GetValue()
        model = self.model.GetValue()
        def load():
            self.loaded = None
            if self.client:
                self.client.close()
            self.client = EngineClient(folder)
            try:
                result = self.client.request('load', model=model)
            except Exception:
                self.client.close()
                self.client = None
                raise
            self.loaded = (folder, model)
            return (f'Model loaded. RVC {result["version"]}, {result["sample_rate"]} Hz, '
                    f'{"pitch enabled" if result["f0"] else "no pitch support"}, {result["gpu"]}. Ready to record.')
        self.run('Loading voice model and HuBERT. Please wait.', load)

    def conversion_settings(self):
        if not self.client or self.loaded != (self.engine.GetValue(), self.model.GetValue()):
            raise ValueError('Load the selected voice model first.')
        return {'pitch': self.pitch.GetValue(), 'method': self.method.GetStringSelection(),
                'index': self.index.GetValue(), 'index_rate': self.index_rate.GetValue() / 100}

    def destinations(self):
        output = self.selected('output')
        result = [output]
        selection = self.choices['monitor'].GetSelection()
        if selection > 0:
            monitor = self.device_lists['monitor'][selection]
            item = next((d for d in endpoints() if d.id == monitor), None)
            if not item or any(word in item.name.casefold() for word in ('cable', 'line 1')):
                raise ValueError('Choose physical headphones for optional monitoring, or select None.')
            result.append(monitor)
        return result

    def converted_playback(self, source, settings, destinations):
        target = ROOT / 'outputs' / (datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:8] + '.wav')
        result = self.client.request('convert', input=str(source), output=str(target), **settings)
        audio, rate = sf.read(target, dtype='float32')
        try:
            playback_many(destinations, audio, rate)
        except Exception as error:
            raise RuntimeError(f'Conversion saved to {target}, but playback failed: {error}') from error
        return (f'Conversion finished and playback completed. Processing {result["processing_seconds"]:.2f} seconds. '
                f'Index searches: {result["index"]["search_calls"]}. Saved to {target}.')

    def convert_file(self, event):
        try:
            settings = self.conversion_settings()
            destinations = self.destinations()
        except Exception as error:
            self.announce(str(error))
            return
        with wx.FileDialog(self, 'Choose an audio file to convert',
                           wildcard='Audio files (*.wav;*.flac;*.ogg)|*.wav;*.flac;*.ogg',
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() != wx.ID_OK:
                return
            source = dialog.GetPath()
        self.run('Converting audio. Playback will follow.', lambda: self.converted_playback(source, settings, destinations))

    def begin_recording(self, event=None):
        if self.busy or self.recorder:
            return
        try:
            self.recording_focus = wx.Window.FindFocus() if self.IsActive() else None
            self.capture_settings = self.conversion_settings()
            self.capture_destinations = self.destinations()
            recorder = Recorder(self.selected('microphone'))
            # Announce before opening capture to reduce speech feedback into the recording.
            self.announce('Recording.')
            recorder.start()
            self.recorder = recorder
            self.panel_controls(False)
            if self.IsActive() and not getattr(self, 'preserve_recording_focus', False):
                self.stop_button.SetFocus()
        except Exception as error:
            self.recording_focus = None
            self.announce('Cannot record: ' + str(error))

    def end_recording(self, event=None):
        if not self.recorder:
            return
        recorder = self.recorder
        self.recorder = None
        self.stop_button.Disable()
        try:
            audio, rate = recorder.stop()
        except Exception as error:
            self.panel_controls(True)
            focus = getattr(self, 'recording_focus', None)
            if self.IsActive() and focus and focus.IsEnabled(): focus.SetFocus()
            self.recording_focus = None
            self.announce(str(error))
            return
        settings = self.capture_settings
        destinations = self.capture_destinations
        def convert():
            folder = ROOT / '.cache' / 'recordings'
            folder.mkdir(parents=True, exist_ok=True)
            source = folder / (uuid4().hex + '.wav')
            created = False
            try:
                with source.open('xb') as file:
                    created = True
                    sf.write(file, audio, rate, format='WAV', subtype='PCM_16')
                return self.converted_playback(source, settings, destinations)
            finally:
                if created:
                    source.unlink(missing_ok=True)
        self.run('Recording finished. Converting voice, then playing to your selected output.', convert)

    def toggle_hotkey(self, event):
        if self.hotkey_hooks:
            for hook in self.hotkey_hooks:
                keyboard.unhook(hook)
            self.hotkey_hooks = []
            self.hotkey.Enable()
            self.hotkey_button.SetLabel('Enable push-to-talk hotkey')
            self.announce('Push-to-talk hotkey disabled.')
            return
        try:
            key = self.hotkey.GetValue().strip().lower()
            if not key:
                raise ValueError('Enter a push-to-talk key first.')
            self.conversion_settings()
            self.destinations()
            self.hotkey_hooks.append(keyboard.on_press_key(key, lambda event: wx.CallAfter(self.hotkey_press)))
            self.hotkey_hooks.append(keyboard.on_release_key(key, lambda event: wx.CallAfter(self.end_recording)))
            self.hotkey.Disable()
            self.hotkey_button.SetLabel('Disable push-to-talk hotkey')
            self.announce('Push-to-talk enabled. Hold ' + key + ' to speak, then release to convert. The key also reaches your game.')
        except Exception as error:
            for hook in self.hotkey_hooks:
                keyboard.unhook(hook)
            self.hotkey_hooks = []
            self.announce('Cannot enable push-to-talk: ' + str(error))

    def hotkey_press(self):
        if self.hotkey_hooks:
            self.begin_recording()

    def check_recording(self, event):
        if self.recorder and (self.recorder.errors or not self.recorder.stream.active):
            self.end_recording()

    def check(self, event):
        result = inspect_engine(self.engine.GetValue())
        self.announce('Missing RVC files: ' + ', '.join(result['missing']) if result['missing'] else
                      'Legacy RVC files found. Dependencies and actual model inference still need verification.')

    def test_microphone(self, event):
        try:
            selected = self.selected('microphone')
        except ValueError as error:
            self.announce(str(error))
            return
        def test():
            _, _, peak, rms = microphone_test(selected)
            return f'Microphone test finished. Peak {peak:.3f}, RMS {rms:.3f}. ' + (
                'No meaningful signal detected.' if peak < 0.001 else 'Audio signal detected; quality has not been assessed.')
        self.run('Recording microphone level for three seconds. Speak now.', test)

    def tone(self, key):
        try:
            selected = self.selected(key)
        except ValueError as error:
            self.announce(str(error))
            return
        def test():
            rate = 48000
            t = np.arange(rate // 2) / rate
            audio = (0.025 * np.sin(2 * np.pi * 440 * t) * np.sin(np.pi * np.arange(len(t)) / len(t)) ** 2).astype('float32')
            playback(selected, audio, rate)
            return 'Test tone sent to the selected device. This does not verify reception in a game or Discord.'
        self.run('Sending a quiet half-second test tone.', test)

    def test_output(self, event):
        self.tone('output')

    def test_monitor(self, event):
        self.tone('monitor')

    def diagnostics(self, event):
        folder = self.engine.GetValue()
        def generate():
            target = ROOT / 'diagnostics.json'
            target.write_text(json.dumps(report(folder), indent=2), encoding='utf-8')
            return 'Diagnostic report saved to ' + str(target)
        self.run('Checking Python, GPU, legacy RVC files and audio endpoints.', generate)

    def close(self, event):
        if self.recorder:
            self.announce('Stop recording before closing.')
            event.Veto()
            return
        if self.busy:
            self.announce('Wait for the current audio test or diagnostic check to finish before closing.')
            event.Veto()
            return
        self.save()
        self.timer.Stop()
        for hook in self.hotkey_hooks:
            keyboard.unhook(hook)
        if self.client:
            self.client.close()
        self.Destroy()


if __name__ == '__main__':
    app = wx.App(False)
    frame = Workbench()
    frame.Show()
    app.MainLoop()
