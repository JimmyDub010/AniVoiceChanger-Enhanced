"""Enhanced application: isolated library, safe import and resilient global PTT."""
import json
import os
from pathlib import Path
from threading import Thread
from datetime import datetime
from uuid import uuid4

import comtypes
import wx
import soundfile as sf
from wx.lib.scrolledpanel import ScrolledPanel

import desktop
from desktop import Workbench, NamedAccessible
from voice_library import VoiceLibrary, SelectionRequired, atomic_json, AUTO
from voice_dialog import VoiceDialog, METHOD_LABELS, method_key, method_label, restored_method
from global_hotkeys import CaptureKeys, GlobalKeyBridge
from audio_effects import settings as effect_settings
from routing import playback_many
from activity import Generations

ROOT = Path(__file__).resolve().parent


class Enhanced(Workbench):
    preserve_recording_focus = True
    def __init__(self):
        self.applying_voice = True
        self.library = None
        self.current_voice = None
        self.effects = {}
        self.bridge = None
        self.generations = Generations()
        self.pending_voice = None
        self.active_voice = None
        self.loading_voice = None
        self.switch_timer = None
        self.calibration = None
        try: self.extra = json.loads(desktop.SETTINGS.read_text(encoding='utf-8'))
        except (ValueError, OSError): self.extra = {}
        super().__init__()
        self.SetTitle('AniVoiceChanger Enhanced — Voice Library and Push-to-Talk')
        self.book = wx.Notebook(self)
        self.panel.Reparent(self.book)
        self.library_panel = ScrolledPanel(self.book)
        self.library_layout = wx.BoxSizer(wx.VERTICAL)
        self.library_panel.SetSizer(self.library_layout)
        self.library_layout.Add(wx.StaticText(self.library_panel, label='Import voice files into your own library. Source ZIPs and downloads are never deleted.\nSelect a saved voice to load it. Use Control+Tab to switch to recording and routing controls.'), 0, wx.ALL, 8)
        self.library_label('Saved Voices')
        self.saved_voices = wx.Choice(self.library_panel)
        self.saved_voices.SetName('Saved Voices')
        self.saved_voices.SetAccessible(NamedAccessible(self.saved_voices, 'Saved Voices'))
        self.library_layout.Add(self.saved_voices, 0, wx.EXPAND | wx.ALL, 8)
        self.saved_voices.Bind(wx.EVT_CHOICE, self.voice_selected)
        self.active_label = wx.StaticText(self.library_panel, label='Active voice: none loaded')
        self.library_layout.Add(self.active_label, 0, wx.ALL, 8)
        self.library_label('Voice Library folder')
        self.library_path = wx.TextCtrl(self.library_panel, style=wx.TE_READONLY)
        self.library_path.SetAccessible(NamedAccessible(self.library_path, 'Voice Library folder'))
        self.library_layout.Add(self.library_path, 0, wx.EXPAND | wx.ALL, 8)
        for label, handler in [('&Import Voice from ZIP', self.import_zip), ('Import &Extracted Voice', self.import_extracted),
                               ('&Rename Voice', self.rename_voice), ('Edit Voice &Settings', self.edit_voice),
                               ('&Remove Voice from Library (keep files)', self.remove_voice),
                               ('Delete Imported Voice &Files', self.delete_voice_files),
                               ('&Browse and Relink Missing Voice', self.relink_voice),
                               ('Re&fresh Voice Library', self.refresh_library), ('&Open Voice Library Folder', self.open_library),
                               ('Choose Voice Library &Folder', self.choose_library)]:
            button = wx.Button(self.library_panel, label=label)
            button.Bind(wx.EVT_BUTTON, handler)
            self.library_layout.Add(button, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        calibration_button = wx.Button(self.library_panel, label='Auto &Pitch Calibration')
        calibration_button.Bind(wx.EVT_BUTTON, self.open_calibration)
        self.library_layout.Add(calibration_button, 0, wx.EXPAND | wx.ALL, 8)
        self.book.AddPage(self.library_panel, 'Voice Library', select=True)
        self.book.AddPage(self.panel, 'Conversion and Audio Routing')
        self.library_panel.SetupScrolling(scroll_x=False)
        # The global status stays available regardless of the selected notebook page.
        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(self.book, 1, wx.EXPAND)
        layout.Add(wx.StaticText(self, label='Enhanced status'), 0, wx.LEFT, 8)
        self.global_status = wx.TextCtrl(self, style=wx.TE_READONLY | wx.TE_MULTILINE, size=(-1, 75))
        self.global_status.SetAccessible(NamedAccessible(self.global_status, 'Enhanced status'))
        layout.Add(self.global_status, 0, wx.EXPAND | wx.ALL, 8)
        self.SetSizer(layout)
        self.method.SetItems(METHOD_LABELS)
        self.method.SetStringSelection(method_label(self.extra.get('method', 'rmvpe')))
        self.method.Unbind(wx.EVT_CHOICE)
        self.method.Bind(wx.EVT_CHOICE, self.method_changed)
        self.pitch.Bind(wx.EVT_TEXT, lambda event: self.save())
        self.index_rate.Bind(wx.EVT_TEXT, lambda event: self.save())
        self.layout.Insert(self.layout.GetItemCount() - 2, wx.StaticText(self.panel, label='Global recording mode'), 0, wx.LEFT, 8)
        self.toggle_record = wx.CheckBox(self.panel, label='Toggle-to-record instead of hold-to-talk')
        self.toggle_record.SetValue(self.extra.get('toggle_record', False))
        self.toggle_record.Bind(wx.EVT_CHECKBOX, self.record_mode_changed)
        self.layout.Insert(self.layout.GetItemCount() - 2, self.toggle_record, 0, wx.ALL, 8)
        self.keys = CaptureKeys(self.hotkey_start, self.hotkey_stop, self.hotkey_ready, self.announce,
                                lambda: self.bridge.physically_down())
        self.bridge = GlobalKeyBridge(self.keys, wx.CallAfter)
        self.Bind(wx.EVT_CHAR_HOOK, self.cancel_key)
        self.cancel_button = wx.Button(self, label='Cancel recording (Escape)')
        self.cancel_button.Bind(wx.EVT_BUTTON, self.cancel_recording)
        layout.Add(self.cancel_button, 0, wx.ALL, 8)
        self.timer.Start(100)
        self.applying_voice = False
        self.set_library(self.extra.get('library_root') or ROOT / 'VoiceLibrary')
        self.Layout()
        wx.CallAfter(self.saved_voices.SetFocus)
        last = self.library.selected() if self.library else None
        if last: wx.CallAfter(self.select_voice, last)

    def library_label(self, label):
        self.library_layout.Add(wx.StaticText(self.library_panel, label=label), 0, wx.LEFT | wx.RIGHT, 8)

    def announce(self, message):
        super().announce(message)
        if hasattr(self, 'global_status'): self.global_status.SetValue(message)

    def save(self):
        if self.applying_voice or not hasattr(self, 'status'): return
        if self.library and self.current_voice:
            try:
                voice = self.library.get(self.current_voice)
                if (self.model.GetValue() != str(self.library.path_for(voice, 'model'))
                        or self.index.GetValue() != str(self.library.path_for(voice, 'index') or '')):
                    self.current_voice = None
                    self.saved_voices.SetSelection(wx.NOT_FOUND)
                    self.library.select(None)
            except StopIteration: self.current_voice = None
        super().save()
        self.settings['method'] = method_key(self.method.GetStringSelection())
        self.settings['toggle_record'] = self.toggle_record.GetValue() if hasattr(self, 'toggle_record') else self.extra.get('toggle_record', False)
        if self.library: self.settings['library_root'] = str(self.library.root)
        atomic_json(desktop.SETTINGS, self.settings)
        if self.library and self.current_voice:
            try:
                voice = self.library.get(self.current_voice)
                prefs = dict(voice['preferences'])
                prefs.update({'pitch': self.pitch.GetValue(), 'method': method_key(self.method.GetStringSelection()),
                         'index_rate': self.index_rate.GetValue() / 100 if voice.get('index') else 0,
                         'effects': self.effects})
                if prefs != voice['preferences']: self.library.edit(self.current_voice, preferences=prefs)
            except (ValueError, StopIteration) as error:
                self.announce('Voice preferences could not be saved: ' + str(error))

    def method_changed(self, event):
        # Programmatic restores use SetStringSelection, which emits no choice event.
        if self.applying_voice: return
        self.save()
        self.announce('Pitch extraction: ' + self.method.GetStringSelection() +
                      ('. Saved for this voice.' if self.current_voice else '.'))

    def panel_controls(self, enabled):
        super().panel_controls(enabled)
        # Capture settings are snapshotted before recording/conversion. These
        # controls can safely set the next utterance without moving keyboard focus.
        self.pitch.Enable()
        self.index_rate.Enable()
        if hasattr(self, 'library_panel'):
            for child in self.library_panel.GetChildren():
                if isinstance(child, wx.Button): child.Enable(enabled)
            # Arrow navigation remains available; only the latest request is loaded.
            self.saved_voices.Enable(self.calibration is None)
        if self.bridge and self.bridge.hooks:
            self.hotkey.Disable()
            self.toggle_record.Disable()

    def run(self, message, function, completed=None):
        if self.busy or self.recorder: return
        if completed is None:
            return super().run(message, function)
        if self.busy or self.recorder: return
        self.return_focus = wx.Window.FindFocus() if self.IsActive() else None
        self.busy = True
        self.panel_controls(False)
        self.announce(message)
        def worker():
            comtypes.CoInitialize()
            try: result = function()
            except Exception as error: result = error
            finally: comtypes.CoUninitialize()
            wx.CallAfter(finish, result)
        def finish(result):
            if self.pending_voice:
                self.refresh_library()
                self.done('Operation finished.')
                return
            self.done('Operation finished.' if not isinstance(result, Exception) else str(result))
            completed(result)
        Thread(target=worker, daemon=True).start()

    def set_library(self, folder):
        try:
            selected = Path(folder).expanduser().resolve()
            if (self.engine.GetValue().strip() and selected.is_relative_to(Path(self.engine.GetValue()).resolve())) or '_preserved' in selected.parts:
                raise ValueError('Choose a Voice Library folder separate from the RVC engine and protected backups.')
            self.library = VoiceLibrary(selected)
            self.current_voice = None
            self.library_path.ChangeValue(str(self.library.root))
            self.refresh_library()
            self.save()
            if self.library.recovered: self.announce(self.library.warning)
        except Exception as error:
            self.announce('Voice Library could not be opened: ' + str(error))

    def refresh_library(self, event=None):
        if not self.library: return
        try:
            self.voice_items = self.library.voices()
            counts = {}
            for voice in self.voice_items: counts[voice['name']] = counts.get(voice['name'], 0) + 1
            labels = [v['name'] + (' — ' + v['id'][:6] if counts[v['name']] > 1 else '') for v in self.voice_items]
            self.saved_voices.SetItems(labels)
            ids = [v['id'] for v in self.voice_items]
            if self.current_voice in ids: self.saved_voices.SetSelection(ids.index(self.current_voice))
            self.library_panel.Layout()
            if event: self.announce(f'Voice Library refreshed. {len(labels)} saved voices.')
        except Exception as error:
            self.announce('Voice Library error: ' + str(error))

    def selected_voice(self):
        index = self.saved_voices.GetSelection()
        if index < 0: raise ValueError('Choose a saved voice first.')
        return self.voice_items[index]

    def voice_selected(self, event):
        self.request_voice(self.selected_voice()['id'])

    def request_voice(self, voice_id):
        if self.calibration:
            self.announce('Close calibration before switching voices.')
            self.refresh_library()
            return
        self.generations.advance()
        self.pending_voice = voice_id
        self.loaded = None
        if self.recorder: self.cancel_recording()
        self.active_voice = None
        self.active_label.SetLabel('Active voice: switching; recording unavailable')
        if self.switch_timer: self.switch_timer.Stop()
        self.switch_timer = wx.CallLater(300, self.drain_voice)

    def drain_voice(self):
        if self.busy or not self.pending_voice: return
        voice_id, self.pending_voice = self.pending_voice, None
        self.select_voice(voice_id)

    def select_voice(self, voice_id):
        if self.busy or self.recorder:
            self.request_voice(voice_id)
            return
        try:
            self.save()
            voice = self.library.get(voice_id)
            incoming_method = restored_method(voice['preferences'], self.method.GetStringSelection())
            missing = [key for key in ('model', 'index') if voice.get(key) and not self.library.path_for(voice, key).is_file()]
            if missing:
                self.active_voice = self.current_voice = None
                self.loaded = None
                self.active_label.SetLabel('Active voice: none; selected voice files are missing')
                self.library.select(None)
                self.announce('Saved voice files are missing. Browse and relink them to restore this voice.')
                if wx.MessageBox('This voice has missing model or index files. Browse and relink them?', 'Missing voice files', wx.YES_NO | wx.ICON_QUESTION, self) == wx.YES:
                    self.relink_voice(None, voice)
                else: self.refresh_library()
                return
            self.applying_voice = True
            self.current_voice = None
            self.active_voice = None
            self.loading_voice = voice_id
            self.model.ChangeValue(str(self.library.path_for(voice, 'model')))
            self.index.ChangeValue(str(self.library.path_for(voice, 'index') or ''))
            prefs = voice['preferences']
            self.pitch.SetValue(prefs['pitch'])
            self.index_rate.SetValue(round(prefs['index_rate'] * 100))
            if not self.method.SetStringSelection(method_label(incoming_method)):
                raise ValueError('The saved pitch extraction method cannot be displayed. No model was activated.')
            self.effects = effect_settings(prefs.get('effects'))
            self.applying_voice = False
            ids = [v['id'] for v in self.voice_items]
            self.saved_voices.SetSelection(ids.index(voice_id))
            self.active_label.SetLabel('Active voice: loading ' + voice['name'])
            self.announce('Loading saved voice ' + voice['name'] + '.')
            self.load_model()
        except Exception as error:
            self.applying_voice = False
            self.loading_voice = None
            self.loaded = None
            self.active_voice = self.current_voice = None
            self.active_label.SetLabel('Active voice: none; selection failed')
            self.announce('Voice selection failed: ' + str(error))

    def done(self, message):
        if self.pending_voice:
            self.busy = False
            self.loading_voice = None
            wx.CallAfter(self.drain_voice)
            return
        super().done(message)
        if message.startswith('Model loaded.') and not self.loading_voice:
            self.active_label.SetLabel('Active voice: custom model ' + Path(self.model.GetValue()).stem)
        if self.loading_voice:
            voice_id, self.loading_voice = self.loading_voice, None
            if not message.startswith('Model loaded.'):
                self.current_voice = self.active_voice = None
                self.loaded = None
                self.active_label.SetLabel('Active voice: none; loading failed')
                self.library.select(None)
                self.announce('Voice loading failed. No voice is active. ' + message)
                return
            self.current_voice = self.active_voice = voice_id
            self.library.select(voice_id)
            self.save()
            self.active_label.SetLabel('Active voice: ' + self.library.get(voice_id)['name'])
        if message.startswith('Model loaded.') and self.active_voice:
            try:
                suffix = ' Singing pitch correction is enabled.' if self.effects.get('singing_mode') and self.effects.get('pitch_correction') else ''
                self.announce(self.library.get(self.current_voice)['name'] + ' is ready. Pitch extraction: ' +
                              self.method.GetStringSelection() + '.' + suffix)
            except StopIteration: pass

    def conversion_settings(self):
        if self.pending_voice or self.loading_voice:
            raise ValueError('Wait for the selected voice to finish loading.')
        result = super().conversion_settings()
        result['method'] = method_key(result['method'])
        result['effects'] = effect_settings(self.effects)
        result['_generation'] = self.generations.snapshot()[0]
        return result

    def converted_playback(self, source, settings, destinations):
        settings = dict(settings)
        generation = settings.pop('_generation')
        number, cancel = self.generations.snapshot()
        if generation != number or cancel.is_set(): return 'Previous voice conversion canceled.'
        client = self.client
        target = ROOT / 'outputs' / (datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:8] + '.wav')
        result = client.request('convert', input=str(source), output=str(target), **settings)
        if not self.generations.valid(generation): return 'Previous voice conversion discarded.'
        warnings = ' '.join(result.get('warnings', []))
        if warnings: wx.CallAfter(self.announce, warnings)
        audio, rate = sf.read(target, dtype='float32')
        try: playback_many(destinations, audio, rate, cancel)
        except Exception as error: raise RuntimeError(f'Conversion saved to {target}, but playback failed: {error}') from error
        return (f'Conversion and playback finished. Processing {result["processing_seconds"]:.2f} seconds. '
                f'Index searches: {result["index"]["search_calls"]}. {warnings} Saved to {target}.')

    def load_model(self, event=None):
        if self.busy or self.recorder: return
        if not self.loading_voice:
            self.generations.advance()
            self.active_voice = self.current_voice = None
            self.active_label.SetLabel('Active voice: loading custom model')
        super().load_model(event)

    def import_preferences(self):
        return {'pitch': self.pitch.GetValue(), 'method': method_key(self.method.GetStringSelection()),
                'index_rate': self.index_rate.GetValue() / 100, 'effects': {}}

    def import_zip(self, event=None, source=None, model=None, index=AUTO):
        if not self.library: return
        if source is None:
            with wx.FileDialog(self, 'Import Voice from ZIP', wildcard='ZIP archives (*.zip)|*.zip', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
                if dialog.ShowModal() != wx.ID_OK: return
                source = dialog.GetPath()
        prefs = self.import_preferences()
        def completed(result):
            if isinstance(result, SelectionRequired):
                labels = ['No index (optional)' if item is None else item for item in result.choices]
                self.announce('More than one compatible choice was found. ' + str(result))
                with wx.SingleChoiceDialog(self, str(result), 'Choose Voice ' + result.kind, labels) as dialog:
                    if dialog.ShowModal() != wx.ID_OK:
                        self.announce('Import canceled. No voice was added.')
                        return
                    chosen = result.choices[dialog.GetSelection()]
                self.import_zip(source=source, model=chosen if result.kind == 'model' else result.model,
                                index=AUTO if result.kind == 'model' else chosen)
            elif isinstance(result, Exception): self.announce('Voice import failed: ' + str(result))
            else: self.import_finished(result)
        self.run('Checking ZIP and validating voice tensors. Please wait.',
                 lambda: self.library.import_zip(source, model, index, preferences=prefs), completed)

    def import_finished(self, result):
        voice = result['voice']
        self.refresh_library()
        self.announce(('This model/index pair is already imported: ' if result['duplicate'] else 'Voice imported: ') + voice['name'] + '. The original download is no longer required. ' + result.get('warning', ''))
        self.select_voice(voice['id'])

    def browse_extracted(self):
        with wx.FileDialog(self, 'Choose extracted RVC model', wildcard='RVC models (*.pth)|*.pth', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() != wx.ID_OK: return None
            model = dialog.GetPath()
        candidate = Path(model).with_suffix('.index')
        if candidate.is_file(): return model, str(candidate)
        with wx.MessageDialog(self, 'Does this voice have an index file? Choose Yes to browse, or No to import without one.', 'Optional retrieval index', wx.YES_NO | wx.CANCEL | wx.ICON_QUESTION) as question:
            choice = question.ShowModal()
        if choice == wx.ID_CANCEL: return None
        if choice == wx.ID_NO: return model, None
        with wx.FileDialog(self, 'Choose matching RVC index', defaultDir=str(Path(model).parent), wildcard='RVC indexes (*.index)|*.index', style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() != wx.ID_OK: return None
            return model, dialog.GetPath()

    def import_extracted(self, event):
        files = self.browse_extracted()
        if files:
            prefs = self.import_preferences()
            self.run('Copying and validating extracted voice files.', lambda: self.library.import_files(*files, preferences=prefs),
                     lambda result: self.announce('Voice import failed: ' + str(result)) if isinstance(result, Exception) else self.import_finished(result))

    def rename_voice(self, event):
        try: voice = self.selected_voice()
        except ValueError as error: self.announce(str(error)); return
        with wx.TextEntryDialog(self, 'Voice display name', 'Rename Voice', voice['name']) as dialog:
            if dialog.ShowModal() != wx.ID_OK: return
            name = dialog.GetValue().strip()
        try:
            self.library.edit(voice['id'], name=name)
            self.refresh_library()
            self.announce('Voice renamed to ' + name + '.')
        except Exception as error: self.announce(str(error))

    def edit_voice(self, event):
        try: voice = self.selected_voice()
        except ValueError as error: self.announce(str(error)); return
        with VoiceDialog(self, voice) as dialog:
            if dialog.ShowModal() != wx.ID_OK: return
            values = dialog.values()
        self.library.edit(voice['id'], **values)
        self.select_voice(voice['id'])

    def unload_current(self):
        if self.busy or self.recorder:
            raise ValueError('Wait for the active operation before unloading a voice.')
        self.generations.advance()
        if self.bridge: self.bridge.disable()
        self.hotkey_button.SetLabel('Enable push-to-talk hotkey')
        if self.client: self.client.close()
        self.client = None
        self.loaded = None
        self.current_voice = None
        self.active_voice = None
        self.active_label.SetLabel('Active voice: none loaded')

    def remove_voice(self, event):
        try: voice = self.selected_voice()
        except ValueError as error: self.announce(str(error)); return
        if wx.MessageBox('Remove ' + voice['name'] + ' from Saved Voices? Its imported files will be kept.', 'Remove Voice from Library', wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION, self) != wx.YES: return
        if voice['id'] == self.current_voice: self.unload_current()
        self.library.remove(voice['id'])
        self.refresh_library()
        self.announce('Voice removed from the registry. Imported files were kept.')

    def delete_voice_files(self, event):
        voices = self.library.voices(include_removed=True)
        if not voices: self.announce('There are no imported voice files to delete.'); return
        with wx.SingleChoiceDialog(self, 'Choose imported files to delete. Original downloads are never deleted.', 'Delete Imported Voice Files', [v['name'] + ' — ' + v['id'][:6] for v in voices]) as dialog:
            if dialog.ShowModal() != wx.ID_OK: return
            voice = voices[dialog.GetSelection()]
        if wx.MessageBox('Permanently delete this imported library copy of ' + voice['name'] + '? This cannot be undone. Original downloads will not be touched.', 'Confirm deletion of imported files', wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING, self) != wx.YES: return
        try:
            if voice['id'] == self.current_voice: self.unload_current()
            self.library.delete_files(voice['id'], confirmed=True)
            self.refresh_library()
            self.announce('Imported files deleted. Original downloads were not changed.')
        except Exception as error: self.announce('Deletion failed: ' + str(error))

    def relink_voice(self, event, voice=None):
        try: voice = voice or self.selected_voice()
        except ValueError as error: self.announce(str(error)); return
        files = self.browse_extracted()
        if not files: return
        self.run('Copying replacement files into a new library directory.',
                 lambda: self.library.import_files(*files, preferences=voice['preferences'], relink_id=voice['id']),
                 lambda result: self.announce('Relink failed: ' + str(result)) if isinstance(result, Exception) else self.import_finished(result))

    def choose_library(self, event):
        with wx.DirDialog(self, 'Choose Voice Library folder', defaultPath=str(self.library.root)) as dialog:
            if dialog.ShowModal() != wx.ID_OK: return
            folder = dialog.GetPath()
        self.unload_current()
        self.set_library(folder)
        self.announce('Voice Library folder selected. Existing libraries were not moved or deleted.')

    def open_library(self, event):
        os.startfile(self.library.root)

    def record_mode_changed(self, event):
        if self.bridge and self.bridge.hooks:
            self.announce('Disable the global hotkey before changing its recording mode.')
            return
        self.save()

    def hotkey_ready(self):
        if self.busy or self.recorder or self.calibration or self.pending_voice or self.loading_voice: return False
        try:
            self.conversion_settings(); self.destinations()
            return True
        except Exception as error:
            self.announce(str(error)); return False

    def hotkey_start(self):
        self.begin_recording()
        return self.recorder is not None

    def begin_recording(self, event=None):
        if self.calibration:
            self.announce('Close calibration before normal recording.')
            return
        if self.IsActive() and not self.busy and not self.recorder and hasattr(self, 'book'):
            self.book.ChangeSelection(1)
        return super().begin_recording(event)

    def hotkey_stop(self, reason):
        if self.recorder: self.end_recording()
        if 'Recovered' in reason or 'limit' in reason: self.announce(reason)

    def cancel_key(self, event):
        key = event.GetKeyCode()
        if key == wx.WXK_TAB and event.ControlDown():
            self.book.AdvanceSelection(not event.ShiftDown())
            (self.saved_voices if self.book.GetSelection() == 0 else self.engine).SetFocus()
            self.announce(self.book.GetPageText(self.book.GetSelection()))
        elif key == ord('P') and event.AltDown():
            self.open_calibration()
        elif key == wx.WXK_TAB and wx.Window.FindFocus() == self:
            (self.saved_voices if self.book.GetSelection() == 0 else self.engine).SetFocus()
        elif key == wx.WXK_ESCAPE and self.recorder:
            self.cancel_recording()
        else:
            event.Skip()

    def cancel_recording(self, event=None):
        if not self.recorder: return
        recorder, self.recorder = self.recorder, None
        self.keys.recording = False
        try:
            recorder.stream.abort()
        finally:
            recorder.stream.close()
            self.panel_controls(True)
            focus = getattr(self, 'recording_focus', None)
            if self.IsActive() and focus and focus.IsEnabled(): focus.SetFocus()
            self.recording_focus = None
            self.announce('Recording canceled. No audio was converted or played.')

    def toggle_hotkey(self, event):
        if self.bridge.hooks:
            self.bridge.disable()
            self.hotkey_button.SetLabel('Enable push-to-talk hotkey')
            self.hotkey.Enable(); self.toggle_record.Enable()
            self.announce('Global push-to-talk disabled.')
            return
        if not self.hotkey_ready(): return
        try:
            self.bridge.enable(self.hotkey.GetValue(), self.toggle_record.GetValue())
            self.hotkey.Disable(); self.toggle_record.Disable()
            self.hotkey_button.SetLabel('Disable push-to-talk hotkey')
            self.announce('Global ' + self.hotkey.GetValue() + (' toggle recording enabled.' if self.toggle_record.GetValue() else ' hold-to-talk enabled, including while minimized.'))
            self.save()
        except Exception as error: self.announce('Global hotkey could not be enabled: ' + str(error))

    def end_recording(self, event=None):
        if hasattr(self, 'keys'): self.keys.recording = False
        return super().end_recording(event)

    def check_recording(self, event):
        if self.bridge:
            self.keys.tick()
        super().check_recording(event)

    def close(self, event):
        if self.calibration:
            self.announce('Close calibration before closing Enhanced.')
            event.Veto()
            return
        if not self.busy and not self.recorder:
            if self.switch_timer: self.switch_timer.Stop()
            self.pending_voice = None
            self.generations.advance()
        if not self.busy and not self.recorder and self.bridge: self.bridge.disable()
        super().close(event)

    def open_calibration(self, event=None):
        if self.busy or self.recorder or not self.active_voice:
            self.announce('Load a saved voice and finish the current operation first.')
            return
        from calibration_dialog import CalibrationDialog
        dialog = CalibrationDialog(self)
        self.calibration = dialog
        try: dialog.ShowModal()
        finally:
            self.calibration = None
            dialog.Destroy()


if __name__ == '__main__':
    app = wx.App(False)
    # Avoid two Enhanced instances competing for hotkeys and the registry.
    instance = wx.SingleInstanceChecker('AniVoiceChanger-Enhanced-' + wx.GetUserId())
    if instance.IsAnotherRunning():
        wx.MessageBox('AniVoiceChanger Enhanced is already running.', 'Enhanced', wx.OK | wx.ICON_INFORMATION)
    else:
        frame = Enhanced()
        frame.Show()
        app.MainLoop()
