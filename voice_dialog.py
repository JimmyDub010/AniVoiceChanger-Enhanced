"""Native keyboard-accessible per-voice metadata and processing preferences."""
import wx
from wx.lib.scrolledpanel import ScrolledPanel

from desktop import NamedAccessible
from voice_library import CATEGORIES, USAGES, METHODS
from audio_effects import DEFAULTS, KEYS, SCALES, settings
from pitch_methods import LABELS, DEVICE_NOTE

METHOD_LABELS = [LABELS[key] for key in METHODS]
LEGACY_LABELS = {'RMVPE New': 'rmvpe-new', 'FCPE Experimental': 'fcpe-experimental',
                 'Mangio-style CREPE 80 Experimental': 'mangio-crepe-80', 'RMVPE Range Experimental': 'rmvpe-range',
                 'RMVPE + FCPE Hybrid Experimental': 'hybrid-rmvpe-fcpe', 'SwiftF0 0.3.0 (CPU)': 'swiftf0'}


def method_key(label):
    key = {value: key for key, value in LABELS.items()}.get(label, LEGACY_LABELS.get(label, label))
    if key not in METHODS:
        raise ValueError('Choose a supported pitch extraction method. No fallback method was used.')
    return key


def method_label(key):
    return LABELS.get(LEGACY_LABELS.get(key, key), key)


def restored_method(preferences, current):
    """A missing per-voice value inherits the current method, never a default."""
    return method_key(preferences['method'] if 'method' in preferences else current)


class VoiceDialog(wx.Dialog):
    def __init__(self, parent, voice):
        super().__init__(parent, title='Edit Voice Settings — ' + voice['name'], size=(600, 650), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.voice = voice
        self.controls = {}
        panel = ScrolledPanel(self)
        self.panel = panel
        self.fields = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(self.fields)
        prefs = voice['preferences']
        effects = settings(prefs.get('effects'))
        self.add('name', 'Voice display name', lambda: wx.TextCtrl(panel, value=voice['name']))
        self.choice('category', 'Voice category', CATEGORIES, voice.get('category', 'Other'))
        self.choice('usage', 'Usage type', USAGES, voice.get('usage', 'Unknown'))
        self.spin('pitch', 'Preferred pitch shift in semitones', -24, 24, prefs['pitch'])
        note = wx.StaticText(panel, label=DEVICE_NOTE)
        note.Wrap(520)
        self.fields.Add(note, 0, wx.EXPAND | wx.ALL, 8)
        self.choice('method', 'Preferred pitch extraction', METHOD_LABELS,
                    method_label(restored_method(prefs, parent.method.GetStringSelection())))
        self.spin('index_rate', 'Preferred index influence percent', 0, 100, round(prefs['index_rate'] * 100))
        self.spin('input_gain_db', 'Input gain in dB', -24, 12, round(effects['input_gain_db']))
        self.check('noise_reduction', 'Gentle input noise reduction (experimental)', effects['noise_reduction'])
        self.spin('output_gain_db', 'Output gain in dB', -24, 12, round(effects['output_gain_db']))
        self.check('normalize', 'Normalize output loudness (RMS target minus 20 dBFS)', effects['normalize'])
        self.check('compression', 'Gentle output compression', effects['compression'])
        self.check('limiter', 'Prevent output clipping with a sample-peak limiter', effects['limiter'])
        self.choice('reverb', 'Optional microphone reverb', ['Off', 'Small room', 'Wrestling arena'], effects['reverb'])
        self.check('singing_mode', 'Singing mode — allow experimental pitch correction', effects['singing_mode'])
        self.check('pitch_correction', 'Experimental pitch correction (only in Singing mode)', effects['pitch_correction'])
        self.spin('correction_strength', 'Pitch correction strength percent', 0, 100, round(effects['correction_strength'] * 100))
        self.choice('musical_key', 'Musical key', KEYS, effects['musical_key'])
        self.choice('scale', 'Musical scale', list(SCALES), effects['scale'])
        dry = wx.Button(panel, label='Dry singing — disable all added processing')
        dry.Bind(wx.EVT_BUTTON, self.dry)
        self.fields.Add(dry, 0, wx.ALL, 8)
        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(panel, 1, wx.EXPAND)
        layout.Add(self.CreateButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 8)
        self.SetSizer(layout)
        panel.SetupScrolling(scroll_x=False)
        self.Bind(wx.EVT_BUTTON, self.accept, id=wx.ID_OK)
        wx.CallAfter(self.controls['name'].SetFocus)

    def add(self, key, label, factory):
        self.fields.Add(wx.StaticText(self.panel, label=label), 0, wx.LEFT | wx.RIGHT, 8)
        control = factory()
        control.SetName(label)
        control.SetAccessible(NamedAccessible(control, label))
        self.fields.Add(control, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        self.controls[key] = control
        return control

    def choice(self, key, label, values, selected):
        control = self.add(key, label, lambda: wx.Choice(self.panel, choices=values))
        control.SetStringSelection(selected)

    def spin(self, key, label, low, high, initial):
        self.add(key, label, lambda: wx.SpinCtrl(self.panel, min=low, max=high, initial=initial))

    def check(self, key, label, initial):
        control = wx.CheckBox(self.panel, label=label)
        control.SetValue(initial)
        self.controls[key] = control
        self.fields.Add(control, 0, wx.ALL, 8)

    def dry(self, event):
        for key, value in DEFAULTS.items():
            control = self.controls[key]
            if isinstance(control, wx.Choice): control.SetStringSelection(value)
            elif isinstance(control, wx.SpinCtrl): control.SetValue(round(value * 100) if key == 'correction_strength' else round(value))
            else: control.SetValue(value)
        self.controls['singing_mode'].SetValue(True)
        self.GetParent().announce('Dry singing selected. Added processing and pitch correction are off.')

    def values(self):
        def value(key):
            control = self.controls[key]
            return control.GetStringSelection() if isinstance(control, wx.Choice) else control.GetValue()
        effects = {key: value(key) for key in DEFAULTS}
        effects['correction_strength'] /= 100
        prefs = {'pitch': value('pitch'), 'method': method_key(value('method')),
                 'index_rate': value('index_rate') / 100, 'effects': settings(effects)}
        if not self.voice.get('index'): prefs['index_rate'] = 0
        if not self.voice['rvc']['f0'] and (prefs['pitch'] or effects['pitch_correction']):
            raise ValueError('This non-F0 model requires pitch zero and pitch correction off.')
        name = value('name').strip()
        if not 1 <= len(name) <= 200: raise ValueError('Enter a display name between 1 and 200 characters.')
        return {'name': name, 'category': value('category'), 'usage': value('usage'), 'preferences': prefs}

    def accept(self, event):
        try: self.values()
        except ValueError as error:
            wx.MessageBox(str(error), 'Voice settings', wx.OK | wx.ICON_ERROR, self)
            return
        self.EndModal(wx.ID_OK)
