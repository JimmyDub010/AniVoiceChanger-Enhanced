import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice_dialog import METHOD_LABELS, method_key, method_label, restored_method
from enhanced import Enhanced
from voice_library import VoiceLibrary, atomic_json


class ExtractionMethodTests(unittest.TestCase):
    def test_every_display_label_round_trips_to_engine_key(self):
        for label in METHOD_LABELS:
            self.assertEqual(method_label(method_key(label)), label)

    def test_explicit_per_voice_method_overrides_current(self):
        self.assertEqual(restored_method({'method': 'rmvpe'}, 'FCPE Experimental'), 'rmvpe')
        self.assertEqual(restored_method({'method': 'fcpe-experimental'}, 'rmvpe'), 'fcpe-experimental')

    def test_missing_method_inherits_current_without_modifying_settings(self):
        prefs = {'pitch': 5, 'effects': {'limiter': True}}
        self.assertEqual(restored_method(prefs, 'FCPE Experimental'), 'fcpe-experimental')
        self.assertNotIn('method', prefs)

    def test_invalid_method_does_not_silently_fallback(self):
        with self.assertRaises(ValueError): restored_method({'method': 'unknown'}, 'rmvpe')
        with self.assertRaises(ValueError): method_key('')

    def test_method_event_saves_without_changing_focus(self):
        frame = Mock(applying_voice=False, current_voice='voice')
        frame.method.GetStringSelection.return_value = 'FCPE Experimental'
        Enhanced.method_changed(frame, Mock())
        frame.save.assert_called_once()
        frame.method.SetFocus.assert_not_called()

    def test_programmatic_restore_does_not_trigger_save_handler(self):
        frame = Mock(applying_voice=True)
        Enhanced.method_changed(frame, Mock())
        frame.save.assert_not_called()

    def test_two_voices_persist_distinct_methods_after_restart(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            library = VoiceLibrary(folder)
            voices = [dict(id=letter * 32, name=letter, model=f'voices/{letter * 32}/model.pth',
                           index=None, preferences={'pitch': pitch, 'method': method, 'index_rate': 0, 'effects': {'limiter': True}})
                      for letter, pitch, method in [('a', 5, 'fcpe-experimental'), ('b', -3, 'rmvpe')]]
            atomic_json(library.path, dict(schema=1, voices=voices, removed=[], last_voice=voices[0]['id']))
            for _ in range(5):
                restarted = VoiceLibrary(folder)
                self.assertEqual(restored_method(restarted.get('a' * 32)['preferences'], 'rmvpe'), 'fcpe-experimental')
                self.assertEqual(restored_method(restarted.get('b' * 32)['preferences'], 'FCPE Experimental'), 'rmvpe')
                self.assertEqual(restarted.get('a' * 32)['preferences']['pitch'], 5)
                self.assertTrue(restarted.get('b' * 32)['preferences']['effects']['limiter'])


if __name__ == '__main__': unittest.main()
