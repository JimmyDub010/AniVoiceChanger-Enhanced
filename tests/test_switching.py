import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from threading import Event
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from activity import Generations
import enhanced
import routing


class SwitchingTests(unittest.TestCase):
    def test_control_tab_changes_page_and_moves_keyboard_focus(self):
        app, event = Mock(), Mock()
        event.GetKeyCode.return_value = enhanced.wx.WXK_TAB
        event.ControlDown.return_value = True
        event.ShiftDown.return_value = False
        app.book.GetSelection.return_value = 1
        enhanced.Enhanced.cancel_key(app, event)
        app.book.AdvanceSelection.assert_called_once_with(True)
        app.engine.SetFocus.assert_called_once()
        event.Skip.assert_not_called()

    def test_calibration_keyboard_shortcut(self):
        app, event = Mock(), Mock()
        event.GetKeyCode.return_value = ord('P')
        event.ControlDown.return_value = False
        event.AltDown.return_value = True
        enhanced.Enhanced.cancel_key(app, event)
        app.open_calibration.assert_called_once()

    def test_generation_invalidates_existing_capture_and_playback_tokens(self):
        state = Generations()
        number, cancel = state.snapshot()
        state.advance()
        self.assertTrue(cancel.is_set())
        self.assertFalse(state.valid(number))
        self.assertTrue(state.valid(state.snapshot()[0]))

    def test_latest_navigation_wins_and_waits_for_worker(self):
        app = Mock()
        app.busy = True
        app.pending_voice = 'last'
        enhanced.Enhanced.drain_voice(app)
        app.select_voice.assert_not_called()
        app.busy = False
        enhanced.Enhanced.drain_voice(app)
        app.select_voice.assert_called_once_with('last')
        self.assertIsNone(app.pending_voice)

    def test_request_invalidates_but_never_closes_in_use_client(self):
        app = Mock()
        app.calibration = None
        app.generations = Generations()
        app.recorder = None
        with patch.object(enhanced.wx, 'CallLater'):
            for voice in ['one', 'two', 'three']:
                enhanced.Enhanced.request_voice(app, voice)
        self.assertEqual(app.pending_voice, 'three')
        self.assertIsNone(app.loaded)
        app.client.close.assert_not_called()

    def test_stale_conversion_result_never_plays(self):
        app = Mock()
        app.generations = Generations()
        number, _ = app.generations.snapshot()
        def convert(*args, **kwargs):
            app.generations.advance()
            return {}
        app.client.request.side_effect = convert
        with patch.object(enhanced, 'playback_many') as play:
            result = enhanced.Enhanced.converted_playback(app, 'input.wav', {'_generation': number}, ['headphones'])
        self.assertIn('discarded', result)
        play.assert_not_called()

    def test_already_stale_job_never_calls_worker(self):
        app = Mock()
        app.generations = Generations()
        app.generations.advance()
        enhanced.Enhanced.converted_playback(app, 'input.wav', {'_generation': 0}, ['headphones'])
        app.client.request.assert_not_called()

    def test_stream_aborts_after_generation_cancel(self):
        cancel = Event()
        stream = Mock()
        stream.__enter__ = Mock(return_value=stream)
        stream.__exit__ = Mock(return_value=False)
        stream.write.side_effect = lambda data: cancel.set()
        device = {'default_samplerate': 48000, 'max_output_channels': 2, 'index': 3}
        with patch.object(routing, 'resolve', return_value=(None, device)), patch.object(routing.sd, 'OutputStream', return_value=stream):
            routing.playback('headphones', np.ones(48000, dtype='float32'), 48000, cancel)
        self.assertEqual(stream.write.call_count, 1)
        stream.abort.assert_called_once()

    def test_duplicate_output_monitor_endpoint_plays_once(self):
        with patch.object(routing, 'playback') as play:
            routing.playback_many(['headphones', 'headphones'], np.ones(100), 48000)
        self.assertEqual(play.call_count, 1)


if __name__ == '__main__': unittest.main()
