import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import desktop
from enhanced import Enhanced


class RecordingFocusTests(unittest.TestCase):
    def test_enhanced_recording_keeps_existing_control(self):
        frame = Mock(busy=False, recorder=None, preserve_recording_focus=True)
        frame.IsActive.return_value = True
        original = Mock()
        with patch.object(desktop.wx.Window, 'FindFocus', return_value=original), patch.object(desktop, 'Recorder'):
            desktop.Workbench.begin_recording(frame)
        self.assertIs(frame.recording_focus, original)
        frame.stop_button.SetFocus.assert_not_called()

    def test_processing_restores_pre_recording_control(self):
        original = Mock()
        frame = Mock(busy=False, recording_focus=original, buttons=[])
        frame.IsActive.return_value = True
        with patch.object(desktop.wx.Window, 'FindFocus', return_value=Mock()), patch.object(desktop, 'Thread'):
            desktop.Workbench.run(frame, 'Converting', lambda: 'done')
        self.assertIs(frame.return_focus, original)
        self.assertIsNone(frame.recording_focus)

    def test_recording_error_restores_original_focus(self):
        original = Mock()
        frame = Mock(recording_focus=original)
        frame.IsActive.return_value = True
        frame.recorder.stop.side_effect = ValueError('No microphone signal')
        desktop.Workbench.end_recording(frame)
        original.SetFocus.assert_called_once()
        self.assertIsNone(frame.recording_focus)

    def test_cancel_restores_original_focus(self):
        original = Mock()
        frame = Mock(recording_focus=original)
        frame.IsActive.return_value = True
        Enhanced.cancel_recording(frame)
        original.SetFocus.assert_called_once()
        self.assertIsNone(frame.recording_focus)

    def test_background_completion_does_not_take_focus(self):
        original = Mock()
        frame = Mock(busy=True, return_focus=original, buttons=[])
        frame.IsActive.return_value = False
        desktop.Workbench.done(frame, 'Conversion finished')
        original.SetFocus.assert_not_called()


if __name__ == '__main__': unittest.main()
