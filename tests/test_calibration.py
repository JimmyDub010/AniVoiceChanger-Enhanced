import json
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import Mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pitch_calibration import summarize, require_speech, candidate_pitches, rank_candidates, cache_key, convert_candidates
from voice_library import VoiceLibrary, atomic_json

TEST_ROOT = Path(__file__).resolve().parent.parent / 'test-results'
TEST_ROOT.mkdir(exist_ok=True)


def analysis(hz=130, voiced=1., clipped=0.):
    value = summarize(np.ones(48000, dtype='float32') * .1, 48000,
                      np.r_[np.full(round(100 * voiced), hz), np.zeros(100 - round(100 * voiced))])
    value['clipped_fraction'] = clipped
    return value


class CalibrationTests(unittest.TestCase):
    def test_pitch_distribution_and_unvoiced_frames(self):
        value = analysis(140, .6)
        self.assertEqual(value['median_hz'], 140)
        self.assertEqual(value['voiced_frames'], 60)
        self.assertEqual(value['voiced_fraction'], .6)

    def test_silence_unvoiced_clipping_rejected(self):
        for value in [analysis(voiced=0), analysis(voiced=.1), analysis(clipped=.1)]:
            with self.assertRaises(ValueError): require_speech(value)

    def test_unknown_target_is_symmetric_and_preserves_current_choice(self):
        self.assertEqual(candidate_pitches(analysis(), current=5), [-6, -3, 0, 3, 5, 6])

    def test_reference_changes_candidates_without_gender_or_filename(self):
        values = candidate_pitches(analysis(100), reference=analysis(200))
        self.assertEqual(values, [0, 6, 9, 12, 15, 18])
        self.assertTrue(all(-24 <= v <= 24 for v in candidate_pitches(analysis(50), reference=analysis(1000))))

    def test_non_f0_voice_only_zero(self):
        self.assertEqual(candidate_pitches(analysis(), 12, supports_f0=False), [0])

    def test_rank_reports_measured_tracking_and_clipping_not_confidence(self):
        rows = [{'pitch': 0, 'analysis': analysis(130, clipped=.1)},
                {'pitch': 12, 'analysis': analysis(260)}]
        ranked = rank_candidates(analysis(130), rows)
        self.assertEqual(ranked[0]['pitch'], 12)
        self.assertEqual(ranked[0]['tracking_error_semitones'], 0)
        self.assertNotIn('confidence', ranked[0])

    def test_cancellation_waits_for_current_inference_and_skips_remaining(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            client = Mock()
            cancel = Event()
            client.request.side_effect = lambda *args, **kwargs: cancel.set()
            result = convert_candidates(client, 'input.wav', folder, {}, [0, 3, 6], analysis(), cancel)
            self.assertEqual(result, [])
            self.assertEqual(client.request.call_count, 1)

    def test_candidate_cache_does_not_reconvert_same_sample(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            client = Mock()
            def request(command, **params):
                if command == 'convert': Path(params['output']).write_bytes(b'audio-fixture')
                else: return analysis()
            client.request.side_effect = request
            convert_candidates(client, 'input.wav', folder, {}, [0], analysis(), Event())
            self.assertEqual(client.request.call_count, 2)
            convert_candidates(client, 'input.wav', folder, {}, [0], analysis(), Event())
            self.assertEqual(client.request.call_count, 2)

    def test_cache_key_includes_sample_model_and_settings(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            sample = Path(folder) / 'sample.wav'; sample.write_bytes(b'one')
            key = cache_key(sample, 'model-one', 'index', {'method': 'rmvpe'}, [0])
            self.assertNotEqual(key, cache_key(sample, 'model-two', 'index', {'method': 'rmvpe'}, [0]))
            sample.write_bytes(b'two')
            self.assertNotEqual(key, cache_key(sample, 'model-one', 'index', {'method': 'rmvpe'}, [0]))

    def test_saved_calibration_preserves_other_settings_and_survives_reload(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            lib = VoiceLibrary(folder)
            voice_id = 'a' * 32
            voice = dict(id=voice_id, name='Test voice', model=f'voices/{voice_id}/model.pth', index=None,
                         rvc={'f0': 1}, preferences={'pitch': 0, 'method': 'rmvpe', 'index_rate': 0, 'effects': {'limiter': True}})
            atomic_json(lib.path, dict(schema=1, voices=[voice], removed=[], last_voice=voice_id))
            lib.save_calibration(voice_id, -3, analysis())
            loaded = VoiceLibrary(folder).get(voice_id)
            self.assertEqual(loaded['preferences']['pitch'], -3)
            self.assertTrue(loaded['preferences']['effects']['limiter'])
            self.assertEqual(loaded['calibration']['selected_by'], 'user listening choice')
            lib.edit(voice_id, name='Renamed voice')
            self.assertEqual(lib.get(voice_id)['calibration']['pitch'], -3)


if __name__ == '__main__': unittest.main()
