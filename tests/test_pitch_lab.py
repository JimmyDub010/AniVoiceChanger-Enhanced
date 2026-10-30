import copy
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pitch_lab import compare, make_run, snapshot_audio, save_favorite, describe_result
from pitch_methods import align, hybrid, capabilities
from voice_library import VoiceLibrary, atomic_json


class FakeClient:
    def __init__(self, cancel=None): self.calls, self.cancel = [], cancel
    def request(self, command, **values):
        if command == 'analyze': return {'median_hz': 220}
        self.calls.append(copy.deepcopy(values))
        if values['method'] == 'missing': raise RuntimeError('Missing dependency; no fallback used.')
        sf.write(values['output'], np.ones(16000, dtype='float32') * .1, 16000)
        if self.cancel: self.cancel.set()
        return {'sample_rate': 16000, 'peak_vram_bytes': 100, 'pipeline_stage_seconds': [0, .1, .1],
                'pitch_metrics': {'seconds': .1, 'voiced_fraction': .9}}


class PitchLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(Path(__file__).resolve().parent)
        self.sample = self.root / 'source.wav'
        sf.write(self.sample, np.sin(np.arange(32000) * .1).astype('float32') * .1, 16000)
        self.voice = {'id': 'a' * 32, 'name': 'Lab voice', 'model_hash': 'hash', 'index_hash': 'index'}
    def tearDown(self): self.temp.cleanup()

    def test_identical_parameters_input_and_unique_results(self):
        original = self.sample.read_bytes()
        run = make_run(self.root)
        sample = snapshot_audio(self.sample, run)
        client = FakeClient()
        params = {'pitch': 12, 'index_rate': .88, 'effects': {'limiter': True}, 'method': 'rmvpe', '_generation': 3}
        frozen = copy.deepcopy(params)
        report = compare(client, sample, run, self.voice, params, ['rmvpe', 'crepe'], Event())
        self.assertEqual(params, frozen)
        self.assertEqual(self.sample.read_bytes(), original)
        self.assertEqual(len(report['results']), 2)
        one, two = client.calls
        for key in one.keys() - {'output', 'method'}: self.assertEqual(one[key], two[key])
        self.assertNotEqual(one['output'], two['output'])
        self.assertEqual(report['status'], 'finished')
        self.assertTrue((run / 'results.json').exists())
        self.assertIn('naturalness', describe_result(report['results'][0], report['parameters']))

    def test_failed_conversion_does_not_poison_next_method(self):
        run = make_run(self.root); sample = snapshot_audio(self.sample, run)
        report = compare(FakeClient(), sample, run, self.voice, {'pitch': 0}, ['missing', 'rmvpe'], Event())
        self.assertEqual([r['status'] for r in report['results']], ['failed', 'complete'])
        self.assertIn('no fallback', report['results'][0]['error'])

    def test_cancel_stops_remaining_and_keeps_completed_results(self):
        cancel = Event(); run = make_run(self.root); sample = snapshot_audio(self.sample, run)
        client = FakeClient(cancel)
        report = compare(client, sample, run, self.voice, {'pitch': 0}, ['rmvpe', 'crepe'], cancel)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(report['status'], 'cancelled')
        self.assertTrue((run / report['results'][0]['path']).exists())
        self.assertEqual(report['input_sha256'], __import__('hashlib').sha256(sample.read_bytes()).hexdigest())

    def test_restart_and_favorite_changes_only_selected_method(self):
        library = VoiceLibrary(self.root / 'library')
        prefs = {'pitch': 12, 'method': 'rmvpe', 'index_rate': .88, 'effects': {'limiter': True}, 'calibration': {'keep': True}}
        voice = dict(self.voice, model='voices/' + 'a' * 32 + '/model.pth', index=None, preferences=prefs)
        atomic_json(library.path, {'schema': 1, 'voices': [voice], 'removed': [], 'last_voice': voice['id']})
        before = copy.deepcopy(library.get(voice['id']))
        save_favorite(library, voice['id'], 'crepe', {'crepe'})
        after = VoiceLibrary(library.root).get(voice['id'])
        expected = copy.deepcopy(before); expected['preferences']['method'] = 'crepe'
        self.assertEqual(after, expected)
        with self.assertRaises(ValueError): save_favorite(library, voice['id'], 'swift', {'crepe'})

    def test_range_favorite_retains_required_options(self):
        library = VoiceLibrary(self.root / 'library')
        voice = dict(self.voice, model='voices/' + 'a' * 32 + '/model.pth', index=None, preferences={'pitch': 0, 'method': 'rmvpe'})
        atomic_json(library.path, {'schema': 1, 'voices': [voice], 'removed': [], 'last_voice': None})
        save_favorite(library, voice['id'], 'rmvpe-range', {'rmvpe-range'}, {'f0_min': 80, 'f0_max': 700})
        self.assertEqual(library.get(voice['id'])['preferences']['method_options']['f0_min'], 80)

    def test_capabilities_exclude_missing_weights_and_non_f0(self):
        self.assertFalse(any(c['available'] for c in capabilities(self.root, self.root, f0=False)))
        self.assertFalse(next(c for c in capabilities(self.root, self.root) if c['key'] == 'fcpe-experimental')['available'])
        self.assertNotIn('swift', [c['key'] for c in capabilities(self.root, self.root)])

    def test_frame_alignment_and_hybrid_preserve_unvoiced_authority(self):
        self.assertEqual(align([220, 0, 440], 6)[2], 0)
        result, details = hybrid(np.array([0., 220., 220.]), np.array([220., 225., 440.]))
        self.assertEqual(result[0], 0)
        self.assertEqual(result[2], 220)
        self.assertEqual(details['agreement_frames'], 1)

    def test_never_overwrite_an_existing_sample(self):
        run = make_run(self.root); snapshot_audio(self.sample, run)
        with self.assertRaises(ValueError): snapshot_audio(self.sample, run)


if __name__ == '__main__': unittest.main()
