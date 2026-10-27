import io
import json
import os
from pathlib import Path
import pickle
import shutil
import stat
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import voice_library as library_module
from voice_library import VoiceLibrary, SelectionRequired, archive_catalog
from safe_checkpoint import SafeCheckpoint, CheckpointError
from global_hotkeys import CaptureKeys
from audio_effects import input_audio, output_audio, correct_pitch
import numpy as np


def emit(value):
    if isinstance(value, str):
        data = value.encode(); return b'X' + struct.pack('<I', len(data)) + data
    if type(value) is int: return b'J' + struct.pack('<i', value)
    if isinstance(value, list): return b'](' + b''.join(emit(item) for item in value) + b'e'
    if isinstance(value, tuple): return b'(' + b''.join(emit(item) for item in value) + b't'
    if isinstance(value, dict): return b'}(' + b''.join(emit(k) + emit(v) for k, v in value.items()) + b'u'
    if isinstance(value, TensorFixture):
        shape = value.shape
        stride = (shape[1], 1)
        persistent = b'(' + emit('storage') + b'ctorch\nFloatStorage\n' + emit(value.key) + emit('cpu') + emit(np.prod(shape).item()) + b'tQ'
        return b'ctorch._utils\n_rebuild_tensor_v2\n(' + persistent + emit(0) + emit(shape) + emit(stride) + b'\x89ccollections\nOrderedDict\n)RtR'
    raise TypeError(value)


class TensorFixture:
    def __init__(self, key, shape): self.key, self.shape = key, shape


def model_bytes(version='v1', legacy=False):
    dims = 256 if version == 'v1' else 768
    weights = {'emb_g.weight': TensorFixture('0', (1, 256)), 'enc_p.emb_phone.weight': TensorFixture('1', (192, dims))}
    config = [1025, 32, 192, 192, 768, 2, 6, 3, 0, '1', [3, 7, 11], [[1, 3, 5]] * 3, [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000]
    payload = b'\x80\x02' + emit({'weight': weights, 'config': config, 'version': version, 'f0': 1}) + b'.'
    storages = {'0': bytes(256 * 4), '1': bytes(192 * dims * 4)}
    if legacy:
        result = pickle.dumps(0x1950A86A20F9469CFC6C, protocol=2) + pickle.dumps(1001, protocol=2) + pickle.dumps({'little_endian': True}, protocol=2) + payload + pickle.dumps(list(storages), protocol=2)
        for key, data in storages.items(): result += struct.pack('<q', len(data) // 4) + data
        return result
    output = io.BytesIO()
    with ZipFile(output, 'w') as archive:
        archive.writestr('archive/data.pkl', payload)
        for key, data in storages.items(): archive.writestr('archive/data/' + key, data)
    return output.getvalue()


def index_bytes(dimensions=256, count=16):
    data = bytearray(40)
    data[:4] = b'IwFl'; data[4:8] = struct.pack('<i', dimensions); data[8:16] = struct.pack('<q', count); data[32] = 1
    return bytes(data)


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(Path(__file__).resolve().parent)
        self.library = VoiceLibrary(self.root / 'library')

    def tearDown(self): self.temp.cleanup()

    def archive(self, entries, name='voice.zip'):
        path = self.root / name
        with ZipFile(path, 'w') as archive:
            for key, value in entries.items(): archive.writestr(key, value)
        return path

    def test_nested_pair_standard_names_source_retained(self):
        source = self.archive({'folder/voice.pth': model_bytes(), 'folder/added_IVF16_Flat_nprobe_1_voice_v1.index': index_bytes(), 'folder/photo.jpg': b'ignored'})
        before = source.read_bytes()
        result = self.library.import_zip(source)
        voice = result['voice']
        self.assertEqual(self.library.path_for(voice, 'model').name, 'model.pth')
        self.assertEqual(self.library.path_for(voice, 'index').name, 'model.index')
        self.assertEqual(voice['original_index_name'], 'added_IVF16_Flat_nprobe_1_voice_v1.index')
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(len(self.library.voices()), 1)
        # Source ZIP can disappear: the imported copies are complete independent files.
        source.unlink()
        self.assertEqual(SafeCheckpoint(self.library.path_for(voice, 'model')).metadata['version'], 'v1')

    def test_model_without_index(self):
        voice = self.library.import_zip(self.archive({'voice.pth': model_bytes()}))['voice']
        self.assertIsNone(voice['index'])
        self.assertEqual(voice['preferences']['index_rate'], 0)

    def test_multiple_models_ambiguity_and_selection(self):
        source = self.archive({'one.pth': model_bytes(), 'two.pth': model_bytes('v2'), 'two.index': index_bytes(768)})
        with self.assertRaises(SelectionRequired) as context: self.library.import_zip(source)
        self.assertEqual(context.exception.kind, 'model')
        voice = self.library.import_zip(source, model_choice='two.pth')['voice']
        self.assertEqual(voice['rvc']['version'], 'v2')

    def test_multiple_indexes_ambiguity(self):
        source = self.archive({'voice.pth': model_bytes(), 'a.index': index_bytes(), 'b.index': index_bytes()})
        with self.assertRaises(SelectionRequired) as context: self.library.import_zip(source)
        self.assertIn(None, context.exception.choices)
        self.assertEqual(len(self.library.voices()), 0)
        self.assertFalse(list(self.library.root.glob('.import-*')))
        voice = self.library.import_zip(source, index_choice='b.index')['voice']
        self.assertEqual(voice['original_index_name'], 'b.index')

    def test_dimension_and_trained_index_checks(self):
        source = self.archive({'voice.pth': model_bytes(), 'voice.index': index_bytes(768), 'trained_IVF.index': index_bytes(256, 0)})
        with self.assertRaises(ValueError): self.library.import_zip(source, index_choice='voice.index')
        voice = self.library.import_zip(source)['voice']
        self.assertIsNone(voice['index'])

    def test_duplicate_hashes_different_archive_names(self):
        one = self.library.import_zip(self.archive({'first.pth': model_bytes()}, 'one.zip'))
        two = self.library.import_zip(self.archive({'renamed.pth': model_bytes()}, 'two.zip'))
        self.assertTrue(two['duplicate'])
        self.assertEqual(one['voice']['id'], two['voice']['id'])

    def test_same_display_name_unique_directories(self):
        one = self.library.import_zip(self.archive({'voice.pth': model_bytes()}, 'one.zip'), name='Same')['voice']
        two = self.library.import_zip(self.archive({'voice.pth': model_bytes('v2')}, 'two.zip'), name='Same')['voice']
        self.assertNotEqual(one['model'], two['model'])

    def test_corrupt_and_unsafe_archives(self):
        bad = self.root / 'bad.zip'; bad.write_bytes(b'not a zip')
        with self.assertRaises(ValueError): self.library.import_zip(bad)
        for name in ['../outside.pth', '/absolute.pth', 'C:/absolute.pth', 'folder/../../escape.pth', 'folder\\..\\escape.pth']:
            with self.subTest(name=name):
                with self.assertRaises(ValueError): self.library.import_zip(self.archive({name: model_bytes()}))
        info = ZipInfo('voice.pth'); info.create_system = 3; info.external_attr = (stat.S_IFLNK | 0o777) << 16
        path = self.root / 'link.zip'
        with ZipFile(path, 'w') as archive: archive.writestr(info, b'elsewhere')
        with self.assertRaises(ValueError): self.library.import_zip(path)

    def test_limits_case_collisions_and_bomb(self):
        source = self.archive({'a.pth': model_bytes(), 'A.pth': model_bytes()})
        with self.assertRaises(ValueError): archive_catalog(source)
        source = self.archive({'a.pth': model_bytes(), 'note': b'123'})
        with patch.object(library_module, 'MAX_ENTRIES', 1):
            with self.assertRaises(ValueError): archive_catalog(source)
        with patch.object(library_module, 'MAX_FILE', 100):
            with self.assertRaises(ValueError): archive_catalog(source)
        bomb = self.root / 'bomb.zip'
        with ZipFile(bomb, 'w', compression=ZIP_DEFLATED) as archive: archive.writestr('voice.pth', bytes(1024 ** 2))
        with self.assertRaises(ValueError): archive_catalog(bomb)

    def test_registry_recovery_persistence_and_relocation(self):
        voice = self.library.import_zip(self.archive({'voice.pth': model_bytes()}))['voice']
        self.library.select(voice['id'])
        self.library.edit(voice['id'], name='Renamed', category='Singer', usage='Both')
        self.library.path.write_text('{corrupted')
        recovered = VoiceLibrary(self.library.root)
        self.assertTrue(recovered.recovered)
        self.assertEqual(recovered.get(voice['id'])['name'], 'Renamed')
        self.assertEqual(recovered.selected(), voice['id'])
        target = self.root / 'moved'; shutil.copytree(recovered.root, target)
        moved = VoiceLibrary(target)
        self.assertTrue(moved.path_for(moved.get(voice['id']), 'model').is_file())

    def test_registry_path_validation(self):
        voice = self.library.import_zip(self.archive({'voice.pth': model_bytes()}))['voice']
        doc = self.library.read(); doc['voices'][0]['model'] = '../original.pth'
        with self.assertRaises(ValueError): self.library.write(doc)

    def test_atomic_failure_never_registers_incomplete_voice(self):
        original = library_module.atomic_json
        def fail_primary(path, value):
            if path.name == 'voices.json': raise OSError('injected disk failure')
            original(path, value)
        with patch.object(library_module, 'atomic_json', side_effect=fail_primary):
            with self.assertRaises(OSError): self.library.import_zip(self.archive({'voice.pth': model_bytes()}))
        self.assertEqual(self.library.voices(), [])
        self.assertFalse(list((self.library.root / 'voices').glob('*')))

    def test_missing_files_relink_and_separate_deletion(self):
        model = self.root / 'original.pth'; model.write_bytes(model_bytes())
        voice = self.library.import_files(model)['voice']
        self.library.path_for(voice, 'model').unlink()
        replacement = self.library.import_files(model, relink_id=voice['id'])['voice']
        self.assertEqual(replacement['id'], voice['id'])
        self.assertTrue(self.library.path_for(replacement, 'model').is_file())
        self.library.remove(voice['id'])
        self.assertTrue(self.library.path_for(replacement, 'model').is_file())
        with self.assertRaises(ValueError): self.library.delete_files(voice['id'])
        self.library.delete_files(voice['id'], confirmed=True)
        self.assertTrue(model.is_file())

    def test_legacy_and_malicious_checkpoint_no_execution(self):
        legacy = self.root / 'legacy.pth'; legacy.write_bytes(model_bytes(legacy=True))
        self.assertEqual(SafeCheckpoint(legacy).metadata['dimensions'], 256)
        marker = self.root / 'must-not-exist'
        class Malicious:
            def __reduce__(self): return os.system, ('echo unsafe > "' + str(marker) + '"',)
        path = self.root / 'unsafe.pth'
        with ZipFile(path, 'w') as archive: archive.writestr('archive/data.pkl', pickle.dumps(Malicious(), protocol=2))
        with self.assertRaises(CheckpointError): SafeCheckpoint(path)
        self.assertFalse(marker.exists())


class HotkeyTests(unittest.TestCase):
    def setUp(self):
        self.events, self.now, self.physical, self.ready = [], 0, True, True
        self.keys = CaptureKeys(lambda: self.events.append('start') or True,
                                lambda reason: self.events.append('stop'), lambda: self.ready,
                                lambda text: None, lambda: self.physical, lambda: self.now)
        self.keys.enable()

    def test_hold_auto_repeat_and_release(self):
        for _ in range(20): self.keys.event('down')
        self.keys.event('up'); self.keys.event('up')
        self.assertEqual(self.events, ['start', 'stop'])

    def test_missed_release_recovery(self):
        self.keys.event('down'); self.physical = False
        self.keys.tick(); self.keys.tick(); self.keys.tick()
        self.assertEqual(self.events, ['start', 'stop'])

    def test_toggle_safety_and_disable(self):
        self.keys.enable(toggle=True)
        self.keys.event('down'); self.keys.event('up')
        self.assertEqual(self.events, ['start'])
        self.keys.event('down'); self.keys.event('up')
        self.assertEqual(self.events, ['start', 'stop'])
        self.keys.event('down'); self.now = 60; self.keys.tick()
        self.assertEqual(self.events, ['start', 'stop', 'start', 'stop'])
        self.keys.disable(); self.keys.event('down')
        self.assertEqual(len(self.events), 4)

    def test_busy_hold_does_not_start_after_ready(self):
        self.ready = False; self.keys.event('down')
        self.ready = True; self.keys.event('down')
        self.assertEqual(self.events, [])
        self.keys.event('up'); self.keys.event('down')
        self.assertEqual(self.events, ['start'])

    def test_timeout_does_not_restart_while_key_remains_held(self):
        self.keys.event('down')
        self.now = 60
        self.keys.tick()
        for _ in range(20): self.keys.event('down')
        self.assertEqual(self.events, ['start', 'stop'])
        self.keys.event('up')
        self.keys.event('down')
        self.assertEqual(self.events, ['start', 'stop', 'start'])


class EffectsTests(unittest.TestCase):
    def test_dry_is_exact_and_correction_bypasses_talking(self):
        audio = np.linspace(-.8, .8, 1000, dtype='float32')
        np.testing.assert_array_equal(input_audio(audio, 48000, {})[0], audio)
        np.testing.assert_array_equal(output_audio(audio, 48000, {})[0], audio)
        pitch = np.array([0, 430, 465], dtype='float32')
        np.testing.assert_array_equal(correct_pitch(pitch, {'pitch_correction': True, 'correction_strength': 1}), pitch)
        corrected = correct_pitch(pitch, {'singing_mode': True, 'pitch_correction': True, 'correction_strength': 1})
        self.assertEqual(corrected[0], 0)
        self.assertAlmostEqual(corrected[1], 440, places=2)

    def test_gain_limiter_reverb_and_clipping(self):
        audio = np.sin(np.arange(4800) * .1).astype('float32')
        self.assertTrue(input_audio(audio, 48000, {})[1])
        result, warnings = output_audio(audio, 48000, {'output_gain_db': 12, 'limiter': True, 'reverb': 'Wrestling arena'})
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(float(abs(result).max()), .98001)
        self.assertGreater(len(result), len(audio))


if __name__ == '__main__': unittest.main(verbosity=2)
