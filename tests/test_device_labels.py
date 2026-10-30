import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pitch_methods import LABELS, EXTRACTION_DEVICES, DEVICE_NOTE
from voice_dialog import METHOD_LABELS, LEGACY_LABELS, method_key, method_label, restored_method
from voice_library import METHODS


class DeviceLabelTests(unittest.TestCase):
    def test_all_surfaces_use_one_label_per_internal_key(self):
        self.assertEqual(set(LABELS), set(METHODS))
        self.assertEqual(METHOD_LABELS, [LABELS[key] for key in METHODS])
        for key in METHODS:
            self.assertEqual(method_key(LABELS[key]), key)
            self.assertEqual(method_key(key), key)
            self.assertEqual(restored_method({'method': key}, 'rmvpe'), key)

    def test_legacy_display_values_are_still_accepted(self):
        for label, key in LEGACY_LABELS.items():
            self.assertEqual(method_key(label), key)
            self.assertEqual(method_label(label), LABELS[key])
            self.assertEqual(restored_method({'method': label}, 'rmvpe'), key)

    def test_cpu_and_cuda_labels_match_installed_routing(self):
        self.assertEqual({key for key, device in EXTRACTION_DEVICES.items() if device == 'cpu'}, {'pm', 'harvest', 'swiftf0'})
        for key, label in LABELS.items():
            suffix = '(CPU only)' if EXTRACTION_DEVICES[key] == 'cpu' else '(GPU only)'
            self.assertTrue(label.endswith(suffix), label)
            self.assertEqual(method_label(key), label)
        self.assertIn('requires NVIDIA CUDA', DEVICE_NOTE)
        self.assertIn('CPU-only conversion is not supported', DEVICE_NOTE)


if __name__ == '__main__': unittest.main()
