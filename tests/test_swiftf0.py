from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from swiftf0_adapter import align_result, available
from voice_dialog import method_key, method_label


class SwiftTests(unittest.TestCase):
    def contour(self, pitch=(100, 100, 100, 100), confidence=(1, 1, 1, 1)):
        return SimpleNamespace(timestamps=np.arange(len(pitch)) * .016, pitch_hz=np.array(pitch), confidence=np.array(confidence))

    def test_shift_and_rvc_grid(self):
        for shift, value in ((0, 100), (-12, 50), (12, 200)):
            coarse, continuous = align_result(self.contour(), 6, shift)
            np.testing.assert_allclose(continuous, value)
            self.assertEqual(len(coarse), 6)
            self.assertTrue(((coarse >= 1) & (coarse <= 255)).all())

    def test_no_stretching_or_bridging_unvoiced(self):
        _, values = align_result(self.contour(confidence=(1, 0, 0, 1)), 10)
        self.assertEqual(values[0], 100)
        self.assertTrue((values[1:4] == 0).all())
        self.assertTrue((values[7:] == 0).all())

    def test_confidence_threshold_and_invalid_values(self):
        _, values = align_result(self.contour(pitch=(100, np.nan, 100, 2000), confidence=(.5, 1, np.nan, 1)), 6)
        self.assertEqual(values[0], 100)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[2], 0)
        self.assertEqual(values[5], 0)

    def test_invalid_timestamps_fail(self):
        contour = self.contour(); contour.timestamps[:] = 0
        with self.assertRaises(ValueError): align_result(contour, 6)

    def test_last_frame_is_held_without_extrapolation(self):
        _, values = align_result(self.contour(pitch=(100, 120, 160, 200)), 6)
        self.assertAlmostEqual(values[-1], 200)

    def test_empty_contour_fails(self):
        with self.assertRaises(ValueError): align_result(self.contour((), ()), 6)

    def test_missing_bundle_unavailable(self):
        self.assertFalse(available(Path(__file__).parent / 'no-swift-bundle'))

    def test_label_roundtrip(self):
        self.assertEqual(method_key('SwiftF0 0.3.0 (CPU)'), 'swiftf0')
        self.assertEqual(method_label('swiftf0'), 'SwiftF0 0.3.0 (CPU only)')


if __name__ == '__main__': unittest.main()
