"""Contract and synthetic streaming regression checks (no training labels)."""
import math
import unittest
from copy import deepcopy

import numpy as np

from detector import Detector


FIELDS = ('temperature_c', 'humidity_pct', 'vibration_g', 'acoustic_db', 'air_pressure_kpa', 'co2_ppm')


def stream(n=240, seed=17, zone='test'):
    rng = np.random.default_rng(seed)
    center = np.array([13., 72., .12, 32., 101.5, 510.])
    scale = np.array([.2, .5, .01, .35, .04, 10.])
    return [dict(zip(FIELDS, values), timestamp=i, bunker_zone=zone)
            for i, values in enumerate(center + rng.normal(size=(n, 6)) * scale)]


class DetectorTests(unittest.TestCase):
    def test_deterministic_prefix_and_input_unchanged(self):
        rows = stream()
        original = deepcopy(rows)
        a, b = Detector(), Detector()
        self.assertEqual([a.update(r) for r in rows[:100]], [b.update(r) for r in rows[:100]])
        self.assertEqual(rows, original)

    def test_finite_bounded_scores_for_invalid_and_extreme_values(self):
        detector = Detector()
        for row in stream() + [{}, {'humidity_pct': 'bad'}, {'co2_ppm': float('nan')},
                                {'co2_ppm': float('inf')}, {'co2_ppm': 1e308},
                                {'co2_ppm': -1e308}]:
            value = detector.update(dict(row, bunker_zone='test'))
            self.assertIsInstance(value, float)
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.)
            self.assertLessEqual(value, 1.)
        self.assertGreaterEqual(detector.update({'bunker_zone': 'test', 'co2_ppm': 1e308}), .5)

    def test_zones_do_not_change_each_others_state(self):
        rows_a, rows_b = stream(zone='a'), stream(seed=29, zone='b')
        isolated, interleaved = Detector(), Detector()
        expected = [isolated.update(row) for row in rows_a]
        actual = []
        for a, b in zip(rows_a, rows_b):
            interleaved.update(b)
            actual.append(interleaved.update(a))
        self.assertEqual(expected, actual)

    def test_point_drift_and_persistent_multisensor_change(self):
        base = stream()
        detector = Detector()
        for row in base:
            detector.update(row)
        point = dict(base[-1], vibration_g=.35)
        self.assertGreaterEqual(detector.update(point), .5)
        detector = Detector()
        for row in base:
            detector.update(row)
        scores = [detector.update(dict(row, humidity_pct=row['humidity_pct'] + .08 * i))
                  for i, row in enumerate(stream(100, seed=23))]
        self.assertGreater(sum(s >= .5 for s in scores[-40:]), 30)
        detector = Detector()
        for row in base:
            detector.update(row)
        scores = [detector.update(dict(row, vibration_g=row['vibration_g'] + .025,
                                       acoustic_db=row['acoustic_db'] + .9,
                                       co2_ppm=row['co2_ppm'] + 25))
                  for row in stream(40, seed=29)]
        self.assertGreater(sum(s >= .5 for s in scores[-20:]), 10)


if __name__ == '__main__':
    unittest.main()
