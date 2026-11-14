"""Causal, bounded-memory anomaly detection for bunker sensor streams.

Only the current reading and per-zone history are used.  No files, labels,
network resources, or timestamps from the training set are consulted.
"""
from collections import deque
import math

import numpy as np


_FIELDS = (
    "temperature_c", "humidity_pct", "vibration_g", "acoustic_db",
    "air_pressure_kpa", "co2_ppm",
)
# Resolution/noise floors keep a quiet commissioning period from producing an
# unrealistically sensitive model.  Location and most of the scale are learned.
_FLOORS = np.array([0.30, 0.40, 0.007, 0.25, 0.040, 7.0])


class _Zone:
    def __init__(self):
        self.history = [deque(maxlen=256) for _ in _FIELDS]
        self.center = np.zeros(6)
        self.scale = _FLOORS.copy()
        self.smoothed = np.zeros(6)
        self.count = 0

    def refresh(self):
        for j, values in enumerate(self.history):
            if values:
                a = np.asarray(values, dtype=float)
                m = float(np.median(a))
                self.center[j] = m
                self.scale[j] = max(_FLOORS[j], 1.4826 * float(np.median(np.abs(a - m))))


class Detector:
    """One instance consumes one ordered stream; zones have independent state."""

    def __init__(self):
        self._zones = {}

    def update(self, reading: dict) -> float:
        zone = str(reading.get("bunker_zone", "unknown"))
        state = self._zones.get(zone)
        if state is None:
            state = self._zones[zone] = _Zone()
        values = []
        valid = []
        for name in _FIELDS:
            try:
                value = float(reading.get(name, float("nan")))
            except (TypeError, ValueError, OverflowError):
                value = float("nan")
            valid.append(math.isfinite(value))
            values.append(value)
        mask = np.asarray(valid, dtype=bool)
        if not mask.any():
            return 0.0
        x = np.asarray(values)
        if state.count == 0:
            state.center[mask] = x[mask]
        if state.count < 24 or state.count % 8 == 0:
            state.refresh()
        z = np.zeros(6)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            z[mask] = (x[mask] - state.center[mask]) / state.scale[mask]
        z = np.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)
        z = np.clip(z, -1e6, 1e6)
        # Signed smoothing retains a persistent shift and cancels random noise.
        # Clip spikes so a single event does not create several follow-up alerts.
        state.smoothed = 0.75 * state.smoothed + 0.25 * np.clip(z, -6.0, 6.0)
        abs_z = np.abs(z)
        # Require current evidence in the same direction: a recovered sensor
        # must not keep alarming only because its moving average decays slowly.
        agreement = (z * state.smoothed > 0) & (abs_z >= 1.5)
        slow = np.where(agreement, np.abs(state.smoothed), 0.0)
        point = float(np.max(abs_z)) / 4.5
        collective = float(np.linalg.norm(np.sort(abs_z)[-3:])) / 6.3
        drift = float(np.max(slow)) / 3.7
        persistent = float(np.sort(slow)[-2]) / 2.0
        strength = max(point, collective, drift, persistent)
        # The result is deliberately a detection score rather than a calibrated
        # probability.  Strength 1 is exactly the evaluator's 0.5 boundary.
        score = float(1.0 - 1.0 / (1.0 + strength))
        # Robust, feature-wise admission prevents one bad channel from freezing
        # learning of all the other channels.  Alarmed levels cannot normalize
        # themselves away simply by lasting for a long time.
        for j in range(6):
            if mask[j] and (state.count < 24 or abs_z[j] < 3.0):
                state.history[j].append(float(x[j]))
        state.count += 1
        if state.count < 24:
            return 0.0
        return min(1.0, max(0.0, score))
