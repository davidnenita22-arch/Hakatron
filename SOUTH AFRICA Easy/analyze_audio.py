"""Inspect and suppress persistent narrow-band stadium interference."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import soundfile as sf
from scipy import signal, ndimage

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--wide', action='store_true', help='Reproduce the initial wider notch comparison')
args = ap.parse_args()
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
x, sr = sf.read(ROOT / 'voice_note.wav', dtype='float32')
nfft = 2048 if args.wide else 8192
overlap = 3*nfft//4
f, t, z = signal.stft(x, sr, nperseg=nfft, noverlap=overlap)
p = np.abs(z)**2
profile = np.mean(p, axis=1)
baseline = ndimage.median_filter(profile, size=31 if args.wide else 61)
# Persistent tonal energy lies far above the local spectral floor.
tonal = profile > baseline * 4
tonal = ndimage.maximum_filter1d(tonal, size=5)
gain = np.ones(len(f), dtype=np.float32)
gain[tonal] = 0.015
gain[f < 100] = 0
gain[f > (5000 if args.wide else 4500)] = 0.15 if args.wide else 0.01
_, y = signal.istft(z * gain[:, None], sr, nperseg=nfft, noverlap=overlap)
y = y[:len(x)]
normalization = float(0.9 / max(np.max(np.abs(y)), 1e-8))
y *= normalization
name = 'tonal_cleaned' if args.wide else 'cleaned'
sf.write(OUT / f'{name}.wav', y, sr, subtype='PCM_16')
metadata = dict(input_sha256=hashlib.sha256((ROOT/'voice_note.wav').read_bytes()).hexdigest(),
                sample_rate=sr, duration_seconds=len(x)/sr, fft_size=nfft,
                overlap_samples=overlap, removed_frequency_bins=int(tonal.sum()),
                tonal_gain=0.015, normalization_gain=normalization)
(OUT/f'{name}_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('removed frequency bins:', int(tonal.sum()), flush=True)
print('wrote', OUT / f'{name}.wav', flush=True)
