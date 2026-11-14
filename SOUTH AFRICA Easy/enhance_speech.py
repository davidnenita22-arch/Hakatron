"""Optional DeepFilterNet3 pass after deterministic tonal filtering."""
from pathlib import Path
import argparse
import sys
import types
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

# DF 0.5.6 imports this removed torchaudio class only for annotations.
# Actual file I/O below uses soundfile; no torchaudio backend is needed.
try:
    from torchaudio.backend.common import AudioMetaData
except ModuleNotFoundError:
    common = types.ModuleType('torchaudio.backend.common')
    common.AudioMetaData = object
    sys.modules['torchaudio.backend.common'] = common
from df.enhance import init_df, enhance

ROOT = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio', type=Path)
    ap.add_argument('--seconds', type=float)
    ap.add_argument('--output', type=Path, default=ROOT/'outputs'/'enhanced.wav')
    args = ap.parse_args()
    torch.set_num_threads(4)
    model, state, _ = init_df('DeepFilterNet3', log_file=None)
    model = model.cpu()
    x, sr = sf.read(args.audio, dtype='float32')
    if args.seconds:
        x = x[:int(sr*args.seconds)]
    pieces = []
    for start in range(0, len(x), sr*30):
        a = resample_poly(x[start:start+sr*30], 3, 1).astype(np.float32)
        with torch.inference_mode():
            b = enhance(model, state, torch.from_numpy(a[None]), pad=True)
        pieces.append(resample_poly(b.numpy()[0], 1, 3))
        print('enhanced through', min((start/sr)+30,len(x)/sr), flush=True)
    y = np.concatenate(pieces)[:len(x)]
    sf.write(args.output, y, sr, subtype='PCM_16')

if __name__ == '__main__':
    main()
