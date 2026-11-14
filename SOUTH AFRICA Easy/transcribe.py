"""Timestamped local isiZulu CTC transcription; never infer a flag from noise."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForCTC

ROOT = Path(__file__).resolve().parent
MODEL = 'badrex/w2v-bert-2.0-zulu-asr'
REVISION = '9855bac28489def178407fb385710e8cc519a87f'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio', type=Path)
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--end', type=float)
    ap.add_argument('--chunk', type=float, default=20)
    ap.add_argument('--overlap', type=float, default=3)
    ap.add_argument('--output', type=Path, default=ROOT/'outputs'/'transcript.json')
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('Loading', MODEL, 'on', device, flush=True)
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(MODEL, revision=REVISION, local_files_only=True).eval().to(device)
    x, sr = sf.read(args.audio, dtype='float32')
    assert sr == 16000 and x.ndim == 1
    end = min(len(x)/sr, args.end or len(x)/sr)
    results = []
    for start in np.arange(args.start, end, args.chunk-args.overlap):
        stop = min(start+args.chunk, end)
        part = x[int(start*sr):int(stop*sr)]
        inputs = processor(part, sampling_rate=sr, return_tensors='pt')
        inputs = {k:v.to(device) for k,v in inputs.items()}
        before = time.time()
        with torch.inference_mode():
            logits = model(**inputs).logits
        ids = logits.argmax(-1).cpu()
        decoded = processor.batch_decode(ids)[0]
        result = dict(start=round(float(start),3), end=round(float(stop),3), text=decoded)
        results.append(result)
        print(f'{start:7.2f}-{stop:7.2f} ({time.time()-before:.1f}s): {decoded}', flush=True)
        args.output.write_text(json.dumps({'model':MODEL,'audio':str(args.audio),'segments':results},ensure_ascii=False,indent=2)+'\n')
        if stop >= end:
            break

if __name__ == '__main__':
    main()
