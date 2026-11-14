"""Compare independent signal treatments around the detected keyword."""
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy import signal
import torch
from transformers import AutoProcessor, AutoModelForCTC

ROOT = Path(__file__).resolve().parent
MODEL = 'badrex/w2v-bert-2.0-zulu-asr'
REVISION = '9855bac28489def178407fb385710e8cc519a87f'
torch.set_num_threads(4)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
model = AutoModelForCTC.from_pretrained(MODEL, revision=REVISION, local_files_only=True).eval().to(device)
results = []
for name, offset in [('voice_note.wav',0),('outputs/cleaned.wav',0),('outputs/key_context_wide.wav',425),('outputs/key_context_enhanced.wav',425)]:
    x, sr = sf.read(ROOT/name, dtype='float32')
    for start,end in [(429,441),(436,442),(437,441),(438.3,440.8)]:
        a=x[int((start-offset)*sr):int((end-offset)*sr)]
        inputs=processor(a,sampling_rate=sr,return_tensors='pt')
        with torch.inference_mode():
            logits=model(**{k:v.to(device) for k,v in inputs.items()}).logits
        ids=logits.argmax(-1).cpu()[0]
        decoded=processor.tokenizer.decode(ids,output_word_offsets=True)
        step=(end-start)/len(ids)
        words=[dict(word=w['word'],start=round(start+w['start_offset']*step,3),end=round(start+w['end_offset']*step,3)) for w in decoded.word_offsets]
        result=dict(audio=name,start=start,end=end,text=decoded.text,words=words)
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
(ROOT/'outputs'/'key_verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
