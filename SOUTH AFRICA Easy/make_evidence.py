"""Export the located phrase, comparison figure, and machine-readable result."""
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs'
for source, label in [('voice_note.wav','original'),('outputs/cleaned.wav','cleaned'),('outputs/tonal_cleaned.wav','wide')]:
    x, sr = sf.read(ROOT/source)
    sf.write(OUT/f'key_context_{label}.wav', x[425*sr:445*sr], sr, subtype='PCM_16')
    sf.write(OUT/f'key_phrase_{label}.wav', x[int(437.2*sr):int(440.7*sr)], sr, subtype='PCM_16')
    if label == 'cleaned':
        sf.write(OUT/'key_word.wav',x[int(439.6*sr):int(440.6*sr)],sr,subtype='PCM_16')
        sf.write(OUT/'key_phrase_slow.wav',signal.resample_poly(x[int(437.2*sr):int(440.7*sr)],5,4),sr,subtype='PCM_16')

fig, axes = plt.subplots(2,1,figsize=(12,7),sharex=True,sharey=True)
for ax,label,title in zip(axes,['original','cleaned'],['Original stadium recording','Narrow-band interference removed']):
    x,sr=sf.read(OUT/f'key_context_{label}.wav')
    a=x[11*sr:17*sr]
    f,t,z=signal.stft(a,sr,nperseg=1024,noverlap=896)
    db=20*np.log10(np.abs(z)+1e-10)
    ax.pcolormesh(t+436,f,db-db.max(),vmin=-65,vmax=0,shading='auto',cmap='magma')
    ax.set_ylim(0,4500)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    for start,end,label in [(438.65,439.1,'ikhodi'),(439.83,440.3,'ulwazi')]:
        ax.axvspan(start,end,color='cyan',alpha=.16)
        ax.text((start+end)/2,4200,label,ha='center',color='white')
axes[-1].set_xlabel('Time in original recording (seconds; 439.83 s = 7:19.83)')
fig.tight_layout()
fig.savefig(OUT/'key_spectrogram.png',dpi=160)
plt.close(fig)
result = {
    'flag':'SIGMOID_ULWAZI', 'word':'ulwazi',
    'keyword':'ikhodi', 'keyword_time_seconds':[438.65,439.10],
    'key_time_seconds':[439.83,440.30],
    'phrase':'uzosebenzisa ikhodi ulwazi',
    'evidence':'outputs/key_verification.json',
    'status':'Recovered from local audio; not submitted to or validated by the scoring platform',
    'note':'CTC timestamps are approximate. No reference transcript was supplied.'
}
(ROOT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
