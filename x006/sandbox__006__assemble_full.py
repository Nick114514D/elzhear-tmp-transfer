"""
006 "Requiem" (Elzhear de Nohant) -- full-piece assembly
========================================================
Concatenates the four LOCKED movements into the complete continuous piece with
custom inter-movement transitions. This is an AUDIO-DOMAIN assembly on the
rendered movement masters; the Mvt IV master is itself the audio-domain
recomposition committed as mvt4_LOCKED.py (its r5 source synthesis was lost).

Audio is never committed to the repo -- this script is the record of the
assembly. Inputs (movement masters, kept locally, NOT in repo):
    m1.wav = 006_mvt1_v2.wav        Mvt I                (~3:18, quietest; RMS ~-25.8)
    m2.wav = 006_section2_full.wav  Mvt II               (~2:47;          RMS ~-22.2)
    m3.wav = 006_mvt3_LOCKED.wav    Mvt III "Dies irae"  (~4:38;          RMS ~-21.6)
    m4.wav = 006_mvt4_LOCKED.wav    Mvt IV  "Lux aeterna"(~3:26, loudest; RMS ~-18.1)
    all 48 kHz / stereo / PCM-24.
Output: 006_Requiem_full.wav -- 48 kHz / 24-bit stereo, ~14:07, peak ~-0.80 dBFS.

DESIGN
------
- Per-movement mastered levels are PRESERVED (no loudness flattening). The
  quiet-I -> present-IV dynamic arc across the whole piece is intentional.
- Each movement's leading/trailing silence is trimmed (below -55 dB, 0.15 s pad),
  its own fades left intact.

Inter-movement transitions
- Dead digital silence between movements reads as a "vacuum": Mvt I decays to its
  own ~-58 dB room floor, and an absolute-zero gap then drops the ever-present air
  to nothing. Fix: a low tape-room FLOOR BED (floor_bed) -- decorrelated stereo
  band-limited hiss (90-9000 Hz) plus a touch of 40-300 Hz room tone, at ~-56 dB
  RMS, matching the movements' own fade-tail floor -- bridges each inner breath so
  the air never dies.

- I -> II : the floor holds after Mvt I; Mvt II is eased in via a two-stage gain
  envelope (two_stage) -- a quiet INTRO PLATEAU (~-47 dB, ~29 dB under its natural
  level) that CO-SOUNDS with the floor as two audible layers; then, once the floor
  has faded out (~3:25.5), the harmony SWELLS up to full, arriving at 3:28. The
  order is deliberate: breath + intro co-sound -> breath fades -> harmony rises.

- II -> III : short (~3 s) floored breath, same floor bed bridging the gap.

- III -> IV : ATTACCA. 4 s equal-power crossfade; Mvt III's low C#2 residue thins
  to a thread (never silence) and Mvt IV's C#3 blooms out of it -- the residue
  reborn as light. No silent gap.

NOTES
- Absolute times (S2=200.0; floor hold-to 203.5, out-by 205.5; harmony full 208.0)
  are tuned to the current LOCKED movement durations and must be re-derived if any
  movement master changes.
- RNG seed is fixed (77) so the floor bed is bit-reproducible.
"""
import soundfile as sf, numpy as np
from scipy.signal import butter, sosfilt
SR=48000; rng=np.random.default_rng(77)

def load(f):
    y,sr=sf.read(f); assert sr==SR; return y if y.ndim>1 else np.stack([y,y],1)
def trim(y,thr=-55,pad=0.15):
    m=y.mean(1); w=int(0.02*SR)
    e=np.array([20*np.log10(np.sqrt(np.mean(m[i:i+w]**2))+1e-12) for i in range(0,len(m)-w,w)])
    idx=np.where(e>thr)[0]; a=max(0,int(idx[0]*w-pad*SR)); b=min(len(y),int((idx[-1]+1)*w+pad*SR)); return y[a:b]
def xfade(a,b,sec):
    n=int(sec*SR); n=min(n,len(a),len(b))
    fo=np.sqrt(np.linspace(1,0,n))[:,None]; fi=np.sqrt(np.linspace(0,1,n))[:,None]
    return np.concatenate([a[:-n],a[-n:]*fo+b[:n]*fi,b[n:]],0)
def bp(y,lo,hi,o=2): return sosfilt(butter(o,[lo,min(hi,SR/2-100)],btype="band",fs=SR,output="sos"),y)
def floor_bed(n,rms_db=-56.0):
    x=np.stack([bp(rng.standard_normal(n),90,9000),bp(rng.standard_normal(n),90,9000)],1)
    x+=np.stack([bp(rng.standard_normal(n),40,300),bp(rng.standard_normal(n),40,300)],1)*0.6
    cur=20*np.log10(np.sqrt(np.mean(x**2))+1e-12); return x*10**((rms_db-cur)/20)
def seg_bed(buf,t_a,t_ramp,t_hold_end,t_out_end,rms_db=-56.0):
    a=int(t_a*SR); b=int(t_out_end*SR); n=b-a; seg=floor_bed(n,rms_db); env=np.zeros(n)
    i1=int((t_ramp-t_a)*SR); i2=int((t_hold_end-t_a)*SR)
    env[:i1]=np.linspace(0,1,i1); env[i1:i2]=1.0; env[i2:]=np.linspace(1,0,n-i2)
    buf[a:b]+=seg*env[:,None]; return buf
def rc(n): return 0.5-0.5*np.cos(np.linspace(0,np.pi,n))   # raised-cosine 0->1
def two_stage(y,fin=1.5,pl_end=5.5,sw_end=8.0,g_low=0.035):
    n=len(y); env=np.ones(n); i0=int(fin*SR); i1=int(pl_end*SR); i2=int(sw_end*SR)
    env[:i0]=g_low*rc(i0); env[i0:i1]=g_low; env[i1:i2]=g_low+(1-g_low)*rc(i2-i1); env[i2:]=1.0
    return y*env[:,None]

m1,m2,m3,m4=[trim(load(f)) for f in ("m1.wav","m2.wav","m3.wav","m4.wav")]
L1=len(m1)/SR
S2=200.0; G1=S2-L1                                                # II emerges at 3:20 under the floor
m2e=two_stage(m2, fin=1.5, pl_end=5.5, sw_end=8.0, g_low=0.035)   # quiet intro plateau -> swell; harmony full 208 (3:28)
G2=3.0
head=np.concatenate([m1,np.zeros((int(G1*SR),2)),m2e,np.zeros((int(G2*SR),2)),m3],0)
head=seg_bed(head, L1-2.5, L1-0.5, 203.5, 205.5)                  # I->II floor: hold under I fade + II quiet intro, fade OUT before the swell
s3=L1+G1+len(m2e)/SR
head=seg_bed(head, s3-2.5, s3-0.5, s3+G2, s3+G2+2.0)             # II->III short floored breath
whole=xfade(head,m4,4.0)                                          # III->IV attacca

def env(y,a,b):
    m=y.mean(1); return [(t,20*np.log10(np.sqrt(np.mean(m[int(t*SR):int((t+0.5)*SR)]**2))+1e-12)) for t in np.arange(a,b,0.5)]
print("I->II: floor(~-59) + II quiet intro plateau(~-47) co-sound; floor gone ~205.5; harmony swells 205.5->208:")
for t,d in env(whole,197,209): print(f"  {int(t//60)}:{t%60:05.2f}  {d:6.1f}")
print(f"\ntotal {len(whole)/SR/60:.2f} min  peak {20*np.log10(np.max(np.abs(whole))):.2f} dBFS")
sf.write("006_Requiem_full.wav",whole,SR,subtype="PCM_24")
print("wrote 006_Requiem_full.wav")
