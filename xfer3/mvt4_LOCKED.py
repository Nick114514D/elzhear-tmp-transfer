# ============================================================
# 006 REQUIEM - Movement IV "Lux Aeterna"  [LOCKED 2026-07-03]
#
# The original synthesis script for Mvt IV r5 was never committed and was lost
# with its sandbox. This locked file therefore works in the AUDIO DOMAIN on the
# r5 render, which is the surviving master.
#
# Inputs (audio, never committed to repo):
#   r5.wav    = 006_mvt4_FINAL_r5.wav   (r5 master, 3:26, 48k stereo)
#   op135.wav = Beethoven Op.135 III Lento assai (Musopen), decoded from mp3
#
# Chain:
#   1. Whistle (0:03-0:15): C# residue tone at 0:07 - define fundamental
#      (+1.5 @138 Q2.5), HPF sub-mush (102), cut reed 3rd harmonic (-4 @450)
#      + nasal (-2 @900) for the shrill, push distant (room rt3.0s 26% wet,
#      dry x0.85).
#   2. String (1:16-1:27): major-string phrase at 1:21 - cut boxy 300-600 weight
#      (-3 @450 Q1.4) for the laggard heaviness, shelve edge (-3.5 @3k, -4 @6.5k)
#      for the shrill.
#   3. Opening blooms (0:16, 0:33): two faint focused-grain blooms of the Op.135
#      theme, pitched to D (+1), -25 dBFS - the "first section" sample.
#   4. Seam glue (-> 2:01.5): Op.135 theme's final cadential phrase (0:51.5-0:59),
#      aged, -21 dBFS, landing where r5 lifts into the returning theme.
#   Transfiguration (III) and outro carry no sample.
#
# Output: 006_mvt4_LOCKED.wav  (48k/24-bit, peak -2.0 dBFS, 3:26)
# ============================================================

import numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve, resample_poly, lfilter
SR=48000; rng=np.random.default_rng(4400)
def norm(y,p=0.9): return y/(np.max(np.abs(y)) or 1)*p
def rms_db(y): 
    s=y.mean(1) if y.ndim>1 else y; return 20*np.log10(np.sqrt(np.mean(s**2)+1e-24)+1e-12)
def set_rms(y,tgt): return y*10**((tgt-rms_db(y))/20)
def samples(d): return int(round(d*SR))
def hp1(y,lo,o=2): return sosfilt(butter(o,max(20,lo),btype="high",fs=SR,output="sos"),y)
def bp(y,lo,hi,o=2):
    hi=min(hi,SR/2-100); lo=max(20,lo); return sosfilt(butter(o,[lo,hi],btype="band",fs=SR,output="sos"),y)
def synth_ir(rt,seed=1):
    r=np.random.default_rng(seed); n=int(rt*SR); t=np.arange(n)/SR
    ir=r.standard_normal(n)*np.exp(-t/(rt*0.32)); ir[0]=1.0; return ir/np.max(np.abs(ir))
# --- RBJ biquads ---
def peaking(fc,g,Q):
    A=10**(g/40); w=2*np.pi*fc/SR; al=np.sin(w)/(2*Q); c=np.cos(w)
    b=[1+al*A,-2*c,1-al*A]; a=[1+al/A,-2*c,1-al/A]; return [x/a[0] for x in b],[1,a[1]/a[0],a[2]/a[0]]
def hishelf(fc,g):
    A=10**(g/40); w=2*np.pi*fc/SR; c=np.cos(w); s=np.sin(w); al=s/2*np.sqrt((A+1/A)+2)
    b=[A*((A+1)+(A-1)*c+2*np.sqrt(A)*al),-2*A*((A-1)+(A+1)*c),A*((A+1)+(A-1)*c-2*np.sqrt(A)*al)]
    a=[(A+1)-(A-1)*c+2*np.sqrt(A)*al,2*((A-1)-(A+1)*c),(A+1)-(A-1)*c-2*np.sqrt(A)*al]
    return [x/a[0] for x in b],[1,a[1]/a[0],a[2]/a[0]]
def apply_bq(y,chain):
    out=y.copy()
    for ch in range(out.shape[1]):
        v=out[:,ch]
        for b,a in chain: v=lfilter(b,a,v)
        out[:,ch]=v
    return out
def win(n,t0,t1,ramp=0.5):
    e=np.zeros(n); i0,i1,r=int(t0*SR),int(t1*SR),int(ramp*SR)
    e[i0:i1]=1.0; e[i0:i0+r]=np.linspace(0,1,r); e[i1-r:i1]=np.linspace(1,0,r); return e[:,None]

r5,_=sf.read('r5.wav'); n=len(r5)

# ===== whistle fix (0:03-0:15): concrete + distant, no boom(dreary), no reed(shrill) =====
wproc=apply_bq(r5,[peaking(138,1.5,2.5),peaking(450,-4.0,1.2),peaking(900,-2.0,1.4)]); wproc=np.stack([hp1(wproc[:,0],102),hp1(wproc[:,1],102)],1)
IRw=synth_ir(3.0,5); wet=np.stack([fftconvolve(wproc[:,0],IRw)[:n],fftconvolve(wproc[:,1],IRw)[:n]],1)
wet=wet/(np.max(np.abs(wet)) or 1)*np.max(np.abs(wproc))
ew=win(n,3.0,15.0,0.6)
r5f=r5*(1-ew) + (wproc*0.85 + wet*0.26)*ew          # duck dry ~2 dB, add distant room

# ===== string fix (1:16-1:27): de-box (laggard) + de-shrill =====
sproc=apply_bq(r5f,[peaking(450,-3.0,1.4), hishelf(3000,-3.5), hishelf(6500,-4.0)])
es=win(n,76.0,87.0,0.6)
r5f=r5f*(1-es) + sproc*es

# ---- verify the two windows ----
def bands(y,a,b,tag):
    s=(y.mean(1) if y.ndim>1 else y)[int(a*SR):int(b*SR)]*np.hanning(int((b-a)*SR))
    S=np.abs(np.fft.rfft(s)); f=np.fft.rfftfreq(len(s),1/SR); P=S**2; tot=P.sum()+1e-12
    g=lambda lo,hi:P[(f>=lo)&(f<hi)].sum()/tot*100
    print(f"  {tag:16s} <150:{g(0,150):4.1f} 300-600:{g(300,600):4.1f} 6k-11k:{g(6000,11000):5.2f}  RMS {rms_db(y[int(a*SR):int(b*SR)]):.1f}")
print("whistle 0:05-0:12  before -> after:")
bands(r5,5,12,"before"); bands(r5f,5,12,"after")
print("string 1:18-1:25  before -> after:")
bands(r5,78,85,"before"); bands(r5f,78,85,"after")

# ===== rebuild the approved v8 additions on the fixed r5 =====
def room(y,mix,ir): wet=fftconvolve(y,ir)[:len(y)]; return (1-mix)*y+mix*norm(wet,np.max(np.abs(y)) or 1)
IRs=synth_ir(1.4,7)
def tape_warp(y,wow=0.005,fl=0.0015):
    N=len(y); t=np.arange(N)/SR; d=1+wow*np.sin(2*np.pi*0.4*t+rng.uniform(0,6))+fl*np.sin(2*np.pi*6*t+rng.uniform(0,6))
    return np.interp(np.clip(np.cumsum(d),0,N-1),np.arange(N),y)
def bloom(src,dur,pos,pitch,seed):
    N=samples(dur); out=np.zeros(N+SR); s=np.asarray(src,float); ls=len(s); r=np.random.default_rng(seed)
    gl=samples(0.22); w=np.hanning(gl)
    for _ in range(int(14*dur)):
        op=int(r.integers(0,N)); c=min(max(pos+r.uniform(-0.05,0.05),0),1); sp=int(c*max(0,ls-gl))
        rt=2**((pitch+r.uniform(-0.05,0.05))/12); rl=int(gl*rt)
        gr=(s[sp:sp+gl] if(rl<2 or sp+rl>=ls) else np.interp(sp+np.arange(gl)*rt,np.arange(ls),s))
        if len(gr)<gl: gr=np.pad(gr,(0,gl-len(gr)))
        out[op:op+gl]+=gr[:gl]*w*(1-0.3*r.uniform(0,1))
    g=out[:N]/(np.max(np.abs(out[:N])) or 1); t=np.linspace(0,1,len(g)); return g*(np.sin(np.pi*t)**2.0)
x,_=sf.read('op135.wav'); x=x.mean(1) if x.ndim>1 else x; x=resample_poly(x.astype(float),48000,44100)
theme=hp1(x[int(7*SR):int(62*SR)],40)
mix=r5f.astype(np.float64).copy()
lay=np.zeros(n)
for on,dur,pos,pit,rr,sd in [(16.0,2.6,0.03,1,-25,1),(33.0,2.4,0.05,1,-25,2)]:
    g=set_rms(bloom(theme,dur,pos,pit,sd),rr); i=int(on*SR); lay[i:i+len(g)]+=g[:n-i]
lay=bp(np.tanh(lay*1.03),170,6500); lay=room(lay,0.06,IRs)
mix[:n]+=np.stack([lay,np.roll(lay,int(0.008*SR))],1)
run=hp1(x[int(51.5*SR):int(59.0*SR)],40); g=tape_warp(run); g=bp(np.tanh(g*1.03),150,6800); g=room(g,0.08,IRs); g=set_rms(g,-21.0)
fi,fo=int(1.2*SR),int(1.0*SR); g[:fi]*=np.linspace(0,1,fi); g[-fo:]*=np.linspace(1,0,fo)
gS=np.stack([g,np.roll(g,int(0.009*SR))],1); end=int(121.5*SR); st=end-len(gS); mix[st:st+len(gS)]+=gS
mix=mix/(np.max(np.abs(mix)) or 1)*10**(-2.0/20)
print(f"\ntotal {len(mix)/SR:.1f}s  peak {20*np.log10(np.max(np.abs(mix))):.1f}")
sf.write("/mnt/user-data/outputs/006_requiem_voice/006_mvt4_LOCKED.wav",mix,SR,subtype="PCM_24")
print("wrote LOCKED")
