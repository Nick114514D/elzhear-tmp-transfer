import numpy as np, soundfile as sf, librosa, re
from scipy.signal import butter, sosfilt, lfilter, fftconvolve
SR=48000; rng=np.random.default_rng(4100)
def lp(y,hi,o=2): return sosfilt(butter(o,min(hi,SR/2-100),btype="low",fs=SR,output="sos"),y)
def hp(y,lo,o=2): return sosfilt(butter(o,lo,btype="high",fs=SR,output="sos"),y)
def bp(y,lo,hi,o=2): hi=min(hi,SR/2-100); lo=max(20,lo); return sosfilt(butter(o,[lo,hi],btype="band",fs=SR,output="sos"),y)
def norm(y,p=0.9): return y/(np.max(np.abs(y)) or 1)*p
def declick(y,i=3,o=8):
    a=int(i/1000*SR); b=int(o/1000*SR); a=min(a,len(y)//2); b=min(b,len(y)//2); y=y.copy()
    if a>1: y[:a]*=0.5-0.5*np.cos(np.linspace(0,np.pi,a))
    if b>1: y[-b:]*=0.5+0.5*np.cos(np.linspace(0,np.pi,b))
    return y
def synth_ir(rt,seed=1):
    n=int(rt*1.1*SR); t=np.arange(n)/SR; return np.random.default_rng(seed).standard_normal(n)*np.exp(-6.9*t/rt)
_IR={}
def room(y,rt=0.9,mix=0.06,seed=1):
    key=(round(rt,1),seed%6)
    if key not in _IR: _IR[key]=synth_ir(rt,seed)
    yy=declick(y,3,10); wet=fftconvolve(yy,_IR[key])
    out=np.zeros(len(wet)); out[:len(yy)]+=(1-mix)*yy; out+=mix*wet/(np.max(np.abs(wet)) or 1); return out
def place(bus,sig,at):
    sig=declick(sig,3,6); i=int(at*SR)
    if i<0: sig=sig[-i:]; i=0
    e=min(len(bus),i+len(sig)); bus[i:e]+=sig[:max(0,e-i)]
NM={'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11,'Bb':10,'Eb':3}
def f(n): m=re.match(r'([A-G][b#]?)(-?\d)',n); return 440*2**((12*(int(m.group(2))+1)+NM[m.group(1)]-69)/12)
def saw(freq,dur):
    n=int(dur*SR); t=np.arange(n)/SR; y=np.zeros(n); K=min(24,int((SR/2)/max(freq,1)))
    for k in range(1,K+1): y+=(1/k)*np.sin(2*np.pi*k*freq*t)
    return y
def strike_metal(freq,dur,amp):
    n=int(dur*SR); t=np.arange(n)/SR
    atk=bp(rng.standard_normal(n),freq*5,min(SR/2-500,freq*40))*np.exp(-t/0.006)*0.9
    for r in (4.2,5.4,6.8,8.1,9.5,11.7): atk+=np.sin(2*np.pi*freq*r*t+rng.uniform(0,6))*np.exp(-t/0.012)
    body=np.zeros(n)
    for r in (1.0,2.0,2.76,3.52,4.07,5.4,6.8,8.2,10.5):
        if freq*r<SR/2-500: body+=(1/(1+r*0.35))*np.sin(2*np.pi*freq*r*t*(1+0.0025*rng.uniform(-1,1))+rng.uniform(0,6))*np.exp(-t/(dur*(0.55/(1+0.3*r))))
    y=np.tanh((body+atk*0.7)*1.7)*np.minimum(1,t/0.002)
    return declick(norm(hp(lp(y,13000),freq*0.8),1)*amp)
def strafe(freq,dur,amp):
    n=int(dur*SR); t=np.arange(n)/SR; y=np.zeros(n)
    for det in (1.0,1.007,0.992,0.5,2.009): y[:n]+=saw(freq*det,dur)[:n]
    y=np.tanh(y*3.6); env=np.minimum(1,t/0.015)*np.minimum(1,(dur-t)/0.1)
    return declick(norm(lp(y,min(SR/2-500,freq*13))*env,1)*amp)
def corg(notes,dur,amp):
    n=int(dur*SR); t=np.arange(n)/SR; y=np.zeros(n)
    for nm in notes:
        fr=f(nm)
        for r,a in [(0.5,0.5),(1,1.0),(2,0.72),(3,0.36),(4,0.5)]:
            if fr*r<SR/2-500: y+=a*np.sin(2*np.pi*fr*r*t*(1+0.004*rng.uniform(-1,1))+rng.uniform(0,6))
    y=np.tanh(y*2.8); env=np.minimum(1,t/0.015)*np.minimum(1,(dur-t)/0.1)
    return declick(norm(lp(y*env,7000),1)*amp)
def mbrass(notes,dur,amp):
    n=int(dur*SR); t=np.arange(n)/SR; y=np.zeros(n)
    for nm in notes:
        fr=f(nm); b=np.zeros(n)
        for k in range(1,12): b+=(1/k**0.8)*np.sin(2*np.pi*k*fr*t+rng.uniform(0,6))
        y+=np.tanh(b*2.6)*np.minimum(1,t/0.03)*np.minimum(1,(dur-t)/0.1)
    return declick(norm(lp(y,10000),1)*amp)
def timp(freq,dur,amp,drive=2.8):
    n=int(dur*SR); t=np.arange(n)/SR; pitch=freq*(1+0.07*np.exp(-t/0.05))
    body=np.sin(2*np.pi*np.cumsum(pitch)/SR)+0.4*np.sin(2*np.pi*1.5*np.cumsum(pitch)/SR)
    atk=bp(rng.standard_normal(n),freq*2,freq*10)*np.exp(-t/0.02)*0.9
    return declick(norm(np.tanh(lp(body*np.exp(-t/(dur*0.5))+atk,600)*drive),1)*amp)
def bass_hit(amp):
    dur=0.5; n=int(dur*SR); t=np.arange(n)/SR
    y=np.sin(2*np.pi*(100*np.exp(-t/0.035)+40)*t)+bp(rng.standard_normal(n),40,240)*np.exp(-t/0.025)*0.8
    return declick(norm(np.tanh(lp(y,320)*1.8),1)*amp)
def crash(dur,amp):
    n=int(dur*SR); t=np.arange(n)/SR
    return declick(norm(bp(rng.standard_normal(n),2600,13000)*np.exp(-t/(dur*0.35)),1)*amp)
def ks(freq,dur,amp=1,decay=0.994,exc=4200):
    N=max(2,int(SR/freq)); n=int(dur*SR); x=np.zeros(n); x[:N]=lp(rng.standard_normal(N),exc)*np.hanning(N)
    a=np.zeros(N+2); a[0]=1; a[N]=-decay*0.5; a[N+1]=-decay*0.5
    return norm(hp(lfilter([1.],a,x),22),1)*amp
def threnody(n,lo,hi,voices=12,gliss=(-2.5,2.5)):
    t=np.arange(n)/SR; y=np.zeros(n)
    for _ in range(voices):
        f0=rng.uniform(lo,hi); f1=f0*2**(rng.uniform(*gliss)/12)
        ff=np.linspace(f0,f1,n)*(1+0.004*np.sin(2*np.pi*rng.uniform(4,8)*t))
        y+=np.sin(2*np.pi*np.cumsum(ff)/SR)*rng.uniform(.5,1)
    return declick(norm(bp(y,lo*0.85,min(SR/2-200,hi*1.6)),1))

def fmpiston(freq,dur,amp):
    n=int(dur*SR); t=np.arange(n)/SR
    mod=np.sin(2*np.pi*freq*t)*2.6*np.exp(-t/0.05)
    car=np.sin(2*np.pi*freq*t+mod)
    knock=bp(rng.standard_normal(n),freq*3,min(SR/2-500,freq*9))*np.exp(-t/0.004)*0.45
    env=np.exp(-t/(dur*0.4))*np.minimum(1,t/0.0018)
    return declick(norm(np.tanh((car+knock)*1.5)*env,1)*amp)
def marimba(freq,dur,amp):
    n=int(dur*SR); t=np.arange(n)/SR
    y=np.sin(2*np.pi*freq*t)*np.exp(-t/(dur*0.35))
    y+=0.34*np.sin(2*np.pi*freq*3.98*t)*np.exp(-t/(dur*0.12))
    y+=0.11*np.sin(2*np.pi*freq*9.6*t)*np.exp(-t/(dur*0.06))
    mallet=bp(rng.standard_normal(n),freq*2,min(SR/2-500,freq*8))*np.exp(-t/0.003)*0.3
    return declick(norm((y+mallet)*np.minimum(1,t/0.0015),1)*amp)

# voices
vlat,_=librosa.load("/mnt/user-data/outputs/006_requiem_voice/006_05_dies_irae.wav",sr=SR,mono=True); vlat=norm(hp(vlat,80),0.95)
vrex,_=librosa.load("/mnt/user-data/outputs/006_requiem_voice/006_06_rex_tremendae.wav",sr=SR,mono=True); vrex=norm(hp(vrex,80),0.95)
vcon,_=librosa.load("/mnt/user-data/outputs/006_requiem_voice/006_07_confutatis.wav",sr=SR,mono=True); vcon=norm(hp(vcon,80),0.95)
vhum,_=librosa.load("/mnt/user-data/uploads/demonstrationapostolicpreaching_01_irenaeus_64kb.mp3",sr=SR,mono=True,offset=200,duration=30); vhum=norm(hp(vhum,70),0.95)
VB=120.0
def utter(srcv,freq,dur,amp,dist=2.0,lphz=3000,bend=0.0,shape=(0.4,0.4),density=110,gl_ms=420):
    n=int(dur*SR); out=np.zeros(n); gl=int(gl_ms/1000*SR); win=np.hanning(gl); tg=[freq,freq*1.004]
    ratio0=freq/VB; need0=int(gl/ratio0); center=rng.integers(need0+1,max(need0+2,len(srcv)-need0-1)); jit=int(0.10*SR)
    for _ in range(density):
        tgt=tg[rng.integers(2)]; ratio=tgt/VB; need=int(gl/ratio)
        s=int(np.clip(center+rng.integers(-jit,jit),0,len(srcv)-need-1)); seg=srcv[s:s+need]
        g=np.interp(np.linspace(0,len(seg)-1,gl),np.arange(len(seg)),seg)*win
        d=rng.integers(0,n-gl); out[d:d+gl]+=g*rng.uniform(0.6,1)
    y=np.tanh(bp(out,80,lphz)*dist)
    if bend!=0:
        sp=1+(2**(bend/12)-1)*np.linspace(0,1,n); idx=np.clip(np.cumsum(sp),0,n-1); y=np.interp(idx,np.arange(n),y)
    t=np.arange(n)/SR; a,r=shape; env=np.clip(t/(a*dur),0,1)**1.4*np.clip((dur-t)/(r*dur),0,1)**1.2
    return declick(norm(y*env,1)*amp)
def vstab(srcv,dur,amp,freq,dist=3.0):    # shouted choir hammer (grave)
    return utter(srcv,freq,dur,amp,dist=dist,lphz=5000,shape=(0.06,0.4),density=120,gl_ms=120)
def vwall(srcv,dur,amp,targets,density=340,dist=2.2):   # damned wall
    n=int(dur*SR); out=np.zeros(n); gl=int(150/1000*SR); win=np.hanning(gl)
    for _ in range(density):
        tgt=targets[rng.integers(len(targets))]; ratio=tgt/VB; need=int(gl/ratio)
        if need<8 or need>=len(srcv): continue
        s=rng.integers(0,len(srcv)-need); seg=srcv[s:s+need]
        g=np.interp(np.linspace(0,len(seg)-1,gl),np.arange(len(seg)),seg)*win
        d=rng.integers(0,n-gl); out[d:d+gl]+=g*rng.uniform(.5,1)
    return declick(norm(np.tanh(bp(out,80,4200)*dist),1)*amp)


def ironknock(freq,dur,amp):
    n=int(dur*SR); t=np.arange(n)/SR; y=np.zeros(n)
    for r,a,dc in [(1.0,1.0,0.45),(2.76,0.6,0.18),(4.9,0.4,0.11),(7.7,0.28,0.07),(11.3,0.16,0.05)]:
        if freq*r<SR/2-500: y+=a*np.sin(2*np.pi*freq*r*t+rng.uniform(0,6))*np.exp(-t/(dur*dc))
    clang=bp(rng.standard_normal(n),freq*3,min(SR/2-500,freq*15))*np.exp(-t/0.005)*0.7
    y=np.tanh((y+clang)*1.9)*np.minimum(1,t/0.0012)
    return declick(norm(hp(y,freq*0.7),1)*amp)
def ikick(amp):
    dur=0.4; n=int(dur*SR); t=np.arange(n)/SR
    y=np.sin(2*np.pi*(120*np.exp(-t/0.03)+45)*t)+bp(rng.standard_normal(n),50,200)*np.exp(-t/0.02)*0.7
    return declick(norm(np.tanh(lp(y,300)*2.2),1)*amp)
def iclank(amp):
    dur=0.35; n=int(dur*SR); t=np.arange(n)/SR; f0=380; y=np.zeros(n)
    for r in (1,2.4,3.8,5.7,7.9): y+=np.sin(2*np.pi*f0*r*t+rng.uniform(0,6))*np.exp(-t/(dur*0.3/(1+0.2*r)))
    noise=bp(rng.standard_normal(n),1500,7000)*np.exp(-t/0.04)*0.8
    return declick(norm(np.tanh((y+noise)*1.6),1)*amp)
def keen(freq,dur,amp):    # keening reed line (clarinet nod)
    n=int(dur*SR); t=np.arange(n)/SR; vib=1+0.013*np.sin(2*np.pi*5.5*t)
    ph=2*np.pi*freq*np.cumsum(vib)/SR; y=np.sin(ph)+0.4*np.sin(2*ph)+0.2*np.sin(3*ph)
    y=np.tanh(y*1.7); env=np.sin(np.pi*np.clip(t/dur,0,1))**1.2
    return declick(norm(bp(y,freq*0.7,min(SR/2-500,freq*7))*env,1)*amp)

# ---- Mozart ghosts (Christopherson tape treatment): Rex + Lacrimosa ----
def _load(p): y,_=librosa.load(p,sr=SR,mono=True); return norm(y,0.95)
_rex=_load("/mnt/user-data/uploads/Classicals_de_-_Mozart_-_Requiem_in_D_minor__K_626_-_IIIc__Rex_tremendae__Choeur_des_Marais_.mp3")
_lac=_load("/mnt/user-data/uploads/Classicals_de_-_Mozart_-_Requiem_in_D_minor__K_626_-_IIIf__Lacrimosa__Choeur_des_Marais_.mp3")
def _sg(src,a,b): return src[int(a*SR):int(b*SR)].copy()
_REX1=_sg(_rex,11.7,13.7); _REX2=_sg(_rex,15.9,18.0); _SUS=_sg(_rex,3.0,8.5)
_ahh_src=_sg(_lac,37.0,40.6)
_LAC1=_sg(_lac,11.5,20.5); _LACS=_sg(_lac,1.0,9.0)
def tape_warp(y,wow=0.010,flutter=0.0035,wow_hz=0.45,flutter_hz=6.5,sag=0.0):
    n=len(y); t=np.arange(n)/SR
    drift=1+sag+wow*np.sin(2*np.pi*wow_hz*t+rng.uniform(0,6))+flutter*np.sin(2*np.pi*flutter_hz*t+rng.uniform(0,6))
    idx=np.clip(np.cumsum(drift),0,n-1); return np.interp(idx,np.arange(n),y)
def corrode(y,drive=1.9,lo=170,hi=4300,crush=6,hiss=0.012,crackle=6e-4):
    y=np.tanh(y*drive); y=bp(y,lo,hi)
    ds=max(1,crush); y2=np.repeat(y[::ds],ds)[:len(y)]; y=0.62*y+0.38*y2
    n=len(y); y=y+bp(rng.standard_normal(n),2200,9000)*hiss
    cr=(rng.random(n)<crackle); y=y+cr*rng.uniform(-1,1,n)*0.26
    return norm(y,1)
def gpitch(y,semi):
    if abs(semi)<1e-3: return y
    idx=np.clip(np.arange(0,len(y),2**(semi/12)),0,len(y)-1); return np.interp(idx,np.arange(len(y)),y)
def granulate(src,dur,gl_ms=180,rev=0.4,dens=0.8):
    n=int(dur*SR); out=np.zeros(n); gl=int(gl_ms/1000*SR); win=np.hanning(gl); step=max(1,int(gl*0.5/dens)); pos=0
    while pos<n-gl:
        s=int(rng.uniform(0,len(src)-gl-1)); g=src[s:s+gl]*win
        if rng.random()<rev: g=g[::-1]
        d=pos+int(rng.uniform(-1,1)*gl*0.4); d=max(0,min(n-gl,d)); out[d:d+len(g)]+=g*rng.uniform(0.6,1); pos+=step
    return declick(norm(out,1))
_ahh_h,_=librosa.effects.hpss(_ahh_src,margin=(1.0,4.0))
_ahh_h=bp(_ahh_h,420,3400)
_ahh_h=norm(_ahh_h,0.9)*np.hanning(len(_ahh_h))**0.35   # deep taper: strips any word-onset/offset hardness
SOPRANO_AHH=corrode(tape_warp(_ahh_h,wow=0.012,flutter=0.0035),drive=1.15,lo=350,hi=3600,crush=2,hiss=0.006,crackle=2e-4)
SOPRANO_AHH=declick(SOPRANO_AHH,260,500)
G_FORE=corrode(tape_warp(granulate(_SUS,10.0,gl_ms=210,rev=0.42),wow=0.014,flutter=0.004),drive=1.6,hi=3600,crush=8)
G_REX1=corrode(tape_warp(_REX1,wow=0.008,flutter=0.003),drive=2.1,hi=4600,crush=5)
G_REX2=corrode(tape_warp(_REX2,wow=0.009,flutter=0.0035),drive=2.1,hi=4600,crush=5)
G_REXLOW=corrode(gpitch(_REX1,-12),drive=1.8,hi=2600,crush=8)
G_DECAY=corrode(tape_warp(granulate(_REX2,11.0,gl_ms=170,rev=0.5,dens=0.7),wow=0.02,flutter=0.005,sag=-0.02),drive=1.5,hi=3200,crush=9)
G_LAC=corrode(tape_warp(gpitch(_LAC1,-1.0),wow=0.017,flutter=0.003),drive=1.7,hi=4200,crush=4)          # the weep (to C#)
G_LACSM=corrode(tape_warp(granulate(_LACS,12.0,gl_ms=240,rev=0.4),wow=0.02,flutter=0.004),drive=1.5,hi=3400,crush=7)
_harp=_load("harpsi/Classicals.de - Mozart - Requiem in D minor, K.626 - III. Dies Irae - Arranged for Harpsichord.mp3")
HARP_FRAG=corrode(tape_warp(_sg(_harp,1.0,4.0),wow=0.012,flutter=0.005),drive=1.6,hi=5000,crush=5)
HARP_DRV=corrode(tape_warp(gpitch(_sg(_harp,6.0,14.0),-1.0),wow=0.010,flutter=0.004),drive=1.7,hi=5200,crush=4)
HARP_CONF=corrode(tape_warp(gpitch(_sg(_harp,15.0,27.0),-1.0),wow=0.012,flutter=0.004),drive=1.8,hi=4800,crush=5)
OFFKEY=corrode(tape_warp(gpitch(_sg(_harp,20.0,24.0),0.6),wow=0.02,flutter=0.006),drive=1.6,hi=4200,crush=6)
_W1=corrode(tape_warp(_sg(_rex,18.6,19.5),wow=0.008,flutter=0.003),drive=1.5,hi=4600,crush=5)
_W2=corrode(tape_warp(_sg(_lac,12.6,13.6),wow=0.010,flutter=0.004),drive=1.5,hi=4400,crush=5)
_W3=corrode(tape_warp(_sg(_rex,16.2,17.2),wow=0.008,flutter=0.003),drive=1.5,hi=4600,crush=5)

# ---- precompute ----
STR={'C#L':strike_metal(f('C#2'),2.4,1.0),'G L':strike_metal(f('G2'),2.4,1.0),'C#H':strike_metal(f('C#3'),1.9,1.0)}
IKICK=ikick(1.0); ICLANK=iclank(1.0)
DRIVE=[["G2","Bb2","D3"],["C3","Eb3","G3"],["F2","A2","C3"],["Bb2","D3","F3"],["Eb2","G2","Bb2"],["C3","Eb3","F#3"],["D3","F#3","A3"],["G2","Bb2","D3"]]
RD=["G1","C2","F1","Bb1","Eb1","A1","D2","G1"]
WALL=[]
for ch,rt in zip(DRIVE,RD):
    ch2=ch+[ch[0][:-1]+str(int(ch[0][-1])+1)]
    b=corg(ch2,0.5,0.95)[:int(0.5*SR)]
    s2=strafe(f(rt),0.5,0.82)[:int(0.5*SR)]; b[:len(s2)]+=s2
    sb=mbrass(ch,0.5,0.58)[:int(0.5*SR)]; b[:len(sb)]+=sb
    WALL.append(declick(np.tanh(b*1.5)))
OUT=278.0; N=int(OUT*SR); bus=np.zeros(N); tg=np.arange(N)/SR
def soft(y,fin_ms=55,fout_ms=130):
    a=int(fin_ms/1000*SR); b=int(fout_ms/1000*SR); y=y.copy()
    if 1<a<len(y): y[:a]*=np.clip(np.arange(a)/a,0,1)**1.3
    if 1<b<len(y): y[-b:]*=np.clip(np.arange(b,0,-1)/b,0,1)**1.1
    return y
def bass_hit(amp):
    dur=0.5; n=int(dur*SR); t=np.arange(n)/SR
    y=np.sin(2*np.pi*(100*np.exp(-t/0.035)+40)*t)+bp(rng.standard_normal(n),40,240)*np.exp(-t/0.025)*0.55
    y=np.tanh(lp(y,320)*1.7)*np.minimum(1,t/0.007)
    return declick(norm(y,1)*amp)
def place(bus,sig,at):
    sig=declick(sig,5,30); i=int(at*SR)
    if i<0: sig=sig[-i:]; i=0
    e=min(len(bus),i+len(sig)); bus[i:e]+=sig[:max(0,e-i)]
__thr=threnody
def threnody(n,lo,hi,voices=12,gliss=(-2.5,2.5)): return soft(__thr(n,lo,hi,voices,gliss),90,220)
def gh(at,g,amp,rt=2.2,mix=0.18,seed=2): place(bus, room(soft(g,130,350),rt,mix,seed)*amp, at)
def hplace(at,buf,amp,rt=1.2,mix=0.10,seed=1): place(bus, room(soft(buf,50,140),rt,mix,seed)*amp, at)
def subswell(t0,dur,amp):
    n=int(dur*SR); ss=np.sin(2*np.pi*f('C#1')*np.arange(n)/SR)*amp*np.sin(np.pi*np.arange(n)/n); place(bus,ss,t0)
def contra(t0,amp,span=4.8):
    notes=['C#4','E4','G#4','B4','C#5','E5']; g=span/len(notes)
    for i,pc in enumerate(notes):
        a=amp*(0.55+0.55*i/(len(notes)-1))
        place(bus, room(ks(f(pc),2.0,a,decay=0.9958),1.1,0.06,seed=int((t0+i)*7)%6), t0+0.15+i*g)
DESC=['D4','C4','Bb3','A3','G3','F#3','Eb3','D3']
def desc_line(amp,note_dur=0.7,gap=0.55,decay=0.990):
    total=int((len(DESC)*gap+note_dur+0.6)*SR); buf=np.zeros(total)
    for i,pc in enumerate(DESC):
        seg=ks(f(pc),note_dur+0.6,amp,decay=decay); j=int(i*gap*SR); e=min(total,j+len(seg)); buf[j:e]+=seg[:e-j]
    return declick(norm(buf,1),4,60)
def cassette(y,grid_s=0.55,n_glitch=3):
    g=max(1,int(grid_s*SR)); ncells=max(1,len(y)//g)
    cells=[declick(y[i*g:(i+1)*g],2,9) for i in range(ncells)]
    if len(y)>ncells*g: cells.append(declick(y[ncells*g:],2,9))
    glitch_at=set()
    if len(cells)>3 and n_glitch>0:
        glitch_at=set(int(x) for x in rng.choice(range(1,len(cells)-1), size=min(n_glitch,len(cells)-2), replace=False))
    out=[]
    for i in range(len(cells)):
        out.append(cells[i])
        if i in glitch_at:
            for _ in range(int(rng.integers(2,4))): out.append(cells[i]*np.linspace(1,0.7,len(cells[i])))
            k=min(2,i+1); rw=np.concatenate(cells[i-k+1:i+1])[::-1]
            idx=np.clip(np.cumsum(np.linspace(1,2.2,len(rw))),0,len(rw)-1)
            rw=declick(np.interp(idx,np.arange(len(rw)),rw)*np.linspace(1,0.45,len(rw)),2,12)
            out.append(rw)
    return declick(norm(np.concatenate(out),1))
def leadin(t_drive,wbuf,amp,seed=1):
    t=np.arange(len(wbuf))/SR; w=wbuf*np.clip(t/(len(wbuf)/SR),0,1)**0.6
    place(bus, room(soft(w,60,90),1.6,0.14,seed=seed)*amp, t_drive-len(wbuf)/SR+0.12)
def dreamy(buf,trem_hz=5.2,depth=0.32):
    y=tape_warp(buf,wow=0.022,flutter=0.005); t=np.arange(len(y))/SR
    return declick(y*(1-depth+depth*np.sin(2*np.pi*trem_hz*t)))
def groove_out(t0,t1,base):
    beat=0.6; bar=beat*4; slot=beat/2; at=t0
    while at<t1-bar:
        intens=base*np.interp(at,[t0,t0+0.65*(t1-t0),t1],[0.92,0.82,0.04])
        for e,a in [(0,1.0),(4,0.9),(7,0.5)]: pc_place(IKICK, at+e*slot, 0.40*intens*a)
        for e,a in [(2,1.0),(6,1.0),(3,0.4)]: pc_place(ICLANK, at+e*slot, 0.26*intens*a)
        pc_place(STR['C#L'], at, 0.15*intens)
        at+=bar
    return at
def pc_place(buf,at,amp):
    i=int(at*SR); e=min(N,i+len(buf))
    if i<N: bus[i:e]+=buf[:e-i]*amp

SPINE_OCT=['C#3','B2','A2','G#2','F#2','E2','D#3','C#3']
def ritual(t0,t1,base,dense=False,bar=1.9,skip=0.0,jit=0.0,dense_after=None):
    if dense_after is None: dense_after=t0
    ti=0; at=t0
    while at<t1-bar:
        intens=base*np.interp(at,[t0,t1],[0.7,1.2]); jt=rng.uniform(-jit,jit)
        d_here = dense and (at>=dense_after)
        if rng.random()>=skip:
            for off,a in [(0.0,1.0),(0.9,0.6),(1.35,0.42)]:
                place(bus, fmpiston(f('C#2'),0.5,0.22*intens*a), at+off+jt)
                if d_here: place(bus, fmpiston(f('C#3'),0.4,0.11*intens*a), at+off+jt)
            pc=SPINE_OCT[ti%len(SPINE_OCT)]
            place(bus, room(ironknock(f(pc),0.8,0.48*intens),0.7,0.05,seed=ti%6), at+jt)
            if d_here:
                pc2=SPINE_OCT[(ti+1)%len(SPINE_OCT)]
                place(bus, room(ironknock(f(pc2),0.6,0.32*intens),0.7,0.05,seed=(ti+1)%6), at+0.95+jt)
        ti+=1; at+=bar
    return at
MOTIF=['G4','C#5','A4','G4','D#5','C#5']
def ks_batch(t0,amp,gap0=0.46,accel=0.84,reps=2,cluster=True):
    at=t0; gap=gap0; s=0
    for _ in range(reps):
        for pc in MOTIF:
            place(bus, room(ks(f(pc),0.9,amp),0.8,0.05,seed=s%6), at); s+=1
            at+=gap; gap=max(0.11,gap*accel)
    if cluster:
        for pc in MOTIF+['C#5','G4']:
            place(bus, room(ks(f(pc),1.5,amp*0.85,decay=0.992),0.9,0.06,seed=s%6), at+rng.uniform(0,0.03)); s+=1
        at+=1.2
    return at
def wall_swell(t0,root,chord,amp,dur=5.0):
    body=corg(chord,dur,0.95); s=strafe(f(root),dur,0.7); body[:len(s)]+=s
    br=mbrass(chord,dur,0.5); body[:len(br)]+=br
    t=np.arange(len(body))/SR; env=np.sin(np.pi*np.clip(t/dur,0,1))**1.5
    place(bus, room(np.tanh(body*env*1.4),1.0,0.07,seed=int(t0)%6)*amp, t0)
def wall_pass(t0,intens,chord_dur,roll=0.11,nchords=8,ramp=1.9,fo=2.4):
    total=nchords*chord_dur; at=t0
    def rmp(rel): return (0.10+0.90*min(1,rel/ramp))*min(1,max(0,(total-rel)/fo))**1.3
    for k in range(nchords):
        stab=WALL[k%len(DRIVE)]; rtt=0.0
        while rtt<chord_dur-0.12:
            rf=rmp((at+rtt)-t0)
            place(bus, stab*(intens*rf*(0.75+0.25*np.sin(2*np.pi*12*rtt))), at+rtt); rtt+=roll
        rtn=RD[k%len(RD)]; tr=0.0
        while tr<chord_dur-0.15:
            rf=rmp((at+tr)-t0)
            place(bus, timp(f(rtn),0.45,(0.45*intens+0.1)*rf,drive=3.0), at+tr); tr+=0.18
        place(bus, bass_hit((0.35+0.3*intens)*rmp(at-t0)), at); at+=chord_dur
    return at
def groove(t0,t1,base):     # Solar Lodge: merciless primitive drum-machine, building/spiraling
    beat=0.55; bar=beat*4; slot=beat/2; at=t0
    while at<t1-bar:
        intens=base*np.interp(at,[t0,t0+0.72*(t1-t0),t1],[0.55,1.35,0.45])
        for e,a in [(0,1.0),(4,0.9),(7,0.55)]: pc_place(IKICK, at+e*slot, 0.5*intens*a)
        for e,a in [(2,1.0),(6,1.0),(3,0.4),(5,0.4)]: pc_place(ICLANK, at+e*slot, 0.32*intens*a)
        pc_place(STR['C#L'], at, 0.20*intens)
        if intens>0.85: pc_place(STR['G L'], at+2*beat, 0.15*intens)
        for e in (1,3,5,7): place(bus, fmpiston(f('C#2'),0.4,0.09*intens), at+e*slot)
        at+=bar
    return at

# RUMBLE strike punctuation (only 0-115; groove owns 120-205)
rp=2.0; kk=0
rlev=lambda t: 0.22 + (0.22 if 62<=t<108 else 0.0)
while rp<115:
    strike = STR['G L'] if kk%4==3 else (STR['C#H'] if (62<=rp<108 and kk%2==0) else STR['C#L'])
    place(bus, strike*max(0.10,rlev(rp)*(1.1 if kk%2==0 else 0.7)), rp)
    rp+= (3.4 if rp<62 else 1.7); kk+=1

# ===== 0-62 DIES IRAE BUILD + faint ghost foreshadow =====
csub0=np.sin(2*np.pi*f('C#1')*np.arange(int(16*SR))/SR)*0.15; csub0*=np.clip(np.arange(len(csub0))/SR/3,0,1)
place(bus, csub0, 0.0)
place(bus, room(SOPRANO_AHH,3.0,0.30,seed=7)*0.30, 0.4)
ritual(6.0,14.0,0.42); ritual(14.0,38.0,0.5,bar=2.4,skip=0.14,jit=0.12)
place(bus, room(cassette(G_FORE,0.6,2),2.4,0.20,seed=5)*0.15, 9.0)
place(bus, room(cassette(HARP_FRAG,0.5,2),1.6,0.14,seed=2)*0.13, 17.5)
_hrev=_sg(_harp,30.0,40.0)[::-1]
G_INTRO=dreamy(corrode(tape_warp(granulate(_hrev,9.0,gl_ms=260,rev=0.5),wow=0.02,flutter=0.005),drive=1.5,hi=4200,crush=6),trem_hz=4.8,depth=0.40)
place(bus, room(G_INTRO,2.8,0.26,seed=3)*0.16, 8.0)

ks_batch(29.0,0.26,reps=1,cluster=False); gh(20.0,G_FORE,0.13,2.6,0.22,5)
ritual(38.0,62.0,0.62,dense=True,jit=0.10,dense_after=46.0)
_thr_n=int(20*SR); _thr_t=np.arange(_thr_n)/SR
_thr_env=np.clip(_thr_t/3.0,0,1)*np.clip((20-_thr_t)/4.5,0,1)   # slow fade in/out, spanning 30-50s
place(bus, room(threnody(_thr_n,2400,4200,9,gliss=(-1.8,1.8))*_thr_env*0.16,1.0,0.06,seed=9), 30.0)

wall_swell(50.0,'G1',["G2","Bb2","D3"],0.46,5.5); ks_batch(56.0,0.34,gap0=0.4,accel=0.82,cluster=False)
place(bus, room(threnody(int(20*SR),2200,4600,9)*np.clip((np.arange(int(20*SR))/SR)/14,0,1)*0.15,1.0,0.06,seed=1), 54.0)
gh(46.0,G_FORE,0.15,2.6,0.22,1)
hplace(24.0,HARP_FRAG,0.13,1.6,0.14,2); hplace(48.0,HARP_FRAG,0.15,1.6,0.14,3)

# ===== 62-108 REX CLIMAX: two Rex, OPEN breaks (true silence so the roar is clear) =====
def break_sub(at,dur=4.0):
    n=int(dur*SR); s=np.sin(2*np.pi*f('C#1')*np.arange(n)/SR)*0.10*np.sin(np.pi*np.arange(n)/n); place(bus,s,at)
leadin(62.0,_W1,0.34,1)
e=wall_pass(62.0,0.36,1.2,nchords=5); ritual(62.0,e,0.6); contra(62.0,0.40)
gh(e-1.0,G_REX1,1.0,1.9,0.16,2); gh(e-0.9,G_REXLOW,0.6,2.2,0.22,4)

g1=e+3.6
leadin(g1,_W2,0.36,2)
e2=wall_pass(g1,0.42,1.12,nchords=6,ramp=2.3); ritual(g1,e2,0.7); contra(g1,0.43)
gh(e2-1.0,G_REX2,1.05,1.9,0.16,3)

g2=e2+3.6
leadin(g2,_W3,0.38,3)
e3=wall_pass(g2,0.48,1.05,nchords=8,ramp=2.3); ritual(g2,e3,0.8); contra(g2,0.45)
ks_batch(g2+1.0,0.4,gap0=0.32,accel=0.8,reps=1)
place(bus, room(threnody(int(16*SR),3200,6800,12)*0.24,1.0,0.06,seed=3), g2)
gh(e3-1.0,G_REX1,0.85,1.9,0.16,2)

# 108-120 subside
gh(110.0,G_DECAY,0.24,3.0,0.30,5)
place(bus, room(threnody(int(12*SR),3000,5600,10,gliss=(-6,-2))*0.18,1.0,0.06,seed=1), 110.0)

# ===== 120-205 SOLAR LODGE: prolonged merciless groove + Lacrimosa weep + Rex chant + keen =====
groove(120.0,204.0,0.6)
for _at,_a in [(128.0,0.16),(144.0,0.20),(160.0,0.24),(176.0,0.27),(192.0,0.30)]: hplace(_at,HARP_DRV,_a,1.3,0.12,int(_at)%6)
# Lacrimosa weeps over the machine (corroded, on C#)
for at,a in [(128.0,0.5),(150.0,0.58),(172.0,0.62),(190.0,0.7)]:
    gh(at,G_LAC,a,2.6,0.24,int(at)%6)
gh(134.0,G_LACSM,0.26,3.0,0.3,2)
# Rex ghost chant recurs (the deity's domain)
for at,a in [(140.0,0.5),(165.0,0.55),(186.0,0.6)]:
    gh(at,G_REX2 if int(at)%2 else G_REX1,a,2.0,0.18,int(at)%6)
# keening reed line (clarinet nod)
for at,fr in [(145.0,'D5'),(160.0,'F5'),(178.0,'A5'),(196.0,'G5')]:
    place(bus, room(keen(f(fr),4.5,0.16),1.4,0.10,seed=int(at)%6), at)
# black sun rising: threnody + strike surge into the peak (~180-200)
place(bus, room(threnody(int(28*SR),2800,6400,12)*np.clip((np.arange(int(28*SR))/SR)/18,0,1)*0.24,1.0,0.06,seed=3), 176.0)
for at in np.arange(182,202,1.3):
    place(bus, STR['C#H']*0.24, at)
# low mass under the groove (thin), swelling to the black-sun peak
mass=np.zeros(N); tt=120.0
while tt<204.0: place(mass, ks(rng.uniform(150,1300),0.9,rng.uniform(.3,.5)), tt); tt+=rng.uniform(0.12,0.2)
menv=np.interp(tg,[0,76,82,105,118,120,176,196,205,245,OUT],[0,0,0.15,0.28,0.14,0.10,0.14,0.30,0.14,0.06,0])
place(bus, norm(bp(mass,90,6200),1)*menv,0)

# ===== 205-245 CONFUTATIS (the sentence / damned / flames) =====


place(bus, room(soft(vwall(vhum,20,0.18,[f('C#2'),f('G2')],dist=1.5),2600,320),1.2,0.06,seed=11), 212.0)
gh(216.0,G_LAC,0.30,3.0,0.3,4)                                   # the weep haunts the sentence

place(bus, room(threnody(int(26*SR),3000,6000,10,gliss=(-8,-3))*0.15*np.clip(np.arange(int(26*SR))/SR/8,0,1),1.0,0.06,seed=3), 206.0)
for at in np.arange(206,238,2.0):
    place(bus, room(strafe(f(rng.choice(['C#2','G2','Eb2','A2'])),1.4,0.18),0.9,0.06,seed=int(at)), at)
place(bus, room(soft(HARP_CONF,2600,240),1.4,0.12,seed=2)*0.16, 206.0); hplace(220.0,HARP_CONF,0.20,1.4,0.12,3)
subswell(202.0,12.0,0.10); subswell(236.0,18.0,0.09)
gh(202.0,G_LAC,0.34,3.0,0.30,5)
ks_batch(212.0,0.15,gap0=0.5,accel=0.9,reps=1,cluster=False)
ks_batch(230.0,0.15,gap0=0.5,accel=0.9,reps=1,cluster=False)
DLINE=desc_line(0.20)
for _at,_a in [(216.0,1.0),(232.0,0.85)]:
    place(bus, room(cassette(DLINE,0.55,2),1.5,0.12,seed=int(_at)%6)*_a, _at)
    hplace(_at+0.6, OFFKEY, 0.15*_a, 1.6, 0.14, int(_at)%6)
place(bus, room(cassette(HARP_CONF[:int(4.0*SR)],0.5,2),1.3,0.12,seed=3)*0.5, 234.0)
groove_out(240.0,276.0,0.5)

# ===== 248-270 COLLAPSE to bare C# (+ final ghost trace) =====
csub=np.sin(2*np.pi*f('C#1')*np.arange(int(27*SR))/SR)*np.clip(np.arange(int(27*SR))/SR/2,0,1)*0.4
csub*=np.minimum(1,(len(csub)-np.arange(len(csub)))/(4*SR)); place(bus, csub, 250.0)
gh(252.0,G_LACSM,0.18,3.4,0.36,3)


mix=hp(bus,26); mix=lp(mix,16000)
mix*=np.clip(tg/0.5,0,1)*np.clip((OUT-tg)/6.0,0,1)
mix=np.tanh(mix*0.92); mix=mix/np.max(np.abs(mix))*10**(-0.8/20)
st=np.stack([mix,np.concatenate([np.zeros(int(.011*SR)),mix])[:len(mix)]],1)
sf.write("/mnt/user-data/outputs/006_requiem_voice/006_mvt3_full.wav",st,SR,subtype="PCM_24")
d=len(mix)/SR; print("Movement III v17 %.1fs (%d:%02d)"%(d,d//60,d%60))
