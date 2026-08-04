"""
007 — Movement III, v2: NOTTURNO (the sphere).
Scelsi physics: one note inhabited. Six breaths, each ~50-60 s of
overlapping swells orbiting Ab at cent-distances. The spread widens
breath by breath — 10c, 22c, 38c — until the A-natural itself enters
as the outermost orbit (breath 4, the dyad become a world), then the
sphere contracts back toward unison and the torch goes out.
No sustain is flat; everything breathes. The hours still strike.
Chopin is heard once, through the sphere, from the town below.
"""
import numpy as np, soundfile as sf, subprocess, os
from scipy.signal import butter, sosfilt, fftconvolve
SR=48000; rng=np.random.default_rng(672004)
def hp(x,f,o=2): return sosfilt(butter(o,f,btype='high',fs=SR,output='sos'),x)
def lp(x,f,o=2): return sosfilt(butter(o,min(f,SR/2-200),btype='low',fs=SR,output='sos'),x)
def bp(x,lo,hi,o=2): return sosfilt(butter(o,[lo,min(hi,SR/2-200)],btype='band',fs=SR,output='sos'),x)
def drv(x,a): return np.tanh(x*a)/np.tanh(a)
def norm(y): return y/(np.max(np.abs(y))+1e-9)
def resample(y,rate):
    nn=max(4,int(len(y)/rate)); idx=np.arange(nn)*rate
    i0=np.clip(idx.astype(int),0,len(y)-2); fr=idx-i0
    return y[i0]*(1-fr)+y[i0+1]*fr
def gstretch(y,factor,grain=0.09):
    g=int(grain*SR); hin=int(g*0.5); hout=int(hin*factor)
    nn=int(len(y)*factor)+g; out=np.zeros(nn); win=np.hanning(g); pos=0; opos=0
    while pos+g<len(y) and opos+g<nn:
        out[opos:opos+g]+=y[pos:pos+g]*win
        pos+=hin; opos+=hout
    return norm(out[:int(len(y)*factor)])
def edge(y,fi=0.5,fo=0.5):
    y=y.copy(); a=min(int(fi*SR),len(y)//2); b=min(int(fo*SR),len(y)//2)
    if a>0: y[:a]*=np.linspace(0,1,a)
    if b>0: y[-b:]*=np.linspace(1,0,b)
    return y
def make_ir(dur,dec,hpf,seed):
    n=int(dur*SR); r=np.random.default_rng(seed)
    ir=hp(r.standard_normal(n)*np.exp(-np.arange(n)/SR*dec),hpf); ir[:150]*=np.linspace(0,1,150)
    return ir/(np.max(np.abs(ir))+1e-9)

D="/home/claude/007/"
def load48(path,tag):
    dst=D+f"alt48/{tag}.wav"
    if not os.path.exists(dst):
        subprocess.run(["ffmpeg","-y","-i",path,"-ar","48000","-ac","1",dst],capture_output=True)
    y,_=sf.read(dst); return y
def seg(y,a,b): return norm(y[int(a*SR):int(b*SR)].copy())

wil,_=sf.read(D+"dewitt/willie.wav")
lomax,_=sf.read(D+"dewitt/lomax39.wav")
TROLL=norm(load48(D+"vsco_pack/Percussion/Timpani/Rolls/Timpani3_Roll_v3_rr1_Sum.wav","trolln") if os.path.exists(D+"vsco_pack/Percussion/Timpani/Rolls/Timpani3_Roll_v3_rr1_Sum.wav") else sf.read(D+"vsco_pack/Percussion/Timpani/Rolls/Timpani3_Roll_v3_rr1_Sum.wav")[0])
morF,_=sf.read(D+"dewitt/moreschi___08-G_T_54764.wav")
bells,_=sf.read(D+"dewitt/bells__clockstrikesnoonfrenchquarterNOLA9512.wav")
chop=load48(D+"nocturne/chopin_nocturne_1923.mp3","chop")
boul=load48(D+"nocturne/boulanger_nocturne_heifetz_1924.mp3","boul")
nice=load48(D+"nocturne/nice_bells_birds_pd.mp3","nice")
if not os.path.exists(D+"alt48/ocean90.wav"):
    subprocess.run(["ffmpeg","-y","-ss","60","-t","95","-i",D+"nocturne/ocean_waves_pd.mp3","-ar","48000","-ac","1",D+"alt48/ocean90.wav"],capture_output=True)
ocean,_=sf.read(D+"alt48/ocean90.wav")
ef=load48(D+"nocturne/eternal_father_1912.mp3","efath")
shk,_=sf.read(D+"nocturne/shackleton.wav")
if not os.path.exists(D+"alt48/freya1.wav"):
    subprocess.run(["ffmpeg","-y","-ss","55","-t","75","-i","/mnt/user-data/uploads/twixtlandandsea_11_conrad_128kb.mp3","-ar","48000","-ac","1",D+"alt48/freya1.wav"],capture_output=True)
freya,_=sf.read(D+"alt48/freya1.wav")
shen=load48(D+"nocturne/shenandoah_1925.mp3","shen")
sist=load48(D+"nocturne/sistine_ave_maria_1924.mp3","sist")
V=D+"nocturne/vsco/"
def vs(f): return norm(load48(V+f, f.replace('.wav','').replace('#','s')))
HARP_A2=vs("Strings__Harp__KSHarp_A2_mf.wav")
HARP_A4=vs("Strings__Harp__KSHarp_A4_mf.wav")
HARP_A6=vs("Strings__Harp__KSHarp_A6_mf.wav")
NEP=[vs(f"Miscellania_Raw__Misc_2__NepaleseBells__fx1_short_r0{k}_main.wav") for k in range(1,6)]
GONG_PP=vs("Miscellania_Raw__Misc_1__cymbgong_pp.wav")
GONG_W=vs("Miscellania_Raw__Misc_1__cymbgong_weird.wav")
ALIEN=vs("Miscellania_Raw__Misc_1__alien2_pp.wav")
V2=D+"vsco_pack/"
CTB1=norm(load48(V2+"Strings/Solo_Contrabass/SusNV/BKCtbss_SusNV_G#1_v1_rr1.wav","ctb1n"))
CTB2=norm(load48(V2+"Strings/Solo_Contrabass/SusNV/BKCtbss_SusNV_G#2_v1_rr1.wav","ctb2n"))
TBN1=norm(load48(V2+"Brass/OldTrombone/Sustain/Trombone_Sustain_A1_v3_1.wav","tbn1n"))
TBN2=norm(load48(V2+"Brass/OldTrombone/Sustain/Trombone_Sustain_A2_v3_1.wav","tbn2n"))
VLN_TR=norm(load48(V2+"Strings/Violin_Section/Trem/VlnEns_Trem_A2_v1.wav","vlntrn"))
WHUM=seg(wil,162.0,174.0)
BSTR=seg(bells,66.0,80.0)
def quietest(y,win=2.0):
    w=int(win*SR); best=None; bi=0
    for a in range(0,len(y)-w,w//2):
        e=float(np.mean(y[a:a+w]**2))
        if best is None or e<best: best=e; bi=a
    return y[bi:bi+w].copy()
CRACK=norm(np.concatenate([quietest(morF),quietest(wil)]))
def sizzle_long(src):
    y=src.copy(); nn=len(y); tt=np.arange(nn)/SR
    fc=np.interp(tt,[0,tt[-1]*0.5,tt[-1]],(2400,3600,2800))
    rm=hp(y*np.sin(2*np.pi*np.cumsum(fc)/SR),2000)
    envf=lp(np.abs(y),8.0); envf/=(np.max(envf)+1e-9)
    crk=np.tile(CRACK,nn//len(CRACK)+2)[:nn]
    return norm(hp(drv(drv(rm*3.0,4.0)+crk*envf*1.3,2.5),1700))

# trombone/contrabass are on A; the sphere's center is Ab
dn=1/2**(-1/12)          # resample rate for A -> Ab (rate=2**(1/12) lowers by 1 st)
TBN1=norm(resample(TBN1,dn)); TBN2=norm(resample(TBN2,dn))
VLN_AB=norm(resample(VLN_TR,dn))
HARP_AB2=norm(resample(HARP_A2,dn)); HARP_AB4=norm(resample(HARP_A4,dn)); HARP_AB6=norm(resample(HARP_A6,dn))

def sustain(src,dur):
    """tile with crossfades to reach dur — the source re-articulated, never frozen"""
    xf=int(1.2*SR); out=src.copy()
    while len(out)<int(dur*SR):
        nxt=src.copy()
        a=out[-xf:]*np.linspace(1,0,xf); b2=nxt[:xf]*np.linspace(0,1,xf)
        out=np.concatenate([out[:-xf],a+b2,nxt[xf:]])
    return out[:int(dur*SR)]

DUR=540.0; TOT=int(DUR*SR); L=np.zeros(TOT); R=np.zeros(TOT)
def place(sig,t,g=1.0,pan=0.0):
    i=int(t*SR); e=min(TOT,i+len(sig))
    if e<=i: return
    gl,gr=np.sqrt(0.5-pan/2),np.sqrt(0.5+pan/2)
    L[i:e]+=sig[:e-i]*g*gl; R[i:e]+=sig[:e-i]*g*gr
reps=TOT//len(CRACK)+2
bedy=np.tile(CRACK,reps)[:TOT].copy()
bedy[:int(4*SR)]*=np.linspace(0,1,int(4*SR)); bedy[-int(6*SR):]*=np.linspace(1,0,int(6*SR))
L+=bedy*10**(-35/20); R+=np.roll(bedy,int(0.013*SR))*10**(-35/20)
del bedy
irN=make_ir(1.6,2.6,240,31)
def far(v,wet=0.4):
    w2=fftconvolve(norm(v),irN)[:len(v)]
    return norm(norm(v)*(1-wet)+norm(w2)*wet)

irD=make_ir(2.6,1.7,180,77)
def depop(y):
    env=lp(np.abs(y),40)+1e-9
    sp=np.abs(y)>4.0*env
    y=y.copy(); y[sp]*=0.22
    return y
def oceanvoice(v,t,g,pan,surf_lo=900,surf_hi=3400,rate_wob=0.006):
    y=depop(norm(v))
    y=hp(y,200)
    ch2=int(0.5*SR); ov=int(0.06*SR); out=np.zeros(len(y)+ch2)
    i=0
    while i<len(y):
        rt=1.0+rate_wob*np.sin(2*np.pi*0.09*(i/SR)+0.7)
        gr=resample(y[i:i+ch2+800],rt)[:ch2]
        if len(gr)<ch2: gr=np.pad(gr,(0,ch2-len(gr)))
        gr=gr.copy(); gr[:ov]*=np.linspace(0,1,ov); gr[-ov:]*=np.linspace(1,0,ov)
        out[i:i+ch2]+=gr
        i+=ch2-ov
    y=out[:len(v)]
    tt4=np.arange(len(y))/SR
    cutd=surf_lo+(surf_hi-surf_lo)*(0.5+0.5*np.sin(2*np.pi*0.05*tt4+1.1))
    ylo=lp(y,surf_lo+80); yhi=lp(y,surf_hi)
    mw=np.clip((cutd-surf_lo)/(surf_hi-surf_lo),0,1)
    y=ylo*(1-mw)+yhi*mw
    w3=fftconvolve(y,irD)[:len(y)]
    tailw=0.16+0.3*(0.5+0.5*np.sin(2*np.pi*0.05*tt4-0.4))
    y=y*0.8+norm(w3)*tailw*np.max(np.abs(y))
    y=norm(y)
    press=lp(y,480)*0.5                              # the body under the surface
    place(edge(y,1.2,1.8),t,g,pan)
    place(edge(press,1.6,2.2),t,g*0.55,pan*0.4)
    wh=hp(rng.standard_normal(len(y)),1400)*lp(np.abs(y),6.0)*2.2   # the whisper ghost
    place(edge(norm(wh)*0.5,1.5,2.0),t+0.05,g*0.28,-pan)
def spiral(y,factor,grain=0.045,semis_start=0.0,semis_end=-5.0):
    g2=int(grain*SR); hin=int(g2*0.5); hout=int(hin*factor)
    nout=int(len(y)*factor)+g2; out=np.zeros(nout); win=np.hanning(g2)
    pos=0; opos=0
    while pos+int(g2*1.8)<len(y) and opos+g2<nout:
        prog=opos/max(1,nout)
        rt=2**((semis_start+(semis_end-semis_start)*prog)/12.0)
        src2=y[pos:pos+max(4,int(g2*rt)+2)]
        gr=resample(src2,rt)[:g2]
        if len(gr)<g2: gr=np.pad(gr,(0,g2-len(gr)))
        out[opos:opos+g2]+=gr*win
        pos+=hin; opos+=hout
    return norm(out[:int(len(y)*factor)])
GSRC=seg(lomax,162.0,163.1)
def gull(t,g=0.16,pan=0.4):
    dt=rng.uniform(-1.5,1.5)
    cry=hp(spiral(GSRC,0.85,grain=0.03,semis_start=7+dt,semis_end=-3+dt),900)
    place(edge(norm(cry),0.05,0.25),t,g,pan)
def orb_lean(src,lean,t,dur,g,pan):
    y=sustain(src,dur+1.5)
    n=int(dur*SR); ch=int(0.5*SR); ov=int(0.06*SR)
    out=np.zeros(n+ch)
    i=0
    while i<n:
        prog=i/n
        c=np.interp(prog,[0,0.3,0.5,0.75,1.0],[0,lean,0,-lean*0.6,0])
        gr=resample(y[i:i+ch+800],2**(c/1200))[:ch]
        if len(gr)<ch: gr=np.pad(gr,(0,ch-len(gr)))
        gr=gr.copy(); gr[:ov]*=np.linspace(0,1,ov); gr[-ov:]*=np.linspace(1,0,ov)
        out[i:i+ch]+=gr
        i+=ch-ov
    yy=out[:n]
    env=np.interp(np.arange(n)/SR,[0,dur*0.4,dur*0.75,dur],[0,1,0.8,0])**1.2
    place(norm(yy)*env,t,g,pan)
def orb(src,cents,t,dur,g,pan,att=0.45,cut=None):
    """one orbit: a breathing swell of the source, detuned by cents"""
    y=sustain(src,dur+1.0)
    y=norm(resample(y,2**(cents/1200)))[:int(dur*SR)]
    if cut: y=lp(y,cut)
    tt=np.arange(len(y))/SR
    Ld=len(y)/SR
    env=np.interp(tt,[0,Ld*att,Ld*0.72,Ld],[0,1,0.75,0])**1.25
    wav=1.0+0.05*np.sin(2*np.pi*0.11*tt+rng.uniform(0,6.28))
    place(y*env*wav,t,g,pan)

# ===== THE TORCH: the sphere's surface =====
TSRC=np.tile(WHUM,7)[:int(84*SR)]
TORCH=sizzle_long(TSRC)
fl=lp(rng.standard_normal(TOT),0.5); fl=0.85+0.3*(fl/np.max(np.abs(fl)+1e-9))
tor=np.zeros(TOT); tpos=int(5*SR)
while tpos<TOT-int(2*SR):
    ch=edge(TORCH.copy(),1.5,1.5)
    e2=min(TOT,tpos+len(ch)); tor[tpos:e2]+=ch[:e2-tpos]
    tpos+=len(ch)-int(3*SR)
tglob=np.interp(np.arange(TOT)/SR,[0,5,12,455,486,540],[0,0,0.1,0.1,0.015,0.0])
ts=tor*fl*tglob
L+=ts*0.7; R+=np.roll(ts,int(0.011*SR))*0.7
del tor, ts, fl, tglob
WV=norm(hp(lp(ocean,6000),140))
wsig=np.zeros(TOT); wpos=0
while wpos<TOT:
    chw=edge(WV.copy(),2.0,2.0)
    e3=min(TOT,wpos+len(chw)); wsig[wpos:e3]+=chw[:e3-wpos]
    wpos+=len(chw)-int(4*SR)
wglob=np.interp(np.arange(TOT)/SR,[0,8,20,532,540],[0,0.02,0.11,0.11,0.0])
wv=wsig*wglob
mig=0.5+0.35*np.sin(2*np.pi*np.arange(TOT)/SR/47.0)
L+=wv*0.72*mig; R+=np.roll(wv,int(0.017*SR))*0.72*(1-mig+0.5)
del wsig, wv, mig, wglob
RUM=norm(lp(ocean,300))
rs=np.zeros(TOT); rp=int(7*SR)
while rp<TOT:
    chr_=edge(RUM.copy(),2.5,2.5); e4=min(TOT,rp+len(chr_)); rs[rp:e4]+=chr_[:e4-rp]
    rp+=len(chr_)-int(5*SR)
rglob=np.interp(np.arange(TOT)/SR,[0,15,30,528,540],[0,0.02,0.055,0.055,0])
L+=rs*rglob; R+=np.roll(rs,int(0.023*SR))*rglob
del rs, rglob
UND=norm(lp(ocean,800))
us=np.zeros(TOT); up=int(13*SR)
while up<TOT:
    chu=edge(UND.copy(),3.0,3.0); e5=min(TOT,up+len(chu)); us[up:e5]+=chu[:e5-up]
    up+=len(chu)-int(6*SR)
ulfo=0.5+0.5*np.sin(2*np.pi*np.arange(TOT)/SR/31.0)
uglob=np.interp(np.arange(TOT)/SR,[0,20,40,530,540],[0,0.01,0.045,0.045,0])*ulfo
L+=us*uglob*0.9; R+=np.roll(us,int(0.02*SR))*uglob
del us, uglob, ulfo, UND, RUM

# ===== THE BREATHS =====
# (t0, dur, spread_cents, intensity, A-natural present)
B=[(10.0,52,8,0.48,False),(64.0,54,18,0.6,False),(120.0,56,30,0.72,False),
   (178.0,60,44,0.84,False),(240.0,62,56,0.95,True),(304.0,58,34,0.74,True),
   (364.0,60,16,0.52,False)]
for bi,(t0,dur,sp,inten,hasA) in enumerate(B):
    # core: two contrabasses straddling the center
    orb(CTB1,-sp*0.5,t0,dur*0.9,0.3*inten,-0.06,att=0.5,cut=700)
    orb(CTB1,+sp*0.35,t0+dur*0.08,dur*0.85,0.26*inten,0.06,att=0.5,cut=700)
    orb(CTB2,+sp*0.8,t0+dur*0.16,dur*0.7,0.2*inten,-0.15,cut=1200)
    # foghorn: old trombones, staggered, beating
    orb(TBN1,-sp*0.7,t0+dur*0.1,dur*0.72,0.3*inten,0.18,att=0.42)
    orb(TBN1,+sp*0.9,t0+dur*0.22,dur*0.62,0.27*inten,-0.2,att=0.42)
    if inten>0.6: orb(TBN2,+sp*0.5,t0+dur*0.3,dur*0.5,0.22*inten,0.3,att=0.4)
    # shell: tremolo strings, octave up, wider orbits
    orb(VLN_AB,-sp*1.2,t0+dur*0.2,dur*0.6,0.14*inten,0.4,att=0.4,cut=5200)
    orb(VLN_AB,+sp*1.4,t0+dur*0.3,dur*0.55,0.13*inten,-0.42,att=0.4,cut=5200)
    # the A-natural: outermost orbit, +100 cents — the dyad become a world
    if hasA:
        orb(TBN1,100-sp*0.2,t0+dur*0.26,dur*0.6,0.24*inten,-0.3,att=0.4)
        orb(VLN_TR,100+sp*0.3,t0+dur*0.36,dur*0.5,0.13*inten,0.46,att=0.4,cut=5200)
    orb_lean(TBN1,sp*1.3,t0+dur*0.15,dur*0.65,0.24*inten,0.08)
    # crest glints: harp and small bells at the breath's peak
    tc=t0+dur*0.46
    if bi==2:
        place(HARP_AB2,tc,0.34,-0.12)
        place(norm(resample(HARP_A2,2**(-5/12))),tc+0.02,0.28,0.12)   # Ab against D: the tritone glint
    elif bi==5:
        place(HARP_AB4,tc,0.3,-0.2)
        place(norm(resample(HARP_A4,2**(-0.5/12))),tc+0.02,0.26,0.2)  # a quarter-sharp lie
    elif bi%2==0:
        place(HARP_AB2,tc,0.34,-0.12); place(HARP_A2,tc+0.02,0.3,0.12)
    else:
        place(HARP_AB4,tc,0.3,-0.2); place(HARP_A4,tc+0.02,0.26,0.2)
    place(far(NEP[bi%5],0.5),tc+rng.uniform(4,7),0.22,(-1)**bi*0.35)
    # exhale marker: a pp gong swell dying into the next inhale
    if bi in (1,3): place(edge(norm(GONG_PP if bi==1 else GONG_W),0.4,3.0),t0+dur*0.8,0.24,(-1)**bi*0.12)

# ===== THE HOURS: six strikes, darker each =====
HOURS=[12.0,70.0,132.0,192.0,258.0,330.0,415.0]
for k,th in enumerate(HOURS):
    KN=lp(resample(BSTR[:int(6.0*SR)],0.5),2400-k*170)
    place(edge(norm(KN),0.02,1.8),th,0.38-0.014*k,0.0)

# ===== THE WORLD OUTSIDE: nice bells+birds, far, twice =====
place(edge(far(hp(seg(nice,58.0,76.0),300),0.5),3.0,4.0),98.0,0.14,0.42)
place(edge(far(hp(seg(nice,176.0,196.0),300),0.5),3.5,4.5),398.0,0.13,-0.42)

# ===== CHOPIN, THROUGH THE SPHERE (inside breath 4's exhale) =====
CHOPX=seg(chop,18.0,84.0)
CH1=gstretch(CHOPX,1.6,grain=0.11)
nnc=len(CH1); ttc=np.arange(nnc)/SR
cutp=np.interp(ttc,[0,20,26,34,40,nnc/SR],[1100,1200,4200,4200,1200,900])
loc=lp(CH1,1050); hic=lp(CH1,4400)
mixw=np.clip((cutp-1100)/3300,0,1)
CHSUB=loc*(1-mixw)+hic*mixw
cenv=np.interp(ttc,[0,8,22,30,44,60,nnc/SR],[0,0.2,0.26,0.3,0.24,0.14,0])
place(norm(CHSUB)*cenv,262.0,1.0,0.05)
def sungwin(y,lo=200,hi=2000,dur=20):
    v2b=bp(y,lo,hi); w2=SR*2
    rows2=[(i*2,float(np.mean(v2b[i*w2:(i+1)*w2]**2))) for i in range(4,len(y)//w2-int(dur/2))]
    return max(rows2,key=lambda r:r[1])[0]
efs=sungwin(ef)
EF_A=seg(ef,efs,efs+22); EF_B=seg(ef,min(len(ef)/SR-26,efs+40),min(len(ef)/SR-4,efs+62))
place(edge(far(lp(hp(EF_A,280),3400),0.45),2.5,3.0),382.0,0.28,0.1)   # the hymn, mid-exhale
place(edge(far(lp(hp(EF_B,280),3200),0.5),2.5,2.0),452.0,0.3,0.05)    # crossing into the unison
shs=sungwin(shen)
place(edge(far(lp(hp(seg(shen,shs,shs+14),280),3400),0.5),2.0,2.5),138.0,0.22,-0.24)
place(edge(far(lp(seg(sist,30.0,52.0),2000),0.6),3.0,3.5),252.0,0.16,0.18)  # ave maria, below hearing
# THE READINGS, in the water
oceanvoice(seg(shk,50.0,74.0),92.0,0.5,0.06)                    # shackleton: the log
oceanvoice(seg(freya,4.0,62.0),268.0,0.46,-0.08,surf_lo=800)    # conrad: the dream, over the waterflow
oceanvoice(seg(shk,186.0,214.0),336.0,0.52,0.08)                # shackleton: the pole passage
for tg,gg,pg in [(75.0,0.22,0.45),(148.0,0.2,-0.5),(246.0,0.26,0.38),(318.0,0.21,-0.42),(360.0,0.24,0.25),(388.0,0.19,0.5)]:
    gull(tg,gg,pg)
TH=drv(lp(TROLL,350),1.5)
for tt5,gt in [(172.0,0.2),(285.0,0.26),(402.0,0.15)]:
    place(edge(norm(TH[:int(11*SR)]),2.5,3.5),tt5,gt,rng.uniform(-0.2,0.2))
# THE PROTAGONIST: five notes, three appearances
def protagonist(t0,g=0.24):
    for cents,dt,dl in [(0,0,9),(300,8,8),(100,15.5,8),(-200,23,9),(0,31,11)]:
        orb(CTB1,cents,t0+dt,dl,g,-0.04,att=0.42,cut=850)
protagonist(30.0); protagonist(410.0,g=0.2)
# the boulanger violin, once, inside the contraction
vb=bp(boul,300,2600); w=SR*2
rows=[(i*2,float(np.mean(vb[i*w:(i+1)*w]**2))) for i in range(5,len(boul)//w-2)]
bstart=max(rows,key=lambda r:r[1])[0]
place(edge(far(lp(hp(seg(boul,bstart,bstart+20),240),4600),0.35),2.5,3.0),352.0,0.3,0.24)
place(edge(far(ALIEN,0.5),2.0,3.0),222.0,0.14,-0.28)

# ===== THE END: contraction to Ab, the torch out =====
place(HARP_AB2,468.0,0.32,-0.05)
place(HARP_AB2,482.0,0.26,-0.05)
place(norm(resample(HARP_AB6,1.0)),492.0,0.14,0.3)
place(far(NEP[2],0.6),503.0,0.15,-0.2)
orb(CTB1,0.0,442.0,42,0.2,-0.06,att=0.35,cut=600)   # the first unison:
orb(TBN1,0.0,444.0,38,0.17,0.1,att=0.4)              # nothing beats against anything
# THE DAWN: light above the unison
orb(VLN_AB,-3,478.0,34,0.1,0.34,att=0.5,cut=6000)
VLN_HI=norm(resample(VLN_AB,0.5))
orb(VLN_HI,+4,486.0,30,0.08,-0.36,att=0.5,cut=7000)
place(HARP_AB6,497.0,0.14,-0.25)
place(norm(resample(HARP_AB6,2**(2/1200))),508.0,0.11,0.3)
fadeN=int(12.0*SR); L[-fadeN:]*=np.linspace(1,0,fadeN)**1.2; R[-fadeN:]*=np.linspace(1,0,fadeN)**1.2
st=np.stack([L,R],1); st=norm(st)*10**(-1.0/20)
sf.write("/mnt/user-data/outputs/007_mvt3_v7.wav",st,SR,subtype="PCM_24")
print(f"NOTTURNO v7: {len(st)/SR:.1f}s | hour tails faded | fragments cut | water seams staggered | biology forward")
