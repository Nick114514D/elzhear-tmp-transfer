"""
007 — Movement II, impro c1: LA KERMESSE.
BPM 120 (BAR = 1.0 s). Free length ~5:40. The mixture, each given a round:
  S0 0:00 SLATE — the archivist reads catalog text; the machine assembles
  S1 0:14 MARIONETTE — the mock funeral becomes the carnival's melody
  S2 1:14 MAELSTROM — the manic riffs take the riff chair (alt rock round)
  S3 2:08 GROTESQUE — old trombone, trumpet, anvil, timpani (classical round)
  S4 2:52 READING — spoken text over skeleton; SIEGFRIED erupts, is cut off
  S5 3:24 ALL — every round at once; false stop; lurches tighten
  S6 4:36 NIGHT — stop; night bell; Yukhov choir far; the carnival is over
"""
import numpy as np, soundfile as sf, subprocess, os
from scipy.signal import butter, sosfilt, fftconvolve, find_peaks
SR=48000; rng=np.random.default_rng(1672)
def hp(x,f,o=2): return sosfilt(butter(o,f,btype='high',fs=SR,output='sos'),x)
def lp(x,f,o=2): return sosfilt(butter(o,min(f,SR/2-200),btype='low',fs=SR,output='sos'),x)
def bp(x,lo,hi,o=2): return sosfilt(butter(o,[lo,min(hi,SR/2-200)],btype='band',fs=SR,output='sos'),x)
def drv(x,a): return np.tanh(x*a)/np.tanh(a)
def midi_f(m): return 440.0*2**((m-69)/12)
def Nn(nm,o):
    T={'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
    return 12*(o+1)+T[nm]
def norm(y): return y/(np.max(np.abs(y))+1e-9)
def declick(y):
    w=int(0.0015*SR); y[:w]*=np.linspace(0,1,w); y[-w:]*=np.linspace(1,0,w); return y
def resample(y,rate):
    nn=max(4,int(len(y)/rate)); idx=np.arange(nn)*rate
    i0=np.clip(idx.astype(int),0,len(y)-2); fr=idx-i0
    return y[i0]*(1-fr)+y[i0+1]*fr
def spiral(y,factor,grain=0.045,semis_start=0.0,semis_end=-5.0):
    g=int(grain*SR); hin=int(g*0.5); hout=int(hin*factor)
    nout=int(len(y)*factor)+g; out=np.zeros(nout); win=np.hanning(g)
    pos=0; opos=0
    while pos+int(g*1.8)<len(y) and opos+g<nout:
        prog=opos/max(1,nout)
        rt=2**((semis_start+(semis_end-semis_start)*prog)/12.0)
        src=y[pos:pos+max(4,int(g*rt)+2)]
        gr=resample(src,rt)[:g]
        if len(gr)<g: gr=np.pad(gr,(0,g-len(gr)))
        out[opos:opos+g]+=gr*win
        pos+=hin; opos+=hout
    return norm(out[:int(len(y)*factor)])
def gstretch(y,factor,grain=0.09):
    g=int(grain*SR); hin=int(g*0.5); hout=int(hin*factor)
    nn=int(len(y)*factor)+g; out=np.zeros(nn); win=np.hanning(g); pos=0; opos=0
    while pos+g<len(y) and opos+g<nn:
        out[opos:opos+g]+=y[pos:pos+g]*win
        pos+=hin; opos+=hout
    return norm(out[:int(len(y)*factor)])
def edge(y,fi=0.5,fo=0.5):
    y=y.copy(); a=int(fi*SR); b=int(fo*SR)
    if a>0: y[:a]*=np.linspace(0,1,a)
    if b>0: y[-b:]*=np.linspace(1,0,b)
    return y

D="/home/claude/007/"
def load48(path,tag):
    dst=D+f"alt48/{tag}.wav"
    if not os.path.exists(dst):
        subprocess.run(["ffmpeg","-y","-i",path,"-ar","48000","-ac","1",dst],capture_output=True)
    y,_=sf.read(dst); return y
def seg(y,a,b): return norm(y[int(a*SR):int(b*SR)].copy())

m1=load48(D+"altrock/pack/CC0__01-Maelstrom-InAPaintedBlackWorld.mp3","m1")
m2=load48(D+"altrock/pack/CC0__02-Maelstrom-IDreamtForABrighterSky.mp3","m2")
m3=load48(D+"altrock/pack/CC0__04-Maelstrom-...untilTheRestOfMyLife.mp3","m3")
mario=load48(D+"pdmusic/marionette_funeral_1919.mp3","mario")
sieg=load48(D+"pdmusic/siegfried_funeral_1920.mp3","sieg")
yuk=load48(D+"pdmusic/yukhov_choir_1911.mp3","yukhov")
lomax2=sf.read(D+"dewitt/lomax2.wav")[0]
ybr,_=sf.read(D+"dewitt/ybr.wav")
lomax,_=sf.read(D+"dewitt/lomax39.wav")
berg,_=sf.read(D+"dewitt/berg.wav")
lomax39=sf.read(D+"dewitt/lomax39.wav")[0]
chaF,_=sf.read(D+"dewitt/chaliapin__Boris_Godounow-In_the_Town_of_Kaz_n_-_Feodor_Chaliapin.wav")
bells,_=sf.read(D+"dewitt/bells__clockstrikesnoonfrenchquarterNOLA9512.wav")
morF,_=sf.read(D+"dewitt/moreschi___08-G_T_54764.wav")
wil,_=sf.read(D+"dewitt/willie.wav")

V=D+"vsco_pack/"
def vs(path):
    path=path.replace(' ','_')
    y=load48(V+path,os.path.basename(path).replace('.wav','').replace('#','s'))
    return norm(y)
TBN_A1=vs("Brass/OldTrombone/Sustain/Trombone_Sustain_A1_v3_1.wav")
TBN_A2=vs("Brass/OldTrombone/Sustain/Trombone_Sustain_A2_v3_1.wav")
TBN_BZ=[vs("Brass/OldTrombone/Buzz/Trombone_Buzz_A1_v1_1.wav"),vs("Brass/OldTrombone/Buzz/Trombone_Buzz_A2_v1_1.wav")]
TBN_FL=[vs("Brass/OldTrombone/Fall/Trombone_Fall_A1_1.wav"),vs("Brass/OldTrombone/Fall/Trombone_Fall_A2_1.wav"),vs("Brass/OldTrombone/Fall/Trombone_Fall_A#2_1.wav")]
TPT_ST=[vs("Brass/Trumpet/stac/Sum_SHTrumpet_stac_A2_v1_rr1.wav"),vs("Brass/Trumpet/stac/Sum_SHTrumpet_stac_A2_v2_rr1.wav"),vs("Brass/Trumpet/stac/Sum_SHTrumpet_stac_A2_v3_rr1.wav")]
ANV=[vs("Percussion/Anvil_Hit1_v1_Sum.wav"),vs("Percussion/Anvil_Hit1_v2_Sum.wav"),vs("Percussion/Anvil_Hit1_v3_Sum.wav")]
TIMP=vs("Percussion/Timpani/Timpani1_Hit_v3_rr1_Sum.wav")
TROLL=vs("Percussion/Timpani/Rolls/Timpani3_Roll_v3_rr1_Sum.wav")
BDR=vs("Percussion/bassdrum_rub2_v1.wav")
VLN_TR=vs("Strings/Violin_Section/Trem/VlnEns_Trem_A2_v1.wav") if os.path.exists(V+"Strings/Violin_Section/Trem/VlnEns_Trem_A2_v1.wav") else vs("Strings/Violin Section/Trem/VlnEns_Trem_A2_v1.wav")
CTB_G1=vs("Strings/Solo Contrabass/SusNV/BKCtbss_SusNV_G#1_v1_rr1.wav")

dn=2**(-1/12)   # A -> Ab
TBN_Ab1=norm(resample(TBN_A1,1/dn)); TBN_Ab2=norm(resample(TBN_A2,1/dn))
TPT_Ab=[norm(resample(t,1/dn)) for t in TPT_ST]

# found regions
M1R=seg(m1,180.0,192.0); M1R2=seg(m1,148.0,160.0)
M2R=seg(m2,224.0,238.0); M2C=seg(m2,286.0,296.0)
M3R=seg(m3,198.0,210.0)
MARIO_A=seg(mario,4.0,20.0); MARIO_B=seg(mario,104.0,118.0)
SIEG_E=seg(sieg,236.0,252.0)
YUK=seg(yuk,52.0,72.0)
SLATE1=seg(lomax39,0.0,8.0)
SLATE2=seg(lomax2,1.5,12.0)
READ=seg(lomax2,224.0,242.0)
BARK=seg(chaF,144.6,145.15); LAUGH=seg(chaF,145.0,145.35)
BSTR=seg(bells,66.0,80.0)
WHUM=seg(wil,162.0,174.0)

def quietest(y,win=2.0):
    w=int(win*SR); best=None; bi=0
    for a in range(0,len(y)-w,w//2):
        e=float(np.mean(y[a:a+w]**2))
        if best is None or e<best: best=e; bi=a
    return y[bi:bi+w].copy()
CRACK=norm(np.concatenate([quietest(morF),quietest(chaF)]))
hpm=hp(morF,3000)
pk2,_=find_peaks(np.abs(hpm),height=np.max(np.abs(hpm))*0.25,distance=int(0.05*SR))
POPS=[norm(hpm[max(0,i-96):max(0,i-96)+768].copy())*np.hanning(768) for i in pk2[:300] if i+700<len(hpm)]
def pop(): return POPS[int(rng.integers(0,len(POPS)))]

def detect_f0(y,lo,hi):
    w=y*np.hanning(len(y))
    sp=np.abs(np.fft.rfft(w,n=1<<int(np.ceil(np.log2(len(w))+1))))
    fr=np.fft.rfftfreq(2*(len(sp)-1),1/SR)
    m=(fr>=lo)&(fr<=hi); i=int(np.argmax(sp*m))
    return fr[i]
fB=detect_f0(BARK,80,600)
fAb2=midi_f(Nn('G#',2)); fA2=midi_f(Nn('A',2)); fAb1=midi_f(Nn('G#',1))
r=fAb2/fB
while r>1.6: r/=2
while r<0.55: r*=2
CHANT=norm(drv(bp(resample(BARK,r),120,3800),1.8))

def ironknock(freq,dur,amp):
    n=int(dur*SR); t=np.arange(n)/SR; y=np.zeros(n)
    for r_,a,dc in [(1.0,1.0,0.45),(2.76,0.6,0.18),(4.9,0.4,0.11),(7.7,0.28,0.07),(11.3,0.16,0.05)]:
        if freq*r_<SR/2-500: y+=a*np.sin(2*np.pi*freq*r_*t+rng.uniform(0,6))*np.exp(-t/(dur*dc))
    clang=bp(rng.standard_normal(n),freq*3,min(SR/2-500,freq*15))*np.exp(-t/0.005)*0.7
    y=np.tanh((y+clang)*1.9)*np.minimum(1,t/0.0012)
    return declick(norm(hp(y,freq*0.7)))*amp
def fmpiston(freq,dur,amp):
    n=int(dur*SR); t=np.arange(n)/SR
    mod=np.sin(2*np.pi*freq*t)*2.6*np.exp(-t/0.05)
    car=np.sin(2*np.pi*freq*t+mod)
    knock=bp(rng.standard_normal(n),freq*3,min(SR/2-500,freq*9))*np.exp(-t/0.004)*0.45
    env=np.exp(-t/(dur*0.4))*np.minimum(1,t/0.0018)
    return declick(norm(np.tanh((car+knock)*1.5)*env))*amp
def strike_metal(freq,dur,amp):
    n=int(dur*SR); t=np.arange(n)/SR
    atk=bp(rng.standard_normal(n),freq*5,min(SR/2-500,freq*40))*np.exp(-t/0.006)*0.9
    for r_ in (4.2,5.4,6.8,8.1,9.5,11.7): atk+=np.sin(2*np.pi*freq*r_*t+rng.uniform(0,6))*np.exp(-t/0.012)
    body=np.zeros(n)
    for r_ in (1.0,2.0,2.76,3.52,4.07,5.4,6.8,8.2,10.5):
        if freq*r_<SR/2-500: body+=(1/(1+r_*0.35))*np.sin(2*np.pi*freq*r_*t*(1+0.0025*rng.uniform(-1,1))+rng.uniform(0,6))*np.exp(-t/(dur*(0.55/(1+0.3*r_))))
    y=np.tanh((body+atk*0.7)*1.7)*np.minimum(1,t/0.002)
    return declick(norm(hp(lp(y,13000),freq*0.8)))*amp
def stompkick(amp=1.0):
    nn=int(0.28*SR); tt=np.arange(nn)/SR
    y=(np.sin(2*np.pi*84*tt)+0.4*np.sin(2*np.pi*168*tt))*np.exp(-tt/0.14)
    a=int(0.004*SR); y[:a]*=np.linspace(0,1,a)
    return drv(y,2.4)*amp
def bbsnare():
    nn=int(0.16*SR); tt=np.arange(nn)/SR
    tone=(np.sin(2*np.pi*196*tt)+np.sin(2*np.pi*271*tt))*np.exp(-tt/0.035)
    nz=hp(rng.standard_normal(nn),1800)*np.exp(-tt/0.05)
    q=12.0; y=np.round((tone*0.5+nz*1.1)*q)/q
    return drv(y,3.2)
def cclick():
    n=int(0.012*SR)
    return norm(hp(rng.standard_normal(n),4500)*np.exp(-np.arange(n)/(0.002*SR)))
def cping(fq=3100.0,dur=0.05):
    tt=np.arange(int(dur*SR))/SR
    return norm(np.sin(2*np.pi*fq*tt)*np.exp(-tt/(dur*0.25)))
def sawbass(f,dur,drvA=2.8):
    nn=int(dur*SR); tt=np.arange(nn)/SR
    y=2*((f*tt)%1.0)-1+0.5*(2*((f*1.005*tt)%1.0)-1)
    y=lp(y,f*7)
    e=np.ones(nn); a=int(0.006*SR); rl=int(0.05*SR)
    e[:a]=np.linspace(0,1,a); e[-rl:]*=np.linspace(1,0,rl)
    return drv(y*e,drvA)
def sizzle_long(src,fc_path=(2800,4200,3000)):
    y=src.copy(); nn=len(y); tt=np.arange(nn)/SR
    fc=np.interp(tt,[0,tt[-1]*0.5,tt[-1]],fc_path)
    rm=hp(y*np.sin(2*np.pi*np.cumsum(fc)/SR),2300)
    envf=lp(np.abs(y),8.0); envf/=(np.max(envf)+1e-9)
    crk=np.tile(CRACK,nn//len(CRACK)+2)[:nn]
    burn=hp(drv(drv(rm*3.0,4.0)+crk*envf*1.3,2.5),1900)
    e=np.ones(nn); w=int(0.4*SR); e[:w]=np.linspace(0,1,w); e[-w:]*=np.linspace(1,0,w)
    return norm(burn*e)

BPM=120.0; BEAT=0.5; BAR=1.0; g8=BAR/8
DUR=288.0; TOT=int(DUR*SR); L=np.zeros(TOT); R=np.zeros(TOT)
def place(sig,t,g=1.0,pan=0.0):
    i=int(t*SR); e=min(TOT,i+len(sig))
    if e<=i: return
    gl,gr=np.sqrt(0.5-pan/2),np.sqrt(0.5+pan/2)
    L[i:e]+=sig[:e-i]*g*gl; R[i:e]+=sig[:e-i]*g*gr
CUTS=[]
reps=TOT//len(CRACK)+2
bedy=np.tile(CRACK,reps)[:TOT].copy()
bedy[:int(1.5*SR)]*=np.linspace(0,1,int(1.5*SR)); bedy[-int(3*SR):]*=np.linspace(1,0,int(3*SR))
L+=bedy*10**(-38/20); R+=np.roll(bedy,int(0.013*SR))*10**(-38/20)

RIFFS={0:[fAb2,fAb2,fA2,fAb2],1:[fAb2,fA2,fAb2,fAb2],4:[fAb2,fA2,fA2,fAb2]}
VAR={'A':(0,3,6),'B':(0,3,6,7),'C':(0,2,3,6)}
VROT=['A','A','B','A','A','B','C','A']
def stomp_bar(t,bar,riff,density=1.0,harsh=1.0,chant=True,cold=True,snare_all=False,lurch=False,bass=0.4):
    hits=VAR[VROT[bar%8]]
    place(stompkick(),t,0.9*density)
    for p in hits[1:]:
        if p==3: place(ironknock(196,0.3,1.0),t+p*g8,0.5*density)
        elif p==6: place(fmpiston(82,0.4,1.0),t+p*g8,0.6*density)
        else: place(ironknock(124,0.25,1.0),t+p*g8,0.42*density)
    if snare_all or bar%2==1: place(bbsnare(),t+4*g8,0.66*density)
    if snare_all and bar%2==0: place(bbsnare(),t+7*g8,0.4*density)
    if bass>0: place(sawbass(riff[bar%4],BAR*0.8,drvA=2.8*harsh),t,bass*density)
    if chant and bar%4==0:
        ch=CHANT if bar%8==0 else resample(CHANT,2**(-3/12))
        place(norm(ch),t,0.48*density,0.12)
    if cold:
        place(cclick(),t+3*g8,0.38,0.35); place(cclick(),t+7*g8,0.34,-0.35)
        if bar%2==0: place(cping(3100),t+6*g8,0.2,0.2)
    if bar%8==0 and density>0.9: place(strike_metal(148,0.7,1.0),t,0.42)
    return t+(BAR-g8 if lurch else BAR)

def chop(y,n_cells):
    c=int(len(y)/n_cells)
    out=[y[i*c:(i+1)*c].copy() for i in range(n_cells)]
    for cc in out: declick(cc)
    return out
MAR_C=chop(hp(lp(MARIO_A,5200),250),16)          # 1s cells at BAR
MARB_C=chop(hp(lp(MARIO_B,5200),250),14)
MLC=chop(drv(norm(M1R),1.6),12)+chop(drv(norm(M3R),1.6),12)
ML2=chop(drv(norm(M2R),1.7),14)
GAUZE=lp(gstretch(M1R,2.2,grain=0.12),1200)

MSEQ=[0,1,2,3, 0,1,4,5, 8,9,2,3, 12,13,6,7]
def throbline(t0,t1,g0,g1,f=103.8):
    t=t0; k=0
    while t<t1-0.1:
        prog=(t-t0)/max(0.001,(t1-t0)); g=g0+(g1-g0)*prog
        nn=int(0.4*SR); tt=np.arange(nn)/SR
        y=lp(drv(np.sign(np.sin(2*np.pi*f*tt))*np.exp(-tt/0.16),2.6),700)
        place(norm(y)*0.9,t,g)
        if k%2==1:
            n2=int(0.08*SR)
            cl=bp(rng.standard_normal(n2),1500,6000)*np.exp(-np.arange(n2)/(0.012*SR))
            place(norm(cl),t+0.5,g*0.5,0.2*(-1)**(k//2))
        t+=0.5; k+=1
def sharpenv(v):
    v=norm(hp(v,200)); return norm(drv(v*0.75+bp(v,1100,3200)*0.8,1.2))
L2SUNG1=seg(lomax2,30.0,44.0)
L2SUNG2=seg(lomax2,132.0,142.0)
L2SPK=seg(lomax2,224.0,232.0)

RIFFX=seg(m3,198.0,206.0)
def industrial(v,q=10.0,drvA=2.2):
    y=bp(norm(v),300,4500)
    y=np.round(y*q)/q
    return norm(drv(y,drvA))
RX=industrial(RIFFX); RI=industrial(RIFFX[::-1].copy())
def chop2(y,n):
    c=int(len(y)/n); out=[y[i*c:(i+1)*c].copy() for i in range(n)]
    for cc in out: declick(cc)
    return out
RX_C=chop2(RX,8); RI_C=chop2(RI,8)
irH_n=int(1.4*SR)
irH=hp(rng.standard_normal(irH_n)*np.exp(-np.arange(irH_n)/SR*3.0),200)
irH[:150]*=np.linspace(0,1,150); irH/=(np.max(np.abs(irH))+1e-9)
def grindhouse(v):
    y=sharpenv(v)
    ch2=int(0.4*SR); out=[]
    for i2 in range(0,len(y),ch2):
        rt=1.0+0.0045*np.sin(2*np.pi*0.6*(i2/SR)+1.3)
        out.append(resample(y[i2:i2+ch2],rt))
    y=drv(norm(np.concatenate(out)),1.9)
    w=fftconvolve(y,irH)[:len(y)]
    tailw=np.interp(np.arange(len(y))/SR,[0,max(0.1,len(y)/SR-1.2),len(y)/SR],[0.12,0.18,0.5])
    return norm(y*0.82+norm(w)*tailw)
THUD=lp(resample(TIMP,1.4),260)
def horrify(v,st,drvA,q):
    y=norm(resample(v,2**(-st/12)))
    y=bp(y,250,3400); y=np.round(y*q)/q; y=drv(y,drvA)
    w=fftconvolve(y,irH)[:len(y)]
    return norm(y*0.7+norm(w)*0.35)
YBRP=seg(ybr,38.0,46.0); YBRW=seg(ybr,40.0,41.2)

bar=0
# ===== T0 (0-0:20) SLATE & PIT =====
place(grindhouse(SLATE1),1.0,0.72,0.0)
place(norm(THUD),0.92,0.42)
place(TIMP,1.2,0.4); place(edge(CTB_G1,0.8,1.5),2.2,0.34,-0.1)
place(TBN_BZ[0],9.4,0.4,0.3)
place(edge(grindhouse(SLATE2)[:int(7*SR)],0.2,1.2),12.0,0.62,0.22)
place(norm(THUD),11.92,0.3)
place(ANV[1],17.5,0.36,-0.2)
# ===== T1 (0:20-0:48) THE BALLAD STOMP — 28 seconds, once =====
t=20.0
for b in range(10):
    tb=t
    hits=VAR[VROT[b%8]]
    place(stompkick(),tb,0.9)
    for p in hits[1:]:
        if p==3: place(ironknock(196,0.3,1.0),tb+p*g8,0.5)
        elif p==6: place(fmpiston(82,0.4,1.0),tb+p*g8,0.6)
        else: place(ironknock(124,0.25,1.0),tb+p*g8,0.42)
    if b%2==1: place(bbsnare(),tb+4*g8,0.62)
    place(sawbass(RIFFS[0][b%4],BAR*0.8,2.8),tb,0.36)
    if b%8==0: place(norm(CHANT),tb,0.42,0.12)
    place(cclick(),tb+3*g8,0.34,0.35); place(cclick(),tb+7*g8,0.3,-0.35)
    if b%2==1: place(ANV[b%3],tb+7*g8,0.3,-0.22)
    t+=BAR
place(edge(sharpenv(L2SUNG1),0.1,2.0),21.0,0.56,-0.08)   # the singer floats, then outlives the machine
CUTS.append((t-0.35,0.4))                          # we walk past the stall
# ===== T2 THE CAROUSEL — continuous, warping, dying =====
t=t+0.1; T2=t; PROC=50.0
rates=[1.0,0.985,1.01,0.972,0.996,0.955,0.978,0.94]
pos=4.0; tc2=T2+1.0
for k,rt in enumerate(rates):
    ch=seg(mario,pos,pos+4.5); pos+=4.0
    y=edge(norm(resample(hp(lp(ch,5200),250),rt)),0.7,0.9)
    place(y*0.9,tc2,0.44,0.12)
    tc2+=len(y)/SR-0.8
GRIND=lp(gstretch(RX,2.5,grain=0.07),1600)
GRINDR=lp(gstretch(RI,2.5,grain=0.07),1600)
place(edge(GRIND,2.0,1.0),T2+4.0,0.2,-0.2)
place(edge(GRINDR,1.0,3.0),T2+4.0+len(GRIND)/SR-1.5,0.22,0.2)
def bergvoice(a2,b2):
    y=norm(hp(lp(seg(berg,a2,b2),8000),230))
    return norm(y+bp(y,900,2600)*0.5)
BGF1=edge(bergvoice(408.0,430.0),1.2,2.0)
place(BGF1,T2+6.0,0.4,-0.18)
BGF2=edge(bergvoice(430.0,450.0),1.5,2.5)
place(BGF2,T2+27.0,0.38,-0.2)
MELT=lp(spiral(seg(mario,20.0,26.0),1.6,grain=0.05,semis_start=0.0,semis_end=-5.0),3000)
place(edge(norm(MELT)*0.4,1.0,1.5),T2+34.0,1.0,0.0)
tb2=T2; k=0
while tb2<T2+PROC-0.5:
    place(cclick(),tb2+0.375,0.26,0.35); place(cclick(),tb2+0.875,0.22,-0.35)
    if k%4==0: place(TIMP,tb2,0.36)
    if k%4==2: place(ANV[k%3],tb2+0.6,0.3,-0.22)
    if k%8==0: place(edge(norm(TBN_Ab1 if k%16==0 else TBN_Ab2),0.05,0.8),tb2,0.34,0.1)
    if k%8==7: place(TBN_FL[k%3],tb2+0.7,0.32,0.3)
    if k%4==0 and (k//4)%2==1: place(TPT_Ab[k%3],tb2+0.5,0.3,-0.18)
    if k==30:
        place(grindhouse(L2SPK),tb2,0.62,0.2); place(norm(THUD),tb2-0.08,0.3)
    tb2+=1.0; k+=1
t=T2+PROC
UND1=lp(spiral(seg(chaF,100.0,110.0),3.0,grain=0.05,semis_start=0.0,semis_end=-4.0),2200)
UND2=lp(spiral(seg(chaF,110.0,117.0),3.6,grain=0.05,semis_start=-4.0,semis_end=-7.0),2000)
place(UND1*np.linspace(0.5,1.0,len(UND1)),T2+4.0,0.22,-0.05)
place(UND2,T2+4.0+len(UND1)/SR-2.0,0.26,-0.05)
# ===== T3 THE FUSE =====
TF=t
BURNSRC=np.concatenate([seg(chaF,100.0,117.0),seg(m2,222.0,238.0),seg(chaF,137.5,152.0),seg(m3,196.0,210.0)])
BURN=sizzle_long(BURNSRC,(2400,4400,3400))
benv=np.interp(np.arange(len(BURN))/SR,[0,18,40,52,56],[0.05,0.16,0.42,0.68,0.86])
place(BURN*benv,TF,1.0,0.08)
BURNLOW=sizzle_long(BURNSRC,(1200,2000,1500))
BURNLOW=lp(hp(BURNLOW,900),3200)
place(norm(BURNLOW)*benv*0.75,TF,1.0,-0.06)
YSPIR=lp(spiral(YUK[:int(9*SR)],2.4,grain=0.06,semis_start=0.0,semis_end=-5.0),2600)
place(YSPIR*np.interp(np.arange(len(YSPIR))/SR,[0,4,18,21.6],[0,0.26,0.3,0.1]),TF+28.0,1.0,0.18)
FUSE=56.0
tk=TF
while tk<TF+18.0:                                  # the cold ticks fade out of the world
    gg=0.26*(1-(tk-TF)/18.0)
    place(cclick(),tk+0.375,gg,0.35); place(cclick(),tk+0.875,gg*0.9,-0.35)
    tk+=1.0
place(edge(norm(VLN_TR),1.5,2.0),TF+10.0,0.2,0.3)
place(edge(norm(resample(VLN_TR,0.5)),1.5,2.0),TF+24.0,0.18,-0.3)
place(edge(norm(VLN_TR),1.0,1.5),TF+38.0,0.24,0.25)
place(edge(norm(resample(VLN_TR,0.5)),1.0,1.0),TF+46.0,0.22,-0.25)
place(edge(norm(TROLL[:int(14*SR)]),6.0,0.5),TF+40.0,0.32)
place(edge(CTB_G1,1.0,1.5),TF+8.0,0.3,-0.1)
place(edge(CTB_G1,1.0,1.5),TF+28.0,0.3,-0.08)
place(edge(CTB_G1,0.8,0.5),TF+46.0,0.36,-0.05)
place(TBN_BZ[0],TF+16.0,0.32,0.28); place(TBN_BZ[1],TF+33.0,0.34,0.3)
place(grindhouse(READ),TF+22.0,0.8,0.0)
place(norm(THUD),TF+21.9,0.44)
place(edge(lp(resample(BDR,1.0),800),1.0,1.5),TF+36.0,0.34)
place(TIMP,TF+52.0,0.4)
def slabfix(v):
    y=hp(norm(v),170)
    y=y-0.3*bp(y,240,470)
    r2=np.abs(y); r2=r2-np.mean(r2)                 # octave fuzz
    y=drv(y*0.75+r2*0.45,2.4)
    y=y+bp(y,1400,4200)*0.5
    tt3=np.arange(len(y))/SR
    fc=np.interp(tt3,[0,tt3[-1]],[2600,3600])
    sz=hp(y*np.sin(2*np.pi*np.cumsum(fc)/SR),2000)*0.2
    return norm(y+sz)
# ===== T4 THE OUTBURST — riffs + orchestra only, no kit =====
TX=TF+FUSE
SLAB1=edge(slabfix(seg(m2,286.0,296.0)),0.03,0.4)
SLAB1=SLAB1*np.interp(np.arange(len(SLAB1))/SR,[0,0.6,1.2],[0.5,1.0,1.0])
SLAB2=edge(drv(norm(hp(seg(m1,178.0,191.0),120)),1.7),0.03,1.0)
SLAB3=edge(slabfix(seg(m3,198.0,210.0)),0.05,1.5)
SLAB2S=lp(spiral(seg(m1,178.0,191.0),1.0,grain=0.05,semis_start=0.0,semis_end=-4.0),5200)
SLAB2S=edge(slabfix(SLAB2S),0.03,1.0)
place(SLAB1,TX,0.7,0.15); place(SLAB2S,TX+0.4,0.56,-0.2)
PED=drv(norm(CTB_G1),2.0)
place(edge(PED,0.1,0.8),TX,0.42,-0.05); place(edge(PED,0.1,1.2),TX+7.0,0.42,-0.05); place(edge(PED,0.1,1.5),TX+14.5,0.4,-0.05)
place(SLAB3,TX+9.5,0.66,0.1)
SPIRE=edge(norm(hp(seg(berg,452.0,462.0),500)),1.5,1.5)
place(SPIRE,TX+8.0,0.36,0.12)                        # the Berg clarinet crowns the tower
FIDS=edge(drv(norm(hp(resample(seg(lomax,160.0,173.0),2**(5/12)),400)),1.4),0.5,1.5)
place(FIDS,TX+9.0,0.38,-0.28)
for dt,g2 in [(0.0,0.5),(2.6,0.42),(5.3,0.46),(8.1,0.42),(10.6,0.5),(13.4,0.44),(16.2,0.48),(19.0,0.5)]:
    place(TIMP,TX+dt,g2)
for dt,k2 in [(1.3,0),(4.0,2),(6.7,1),(9.4,0),(12.1,2),(14.8,1),(17.6,0),(20.3,2)]:
    place(ANV[k2],TX+dt,0.42,(-1)**k2*0.25)
place(TBN_FL[0],TX+3.4,0.4,0.3); place(TBN_FL[2],TX+11.2,0.4,-0.3); place(TBN_FL[1],TX+18.4,0.42,0.3)
place(norm(drv(bp(resample(BARK,r*0.5),120,3000),3.0)),TX+6.0,0.48,-0.15)
TC=TX+22.5
CUTS.append((TC,0.9))
DEBRIS=lp(spiral(seg(m2,290.0,293.0),1.0,grain=0.04,semis_start=-2.0,semis_end=-9.0),3400)
place(edge(norm(DEBRIS)*0.32,0.02,1.8),TC+0.92,1.0,0.12)   # the rubble lands after the cut
# ===== T5 SIEGFRIED UNOPPOSED =====
TS=TC+0.9
SGU=edge(norm(hp(seg(sieg,228.0,252.0),160)),0.8,3.5)
SGU=SGU*np.interp(np.arange(len(SGU))/SR,[0,6,18,24],[0.42,0.6,0.66,0.5])
place(SGU,TS,1.0,0.0)
place(TIMP,TS+4.0,0.26); place(TIMP,TS+14.0,0.24)
place(edge(CTB_G1,1.0,2.0),TS+2.0,0.26,-0.1)
place(edge(norm(VLN_TR),2.0,3.0),TS+10.0,0.14,0.3)
# ===== T5b THE EMBER — the machinery heard once more, far off =====
EMB=TS+24.0
BURN2=sizzle_long(np.concatenate([seg(chaF,152.0,165.0),seg(m1,316.0,326.0)]),(2200,3600,2600))
per=1.15                                            # 52 BPM
CLK=[]
for k in range(17):
    tk2=EMB+1.0+k*per; CLK.append(tk2)
    place(ANV[k%3],tk2,0.34+0.02*k,(-1)**k*0.15)
    place(ironknock(98.0,0.3,1.0),tk2,0.26+0.015*k)
tt2=np.arange(len(BURN2))/SR
base=np.interp(tt2,[0,6,14,20,23],[0.02,0.12,0.26,0.32,0.12])
bump=np.zeros(len(BURN2))
for tk2 in CLK:
    dt=tk2-(EMB+1.0)
    if dt<tt2[-1]:
        bump+=0.14*np.exp(-np.maximum(0,tt2-dt)/0.9)*(tt2>=dt)
place(BURN2*np.clip(base+bump,0,0.5),EMB+1.0,1.0,-0.1)
THREN=lp(hp(gstretch(seg(lomax,160.0,176.0),2.05,grain=0.11),300),4500)
tenv=np.interp(np.arange(len(THREN))/SR,[0,5,17,28,32.8],[0.0,0.18,0.26,0.22,0.0])
place(THREN*tenv,EMB+3.5,1.0,0.22)
place(horrify(YBRP,2,1.5,14.0),EMB+6.0,0.5,0.05)
place(edge(sharpenv(YBRP),0.7,0.9),EMB+10.2,0.42,0.03)      # the choir, nearly plain
place(edge(horrify(YBRP,2,1.4,16.0),0.7,0.7),EMB+14.6,0.44,0.03)  # once more, going wrong
for j in range(3):
    stt=horrify(YBRW,5,2.0,9.0)
    if j==0: stt=lp(stt,1400)
    place(edge(norm(stt),0.18,0.25),EMB+19.6+j*0.28,0.3+0.05*j,(-1)**j*0.2)
L2SPKB=seg(lomax2,240.5,244.0)
place(edge(grindhouse(L2SPKB),1.0,0.4),EMB+10.6,0.55,0.0); place(norm(THUD),EMB+17.3,0.2)
ST2=horrify(YBRP,5,2.2,8.0)
ST2lo=lp(ST2,900)
wop=np.interp(np.arange(len(ST2))/SR,[0,1.4],[1.0,0.0])
ST2=ST2lo*wop+ST2*(1-wop)
place(edge(norm(ST2),0.8,0.1),EMB+20.8,0.46,0.05)
place(horrify(YBRP,8,3.0,5.0),EMB+27.0,0.6,0.05)
DEB2=lp(spiral(seg(ybr,44.0,46.0),1.2,grain=0.04,semis_start=0.0,semis_end=-6.0),2800)
place(edge(norm(DEB2)*0.3,0.05,1.5),EMB+34.0,1.0,0.1)
# ===== T6 THE NIGHT =====
# ===== T6 THE NIGHT =====
STOP=EMB+39.0
KN=lp(resample(BSTR[:int(6.0*SR)],0.5),2400)
place(norm(KN),STOP+0.5,0.42,0.0)
place(edge(norm(hp(YUK,300)),3.0,4.5),STOP+2.5,0.34,0.15)
place(edge(norm(VLN_TR),1.5,3.0),STOP+9.0,0.16,-0.2)
gh=lp(gstretch(WHUM[:int(5*SR)],4.0),1800)
place(gh,STOP+20.0,0.22,0.05)
place(edge(norm(seg(berg,268.0,276.0)),1.0,2.5),STOP+23.0,0.28,-0.08)
place(norm(KN)*0.7,STOP+28.0,0.4,0.0)
place(edge(CTB_G1,1.0,3.0),STOP+37.0,0.26,-0.05)
for (tc,dc) in CUTS:
    a=int(tc*SR); b2=int((tc+dc)*SR); dn2=int(0.0015*SR)
    L[a-dn2:a]*=np.linspace(1,0,dn2); R[a-dn2:a]*=np.linspace(1,0,dn2)
    L[a:b2]=0.0; R[a:b2]=0.0
    L[b2:b2+dn2]*=np.linspace(0,1,dn2); R[b2:b2+dn2]*=np.linspace(0,1,dn2)
fadeN=int(5.0*SR); L[-fadeN:]*=np.linspace(1,0,fadeN)**1.2; R[-fadeN:]*=np.linspace(1,0,fadeN)**1.2
st=np.stack([L,R],1); st=norm(st)*10**(-1.0/20)
sf.write("/mnt/user-data/outputs/007_mvt2_n12.wav",st,SR,subtype="PCM_24")
print(f"n12: {len(st)/SR:.1f}s | stutter cluster softened: ascending, faded, first behind the filter")
