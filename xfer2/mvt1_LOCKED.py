"""
007 — impro II: LE SUPPLICE (passacaglia).
One ground, stated at the start and never absent: four bars of Ab, four
bars of A, the ironknock of 006/III on the step, the coil pulse beneath.
Verses pass over it. The coda strips the ground to the knock alone.
  GROUND ASSEMBLES  0:00   knock alone; pulse joins; sub joins; swell joins
  V1  WILLIE        0:17   his record sings over the ground, 2 loops
  V2  THE BODY      0:51   Moreschi; the loved ghost; Willie answers
  V3  THE SONG      1:26   13's soprano and piano over a brightened ground
  V4  THE BURN      2:00   the sizzle subtheme: his singing + harsh coil
  V5  THE MOB       2:34   Kazan, laughter, flood; the falls, the climb,
                           the outburst, the tone; ground at obsession
  GLIDE / STOP      3:14   her Ab->A; hard stop
  CODA              3:18   the ground survives as knock alone; his hum;
                           piano ghost; Ab. No bells after the stop.
"""
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve, find_peaks
SR=48000; rng=np.random.default_rng(672)
def hp(x,f,o=2): return sosfilt(butter(o,f,btype='high',fs=SR,output='sos'),x)
def lp(x,f,o=2): return sosfilt(butter(o,min(f,SR/2-200),btype='low',fs=SR,output='sos'),x)
def bp(x,lo,hi,o=2): return sosfilt(butter(o,[lo,min(hi,SR/2-200)],btype='band',fs=SR,output='sos'),x)
def drv(x,a): return np.tanh(x*a)/np.tanh(a)
def midi_f(m): return 440.0*2**((m-69)/12)
def Nn(nm,o):
    T={'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
    return 12*(o+1)+T[nm]
def make_ir(dur,dec,hpf,seed):
    n=int(dur*SR); r=np.random.default_rng(seed)
    ir=hp(r.standard_normal(n)*np.exp(-np.arange(n)/SR*dec),hpf); ir[:150]*=np.linspace(0,1,150)
    return ir/(np.max(np.abs(ir))+1e-9)
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

D="/home/claude/007/dewitt/"
morF,_=sf.read(D+"moreschi___08-G_T_54764.wav")
chaF,_=sf.read(D+"chaliapin__Boris_Godounow-In_the_Town_of_Kaz_n_-_Feodor_Chaliapin.wav")
m13,_=sf.read(D+"mor13.wav")
paris,_=sf.read(D+"bellsParis.wav")
wil,_=sf.read(D+"willie.wav")
jeru,_=sf.read(D+"jeru.wav")
lomax,_=sf.read(D+"lomax39.wav")
lomax2,_=sf.read(D+"lomax2.wav")
dwn,_=sf.read(D+"dwn.wav")
def seg(y,a,b): return norm(y[int(a*SR):int(b*SR)].copy())

MOR_A=seg(morF,147.5,163.0)
CHA_C=seg(chaF,137.5,152.0); ZER=seg(chaF,100.0,117.0)
LAUGH=seg(chaF,145.0,145.35)
C13=[seg(m13,21.0,25.0),seg(m13,33.0,36.6),seg(m13,45.0,48.4)]
L2SNG1=seg(lomax2,30.0,33.6); L2SPK1=seg(lomax2,224.0,227.8)
L2SPK2=seg(lomax2,240.5,243.6); L2SNG2=seg(lomax2,78.5,84.8)
PIANO=seg(m13,12.5,19.8); SOPR=seg(m13,216.5,231.5); SOPR_CAD=seg(m13,247.0,254.0)
BFLOOD=seg(paris,42.0,118.0)
WOPEN=seg(wil,2.0,14.0); WCONT=seg(wil,14.0,44.0)
WMOAN=seg(wil,90.0,102.0); WCALL=seg(wil,60.0,68.0); WHUM=seg(wil,162.0,174.0)
WS1=seg(wil,42.0,56.0); WS2=seg(wil,96.0,108.0)
JV1=seg(jeru,16.0,31.0); JV2=seg(jeru,40.0,52.0)
FID_BODY=seg(lomax,120.0,142.0)                 # the full-voiced plateau
FID_KEEN=seg(lomax,160.0,176.0)                 # the high keening register
def edge(y,fi=1.5,fo=2.0):
    y=y.copy(); a=int(fi*SR); b=int(fo*SR)
    y[:a]*=np.linspace(0,1,a); y[-b:]*=np.linspace(1,0,b)
    return y

def quietest(y,win=2.0):
    w=int(win*SR); best=None; bi=0
    for a in range(0,len(y)-w,w//2):
        e=float(np.mean(y[a:a+w]**2))
        if best is None or e<best: best=e; bi=a
    return y[bi:bi+w].copy()
CRACK=norm(np.concatenate([quietest(morF),quietest(chaF),quietest(wil)]))
hpm=hp(morF,3000)
pk2,_=find_peaks(np.abs(hpm),height=np.max(np.abs(hpm))*0.25,distance=int(0.05*SR))
POPS=[norm(hpm[max(0,i-96):max(0,i-96)+768].copy())*np.hanning(768) for i in pk2[:300] if i+700<len(hpm)]
def pop(): return POPS[int(rng.integers(0,len(POPS)))]

def declick(y):
    w=int(0.0015*SR); y[:w]*=np.linspace(0,1,w); y[-w:]*=np.linspace(1,0,w); return y
def ironknock(freq,dur,amp):                       # 006 / movement III
    n=int(dur*SR); t=np.arange(n)/SR; y=np.zeros(n)
    for r_,a,dc in [(1.0,1.0,0.45),(2.76,0.6,0.18),(4.9,0.4,0.11),(7.7,0.28,0.07),(11.3,0.16,0.05)]:
        if freq*r_<SR/2-500: y+=a*np.sin(2*np.pi*freq*r_*t+rng.uniform(0,6))*np.exp(-t/(dur*dc))
    clang=bp(rng.standard_normal(n),freq*3,min(SR/2-500,freq*15))*np.exp(-t/0.005)*0.7
    y=np.tanh((y+clang)*1.9)*np.minimum(1,t/0.0012)
    return declick(norm(hp(y,freq*0.7))*amp)
def fmpiston(freq,dur,amp):                        # 006/III — the ritual piston
    n=int(dur*SR); t=np.arange(n)/SR
    mod=np.sin(2*np.pi*freq*t)*2.6*np.exp(-t/0.05)
    car=np.sin(2*np.pi*freq*t+mod)
    knock=bp(rng.standard_normal(n),freq*3,min(SR/2-500,freq*9))*np.exp(-t/0.004)*0.45
    env=np.exp(-t/(dur*0.4))*np.minimum(1,t/0.0018)
    return declick(norm(np.tanh((car+knock)*1.5)*env))*amp
def strike_metal(freq,dur,amp):                    # 006/III — the downbeat strike
    n=int(dur*SR); t=np.arange(n)/SR
    atk=bp(rng.standard_normal(n),freq*5,min(SR/2-500,freq*40))*np.exp(-t/0.006)*0.9
    for r_ in (4.2,5.4,6.8,8.1,9.5,11.7): atk+=np.sin(2*np.pi*freq*r_*t+rng.uniform(0,6))*np.exp(-t/0.012)
    body=np.zeros(n)
    for r_ in (1.0,2.0,2.76,3.52,4.07,5.4,6.8,8.2,10.5):
        if freq*r_<SR/2-500: body+=(1/(1+r_*0.35))*np.sin(2*np.pi*freq*r_*t*(1+0.0025*rng.uniform(-1,1))+rng.uniform(0,6))*np.exp(-t/(dur*(0.55/(1+0.3*r_))))
    y=np.tanh((body+atk*0.7)*1.7)*np.minimum(1,t/0.002)
    return declick(norm(hp(lp(y,13000),freq*0.8)))*amp

def deep_pulse(f=80.0,harsh=1.0):
    nn=int(0.55*SR); tt=np.arange(nn)/SR
    y=(np.sin(2*np.pi*f*tt)+0.35*np.sin(2*np.pi*2*f*tt))*np.exp(-tt/0.28)
    a2=int(0.012*SR); y[:a2]*=np.linspace(0,1,a2)
    return drv(y,1.6*harsh)
def organ(f0,dur,att,rel,trem=0.07):
    nn=int((dur+rel)*SR); tt=np.arange(nn)/SR; y=np.zeros(nn)
    for p,gg in [(1,1.0),(2,0.5),(3,0.28),(4,0.15)]:
        for c in (-3,3):
            y+=gg*0.5*np.sin(2*np.pi*f0*p*2**(c/1200)*tt+rng.uniform(0,6.28))
    e=np.ones(nn); a=int(att*SR); rl=int(rel*SR)
    e[:a]=np.linspace(0,1,a)**1.4; e[-rl:]*=np.linspace(1,0,rl)**1.2
    e*=1.0+trem*np.sin(2*np.pi*5.5*tt+rng.uniform(0,6.28))
    return norm(y*e)
def swell_root(nm,o,dur):
    m0=Nn(nm,o); out=np.zeros(int((dur+1.8)*SR))
    for k,m in enumerate([m0,m0+7,m0+15]):
        v=organ(midi_f(m),dur,att=0.8,rel=1.8)
        out[:len(v)]+=v*(1.0-0.09*k)
    return norm(out)
def sub(f,dur,harsh=1.0):
    nn=int(dur*SR); tt=np.arange(nn)/SR
    y=np.sin(2*np.pi*f*tt)+0.3*np.sin(2*np.pi*2*f*tt)
    e=np.ones(nn); a=int(0.02*SR); rl=int(0.15*SR)
    e[:a]=np.linspace(0,1,a); e[-rl:]*=np.linspace(1,0,rl)
    return drv(y*e*np.exp(-tt/(dur*1.5)),1.4*harsh)
def sizzle_long(src,fc_path=(2800,4200,3000)):
    y=src.copy(); nn=len(y); tt=np.arange(nn)/SR
    fc=np.interp(tt,[0,tt[-1]*0.5,tt[-1]],fc_path)
    rm=hp(y*np.sin(2*np.pi*np.cumsum(fc)/SR),2300)
    envf=lp(np.abs(y),8.0); envf/=(np.max(envf)+1e-9)
    crk=np.tile(CRACK,nn//len(CRACK)+2)[:nn]
    burn=drv(rm*3.0,4.0)+crk*envf*1.3
    burn=hp(drv(burn,2.5),1900)
    e=np.ones(nn); w=int(0.8*SR); e[:w]=np.linspace(0,1,w); e[-w:]*=np.linspace(1,0,w)
    return norm(burn*e)
def strafe(f0,dur,rate=8.3):
    nn=int(dur*SR); tt=np.arange(nn)/SR
    saw=2*((f0*tt)%1.0)-1
    trem=0.5+0.5*np.sign(np.sin(2*np.pi*rate*tt))
    y=drv(lp(saw,f0*6),2.6)*trem
    e=np.ones(nn); w=int(1.0*SR); e[:w]=np.linspace(0,1,w); e[-w:]*=np.linspace(1,0,w)
    return norm(y*e)

def brass(f0,dur,drvA=2.4,vibd=0.012):
    nn=int(dur*SR); tt=np.arange(nn)/SR
    sc=2**((-70*np.exp(-tt/0.03))/1200)
    vib=1+vibd*np.sin(2*np.pi*5.3*tt)*np.clip((tt-0.18)/0.25,0,1)
    ph=2*np.pi*np.cumsum(f0*sc*vib)/SR
    y=(2*((ph/(2*np.pi))%1.0)-1)*0.8+np.sin(ph)*0.3
    y=bp(y,max(80,f0*0.8),4200)
    y=y*0.5+bp(y,700,1300)*1.4+bp(y,1600,2600)*0.9
    e=np.ones(nn); a=int(0.035*SR); r_=int(0.12*SR)
    e[:a]=np.linspace(0,1,a); e[-r_:]*=np.linspace(1,0,r_)
    return norm(drv(y*e,drvA))
def brass_line(notes,t0,g=0.34,pan=0.15,drvA=2.4):
    t=t0
    for nm,o,d in notes:
        if nm is None: t+=d; continue
        place(brass(midi_f(Nn(nm,o)),d*0.95,drvA=drvA),t,g*rng.uniform(0.9,1.05),pan)
        t+=d
    return t

def cping(fq=3100.0,dur=0.05):
    tt=np.arange(int(dur*SR))/SR
    return norm(np.sin(2*np.pi*fq*tt)*np.exp(-tt/(dur*0.25)))
def cclick():
    n=int(0.012*SR)
    return norm(hp(rng.standard_normal(n),4500)*np.exp(-np.arange(n)/(0.002*SR)))
def csnare():
    n=int(0.07*SR); y=hp(rng.standard_normal(n),2000)
    gate=np.ones(n); gate[int(0.055*SR):]=0
    return norm(y*gate*np.exp(-np.arange(n)/(0.05*SR)))
def coldkit(t0,t1,g0,g1):
    t=t0; bar=0
    while t<t1-0.05:
        g8=STEP/8
        prog=(t-t0)/max(0.001,(t1-t0)); g=g0+(g1-g0)*prog
        place(cclick(),t+3*g8,0.5*g,0.35)
        place(cclick(),t+7*g8,0.45*g,-0.35)
        if bar%2==1:
            place(csnare(),t+4*g8,0.55*g,0.0)
            for j in range(3): place(cclick(),t+7*g8+j*g8/3,0.4*g,-0.1+0.1*j)
        place(cping(3100),t+6*g8,0.28*g,0.2)
        t+=STEP; bar+=1

BEAT=60.0/56.0; STEP=2*BEAT; LOOP=8*STEP            # 17.14s: 4 bars Ab, 4 bars A
fAb1=midi_f(Nn('G#',1)); fA1=midi_f(Nn('A',1))
DUR=287.0; TOT=int(DUR*SR); L=np.zeros(TOT); R=np.zeros(TOT)
def place(sig,t,g=1.0,pan=0.0):
    i=int(t*SR); e=min(TOT,i+len(sig))
    if e<=i: return
    gl,gr=np.sqrt(0.5-pan/2),np.sqrt(0.5+pan/2)
    L[i:e]+=sig[:e-i]*g*gl; R[i:e]+=sig[:e-i]*g*gr
CUTS=[]
reps=TOT//len(CRACK)+2
bedy=np.tile(CRACK,reps)[:TOT].copy()
bedy[:int(3*SR)]*=np.linspace(0,1,int(3*SR)); bedy[-int(4*SR):]*=np.linspace(1,0,int(4*SR))
L+=bedy*10**(-36/20); R+=np.roll(bedy,int(0.013*SR))*10**(-36/20)
irSm=make_ir(0.9,4.5,300,55)
def sharpen(v):
    v=norm(hp(v,160)); return norm(drv(v*0.7+bp(v,1100,3200)*0.9,1.3))
irTalk=make_ir(1.1,3.8,260,91)
def converse(frag,t,pan,g=0.45,fade_out=0.0):
    y=lp(norm(hp(frag,240)),3600)
    env=lp(np.abs(y),6.0); env/=(np.max(env)+1e-9)
    gate=0.22+0.78*np.clip(env/0.22,0,1)
    y=y*gate
    w=fftconvolve(y,irTalk)[:len(y)]
    y=y*0.78+norm(w)*0.26*np.max(np.abs(y))
    if fade_out>0:
        b2=int(fade_out*SR); y[-b2:]*=np.linspace(1,0,b2)
    place(norm(y),t,g,pan)
def speak(frag,t,pan,g=0.5,dry=True):
    y=hp(frag,180)
    if not dry:
        w=fftconvolve(y,irSm)[:len(y)]; y=y*0.8+norm(w)*0.22
    place(norm(y),t,g,pan)

# ============ THE GROUND ============
T0=0.8; TB0=135.8; TEP1=185.1; STOP_T=228.6
def ground(t0,t1,density,harsh=1.0,pattern='full'):
    """Scenario-reactive percussion counterpoint.
    float:    spine downbeat every 2nd step only — under a leading voice
    tread:    spine (0,4) even quarters, one answer each 4th bar — procession
    walk:     spine (0,3,6), floor, no answer — neutral motion
    full:     spine + strict answer + pops + floor (+ strike at loop heads)
    skeleton: the dotted spine alone
    Timbre roles still swap with harmony: piston on Ab, iron on A."""
    t=t0; bar=0; g8=STEP/8
    while t<t1-0.1:
        root_A=(int((t-T0)/STEP)//4)%2==1
        f=fA1 if root_A else fAb1
        nm='A' if root_A else 'G#'
        if root_A:
            spine=lambda a2: ironknock(196.0,0.35,a2); answ=lambda a2: fmpiston(82.0,0.6,a2)
        else:
            spine=lambda a2: fmpiston(82.0,0.6,a2); answ=lambda a2: ironknock(196.0,0.35,a2)
        if pattern=='skeleton':
            for p in (0,3,6): place(spine(1.0),t+p*g8,(0.6 if p==0 else 0.46)*density)
            t+=STEP; bar+=1; continue
        if pattern=='float':
            if bar%2==0: place(spine(1.0),t,0.55*density)
        elif pattern=='tread':
            for p in (0,4): place(spine(1.0),t+p*g8,(0.6 if p==0 else 0.44)*density)
            if bar%4==3: place(answ(1.0),t+6*g8,0.38*density)
        else:
            for p in (0,3,6): place(spine(1.0),t+p*g8,(0.62 if p==0 else 0.48)*density)
            if pattern=='full':
                for p in ((2,7) if bar%2==0 else (4,)):
                    place(answ(1.0),t+p*g8,0.4*density)
                if density>=0.6:
                    for p in ((5,) if bar%2==0 else (1,5)):
                        place(pop(),t+p*g8,0.3,0.3 if p==1 else -0.3)
                if density>0.8 and bar%8==0: place(strike_metal(148.0,0.8,1.0),t,0.48)
        if density>0.35: place(deep_pulse(80,harsh),t,0.62*density)
        if density>0.45 and pattern!='float': place(sub(f*2,STEP*0.9,harsh=min(harsh,1.6)),t,0.36*density)
        if pattern=='float' and bar%2==0: place(sub(f*2,STEP*1.8),t,0.3*density)
        if bar%4==0 and density>0.3:
            place(swell_root(nm,1,STEP*3.6),t,0.2*min(1.0,density))
        t+=STEP; bar+=1
# the scene opens shrilling: the keening fiddle, its higher shadow, the knock discovered beneath
FIDO=edge(norm(hp(FID_KEEN,350)),1.8,3.0)
place(FIDO,1.2,0.44,0.15)
FIDO2=edge(norm(hp(resample(FID_KEEN,2**(7/12)),700))[:int(11*SR)],2.5,3.0)
place(FIDO2,5.5,0.2,-0.3)
t=T0; g8=STEP/8
for bar in range(8):
    root_A=(bar//4)%2==1; f=fA1 if root_A else fAb1; nm='A' if root_A else 'G#'
    if root_A:
        spine=lambda a2: ironknock(196.0,0.35,a2); answ=lambda a2: fmpiston(82.0,0.6,a2)
    else:
        spine=lambda a2: fmpiston(82.0,0.6,a2); answ=lambda a2: ironknock(196.0,0.35,a2)
    if bar>=2:
        gA=min(1.0,(bar-1)/4.0)
        for p in (0,3,6):
            place(spine(1.0),t+p*g8,(0.6 if p==0 else 0.46)*gA)
    if bar>=2:
        for p in ((2,7) if bar%2==0 else (4,)):
            place(answ(1.0),t+p*g8,0.38)
    if bar>=4:
        place(deep_pulse(80),t,0.58)
        place(sub(f*2,STEP*0.9),t,0.34)
    if bar>=6:
        for p in ((5,) if bar%2==0 else (1,5)):
            place(pop(),t+p*g8,0.28,0.3 if p==1 else -0.3)
        if bar%2==0: place(swell_root(nm,1,STEP*1.8),t,0.2)
    t+=STEP

# ============ V1 — WILLIE (17.1–51.4) ============
ground(T0+LOOP,T0+3*LOOP,0.5,pattern='float')
# willie and the soprano merged: one constant line, handed back and forth
def onevoice(v):
    y=sharpen(v)
    w=fftconvolve(y,irSm)[:len(y)]
    return norm(y*0.82+norm(w)*0.2)
def f0med(v,lo,hi):
    n=1<<int(np.ceil(np.log2(len(v))))
    sp=np.abs(np.fft.rfft(v*np.hanning(len(v)),n=n))
    fr=np.fft.rfftfreq(n,1/SR); m=(fr>=lo)&(fr<=hi)
    return fr[m][int(np.argmax(sp[m]))]
_f0w=f0med(WOPEN[:int(6*SR)],85,320)
_f0s=f0med(C13[0],240,700)
_semis=12*np.log2(_f0s/_f0w)
def herdown(v):
    y=norm(resample(v,2**(-_semis/12)))          # her fundamental onto his
    y=lp(hp(y,240),2800)                          # boxed; her bass rumble cut
    env=lp(np.abs(y),3.0); env=env/(np.max(env)+1e-9)
    y=y/(env*0.8+0.2)                             # the rises and falls flattened
    return norm(y)
def embrace(vw,vs,dur):
    n=int(dur*SR)
    vw2=np.tile(vw,n//len(vw)+1)[:n]; vs2=np.tile(vs,n//len(vs)+1)[:n]
    g2=int(0.075*SR); win=np.hanning(g2); hop=g2//2
    out=np.zeros(n); pos=0; k=0
    while pos+g2<n:
        srcv=vw2 if k%2==0 else vs2
        othv=vs2 if k%2==0 else vw2
        e2=np.abs(othv[pos:pos+g2]); e2=e2/(np.max(e2)+1e-9)
        out[pos:pos+g2]+=srcv[pos:pos+g2]*win*(0.5+0.5*e2)
        pos+=hop; k+=1
    y=norm(out)
    y=norm(y+norm(vw2*vs2)*0.14)                  # the ring whisper: their product
    return onevoice(lp(hp(y,180),3200))
EMB=[(embrace(norm(WOPEN),herdown(C13[0]),12.0),0.54),
     (embrace(norm(WCONT[:int(8.0*SR)]),herdown(C13[1]),8.0),0.52),
     (embrace(norm(WCALL),herdown(C13[2]),8.0),0.52)]
tr=T0+LOOP+0.6; XF=1.1
for y,gg in EMB:
    y=edge(y,XF*0.9,XF*1.1)
    place(y,tr,gg,-0.05)
    tr+=len(y)/SR-XF
# the guitar as mortar at the seams
def gglue(a,b): 
    y=norm(lp(hp(seg(dwn,a,b),130),4500))
    w2=fftconvolve(y,irSm)[:len(y)]
    return norm(y*0.85+norm(w2)*0.18)
place(edge(gglue(1.5,7.5),0.8,1.2),17.0,0.34,0.15)
place(edge(gglue(13.5,19.5),0.8,1.2),29.0,0.32,0.18)
place(edge(gglue(37.5,45.5),0.8,1.5),44.5,0.34,0.15)   # runs under her last word into V2
place(edge(gglue(3.0,7.0),1.0,1.5),84.5,0.28,0.2)      # glues V2 into V3

# ============ V2 — THE BODY (51.4–85.7) ============
ground(T0+3*LOOP,T0+5*LOOP,0.6,pattern='tread')
_f0m=f0med(MOR_A[:int(6*SR)],300,900)
_semM=12*np.log2(_f0m/_f0w)
MORd=lp(hp(norm(resample(MOR_A[:int(12.0*SR)],2**(-_semM/12))),240),2800)
env=lp(np.abs(MORd),3.0); env=env/(np.max(env)+1e-9); MORd=norm(MORd/(env*0.8+0.2))
EMB_M=embrace(norm(WCONT[int(8.0*SR):int(16.0*SR)]),MORd,14.0)
place(edge(EMB_M,1.4,1.6),T0+3*LOOP+0.5,0.56,-0.03)
ghost=lp(gstretch(MOR_A[:int(8.0*SR)],2.6),2600)
place(ghost,T0+3*LOOP+16.0,0.32,0.1)
place(sharpen(WCALL),T0+4*LOOP+6.5,0.5,-0.33)
place(edge(norm(lp(hp(FID_BODY,160),5200))),66.0,0.42,0.18)
place(sharpen(WOPEN[:int(5.5*SR)]),T0+5*LOOP+0.2-LOOP+16.6,0.4,-0.1)

# ============ V3 — THE SONG (85.7–120.0) ============
ground(T0+5*LOOP,T0+7*LOOP,0.55,pattern='float')
S=hp(SOPR,180)
place(norm(S)*np.interp(np.arange(len(S))/SR,[0,2.6,12.2,15],[0,1,0.95,0]),T0+5*LOOP+1.0,0.6,0.05)
place(edge(norm(hp(PIANO,150)),0.5,2.2),T0+5*LOOP+17.0,0.4,-0.2)
place(edge(sharpen(WOPEN[int(6.0*SR):int(13.5*SR)]),0.3,2.2),T0+6*LOOP+9.0,0.4,-0.2)

# ============ LEAD-IN (120.8–135.8): jerusalem voices + brass ============
ground(T0+7*LOOP,T0+7*LOOP+6.5,0.38,harsh=1.1,pattern='tread')
ground(T0+7*LOOP+6.5,TB0,0.5,harsh=1.2,pattern='tread')
JV1s=norm(hp(JV1,200))
JV1s=JV1s*np.interp(np.arange(len(JV1s))/SR,[0,3.2,13,15],[0,1,1,0.55])
place(JV1s,120.6,0.42,0.2)
place(swell_root('G#',1,4.5),124.0,0.24); place(swell_root('A',1,4.5),130.0,0.24)
LEAD=[('G#',4,1.6),('G',4,0.55),('G#',4,0.8),('A#',4,0.8),('G#',4,1.2),(None,0,0.8),
      ('E',4,0.9),('F',4,0.5),('G#',4,2.2),(None,0,1.0),('A',4,0.9),('G#',4,2.4)]
brass_line(LEAD,126.5,g=0.24,pan=0.2,drvA=1.8)

# ============ V4 — THE BURN (135.8–170.1) ============
ground(TB0,TB0+2*LOOP,0.75,harsh=1.9,pattern='full')
WSUB=np.concatenate([WS1,WS2])
place(sizzle_long(WSUB,(2800,4200,3000)),TB0+0.7,0.32,0.12)
place(sharpen(WS1),TB0+1.8,0.58,-0.06)
place(edge(sharpen(WS2[:int(10.0*SR)]),0.3,1.6),TB0+15.8,0.58,-0.06)
place(strafe(fAb1*2,30.0),TB0+1.0,0.16,0.0)
BURN1=[('B',4,0.7),('A#',4,0.5),('C#',5,0.7),('G#',4,1.1),(None,0,0.6),
       ('D#',5,0.6),('C#',5,0.5),('B',4,0.5),('G#',4,1.8)]
brass_line(BURN1,TB0+4.0,g=0.28,pan=-0.02,drvA=2.6)
BURN2=[('D#',5,0.7),('E',5,0.6),('C#',5,0.7),('B',4,0.6),('C#',5,0.5),('G#',4,2.6)]
brass_line(BURN2,TB0+18.0,g=0.28,pan=-0.02,drvA=2.6)

# ============ EPILOGUE (170.1–185.1): cooling, voices return ============
ground(TB0+2*LOOP,TEP1,0.42,harsh=1.0,pattern='tread')
place(norm(hp(JV2,200)),171.0,0.4,0.2)
place(swell_root('A',1,4.5),172.0,0.22); place(swell_root('G#',1,4.5),178.0,0.24)
place(sharpen(WCALL),178.5,0.36,-0.15)
place(edge(norm(lp(hp(FID_KEEN,200),5200)),2.0,2.5),171.5,0.26,0.3)
EPI=[('A#',4,1.2),('G#',4,1.0),('G',4,1.0),('F',4,1.2),('E',4,1.4),(None,0,0.8),('G#',3,4.5)]
brass_line(EPI,175.5,g=0.24,pan=0.2,drvA=1.7)
place(brass(midi_f(Nn('G#',2)),4.3,drvA=1.6),182.1,0.14,0.05)

coldkit(TB0,TB0+2*LOOP,0.0,0.85)          # fades in across the burn
coldkit(TB0+2*LOOP,TEP1,0.25,0.35)         # cold ticking through the epilogue
coldkit(TEP1,224.5,0.85,1.0)               # full in the mob

# ============ V5 — THE MOB (154.3–193.4) ============
ground(TEP1,224.5,0.85,harsh=1.4,pattern='full')
cells=[CHA_C[i*int(BEAT*SR):(i+1)*int(BEAT*SR)].copy() for i in range(int(len(CHA_C)/(BEAT*SR)))]
for c in cells:
    w2=int(0.004*SR); c[:w2]*=np.linspace(0,1,w2); c[-w2:]*=np.linspace(1,0,w2)
SEQ=[0,0,1,2, 0,0,3,3, 4,5,4,6, 7,7,8,2, 0,1,0,9, 10,10,3,3, 4,4,11,6, 0,2,1,12, 3,0,5,9]
T5=TEP1
for j,ci in enumerate(SEQ):
    tt=T5+j*BEAT
    if tt>220.0: break
    if ci<len(cells):
        c=cells[ci]
        if j%8==7:
            h=c[:len(c)//2]
            place(h,tt,0.56,0.05); place(h*0.8,tt+BEAT/2,0.44,0.05)
        else:
            place(c,tt,0.56,0.05)
BF=hp(BFLOOD[:int(43.0*SR)],160)
genv=np.interp(np.arange(len(BF))/SR,[0,14,26,38,43],[0.14,0.26,0.42,0.55,0.55])
place(norm(BF)*genv,T5,1.0,0.08)
def laughter(t0,nrep,g=0.55,accel=1.0):
    dt=0.16
    for k2 in range(nrep):
        place(hp(LAUGH,250),t0+sum(dt*(accel**j) for j in range(k2)),g*(1-0.05*k2),(-1)**k2*0.35)
for k2 in range(5):
    place(hp(LAUGH,250),T5+16.5+sum(0.16*(0.94**j) for j in range(k2)),0.28+0.055*k2,(-1)**k2*0.3)
wf1=lp(spiral(WMOAN[:int(5.0*SR)],3.2,grain=0.05,semis_start=0.0,semis_end=-7.0),3000)
place(wf1,T5+9.0,0.5,0.24)
zerA=lp(spiral(ZER,1.2,grain=0.05,semis_start=0.0,semis_end=+5.0),4600)
zerA=drv(norm(zerA)*np.linspace(1.0,1.8,len(zerA)),1.7)
place(norm(zerA)*np.linspace(0.28,0.58,len(zerA)),T5+12.0,1.0,-0.1)
wf2=lp(spiral(WMOAN[int(5.0*SR):int(10.0*SR)],3.6,grain=0.045,semis_start=-3.0,semis_end=-10.0),2600)
place(wf2,T5+26.0,0.48,0.28)
nnT=int(9.0*SR); ttT=np.arange(nnT)/SR
fT0_=midi_f(Nn('D',5)); fT1_=midi_f(Nn('F',5))
fcur=fT0_*(fT1_/fT0_)**(ttT/9.0)
vib=1+0.006*np.sin(2*np.pi*5.2*ttT)*np.minimum(1,ttT/2)
toneZ=np.sin(2*np.pi*np.cumsum(fcur*vib)/SR)
place(hp(toneZ,800)*np.interp(ttT,[0,1.8,7.5,9.0],[0,0.5,0.55,0.45]),T5+31.0,0.46,0.02)
CUTS.append((T5+20.5,0.9))                          # the drop, mid-mob
glc=SOPR_CAD[:int(STEP*SR*1.4)]
nn2=len(glc); rmp=np.linspace(1.0,2**(1/12),nn2)
idx=np.cumsum(rmp); idx=np.clip(idx,0,nn2-2)
i0=idx.astype(int); fr=idx-i0
glid=glc[i0]*(1-fr)+glc[i0+1]*fr
place(norm(glid)*np.interp(np.arange(len(glid))/SR,[0,0.45,2.6,3.0],[0,1,1,0.85]),224.6,0.68,0.0)
place(sub(fAb1*2,STEP*0.5),224.6,0.46)
place(sub(fA1*2,STEP*0.5),224.6+STEP*0.5,0.52)
CUTS.append((STOP_T,0.7))                           # THE STOP

# ============ CODA (198.2–260): the ground as skeleton ============
ground(230.0,267.0,0.5,pattern='skeleton')             # the knock alone keeps time
gF=lp(gstretch(WHUM[:int(5.0*SR)],4.5,grain=0.06),1800)
place(gF,234.2,0.3,0.05)
pg=lp(gstretch(PIANO,2.2),2400)
place(pg,250.2,0.22,0.18)
gF2=lp(gstretch(WHUM[int(5.0*SR):int(9.0*SR)],5.5,grain=0.05),1500)
place(gF2,259.2,0.24,-0.1)
# the spiral fade-out: everything falls away, clean
place(edge(sharpen(WOPEN[:int(7.0*SR)]),0.5,2.5),256.5,0.4,-0.05)
SPE=lp(spiral(np.concatenate([WHUM[:int(4.0*SR)],PIANO[:int(4.0*SR)]]),2.4,grain=0.05,semis_start=0.0,semis_end=-12.0),1600)
place(norm(SPE)*np.interp(np.arange(len(SPE))/SR,[0,2,10,19.2],[0,0.32,0.26,0]),265.0,1.0,0.08)
fadeN=int(9.0*SR)
for (tc,dc) in CUTS:
    a=int(tc*SR); b=int((tc+dc)*SR); dn=int(0.0015*SR)
    L[a-dn:a]*=np.linspace(1,0,dn); R[a-dn:a]*=np.linspace(1,0,dn)
    L[a:b]=0.0; R[a:b]=0.0
    L[b:b+dn]*=np.linspace(0,1,dn); R[b:b+dn]*=np.linspace(0,1,dn)
L[-fadeN:]*=np.linspace(1,0,fadeN)**1.3; R[-fadeN:]*=np.linspace(1,0,fadeN)**1.3
st=np.stack([L,R],1); st=norm(st)*10**(-1.0/20)
sf.write("/mnt/user-data/outputs/007_mvt1_r9.wav",st,SR,subtype="PCM_24")
print(f"r9 (burn ah deleted): {len(st)/SR:.1f}s | leadin 120.8-135.8 | burn 135.8-170.1 | epilogue 170.1-185.1 | mob 185.1-224.5 | stop@{STOP_T} | coda 230-291")
