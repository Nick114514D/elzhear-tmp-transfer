"""
007 — Movement IV, impro d6: THE RALLY — seven cinematic scenes: tempo breathes (curves per section), KS subtheme in six appearances, rotating accent grids with 7/8 bars, the ocean woven through, effector-chained walls.
The lattice law: every sample is repitched to the nearest G minor degree.
The conjugation: voices are chopped to syllables and made to sing written
degree-lines. The wall law: noise is amplitude-gated on the motor's own
eighth-grid — the effector plays the rhythm. The seize: the lattice
tightens to G and Ab only (the pincer). At the end, one phrase of the
fatherland is heard whole — the only unreproduced sound in the movement.
"""
import numpy as np, soundfile as sf, os
from scipy.signal import butter, sosfilt
SR=48000; rng=np.random.default_rng(1672)
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
        out[opos:opos+g]+=y[pos:pos+g]*win; pos+=hin; opos+=hout
    return norm(out[:int(len(y)*factor)])
def edge(y,fi=0.5,fo=0.5):
    y=y.copy(); a=min(int(fi*SR),len(y)//2); b=min(int(fo*SR),len(y)//2)
    if a>0: y[:a]*=np.linspace(0,1,a)
    if b>0: y[-b:]*=np.linspace(1,0,b)
    return y
def declick(y,w=180):
    y=y.copy(); y[:w]*=np.linspace(0,1,w); y[-w:]*=np.linspace(1,0,w); return y

D="/home/claude/007/iv48/"
def L48(tag): 
    y,_=sf.read(D+tag+".wav"); return y
vader=L48("vader"); urlus1=L48("urlus1"); urlus2=L48("urlus2"); rooy=L48("rooy")
vball=L48("vball"); prel=L48("prel"); haag=L48("haag"); tulip=L48("tulip")
adag=L48("adag"); gulls=L48("gulls"); alkm=L48("alkm")
roto=L48("roto1"); spore=L48("spore"); chern1=L48("chern1"); chern3=L48("chern3")
def seg(y,a,b): return norm(y[int(a*SR):int(b*SR)].copy())

# ===== THE LATTICE =====
G2=98.0
LAT=[0,2,3,5,7,8,10]                                # G minor degrees (semitones from G)
def f0med(y,lo=120,hi=800):
    v=y[:int(6*SR)]
    n=1<<int(np.ceil(np.log2(len(v))))
    sp=np.abs(np.fft.rfft(v*np.hanning(len(v)),n=n))
    fr=np.fft.rfftfreq(n,1/SR); m=(fr>=lo)&(fr<=hi)
    return float(fr[m][int(np.argmax(sp[m]))])
def latticize(y,lo=120,hi=800,degrees=None):
    degs=degrees if degrees is not None else LAT
    f=f0med(y,lo,hi)
    st=12*np.log2(f/G2)
    cands=[d+12*k for k in range(-2,7) for d in degs]
    tgt=min(cands,key=lambda c:abs(c-st))
    return norm(resample(y,2**((st-tgt)/12)))
def to_degree(y,base_deg,new_deg):
    return norm(resample(y,2**((base_deg-new_deg)/12)))
def syl(y,n):
    c=int(len(y)/n)
    return [declick(y[i*c:(i+1)*c].copy()) for i in range(n)]

BPM=156.0; BEAT=60/BPM; E8=BEAT/2; BAR=4*BEAT
DUR=460.0; TOT=int(DUR*SR); Lc=np.zeros(TOT); Rc=np.zeros(TOT)
def place(sig,t,g=1.0,pan=0.0):
    i=int(t*SR); e=min(TOT,i+len(sig))
    if e<=i: return
    gl,gr=np.sqrt(0.5-pan/2),np.sqrt(0.5+pan/2)
    Lc[i:e]+=sig[:e-i]*g*gl; Rc[i:e]+=sig[:e-i]*g*gr
CUTS=[]
# floor bed from the shellac's own quiet
def quietest(y,win=1.5):
    w=int(win*SR); best=None; bi=0
    for a in range(0,len(y)-w,w//2):
        e=float(np.mean(y[a:a+w]**2))
        if best is None or e<best: best=e; bi=a
    return y[bi:bi+w].copy()
CRACK=norm(np.concatenate([quietest(vader),quietest(vball)]))
bed=np.tile(CRACK,TOT//len(CRACK)+2)[:TOT].copy()
bed[:int(2*SR)]*=np.linspace(0,1,int(2*SR)); bed[-int(4*SR):]*=np.linspace(1,0,int(4*SR))
Lc+=bed*10**(-36/20); Rc+=np.roll(bed,int(0.012*SR))*10**(-36/20)

# ===== THE MOTOR =====
def pluck(f,dur=0.22,bright=0.6):
    n=int(dur*SR); dl=int(SR/f); buf=rng.standard_normal(dl)*0.8
    out=np.zeros(n)
    for i in range(n):
        out[i]=buf[i%dl]
        buf[i%dl]=(buf[i%dl]+buf[(i+1)%dl])*0.5*(0.994+0.004*bright)
    e=np.ones(n); e[:60]=np.linspace(0,1,60); e[-int(0.02*SR):]*=np.linspace(1,0,int(0.02*SR))
    return norm(drv(out*e,1.6))
def sawb(f,dur,dA=3.0):
    n=int(dur*SR); tt=np.arange(n)/SR
    y=2*((f*tt)%1)-1+0.5*(2*((f*1.006*tt)%1)-1)
    y=lp(y,f*9)
    e=np.ones(n); e[:int(0.004*SR)]=np.linspace(0,1,int(0.004*SR)); e[-int(0.03*SR):]*=np.linspace(1,0,int(0.03*SR))
    return drv(y*e,dA)
def piston(f,dur,amp):
    n=int(dur*SR); tt=np.arange(n)/SR
    mod=np.sin(2*np.pi*f*tt)*2.6*np.exp(-tt/0.05)
    car=np.sin(2*np.pi*f*tt+mod)
    kn=bp(rng.standard_normal(n),f*3,min(SR/2-500,f*9))*np.exp(-tt/0.004)*0.45
    return declick(norm(np.tanh((car+kn)*1.5)*np.exp(-tt/(dur*0.4))*np.minimum(1,tt/0.0018)))*amp
def iron(f,dur,amp):
    n=int(dur*SR); tt=np.arange(n)/SR; y=np.zeros(n)
    for r_,a,dc in [(1.0,1.0,0.4),(2.76,0.6,0.16),(4.9,0.4,0.1),(7.7,0.25,0.06)]:
        if f*r_<SR/2-500: y+=a*np.sin(2*np.pi*f*r_*tt+rng.uniform(0,6))*np.exp(-tt/(dur*dc))
    cl=bp(rng.standard_normal(n),f*3,min(SR/2-500,f*14))*np.exp(-tt/0.005)*0.7
    return declick(norm(hp(np.tanh((y+cl)*1.9)*np.minimum(1,tt/0.0012),f*0.7)))*amp
def smetal(f,dur,amp):
    n=int(dur*SR); tt=np.arange(n)/SR
    atk=bp(rng.standard_normal(n),f*5,min(SR/2-500,f*40))*np.exp(-tt/0.006)*0.9
    body=np.zeros(n)
    for r_ in (1.0,2.0,2.76,4.07,5.4,6.8,8.2):
        if f*r_<SR/2-500: body+=(1/(1+r_*0.35))*np.sin(2*np.pi*f*r_*tt+rng.uniform(0,6))*np.exp(-tt/(dur*0.4/(1+0.3*r_)))
    return declick(norm(hp(lp(np.tanh((body+atk*0.7)*1.7),12000),f*0.8)*np.minimum(1,tt/0.002)))*amp
G1=49.0
GRIDS=[[0,3,6],[0,2,5],[0,3,5]]
def spiral(y,factor,grain=0.045,semis_start=0.0,semis_end=-5.0):
    g2=int(grain*SR); hin=int(g2*0.5); hout=int(hin*factor)
    nout=int(len(y)*factor)+g2; out=np.zeros(nout); win=np.hanning(g2)
    pos=0; opos=0
    while pos+int(g2*1.8)<len(y) and opos+g2<nout:
        prog=opos/max(1,nout)
        rt=2**((semis_start+(semis_end-semis_start)*prog)/12.0)
        gr=resample(y[pos:pos+max(4,int(g2*rt)+2)],rt)[:g2]
        if len(gr)<g2: gr=np.pad(gr,(0,g2-len(gr)))
        out[opos:opos+g2]+=gr*win
        pos+=hin; opos+=hout
    return norm(out[:int(len(y)*factor)])
def sizzle_long(src,fc_path=(2400,4200,3000)):
    y=src.copy(); nn=len(y); tt=np.arange(nn)/SR
    fc=np.interp(tt,[0,tt[-1]*0.5,tt[-1]],fc_path)
    rm=hp(y*np.sin(2*np.pi*np.cumsum(fc)/SR),2000)
    envf=lp(np.abs(y),8.0); envf/=(np.max(envf)+1e-9)
    crk=np.tile(CRACK,nn//len(CRACK)+2)[:nn]
    return norm(hp(drv(drv(rm*3.0,4.0)+crk*envf*1.3,2.5),1700))
def efx(y):
    """the effector, audible: mud-cut, octave fuzz, drive, slow wah"""
    y=hp(norm(y),170)
    y=y-0.3*bp(y,240,470)
    r2=np.abs(y); r2=r2-np.mean(r2)
    y=drv(y*0.72+r2*0.5,2.4)
    tt=np.arange(len(y))/SR
    fc=1200+900*np.sin(2*np.pi*0.4*tt)
    lo2=bp(y,380,1400); hi2=bp(y,1200,3400)
    w=np.clip((fc-380)/3020,0,1)
    y=y*0.7+(lo2*(1-w)+hi2*w)*0.6
    return norm(y)
def wall_gate(y,t0,dur,g,e8,duty=0.5,pan=0.0,band=(280,6200)):
    n=int(dur*SR)
    w=efx(bp(np.tile(y,n//len(y)+2)[:n],band[0],band[1]))
    gate=np.zeros(n); step=int(e8*SR); hold=int(duty*e8*SR); a=int(0.004*SR); r=int(0.03*SR)
    i=0
    while i<n:
        j=min(n,i+hold); gate[i:j]=1.0
        if i+a<n: gate[i:i+a]=np.linspace(0,1,a)
        if j<n: gate[j:min(n,j+r)]=np.linspace(1,0,min(r,n-j))
        i+=step
    place(edge(w*gate,0.3,0.8),t0,g,pan)

# ===== SOURCES =====
VAD=latticize(seg(vader,86.0,100.0),180,600); VS=syl(VAD,36)
VB=latticize(seg(vball,26.0,32.0),90,700); VBc=syl(VB,16)
VB2=latticize(seg(vball,198.0,206.0),90,700); VB2c=syl(VB2,16)
CRY1=latticize(seg(urlus1,148.0,156.0),200,700)
CRY2=latticize(seg(urlus2,22.0,30.0),200,700)
ROOYC=norm(seg(rooy,138.0,146.0)); RSy=syl(ROOYC,16)
TUL=latticize(seg(tulip,52.0,66.0),150,600); TSy=syl(TUL,28)
PRE=latticize(seg(prel,112.0,128.0),90,600); PRc=syl(PRE,40)
HAAGS=norm(resample(seg(haag,162.0,168.0),2**(-5/12)))
ADG=lp(latticize(seg(adag,176.0,188.0),120,700),2400)
GULL=norm(hp(seg(gulls,296.0,306.0),1500))
ALKB=latticize(seg(alkm,30.0,36.0),200,900)
CH1_ARIA=norm(seg(chern1,106.0,126.0))
CH1_MAN=norm(seg(chern1,16.0,24.0))
ds3w,_=sf.read("/home/claude/007/iv48/ds3.wav")
ffs1w,_=sf.read("/home/claude/007/iv48/ffs1.wav")
ffs2bw,_=sf.read("/home/claude/007/iv48/ffs2b.wav")
def gtr(y):
    y=hp(norm(y),150)
    y=y-0.32*bp(y,230,450)
    r2=np.abs(y); r2-=np.mean(r2)
    y=drv(y*0.65+r2*0.6,3.2)
    y=y+bp(y,1800,3500)*0.5
    y=y+np.sin(2*np.pi*55*np.arange(len(y))/SR)*bp(y,80,200)*0.25
    return norm(y)
RIFF_A=gtr(latticize(seg(ds3w,0.0,12.0),70,300))
RIFF_B=gtr(latticize(seg(ds3w,21.0,36.0),70,300))
BITE1=gtr(seg(ffs1w,246.0,252.0))
BITE2=gtr(seg(ffs1w,21.0,27.0))
WALLG=gtr(seg(ffs2bw,279.0,291.0))
WALLG=norm(WALLG-0.28*bp(WALLG,2100,3700))
def rdspan(p,a,b): 
    y,_=sf.read(p,start=int(a*SR),frames=int((b-a)*SR)); return y
RT1a=rdspan("/home/claude/007/iv48/read_tulip_ch1.wav",60,74)
RT1b=rdspan("/home/claude/007/iv48/read_tulip_ch1.wav",122,136)
RT2a=rdspan("/home/claude/007/iv48/read_tulip_ch2.wav",100,118)
RT2b=rdspan("/home/claude/007/iv48/read_tulip_ch2.wav",300,318)
RNLa=rdspan("/home/claude/007/iv48/read_havelaar_nl.wav",60,72)
def readv(y):
    y=lp(hp(norm(y),180),3400)
    y=y*0.85+np.roll(y,int(0.085*SR))*0.2
    return norm(drv(y,1.2))
def mobtongue(y):
    return norm(drv(hp(norm(y),300),1.7))
OC,_=sf.read("/home/claude/007/alt48/ocean90.wav")
SURF=norm(hp(lp(OC,5500),140))
UNDW=norm(lp(OC,800))
pk=int(np.argmax(np.abs(OC[:int(80*SR)])))
WCRASH=norm(drv(hp(OC[max(0,pk-int(0.3*SR)):pk+int(2.7*SR)].copy(),180),1.5))
def sea(t0,dur,g_surf,g_und,fi=3.0,fo=3.0):
    n=int(dur*SR)
    s2=np.tile(SURF,n//len(SURF)+2)[:n]; u2=np.tile(UNDW,n//len(UNDW)+2)[:n]
    place(edge(s2,fi,fo),t0,g_surf,0.12)
    place(edge(u2,fi,fo),t0,g_und,-0.1)
def to_deg(y,dg): return norm(resample(y,2**(-dg/12)))
def burst(cells,t0,degs,gap,g=0.5,pan=0.0):
    for k,dg in enumerate(degs):
        cc=to_deg(cells[k%len(cells)],dg) if dg!=0 else cells[k%len(cells)]
        cc=hp(norm(cc)[:int(gap*SR*0.94)],190)
        aw=min(int(0.028*SR),len(cc))
        cc[:aw]*=np.linspace(0.25,1,aw)**0.7
        place(cc,t0+k*gap,g,pan)

# ===== THE SUBTHEME (Karplus identity) =====
KS_D=[0,3,2,0,7,8,7,5,3,2,0]
KS_R=[1,0.5,0.5,1, 1,0.5,0.5,1, 1,0.5,2]
def ks_theme(t0,bt,g=0.4,transpose=0,frag=None,bright=0.6,pincer=False):
    seq=list(zip(KS_D,KS_R))
    if frag: seq=seq[frag[0]:frag[1]]
    tt2=t0
    for dg,r in seq:
        d2=dg+transpose
        if pincer: d2=0 if dg in (0,2,5) else 1
        place(pluck(G2*2**(d2/12),min(0.6,r*bt*0.92),bright),tt2,g,0.14*np.sign(np.sin(tt2)))
        tt2+=r*bt
    return tt2

def orch_bar(t,bar,bt,density=1.0,harsh=1.0,cluster=False,seven=False):
    e8=bt/2
    grid=GRIDS[(bar//4)%3]
    rot=VBc if bar%4<3 else VB2c
    n8=7 if seven else 8
    for j,e in enumerate(grid):
        if e>=n8: continue
        te=t+e*e8
        dg=0 if not cluster else (0 if (bar+j)%2==0 else 1)
        cc=to_deg(rot[(bar*3+j)%len(rot)],dg) if dg else rot[(bar*3+j)%len(rot)]
        place(norm(cc)[:int(e8*SR*1.6)],te,0.5*density,0.12*(-1)**j)
        place(piston(G1*2,0.3,1.0),te,0.4*density)
        if density>0.9: place(sawb(G1*2**(dg/12),e8*1.6,2.8*harsh),te,0.26*density)
    place(iron(196,0.22,1.0),t+4*e8,0.44*density)
    if bar%2==0:
        place(smetal(640,0.45,1.0),t,0.36*density); place(iron(98,0.28,1.0),t,0.32*density)
    else:
        place(smetal(640,0.4,1.0),t+5*e8,0.28*density); place(iron(98,0.26,1.0),t+5*e8,0.26*density)
    if bar%4==2: place(PRc[bar%len(PRc)],t+4*e8,0.42*density,-0.18)
    if bar%2==1 and not seven: place(iron(124,0.2,1.0),t+7*e8,0.34*density)
    if bar%8==0 and density>0.85: place(smetal(148,0.7,1.0),t,0.4)
    return t+n8*e8

def bpmramp(b0,b1,nbars):
    return [60/ (b0+(b1-b0)*k/max(1,nbars-1)) for k in range(nbars)]

# ============================================================
# d6 — THE RALLY: seven scenes, through-composed
# 1 HARBOR (doc, no processing) | 2 RUMOR (broken, D->Bb)
# 3 THE SPEECH (call-response rally, G vs Ab) | 4 THE HUNT
# (dry motor, transposing G-Bb-C-Eb) | 5 THE KILLING (pincer,
# saturated) | 6 THE PIECES (cavernous, Eb-C-Bb-Ab-G descent)
# 7 THE SEA (open air)
# ============================================================
from scipy.signal import fftconvolve
def anv(t,g):
    place(smetal(640,0.5,1.0),t,g*0.9); place(iron(98,0.3,1.0),t,g*0.8)
RT1w=rdspan("/home/claude/007/iv48/read_tulip_ch1.wav",60,88)
RNLb=rdspan("/home/claude/007/iv48/read_havelaar_nl.wav",300,314)
irC=rng.standard_normal(int(2.2*SR))*np.exp(-np.arange(int(2.2*SR))/(0.5*SR))
irC=lp(irC,3800)*0.25

# ============================================================
# d20 — LE CARNAGE (007, Movement IV) — written whole.
#
# I HARBOR  II RUMOR  III THE SPEECH  IV THE HUNT
# V THE KILLING  VI THE PIECES  VII THE SEA
#
# Rhymes binding the whole: the speech ASCENDS 0-3-5-7-8 and the
# relics DESCEND 8-5-3-1-0; the music box is seeded far off in the
# harbour, reigns above the killing, jams over the relics; the
# collapsing string sting recurs ever sooner through the approach
# and returns once, slowed, above the corpse; Ab is taught to the
# mob in the roars, fought in the killing, and is the fourth relic.
# One pulse runs 0:08 to 2:38 without stopping, becoming the rally.
# ============================================================
from scipy.signal import fftconvolve

def kal(f,dur=0.42,amp=1.0):
    n=int(dur*SR); t2=np.arange(n)/SR; y=np.zeros(n)
    for r_,a,dc in [(1.0,1.0,0.5),(2.9,0.5,0.22),(5.1,0.3,0.12),(7.8,0.18,0.07)]:
        if f*r_<SR/2-500: y+=a*np.sin(2*np.pi*f*r_*t2+rng.uniform(0,6))*np.exp(-t2/(dur*dc))
    cl=hp(rng.standard_normal(n),3200)*np.exp(-t2/0.0025)*0.5
    return declick(norm((y+cl)*np.minimum(1,t2/0.0012)))*amp
def rewind(f,dur=0.5):
    y=kal(f,dur,1.0)[::-1].copy(); n=len(y)
    idx=np.clip(np.cumsum(np.linspace(1.0,2.1,n)),0,n-2)
    i0=idx.astype(int); fr=idx-i0
    return norm(hp(y[i0]*(1-fr)+y[i0+1]*fr,700))*np.hanning(n)**0.6
def subthud(f,dur):
    n=int(dur*SR); t2=np.arange(n)/SR
    y=np.sin(2*np.pi*f*t2*(1+0.06*np.exp(-t2/0.05)))*np.exp(-t2/(dur*0.38))
    return declick(norm(np.tanh(y*2.2))*np.minimum(1,t2/0.004))
def pulpit(y):
    y=lp(hp(norm(y),260),3600)
    return norm(drv(y*0.85+np.roll(y,int(0.07*SR))*0.2,1.35))
irBox=hp(rng.standard_normal(int(0.85*SR))*np.exp(-np.arange(int(0.85*SR))/(0.15*SR)),1200)*0.3

ASC=[0,3,5,7,8]                      # the speech climbs
DESC=[8,5,3,1,0]                     # the relics fall

# ============ I. THE HARBOUR (0-38) ============
sea(0.0,104.0,0.12,0.08,fi=1.5,fo=18.0)
place(edge(pulpit(RT1w),0.5,1.5),4.0,0.5,0.0)
STR=lp(gstretch(norm(seg(haag,150.0,170.0)),7.6),3400)
tS_=np.arange(len(STR))/SR
place(STR*np.interp(tS_,[0,7,30,55,82,112,142,len(STR)/SR],[0,0.26,0.2,0.3,0.22,0.32,0.26,0.15]),6.0,1.0,0.18)
STRL=lp(to_deg(gstretch(norm(seg(haag,152.0,166.0)),6.4),12),1100)[:int(130*SR)]
tL_=np.arange(len(STRL))/SR
place(STRL*np.interp(tL_,[0,10,60,100,130],[0,0.2,0.26,0.22,0.1]),22.0,1.0,-0.22)
place(lp(kal(G2*8*2**(7/12),1.2,1.0),2600),26.5,0.15,0.32)      # the box, seeded, far off

# ---- the pulse: one accelerando, heart into rally ----
SLAB1,SLAB2=108.5,131.5
tt=8.0
while tt<157.0:
    prog=float(np.clip((tt-8.0)/149.0,0,1))
    iv=3.0*(0.4545/3.0)**(prog**1.35)
    g=0.2+0.24*prog
    place(subthud(G1,0.5+0.22*(1-prog)),tt,g*rng.uniform(0.92,1.08))
    if prog>0.30: place(iron(196,0.2,1.0),tt,0.09+0.15*prog)
    if tt>SLAB1:
        place(piston(G1*2,0.28,1.0),tt+iv*0.5,0.12+0.2*prog)
        place(subthud(G1,0.32),tt+iv*0.5,0.15+0.1*prog)
    if tt>SLAB2: place(iron(124,0.18,1.0),tt+iv*0.5,0.12+0.12*prog)
    tt+=iv*rng.uniform(0.94,1.06)

# ---- the collapsing sting, recurring ever sooner ----
STING_L=lp(spiral(norm(seg(haag,158.0,163.0)),1.2,grain=0.05,semis_start=0.0,semis_end=-14.0),3000)
STING_S=lp(spiral(norm(seg(haag,164.0,167.0)),1.15,grain=0.045,semis_start=2.0,semis_end=-12.0),3600)
for ts_,gg,sh in [(30.0,0.24,0),(58.0,0.26,0),(80.0,0.30,0),(101.0,0.30,1),
                  (118.0,0.32,1),(154.5,0.42,1)]:
    S2=STING_S if sh else STING_L
    place(edge(S2,0.12,1.0),ts_,gg,0.12*(-1)**sh)
    place(subthud(G1,1.05),ts_+len(S2)/SR-0.25,gg*0.55)

# ============ II. THE RUMOUR (38-86) ============
RT1c=rdspan("/home/claude/007/iv48/read_tulip_ch1.wav",88,116)
V2=pulpit(RT1c)
place(edge(V2[:int(22.5*SR)],1.2,0.7),40.0,0.48,-0.04)
VWAIL=lp(drv(spiral(V2[int(21.0*SR):int(27.0*SR)],1.9,grain=0.04,semis_start=3.0,semis_end=-13.0),1.5),3200)
place(norm(VWAIL)*np.interp(np.arange(len(VWAIL))/SR,[0,0.4,4,8,11.4],[0,0.44,0.4,0.24,0]),61.9,1.0,-0.04)
BURN1=sizzle_long(np.concatenate([CH1_MAN,VB]))
b1=np.tile(BURN1,int(124*SR)//len(BURN1)+2)[:int(124*SR)]
place(b1*np.interp(np.arange(len(b1))/SR,[0,22,64,100,124],[0.02,0.07,0.16,0.26,0.3]),42.0,1.0,0.08)
MOAN=lp(gtr(seg(ffs1w,258.0,272.0)),2000)
place(norm(MOAN)*np.interp(np.arange(len(MOAN))/SR,[0,5,11,14],[0,0.2,0.18,0]),64.0,1.0,-0.28)
place(lp(kal(G2*8*2**(3/12),1.0,1.0),2400),74.5,0.13,-0.3)       # the box, seeded again

# ============ III. THE SPEECH (86-166) ============
OR=np.concatenate([seg(chern1,106.0,126.0),seg(chern1,138.0,150.0),
                   seg(chern1,16.0,24.0),seg(chern1,145.0,150.0)])
vo=pulpit(OR)*0.8+efx(OR)*0.35
nv=len(vo); tv=np.arange(nv)/SR; LV=nv/SR
env=np.interp(tv,[0,0.7,8,8.9,16.5,17.4,26,26.9,33,33.9,40,40.9,LV-1.2,LV],
                 [0,1,1,0.72,1,0.74,1,0.75,1,0.78,1,0.76,0.9,0])
for dk in (SLAB1-88.0,SLAB2-88.0):
    env[(tv>dk-2.6)&(tv<dk)]*=1.15
    env[(tv>dk)&(tv<dk+1.3)]*=0.62
VOH=norm(drv(hp(OR,300),2.1))*0.9+efx(OR)*0.5                   # the inflamed voice
msk=np.clip((tv-(SLAB1-88.0))/1.4,0,1)
place(vo*np.clip(env,0,1.15)*(1-0.5*msk),88.0,0.56,0.0)
place(VOH*np.clip(env,0,1.15)*msk,88.0,0.5,0.0)
place(pulpit(RNLa)[:int(11*SR)]*0.5,92.5,0.55,-0.14)
place(pulpit(RNLb)[:int(11*SR)]*0.5,119.0,0.55,0.14)
AH1=pulpit(VS[5])[:int(0.34*SR)]
AH1=norm(AH1)*np.interp(np.arange(len(AH1))/SR,[0,0.03,0.17,0.34],[0,1,0.78,0])
AH2=lp(AH1,2400)
def ah_round(t0,t1,per,g,pan,dark=False):
    tt=t0
    while tt<t1:
        place((AH2 if dark else AH1)*rng.uniform(0.9,1.07),tt,g,pan)
        tt+=per
ah_round(96.0,SLAB1-0.35,1.00,0.19,0.0,dark=True)               # half-rate before the blow
ah_round(SLAB1+0.04,134.0,0.50,0.27,0.0)                        # the blow doubles it, and opens it
ah_round(136.0,158.5,0.500,0.27,-0.25)                          # the second round: two loops,
ah_round(136.0,158.5,0.531,0.23,0.25)                           # phasing apart and back
ROAR=np.zeros(int(1.0*SR))
for c in VS[:14]:
    cc=pulpit(c)[:int(0.9*SR)]; ROAR[:len(cc)]+=cc*0.16
ROAR=norm(ROAR+norm(VB2)[:int(1.0*SR)]*0.7)
ROAR_Ab=to_deg(norm(ROAR),1)
for k,(ts_,d_) in enumerate(zip([88.0,97.0,106.0,115.0,124.0],ASC)):
    SW=lp(to_deg(gstretch(VB2,1.35),d_),3400)[:int(10.0*SR)]
    place(norm(SW)*np.interp(np.arange(len(SW))/SR,[0,5.5,10],[0,0.3,0.36]),ts_,1.0,-0.18)
def slab(t0,tail=False,big=False):
    LN=1.65 if big else 1.2
    pe=norm(VB2)[:int(1.15*SR)][::-1].copy()                              # the blow, heard backwards first
    place(pe*np.linspace(0,1,len(pe))**1.8*(0.34 if big else 0.26),t0-1.15,1.0,0.0)
    place(norm(drv(sawb(G1,LN,3.4),1.4)),t0,0.6,0.0)
    place(norm(VB2)[:int(LN*SR)],t0+0.012,0.58,-0.1)
    place(edge(BITE1[:int((LN+0.1)*SR)],0.01,0.3),t0+0.006,0.55,0.12)
    anv(t0,0.5); anv(t0+0.62,0.4)
    if big:
        SMR=spiral(norm(VB2),1.5,grain=0.05,semis_start=0.0,semis_end=-2.0)   # the chord smears down after
        place(edge(SMR,0.05,1.0)*0.4,t0+0.9,1.0,-0.05)
        place(subthud(G1,1.6),t0+0.9,0.36)
    else:
        place(iron(196,0.3,1.0),t0+0.52,0.34,0.2)                             # the blow answers itself
        place(iron(124,0.28,1.0),t0+0.86,0.26,-0.2)
    if tail:
        FBT=lp(gtr(seg(ffs1w,252.0,258.0)),1800)
        place(norm(FBT)*np.interp(np.arange(len(FBT))/SR,[0,0.5,4,6],[0.34,0.3,0.12,0]),t0+1.5,1.0,0.15)
def prep(t0):
    t4=t0-2.6; iv2=0.24
    while t4<t0-0.05:
        place(iron(196,0.14,1.0),t4,0.16+0.14*(1-(t0-t4)/2.6))
        t4+=iv2; iv2=max(iv2*0.9,0.085)
    CR=lp(sawb(G1,2.6,2.6),900)
    place(norm(CR)*np.interp(np.arange(len(CR))/SR,[0,2.0,2.6],[0,0.26,0.34]),t0-2.6,1.0,-0.05)
    burst(PRc,t0-1.35,[0,3,5,7,8,10,12],0.185,g=0.46,pan=0.15)
for k,(tr_,ab) in enumerate([(95.5,0),(104.0,0),(113.5,1),(122.0,1),(130.0,1)]):
    base=ASC[min(4,int((tr_-88.0)/9.0))]
    burst(PRc,tr_-0.55,[base,base+3,base+5,base+7],0.16,g=0.42,pan=0.2)
    place(edge(ROAR_Ab if ab else ROAR,0.02,0.35),tr_,0.62,0.0)
    anv(tr_,0.4)
prep(SLAB1); slab(SLAB1)
prep(SLAB2); slab(SLAB2,tail=True,big=True)
STRH=lp(to_deg(gstretch(norm(seg(haag,150.0,162.0)),4.4),-12),6000)   # after slab I: the strings climb
place(STRH*np.interp(np.arange(len(STRH))/SR,[0,3,18,26.4],[0,0.18,0.2,0.12]),SLAB1,1.0,0.26)
BURNb=sizzle_long(np.concatenate([seg(roto,196.0,204.0),CH1_MAN]))    # after slab II: the fire steps
bb=np.tile(BURNb,int(32*SR)//len(BURNb)+2)[:int(32*SR)]
place(bb*np.interp(np.arange(len(bb))/SR,[0,3,22,32],[0,0.14,0.2,0.22]),SLAB2,1.0,-0.1)
CROWD=lp(gstretch(np.concatenate([ROAR,ROAR_Ab,ROAR]),8.0),3200)     # one mass, rising, no debris
tC=np.arange(len(CROWD))/SR
place(norm(CROWD)*np.interp(tC,[0,7,17,22,24],[0,0.2,0.3,0.34,0.22]),133.5,1.0,0.0)
burst(PRc,149.4,[0,3,5,7,8,10],0.17,g=0.44,pan=0.18)                  # one run, into one peak
place(edge(ROAR,0.02,0.45),150.6,0.62,0.0)
place(edge(ROAR_Ab,0.02,0.5),152.1,0.5,0.12)
t5=158.0; iv3=0.30                                                     # the stumble
while t5<166.0:
    place(iron(196,0.16,1.0),t5,0.24+0.05*(t5-158.0))
    place(subthud(G1,0.34),t5,0.2)
    t5+=iv3*rng.uniform(0.72,1.2); iv3=max(iv3*0.88,0.095)
TREM=lp(gstretch(norm(seg(haag,166.0,170.0)),1.9),4200)
place(norm(TREM)*np.interp(np.arange(len(TREM))/SR,[0,3,6,7.6],[0,0.28,0.44,0.3]),158.4,1.0,0.22)

# ============ IV. THE HUNT (166-240) ============
tH=166.0
place(WCRASH,tH-0.35,0.5,0.05)
place(smetal(196,0.9,1.0),tH,0.55)
place(norm(drv(sawb(G1,1.6,3.4),1.4)),tH,0.62,0.0)
place(norm(VB2)[:int(1.5*SR)],tH+0.012,0.6,-0.08)
place(edge(BITE1[:int(1.5*SR)],0.01,0.35),tH+0.006,0.56,0.12)
place(subthud(G1,1.5),tH,0.52); anv(tH,0.55); anv(tH+0.44,0.4)
BURNh=sizzle_long(np.concatenate([CH1_MAN,seg(roto,196.0,206.0),VB2]))
bh=np.tile(BURNh,int(80*SR)//len(BURNh)+2)[:int(80*SR)]
place(bh*np.interp(np.arange(len(bh))/SR,[0,10,46,80],[0.26,0.32,0.36,0.42]),tH-1.5,1.0,0.06)
wall_gate(seg(roto,10.0,26.0),tH+3.0,66.0,0.4,60/166.0/2,duty=0.5,pan=0.1)
TRANS=[0,0,0,3,3,5,5,5,8,8,3,1,0,0,0]
def mbar(t,bar,bt,tr,density=1.0,harsh=1.0,seven=False,anvils=True):
    e8=bt/2; grid=GRIDS[(bar//3)%3]; rot=VBc if bar%4<3 else VB2c; n8=7 if seven else 8
    for j,e in enumerate(grid):
        if e>=n8: continue
        te=t+e*e8
        place(norm(to_deg(rot[(bar*3+j)%len(rot)],tr))[:int(e8*SR*1.6)],te,0.5*density,0.12*(-1)**j)
        place(piston(G1*2**(tr/12)*2,0.3,1.0),te,0.4*density)
        if density>0.9: place(sawb(G1*2**(tr/12),e8*1.6,2.8*harsh),te,0.26*density)
    place(iron(196,0.22,1.0),t+4*e8,0.42*density)
    if anvils:
        if bar%2==0: anv(t,0.34*density)
        else: anv(t+5*e8,0.26*density)
    if bar%4==2: place(to_deg(PRc[bar%len(PRc)],tr),t+4*e8,0.4*density,-0.18)
    return t+n8*e8
t=tH+0.2; bar=0
HB=bpmramp(158,170,50)
sevens={5,11,20,27,34,41,47}
storms={6:(VS,[0,3,5,7],0.52,0.14),14:(VS[9:],[8,7,5,3],0.5,-0.12),20:(RSy,[0,3,0],0.48,-0.2),
        31:(VS,[7,8,7,5,3],0.52,0.12),38:(TSy,[0,3,0,5],0.46,0.2),45:(VS[6:],[0,5,3,0],0.52,0.1)}
tR=None
for bi,bt in enumerate(HB):
    tb=t
    if bi==24: tR=tb
    tr=TRANS[(bi//3)%len(TRANS)]
    dens=0.72 if (24<=bi<41) else 1.0
    if bi<6: t+=rng.uniform(-0.02,0.055)
    t=mbar(t,bar,bt,tr,dens,1.15,seven=(bi in sevens)); bar+=1
    if bi in storms:
        cells,degs,gg,pn=storms[bi]
        burst(cells,tb,[d+tr for d in degs],bt/2,g=gg,pan=pn)
    if bi==17: place(edge(spiral(HAAGS,2.1,grain=0.05,semis_start=2.0,semis_end=-8.0),0.25,0.9),tb,0.4,0.25)
    if bi==29: place(edge(spiral(TUL,2.0,grain=0.05,semis_start=0.0,semis_end=-9.0),0.25,0.9),tb,0.4,-0.2)
    if bi==43: place(edge(gstretch(CRY1,1.4),0.6,1.2)*0.48,tb,1.0,0.15)
place(norm(lp(RIFF_A[:int(3.0*SR)],1300))*np.interp(np.arange(int(3.0*SR))/SR,[0,1.5,3],[0,0.3,0.42]),tR-3.0,1.0,-0.08)
place(edge(RIFF_A,0.15,0.4),tR,0.5,-0.08)                              # the riff, whole
place(edge(norm(drv(to_deg(RIFF_A,3),1.4)),0.1,0.4),tR+12.4,0.52,0.08) # again, up a third
RTL=lp(RIFF_A[-int(3.2*SR):],1400)
place(norm(RTL)*np.interp(np.arange(len(RTL))/SR,[0,1,3.2],[0.4,0.3,0]),tR+24.8,1.0,0.0)
T5=t

# ============ V. THE KILLING ============
BURNK=sizzle_long(np.concatenate([CH1_MAN,seg(roto,194.0,206.0),VB2]))
bk=np.tile(BURNK,int(66*SR)//len(BURNK)+2)[:int(66*SR)]
place(bk*np.interp(np.arange(len(bk))/SR,[0,20,48,66],[0.34,0.44,0.56,0.5]),T5,1.0,-0.06)
wg=np.tile(norm(WALLG-0.28*bp(WALLG,2100,3700)),int(52*SR)//len(WALLG)+2)[:int(52*SR)]
place(wg*np.interp(np.arange(len(wg))/SR,[0,6,40,52],[0,0.3,0.34,0.24]),T5+4.0,1.0,-0.12)
wall_gate(seg(chern3,140.0,152.0),T5+8.0,50.0,0.2,60/168.0/2,duty=0.32,pan=-0.15,band=(400,6500))
HALO=lp(hp(gstretch(to_deg(latticize(seg(haag,154.0,162.0),150,900),-12),3.2),700),4200)
tHa=np.arange(len(HALO))/SR
place(norm(HALO)*np.interp(tHa,[0,4,10,16,22,25.6],[0,0.28,0.2,0.3,0.22,0]),T5+3.0,1.0,0.3)
place(norm(HALO)*np.interp(tHa,[0,5,11,17,23,25.6],[0,0.24,0.3,0.2,0.26,0]),T5+27.0,1.0,-0.3)
place(edge(RIFF_B,0.2,0.5),T5+2.0,0.46,0.1)
place(edge(norm(drv(to_deg(RIFF_B,1),1.3)),0.15,0.5),T5+17.5,0.48,-0.1)
BIGSP=spiral(np.concatenate([VAD[:int(6*SR)],VB2,CH1_MAN]),1.5,grain=0.05,semis_start=0.0,semis_end=-12.0)
place(edge(BIGSP,0.4,1.5)*0.46,T5+30.0,1.0,0.0)
KFR=[]; KBARS=[]; GAPS=[]
t=T5; barc=0
KB=[60/168.0]*26+bpmramp(168,146,12)
for bi,bt in enumerate(KB):
    tb=t
    t=mbar(t,barc,bt,(0 if bi%2==0 else 1),0.92,1.6,seven=(bi in (7,17,29)),anvils=False); barc+=1
    KBARS.append((tb,bi,bt))
    if bi%2==0: burst(VS[10:],tb,[0,1,0,1],bt/2,g=0.5,pan=0.12*(-1)**bi)
    if bi>=24 and bi%2==0 and rng.random()>0.2+0.6*(bi-24)/14:
        place(edge(to_deg(RIFF_B[:int(1.3*SR)],(0 if bi%4==0 else 1)),0.02,0.3),tb,0.44,0.1)
    if bi in (12,24):
        gd=0.36 if bi==12 else 0.72                              # a stumble, then a true caesura
        KFR.append(t); GAPS.append((t,gd)); CUTS.append((t,gd)); t+=gd
# ---- the music box: seeded in the harbour, sovereign here ----
NEWKEY=[[8,0,3],[1,5,8]]                                   # Eb, then Ab major: where the box lands
for fi,(f,gd) in enumerate(GAPS):
    place(rewind(G2*8*2**(NEWKEY[fi][0]/12),0.95),f-0.98,0.5,0.1)       # the sweep
    for j,off in enumerate([0.62,0.40,0.22]):                            # the brake: three strikes, slowing
        place(iron(196,0.16,1.0),f-off,0.28+0.09*j,0.12*(-1)**j)
    place(subthud(G1,0.5),f-0.22,0.34)
    place(norm(to_deg(VB2,NEWKEY[fi][0]))[:int(0.95*SR)],f+gd,0.5,-0.1)  # the orchestra affirms the new root
    place(norm(drv(sawb(G1*2**(NEWKEY[fi][0]/12),0.85,3.0),1.3)),f+gd,0.4,0.0)
    for j,d in enumerate(NEWKEY[fi]):
        place(kal(G2*4*2**(d/12),0.95,1.0),f+gd+0.1+j*0.27,0.5-0.05*j,0.22*(-1)**j)
def boxn(f2,d2,hard):
    y=kal(f2,d2,1.0); n2=len(y)
    if hard>0.02:
        t2=np.arange(n2)/SR
        wow=1.0+(0.004+0.010*hard)*np.sin(2*np.pi*(0.5+0.6*hard)*t2+rng.uniform(0,6))
        idx=np.clip(np.cumsum(wow),0,n2-2); i0=idx.astype(int); fr=idx-i0
        y=y[i0]*(1-fr)+y[i0+1]*fr
        y=np.tanh(hp(y,240+300*hard)*(1.0+1.9*hard))
        wd=np.sin(2*np.pi*f2*0.25*t2)*np.exp(-t2/0.032)*0.42*hard
        ck=hp(rng.standard_normal(n2),4200)*np.exp(-t2/0.0011)*0.5*hard
        y=y+bp(wd,120,900)+ck
        tl=fftconvolve(y,irBox)[:n2+int(0.85*SR)]
        y=np.pad(y,(0,len(tl)-n2))*(1.0-0.18*hard)+norm(tl)*(0.34*hard)
        y=norm(y)
    return y
F0=KFR[0] if KFR else T5+18.0
PHASES=[([7,3,0,7,10,3],8,1.92),        # I  G minor, high and spacious
        ([8,3,0,8,7,12],4,1.66),        # II Eb, the relative major
        ([5,8,3,0,5,8],4,1.48),         # III C minor with its seventh — the richest ground
        ([7,1,7,2,7,1],4,2.10)]         # IV D against Ab: the pincer, unresolved
DPH=T5+47.0
tK=T5+7.0; kn=0
BOXEND=(KFR[1]-1.2) if len(KFR)>1 else T5+55.5
while tK<BOXEND:
    inb=[f for f in KFR if f-1.15<=tK<=f+1.05]
    if inb: tK=max(inb)+1.15; kn+=1; continue
    nfr=sum(1 for f in KFR if tK>f)
    ph=3 if tK>=DPH else min(nfr,2)
    pool,oct_,bstr=PHASES[ph]
    stride=bstr*rng.uniform(0.97,1.03)
    kd=pool[kn%len(pool)]
    hard=0.0 if tK<F0 else float(np.clip((tK-F0)/24.0,0,1))
    dur=0.95-0.15*hard
    g=(0.42+0.06*min(ph,2))*(1.0+0.10*hard)
    det=1.0 if ph<2 else (1+rng.uniform(-0.005,0.005))
    bx=lambda f2,d2,a2: boxn(f2,d2,hard)*a2
    place(bx(G2*oct_*2**(kd/12)*det,dur,1.0),tK,g,0.3*(-1)**kn)
    place(bx(G2*oct_*2*2**(kd/12)*det,dur*0.7,1.0),tK+0.014,g*0.3,-0.3*(-1)**kn)
    if ph>=1 and rng.random()<0.55:
        iv=int(rng.choice([3,4,7]))                                     # the consonant companion
        place(bx(G2*oct_*2**((kd+iv)/12)*det,dur*0.85,1.0),tK+0.03,g*0.44,-0.26*(-1)**kn)
    if ph==2 and rng.random()<0.5:                                      # the chord, at the peak
        for j,iv in enumerate([3,7]):
            place(bx(G2*oct_*2**((kd+iv)/12)*det,dur*0.7,1.0),tK+0.27+j*0.25,g*0.4,0.22*(-1)**j)
    if hard>0.3 and rng.random()<0.16:
        place(bx(G2*oct_*2**((kd+1)/12)*det,dur*0.8,1.0),tK+0.028,g*0.3,-0.28*(-1)**kn)
    if hard>0.45 and rng.random()<0.3:
        rt=hp(rng.standard_normal(int(0.05*SR)),3000)*np.exp(-np.arange(int(0.05*SR))/(0.006*SR))
        place(norm(rt),tK-0.055,g*0.3*hard,0.16)
    tK+=stride; kn+=1
# the box joins the war: G minor against Ab major, bar by bar, on the grid
GM=[0,3,7]; AB=[1,5,8]
F2=KFR[1] if len(KFR)>1 else None
if F2 is not None:
    B2=[(tb,bi,bt) for (tb,bi,bt) in KBARS if tb>F2+1.0]
    nb=max(1,len(B2)-1)
    for idx,(tb,bi,bt) in enumerate(B2):
        prog2=idx/nb
        tri=AB if bi%2==1 else GM
        e8=bt/2
        gg=0.46+0.14*prog2
        hd=0.85+0.15*prog2
        dl=1.5+0.9*prog2
        place(boxn(G2*2*2**(tri[0]/12),dl,hd)*gg*0.92,tb,1.0,0.0)                    # the pedal takes the bar's root
        if idx<len(B2)-3:
            for j,d in enumerate(tri):                                               # the triad, arpeggiated
                place(boxn(G2*8*2**(d/12),0.95+0.25*prog2,hd)*gg*(0.92-0.14*j),tb+j*e8,1.0,0.28*(-1)**j)
            if idx%2==1:
                place(boxn(G2*4*2**(tri[2]/12),1.1,hd)*gg*0.5,tb+3*e8,1.0,-0.24)     # the fifth, answering
        else:
            for j in range(4):                                                        # the war compresses to two teeth
                d=1 if j%2 else 0
                place(boxn(G2*8*2**(d/12),0.8,hd)*gg*0.92,tb+j*e8,1.0,0.3*(-1)**j)
                place(boxn(G2*2*2**(d/12),0.9,hd)*gg*0.5,tb+j*e8,1.0,0.0)
T6=t
place(rewind(G2*8,1.1),T6-1.15,0.5,0.0)                                               # the sweep into the silence
CUTS.append((T6,1.4))
for kd,td in [(0,1.55),(1,1.95),(0,2.35)]:                              # the box alone in the void
    place(kal(G2*8*2**(kd/12),0.7,1.0),T6+td,0.3,0.15*(-1)**kd)

# ============ VI. THE PIECES ============
tP=T6+3.2
EMB=sizzle_long(CH1_MAN)
ep=np.tile(EMB,int(58*SR)//len(EMB)+2)[:int(58*SR)]
place(ep*np.interp(np.arange(len(ep))/SR,[0,20,45,58],[0.28,0.3,0.22,0.12]),tP,1.0,0.06)
GLAM=spiral(lp(gstretch(BITE2,2.4),2600),1.9,grain=0.055,semis_start=0.0,semis_end=-8.0)
place(norm(GLAM)*np.interp(np.arange(len(GLAM))/SR,[0,3,14,24,27.4],[0,0.3,0.32,0.26,0]),tP+1.0,1.0,-0.12)
CHT=[]; gaps=[9.0,8.5,8.0,7.5,15.0]
for k,dg in enumerate(DESC):
    ch=to_deg(norm(VB2),dg)[:int(1.3*SR)]
    chc=norm(np.concatenate([ch,fftconvolve(ch,irC)[len(ch):len(ch)+int(1.8*SR)]]))
    place(edge(chc,0.02,1.2),tP,0.55,0.0)
    deb=spiral(to_deg(VB2,dg),1.9,grain=0.05,semis_start=0.0,semis_end=-9.0)
    place(edge(deb,0.2,1.0)*(0.36-0.03*k),tP+1.4,1.0,0.2*(-1)**k)
    if k in (1,3):
        src=RNLa if k==1 else RNLb
        for j,c in enumerate([norm(src[m*int(0.18*SR):(m+1)*int(0.18*SR)]) for m in range(8,13+k)]):
            place(mobtongue(c)*0.32,tP+2.2+j*0.21,1.0,-0.15)
    CHT.append((tP,dg)); tP+=gaps[k]
tJ=CHT[0][0]+1.2; jn=0; stuck=0; skd=7
JEND=min(tP-0.5,CHT[0][0]+29.5)
while tJ<JEND:
    root=0
    for (tc,dg2) in CHT:
        if tJ>=tc: root=dg2
    if stuck>0: kd=skd; stuck-=1
    else:
        kd=rng.choice([7,8,5,3,2,10,7])
        if rng.random()<0.3: kd+=rng.choice([-1,1])
        if rng.random()<0.16: stuck=rng.integers(2,5); skd=kd
    fJ=G2*4*2**((kd+root)/12)*(1+0.003*jn/40)
    place(kal(fJ,1.0,1.0),tJ,0.17,0.22*(-1)**jn)
    if rng.random()<0.14: place(kal(fJ*2**(1/12),0.9,1.0),tJ+0.02,0.13,-0.22*(-1)**jn)
    if rng.random()<0.13: place(rewind(fJ*2,0.8),tJ+0.34,0.12,-0.2*(-1)**jn)
    tJ+=rng.uniform(0.72,1.15); jn+=1
place(rewind(G2*4*2**(3/12),1.0),JEND+0.4,0.14,-0.15)
STING_END=lp(spiral(norm(seg(haag,158.0,164.0)),2.4,grain=0.06,semis_start=-2.0,semis_end=-17.0),2200)
place(norm(STING_END)*np.interp(np.arange(len(STING_END))/SR,[0,2,8,14.4],[0,0.26,0.2,0]),tP-9.0,1.0,0.1)
place(subthud(G1,1.4),tP-9.0+len(STING_END)/SR-0.3,0.16)               # the sting returns, slowed

# ============ VII. THE SEA ============
sea(tP-4.0,64.0,0.11,0.08,fi=6.0,fo=9.0)
place(edge(ALKB,0.5,2.0),tP+6.0,0.22,0.3)
place(edge(ALKB,0.5,2.5),tP+19.0,0.18,-0.3)
place(edge(readv(RT2b),0.8,2.0),tP+26.0,0.42,0.0)
place(edge(ALKB,0.3,3.0),tP+48.0,0.13,0.1)
END=int(min(DUR,tP+56.0)*SR)
for (tc,dc) in CUTS:
    a=int(tc*SR); b=int((tc+dc)*SR); dn=int(0.0015*SR)
    Lc[a-dn:a]*=np.linspace(1,0,dn); Rc[a-dn:a]*=np.linspace(1,0,dn)
    Lc[a:b]=0.0; Rc[a:b]=0.0
    Lc[b:b+dn]*=np.linspace(0,1,dn); Rc[b:b+dn]*=np.linspace(0,1,dn)
for fi,(gt,gd) in enumerate(GAPS):
    ln=int(gd*SR); ch=np.zeros(ln)
    for d in NEWKEY[fi]:
        c=kal(G2*4*2**(d/12),gd*0.98,1.0)
        ch[:min(ln,len(c))]+=c[:min(ln,len(c))]*0.5
    ch=norm(ch)[::-1].copy()
    place(ch*np.linspace(0,1,ln)**1.7*0.36,gt,1.0,0.0)                   # the chord heard in reverse, inside the gap
Lc=Lc[:END]; Rc=Rc[:END]
fadeN=int(9*SR); Lc[-fadeN:]*=np.linspace(1,0,fadeN)**1.2; Rc[-fadeN:]*=np.linspace(1,0,fadeN)**1.2
st=np.stack([Lc,Rc],1); st=norm(st)*10**(-1.0/20)
sf.write("/mnt/user-data/outputs/007_mvt4_d26.wav",st,SR,subtype="PCM_24")
print(f"IV d26 (the first slab remakes the section): {len(st)/SR:.1f}s | harbour 0-38 | rumour 38-86 | speech 86-166 | hunt 166-{T5:.0f} | killing {T5:.0f}-{T6:.0f} | pieces {T6:.0f}-{tP:.0f} | sea {tP:.0f}+")
