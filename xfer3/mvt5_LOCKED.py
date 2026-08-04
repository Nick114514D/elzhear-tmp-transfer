"""
007 — Movement V: L'INVENTAIRE. v6 — the electronics in front: sweeps, fasce, tape.

THE ROW. Begin on A and alternate outward by expanding intervals:
+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11. The wedge closes on all
twelve pitches and every interval class exactly once:
   A  Bb  Ab  B  G  C  F#  C#  F  D  E  Eb
The album's forbidden A is the origin of the fan. Registers fan with it:
odd members climb, even members descend, so the wedge is audible as shape.

STRUCTURE — cantata in nine, after Il canto sospeso:
  1 INTRODUZIONE     the row stated as a fan, one instrument per member
  2 CORO I           the word dispersed across seven throats, on row pitches
  3 SOLO I           viola sul C alone: the row entire, pp
  4 CONTRAPPUNTO     after Contrappunto dialettico alla mente — filters,
                     micro-intervallic sound-bands, layering, overlapping,
                     intercutting, collage. Tape splices, not fades.
  5 CORO II          the word again, on the retrograde
  6 SOLO II          bells and crotales: the row at the top of hearing
  7 FASCE            sound-bands: the field of A opened microtonally
  8 NON IN UNA TECA  Julia Culp 1917, untouched, the row haloing her
  9 CONGEDO          the wedge closes to A; Ab once; the needle lifts

Instruments are played from chromatic banks sliced out of the Iowa runs:
viola sul C, tenor trombone, tuba, and both bell octaves each yield a
complete twelve.
"""
import numpy as np, soundfile as sf, os, subprocess
from scipy.signal import butter, sosfilt, fftconvolve
SR=48000; rng=np.random.default_rng(1672)
def hp(x,f,o=2): return sosfilt(butter(o,f,btype='high',fs=SR,output='sos'),x)
def lp(x,f,o=2): return sosfilt(butter(o,min(f,SR/2-200),btype='low',fs=SR,output='sos'),x)
def bp(x,lo,hi,o=2):
    lo=max(20,lo); hi=min(hi,SR/2-200)
    if hi<=lo*1.02: hi=lo*1.06
    return sosfilt(butter(o,[lo,hi],btype='band',fs=SR,output='sos'),x)
def drv(x,a): return np.tanh(x*a)/np.tanh(a)
def norm(y):
    return y/(np.max(np.abs(y))+1e-9) if len(y) else np.zeros(1)
def rp(y,semis):
    rate=2**(semis/12.0); nn=max(4,int(len(y)/rate)); idx=np.arange(nn)*rate
    i0=np.clip(idx.astype(int),0,len(y)-2); fr=idx-i0
    return y[i0]*(1-fr)+y[i0+1]*fr
def ct(y,c): return rp(y,c/100.0)
def edge(y,fi=0.05,fo=0.4):
    y=y.copy(); a=min(int(fi*SR),len(y)//2); b=min(int(fo*SR),len(y)//2)
    if a>0: y[:a]*=np.linspace(0,1,a)**0.8
    if b>0: y[-b:]*=np.linspace(1,0,b)**1.2
    return y
def splice(y):
    w=int(0.002*SR)
    if len(y)<2*w: return y
    y=y.copy(); y[:w]*=np.linspace(0,1,w); y[-w:]*=np.linspace(1,0,w); return y
def sustain(y,dur,xf=0.35):
    n=int(dur*SR); core=y[int(0.04*len(y)):]
    if len(core)<4096: core=np.tile(core,8)
    w=max(64,min(int(xf*SR),len(core)//3))
    if len(core)<3*w:
        core=np.tile(core,4); w=max(64,min(w,len(core)//3))
    Ln=len(core); out=np.zeros(n+Ln+8); pos=0
    while pos<n:
        seg=core.copy()
        seg[:w]*=np.linspace(0,1,w)            # linear pair: the overlap sums to exactly 1
        seg[-w:]*=np.linspace(1,0,w)
        out[pos:pos+Ln]+=seg
        pos+=max(1,Ln-w)
    return norm(out[:n])

M="/home/claude/007/mis/"; I4="/home/claude/007/iv48/"; A4="/home/claude/007/alt48/"; DW="/home/claude/007/dewitt/"
def L(p):
    y,_=sf.read(p); return y[:,0] if y.ndim>1 else y
def sp(path,a,b):
    y,_=sf.read(path,start=int(a*SR),frames=int(max(0.02,b-a)*SR),dtype='float32')
    if y.ndim>1: y=y[:,0]
    return norm(y.astype(np.float64))
_FLEN={}
def FB(name,a,b):
    p=M+f"IOWA_fob_{name}.flac"
    if p not in _FLEN: _FLEN[p]=sf.info(p).duration
    d=_FLEN[p]
    if b>d-0.05:
        span=max(0.4,b-a); b=d-0.05; a=max(0.0,b-span)
    if b<=a+0.05: a=max(0.0,d-1.2); b=d-0.05
    return sp(p,a,b)
def conv_if_needed(src,dst):
    if not os.path.exists(dst): subprocess.run(["ffmpeg","-y","-i",src,"-ar","48000","-ac","1",dst],capture_output=True)
    return dst
def notes_of(y,thr=0.05,minlen=0.35):
    w=int(0.03*SR); e=np.convolve(np.abs(y),np.ones(w)/w,'same')
    on=(e>thr*np.max(e)).astype(np.int8); d=np.diff(on)
    st=list(np.where(d==1)[0]); en=list(np.where(d==-1)[0])
    if on[0]==1: st=[0]+st
    if on[-1]==1: en=en+[len(on)-1]
    return [y[max(0,s-int(0.03*SR)):min(len(y),e2+int(0.6*SR))].copy() for s,e2 in zip(st,en) if (e2-s)/SR>=minlen]

# ---------- chromatic banks from the Iowa runs ----------
def bank(path,lo_midi,count):
    y=L(path); ns=notes_of(y)
    if len(ns)<count*0.75:
        c=len(y)//count; ns=[y[i*c:(i+1)*c] for i in range(count)]
    return {lo_midi+i:norm(ns[i]) for i in range(min(count,len(ns)))}
BK={}
BK["vla"]=bank(M+"IOWA_vla1_sulC.pp.C3B3.flac",48,12)
BK["tbn"]=bank(M+"IOWA_tbn2_TenorTrombone.ff.C4B4.flac",60,12)
BK["tuba"]=bank(M+"IOWA_tuba1_Tuba.ff.C1B1.flac",24,12)
BK["bell5"]=bank(M+"IOWA_bell1_plastic.pp.C5B5.flac",72,12)
BK["bell6"]=bank(M+"IOWA_bell2_plastic.pp.C6B6.flac",84,12)
BK["vlnD"]=bank(M+"IOWA_vln2_Violin.arco.pp.sulD.B4A5.flac",71,11)
BK["vlnA"]=bank(M+"IOWA_vln3_Violin.arco.pp.sulA.Bb5Ab6.flac",82,11)
BK["vlnG"]=bank(M+"IOWA_vln1_Violin.arco.pp.sulG.G3B3.flac",55,5)
BK["vlaD"]=bank(M+"IOWA_vla2_sulD.pp.D4B4.flac",62,10)
BK["vlc"]=bank(M+"IOWA_vlc1_Cello.arco.pp.sulC.C2Gb2.flac",36,7)
BK["tbn2"]=bank(M+"IOWA_tbn1_TenorTrombone.ff.E2B2.flac",40,8)
BK["btb"]=bank(M+"IOWA_btbn1_BassTrombone.ff.Db1B1.flac",25,11)
BK["tpt"]=bank(M+"IOWA_tpt1_Trumpet.vib.ff.E3B3.flac",52,8)
BK["tptn"]=bank(M+"IOWA_tpt2_Trumpet.novib.ff.E3B3.flac",52,8)
BK["pno"]={36:L(M+"IOWA_pno_C2_pp.flac"),43:L(M+"IOWA_pno_G2_pp.flac"),51:L(M+"IOWA_pno_Eb3_pp.flac"),
           58:L(M+"IOWA_pno_Bb3_pp.flac"),65:L(M+"IOWA_pno_F4_pp.flac"),72:L(M+"IOWA_pno_C5_pp.flac"),
           80:L(M+"IOWA_pno_Ab5_pp.flac"),87:L(M+"IOWA_pno_Eb6_pp.flac")}
BK["crot"]={84:L(M+"IOWA_crot1_crotale.C6.pp.flac"),89:L(M+"IOWA_crot2_crotale.F6.pp.flac"),
            94:L(M+"IOWA_crot3_crotale.Bb6.pp.flac"),99:L(M+"IOWA_crot4_crotale.Eb7.pp.flac"),
            104:L(M+"IOWA_crot5_crotale.Ab7.pp.flac")}
def note(inst,midi,cents=0.0):
    b=BK[inst]; k=min(b.keys(),key=lambda x:abs(x-midi))
    return norm(rp(b[k],(midi-k)+cents/100.0))

# ---------- the wedge ----------
ROW=[0]
for k in range(1,12): ROW.append((ROW[-1]+(k if k%2 else -k))%12)
UNIT=0.62
def DS(k,mult=1.0): return UNIT*max(1,k%12)*mult          # durations expand as the intervals do
def ROWPAN(i): return ((ROW[i%12]/11.0)*2.0-1.0)*0.88      # and the series places things in the room
def ROWTHROAT(k,off=0): return ROW[(k+off)%12]%7           # and chooses which throat carries a syllable
A4M=69
def rowmidi(i,base=A4M):
    """the register fans with the row: odd members climb, even members fall"""
    pc=ROW[i%12]; step=(i%12)
    oct_= (step+1)//2 * (1 if step%2 else -1)
    m=base+pc+12*oct_
    while m>103: m-=12
    while m<26: m+=12
    return m

TSUS1=norm(L(A4+"Trombone_Sustain_A1_v3_1.wav")); TSUS2=norm(L(A4+"Trombone_Sustain_A2_v3_1.wav"))
TBUZ1=norm(L(A4+"Trombone_Buzz_A1_v1_1.wav")); TBUZ2=norm(L(A4+"Trombone_Buzz_A2_v1_1.wav"))
TFALL1=norm(L(A4+"Trombone_Fall_A1_1.wav")); TFALL2=norm(L(A4+"Trombone_Fall_A2_1.wav"))
SHTPT=[norm(L(A4+f"Sum_SHTrumpet_stac_A2_v{v}_rr1.wav")) for v in (1,2,3)]
TIMP=norm(L(A4+"Timpani1_Hit_v3_rr1_Sum.wav")); TROLL=norm(L(A4+"Timpani3_Roll_v3_rr1_Sum.wav"))
ANV=[norm(L(A4+f"Anvil_Hit1_v{v}_Sum.wav")) for v in (1,2,3)]
VTREM=norm(L(A4+"VlnEns_Trem_A2_v1.wav"))
CYB=[norm(L(M+f"IOWA_cymbow{i}.flac")) for i in (1,2,3)]
GONGW=norm(L(M+"IOWA_gong1_20windgong.pp.flac"))
GONGT=[norm(L(M+"IOWA_gong2_thaigong.C4.ff.flac")),norm(L(M+"IOWA_gong3_thaigong.Gb4.ff.flac"))]
WIND=norm(L(M+"BY__aporee_dartington_wind.mp3")); HUM=norm(L(M+"CC0__goldtape_electricity.mp3"))
P_OCEAN=A4+"ocean90.wav"; P_MARIO=A4+"mario.wav"; P_WILLIE=DW+"willie.wav"; P_MOR=DW+"mor13.wav"
P_CHAL=DW+"chaliapin__Boris_Godounow-In_the_Town_of_Kaz_n_-_Feodor_Chaliapin.wav"
P_VADER=I4+"vader.wav"; P_TYRANT=I4+"chern1.wav"; P_NL=I4+"read_havelaar_nl.wav"; P_EN=I4+"read_tulip_ch1.wav"
P_NUIT=conv_if_needed("/home/claude/007/dutch/pack/nuit_detoiles_culp_1917.mp3",I4+"nuit.wav")

DUR=640.0; TOT=int(DUR*SR); Lc=np.zeros(TOT); Rc=np.zeros(TOT)
def place(sig,t,g=1.0,pan=0.0):
    i=int(t*SR); e=min(TOT,i+len(sig))
    if e<=i or i<0: return
    gl,gr=np.sqrt(0.5-pan/2),np.sqrt(0.5+pan/2)
    Lc[i:e]+=sig[:e-i]*g*gl; Rc[i:e]+=sig[:e-i]*g*gr
_ir1=norm(lp(FB("SmallMetals",40.0,41.3),5200)); irRoom=_ir1*np.exp(-np.arange(len(_ir1))/(0.24*SR))
_ir2=norm(hp(FB("Broken-Glass1",96.0,96.9),700)); irGlass=_ir2*np.exp(-np.arange(len(_ir2))/(0.12*SR))
def room(y,mix=0.2):
    w=fftconvolve(y,irRoom)[:len(y)+int(1.3*SR)]
    return norm(np.pad(y,(0,len(w)-len(y)))*(1-mix)+norm(w)*mix)
def vitrine(y,g=1.0):
    z=bp(norm(y),330,1900)
    w=fftconvolve(z,irGlass)[:len(z)+int(0.9*SR)]
    return norm(np.pad(z,(0,len(w)-len(z)))*0.8+norm(w)*0.3)*g
def longest_loud(y,thr=0.22,win=0.5):
    w=int(win*SR); e=np.array([float(np.sqrt(np.mean(y[i*w:(i+1)*w]**2))) for i in range(len(y)//w)])
    ok=e>thr*np.max(e); best=(0,0); run=None
    for i,v in enumerate(ok):
        if v and run is None: run=i
        if (not v or i==len(ok)-1) and run is not None:
            if i-run>best[1]-best[0]: best=(run,i)
            run=None
    a,b=best[0]*w,best[1]*w
    return y[a:b] if b-a>int(6*SR) else y
WINDL=longest_loud(WIND)
air=sustain(lp(WINDL,380),DUR+2.0,xf=1.6)[:TOT]

hum=np.tile(bp(HUM,90,700),TOT//len(HUM)+2)[:TOT]
place(norm(air*0.7+hum*0.5)*np.interp(np.arange(TOT)/SR,[0,6,120,300,420,500,DUR],[0,0.85,1.0,0.85,0.7,0.4,0])*10**(-30/20),0,1.0,0.0)

OFFSV=[0,-14,14,-25,25,-33,33,-50,50,-66,66,-75]
# ============================================================
# v4 — the row becomes HARMONY. Wedge pairs give six chords:
#   I  A+Bb   II Ab+B   III G+C   IV F#+C#   V F+D   VI E+Eb
# each sustained by strings and brass across five octaves, each
# crossfading into the next: the piece never stops sounding.
# Every voice is pitch-detected and retuned onto a chord tone, and
# every voice is DOUBLED by an instrument holding its pitch, entering
# before it and outlasting it. Nothing enters cold.
# ============================================================
def f0of(y,lo=70,hi=900):
    v=y[:int(6*SR)]
    if len(v)<2048: return 220.0
    n=1<<int(np.ceil(np.log2(len(v))))
    S=np.abs(np.fft.rfft(v*np.hanning(len(v)),n=n))
    fr=np.fft.rfftfreq(n,1/SR); m=(fr>=lo)&(fr<=hi)
    return float(fr[m][int(np.argmax(S[m]))])
def tune(y,pcs,lo=70,hi=900,maxshift=7.0):
    """retune a sample so its fundamental lands on a chord tone"""
    f=f0of(y,lo,hi); midi=69+12*np.log2(max(20.0,f)/440.0)
    cands=[]
    for pc in pcs:
        for o in range(1,9):
            cands.append(pc+12*o)
    tgt=min(cands,key=lambda m:abs(m-midi))
    d=tgt-midi
    if abs(d)>maxshift: d=np.sign(d)*maxshift
    return norm(rp(y,d)),midi+d

# ============================================================
# v5 — TWO AXES.
#   HARMONY: the six wedge-pair chords, as before, but now each is
#   orchestrated differently and gains a tone as the piece proceeds.
#   NOISE <-> PITCH: scored as its own parameter, NP(t) in [0,1].
#     brass   NP=0 air only through the bore ... NP=1 full tone
#     strings NP=0 overpressure scratch      ... NP=1 clean arco
#     bowed metal peaks at NP~0.45 (its home is the middle)
#     voices  NP=0 breath, unvoiced          ... NP=1 full vowel
#     tape    NP=0 wide-band                 ... NP=1 narrow, on chord tones
#   The form is the curve of that axis as much as the progression.
#
# And a CHOIR, synthesised: solo soprano (phonetic decomposition,
# quarter-tones, breath as notated material), solo contralto, and a
# chorus of sixteen, 4 per part, each voice its own detuning, its own
# vibrato phase, its own entry.
# ============================================================
NPX=[0,40,70,110,135,175,200,235,265,300,345,395,430,455,478,500]
NPY=[0.08,0.18,0.34,0.82,0.88,0.62,0.14,0.20,0.34,0.72,0.95,0.98,0.70,0.30,0.08,0.02]
def NP(t): return float(np.interp(t,NPX,NPY))

VOW={'a':[(800,0),(1150,-6),(2900,-32),(3900,-20),(4950,-50)],
     'e':[(350,0),(2000,-20),(2800,-15),(3600,-40),(4950,-56)],
     'i':[(270,0),(2140,-12),(2950,-26),(3900,-26),(4950,-44)],
     'o':[(450,0),(800,-11),(2830,-22),(3800,-22),(4950,-50)],
     'u':[(325,0),(700,-16),(2700,-35),(3800,-40),(4950,-60)]}
def fmt(src,vowel,scale=1.0,q0=9.0):
    out=np.zeros(len(src))
    for f,db in VOW[vowel]:
        fc=f*scale
        if fc>SR/2-600 or fc<70: continue
        q=q0 if fc<1300 else q0*1.5
        bw=max(40.0,fc/q)
        out+=bp(src,max(60,fc-bw/2),fc+bw/2,o=2)*(10**(db/20))
    return out
def sing_note(midi,dur,vowel='a',breath=0.2,press=0.0,vibd=28.0,cents=0.0,scale=1.0,seed=None):
    r=np.random.default_rng(seed if seed is not None else int(rng.integers(1e9)))
    n=int(dur*SR); t=np.arange(n)/SR
    f0=440.0*2**((midi-69+cents/100.0)/12.0)
    vr=4.6+r.uniform(-0.5,0.7)
    onset=np.clip(t/max(0.35,dur*0.22),0,1)
    vib=1.0+(vibd/1200.0)*np.sin(2*np.pi*vr*t+r.uniform(0,6))*onset
    jit=1.0+0.0035*np.cumsum(r.standard_normal(n))/np.sqrt(n)
    fcurve=f0*vib*jit
    ph=np.cumsum(fcurve)/SR
    nh=min(28,int((SR/2-800)/max(60.0,f0)))
    src=np.zeros(n)
    for h in range(1,max(2,nh)):
        src+=np.sin(2*np.pi*h*ph+r.uniform(0,6))/(h**(1.25-0.5*press))
    voiced=fmt(norm(src),vowel,scale)
    bn=fmt(r.standard_normal(n)*0.6,vowel,scale)*1.6
    y=voiced*(1.0-breath)+bn*breath
    if press>0: y=drv(y,1.0+2.2*press)
    env=np.interp(t,[0,dur*0.16,dur*0.7,dur],[0,1,0.92,0])**1.05
    return norm(y*env)
SATB=[("S",67,79,'a'),("A",60,72,'a'),("T",55,67,'o'),("B",43,55,'o')]
def chord_tones(pcs,lo,hi):
    out=[m for m in range(lo,hi+1) if (m%12) in [p%12 for p in pcs]]
    return out if out else [lo]
def chorus(t0,dur,pcs,gain,npv,vowel=None,per=4,spread=9.0,pan=0.62):
    for si,(nm,lo,hi,vw) in enumerate(SATB):
        opts=chord_tones(pcs,lo,hi)
        for k in range(per):
            m=opts[(k+si)%len(opts)]
            det=(k-1.5)*spread*0.5+rng.uniform(-4,4)
            br=float(np.clip(1.0-npv,0.05,0.95))
            y=sing_note(m,dur*rng.uniform(0.92,1.0),vowel or vw,breath=br,
                        press=0.15*(1-npv),vibd=22+10*npv,cents=det)
            place(y,t0+rng.uniform(0,0.5),gain*(0.85+0.3*rng.random()),
                  pan*np.sin(si*1.6+k*0.9))
def soprano(t0,notes,gain,npv_fn,vowels=('u','o','a','e','i')):
    """the solo line: quarter-tones, and phonetics that decompose into breath"""
    t=t0
    for k,(m,dur,ct_,vi) in enumerate(notes):
        npv=npv_fn(t)
        br=float(np.clip(1.05-npv,0.05,0.97))
        y=sing_note(m,dur,vowels[vi%len(vowels)],breath=br,press=0.1,vibd=18+16*npv,cents=ct_)
        place(edge(room(y,0.16),0.25,0.9),t,gain,0.10*np.sin(k*1.3))
        t+=dur*0.92
    return t
def contralto(t0,notes,gain,npv_fn,vowels=('o','u','a')):
    t=t0
    for k,(m,dur,ct_,vi) in enumerate(notes):
        npv=npv_fn(t); br=float(np.clip(1.0-npv,0.05,0.95))
        y=sing_note(m,dur,vowels[vi%len(vowels)],breath=br,press=0.18,vibd=16+12*npv,cents=ct_,scale=0.94)
        place(edge(room(y,0.18),0.35,1.1),t,gain,-0.16*np.sin(k*1.1))
        t+=dur*0.94
    return t

# ---- the axis applied to the instruments ----
def air_brass(y,npv):
    """NP=0: only the breath through the bore. NP=1: the tone."""
    n=len(y); r=np.random.default_rng(7)
    envf=np.abs(lp(np.abs(y),6.0)); envf/= (np.max(envf)+1e-9)
    breath=bp(r.standard_normal(n),260,3400)*envf
    return norm(y*npv+breath*(1.0-npv)*0.9)
def press_string(y,npv):
    """NP=0: overpressure scratch. NP=1: clean arco."""
    n=len(y); r=np.random.default_rng(11)
    envf=np.abs(lp(np.abs(y),9.0)); envf/=(np.max(envf)+1e-9)
    scr=drv(bp(r.standard_normal(n),180,2600)*envf,3.2)
    grind=drv(y,1.0+5.0*(1.0-npv))
    return norm(y*npv+ (grind*0.5+scr*0.85)*(1.0-npv))

def place_pan(sig,t,g,p0,p1,rate=0.0,depth=0.0):
    n=len(sig)
    if n<8: return
    tt=np.arange(n)/SR
    p=np.linspace(p0,p1,n)
    if rate>0: p=p+depth*np.sin(2*np.pi*rate*tt)
    p=np.clip(p,-0.98,0.98)
    gl=np.sqrt(0.5-p/2); gr=np.sqrt(0.5+p/2)
    i=int(t*SR); e=min(TOT,i+n)
    if e<=i: return
    Lc[i:e]+=sig[:e-i]*g*gl[:e-i]; Rc[i:e]+=sig[:e-i]*g*gr[:e-i]
def gliss(y,c0,c1):
    n=len(y)
    if n<64: return y
    rates=2**(np.linspace(c0,c1,n)/1200.0)
    idx=np.cumsum(rates); idx=np.clip(idx,0,n-2)
    i0=idx.astype(int); fr=idx-i0
    return norm(y[i0]*(1-fr)+y[i0+1]*fr)
def pshift(y,cents,grain=0.085):
    r=2**(cents/1200.0); g=int(grain*SR); hop=g//2
    if len(y)<3*g: return y
    out=np.zeros(len(y)+g); win=np.hanning(g); pos=0
    while pos+int(g*r)+4<len(y):
        src=y[pos:pos+int(g*r)+4]
        nn2=max(4,int(len(src)/r)); idx=np.arange(nn2)*r
        i0=np.clip(idx.astype(int),0,len(src)-2); fr=idx-i0
        seg=(src[i0]*(1-fr)+src[i0+1]*fr)[:g]
        if len(seg)<g: seg=np.pad(seg,(0,g-len(seg)))
        out[pos:pos+g]+=seg*win
        pos+=hop
    return norm(out[:len(y)])
def halaphon(t0,y,cents=9.0,taps=6,delay=1.5,fb=0.66,gain=0.22,spread=0.62,lpf=6200,cumulative=True):
    """the room answers with a pitch that has no name: each return shifted further"""
    cur=norm(y)
    for k in range(1,taps+1):
        c=cents*k if cumulative else cents
        cur=pshift(cur,cents) if cumulative else pshift(norm(y),c)
        z=lp(cur,lpf-500*k)
        z=edge(z,0.25,1.4)
        pa=spread*np.sin(k*1.83)
        place_pan(z,t0+k*delay,gain*(fb**k),pa,-pa*0.85,rate=0.035+0.01*k,depth=0.18)
    # the original and its double, beating
    place(edge(pshift(norm(y),cents*0.55),0.2,1.2),t0+0.05,gain*0.75,-spread*0.4)
def sweepfilt(y,f0_,f1_,q=22.0,blk=0.11):
    n=len(y)
    if n<2048: return y
    step=max(256,int(blk*SR)); nb=max(4,n//step)
    out=np.zeros(n); win=np.hanning(2*step)
    for i in range(nb):
        a=i*step; b=min(n,a+2*step)
        if b-a<64: continue
        fc=f0_*(f1_/f0_)**(i/max(1,nb-1))
        bw=max(28.0,fc/q)
        seg=bp(y[a:b],max(40,fc-bw/2),fc+bw/2,o=2)
        out[a:b]+=seg*win[:b-a]
    return norm(out)
PAIRS=[(ROW[0],ROW[1]),(ROW[2],ROW[3]),(ROW[4],ROW[5]),(ROW[6],ROW[7]),(ROW[8],ROW[9]),(ROW[10],ROW[11])]
CH_PC=[[(69+a)%12,(69+b)%12] for a,b in PAIRS]
for i in range(1,6): CH_PC[i]=CH_PC[i]+[9]                          # A returns as the axis of the fan
SEC=[(2.0,86.0),(78.0,168.0),(160.0,252.0),(244.0,318.0),(310.0,404.0),(396.0,470.0)]
ORCH=[[(33,"tuba",0.24),(45,"btb",0.22),(52,"vlc",0.16)],                                  # low, dark
      [(57,"tbn",0.20),(64,"vla",0.17),(69,"vlaD",0.15),(76,"vlnD",0.12)],                 # middle strings
      [(45,"btb",0.20),(69,"vlaD",0.14),(81,"vlnA",0.11),(88,"bell5",0.09)],               # opened out
      [(33,"tuba",0.22),(57,"tbn",0.20),(64,"vla",0.15),(76,"vlnD",0.12),(88,"bell5",0.08)],
      [(45,"btb",0.18),(52,"vlc",0.15),(69,"vlaD",0.14),(81,"vlnA",0.10)],                 # for Culp
      [(33,"tuba",0.20),(52,"vlc",0.14),(64,"vla",0.13),(81,"vlnA",0.10),(88,"bell5",0.08)]]
for ci,(t0,t1) in enumerate(SEC):
    dur=t1-t0
    GQ=[0.26,0.46,0.5,0.6,1.0,1.0][ci]
    for k,(base,inst,g) in enumerate(ORCH[ci]):
        g=g*GQ
        pcs=CH_PC[ci]
        m=min([p+12*o for p in pcs for o in range(1,9)],key=lambda x:abs(x-base))
        y=sustain(note(inst,m,cents=[0,-25,25,-50,50][k%5]),dur,xf=0.55)
        npv=NP((t0+t1)/2)
        y=air_brass(y,npv) if inst in ("tuba","btb","tbn","tbn2","tpt","tptn") else press_string(y,npv)
        n=len(y); tt=np.arange(n)/SR
        if ci<4:
            fc=440.0*2**((m-69)/12.0)
            y=norm(y*0.55+sweepfilt(y,fc*0.75,fc*2.6,q=9.0,blk=0.5)*0.75)
        place(y*np.interp(tt,[0,9.0,dur-10.0,dur],[0,1.0,1.0,0]),t0,g,0.62*np.sin(k*1.47+ci))
    # bowed metal lives in the middle of the axis
    npm=NP((t0+t1)/2); bm=1.0-abs(npm-0.45)*1.9
    if bm>0.05:
        place(edge(CYB[ci%3],4.0,6.0),t0+dur*0.35,0.15*bm,0.4*(-1)**ci)
        place(edge(GONGW,5.0,8.0),t0+dur*0.62,0.13*bm,-0.3*(-1)**ci)

# ============================================================
# v6 — the electronics take the front.
#   Sections I-IV are rebuilt around three devices:
#     SWEEP   a resonant band crawling across held material: a LINE
#             made of filtering, not of pitch.
#     FASCE   micro-intervallic band-clusters — seven to ten narrow
#             bands spaced 20-70 cents apart, each gliding, so the
#             cluster beats against itself.
#     NASTRO  tape: layering, overlapping, intercutting, collage —
#             present from the first minute, thickening throughout.
#   The harmony stays, quieter, filtered and slowly moving: a ground.
# ============================================================
def sweep(t0,dur,src,f_a,f_b,gain,q=22.0,pan=0.0,fi=1.2,fo=2.2):
    base=sustain(src,dur,xf=0.4)
    place(edge(sweepfilt(base,f_a,f_b,q),fi,fo),t0,gain,pan)
def fasce(t0,dur,src,center_hz,n=8,spread_c=70.0,glide_c=45.0,gain=0.20,pan=0.62,q=30.0):
    base=sustain(src,dur,xf=0.45)
    for k in range(n):
        u=(k-(n-1)/2.0)/max(1.0,(n-1)/2.0)
        c0=center_hz*2**((u*spread_c)/1200.0)
        c1=c0*2**((glide_c*np.sin(k*1.27+0.6))/1200.0)
        y=sweepfilt(base,c0,c1,q=q,blk=0.14)
        nn2=len(y); tt2=np.arange(nn2)/SR
        env=np.interp(tt2,[0,dur*0.22,dur*0.75,dur],[0,1,0.9,0])
        p0=pan*np.sin(k*1.7)
        place_pan(y*env,t0+abs(u)*1.6,gain/np.sqrt(n)*(1.0+0.3*(1-abs(u))),p0,-p0,
                  rate=0.018+0.012*(k%4),depth=0.22)
def nastro(t0,dur,cell,gain,layers,ci,srcs,drive=1.0,pan=0.6,widen=0.0):
    """tape: intercut, overlap, collage. widen>0 opens the bands toward noise."""
    bands=chordhz(ci)
    for Lx in range(layers):
        t=t0+rng.uniform(0,0.4)
        while t<t0+dur:
            d=rng.uniform(*cell); npv=NP(t)
            src=srcs[int(rng.integers(len(srcs)))]
            a=rng.uniform(0,max(0.05,len(src)/SR-d-0.05))
            y=src[int(a*SR):int((a+d)*SR)].copy()
            if len(y)<int(0.03*SR): t+=d; continue
            c=bands[int(rng.integers(len(bands)))]*(2**int(rng.integers(0,3)))
            w_=0.35+widen+0.55*(1.0-npv)
            y=bp(y,c*max(0.25,1-w_*0.55),c*(1+w_*1.25),o=3)
            if drive>1.0: y=drv(y,drive)
            wf=int(0.02*SR); y=norm(y)
            y[:wf]*=np.linspace(0,1,wf)**0.7; y[-wf:]*=np.linspace(1,0,wf)**0.7
            gg=gain*np.interp(t,[t0,t0+dur*0.22,t0+dur*0.82,t0+dur],[0.2,1,1,0.18])
            if rng.random()<0.42: pn=0.92*(1 if int(rng.integers(2)) else -1)   # spatial cut
            else: pn=pan*np.sin(Lx*2.0+t)
            place(y*rng.uniform(0.6,1.0),t,gg,pn)
            t+=d*rng.uniform(0.88,1.08)

VOX=[sp(P_WILLIE,26,36),sp(P_MOR,18,27),sp(P_CHAL,44,53),sp(P_VADER,88,98),
     sp(P_NUIT,50,66),sp(P_TYRANT,108,120),sp(P_NL,60,72),sp(P_EN,64,78),sp(P_MARIO,32,42)]
VOXT=[tune(v,CH_PC[2],90,700,maxshift=9.0)[0] for v in VOX]
def chordhz(ci):
    out=[]
    for base,inst,g in ORCH[ci]:
        m=min([p+12*o for p in CH_PC[ci] for o in range(1,9)],key=lambda x:abs(x-base))
        out.append(440.0*2**((m-69)/12.0))
    return out
HZ=lambda m: 440.0*2**((m-69)/12.0)

# ---- Debussy, played by hands, 1919 and 1921 ----
P_FILLE=I4+"DEB_fille_1919.wav"; P_GOLLI=I4+"DEB_golliwogg_rach_1921.wav"
_rp=np.random.default_rng(31); _n=int(4.6*SR); _t=np.arange(_n)/SR
_ir=lp(_rp.standard_normal(_n),1700)*np.exp(-_t/1.7)+bp(_rp.standard_normal(_n),1700,4000)*np.exp(-_t/0.5)
irPno=_ir/(np.sqrt(np.sum(_ir**2))+1e-9)
def debussy(t,a,b,g,pan,src="g",lo=175,hi=4400,fi=0.7,fo=4.6,dark=0.0):
    y=sp(P_GOLLI if src=="g" else P_FILLE,a,b)
    y=bp(y,lo,hi-1400*dark)
    env=np.convolve(np.abs(y),np.ones(int(0.02*SR))/int(0.02*SR),'same')
    y=norm(y*np.clip(env/(np.percentile(env,68)+1e-9),0.18,1.0))       # the surface ducks under the notes
    w=fftconvolve(y,irPno)[:len(y)+len(irPno)]
    y=np.pad(y,(0,len(w)-len(y)))*0.60+norm(w)*0.62                    # and it blooms
    place(edge(norm(y),fi,fo),t,g,pan)

# ============ 0. THE FAN — the wedge stated (4-56) ============
FANI=[("vlc",0.15),("tbn2",0.24),("vla",0.16),("btb",0.23),("vlnG",0.14),("tuba",0.23),
      ("vlaD",0.15),("tpt",0.19),("vlnD",0.13),("tbn",0.21),("vlnA",0.12),("bell5",0.13)]
tFAN=4.0
for i in range(12):
    inst,g=FANI[i]
    y=sustain(note(inst,rowmidi(i),cents=OFFSV[i]),DS(i,1.9)+2.2,xf=0.5)
    y=air_brass(y,NP(tFAN)) if inst in ("tuba","btb","tbn","tbn2","tpt") else press_string(y,NP(tFAN))
    p=ROWPAN(i)
    place_pan(edge(room(y),1.5,2.8),tFAN,g,p,-p*0.65,rate=0.017+0.004*i,depth=0.16)
    tFAN+=DS(i,1.05)+0.45

# the piano sits in the measured silences: 16, 36, 76, 160, 216, 246, 288, 326
DEBS=[(16.0,"g",118.0,122.6,0.34,-0.24,0.30),(36.0,"f",152.0,156.6,0.33,0.26,0.20),
      (76.0,"g",46.0,50.6,0.36,0.18,0.0),(160.0,"f",178.0,182.6,0.34,-0.28,0.25),
      (216.0,"g",98.0,102.6,0.36,0.22,0.0),(246.0,"g",170.0,174.6,0.35,-0.20,0.15),
      (288.0,"f",194.0,198.6,0.33,0.24,0.30),(326.0,"g",130.0,134.6,0.31,-0.16,0.35)]
for _t0,_s,_a,_b,_g,_p,_d in DEBS:
    debussy(_t0,_a,_b,_g,_p,_s,dark=_d)

# ============ I. FILTERS (0-72) ============
place(edge(room(FB("MechanicalMetals",62.0,63.1)),0.05,0.9),7.0,0.19,-0.15)
sweep(10.0,DS(7,6),VOX[0],240.0,1750.0,0.30,q=24,pan=-0.2)                 # a line made of filtering
sweep(20.0,DS(6,6),VOX[5],1900.0,320.0,0.24,q=20,pan=0.26)                 # its contrary motion
fasce(30.0,DS(7,6),VOX[3],HZ(69),n=8,spread_c=64,glide_c=38,gain=0.26,pan=0.6)   # the first band-cluster, on A
nastro(24.0,DS(7,6),(0.34,0.72),0.13,2,0,VOXT,widen=0.25)
place(edge(room(note("pno",69)),0.004,1.5),16.0,0.10,0.0)
soprano(44.0,[(69,7.0,0,0),(70,6.0,-50,0),(69,8.0,25,1)],0.26,NP)
sweep(50.0,DS(6,6),VOX[2],3200.0,420.0,0.22,q=26,pan=-0.3)
fasce(56.0,DS(5,6),VOX[6],HZ(57),n=7,spread_c=48,glide_c=30,gain=0.22,pan=-0.55)
nastro(52.0,DS(6,6),(0.26,0.58),0.15,3,0,VOXT,widen=0.15)

# ============ II. FASCE (72-160) ============
contralto(74.0,[(57,9.0,-25,0),(58,8.0,50,1)],0.22,NP)
fasce(78.0,DS(8,6),VOX[4],HZ(76),n=9,spread_c=80,glide_c=55,gain=0.28,pan=0.62)
fasce(88.0,DS(7,6),VOX[1],HZ(64),n=8,spread_c=56,glide_c=-40,gain=0.24,pan=-0.6)
nastro(84.0,DS(8,6),(0.20,0.46),0.17,4,1,VOXT,drive=1.15)
sweep(96.0,DS(7,6),VOX[8],300.0,4200.0,0.24,q=30,pan=0.18)
soprano(104.0,[(76,6.5,-50,1),(77,6.0,0,2),(79,7.0,25,2),(76,6.5,-25,3),(74,8.0,0,2)],0.28,NP)
fasce(116.0,DS(9,6),VOX[7],HZ(81),n=10,spread_c=92,glide_c=64,gain=0.26,pan=-0.5)
nastro(114.0,DS(9,6),(0.15,0.36),0.19,5,1,VOXT,drive=1.25)
chorus(128.0,DS(4,6),CH_PC[1],0.075,NP(128.0))                              # the choir emerges from the bands
chorus(144.0,DS(4,6),CH_PC[1],0.085,NP(144.0),vowel='o')
sweep(140.0,DS(6,6),VOX[5],4600.0,260.0,0.22,q=24,pan=0.3)
place(edge(TIMP,0.06,1.8),157.0,0.16,0.0)

# ============ III. NASTRO — the word inside the tape (160-244) ============
THROATS=[(P_WILLIE,28.0,34.0,70,400),(P_MOR,19.0,25.0,200,800),(P_CHAL,45.0,51.0,70,400),
         (P_VADER,89.0,95.0,150,700),(P_NUIT,58.0,64.0,150,800),(P_TYRANT,112.0,118.0,120,900),
         (P_NL,62.0,68.0,90,500)]
RHY=[0.32,0.28,0.60, 0.32,0.28,0.60, 0.36,0.28,0.66]
def syll(i,dur,ci):
    p,a,b,lo,hi=THROATS[i%len(THROATS)]
    off=rng.uniform(a,max(a+0.05,b-dur-0.05))
    y=bp(sp(p,off,off+dur),280,3300)
    y,mid=tune(y,CH_PC[ci],lo,hi,maxshift=9.0)
    return edge(norm(y*0.85+np.roll(y,int(0.05*SR))*0.2),0.07,min(0.24,dur*0.45)),mid
def word_in_tape(t0,spread,gain,rowoff,ci,answer=True):
    t=t0
    for k,d in enumerate(RHY):
        y,mid=syll(ROWTHROAT(k,rowoff),d,ci)
        place(y,t,gain,ROWPAN(k+rowoff))
        if answer and k%3==2:
            chorus(t+d*0.7,4.6,CH_PC[ci],0.07,NP(t),per=2,spread=7.0)
        t+=d+spread*(1.0 if k%3!=2 else 1.7)
    return t
tX=162.0
nastro(tX,DS(9,6),(0.24,0.52),0.18,4,2,VOXT)
tW=word_in_tape(tX+4.0,0.85,0.22,0,2)
fasce(tX+16.0,DS(7,6),VOX[2],HZ(45),n=8,spread_c=70,glide_c=48,gain=0.24,pan=0.45)
nastro(tX+30.0,DS(9,6),(0.14,0.34),0.20,5,2,VOXT,drive=1.3)
tW=word_in_tape(tW+3.0,0.34,0.26,4,2)
sweep(tX+44.0,DS(6,6),VOX[6],5200.0,300.0,0.24,q=28,pan=-0.24)
for k in range(4):
    chorus(tX+10.0+k*17.0,DS(3,6),CH_PC[2],0.08,0.12,vowel='u',per=3,spread=12.0)   # unvoiced choir
nastro(tX+58.0,DS(7,6),(0.08,0.20),0.22,7,2,VOXT,drive=1.7)
tW=word_in_tape(tW+14.0,0.06,0.28,7,2,answer=False)
SWL=sustain(note("tbn",57,cents=25),8.0)
place(edge(drv(air_brass(SWL,0.75),2.0),4.0,2.2),tW-5.0,0.22,0.0)
place(edge(GONGT[0][:int(2.0*SR)],0.03,1.8),tW+0.3,0.38,0.0)

# ============ IV. THE AXIS TURNS (244-318) ============
tY=tX+86.0
fasce(tY,DS(8,6),VOX[3],HZ(69),n=10,spread_c=88,glide_c=-60,gain=0.26,pan=0.55)
nastro(tY+2.0,DS(8,6),(0.18,0.42),0.18,4,3,VOXT,drive=1.2)
sweep(tY+8.0,DS(7,6),VOX[4],380.0,3600.0,0.22,q=26,pan=-0.28)
place(edge(drv(air_brass(sustain(note("tbn",57,cents=-25),9.0),0.5),2.0),4.5,2.4),tY+26.0,0.24,0.1)
place(edge(GONGT[1][:int(2.2*SR)],0.03,2.0),tY+32.0,0.40,0.06)
place(edge(room(press_string(sustain(note("vlc",52,cents=25),15.0),0.3)),4.0,5.0),tY+36.0,0.15,-0.28)
fasce(tY+40.0,DS(7,6),VOX[0],HZ(81),n=8,spread_c=54,glide_c=34,gain=0.20,pan=-0.45)
nastro(tY+42.0,DS(6,6),(0.26,0.60),0.14,3,3,VOXT)
chorus(tY+50.0,DS(4,6),CH_PC[3],0.085,NP(tY+50.0),vowel='o',per=3)
soprano(tY+56.0,[(79,5.5,50,3),(78,5.0,-50,4),(76,6.5,0,3)],0.26,NP)

def sing(path,a,b,t,g,pan,ci,inst_double="vla",lo=70,hi=900,glass=False,fi=1.4,fo=2.6):
    src=sp(path,a,b)
    y,mid=tune(src,CH_PC[ci%6],lo,hi)
    if glass: y=vitrine(y)
    y=edge(room(bp(y,240,3600)),fi,fo)
    dbl=(b-a)+fi+fo+3.0
    d=sustain(note(inst_double,int(round(mid)),cents=-14),dbl,xf=0.5)
    dn=len(d); dt=np.arange(dn)/SR
    place(press_string(d,NP(t))*np.interp(dt,[0,2.6,dbl-4.0,dbl],[0,0.85,0.8,0])*0.12,t-1.6,1.0,-pan*0.6)
    place(y,t,g,pan)
    return mid


# ============ V. THE CASES, HELD BY THE CHOIR (318-404) ============
CASES=[(P_WILLIE,26.0,34.0,"vla",70,400),(P_MOR,18.0,25.0,"vlaD",200,800),
       (P_MARIO,32.0,40.0,"vlnD",150,900),(P_CHAL,44.0,52.0,"vlc",70,400),
       (P_OCEAN,20.0,30.0,"tbn",90,700),(P_VADER,88.0,96.0,"vlnA",150,700),
       (P_TYRANT,110.0,119.0,"btb",120,900)]
tK=tX+80.0
for i,(p,a,b,inst,lo,hi) in enumerate(CASES):
    place(edge(room(FB("SmallMetals",60.0+i*7.0,61.1+i*7.0)),0.08,0.9),tK,0.13,(-1)**i*0.2)
    mid=sing(p,a,b,tK+1.8,0.15 if i<6 else 0.18,0.0,4,inst,lo,hi,glass=True,fi=1.3,fo=2.5)
    chorus(tK+0.6,(b-a)+4.0,CH_PC[4],0.075,NP(tK),per=2,spread=8.0)   # the choir holds the case open
    halaphon(tK+2.4,vitrine(sp(p,a,min(b,a+3.2))),cents=[6,-8,11,-13,9,-7,14][i],
             taps=5,delay=1.45,fb=0.62,gain=0.10,spread=0.7)             # the case answers
    if i%2==0: place(edge(CYB[i%3],3.6,4.6),tK+3.0,0.09,(-1)**i*0.3)
    tK+=13.0+rng.uniform(-0.4,0.6)
soprano(tK-26.0,[(81,6.0,25,2),(79,5.5,-25,3),(81,7.0,0,2)],0.26,NP)

def threnody(t0,dur,gain):
    """the death: quarter-tone clusters, contrary glissandi, overpressure"""
    INST=["vlnA","vlnA","vlnD","vlnD","vlaD","vlaD","vla","vla","vlc","vlc","vlc","btb"]
    for k in range(26):
        i=k%12
        inst=INST[i]
        m=rowmidi(i)
        while m>92: m-=12
        while m<40: m+=12
        c0=(ROW[i]-5.5)*9.0+rng.uniform(-3,3)
        d=dur*(0.34+0.04*max(1,i))
        y=sustain(note(inst,m,cents=c0),d,xf=0.4)
        up=(i%2==0)
        y=gliss(y,0.0,(1.0 if up else -1.2)*(i+1)*13.5)
        y=press_string(y,float(np.clip(0.62-0.34*(i/11.0),0.12,0.9)))
        n2=len(y); t2=np.arange(n2)/SR; dd=n2/SR
        env=np.interp(t2,[0,dd*0.30,dd*0.78,dd],[0,1,0.85,0])
        p=ROWPAN(i)
        place_pan(y*env,t0+DS(i,0.9)*rng.uniform(0.8,1.2),gain*(0.55+0.5*rng.random()),p,-p,
                  rate=0.02+0.02*(i%5),depth=0.12)
    for k in range(22):                                            # col legno: dry strikes across the field
        tt2=t0+2.0+rng.uniform(0,dur*0.8)
        r=np.random.default_rng(int(tt2*97))
        n3=int(0.05*SR); cl=bp(r.standard_normal(n3),700+1800*r.random(),6200)*np.exp(-np.arange(n3)/(0.006*SR))
        place(norm(cl),tt2,gain*0.5,0.9*(1 if k%2 else -1))
    for k,(m,inst) in enumerate([(33,"tuba"),(45,"btb"),(57,"tbn")]):
        y=sustain(note(inst,m,cents=[-33,25,-14][k]),dur*0.55,xf=0.5)
        y=air_brass(y,0.55); y=gliss(y,0.0,-60.0)
        n2=len(y); t2=np.arange(n2)/SR
        place_pan(y*np.interp(t2,[0,6,n2/SR-8,n2/SR],[0,1,0.9,0]),t0+dur*0.30,gain*0.75,
                  (-1)**k*0.6,(-1)**(k+1)*0.6,rate=0.03,depth=0.2)
    place(edge(GONGT[1][:int(2.4*SR)],0.05,2.2),t0+dur*0.62,0.34,0.0)
    place(edge(TROLL,1.5,3.0),t0+dur*0.58,0.20,0.0)

# ============ VI. CULP — EVERYTHING TUNES TO HER (404-470) ============
threnody(tK+1.0,DS(11,4.05),0.26)
tV=tK+52.0
NU=sp(P_NUIT,42.0,118.0)
nn=len(NU); tt=np.arange(nn)/SR
place(NU*np.interp(tt,[0,4.0,nn/SR-7.0,nn/SR],[0,0.34,0.32,0]),tV,1.0,0.0)
fC=f0of(NU[int(6*SR):int(20*SR)],150,700); mC=int(round(69+12*np.log2(fC/440.0)))
for k,(off,inst,g) in enumerate([(-24,"tuba",0.15),(-12,"vlc",0.14),(0,"vla",0.12),(+7,"vlaD",0.10)]):
    d=60.0
    y=sustain(note(inst,mC+off,cents=[-14,14,-25,25][k]),d,xf=0.6)
    y=air_brass(y,0.95) if inst=="tuba" else press_string(y,0.95)
    n2=len(y); t2=np.arange(n2)/SR
    place(y*np.interp(t2,[0,9,d-10,d],[0,1,0.95,0]),tV+6.0,g,0.5*np.sin(k*1.7))
chorus(tV+14.0,DS(5,6),[mC%12,(mC+7)%12,9],0.075,0.95,vowel='o',per=2)
chorus(tV+40.0,DS(6,6),[mC%12,(mC+3)%12,9],0.08,0.98,vowel='a',per=3)
halaphon(tV+26.0,NU[int(10*SR):int(15.0*SR)],cents=5.0,taps=6,delay=2.1,fb=0.7,gain=0.085,spread=0.75,lpf=5200)
halaphon(tV+54.0,NU[int(40*SR):int(45.0*SR)],cents=-6.0,taps=5,delay=2.4,fb=0.68,gain=0.075,spread=0.7,lpf=4600)
soprano(tV+30.0,[(mC+12,7.0,0,2),(mC+10,6.0,-25,3)],0.20,NP)

# ============ VII. CONGEDO (470+) ============
tA=tV+nn/SR+2.0
for i in range(11,-1,-1):
    m=rowmidi(i)
    inst=["bell6","vlnA","tbn","vlnD","tpt","vlaD","tuba","vlnG","btb","vla","tbn2","vlc"][i]
    y=sustain(note(inst,m,cents=OFFSV[i]*0.5),3.2)
    y=air_brass(y,NP(tA)) if inst in ("tuba","btb","tbn","tbn2","tpt") else press_string(y,NP(tA))
    place(edge(room(y),1.6,3.0),tA+(11-i)*1.7,0.125*(0.55+0.045*i),0.5*np.sin(i*1.5))
tZ=tA+18.0
place(edge(GONGW[:int(7.0*SR)],1.8,4.6),tZ,0.20,0.0)
SWL2=sustain(note("vla",69,cents=-14),6.0,xf=0.5)
place(press_string(SWL2,0.9)*np.interp(np.arange(len(SWL2))/SR,[0,4.2,5.4,6.0],[0,0.11,0.06,0]),tZ-1.4,1.0,-0.2)
SWL3=sustain(note("vlc",57,cents=14),6.4,xf=0.5)
place(press_string(SWL3,0.9)*np.interp(np.arange(len(SWL3))/SR,[0,4.6,5.8,6.4],[0,0.09,0.05,0]),tZ-1.0,1.0,0.22)
# the room does not slam shut — but it dims rather than glitters
def darkloop(y,dur,xf=1.4,cut=3000):
    core=y[int(0.28*len(y)):]                      # start past the attack: no re-strike
    if len(core)<4096: core=np.tile(core,8)
    w=max(64,min(int(xf*SR),len(core)//3))
    if len(core)<3*w: core=np.tile(core,4); w=max(64,min(w,len(core)//3))
    n=int(dur*SR); out=np.zeros(n+len(core)+8); pos=0
    while pos<n:
        b=min(len(core),n-pos); seg=core[:b]*1.0
        if b>w:
            seg[:w]*=np.linspace(0,1,w)
            seg[-w:]*=np.linspace(1,0,w)
        out[pos:pos+b]+=seg
        pos+=max(1,len(core)-w)
    return lp(norm(out[:n]),cut)
BRG=darkloop(CYB[1],22.0,xf=2.2,cut=2600)                     # bowed metal, dimmed
_b=bp(room(BRG),420,2400); _t=np.arange(len(_b))/SR
place_pan(_b*np.interp(_t,[0,3.2,9.0,16.0,len(_b)/SR],[0,1.0,0.62,0.28,0])*0.165,tZ+0.5,1.0,0.32,-0.32,rate=0.018,depth=0.2)
BRG2=darkloop(GONGW,18.0,xf=2.6,cut=2200)                     # the wind gong, felt not heard
_b2=bp(room(BRG2),300,1900); _t2=np.arange(len(_b2))/SR
place_pan(_b2*np.interp(_t2,[0,3.0,8.5,14.0,len(_b2)/SR],[0,1.0,0.58,0.24,0])*0.135,tZ+2.4,1.0,-0.28,0.28,rate=0.022,depth=0.22)
BRG3=darkloop(note("vlnA",83,cents=-33),20.0,xf=1.8,cut=2400)  # a low sul-A string, no gloss
_b3=press_string(room(BRG3),0.75); _t3=np.arange(len(_b3))/SR
place_pan(_b3*np.interp(_t3,[0,3.4,10.0,16.0,len(_b3)/SR],[0,1.0,0.55,0.22,0])*0.095,tZ+1.2,1.0,-0.24,0.24,rate=0.015,depth=0.24)
BRG4=darkloop(note("vla",71,cents=25),16.0,xf=1.6,cut=1800)
_b4=room(BRG4); _t4=np.arange(len(_b4))/SR
place_pan(_b4*np.interp(_t4,[0,3.0,8.0,13.0,len(_b4)/SR],[0,1.0,0.5,0.2,0])*0.075,tZ+4.0,1.0,0.2,-0.2,rate=0.02,depth=0.18)
for k,m in enumerate([33,45,57,69,81]):
    place(edge(lp(room(note("pno",m)),5200),0.05,2.6),tZ+4.4+k*0.30,[0.105,0.115,0.11,0.095,0.085][k],[0,-0.14,0.14,-0.08,0.08][k])
place(edge(lp(note("bell5",81),2100),0.45,5.6),tZ+3.6,0.075,0.0)
place(edge(room(note("pno",80)),0.02,2.0),tZ+12.6,0.085,0.0)
LASTA=edge(room(note("pno",81)),0.004,2.4)
place(LASTA,tZ+16.5,0.085,0.0)
halaphon(tZ+17.0,note("pno",81)[:int(3.4*SR)],cents=8.0,taps=7,delay=1.9,fb=0.70,gain=0.075,spread=0.8,lpf=5000)
halaphon(tZ+19.4,note("bell5",81)[:int(3.0*SR)],cents=-11.0,taps=6,delay=2.3,fb=0.66,gain=0.055,spread=0.75,lpf=4200)
def quietest(y,win=1.6):
    w=int(win*SR); best=None; bi=0
    for a2 in range(0,max(1,len(y)-w),w//2):
        e=float(np.mean(y[a2:a2+w]**2))
        if best is None or e<best: best=e; bi=a2
    return y[bi:bi+w].copy()
CRACK=norm(np.concatenate([quietest(sp(P_VADER,60,78)),quietest(sp(P_MARIO,10,28)),quietest(sp(P_NUIT,150,168))]))
tR=tZ+20.0
place(edge(room(FB("MechanicalMetals",150.0,150.8)),0.05,0.9),tR-1.4,0.11,-0.1)
EB=44.0
te=np.arange(int(EB*SR))/SR
rr=np.random.default_rng(20260730)
surf=lp(rr.standard_normal(len(te)),2600)*0.22
imp=np.zeros(len(te))
for _ in range(int(EB*11)):
    i=int(rr.uniform(0,len(te)-400)); Lp=int(rr.uniform(0.0008,0.0045)*SR)
    imp[i:i+Lp]+=(rr.uniform(0.15,1.0)**2.6)*np.exp(-np.arange(Lp)/(Lp*0.4))*(1 if rr.random()<0.5 else -1)
bed=norm(surf*0.65+hp(imp,700)*0.9)
tide=np.interp(te,[0,3,14,28,EB],[0,0.135,0.105,0.045,0])*(0.88+0.12*np.sin(2*np.pi*0.027*te))
place_pan(bed*tide,tR,1.0,-0.12,0.12,rate=0.018,depth=0.1)
AIR=sustain(lp(WINDL,260),EB+1.0,xf=1.4)[:len(te)]
place_pan(AIR*np.interp(te,[0,7,24,EB],[0.045,0.075,0.04,0]),tR,1.0,0.2,-0.2,rate=0.014,depth=0.18)
place(edge(lp(note("bell5",81,cents=-9),1700),2.2,6.0),tR+5.0,0.019,0.28)
# ---- the house: a hall send over the first four minutes ----
def hall_ir(dur=3.4,seed=5):
    r=np.random.default_rng(seed)
    n=int(dur*SR); t=np.arange(n)/SR
    y=r.standard_normal(n)
    lo=lp(y,900)*np.exp(-t/(dur*0.44)); mid=bp(y,900,2500)*np.exp(-t/(dur*0.27))
    hi=hp(y,2500)*np.exp(-t/(dur*0.13))
    z=lo*0.85+mid*0.9+hi*0.45
    for d,g in [(0.013,0.45),(0.021,0.36),(0.034,0.29),(0.052,0.22),(0.073,0.17)]:
        i=int(d*SR); z[i:i+500]+=g*np.hanning(500)*0.5
    return z/ (np.sqrt(np.sum(z**2))+1e-9)
def send(buf,ir,t_end,wet):
    n_end=int(t_end*SR); BL=int(45*SR); out=np.zeros(len(buf))
    pos=0
    while pos<n_end:
        b=min(n_end,pos+BL)
        w=fftconvolve(buf[pos:b],ir)
        e=min(len(out),pos+len(w))
        out[pos:e]+=w[:e-pos]
        pos=b
    tt=np.arange(len(out))/SR
    ramp=np.interp(tt,[0,8,230,286,308],[0.9,1.0,1.0,0.4,0.0])
    return out*ramp*wet
irL=hall_ir(3.5,5); irR=hall_ir(3.8,11)
Lc+=send(Lc,irL,300.0,0.42); Rc+=send(Rc,irR,300.0,0.42)
END=int(min(DUR,tR+EB+2.0)*SR); Lc=Lc[:END]; Rc=Rc[:END]
fade=int(14.0*SR)
Lc[-fade:]*=np.linspace(1,0,fade)**2.0; Rc[-fade:]*=np.linspace(1,0,fade)**2.0
def declick(ch,thr=5.0,win=96):
    d=np.diff(ch,prepend=ch[0])
    loc=np.convolve(np.abs(ch),np.ones(4800)/4800,'same')+1e-6
    bad=np.where(np.abs(d)>thr*loc)[0]
    if len(bad)==0: return ch,0
    out=ch.copy(); n=0
    for i in bad:
        a=max(0,i-win); b=min(len(ch),i+win)
        if b-a<8: continue
        ramp=np.hanning(b-a)
        smooth=np.convolve(ch[a:b],np.ones(9)/9,'same')
        out[a:b]=ch[a:b]*(1-ramp)+smooth*ramp
        n+=1
    return out,(n,[round(float(i)/SR,1) for i in bad[:8]])
Lc,n1=declick(Lc); Rc,n2=declick(Rc)
print(f"  declicked L={n1[0]} at {n1[1]} | R={n2[0]}")
st=np.stack([Lc,Rc],1); st=norm(st)*10**(-1.0/20)
sf.write("/mnt/user-data/outputs/007_mvt5_v20.wav",st,SR,subtype="PCM_24")
print(f"V v20: {len(st)/SR:.1f}s | air 0-78 | voice 78-168 | word 172-{tW:.0f} | contrappunto {tX:.0f}-{tX+74:.0f} | cases -{tK:.0f} | Culp {tV:.0f} on midi {mC} | congedo {tA:.0f}+")
