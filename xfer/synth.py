"""
engine/synth.py — DSP primitives for the Elzhéar assemblage.

Pure functions returning float64 numpy arrays at SR. Signals are mono 1-D unless
noted; spatialize with pan / to_stereo, combine with mix, and render to disk with
write_wav (24-bit PCM by default).

Groups
------
oscillators : sine, fm, additive, karplus_strong
noise       : noise (white / pink / brown)
envelopes   : adsr, perc_env, apply_env
tape        : saturate, wow_flutter, dropout
digital     : bit_crush, decimate, click
granular    : granulate
industrial  : modal, sub
room        : synth_ir, convolve_reverb
util / IO   : db_to_amp, amp_to_db, samples, normalize, pan, to_stereo, fade,
              mix, write_wav

All frequencies in Hz, all durations in seconds, all gains linear unless the
name says dB.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, lfilter

from .config import HEADROOM_DB, SR, SUBTYPE

TWO_PI = 2.0 * np.pi
ArrayLike = Union[float, Sequence[float], np.ndarray]

__all__ = [
    "sine", "fm", "additive", "karplus_strong", "noise",
    "adsr", "perc_env", "apply_env",
    "saturate", "wow_flutter", "dropout",
    "bit_crush", "decimate", "click",
    "granulate", "modal", "sub",
    "synth_ir", "convolve_reverb",
    "db_to_amp", "amp_to_db", "samples",
    "normalize", "pan", "to_stereo", "fade", "mix", "write_wav",
]


# ---------------------------------------------------------------- helpers ----
def samples(dur: float, sr: int = SR) -> int:
    """Number of samples in `dur` seconds."""
    return int(round(dur * sr))


def _t(n: int, sr: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / sr


def db_to_amp(db: float) -> float:
    return 10.0 ** (db / 20.0)


def amp_to_db(a: float) -> float:
    return 20.0 * np.log10(max(float(a), 1e-12))


def _as_env(x: ArrayLike, n: int) -> np.ndarray:
    """Broadcast a scalar to length n, or resample a sequence to length n."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if len(arr) == n:
        return arr
    return np.interp(np.linspace(0.0, 1.0, n),
                     np.linspace(0.0, 1.0, len(arr)), arr)


# ----------------------------------------------------------- oscillators ----
def sine(freq: ArrayLike, dur: float, sr: int = SR,
         amp: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Sine. `freq` may be a scalar or a per-sample envelope (instantaneous freq)."""
    n = samples(dur, sr)
    f = _as_env(freq, n)
    ph = TWO_PI * np.cumsum(f) / sr + phase
    return amp * np.sin(ph)


def fm(carrier: float, ratio: float, index: ArrayLike, dur: float,
       sr: int = SR, amp: float = 1.0) -> np.ndarray:
    """2-operator phase-modulation FM. `index` may be a scalar or envelope
    (e.g. an exponentially decaying index for a brightness sweep)."""
    n = samples(dur, sr)
    t = _t(n, sr)
    idx = _as_env(index, n)
    modulator = np.sin(TWO_PI * carrier * ratio * t)
    return amp * np.sin(TWO_PI * carrier * t + idx * modulator)


def additive(f0: float, dur: float, sr: int = SR, partials: int = 8,
             amps: Optional[Sequence[float]] = None,
             ratios: Optional[Sequence[float]] = None,
             decays: Optional[Sequence[float]] = None,
             amp: float = 1.0) -> np.ndarray:
    """Sum of sinusoidal partials. `ratios` default to the harmonic series;
    `decays` (seconds, per partial) apply independent exponential decay."""
    n = samples(dur, sr)
    t = _t(n, sr)
    if ratios is None:
        ratios = [k + 1 for k in range(partials)]
    if amps is None:
        amps = [1.0 / (k + 1) for k in range(len(ratios))]
    out = np.zeros(n)
    for i, r in enumerate(ratios):
        a = amps[i] if i < len(amps) else 0.0
        comp = a * np.sin(TWO_PI * f0 * r * t)
        if decays is not None and i < len(decays) and decays[i] is not None:
            comp *= np.exp(-t / max(float(decays[i]), 1e-4))
        out += comp
    return amp * out


def karplus_strong(freq: float, dur: float, sr: int = SR,
                   decay: float = 0.996, seed: Optional[int] = None,
                   amp: float = 1.0) -> np.ndarray:
    """Plucked-string (Karplus-Strong) via its exact LTI feedback form.
    `decay` near 1.0 lengthens sustain."""
    n = samples(dur, sr)
    L = max(2, int(round(sr / float(freq))))
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    burst = min(L, n)
    x[:burst] = rng.uniform(-1.0, 1.0, burst)
    a = np.zeros(L + 2)
    a[0] = 1.0
    a[L] = -0.5 * decay
    a[L + 1] = -0.5 * decay
    y = lfilter([1.0], a, x)
    peak = np.max(np.abs(y)) or 1.0
    return amp * y / peak


def noise(dur: float, sr: int = SR, color: str = "white",
          seed: Optional[int] = None, amp: float = 1.0) -> np.ndarray:
    """White, pink (1/f), or brown noise, peak-normalized."""
    n = samples(dur, sr)
    rng = np.random.default_rng(seed)
    if color == "white":
        x = rng.uniform(-1.0, 1.0, n)
    elif color == "pink":
        w = rng.standard_normal(n)
        X = np.fft.rfft(w)
        f = np.fft.rfftfreq(n)
        if len(f) > 1:
            f[0] = f[1]
        else:
            f[0] = 1.0
        x = np.fft.irfft(X / np.sqrt(f), n)
    elif color == "brown":
        x = np.cumsum(rng.standard_normal(n))
    else:
        raise ValueError(f"unknown noise color: {color!r}")
    peak = np.max(np.abs(x)) or 1.0
    return amp * x / peak


# ------------------------------------------------------------- envelopes ----
def adsr(n: int, sr: int = SR, a: float = 0.01, d: float = 0.1,
         s: float = 0.7, r: float = 0.2) -> np.ndarray:
    """ADSR envelope of length n samples. Stages are clamped to fit n."""
    A, D, R = samples(a, sr), samples(d, sr), samples(r, sr)
    if A + D + R > n:  # compress to fit
        scale = n / max(A + D + R, 1)
        A, D, R = int(A * scale), int(D * scale), int(R * scale)
    S = max(0, n - A - D - R)
    env = np.empty(0)
    if A:
        env = np.concatenate([env, np.linspace(0.0, 1.0, A)])
    if D:
        env = np.concatenate([env, np.linspace(1.0, s, D)])
    if S:
        env = np.concatenate([env, np.full(S, s)])
    if R:
        start = env[-1] if env.size else s
        env = np.concatenate([env, np.linspace(start, 0.0, R)])
    if env.size < n:
        env = np.pad(env, (0, n - env.size))
    return env[:n]


def perc_env(n: int, sr: int = SR, attack: float = 0.002,
             tau: float = 0.4) -> np.ndarray:
    """Percussive envelope: fast attack then exponential decay (tau seconds)."""
    t = _t(n, sr)
    env = np.exp(-t / max(tau, 1e-4))
    A = samples(attack, sr)
    if A:
        env[:A] *= np.linspace(0.0, 1.0, A)
    return env


def apply_env(sig: np.ndarray, env: ArrayLike) -> np.ndarray:
    return sig * _as_env(env, len(sig))


# ------------------------------------------------------------ tape stratum ---
def saturate(sig: np.ndarray, drive: float = 2.0, amp: float = 1.0) -> np.ndarray:
    """Tanh soft-clip; `drive` raises harmonic content. Output stays ~unity."""
    d = max(float(drive), 1e-3)
    return amp * np.tanh(d * sig) / np.tanh(d)


def wow_flutter(sig: np.ndarray, sr: int = SR,
                wow_rate: float = 0.6, wow_depth: float = 0.0025,
                flutter_rate: float = 7.0, flutter_depth: float = 0.0006,
                seed: Optional[int] = None) -> np.ndarray:
    """Pitch instability via time-varying fractional resampling. Depths in seconds."""
    n = len(sig)
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    pw, pf = rng.uniform(0, TWO_PI), rng.uniform(0, TWO_PI)
    mod = (wow_depth * np.sin(TWO_PI * wow_rate * t / sr + pw)
           + flutter_depth * np.sin(TWO_PI * flutter_rate * t / sr + pf))
    read = np.clip(t - mod * sr, 0, n - 1)
    return np.interp(read, t, sig)


def dropout(sig: np.ndarray, sr: int = SR, rate: float = 0.4,
            min_ms: float = 20.0, max_ms: float = 120.0,
            depth: float = 0.85, seed: Optional[int] = None) -> np.ndarray:
    """Random brief amplitude dips (tape dropout). `rate` = events per second."""
    n = len(sig)
    out = sig.copy()
    rng = np.random.default_rng(seed)
    for _ in range(int(rate * (n / sr))):
        start = int(rng.integers(0, n))
        length = samples(rng.uniform(min_ms, max_ms) / 1000.0, sr)
        end = min(n, start + length)
        if end > start:
            out[start:end] *= (1.0 - depth * np.hanning(end - start))
    return out


# --------------------------------------------------------- digital stratum ---
def bit_crush(sig: np.ndarray, bits: int = 8) -> np.ndarray:
    """Quantize to `bits` of resolution."""
    levels = 2 ** (max(1, int(bits)) - 1)
    return np.round(sig * levels) / levels


def decimate(sig: np.ndarray, factor: int = 4) -> np.ndarray:
    """Sample-and-hold downsampling without anti-aliasing (intentional grit)."""
    factor = max(1, int(factor))
    return np.repeat(sig[::factor], factor)[:len(sig)]


def click(sr: int = SR, freq: float = 3000.0, dur: float = 0.004,
          amp: float = 1.0) -> np.ndarray:
    """Short windowed sine — the Raster sine-click."""
    n = samples(dur, sr)
    win = np.hanning(n) if n > 1 else np.ones(n)
    return amp * np.sin(TWO_PI * freq * np.arange(n) / sr) * win


# -------------------------------------------------------------- granular -----
def granulate(source: np.ndarray, sr: int = SR, dur: float = 4.0,
              grain_ms: float = 80.0, density: float = 24.0,
              pitch_semitones: float = 0.0, pitch_jitter: float = 0.0,
              pos: Optional[float] = None, pos_jitter: float = 0.15,
              amp_jitter: float = 0.4, seed: Optional[int] = None,
              amp: float = 1.0) -> np.ndarray:
    """Grain cloud from a mono `source` buffer.

    density  : grains per second
    pos      : normalized read position 0..1 (None = scatter across the source)
    *_jitter : random spread applied per grain
    Returns a mono buffer of `dur` seconds, peak-normalized.
    """
    n = samples(dur, sr)
    out = np.zeros(n + sr)  # headroom for grains near the end
    src = np.asarray(source, dtype=np.float64)
    if src.ndim > 1:
        src = src.mean(axis=1)
    ls = len(src)
    rng = np.random.default_rng(seed)
    gl = max(8, samples(grain_ms / 1000.0, sr))
    window = np.hanning(gl)
    for _ in range(max(1, int(density * dur))):
        op = int(rng.integers(0, n))
        center = (rng.uniform(0, 1) if pos is None else float(pos))
        center = min(max(center + rng.uniform(-pos_jitter, pos_jitter), 0.0), 1.0)
        sp = int(center * max(0, ls - gl))
        semis = pitch_semitones + rng.uniform(-pitch_jitter, pitch_jitter)
        ratio = 2.0 ** (semis / 12.0)
        read_len = int(gl * ratio)
        if read_len < 2 or sp + read_len >= ls:
            grain = src[sp:sp + gl]
            if len(grain) < gl:
                grain = np.pad(grain, (0, gl - len(grain)))
        else:
            idx = sp + np.arange(gl) * ratio
            grain = np.interp(idx, np.arange(ls), src)
        gain = 1.0 - amp_jitter * rng.uniform(0, 1)
        out[op:op + gl] += grain[:gl] * window * gain
    out = out[:n]
    peak = np.max(np.abs(out)) or 1.0
    return amp * out / peak


# ----------------------------------------------------------- industrial -----
def modal(freq: float, dur: float, sr: int = SR,
          ratios: Sequence[float] = (1.0, 2.76, 5.40, 8.93),
          decays: Sequence[float] = (0.8, 0.5, 0.3, 0.2),
          amps: Sequence[float] = (1.0, 0.6, 0.4, 0.25),
          attack: float = 0.001, amp: float = 1.0) -> np.ndarray:
    """Modal/bell synthesis: decaying inharmonic partials. Defaults read metallic."""
    n = samples(dur, sr)
    t = _t(n, sr)
    out = np.zeros(n)
    for r, dec, a in zip(ratios, decays, amps):
        out += a * np.sin(TWO_PI * freq * r * t) * np.exp(-t / max(dec, 1e-4))
    A = samples(attack, sr)
    if A:
        out[:A] *= np.linspace(0.0, 1.0, A)
    peak = np.max(np.abs(out)) or 1.0
    return amp * out / peak


def sub(freq: float, dur: float, sr: int = SR, drive: float = 1.5,
        attack: float = 0.005, release: float = 0.05,
        amp: float = 1.0) -> np.ndarray:
    """Saturated sine sub-bass with a short AR envelope."""
    s = saturate(sine(freq, dur, sr), drive)
    env = adsr(samples(dur, sr), sr, a=attack, d=0.0, s=1.0, r=release)
    return amp * s * env


# ---------------------------------------------------------------- room ------
def synth_ir(sr: int = SR, rt60: float = 2.0, predelay_ms: float = 0.0,
             stereo: bool = True, seed: Optional[int] = None) -> np.ndarray:
    """Synthetic impulse response: exponentially decaying noise (RT60 seconds).
    Returns (m,) mono or (m, 2) stereo with decorrelated channels."""
    n = int(max(0.1, rt60) * sr * 1.1)
    t = np.arange(n) / sr
    decay = np.exp(-6.9078 * t / max(rt60, 1e-3))  # -60 dB at t = rt60
    pre = samples(predelay_ms / 1000.0, sr)

    def channel(s: Optional[int]) -> np.ndarray:
        x = np.random.default_rng(s).uniform(-1.0, 1.0, n) * decay
        return np.concatenate([np.zeros(pre), x]) if pre else x

    if stereo:
        L = channel(1 if seed is None else seed)
        R = channel(2 if seed is None else seed + 1)
        peak = max(np.max(np.abs(L)), np.max(np.abs(R))) or 1.0
        return np.stack([L / peak, R / peak], axis=1)
    x = channel(seed)
    peak = np.max(np.abs(x)) or 1.0
    return x / peak


def convolve_reverb(sig: np.ndarray, ir: np.ndarray, mix: float = 0.3) -> np.ndarray:
    """Convolution reverb. Accepts mono or stereo sig/ir; returns stereo of length
    len(sig)+len(ir)-1 so the tail survives. `mix` blends wet into dry."""
    s = to_stereo(sig)
    h = to_stereo(ir)
    out_len = s.shape[0] + h.shape[0] - 1
    wet = np.stack([fftconvolve(s[:, 0], h[:, 0]),
                    fftconvolve(s[:, 1], h[:, 1])], axis=1)
    peak = np.max(np.abs(wet)) or 1.0
    wet /= peak
    dry = np.zeros((out_len, 2))
    dry[:s.shape[0]] = s
    return (1.0 - mix) * dry + mix * wet


# --------------------------------------------------------- util / mix / IO ---
def to_stereo(x: np.ndarray) -> np.ndarray:
    """Return an (n, 2) array. Mono input is duplicated to both channels."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x


def pan(mono: np.ndarray, position: float = 0.0) -> np.ndarray:
    """Constant-power pan of a mono signal. position -1 (L) .. +1 (R)."""
    p = float(np.clip(position, -1.0, 1.0))
    angle = (p + 1.0) * 0.25 * np.pi
    m = np.asarray(mono, dtype=np.float64)
    return np.stack([m * np.cos(angle), m * np.sin(angle)], axis=1)


def fade(sig: np.ndarray, fade_in: float = 0.01, fade_out: float = 0.05,
         sr: int = SR) -> np.ndarray:
    """Linear fades at head and tail; works on mono or stereo."""
    x = np.array(sig, dtype=np.float64, copy=True)
    nin, nout = samples(fade_in, sr), samples(fade_out, sr)
    if x.ndim == 1:
        if nin:
            x[:nin] *= np.linspace(0.0, 1.0, nin)
        if nout:
            x[-nout:] *= np.linspace(1.0, 0.0, nout)
    else:
        if nin:
            x[:nin] *= np.linspace(0.0, 1.0, nin)[:, None]
        if nout:
            x[-nout:] *= np.linspace(1.0, 0.0, nout)[:, None]
    return x


def mix(layers: Sequence[np.ndarray],
        gains: Optional[Sequence[float]] = None) -> np.ndarray:
    """Sum layers (mono or stereo) into a stereo bus, aligning to the longest."""
    stereo = [to_stereo(layer) for layer in layers]
    n = max(s.shape[0] for s in stereo)
    acc = np.zeros((n, 2))
    for i, s in enumerate(stereo):
        g = 1.0 if gains is None else float(gains[i])
        acc[:s.shape[0]] += g * s
    return acc


def normalize(sig: np.ndarray, peak_db: float = HEADROOM_DB) -> np.ndarray:
    """Scale so the absolute peak sits at `peak_db` dBFS."""
    x = np.asarray(sig, dtype=np.float64)
    peak = np.max(np.abs(x)) or 1.0
    return x * (db_to_amp(peak_db) / peak)


def write_wav(path: Union[str, Path], sig: np.ndarray, sr: int = SR,
              subtype: str = SUBTYPE, peak_db: float = HEADROOM_DB,
              normalize_output: bool = True) -> str:
    """Write a 24-bit PCM WAV (default). Accepts mono (n,) or stereo (n, 2)."""
    x = np.asarray(sig, dtype=np.float64)
    if normalize_output:
        x = normalize(x, peak_db)
    x = np.clip(x, -1.0, 1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype=subtype)
    return str(path)
