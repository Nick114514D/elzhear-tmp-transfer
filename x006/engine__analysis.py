"""
engine/analysis.py — feature extraction (the Curator's ear).

analyze() returns a flat dict of descriptors for a render: levels and dynamics,
tempo and onset density, spectral shape, timbre (MFCC), harmony (chroma), and a
harmonic/percussive balance. These are the measurable quantities curator.py will
weigh against brief.md.

feature_vector() / distance() give a compact, roughly-normalized comparison used
to read the "residue" between consecutive pieces (the transductive law): a small
distance means the next piece stayed close, a large one means it departed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import librosa

from .config import SR

__all__ = ["load", "analyze", "summary", "feature_vector", "distance"]

Source = Union[str, Path, np.ndarray]


def load(path: Union[str, Path], sr: int = SR, mono: bool = True):
    y, sr = librosa.load(str(path), sr=sr, mono=mono)
    return y, sr


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def _mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return y
    ch_axis = 0 if y.shape[0] <= y.shape[1] else 1  # average the length-2 axis
    return y.mean(axis=ch_axis)


def analyze(source: Source, sr: int = SR) -> dict:
    """Extract descriptors from a file path or a mono/stereo array."""
    if isinstance(source, (str, Path)):
        y, sr = load(source, sr=sr, mono=True)
        path = str(source)
    else:
        y, path = _mono(source), None
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        raise ValueError("empty audio")

    dur = len(y) / sr
    peak = float(np.max(np.abs(y)))

    rms_frames = librosa.feature.rms(y=y)[0]
    rms = float(np.mean(rms_frames))
    rms_db_frames = 20.0 * np.log10(np.maximum(rms_frames, 1e-12))
    dyn = float(np.percentile(rms_db_frames, 95) - np.percentile(rms_db_frames, 5))

    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(np.atleast_1d(
            librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0])[0])
    except Exception:
        tempo = float("nan")
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        onset_rate = float(len(onsets) / dur) if dur > 0 else 0.0
    except Exception:
        onset_rate = float("nan")

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    try:
        H, P = librosa.effects.hpss(y)
        eh, ep = float(np.sum(H ** 2)), float(np.sum(P ** 2))
        harm_ratio = eh / (eh + ep) if (eh + ep) > 0 else float("nan")
    except Exception:
        harm_ratio = float("nan")

    def opt(x: float, nd: int):
        return round(x, nd) if x == x else None  # NaN -> None

    return {
        "path": path,
        "sr": int(sr),
        "duration_s": round(dur, 3),
        "peak_dbfs": round(_db(peak), 2),
        "rms_dbfs": round(_db(rms), 2),
        "crest_db": round(_db(peak) - _db(rms), 2),
        "dynamic_range_db": round(dyn, 2),
        "tempo_bpm": opt(tempo, 2),
        "onset_rate_hz": opt(onset_rate, 3),
        "centroid_hz_mean": round(float(np.mean(cent)), 1),
        "centroid_hz_std": round(float(np.std(cent)), 1),
        "bandwidth_hz_mean": round(float(np.mean(bw)), 1),
        "rolloff_hz_mean": round(float(np.mean(roll)), 1),
        "flatness_mean": round(float(np.mean(flat)), 5),
        "zcr_mean": round(float(np.mean(zcr)), 5),
        "harmonic_ratio": opt(harm_ratio, 3),
        "mfcc_mean": [round(float(v), 3) for v in np.mean(mfcc, axis=1)],
        "chroma_mean": [round(float(v), 3) for v in np.mean(chroma, axis=1)],
    }


def summary(feats: dict) -> str:
    """Compact human-readable digest of analyze() output."""
    return "\n".join([
        f"dur {feats['duration_s']}s  sr {feats['sr']}",
        f"peak {feats['peak_dbfs']} dBFS  rms {feats['rms_dbfs']} dBFS  "
        f"crest {feats['crest_db']} dB  dyn {feats['dynamic_range_db']} dB",
        f"tempo {feats['tempo_bpm']} bpm  onsets {feats['onset_rate_hz']}/s  "
        f"harmonic {feats['harmonic_ratio']}",
        f"centroid {feats['centroid_hz_mean']} Hz  rolloff {feats['rolloff_hz_mean']} Hz  "
        f"flatness {feats['flatness_mean']}  zcr {feats['zcr_mean']}",
    ])


# rough scales to put dissimilar features on comparable footing
_VEC = {
    "centroid_hz_mean": 4000.0, "bandwidth_hz_mean": 3000.0,
    "rolloff_hz_mean": 8000.0, "flatness_mean": 0.2, "zcr_mean": 0.2,
    "crest_db": 24.0, "harmonic_ratio": 1.0, "onset_rate_hz": 8.0,
    "tempo_bpm": 160.0,
}


def feature_vector(feats: dict) -> np.ndarray:
    """Roughly-normalized vector (spectral/dynamic scalars + MFCC timbre)."""
    v = []
    for k, scale in _VEC.items():
        x = feats.get(k)
        v.append((0.0 if x is None else float(x)) / scale)
    v.extend(c / 50.0 for c in feats.get("mfcc_mean", [])[:13])
    return np.asarray(v, dtype=np.float64)


def distance(a: dict, b: dict) -> float:
    """Euclidean distance between two analyses' feature vectors (residue size)."""
    va, vb = feature_vector(a), feature_vector(b)
    m = min(len(va), len(vb))
    return float(np.linalg.norm(va[:m] - vb[:m]))
