# 007 — Movement I: Le Supplice — code notes (timestamped)

Locked script: `sandbox/007/mvt1_LOCKED.py`. Lineage: impro2 → impro13 (first lock,
superseded) → **r2 → r3 → r4 → r6 → r7 → r8 → r9** (current lock, commit `515bb1b1`).
Duration 287.0 s. SR 48000, PCM_24 master at −1.0 dBFS; demo encode 256 kbps MP3.
BPM 56 — BEAT 1.0714 s, STEP (bar) 2.1429 s, LOOP 8 bars = 17.143 s. RNG seed 672.

**Why it was reopened.** The impro13 lock read as a stacking ziggurat: 8 recordings in
24 segments, Willie spent 11 times, V1 hosting two complete dramas at once, a 9-layer
mob, a stranger brass arriving unprepared. The rebuild kept the original base and
grafted onto it; a from-scratch rebuild (r1) was made and rejected.

Form: passacaglia. Ground = 4 bars A♭ / 4 bars A, permanent until the stop, skeleton
after it. The A♭/A pair is the entire tonal argument. Two hard cuts only: the drop
(3:25.6, 0.9 s) and the stop (3:48.6, 0.7 s). Global bed: crackle from the quietest
2 s of the Moreschi, Chaliapin and Johnson discs, tiled, −36 dB, 13 ms L/R
decorrelation.

## What r2–r9 changed

| rev | change |
|---|---|
| r2 | V1's seven-exchange debate deleted. Replaced by a **merged Willie–soprano relay**: six segments through one voice-chain (`onevoice()` — presence + room convolution, single position, 1 s crossfades), so the verse is one continuous line that changes nature without a seam. |
| r3 | The soprano latticed to Willie: `f0med()` detects both fundamentals, she is repitched by the exact interval, band-boxed to his 160–2800, and her amplitude contour divided out so her rises and falls stop competing. |
| r4 | "Reproduce" read as fusion, not matching. `embrace()` — 75 ms grains at half-overlap, each voice **amplitude-shaped by the other's envelope**, plus a 14 % ring-modulation of their product. First guitar mortar from *Dark Was the Night*. |
| r6 | Moreschi drawn into the same language (own f0 detected, own interval, embraced with WCONT[8:16]). V3 soprano fades in from zero. Ending rebuilt as a **spiral fade-out**; the closing knocks and final sub deleted. |
| r7 | Shrilling intro: FID_KEEN at 1.2 s plus its +7-semitone shadow at 5.5 s; the assembly knock gated to bar ≥ 2 with a ramp, so the machine is discovered inside the keening. |
| r8 | V3 soprano's envelope closed to true zero (the 1:42 edge). |
| r9 | WS2 trimmed to 10 s with a 1.6 s fade — the cut-off "ah" at 2:43 deleted. |

## Source segments (source-file timecodes)

| tag | file | cut | role |
|---|---|---|---|
| WOPEN / WCONT / WCALL | 29075_Dark_was_the_night | 2–14 / 14–44 / 60–68 | embraced V1 and V2 material |
| WS1 / WS2 | same | 42–56 / 96–106 | burn vocals (WS2 trimmed at r9) |
| WMOAN / WHUM | same | 90–102 / 162–174 | mob spiral, coda ghost, spiral source |
| dwn (gglue) | same | 1.5–7.5 / 13.5–19.5 / 37.5–45.5 / 3–7 | guitar mortar at four seams |
| MOR_A | moreschi 08 (Crucifixus) | 147.5–163 | the body's statement, embraced |
| SOPR / SOPR_CAD / PIANO | moreschi 13 | 216.5–231.5 / 247–254 / 12.5–19.8 | V3 verse; the glide; ghosts |
| C13[0..2] | moreschi 13 | 21–25 / 33–36.6 / 45–48.4 | the female side of the chimera |
| FID_BODY / FID_KEEN | lomax afs02247a | 120–142 / 160–176 | threnody; the intro keening |
| CHA_C / ZER / LAUGH | chaliapin (Kazan) | 137.5–152 / 100–117 / 145–145.35 | mob cells, climb, laughter |
| POPS / CRACK | harvested | 300 transients hp 3 kHz, 16 ms / quietest-2 s concat | vinyl percussion, bed |

**Deleted at the rebuild:** the L2 ballad-singer fragments (the debate), the Jerusalem
aporee voices JV1/JV2, the Ménilmontant bell flood BFLOOD, the written brass. None
appear in the locked score.

## Devices

`embrace(vw, vs, dur)` — the chimera. Two normalised voices tiled to length; 75 ms
Hanning grains at half-overlap alternate source; each grain multiplied by
`0.5 + 0.5 · (the other voice's normalised envelope)`; the product of both voices added
at 0.14 as a ring whisper; result high-passed 180, low-passed 3200, then through
`onevoice()`.

`herdown(v)` — resample by the detected interval, hp 240 (her rumble cut at r6),
lp 2800, then divide by `env·0.8 + 0.2` to flatten the contour.

`gglue(a, b)` — guitar segment, hp 130 / lp 4500, 18 % room convolution; placed at
17.0, 29.0, 44.5 (runs under the last woven word into V2) and 84.5 (V2 into V3).

Spiral ending — WHUM[0:4 s] and PIANO[0:4 s] concatenated, `spiral()` at factor 2.4,
50 ms grains, 0 → −12 semitones, lp 1600, envelope peaking 0.32 at 2 s and reaching
zero at 19.2 s, placed 265.0. Final fade 9 s, curve 1.2.

Intro — FID_KEEN hp 350, `edge(1.8, 3.0)`, gain 0.44 at 1.2 s; its +7-semitone shadow
hp 700, 11 s, gain 0.20 at 5.5 s; ground knock suppressed until bar 2, ramping to full
by bar 6.

## Timeline

0:01 keening · 0:05 shadow · 0:17 V1 (three embraced blocks, guitar at the seams) ·
0:51 V2, Moreschi embraced with WCONT · 1:26 V3 soprano (rises from zero, closes to
zero) · 2:00 lead-in · 2:16 burn (sizzle swells in; WS1, WS2, strafe) · 2:50 epilogue ·
3:05 mob · 3:25.6 drop · 3:44 glide (0.45 s entry breath) · 3:48.6 stop · 3:50 coda ·
4:16 Willie's enveloped farewell · 4:25 spiral, falling twelve semitones · 4:47 out.

**Rights.** The *Dark Was the Night* guitar enters under the same flag as the Johnson
vocal material: recording caution logged to 2029. See the repository README.
