# 007 — Movement II: La Kermesse — code notes

Locked script: `sandbox/007/mvt2_LOCKED.py` (lineage c1–c3, n1 → **n12**), commit
`cceb9c81`. Duration 288.0 s. BPM 120; BEAT 0.5 s, BAR 2.0 s. PCM_24, −1.0 dBFS.

The scherzo: the fair that followed the murder. Collage discipline throughout — the
samples are quoted, not conjugated (that waits for IV).

## Form

| t | section |
|---|---|
| 0:00–0:20 | grindhoused archivist slates, THUDs, pit heckle |
| 0:20–0:30 | ballad stomp — exactly ten straight bars, the lurch removed; the Lomax singer floats over it and outlives the cut a cappella |
| 0:30–1:20 | **the carousel** — Marionette (1919) in eight chained chunks at dying rates 1.0 → 0.94; GRIND = the industrialised M3 riff plus its retrograde gstretched 2.5×; Berg Op. 5 (source 408–450 s) as a near-continuous rubato counter-voice through `bergvoice()`; Chaliapin spiral undertow 0 → −7 st; orchestral punctuation; L2SPK passing speech |
| 1:20–2:16 | **the fuse** — 56 s burn rising 0.05 → 0.86, sub-burn an octave down, Yukhov choir spiralling −5, stacking tremolos, timpani roll, grindhoused READ; the cold ticks fade out |
| 2:16–2:38 | **the outburst** — Maelstrom slabs via `slabfix()`, emergence 0.6 s, gains 0.7 / 0.56 / 0.66, SLAB2 spiralled −4; driven CTB pedal ×3; orchestra-as-percussion (8 timpani, 8 anvils, 3 trombone falls, **no kit**); Berg clarinet spire (452–462) over the wall; shrill fiddle +5 st; debris spiral after the cut |
| 2:38–3:03 | Siegfried, unopposed |
| 3:03–3:42 | **ember / horror** — 17-strike anvil clock at 52 BPM (1.15 s apart), each doubled by ironknock at 98 Hz; the burn envelope is base plus per-strike ignition bumps, so fire and iron are one organism; fiddle threnody gstretched 2.05× as a continuous floor; the *You Better Run* sequence in five stages |
| 3:42–4:48 | **night** — half-speed NOLA bell, Yukhov, violin tremolo, Berg quiet piano (268–276), hum ghost, bell, contrabass |

## Devices

`bergvoice()` — hp 230 / lp 8000 with a presence lift 900–2600; keeps the Berg quartet
legible as a rubato line under the carousel without letting it foreground.

`slabfix()` — hp 170, mud-cut at 0.3, octave fuzz via |y|, presence lift, ring sheen.
The wall never carries added kit; the orchestra is the percussion.

The *You Better Run* sequence: `horrify(−2)` at 3:09 → choir nearly plain at 3:13 →
mildly wrong at 3:18 → grindhoused narration (lomax2 240.5–244) faded over 1.0 s →
ascending faded stutters (the first through lp 1400) → stage 2 through an opening
filter → stage 3 at (−8, q 5, drv 3) → debris spiral −6.

## Rules the movement established

- The stomp is one cameo of ten seconds or less; it does not return.
- Maelstrom never carries an added drum kit.
- Voices are grindhouse-treated as a class, so speech and song share one grain.
- Fixes emerge: fades, filter-opens, ascending gains — nothing switches on.
- A threnody held as a continuous frequency kills edges better than any fade.
- Hour-bell material must have a faded tail.

## Prunable

`sandbox/007/wip/mvt2_n3.py` and `mvt2_n4.py` are development snapshots superseded by
the lock and may be removed.
