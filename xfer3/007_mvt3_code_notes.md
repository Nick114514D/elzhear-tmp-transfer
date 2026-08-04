# 007 — Movement III: Notturno — code notes

Locked script: `sandbox/007/mvt3_LOCKED.py` (lineage v1 → **v7**), commit `fac0bc83`.
Duration 540.0 s. RNG seed 672004. PCM_24, −1.0 dBFS.

A Scelsian ocean-sphere: nine minutes on one pitch, A♭, examined from the inside.
Homage to Scelsi; the movement that taught the album restraint.

## The seven breaths

`orb()` swells around A♭ at spreads of **8, 18, 30, 44, 56, 34, 16 cents**. Breaths 5
and 6 carry A♮ as the outermost orbit — the album's wound, held at the edge of the
sphere. Each breath has an `orb_lean` cantor trombone (the lesson of *Tre preghiere
latine*). The orb envelope is bound to the true post-resample length; an earlier
truncation bug cut the swells short and was fixed at v4.

## Water

Three layers — surf, shore rumble (lp 300), and an underwater bed (lp 800 with a 31 s
LFO) — tiled at staggered offsets 0 / 7 / 13 s, with a 47 s stereo migration. The torch
sizzle dies between 5:35 and 6:00; the sea persists to the end.

## Voices

`oceanvoice()` — a depop transient suppressor, chunked varispeed wobble, submersion
filter drifting 900–3400 at 0.05 Hz, a drowned IR tail, an lp 480 pressure double, and
an envelope whisper ghost. Applied to Shackleton (EDIS-SRP-0158-01, PD 1909/10; the
strong regions are 54, 82–86 and 188–200) at 92 s and 336 s, and to Conrad's *'Twixt
Land and Sea* (LibriVox) at 268 s over restored waterflow — a submerged Chopin arc
runs 262–328 with an 8 s surfacing.

## The indifferent prayers

The hymns do not breathe with the sphere: *Shenandoah* at 138, the Sistine *Ave Maria*
below hearing at 252, *Eternal Father* at 382 and 452 — the second crossing the album's
**first unison**, contrabass and trombone at zero cents, at 442. Dawn (violin harmonics
doubled, high harps ppp) rises above that unison.

## Detail

Seven hour-bells, each with a 1.8 s tail fade. A protagonist five-note contrabass line
(A♭, B, A, F♯, A♭) at 0:30 and 6:50 only. Six spiral gulls — lomax39 fiddle 162–163.1
through a spiral +7 → −3, hp 900. Three thunders (TROLL, driven, lp 350), one beneath
Conrad. A wrong harp crests breath 3 (A♭ plus D, the tritone) and breath 6 (a
quarter-sharp). Boulanger threnody at 352 (gain 0.3, lp 4600, 20 s). Twelve-second
final fade.

## What it deposited

The ocean of III becomes structural material in IV: it opens the finale, floats beneath
the noise aria, punctuates the scene joints as wave-crashes, and returns to close.
