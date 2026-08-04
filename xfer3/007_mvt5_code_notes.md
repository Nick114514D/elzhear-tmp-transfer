# 007 — Movement V: L'Inventaire — code notes

Locked script: `sandbox/007/mvt5_LOCKED.py` (lineage v1 → v20), commit `6632d78d`.
Duration 546.7 s. PCM_24, −1.0 dBFS peak.

The reliquary. Johan de Witt's tongue and finger are still in a case in The Hague.
Homage to Nono and Feldman; both composers are in copyright, so the homage is technique
alone and no material of theirs is sampled.

## The row

Begin on **A** and alternate outward by expanding intervals — +1, −2, +3, −4, +5, −6,
+7, −8, +9, −10, +11. The wedge closes on all twelve pitches with every interval class
exactly once:

> **A · B♭ · A♭ · B · G · C · F♯ · C♯ · F · D · E · E♭**

Movement I ended *"A♭. Never A."* Four movements held that refusal; here A is the origin
of the fan, and its first two neighbours are the two wounds the record has carried.
Registers fan with the row: odd members climb, even members fall.

**The row governs four parameters.** Pitch (`rowmidi`). **Time** — a unit of 0.62 s
multiplied by the interval that produced each member, so durations expand 1, 2, 3 … 11
exactly as the intervals do; thirty-five gesture lengths read off it and no round
numbers remain. **Space** — `ROWPAN` maps each member to a position. **Voice** —
`ROWTHROAT` chooses which of the seven throats carries each syllable.

## Harmony

The wedge pairs give six chords — **A+B♭ · A♭+B · G+C · F♯+C♯ · F+D · E+E♭** — opening
from a semitone to the tritone and closing on a semitone. A returns as the axis in
chords II–VI. Each is voiced across up to nine instruments and five octaves, orchestrated
differently, and crossfaded into the next over ten seconds: the piece never stops
sounding.

## The noise ↔ pitch axis

Scored as its own parameter, `NP(t)`, and read by every layer. Brass: air through the
bore at 0 → full tone at 1. Strings: overpressure scratch → clean arco. Bowed metal
peaks at ≈ 0.45. Voices: breath → vowel. Tape: wide-band → narrow bands on chord tones.
The curve opens at 0.08, climbs to 0.88 as the voice acquires pitch, collapses to 0.14
for the tape movement, rises to 0.98 for Culp, and falls to nothing at the needle.

## The choir

Formant synthesis — glottal source with jitter and shimmer, five formants per vowel,
vibrato with its own onset, breath as a mixable layer. **Solo soprano** entering as
breath with barely a pitch inside it, phonetics decomposing u → o → a → e → i, every
note carrying a quarter-tone displacement, her breath fraction literally `1 − NP(t)`.
**Solo contralto**, lower and pressed. **Chorus of sixteen, 4·4·4·4**, each voice with
its own detuning across nine cents, its own vibrato phase and entry. In the tape
movement they sing unvoiced — sixteen people breathing the harmony.

## The suspended word

After *Il canto sospeso*: the Dutch name of 1672 — *redeloos, radeloos, reddeloos* — is
assembled one syllable at a time from **seven dead throats** (Willie, Moreschi,
Chaliapin, Culp twice, the tyrant's noise, the Dutch reader), all through one shared
throat-space, each syllable retuned onto a chord tone and each throat chosen by the row.
Four passes at contracting spread. No voice owns the word, and it is never spoken.
Three gong interruptions carry the three words' durations and are likewise never spoken.

## Sections

fan 0:04–0:56 (the row stated) · filters and sweeps · fasce, the micro-intervallic
band-clusters · the tape movement 2:42–3:56 · **the threnody 5:34–6:20** — 26 string
voices in quarter-tone clusters, each gliding by the interval that generated it and
sweeping across the field in the direction it climbs, with col legno strikes hard-panned
alternately · the seven cases, each relic heard through glass and answered by the room ·
**Julia Culp, 1917, whole and untouched, the orchestra tuning itself to her** ·
the congedo, the fan folding twelve to one · A♭ once, A answering · the ebb.

## Halaphon

A granular harmonizer in a feedback loop, each return transposed further than the last,
each tap crossing the field as it decays. On the soprano's phrases at +7 cents; on each
of the seven cases at its own interval (+6, −8, +11, −13, +9, −7, +14); twice on Culp;
and on the closing A. The original always sounds against its own double at half the
shift, so the beating is continuous.

## Repairs at v16–v20

Four gongs had been faded and then truncated, cutting each off mid-ring; all now
truncate first. The closing chord is prepared by two overpressure swells and rolled
across a second and a half. The seam at 8:05 is bridged by bowed and gonged material
looped **past its attack**, low-passed, shaped as a decay. `sustain()` was rewritten:
it had faded each loop segment in but never out, severing every cycle — the fix is a
complementary linear pair. The wind bed was gating the whole movement on a 101 s cycle
because the source recording fades to silence; it now uses the longest continuous span.
A declicker runs on the master. The ebb is synthesised and never repeats.

## Sources

University of Iowa MIS (arco pp per string, crotales, bells, bowed cymbals, gongs, piano
pp, found objects); GOLD TAPE CC0; aporee Dartington wind (**CC BY 3.0, attribution
required**); Rachmaninoff playing Debussy's *Golliwogg's Cake-Walk* (1921) and *La Fille
aux cheveux de lin* (1919); Julia Culp, *Nuit d'étoiles* (1917); and seven relics
quoted from the album's own earlier movements.
