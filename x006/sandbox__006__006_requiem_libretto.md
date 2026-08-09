> **Superseded — pre-production artifact.**
> This libretto plans three panels; the finished piece has four movements, and
> *Lux aeterna* appears nowhere below. The voice layer also departs from the plan:
> movement I uses a corroded LibriVox reading of Irenaeus's *Demonstration of the
> Apostolic Preaching* in place of the synthetic-Latin espeak pipeline specified here.
> Kept as a record of the piece's first shape. For what 006 became, see
> `corpus/pool/006_fifth_chapter_notes.md`.

# 006 "Requiem" — voice libretto (source layer)

Public-domain liturgical Latin (Requiem Mass / Dies Irae sequence, medieval).
Recitation, not bel canto: the source is corroded and granulated into a vocal
cloud (the method proven in 004). Intelligibility is optional — heavy corrosion
and clustering turn it into a choir of the dead, in the Penderecki manner
(St Luke Passion / Dies Irae: whisper, hiss, cluster, glide).

Pipeline per line:  espeak-ng -v la  ->  48 kHz/24-bit mono  ->  corrode_voice.py
->  granular cloud in the piece. Synthetic Latin keeps provenance clean (no
copyright), same posture as 004's public-domain reading.

---

## Panel I — Lament  (Requiem aeternam / Lacrimosa)
Near-silence out of 005's dead machine, held C#. Voice enters as thin mist.
Treatment: low, slow, heavily reverbed; sparse.

1. Requiem aeternam dona eis, Domine, et lux perpetua luceat eis.
   (Eternal rest grant unto them, O Lord, and let perpetual light shine on them.)
2. Lacrimosa dies illa, qua resurget ex favilla judicandus homo reus.
   (Tearful that day, when from the ashes shall rise the guilty to be judged.)

## Panel II — Pie Jesu  (the hinge)
Texture thins to almost nothing. The plea, most exposed the cloud ever gets,
over a single held tone that begins to sour. Unanswered.
Treatment: one near-clean line, minimal corrosion, long tail.

3. Pie Jesu Domine, dona eis requiem.
   (Merciful Lord Jesus, grant them rest.)
4. Pie Jesu Domine, dona eis requiem sempiternam.
   (...grant them rest everlasting.)

## Panel III — Dies Irae  (terror)
Cluster-masses swallow the plea. Full sonorist apparatus. The choir-cloud
shouts, hisses, clusters.
Treatment: dense, layered, pitch-multiplied, distorted; off-grid.

5. Dies irae, dies illa, solvet saeclum in favilla, teste David cum Sibylla.
   (Day of wrath, that day, will dissolve the world in ashes...)
6. Rex tremendae majestatis, qui salvandos salvas gratis, salva me, fons pietatis.
   (King of tremendous majesty... save me, fount of mercy.)
7. Confutatis maledictis, flammis acribus addictis, voca me cum benedictis.
   (When the accursed are confounded, consigned to the bitter flames...)

Seam: opens on C# (dominant of 005's F# minor); bells toll C#/D#/D, the inherited
pitch centre 005 leaves. The lament's C# is buried alive in the Dies Irae wall.
