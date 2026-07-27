# History — the archive. NOT the source of truth.

Everything in this directory is **frozen archival material**: the
documents that *produced* the ratified spec, kept for archaeology only.
Nothing here is maintained; much of it is superseded outright, and any
statement that disagrees with the living canon is simply wrong about
the current system. Every file carries the same warning banner.

**The living canon is `docs/design/200_the-spec.md`**, with its
companions: 210 (backend notes), 220 (principles), 230 (the executable
syntax tour), 250 (the coordinate algebra), 260 (the L4 brief). Read
those; come back here only to ask "why is it this way?" — the answer is
usually in 140–195.

## The map (read nothing below unless you're doing archaeology)

- **010–060 — the original architecture era.** The first proposal
  (010), its implementation plan (020) with the closure-specialization
  and caching evidence analyses (022, 024), and the early
  deep-learning, combinator, provenance, and rendering notes (030–060).
- **070–130 — the first system's build notes.** Backends and their
  organization (070, 080), core-vs-extensions (090), arrays/axes and
  transforms/derivatives (100, 110), events (120), tiles (130). The
  surviving knowledge was distilled into 210 and 220.
- **140–150 — the critical assessment.** The charter and the report:
  the adversarial review that motivated the clean break. The
  `research/` subdirectory holds its persona files (J1–J3, P1–P2).
- **160–195 — the integration decisions.** The documents that argued
  out and specified the clean break (160, 170), the GPT-2 end-to-end
  target (180), the integration spec (190), and indexed computation
  (195) — these were *inputs* to 200; 200 superseded them on
  ratification.
- **240 — the cleanup pivot**, executed; its content lives on in the
  current code and in 200's amendments.

The site build excludes this directory. Delete-don't-archive remains
the repo's working doctrine; this directory is the one deliberate
exception, held as the spec's paper trail by owner decision (P9 close).
