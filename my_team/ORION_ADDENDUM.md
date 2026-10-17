# Addendum — the fresh-start thread (Orion + learned-tier autopsy), 17 Oct ~16:00

Companion to REPORT.md. Everything below was measured this afternoon with the
same paired-seed protocol (`catan_kit.evaluate`, seeds 990000+, 0 faults
unless stated). One sentence up front: **v4 stands (0.5070 ±0.031 over a
fresh 1,000-game confirm on this machine); the exact-opponent engine was
built, tested twice, and is negative; the learned tier's collapse is now
fully diagnosed with a one-line root cause and a concrete route forward.**

## A. Orion — an engine that plays F's exact policy inside its own search

Motivation: every prior cross-turn attempt (v4xct teacher, v14 DART, v16)
modeled the opponent with our own eval — which misevaluates F by construction
(F has no enemy-VP term, never prices LR/LA denial, races production). F's
source ships in the kit: it is a deterministic argmax over catanatron's
`base_fn` on 1-ply successors. So we built the real thing: for each top
candidate action, play our END_TURN, sample a roll, let F take its EXACT
replies (same weights, same tie-break, robbers included), then score with
v4's eval; blend 50/50 with the 1-ply value. At initial placement, simulate
F's next two placements exactly and keep the plan that survives F's greedy
grab (real denial, not the −1e8 production proxy).

Measured costs made this look free: copy 34 µs, copy+execute 78 µs, F eval
71 µs, full F reply chain ~3 ms, F's 54-node placement argmax ~7 ms — against
~650 ms/game of unused clock.

**Result — two lessons, one embarrassing, one decisive.**

1. v1 (accidentally inert): a `current_color` *method* was compared as an
   attribute, so every "is it F's turn?" guard fired immediately and the
   layer only ever saw one dice roll ahead — v4 plus dice noise. It scored
   0.5060 ±0.031 / 1,000 games = exactly v4 (0.5070). Neutrality was the bug
   talking. (Screen-stage numbers are untrustworthy: a +0.03/200-game delta
   evaporated at 1,000.)
2. v2 (fixed, layer live): 0.4600 ±0.069 / 200 paired games vs v4's 0.4950 on
   identical seeds — **the exact-opponent layer is negative**. With v14
   (approximated, −0.09) and v16 (top-2 re-rank, neutral), the in-play
   cross-turn family is now closed at its strongest possible configuration.
   Single-roll reply noise plus a myopic leaf eval swamp the 30%-disagreement
   signal the offline teacher found. The signal is real; it is not
   deployable at 12.5 ms/decision in Python.

## B. Why every fitted eval collapsed to ~0.19 — the autopsy, solved

Eight learned variants today died at the same number. Labels were audited
sane, two architectures gave identical collapse, and the ledger recorded the
cause as an "unresolved systematic issue". We resolved it completely, with a
functional probe of the deployed checkpoint (probe_net.py) plus training-shard
forensics plus one instrumented retrain:

1. **The shipped checkpoint was dead** — on its own 20,000 training rows its
   logit spanned [−0.01, +0.13]; winners vs losers differed by 0.009 logits.
   As an argmax function it was a constant; degenerate play gives 0.19
   regardless of architecture.
2. **It was dead because it was undertrained by two orders of magnitude:**
   one epoch over the 578k rows costs ~1.7 s on this machine. "8 epochs" was
   ~14 seconds of SGD. The validation AUC was still climbing steeply when it
   stopped.
3. **And the recipe had a bomb in it: the regression target exceeded 1.**
   `y = 0.35·Z + 0.65·VPDR` reaches 1.13 (VPDR caps at 1.2), and BCE-with-
   logits treats (1−y) < 0 as a *reward* for unbounded logit growth. Trained
   past ~25 epochs the loss goes negative (−32.8 by epoch 30) and the logits
   explode to ±28. The 8-epoch run stopped just short of the cliff; anyone
   training "properly" with this target got divergence and concluded the data
   was broken.
4. **The one-line fix** (`y = clip(0.35Z + 0.65·min(VPDR,1), 0.01, 0.99)`)
   plus a real training run (120 epochs, ~3.5 min) produced a live network:
   **val AUC 0.8275** on terminal outcomes — far above the 0.711 AUC of
   v4's own eval terms on the same rows.

What the AUC did and did not buy in play (all vs F, paired seeds 990000+,
0 faults, ≤906 ms):

| Agent | Games | Win rate | Note |
|---|---|---|---|
| dead net (v2 weights) | 400 | 0.1875 | constant function |
| v3 pure net argmax | 200 | 0.4000 ±0.068 | outcome calibration ≠ live play |
| v3 ⊕ v4 rank-fusion 50/50 | 200 | 0.5250 ±0.069 | vs v4's 0.4950 paired |
| v3 ⊕ v4 rank-fusion 50/50 | 1,000 | see experiments.md #30 | confirm |

Reading: a well-trained outcome net alone still loses to a hand-tuned tactical
eval (successor-ranking ≠ position-labelling), but the fusion is the first
learned agent all day to sit at v4's level or above — and the diagnosis makes
the path explicit: teacher labels (the still-zero T column), then iterated
self-play rounds with the fused agent in the generation pool. That is the
leaders' recipe; it needs hours, not the one-line fix we had time for.

## C. Reading the leaderboard honestly

Top-10 at 15:09 IST: vsF 0.78–0.84, vsO 0.61–0.73, mean 0.71–0.79, at
107–303 ms CPU/game. At those speeds nobody out-searches anything in this
engine; that profile is the signature of fast learned evaluations. Our
measured results triangulate the same conclusion from below: handcrafted-eval
mutations (25 variants) plateau at ~0.52, search can't be bought with the
clock (10× depth agreed with v4 on 0/391 decisions), the exact-opponent
best-response is negative, and the learned tier is one dead label column
away from having been the right bet all along.

## D. Final numbers (this machine, PYTHONHASHSEED=0, seeds 990000+)

| Agent | Games | Win rate vs F | CPU/game | Over/errs |
|---|---|---|---|---|
| v4 (= sub1) | 1,000 | 0.5070 ±0.031 | 364 ms | 0 / 0 |
| orion v1 (inert layer) | 1,000 | 0.5060 ±0.031 | 428 ms | 0 / 0 |
| orion v1 (screen) | 200 | 0.5250 ±0.069 | 538 ms | 0 / 0 |
| orion v2 (live layer) | 200 | 0.4600 ±0.069 | 541 ms | 0 / 0 |
| v4 (screen slice) | 200 | 0.4950 ±0.069 | 482 ms | 0 / 0 |

Submission guidance: sub1 (v4) stands. sub2_v11.zip (0.5175/400) and
sub3_v16.zip (0.5075/400) are packaged as near-tie variance hedges — the
portal ranks the team by its best submission, so an extra upload of a
~0.51 agent is a free extra draw, nothing more. Orion is not worth a slot.
