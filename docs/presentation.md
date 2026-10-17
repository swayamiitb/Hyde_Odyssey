# Odyssey presentation

Six slides. Suggested delivery: six minutes, followed by four minutes of questions.

## Slide 1 Catan under a CPU budget

Suggested time: 40 seconds.

We are Team Odyssey. Our agent combines resource-aware search with a compact learned value model. The frozen release won 797 out of 1,000 games against F under the official local harness, with no recorded faults or clock overruns. This is a local result, not a leaderboard score. We will explain how we allocated compute, what learning added, and which experiments failed to justify more complexity. The aim was reliable decision quality within the game clock.

## Slide 2 Decision method

Suggested time: 80 seconds.

The allowance makes search allocation a design decision. We first value the two opening settlements together because resource coverage and ports shape the whole game. We use an approximate model of F to anticipate which sites may disappear. During ordinary play, we score the state after each legal action. The evaluation combines points with a concave production utility, useful hand resources, reachable expansion and progress toward road and army awards. A narrow beam then explores follow-up actions within the same turn. Its limits are six candidates, five action levels and 160 successor nodes, with at most 18 milliseconds for follow-up search. That limit excludes root scoring and opening placement. At nine points, a targeted trade planner checks short sequences that enable a win. Below 40 milliseconds of remaining allowance, the policy returns a legal priority move. Development-card purchases use the expected value over the publicly unseen multiset, so a lucky sampled deck does not drive buying.

## Slide 3 Learning a small value model

Suggested time: 65 seconds.

The training set contains 124,344 end-of-turn observations from 1,000 heuristic-versus-F games, viewed from both players. The labels are eventual wins. Ninety-one features describe permitted points, production, resources, ports, robber exposure, structures and build affordability. The opponent development-card identities are excluded. We split by whole game to prevent positions from the same game appearing on both sides. The classifier has 64 shallow gradient-boosted trees. Its validation AUC is 0.8848, but that is a prediction metric and is not a game win rate. We export the trees into plain Python and check the exported scores against the fitted implementation on 100 held-out vectors. The runtime agent embeds the weights, so it needs no model server or additional weight-loading step.

## Slide 4 Candidate selection and fresh validation

Suggested time: 70 seconds.

We compared the heuristic control and two blend strengths on the same 400 games. The 0.4 blend won 318 games, two more than the control. The selection rule chose the most wins among candidates with no faults or clock overruns. This is a small lead, not strong evidence for a learning gain. We then froze that source before running 1,000 fresh games. It won 797, with a board-pair clustered interval from 77.18 to 82.22 percent. Each board is played from both seats with the same dice stream. Fixed batches preserve game indices and agent lifetimes when worker counts change. Manifests include source and harness hashes. We retained all per-game records and verified complete counts. A separate heuristic holdout scored 79.6 percent on different boards, so its difference from 79.7 percent does not prove improvement.

## Slide 5 What the experiments taught us

Suggested time: 65 seconds.

The compact beam won 331 out of 400 games, versus 329 for the broad beam and 327 for the wide beam. Those win differences are small, but the broad beam consumed about 529 milliseconds per game compared with 337 for the compact beam. The observed cost did not buy a measured advantage. We also examined excessive development-card buying after Largest Army. Lowering spare-knight value sounded reasonable, but it won 326 games against a control with 334 on the same screen. Broader card reweighting won 323. We did not ship either change. The general lesson is to test plausible strategy changes against a frozen control and keep the gameplay outcome separate from a model metric or an appealing explanation. Failed experiments helped us narrow the policy without overstating certainty.

## Slide 6 Reliability and remaining questions

Suggested time: 40 seconds.

The behavior checks cover legal immutable openings, a two-trade winning city, development-card expectation, the free-road cache and a low-clock fallback. The runtime ZIP contains one file because the learned trees are embedded. Its source hash matches the measured release. The limitations are material: we have no O or group-opponent result for this snapshot, and local timing is not organizer-machine timing. Our next experiments would broaden the opponent distribution and improve action-ranking labels before adding more runtime search. The evidence supports reliable local play against F, while broader robustness remains an open question.

## Questions and answers

**Did learning improve the policy?** The 400-game common screen showed a two-win lead. This is weak evidence. The 79.6% and 79.7% holdouts use different boards and do not establish a significant gain.

**Why one file?** The standalone agent embeds ML_BIAS and ML_TREES. The JSON model in models/ is an inspection export, not a missing dependency.

**Is 0.8848 AUC an 88.48% win rate?** No. AUC evaluates outcome ranking on held-out training games. The gameplay result is 797/1000 against F.

**Can the agent see hidden development cards?** No real hidden types enter its features. Card-purchase evaluation averages over the publicly unseen multiset reconstructed from a permitted sampled world.

**How was leakage controlled?** Whole games form the training split. Gameplay evaluation uses new board seeds and a separate dice key after source freezing.

**Why no full MCTS?** This release prioritizes same-turn executable sequences under the measured CPU clock. It does not demonstrate that MCTS cannot work.

**What does zero overruns prove?** It describes this completed local run. It does not guarantee organizer-machine timing.

**What is your robustness score?** Unmeasured for this snapshot. F performance does not determine results against O or the weakest half of group matchups.
