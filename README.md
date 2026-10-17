# Odyssey Catan Agent

Team Odyssey's agent for the [Hyde × WnCC Catan challenge](https://github.com/lyric-hyde/catan-hackathon-agent/blob/main/PROBLEM_STATEMENT.md).

The verified release combines joint opening placement, resource-aware evaluation, bounded same-turn search and a compact learned value model. It won **797 of 1,000 local games against F (79.7%)**, with no faults or clock overruns recorded for either player.

These are local results from the official match harness on Windows with Python 3.11.17. They do not establish an organizer score, a leaderboard rank, or performance against O. This repository pins the tested learned release. Later experiments are outside this snapshot.

## Submission and presentation

| Deliverable | File |
|---|---|
| Runtime submission ZIP | [Odyssey_submission.zip](releases/Odyssey_submission.zip) |
| Technical report, 4 A4 pages, single column, 11 pt body | [Odyssey_Report.pdf](docs/Odyssey_Report.pdf) |
| Editable presentation, 6 slides | [Odyssey_Presentation.pptx](docs/Odyssey_Presentation.pptx) |
| Six-minute script and technical questions | [Presentation notes](PRESENTATION.md) |
| Submission requirements and file guide | [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) |

## Verified results

| Experiment | Wins against F | Mean local CPU/game | Faults / over-clock |
|---|---:|---:|---:|
| Frozen heuristic holdout | 1,592 / 2,000 (79.6%) | 316.3 ms | 0 / 0 |
| Frozen learned holdout | **797 / 1,000 (79.7%)** | **393.4 ms** | **0 / 0** |

The learned holdout has a board-pair clustered 95% interval of **77.18%–82.22%**. Both seats share each board and dice stream. The two holdouts use different boards, so the 0.1 percentage-point difference is not evidence that learning improves play.

On the matched 400-game selection screen, the heuristic won 316 games, the 0.4 learned blend won 318, and the 0.9 blend won 317. The chosen blend's two-win lead is weak evidence. See the [experiment ledger](experiments.md), [selection record](evidence/selection.json) and [fresh holdout](evidence/learned_final_1000/summary.json).

## How the agent works

1. **Joint opening placement.** Score settlement pairs for resource coverage, ports, expansion and the opponent's likely choices.
2. **Resource-aware afterstates.** Evaluate points, economy, useful hand resources, expansion and award progress after each action.
3. **Bounded same-turn search.** Keep up to six candidates, search up to five action levels and cap expansion at 160 nodes. Follow-up search gets at most 18 ms, adjusted to the remaining clock.
4. **Tactical checks.** Prioritize winning builds and short bank-trade sequences. Use a legal fallback when the clock is low.
5. **Learned value adjustment.** Add 0.4 times a 64-tree model logit outside opening placement.

Development-card purchase scores average over the publicly unseen multiset. The policy does not depend on a favorable sampled deck order or on the opponent's real concealed development-card types.

## Where are the weights

The runtime [agent.py](submission/agent.py) contains both the decision code and the trained parameters, in **ML_BIAS** and **ML_TREES**. That is why the upload ZIP contains only one file.

The inspectable export is [models/value_model.json](models/value_model.json). The training set, feature extractor and trainer are in [training/](training/). This release uses gradient-boosted trees, so it does not require a PyTorch checkpoint at runtime. Older neural checkpoints under my_team/ belong to historical experiments.

Training used 1,000 offline games, 124,344 positions and 91 features. Whole-game validation produced AUC 0.8848 and log loss 0.4285. **AUC is a prediction metric, not a win rate.**

## Reproduce

Use Python 3.11 and the exact requirements from a separate clone of the [official kit](https://github.com/lyric-hyde/catan-hackathon-agent). Do not substitute the PyPI catanatron release.

Set the kit path in an activated Python 3.11 environment:

~~~powershell
$env:CATAN_KIT_ROOT = 'C:\absolute\path\to\catan-hackathon-agent'
$env:PYTHONPATH = $env:CATAN_KIT_ROOT
python -m pip install -r "$env:CATAN_KIT_ROOT\requirements.txt"
python -m catan_kit.selfcheck
python -m unittest discover -s tests -v
python tools/verify_evidence.py
~~~

On Linux or macOS, set the same variables with export:

~~~bash
export CATAN_KIT_ROOT=/absolute/path/to/catan-hackathon-agent
export PYTHONPATH="$CATAN_KIT_ROOT"
~~~

From this repository, repeat the final protocol:

~~~bash
python tools/benchmark.py --a submission/agent.py --games 1000 --workers 4 --seed0 990000 --key untouched-learned-final-20261017 --name reproduce_learned
~~~

The wrapper calls the unmodified official match function with clocks enabled. It freezes both sources, uses fixed 40-game batches, saves every game, and rejects incomplete results or a reused run name. Timings and deadline-limited search can vary across machines. This local dice key belongs to our experiment, not the organizer evaluation.

To generate new training data and fit a new model:

~~~bash
python training/train_value.py --games 1000 --workers 4
~~~

New training outputs stay under training/. They do not replace the verified model or release automatically. Re-evaluate changed agents before packaging.

## Repository map

| Location | Purpose |
|---|---|
| [submission/](submission/) | Frozen standalone runtime source |
| [releases/](releases/) | Runtime-only upload ZIP |
| [docs/](docs/) | Report, slides and editable text sources |
| [models/](models/) | Learned parameters and fit metrics |
| [training/](training/) | Dataset, features, heuristic teacher and trainer |
| [evidence/](evidence/) | Completed runs, source snapshots, manifests and raw games |
| [tools/](tools/) | Clocked benchmark, evidence verifier and ZIP builder |
| [tests/](tests/) | Five focused behavior regressions |
| [docs/history/](docs/history/) | Previous README, reports and ledger |
| [my_team/](my_team/) and older root scripts | Historical versions and research tooling |

The previous 52–53% descriptions remain in the history directory. They describe earlier versions and do not identify this release. Older one-off claims remain historical notes, not revalidated results for the current agent.

## Release identity and limitations

Agent SHA-256:

~~~text
a43d291a1cd984f67e59b7e24a4105c79c3110fa8ad164e45b3619e98927ba06
~~~

The [release record](evidence/release.json) identifies the measured source. The evidence verifier checks raw win counts, source hashes, embedded model parameters and ZIP contents.

O performance and group robustness are unmeasured. The F-aware opening and training distribution may specialize to one opponent. Local clocks do not guarantee organizer-machine timing. This repository contains no evidence of a 90% win rate.

Public components and development assistance are described in [NOTICE.md](NOTICE.md). Runtime play uses no language-model calls, networking or worker processes.
