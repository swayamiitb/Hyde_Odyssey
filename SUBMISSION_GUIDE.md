# Submission guide

## Files to use

- Agent upload: [releases/Odyssey_submission.zip](releases/Odyssey_submission.zip).
- Report: [docs/Odyssey_Report.pdf](docs/Odyssey_Report.pdf).
- Slides: [docs/Odyssey_Presentation.pptx](docs/Odyssey_Presentation.pptx).
- Speaker script and questions: [PRESENTATION.md](PRESENTATION.md).

The full repository archive is supporting material. The runtime upload ZIP contains agent.py at its root. The learned parameters are embedded in that file.

## Rules checked

| Requirement | This package |
|---|---|
| Python entrypoint returns a legal action index | Agent.decide(obs, playable_actions) |
| Python source and permitted data, at most 100 MB | One small Python runtime file |
| No runtime network or child programs | No such calls in the release |
| Report uses A4, one column, at least 11 pt | Four A4 pages, 11 pt body and tables |
| Explain method, decisions, insights and experiments | Covered with measured evidence |
| Six-minute presentation, four minutes of questions | Six slides with speaker script and question bank |
| Claims come from actual runs | Raw records and manifests in evidence/ |

The official report limit excludes references. This report includes references within four pages. The six-slide limit comes from the team's requested deliverable.

## Validate before upload

~~~bash
python tools/verify_evidence.py
python -m unittest discover -s tests -v
python tools/package_submission.py
~~~

The behavior tests require the pinned kit and CATAN_KIT_ROOT, as described in the README.

Use the organizer's upload procedure from an authenticated team session. Publishing GitHub files does not upload to the scorer. The original schedule closes agent uploads at 17:00 IST and reports at 17:30 IST on 17 October 2026. This package does not assert acceptance or an extension.

Reference: [Official problem statement](https://github.com/lyric-hyde/catan-hackathon-agent/blob/main/PROBLEM_STATEMENT.md).
