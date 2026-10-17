# Public components and provenance

- [Official hackathon kit](https://github.com/lyric-hyde/catan-hackathon-agent): observation API, evaluation harness, F wrapper and requirements. Opponent sources in evidence/ are preserved reference material, not team-authored algorithms.
- [Catanatron](https://github.com/bcollazo/catanatron), commit d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816: engine and board graph. Catanatron uses GPL-3.0. Upstream license terms continue to apply.
- Python standard library, NetworkX and psutil: runtime or evaluation support.
- scikit-learn, NumPy, SciPy and threadpoolctl: offline learning and data processing. Runtime tree inference is plain Python.

Language-model assistants helped write code and analyze results during development. Runtime play makes no language-model calls or network requests.

Earlier research includes other literature and neural-model experiments. Original references remain in docs/history/ and older research documents. The current report relies on the explicit evidence listed in experiments.md.

Original benchmark manifests retain source paths and environment details. Relocated tools use CATAN_KIT_ROOT to find an official-kit checkout.
