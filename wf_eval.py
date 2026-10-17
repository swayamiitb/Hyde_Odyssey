"""cwd-safe evaluate driver for automation (chdirs to the repo root itself)."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from catan_kit._patch import ensure_hash_seed

ensure_hash_seed()

from catan_kit.evaluate import run


def _flag(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


if __name__ == "__main__":
    run(_flag("--a"), _flag("--b"), int(_flag("--games", "200")),
        int(_flag("--seed0", "990000")))
