"""Multi-agent supervision-checkpoint experiment package.

A single parameterized LangGraph multi-agent coder (Planner, Coder, Reviewer)
solving LiveCodeBench tasks under 4 supervision conditions. Only the plan/final
gates vary across conditions; everything else is identical.
"""

import os as _os
import sys as _sys

# `lcb_runner` is a namespace package (no __init__.py), so pip's editable finder is
# flaky. Put the vendored LiveCodeBench checkout on sys.path directly (path-relative,
# so it stays reproducible if the repo moves).
_LCB_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                         "LiveCodeBench")
if _os.path.isdir(_LCB_DIR) and _LCB_DIR not in _sys.path:
    _sys.path.insert(0, _LCB_DIR)
