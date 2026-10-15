"""The thing that grows.

Everything between the two markers is the only code the self-editor is allowed
to touch. It starts deliberately wrong (all coefficients zero) and has to *learn*
its way to the hidden target by tuning constants and swapping arithmetic
operators — never by adding imports, calls, or anything outside this block.

Run ``python -m ouroboros selfedit`` to watch it improve, and ``--apply`` to let
it (with your confirmation) rewrite the region below in place.
"""

from __future__ import annotations


# EVOLVE-REGION  (self-editor may rewrite only between these two markers)
def solve(x):
    return 0.0 * x * x + 0.0 * x + 0.0
# END-EVOLVE-REGION
