"""Ouroboros: a small GPT built from scratch, and a lineage of them that designs and trains its own successors."""

import warnings

# PyTorch notes (harmlessly) when NumPy is absent. Becky does not need NumPy, so don't alarm anyone.
# This must run before torch is imported, which is why it lives in the package's first lines.
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

__version__ = "0.1.0"
