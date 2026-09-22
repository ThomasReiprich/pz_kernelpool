"""kernelpool -- PZ data challenge entry (task sets 3 and 4): FlexZBoost + GPz pool with weights and calibration
fitted through the many-band label-noise kernel. See pipeline.py and pz_method_writeup.md."""
from .pipeline import VERSION, estimate_only, train_and_estimate  # noqa: F401

__version__ = VERSION
