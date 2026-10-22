"""kernelpool -- PZ data challenge entry (task sets 3 and 4 in v1.0.0; 1 and 2 added in v1.1.0; task-set-2 faint-end BPZ
mixture in v1.2.0): FlexZBoost + GPz pool with weights and calibration fitted through the many-band label-noise kernel
(exact labels on task sets 1/2). See pipeline.py and pz_method_writeup.md."""
from .pipeline import VERSION, estimate_only, train_and_estimate  # noqa: F401

__version__ = VERSION
