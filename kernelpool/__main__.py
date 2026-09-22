"""Command line: python -m kernelpool train --train-file F --test-file T --output O [--model-out M] [--workdir W] [--nmax N]
                python -m kernelpool estimate --model M --test-file T --output O"""
import argparse
import sys

from .pipeline import estimate_only, train_and_estimate


def main() -> int:
    ap = argparse.ArgumentParser(prog="kernelpool", description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train"); t.add_argument("--train-file", required=True); t.add_argument("--test-file", required=True)
    t.add_argument("--output", required=True); t.add_argument("--model-out", default=None); t.add_argument("--workdir", default=None)
    t.add_argument("--nmax", type=int, default=None); t.add_argument("--keep-workdir", action="store_true")
    e = sub.add_parser("estimate"); e.add_argument("--model", required=True); e.add_argument("--test-file", required=True); e.add_argument("--output", required=True)
    a = ap.parse_args()
    if a.mode == "train":
        info = train_and_estimate(a.train_file, a.test_file, a.output, model_file=a.model_out, workdir=a.workdir, nmax=a.nmax, keep_workdir=a.keep_workdir)
    else:
        info = estimate_only(a.model, a.test_file, a.output)
    print(info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
