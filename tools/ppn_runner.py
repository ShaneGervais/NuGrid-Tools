#!/usr/bin/env python3
"""
tools/ppn_runner.py

Minimal launcher for NuPPN (single-zone ppn / multi-zone mppnp) runs:

- copy a run_template-style directory (e.g. nuppn/frames/ppn/run_template)
  into a fresh run directory
- optionally patch ppn_physics.input with reaction rate_index/rate_factor
  overrides
- execute the compiled binary in that directory, capturing a log
- optionally do the above for several run directories in parallel

This is intentionally minimal: no batch/sweep orchestration, no STARLIB
rate-set handling, no decay-time calibration. It's the plumbing later
rate-uncertainty/sensitivity work can build on, not that work itself.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def copy_run_template(template_dir, run_dir, extra_link_dirs=("NPDATA",)):
    """
    Copy a run_template directory into a fresh run directory.

    Any of `extra_link_dirs` that exist as siblings of `template_dir` and
    aren't already present under `run_dir` are symlinked in rather than
    copied -- these tend to be large, shared reference-data directories
    (e.g. NPDATA) that every run needs but none should duplicate.

    Raises FileExistsError if `run_dir` already exists.
    """
    template_dir = Path(template_dir)
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"run_dir already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template_dir, run_dir, symlinks=True)
    for name in extra_link_dirs:
        src = (template_dir / ".." / name).resolve()
        dest = run_dir / name
        if src.is_dir() and not dest.exists():
            dest.symlink_to(src, target_is_directory=True)
    return run_dir


def set_rate_factors(physics_input, indices, factors):
    """
    Append rate_index(n)/rate_factor(n) overrides to a ppn_physics.input
    namelist file, just before its closing '/'. Edits the file in place.

    Parameters
    ----------
    physics_input : path to the ppn_physics.input file to edit.
    indices : reaction indices, as they appear in networksetup.txt.
    factors : multiplicative rate factors, one per index.
    """
    if len(indices) != len(factors):
        raise ValueError("indices and factors must be the same length")
    physics_input = Path(physics_input)
    lines = physics_input.read_text().splitlines(keepends=True)
    out = []
    inserted = False
    for line in lines:
        if not inserted and line.strip() == "/":
            for n, (idx, factor) in enumerate(zip(indices, factors), start=1):
                out.append(f"        rate_index({n}) = {idx}\n")
                out.append(f"        rate_factor({n}) = {factor:.16E}\n")
            inserted = True
        out.append(line)
    if not inserted:
        raise ValueError(f"no namelist terminator ('/') found in {physics_input}")
    physics_input.write_text("".join(out))


def run_ppn(run_dir, exe_name="ppn.exe", log_dir=None):
    """
    Execute the compiled ppn/mppnp binary in `run_dir`, capturing
    stdout+stderr to a log file named after the run directory.

    Returns (logfile_path, elapsed_seconds). Raises FileNotFoundError if
    the binary is missing, RuntimeError on a nonzero exit code.
    """
    run_dir = Path(run_dir)
    exe = run_dir / exe_name
    if not exe.is_file():
        raise FileNotFoundError(f"missing executable: {exe}")
    log_dir = Path(log_dir) if log_dir else run_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{run_dir.name}.log"
    start = time.time()
    with open(logfile, "w", encoding="utf-8") as log:
        result = subprocess.run(
            [f"./{exe_name}"], cwd=run_dir, stdout=log, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"{exe} failed (exit {result.returncode}); see {logfile}")
    return logfile, elapsed


def run_many(run_dirs, exe_name="ppn.exe", log_dir=None, jobs=1):
    """
    Run several run directories, optionally in parallel (`jobs` workers).

    Returns a list of (run_dir, logfile, elapsed) for every run. Raises
    RuntimeError summarizing all failures if any run failed -- successful
    runs are still reflected in the exception message so partial progress
    isn't silently lost.
    """
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(run_ppn, d, exe_name, log_dir): d for d in run_dirs}
        for future in as_completed(futures):
            run_dir = futures[future]
            try:
                logfile, elapsed = future.result()
                results.append((run_dir, logfile, elapsed))
                print(f"finished {run_dir} in {elapsed:.0f}s")
            except Exception as exc:
                failures.append((run_dir, exc))
    if failures:
        summary = "\n".join(f"  {d}: {exc}" for d, exc in failures)
        raise RuntimeError(f"{len(failures)} run(s) failed:\n{summary}")
    return results


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    new_cmd = sub.add_parser("new", help="create a run directory from a run_template")
    new_cmd.add_argument("template_dir", help="path to a run_template directory")
    new_cmd.add_argument("run_dir", help="path of the new run directory to create")
    new_cmd.add_argument(
        "--rate", action="append", default=[], metavar="INDEX:FACTOR",
        help="reaction rate override, e.g. --rate 392:10.0 (repeatable)",
    )

    run_cmd = sub.add_parser("run", help="execute the binary in one or more run directories")
    run_cmd.add_argument("run_dirs", nargs="+", help="run directories to execute")
    run_cmd.add_argument("--exe", default="ppn.exe", help="executable name (default: ppn.exe)")
    run_cmd.add_argument("--jobs", type=int, default=1, help="parallel workers (default: 1)")

    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "new":
        run_dir = copy_run_template(args.template_dir, args.run_dir)
        if args.rate:
            indices, factors = [], []
            for spec in args.rate:
                index_str, factor_str = spec.split(":")
                indices.append(int(index_str))
                factors.append(float(factor_str))
            set_rate_factors(run_dir / "ppn_physics.input", indices, factors)
        print(f"created run directory: {run_dir}")
    elif args.command == "run":
        run_many(args.run_dirs, exe_name=args.exe, jobs=args.jobs)


if __name__ == "__main__":
    main()
