#!/usr/bin/env python3
"""
tools/mesa_runner.py

Minimal launcher for MESA runs: copy a work-directory template into a
fresh run directory, optionally override specific `inlist` parameters,
and execute the MESA binary (`./star` by default), capturing a log.

Deliberately minimal, mirroring tools/ppn_runner.py's scope: no inlist
schema/validation, no batch orchestration -- just enough to script
"copy a template, tweak a couple of parameters, launch it".
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path


def copy_work_template(template_dir, run_dir):
    """
    Copy a MESA work-directory template into a fresh run directory.

    Raises FileExistsError if `run_dir` already exists.
    """
    template_dir = Path(template_dir)
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"run_dir already exists: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template_dir, run_dir, symlinks=True)
    return run_dir


def set_inlist_params(inlist, params):
    """
    Set `name = value` for each (name, value) pair in `params` inside an
    inlist file, in place. A parameter already present anywhere in the
    file (in any namelist section, possibly commented out with a single
    leading `!`) is overwritten there. Only parameters not found
    anywhere are appended, just before the first `/` that closes a
    namelist section -- so if a new parameter needs to land in a
    specific section (e.g. `&controls` rather than `&star_job`), add it
    to the template with a placeholder value first rather than relying
    on this fallback.

    Parameters
    ----------
    inlist : path to the inlist file to edit.
    params : dict of {parameter_name: value}. `value` is written via
        `repr()` for strings and `str()` otherwise, so pass Python
        values (e.g. True, 1.5, 'some_string') rather than pre-formatted
        Fortran literals.
    """
    inlist = Path(inlist)
    lines = inlist.read_text().splitlines(keepends=True)

    def literal_for(value):
        return repr(value) if isinstance(value, str) else str(value)

    # Pass 1: overwrite every existing occurrence, wherever it is in the file.
    remaining = dict(params)
    out = []
    for line in lines:
        stripped = line.strip().lstrip("!").strip()
        matched_name = next(
            (name for name in remaining
             if re.match(rf"^{re.escape(name)}\s*=", stripped)),
            None,
        )
        if matched_name is not None:
            out.append(f"    {matched_name} = {literal_for(remaining.pop(matched_name))}\n")
        else:
            out.append(line)

    # Pass 2: anything not found anywhere gets appended before the first '/'.
    if remaining:
        inserted = False
        final = []
        for line in out:
            if not inserted and line.strip() == "/":
                for name, value in remaining.items():
                    final.append(f"    {name} = {literal_for(value)}\n")
                inserted = True
            final.append(line)
        if not inserted:
            raise ValueError(
                f"could not place new parameter(s) {list(remaining)} in {inlist} "
                "(no namelist terminator '/' found)"
            )
        out = final

    inlist.write_text("".join(out))


def run_mesa(run_dir, exe_name="star", log_dir=None):
    """
    Execute the MESA binary in `run_dir`, capturing stdout+stderr to a
    log file named after the run directory.

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


def _parse_param(spec):
    name, raw_value = spec.split("=", 1)
    name = name.strip()
    raw_value = raw_value.strip()
    try:
        value = int(raw_value)
    except ValueError:
        try:
            value = float(raw_value)
        except ValueError:
            if raw_value.lower() in ("true", "false"):
                value = raw_value.lower() == "true"
            else:
                value = raw_value
    return name, value


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    new_cmd = sub.add_parser("new", help="create a run directory from a work-directory template")
    new_cmd.add_argument("template_dir", help="path to a MESA work-directory template")
    new_cmd.add_argument("run_dir", help="path of the new run directory to create")
    new_cmd.add_argument(
        "--inlist", default="inlist", help="inlist file to patch (default: inlist)"
    )
    new_cmd.add_argument(
        "--param", action="append", default=[], metavar="NAME=VALUE",
        help="inlist parameter override, e.g. --param initial_mass=2.0 (repeatable)",
    )

    run_cmd = sub.add_parser("run", help="execute the binary in a run directory")
    run_cmd.add_argument("run_dir", help="run directory to execute")
    run_cmd.add_argument("--exe", default="star", help="executable name (default: star)")

    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "new":
        run_dir = copy_work_template(args.template_dir, args.run_dir)
        if args.param:
            params = dict(_parse_param(spec) for spec in args.param)
            set_inlist_params(run_dir / args.inlist, params)
        print(f"created run directory: {run_dir}")
    elif args.command == "run":
        run_mesa(args.run_dir, exe_name=args.exe)


if __name__ == "__main__":
    main()
