# tools

Scripts for *launching* nuppn (ppn/mppnp) and MESA runs -- as opposed to
`scripts/`, which analyzes output that already exists.

Both scripts are deliberately minimal: create a run directory from a
template, optionally override a couple of parameters, execute the
binary, capture a log. No batch/sweep orchestration, no rate-set
handling, no run-configuration schema. They're the plumbing that
Phase 4 (rate uncertainty / sensitivity analysis, see the top-level
README) will build on, not that work itself.

## `ppn_runner.py`

Wraps a `run_template`-style directory such as
`nuppn/frames/ppn/run_template` or `nuppn/frames/mppnp/run_template`
(the compiled binary, e.g. `ppn.exe`, must already exist there --
`make` it first per the nuppn build instructions).

```bash
# create a run directory from a template
python tools/ppn_runner.py new /path/to/nuppn/frames/ppn/run_template runs/baseline

# ... with a reaction rate override (index:factor, from networksetup.txt)
python tools/ppn_runner.py new /path/to/run_template runs/23Na_pg_24Mg_x10 --rate 392:10.0

# execute one or more run directories, optionally in parallel
python tools/ppn_runner.py run runs/baseline
python tools/ppn_runner.py run runs/*_x10 runs/*_x0.1 --jobs 4
```

Importable API (used programmatically, e.g. from a notebook or a
future batch script): `copy_run_template`, `set_rate_factors`,
`run_ppn`, `run_many`.

## `mesa_runner.py`

Same idea for a MESA work-directory template (the compiled `star`
binary must already exist there).

```bash
python tools/mesa_runner.py new /path/to/mesa/work runs/2Msun --param initial_mass=2.0
python tools/mesa_runner.py run runs/2Msun
```

Importable API: `copy_work_template`, `set_inlist_params`, `run_mesa`.
