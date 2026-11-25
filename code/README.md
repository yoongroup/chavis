# CHAVIS

CHAVIS searches for transition states (TS) of elementary reactions on metal surfaces with the UMA machine-learning
potential. For each reaction it builds adsorbed initial and final states, pairs them, generates TS seeds
(bond-length scans and CI-NEB), refines the seeds with the dimer method, and validates each TS by its imaginary
mode, its connection to the two endpoints and duplicate checks. A round-robin scheduler stops a reaction at its
first validated TS (`first_success`) or at the first validated TS of every representative endpoint pair
(`all_pairs`). Cost ledgers record measured model calls and identify incomplete accounting; see Accounting scope below.

This package contains the Ni dry-reforming reaction and species databases, the campaign runner, the calculation
drivers and the SMILES add-on with its CPU tests.

## Requirements

- Linux (the campaign launchers refuse other systems; Windows works for the CPU tests only) and an NVIDIA GPU.
- Python 3.12.3 (`environment/python-version.txt`).
- Packages (`environment/requirements-pinned.txt`): ase 3.29.0, e3nn 0.6.0, fairchem-core 2.22.0, numpy 2.4.6,
  pymatgen 2026.9.24, rdkit 2026.03.6, scipy 1.18.1, torch 2.13.0+cu130. Also install matplotlib (figures),
  spglib (surface symmetry) and networkx (atom correspondence). `environment/status.json` lists what these pins
  do not fix (no complete lock file).
- UMA model `uma-s-1p2p1`, task `oc20`, revision `f611b917d9c68566bbbeccbb0aa0f7cad1696cb2`, with the asset
  hashes in `environment/model.json`. Get the assets from the model provider and put them in your Hugging Face
  hub cache (`models--facebook--UMA/snapshots/<revision>/...` below that cache). Runs are offline.

```sh
python3 -m pip install -r environment/requirements-pinned.txt
python3 -m pip install matplotlib spglib networkx
```

## Portable configuration

The campaign runner, TS CLI and launchers share `RUN_ROOT` from `generic_v2/common.py`. Set `CHAVIS_RUN_ROOT`
to select a run root; its default is `~/chavis_runs`. Campaign output must be a physical descendant of this root,
with a fresh output folder for each new run. Symlink paths and paths outside the configured root are rejected.
Use the same output folder only when resuming its matching request.

Python entrypoints use the running interpreter (`sys.executable`), with optional `CHAVIS_PYTHON` selection.
Shell launchers use `CHAVIS_PYTHON` when set, otherwise `python3` from `PATH`; activate the intended environment
before running them. The hub cache resolves from `HF_HUB_CACHE`, then `HF_HOME/hub`, then the Hugging Face
default (`$XDG_CACHE_HOME/huggingface/hub`, or `~/.cache/huggingface/hub` when `XDG_CACHE_HOME` is unset).
Preflight and workers use the same model cache and pinned revision.

In the shell examples, set the package location and use the selected interpreter and run root:

```sh
PKG=/absolute/path/to/package
PYTHON="${CHAVIS_PYTHON:-python3}"
RUN_ROOT="${CHAVIS_RUN_ROOT:-$HOME/chavis_runs}"
cd "$PKG"
```

Resume requires the original code fingerprint, calculated from the runtime files when the request is made;
do not migrate an unfinished run to a changed package. `slab_store/` is part of the distributed package and
its files must remain available.

## CPU checks (no GPU, no model)

The subshell limits the hidden-device setting to these checks; it preserves the parent shell's device selection.

```sh
(
  cd "$PKG" || exit 1
  export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
  bash launch_general_v2.sh --cohort ../examples/ni111_ch3_cohort.json --surface Ni:fcc111 --preflight \
    | diff - ../examples/expected_preflight.json || exit 1
  CHAVIS_PYTHON="$PYTHON" bash addon_smiles_inputs/run_tests.sh || exit 1
)
```

The add-on runs eight tests. A check of the paper campaign setup without a GPU:

```sh
bash "$PKG/launch_general_v2.sh" --cohort paper15 --surface Ni:fcc111 --preflight
```

## Campaign runs

`launch_general_v2.sh` runs `generic_v2.runner`: one sequential campaign, one reaction at a time, one UMA worker,
with a GPU temperature guard (pause at 85 °C, resume at 75 °C). All commands below use the 15-reaction paper set
(`--cohort paper15`), the default settings `reproduce/inputs/ultra_fast/settings.json` (azimuth `auto`, judge v3,
first-success threshold 0.01) and replicate label `--seed k` (k = 1, 2, 3; one new output folder per replicate).
These labels identify execution replicates; the RDKit embedding seed stays fixed at 61453.
Run the commands one after another; the runner holds a lock.

A graceful stop can return exit status zero. Define this helper before the batches: it requires both a successful
launcher exit and `state.json["status"] == "complete"`. Missing or malformed state also stops the batch.

```sh
run_complete() {
  local out="$1"
  shift
  bash "$PKG/launch_general_v2.sh" "$@" --out "$out" || return
  "$PYTHON" -B -c 'import json, sys; state = json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if state.get("status") == "complete" else "Campaign is not complete; resume it before continuing the batch.")' "$out/state.json"
}
```

Figure 2, same-pool comparison on Ni(111). The relaxed reactant records of the baseline run replace the CHAVIS
adsorption search; set `BASELINE_ROOT` to the folder holding `seed1..3/reactions/<rid>/checkpoint.json`:

```sh
BASELINE_ROOT=/absolute/path/to/baseline_records
(
  for k in 1 2 3; do
    run_complete "$RUN_ROOT/same_pool/r$k" --cohort paper15 --surface Ni:fcc111 --seed "$k" \
      --reactant-records "$BASELINE_ROOT" || exit 1
  done
)
```

Figures 3 and 4, native search on the four surfaces, and the all-pair search on Ni(111). The all-pair command
is in the same batch and runs only after every native campaign has completed:

```sh
(
  for s in Ni:fcc111 Ni:fcc100 Cu:fcc111 Rh:fcc111; do
    run_complete "$RUN_ROOT/native/${s/:/_}" --cohort paper15 --surface "$s" --seed 1 || exit 1
  done
  run_complete "$RUN_ROOT/all_pairs/Ni_fcc111" --cohort paper15 --surface Ni:fcc111 --seed 1 \
    --settings "$PKG/reproduce/inputs/ultra_fast/settings_all_pairs.json" || exit 1
)
```

Surfaces other than Ni(111) accept only the default azimuth setting and no `--reactant-records`. The lattice and
slab caches for the four surfaces are in `slab_store/`. Ni(111) requires its cache; for Ni(100), Cu(111) and
Rh(111) a missing cache is computed (EOS and slab relaxation) at the first run.

Stop and resume: create `<OUT>/STOP` to stop after the current reaction. Remove it and repeat that campaign's
same command with `--resume`. A resume needs the same code, settings, cohort, surface and inputs; an unfinished
reaction restarts in a new attempt, and the costs of the abandoned attempt stay in the totals. Resume a stopped
campaign to completion before continuing the batch at its next unstarted entry; do not rerun already completed
entries as fresh runs.

## Outputs

Each campaign writes only below its `<OUT>` folder:

- `request.json`, `cohort.json`, `settings.json`, `params.json`: the frozen request; `state.json`: campaign state
  (`complete` at the end); `ledger.jsonl`, `cost_ledger.json`, `launcher.jsonl`, `campaign_end.json`: event and
  cost records. Worker sessions keep a run snapshot `_snapshot/` (code copy, file hashes, environment).
- `reactions/<RID>/attempt_NNNN/`: one folder per reaction attempt.
- `reactions/<RID>/attempt_NNNN/<SURFACE>/<species>/ts_ultra_<RID>/`: `reaction_result.json` (first and lowest
  validated TS, barriers, per-pair status) and `first_success_checkpoint.json` (model calls up to the first
  validated TS).
- `reactions/<RID>/attempt_NNNN/first_success_accounting.json`: per-reaction first-success cost
  (`per_reaction_first_success_calls`) and the cumulative campaign cost.

### Accounting scope

First-success checkpoints describe the current attempt's first validated TS. They do not necessarily identify
the earliest success across retries. `per_reaction_first_success_calls` includes the full shared preparation,
the reaction's imported preparation charge, its pre-TS stages and its TS checkpoint. Summing these per-reaction
values would count shared preparation repeatedly; use `cost_ledger.json` for aggregate campaign totals.

`cumulative_campaign_calls_at_attempt_first_success` additionally includes earlier reactions, surface
preparation, imported charges and abandoned attempts. `campaign_first_success_calls` is a compatibility alias
for that cumulative field, not a paired per-reaction cost. Imported preparation is charged once per
`(seed, reaction, source checkpoint)` across retries in the campaign ledger.

An interruption can leave missing or unreadable cost records. Known calls remain recorded; unknown costs are
not silently assigned zero. Inspect `unknown_cost_stages`, `unknown_time_stages`, `calls_complete` and
`times_complete` in attempt summaries, `per_reaction_calls_complete` and `campaign_calls_complete` in checkpoint
accounting, and `unknown_cost_attempts` and `costs_complete` in the campaign ledger. `costs_complete` also requires
complete timing and no active attempt. A `first_success_accounting_error.json` with
`first_success_calls_complete: false` marks a checkpoint that could not be accounted for. Treat incomplete
totals as the known portion of the cost.

## Databases

Scientific databases are in `config/` and `campaigns/ni_drm/config/`; run settings and cohorts are in
`reproduce/inputs/`. Use the campaign commands above for first-success, all-pair and other-surface searches.

## License

MIT License (`../LICENSE`). Third-party software, model weights and datasets keep their own terms.
