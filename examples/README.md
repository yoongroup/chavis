# Example: one reaction on Ni(111)

`ni111_ch3_cohort.json` asks for one reaction of the paper set: D02, CH3* -> CH2* + H* on Ni(111).

## 1. Preflight (CPU only, no model)

The preflight checks the reaction database, the settings and the cohort.
It does not load the model and does not use the GPU.

```sh
export CHAVIS_PYTHON=python3            # the interpreter of your environment
export CHAVIS_RUN_ROOT="$HOME/chavis_runs"
(
  cd code || exit 1
  bash launch_general_v2.sh --cohort ../examples/ni111_ch3_cohort.json --surface Ni:fcc111 --preflight \
    | diff - ../examples/expected_preflight.json && echo "preflight OK"
)
```

`expected_preflight.json` is the output of this command for this repository.

## 2. Full run (Linux, NVIDIA GPU, UMA model)

The full run needs the UMA model `uma-s-1p2p1` (task `oc20`) in your Hugging Face hub cache
(see `code/environment/model.json` from the repository root). Use a new output folder below `CHAVIS_RUN_ROOT`.

```sh
(
  cd code || exit 1
  bash launch_general_v2.sh --cohort ../examples/ni111_ch3_cohort.json --surface Ni:fcc111 --seed 1 \
    --out "$CHAVIS_RUN_ROOT/example_ni111_ch3"
)
```

When the run ends, `state.json` in the output folder has `"status": "complete"`.
The result of the reaction is in
`reactions/D02/attempt_0001/Ni_fcc111/CH3/ts_ultra_D02/reaction_result.json`
(first and lowest validated transition state, barriers, status of each endpoint pair).
`code/README.md` describes all output files, stop and resume.
