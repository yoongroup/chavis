# CHAVIS

CHAVIS searches for transition states (TS) of elementary reactions on metal surfaces with the UMA machine-learned
potential. For each reaction it builds adsorbed initial and final states, pairs them, makes TS guesses from
bond-length scans (and CI-NEB when the scans fail), refines them with the dimer method, and validates each TS by its
imaginary mode and its connection to both end states. The paper applies it to 15 dissociation steps of methane dry
reforming on Ni(111), Ni(100), Cu(111) and Rh(111).

| Folder | Content |
|---|---|
| `code/` | the program, reaction and species databases, slab caches and SMILES add-on (`code/README.md` is the full manual) |
| `examples/` | one reaction on Ni(111): cohort file, commands, expected preflight output |
| `code/environment/` | pinned package versions and the UMA model identity |
| `data/` | structures and energies of the validated transition states of the SMILES-started runs (`data/README.md`) |

## Requirements and install

- Python 3.12 (tested with 3.12.3).
- Calculations: Linux and one NVIDIA GPU. The CPU checks below need no GPU and no model.
- Packages: `code/environment/requirements-pinned.txt` (ase, e3nn, fairchem-core, numpy, pymatgen, rdkit, scipy, torch),
  plus matplotlib, spglib and networkx.
- UMA model `uma-s-1p2p1`, task `oc20`, revision `f611b917d9c68566bbbeccbb0aa0f7cad1696cb2` (file hashes in
  `code/environment/model.json`). Get it from the model provider and put it in your Hugging Face hub cache. The model is not
  part of this repository.

```sh
python3 -m pip install -r code/environment/requirements-pinned.txt
python3 -m pip install matplotlib spglib networkx
```

## Quick check (CPU only)

```sh
(
  cd code || exit 1
  export CUDA_VISIBLE_DEVICES=""
  bash launch_general_v2.sh --cohort ../examples/ni111_ch3_cohort.json --surface Ni:fcc111 --preflight \
    | diff - ../examples/expected_preflight.json || exit 1
  bash addon_smiles_inputs/run_tests.sh                   # 8 tests of the SMILES add-on
)
```

The preflight must match the stored JSON; the add-on reports `Ran 8 tests ... OK`.

## Example

`examples/README.md` runs one reaction (CH3* -> CH2* + H* on Ni(111)): first a CPU preflight that must print
`examples/expected_preflight.json`, then the full GPU run.

## Runs behind the paper

The commands that produced the runs behind the figures of the paper are in `code/README.md` (Campaign runs).

## Data availability

`data/` holds the structures and energies behind Figures 3 and 4 of the paper and the two other SMILES-started runs on
Ni(111): every validated transition state with its lowest reactant, the two downhill end points, the frequencies and the
UMA and DFT single-point energies (`data/README.md`). The other run records (trajectories, logs and the DFT inputs and
outputs) are available from the corresponding author upon reasonable request.

## How to cite

Citation to be added after publication.

## License

MIT License (`LICENSE`). Third-party software and the model weights keep their own terms.
