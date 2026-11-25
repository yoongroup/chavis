# Data

Structures and energies of the SMILES-started runs of the paper: three runs on Ni(111) and one run each on Ni(100),
Cu(111) and Rh(111). 85 of the 90 reactions have a validated transition state (TS).

| Path | Content |
|---|---|
| `barriers.csv` | one row per run and reaction: UMA and DFT single-point barriers and reaction energies, the energies of the three structures, the imaginary frequency and the folder |
| `structures/<run>/<reaction>/TS.extxyz` | validated TS |
| `structures/<run>/<reaction>/reactant_lowest.extxyz` | lowest sampled reactant configuration of the parent species (barrier reference) |
| `structures/<run>/<reaction>/downhill_reactant_side.extxyz` | end point of the downhill relaxation from the TS toward the reactant |
| `structures/<run>/<reaction>/downhill_product_side.extxyz` | end point of the downhill relaxation from the TS toward the product (the product of the DFT single point) |
| `structures/<run>/<reaction>/TS_frequencies_cm1.json` | finite-difference frequencies of all free atoms in cm-1; the negative value is the imaginary mode |
| `structures/<run>/<reaction>/TS_imaginary_mode.npy` | imaginary mode as Cartesian displacements (NumPy array, atoms x 3) |

Runs: `Ni111_run1` (Figures 3 and 4), `Ni111_run2` and `Ni111_run3` (Supporting Information), `Ni100`, `Cu111` and
`Rh111` (Figure 4). Reaction IDs are those of `code/campaigns/ni_drm/config/reactions.json`; `barriers.csv` gives the
equations of Table 1 of the paper.

- UMA energies: `energy_eV` in the header of each extxyz file (model `uma-s-1p2p1`, task `oc20`), in eV.
- Barrier = E(TS) - E(reactant_lowest); reaction energy = E(downhill_product_side) - E(reactant_lowest).
- DFT single points: spin-polarized PBE with Quantum ESPRESSO and SSSP precision v2.0 on the same structures, without
  relaxation (settings in the Supporting Information). Total energies in eV.
- `move_mask`: F marks a fixed slab atom.
- Empty cells: reaction without a validated TS (see the Supporting Information).
