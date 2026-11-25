# addon_smiles_inputs: plain SMILES -> CHAVIS input entries

This optional add-on turns the plain SMILES of a reaction into the entries that CHAVIS reads from
`campaigns/ni_drm/config/species.json` and `campaigns/ni_drm/config/reactions.json`.
It does two things: it decides the Cl markers and the mode of each species, and it finds the bond that breaks.
It does not import or change CHAVIS, and CHAVIS does not need it. It prints entries; it writes no CHAVIS file.

Paths: add-on files are relative to this folder (`addon_smiles_inputs/`). CHAVIS files are relative to the
release root, which is the folder above this one.

## Usage

Requires Python 3 and RDKit.

```
cd addon_smiles_inputs
python -m chavis_smiles_inputs --reactant "C" --products "[CH3]" "[H]"            # CH4* -> CH3* + H*
python -m chavis_smiles_inputs --reactant "[CH]" "[OH]" --products "[CH]O"        # CH* + OH* -> CHOH*
```

```python
from chavis_smiles_inputs import derive_reaction
out = derive_reaction("C", ["[CH3]", "[H]"])   # {"species": [...], "reaction": {...}, "derivation": {...}}
```

Write radicals with explicit H counts (`[CH3]`, `[OH]`, `[CH]O`), and closed-shell molecules in their
normal form (`C`, `O=C=O`, `[C-]#[O+]`). Any failure raises `ValueError`; no value is guessed.

## Rules (all general; no table keyed by species name, formula or reaction id)

| Output | Rule |
|---|---|
| mode, single atom | one atom in total -> `free:<element>` |
| mode, radical | radical electrons on exactly one atom -> `Cl`, `binder_element` = that atom (radicals on 2+ atoms -> error) |
| mode, closed shell | no radical electrons -> `free:<least electronegative non-H element>` (Pauling); only H -> `free:H` |
| Cl-marked SMILES | one Cl per radical electron on the binder atom (CH3 -> `ClC`, CHOH -> `ClC(Cl)O`) |
| broken_bond_elements | test every bond of the parent (explicit H): the cut must give two fragments that are graph-isomorphic (element, connectivity, H) to the fragments; all hits must give one element pair |
| pair order | `anchor-moving`; anchor = bond atom in the fragment that holds the parent's binder atom |
| direction | 1 -> 2 = dissociation; 2 -> 1 = association, parent = product, `computational_direction` = parent -> reactants |
| name | single atom -> element; only H -> `H<n>`; closed shell -> Hill formula; radical -> condensed formula from the binder atom (CHOH, COH, CHO) |

## Test

Run `bash run_tests.sh` (8 tests, CPU only, about 1 s; interpreter from `CHAVIS_PYTHON`, default `python3`).
The test reads `campaigns/ni_drm/config/` and `reproduce/inputs/cohorts/paper15.json` from the release root.
The only hand-written input is plain SMILES. It compares 9 fields for each of the 15 paper reactions
(parent_species, equation, broken_bond_elements, category, computational_direction, and name, mode,
binder_element, Cl SMILES for the parent and both fragments) and 4 fields for each of the 15 distinct species.
Result table: `results/paper15_match.md`.

- Reaction table: 135 of 135 cells match. Species table: 60 of 60 cells match.

## Limits

- state_class (the mark for reactants that may be weakly bound) and default_route are not derived. The existing
  code has defaults for both (`src/chavis/judgment/judge_v3.py` and `src/chavis/inputs/minima.py`: state_class
  `adsorbed_to_adsorbed`; `src/chavis/inputs/pairs.py`: route `unknown`).
- Closed-shell molecules get no Cl marker (`free:X` mode); their orientations come from the convex hull in
  `src/chavis/inputs/orientations.py`.
- Closed-shell molecules with C plus O or N (for example CH3OH, HCN) get `free:C` from the electronegativity rule.
  Species with radical electrons on two atoms (for example C2) raise an error.
- `src/chavis/inputs/fresh_one.py` accepts only the modes `Cl`, `free:C`, `free:O`, `free:H`; the add-on can give `free:N`.

### Per-reaction items that CHAVIS keeps outside the two JSON files

The add-on cannot supply these. The generic runner (`generic_v2`, judge_v3) does not read them, except the
`X02` branch; they are used on the older paper-15 path (`catalog.make_catalog`, `fresh_one.prepare_one`,
`canonical_judgment` V3, `new_endpoints`). Paths below are under `src/chavis/` unless stated.

| Item | What it does | Unknown (16th) id | General rule for the 15? |
|---|---|---|---|
| `judgment/policy.py` APPROVED_PRIMARY_REACTIONS | allow-list for `reaction_role_contract` | `ValueError` | "enabled in reactions.json" admits all 15 |
| `judgment/policy.py` PARENT_ROLE_OVERRIDES | allowed parent roles for D01, D05, D13 | default (surface, weak) | no: closed-shell parents D01, D05, D08, D13 get 3 different tuples |
| `judgment/policy.py` D05 detached channel | detached fallback channel for D05 only | False | no |
| `inputs/catalog.py` PRIMARY, NEW | the 15 ids; N01-N04 have no saved endpoints | AssertionError / KeyError | PRIMARY = cohort file; NEW is data availability |
| `inputs/catalog.py` PARENT_SITE | CHOH bridge, COH fcc | KeyError | yes: open valences 2 -> bridge, 3 -> hollow |
| `inputs/catalog.py` CL_COUNT | Cl count for CHOH 2, COH 3 | KeyError | yes: radical electrons on the binder (`marker_count`) |
| `inputs/catalog.py` D06/D10 flag, template id, product site text | per-id branches | silent defaults | template id yes (same broken bond, Cl parent); others no |
| `inputs/new_endpoints.py` PARENTS, PARENT_SITES | same for N01-N04 | KeyError | as PARENT_SITE |
| `judgment/graph.py` PRODUCTS, FORMULAS | product formulas per id, composition per species | `[]` -> target False; `ValueError` | yes: from the SMILES |
| `inputs/historical_sources.py`, `search/generalization.py` | ids with saved historical runs | not emitted | no (data availability) |
| `inputs/seed_geometry.py` X02 branch | swap anchor/moving for X02 (generic path, via `tools/run_wf4.py`) | no swap | X02 is not in the 15 |
| `ultra_fast_v1/launch.py`, `generic_v2/common.py` (release root) | 15-id lists | not consulted by the runner | = cohort file |

## Files

- `chavis_smiles_inputs/` the package (`core.py`; CLI in `__main__.py`)
- `tests/test_paper15.py` the comparison test; `run_tests.sh` runs it
- `results/paper15_match.md` the comparison table written by the test
