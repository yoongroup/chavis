# Paper 15: derived from plain SMILES vs shipped JSON

Each cell: match, or `shipped / derived`. Species columns cover the parent and both fragments.

| reaction | parent_species | equation | broken_bond_elements | category | computational_direction | species.name | species.mode | species.binder_element | species.smiles |
|---|---|---|---|---|---|---|---|---|---|
| D01 | match | match | match | match | match | match | match | match | match |
| D02 | match | match | match | match | match | match | match | match | match |
| D03 | match | match | match | match | match | match | match | match | match |
| D04 | match | match | match | match | match | match | match | match | match |
| D05 | match | match | match | match | match | match | match | match | match |
| D06 | match | match | match | match | match | match | match | match | match |
| D07 | match | match | match | match | match | match | match | match | match |
| D08 | match | match | match | match | match | match | match | match | match |
| D09 | match | match | match | match | match | match | match | match | match |
| D10 | match | match | match | match | match | match | match | match | match |
| D13 | match | match | match | match | match | match | match | match | match |
| N01 | match | match | match | match | match | match | match | match | match |
| N02 | match | match | match | match | match | match | match | match | match |
| N03 | match | match | match | match | match | match | match | match | match |
| N04 | match | match | match | match | match | match | match | match | match |

## Distinct species

| shipped name | plain SMILES | name | mode | binder_element | smiles |
|---|---|---|---|---|---|
| C | `[C]` | match (`C`) | match (`free:C`) | match (`None`) | match (`[C]`) |
| CH | `[CH]` | match (`CH`) | match (`Cl`) | match (`C`) | match (`ClC(Cl)Cl`) |
| CH2 | `[CH2]` | match (`CH2`) | match (`Cl`) | match (`C`) | match (`ClCCl`) |
| CH3 | `[CH3]` | match (`CH3`) | match (`Cl`) | match (`C`) | match (`CCl`) |
| CH4 | `C` | match (`CH4`) | match (`free:C`) | match (`None`) | match (`C`) |
| CHO | `[CH]=O` | match (`CHO`) | match (`Cl`) | match (`C`) | match (`O=CCl`) |
| CHOH | `[CH]O` | match (`CHOH`) | match (`Cl`) | match (`C`) | match (`OC(Cl)Cl`) |
| CO | `[C-]#[O+]` | match (`CO`) | match (`free:C`) | match (`None`) | match (`[C-]#[O+]`) |
| CO2 | `O=C=O` | match (`CO2`) | match (`free:C`) | match (`None`) | match (`O=C=O`) |
| COH | `[C]O` | match (`COH`) | match (`Cl`) | match (`C`) | match (`OC(Cl)(Cl)Cl`) |
| H | `[H]` | match (`H`) | match (`free:H`) | match (`None`) | match (`[H]`) |
| H2 | `[H][H]` | match (`H2`) | match (`free:H`) | match (`None`) | match (`[H][H]`) |
| H2O | `O` | match (`H2O`) | match (`free:O`) | match (`None`) | match (`O`) |
| O | `[O]` | match (`O`) | match (`free:O`) | match (`None`) | match (`[O]`) |
| OH | `[OH]` | match (`OH`) | match (`Cl`) | match (`O`) | match (`OCl`) |

Reaction-table cells: 135/135 match (15 reactions x 9 fields).
Species-table cells: 60/60 match (15 species x 4 fields).
state_class and default_route are not derived (out of scope) and not compared.
