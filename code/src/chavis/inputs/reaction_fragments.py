from __future__ import annotations
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
_SPECIES = Path(__file__).resolve().parents[3] / 'campaigns' / 'ni_drm' / 'config' / 'species.json'


@lru_cache(maxsize=None)
def _species_db(path=str(_SPECIES)):
    return json.loads(Path(path).read_text())


def composition_of_species(name, species=None):
    from rdkit import Chem
    species = species or _species_db()
    if name not in species:
        raise ValueError(f'product {name!r} is not in species.json')
    mol = Chem.AddHs(Chem.MolFromSmiles(species[name]['smiles']))
    c = Counter((a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() != 'Cl'))
    return tuple(sorted(c.items()))


def product_names(reaction):
    eq = reaction['equation']
    left, right = (s.strip() for s in eq.split('->'))
    side = left if reaction.get('requested_direction') == 'association' else right
    names = [p.strip().replace('*', '').replace('(g)', '').strip() for p in side.split('+')]
    return [n for n in names if n]


def expected_products(reaction, species=None):
    names = product_names(reaction)
    if len(names) != 2:
        raise ValueError(f"{reaction['id']}: expected two products, got {names}")
    exp = sorted((composition_of_species(n, species) for n in names))
    db = [reaction.get('moving_fragment'), reaction.get('retained_fragment')]
    if all(db):
        if sorted((composition_of_species(n, species) for n in db)) != exp:
            raise ValueError(f"{reaction['id']}: DB fragments {db} disagree with the equation products {names}")
    return exp


def composition(symbols, atoms):
    return tuple(sorted(Counter((symbols[i] for i in atoms)).items()))


def fragments_after_cut(edges, atoms, cut):
    atoms = list(atoms)
    parent = {a: a for a in atoms}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for a, b in edges:
        if {a, b} == set(cut) or a not in parent or b not in parent:
            continue
        parent[find(a)] = find(b)
    groups = {}
    for a in atoms:
        groups.setdefault(find(a), []).append(a)
    return list(groups.values())


def cut_gives_products(symbols, edges, atoms, cut, expected):
    frags = fragments_after_cut(edges, atoms, cut)
    return len(frags) == 2 and sorted((composition(symbols, f) for f in frags)) == expected
