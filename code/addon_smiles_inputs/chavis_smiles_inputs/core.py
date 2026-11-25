"""Derive CHAVIS species.json / reactions.json entries from plain SMILES.

Only general rules are used (element, valence, radical electrons, closed shell, graph connectivity).
There is no table keyed by species name, formula or reaction id. Anything that cannot be derived raises ValueError.
This module does not import the CHAVIS package.
"""
from __future__ import annotations

import hashlib
from collections import Counter

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

MARKER = "Cl"  # element the existing code reads as an open-valence marker on the binding atom

# Pauling electronegativity (element property, used only to pick the free-mode atom of a closed-shell molecule)
PAULING = {"H": 2.20, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
           "Si": 1.90, "P": 2.19, "S": 2.58, "Br": 2.96, "I": 2.66}

RULES = {
    "mode_single_atom": "one atom in total -> free:<element>",
    "mode_radical": "radical electrons on exactly one atom -> mode Cl, binder = that atom",
    "mode_closed_shell": "no radical electrons -> free:<least electronegative non-H element> (H if only H)",
    "cl_markers": "one Cl per radical electron on the binder atom; radical electrons set to 0",
    "broken_bond": "every bond of the parent (explicit H) whose cut gives two fragments graph-isomorphic "
                   "(element + connectivity incl. H) to the two fragments; all hits must give one ordered element pair",
    "bond_order": "anchor-moving; anchor = bond atom in the fragment that contains the parent's binder atom",
    "name": "single atom -> element; only H -> H<n>; closed shell -> Hill formula; "
            "radical -> condensed formula written from the binder atom (Hill formula if the species has a ring)",
    "direction": "1 reactant + 2 products -> dissociation; 2 reactants + 1 product -> association, "
                 "parent = product, computational_direction = parent -> reactants",
}


# ---------------------------------------------------------------- molecules and graphs

def plain_mol(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"empty SMILES: {smiles!r}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")
    if any(a.GetSymbol() == MARKER for a in mol.GetAtoms()):
        raise ValueError(f"plain SMILES must not contain {MARKER} (reserved as marker): {smiles!r}")
    return mol


def graph(mol):
    """(labels, adjacency) of the molecule with every H as its own node; bond orders and charges are ignored."""
    m = Chem.AddHs(mol)
    labels = [a.GetSymbol() for a in m.GetAtoms()]
    adj = [set() for _ in labels]
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].add(j)
        adj[j].add(i)
    return labels, adj


def _colors(labels, adj):
    col = list(labels)
    for _ in range(len(labels)):
        new = [hashlib.sha1(repr((col[i], sorted(col[j] for j in adj[i]))).encode()).hexdigest()
               for i in range(len(labels))]
        if len(set(new)) == len(set(col)):
            col = new
            break
        col = new
    return col


def isomorphic(g1, g2) -> bool:
    (l1, a1), (l2, a2) = g1, g2
    if len(l1) != len(l2) or sum(map(len, a1)) != sum(map(len, a2)) or Counter(l1) != Counter(l2):
        return False
    # joint colour refinement: refine the disjoint union so colours are comparable between the two graphs
    n = len(l1)
    labels = list(l1) + list(l2)
    adj = [set(s) for s in a1] + [{j + n for j in s} for s in a2]
    col = _colors(labels, adj)
    c1, c2 = col[:n], col[n:]
    if Counter(c1) != Counter(c2):
        return False
    order = sorted(range(n), key=lambda i: (sum(c == c1[i] for c in c1), i))
    mapping, used = {}, set()

    def extend(k):
        if k == n:
            return True
        i = order[k]
        for j in range(n):
            if j in used or c2[j] != c1[i]:
                continue
            if all((mapping[x] in a2[j]) == (x in a1[i]) for x in mapping):
                mapping[i] = j
                used.add(j)
                if extend(k + 1):
                    return True
                del mapping[i]
                used.discard(j)
        return False

    return extend(0)


def _components(n, adj, skip):
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        comp, stack = {s}, [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if {u, v} == skip or v in comp:
                    continue
                comp.add(v)
                stack.append(v)
        seen |= comp
        comps.append(sorted(comp))
    return comps


def _subgraph(labels, adj, nodes, skip):
    idx = {old: k for k, old in enumerate(nodes)}
    sub_adj = [set() for _ in nodes]
    for old in nodes:
        for v in adj[old]:
            if v in idx and {old, v} != skip:
                sub_adj[idx[old]].add(idx[v])
    return [labels[o] for o in nodes], sub_adj


# ---------------------------------------------------------------- species

def _hill(counts: Counter) -> str:
    def part(el):
        return el + (str(counts[el]) if counts[el] > 1 else "")
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else []) + sorted(e for e in counts if e not in ("C", "H"))
    else:
        order = sorted(counts)
    return "".join(part(e) for e in order)


def _condensed(mol, root: int) -> str:
    """Condensed formula written from the binder atom: each heavy atom as El + H count, branches in parentheses,
    identical terminal siblings merged (O, O -> O2)."""
    def token(a):
        h = a.GetTotalNumHs()
        return a.GetSymbol() + ("H" if h == 1 else f"H{h}" if h > 1 else "")

    def walk(i, parent):
        a = mol.GetAtomWithIdx(i)
        kids = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != parent and n.GetSymbol() != "H"]
        texts = [walk(k, i) for k in kids]
        is_leaf = [sum(n.GetSymbol() != "H" for n in mol.GetAtomWithIdx(k).GetNeighbors()) == 1 for k in kids]
        leaf = [t for t, f in zip(texts, is_leaf) if f]
        branch = [t for t, f in zip(texts, is_leaf) if not f]
        parts = []
        for t, c in Counter(leaf).items():
            bare = t.isalpha() and sum(ch.isupper() for ch in t) == 1
            parts.append(t if c == 1 else (f"{t}{c}" if bare else f"({t}){c}"))
        parts += branch
        if len(parts) > 1:
            parts = [p if p.startswith("(") or k == len(parts) - 1 else f"({p})" for k, p in enumerate(parts)]
        return token(a) + "".join(parts)

    return walk(root, -1)


def describe_species(smiles: str) -> dict:
    """Species entry from a plain SMILES: name, Cl-marked smiles, mode, binder_element."""
    mol = plain_mol(smiles)
    full = Chem.AddHs(mol)
    counts = Counter(a.GetSymbol() for a in full.GetAtoms())
    radical_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    entry = {"plain_smiles": smiles, "canonical_plain_smiles": Chem.MolToSmiles(mol)}
    if full.GetNumAtoms() == 1:
        el = full.GetAtomWithIdx(0).GetSymbol()
        entry.update(name=el, smiles=Chem.MolToSmiles(mol), mode=f"free:{el}", binder_element=None,
                     placeholder=False, rule="mode_single_atom")
        entry["_binder_symbols"] = {el}
        return entry
    if not radical_atoms:
        heavy = sorted({e for e in counts if e != "H"})
        missing = [e for e in heavy if e not in PAULING]
        if missing:
            raise ValueError(f"no electronegativity for {missing} in {smiles!r}")
        if heavy:
            el = min(heavy, key=lambda e: PAULING[e])
            if sum(PAULING[e] == PAULING[el] for e in heavy) != 1:
                raise ValueError(f"electronegativity tie for free-mode atom in {smiles!r}")
        else:
            el = "H"
        name = _hill(counts) if heavy else f"H{counts['H']}"
        entry.update(name=name, smiles=Chem.MolToSmiles(mol), mode=f"free:{el}", binder_element=None,
                     placeholder=False, rule="mode_closed_shell")
        entry["_binder_symbols"] = {el}
        return entry
    if len(radical_atoms) != 1:
        raise ValueError(f"radical electrons on {len(radical_atoms)} atoms in {smiles!r}; "
                         "the Cl convention needs one binder atom")
    b = radical_atoms[0]
    n_rad = mol.GetAtomWithIdx(b).GetNumRadicalElectrons()
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(b).SetNumRadicalElectrons(0)
    for _ in range(n_rad):
        k = rw.AddAtom(Chem.Atom(MARKER))
        rw.AddBond(b, k, Chem.BondType.SINGLE)
    marked = rw.GetMol()
    Chem.SanitizeMol(marked)
    if sum(a.GetNumRadicalElectrons() for a in marked.GetAtoms()) != 0:
        raise ValueError(f"radical electrons left after marking {smiles!r}")
    marked_smiles = Chem.MolToSmiles(marked)
    # check: removing the markers gives back the plain composition
    back = Counter(a.GetSymbol() for a in Chem.AddHs(Chem.MolFromSmiles(marked_smiles)).GetAtoms())
    if back.pop(MARKER, 0) != n_rad or back != counts:
        raise AssertionError(f"marker round trip failed for {smiles!r}")
    el = mol.GetAtomWithIdx(b).GetSymbol()
    if el == "H":
        raise ValueError(f"binder atom is H in a multi-atom species {smiles!r}")
    name = _hill(counts) if mol.GetRingInfo().NumRings() else _condensed(mol, b)
    entry.update(name=name, smiles=marked_smiles, mode="Cl", binder_element=el,
                 placeholder=True, rule="mode_radical", marker_count=n_rad)
    entry["_binder_atom"] = b
    return entry


# ---------------------------------------------------------------- reactions

def _binder_nodes(sp, labels):
    if "_binder_atom" in sp:
        return {sp["_binder_atom"]}  # AddHs keeps heavy-atom indices
    return {i for i, s in enumerate(labels) if s in sp["_binder_symbols"]}


def _public(sp):
    return {k: v for k, v in sp.items() if not k.startswith("_")}


def derive_reaction(reactant_smiles, product_smiles_list) -> dict:
    reactants = [reactant_smiles] if isinstance(reactant_smiles, str) else list(reactant_smiles)
    products = [product_smiles_list] if isinstance(product_smiles_list, str) else list(product_smiles_list)
    if len(reactants) == 1 and len(products) == 2:
        direction, parent_smi, frag_smis = "dissociation", reactants[0], products
    elif len(reactants) == 2 and len(products) == 1:
        direction, parent_smi, frag_smis = "association", products[0], reactants
    else:
        raise ValueError(f"need 1 -> 2 (dissociation) or 2 -> 1 (association); got {len(reactants)} -> {len(products)}")
    parent = describe_species(parent_smi)
    frags = [describe_species(s) for s in frag_smis]
    pg = graph(plain_mol(parent_smi))
    fgs = [graph(plain_mol(s)) for s in frag_smis]
    if Counter(pg[0]) != Counter(fgs[0][0]) + Counter(fgs[1][0]):
        raise ValueError(f"not atom balanced: {parent_smi} vs {frag_smis}")
    labels, adj = pg
    binders = _binder_nodes(parent, labels)
    edges = sorted({tuple(sorted((i, j))) for i in range(len(labels)) for j in adj[i]})
    hits, pairs = [], set()
    for i, j in edges:
        skip = {i, j}
        comps = _components(len(labels), adj, skip)
        if len(comps) != 2:
            continue
        subs = [_subgraph(labels, adj, c, skip) for c in comps]
        if not ((isomorphic(subs[0], fgs[0]) and isomorphic(subs[1], fgs[1]))
                or (isomorphic(subs[0], fgs[1]) and isomorphic(subs[1], fgs[0]))):
            continue
        side_i = next(c for c in comps if i in c)
        in_i, in_j = bool(binders & set(side_i)), bool(binders - set(side_i))
        if in_i and not in_j:
            anchor, moving = i, j
        elif in_j and not in_i:
            anchor, moving = j, i
        elif labels[i] == labels[j]:
            anchor, moving = i, j
        else:
            raise ValueError(f"binder atoms on both sides of the {labels[i]}-{labels[j]} cut in {parent_smi}; "
                             "anchor side is ambiguous")
        moving_side = next(c for c in comps if moving in c)
        hits.append({"atoms": [anchor, moving], "moving_fragment_size": len(moving_side)})
        pairs.add(f"{labels[anchor]}-{labels[moving]}")
    if not hits:
        raise ValueError(f"no bond of {parent_smi} splits it into {frag_smis}")
    if len(pairs) != 1:
        raise ValueError(f"cut of {parent_smi} into {frag_smis} gives {len(pairs)} element pairs {sorted(pairs)}")
    bond = pairs.pop()
    names = [parent["name"]] + [f["name"] for f in frags]
    seen = {}
    for sp in [parent] + frags:
        old = seen.setdefault(sp["name"], sp["canonical_plain_smiles"])
        if old != sp["canonical_plain_smiles"]:
            raise ValueError(f"name {sp['name']} given to two structures: {old} and {sp['canonical_plain_smiles']}")
    frag_side = " + ".join(f"{n}*" for n in names[1:])
    reaction = {
        "category": direction.upper(),
        "requested_direction": direction,
        "equation": f"{names[0]}* -> {frag_side}" if direction == "dissociation" else f"{frag_side} -> {names[0]}*",
        "parent_species": names[0],
        "broken_bond_elements": bond,
    }
    if direction == "association":
        reaction["computational_direction"] = f"{names[0]}* -> {frag_side}"
    species, done = [], set()
    for sp in [parent] + frags:
        if sp["name"] not in done:
            done.add(sp["name"])
            species.append(_public(sp))
    return {"species": species, "reaction": reaction,
            "derivation": {"matching_bonds": len(hits), "bond_atoms_explicit_H_indices": [h["atoms"] for h in hits]}}
