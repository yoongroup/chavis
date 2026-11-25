"""Paper 15 reactions: derive every entry from plain SMILES only and compare with the shipped JSON files.

Run (from the add-on folder):  python -m unittest discover -s tests -v
Writes results/paper15_match.md (inside the add-on folder only); results/paper15_match.json too when the
environment variable CHAVIS_ADDON_WRITE_JSON=1 is set. The CHAVIS files are only read.
CHAVIS root: $CHAVIS_RELEASE_ROOT if set; else the folder above this add-on if it holds campaigns/ni_drm/config
(shipped layout); else ../release_public (development layout).
"""
import json
import os
import sys
import unittest
from pathlib import Path

ADDON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON))
from rdkit import Chem  # noqa: E402

from chavis_smiles_inputs import derive_reaction, describe_species  # noqa: E402


def _release_root():
    if os.environ.get("CHAVIS_RELEASE_ROOT"):
        return Path(os.environ["CHAVIS_RELEASE_ROOT"])
    for root in (ADDON.parent, ADDON.parent / "release_public"):
        if (root / "campaigns/ni_drm/config/species.json").is_file():
            return root
    raise FileNotFoundError("CHAVIS root with campaigns/ni_drm/config not found; set CHAVIS_RELEASE_ROOT")


RELEASE = _release_root()
SPECIES_JSON = RELEASE / "campaigns/ni_drm/config/species.json"
REACTIONS_JSON = RELEASE / "campaigns/ni_drm/config/reactions.json"
COHORT_JSON = RELEASE / "reproduce/inputs/cohorts/paper15.json"
RESULTS = ADDON / "results"

# The only hand-written input: plain SMILES. (reactants, products) in the order of the paper equation.
PLAIN = {
    "D01": (["C"], ["[CH3]", "[H]"]),
    "D02": (["[CH3]"], ["[CH2]", "[H]"]),
    "D03": (["[CH2]"], ["[CH]", "[H]"]),
    "D04": (["[CH]"], ["[C]", "[H]"]),
    "D05": (["O=C=O"], ["[C-]#[O+]", "[O]"]),
    "D06": (["[CH]=O"], ["[C-]#[O+]", "[H]"]),
    "D07": (["[OH]"], ["[O]", "[H]"]),
    "D08": (["O"], ["[OH]", "[H]"]),
    "D09": (["[C-]#[O+]"], ["[C]", "[O]"]),
    "D10": (["[CH]=O"], ["[CH]", "[O]"]),
    "D13": (["[H][H]"], ["[H]", "[H]"]),
    "N01": (["[CH]O"], ["[CH]=O", "[H]"]),
    "N02": (["[C]O"], ["[C-]#[O+]", "[H]"]),
    "N03": (["[CH]", "[OH]"], ["[CH]O"]),
    "N04": (["[C]", "[OH]"], ["[C]O"]),
}
REACTION_FIELDS = ["parent_species", "equation", "broken_bond_elements", "category", "computational_direction"]
SPECIES_FIELDS = ["name", "mode", "binder_element", "smiles"]
# state_class and default_route are out of scope (not derived; the existing code has defaults for both).


def canon(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES {smiles!r}")
    return Chem.MolToSmiles(mol)


def names_of(side):
    return [t.strip().rstrip("*").strip() for t in side.split(" + ")]


def compare():
    species = json.loads(SPECIES_JSON.read_text(encoding="utf-8"))
    reactions = {r["id"]: r for r in json.loads(REACTIONS_JSON.read_text(encoding="utf-8"))["reactions"]}
    ids = json.loads(COHORT_JSON.read_text(encoding="utf-8"))["reactions"]
    assert ids == list(PLAIN), (ids, list(PLAIN))
    rows, species_rows = [], {}
    for rid in ids:
        shipped = reactions[rid]
        reac, prod = PLAIN[rid]
        out = derive_reaction(reac if len(reac) > 1 else reac[0], prod)
        derived = out["reaction"]
        cells = {}
        for f in REACTION_FIELDS:
            want, got = shipped.get(f), derived.get(f)
            cells[f] = {"match": want == got, "shipped": want, "derived": got}
        lhs, rhs = shipped["equation"].split(" -> ")
        frag_names = names_of(rhs) if len(reac) == 1 else names_of(lhs)
        frag_smiles = prod if len(reac) == 1 else reac
        positions = [(shipped["parent_species"], (prod[0] if len(reac) > 1 else reac[0]))]
        positions += list(zip(frag_names, frag_smiles))
        assert len(positions) == 3, (rid, positions)
        for f in SPECIES_FIELDS:
            bad = []
            for ship_name, smi in positions:
                d = describe_species(smi)
                s = species[ship_name]
                want = ship_name if f == "name" else s.get(f)
                got = d[f]
                if f == "smiles":
                    ok = canon(want) == canon(got)
                else:
                    ok = want == got
                key = canon(smi)
                srow = species_rows.setdefault(key, {"shipped_name": ship_name, "plain_smiles": smi, "fields": {}})
                assert srow["shipped_name"] == ship_name, (key, srow["shipped_name"], ship_name)
                srow["fields"][f] = {"match": ok, "shipped": want, "derived": got}
                if not ok:
                    bad.append(f"{ship_name}: shipped {want!r} / derived {got!r}")
            cells["species." + f] = {"match": not bad, "detail": bad}
        rows.append({"id": rid, "cells": cells, "derived": out})
    return rows, species_rows


def write_tables(rows, species_rows):
    RESULTS.mkdir(exist_ok=True)
    cols = REACTION_FIELDS + ["species." + f for f in SPECIES_FIELDS]
    lines = ["# Paper 15: derived from plain SMILES vs shipped JSON", "",
             "Each cell: match, or `shipped / derived`. Species columns cover the parent and both fragments.", "",
             "| reaction | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for r in rows:
        out = []
        for c in cols:
            cell = r["cells"][c]
            if cell["match"]:
                out.append("match")
            elif "detail" in cell:
                out.append("; ".join(cell["detail"]))
            else:
                out.append(f"`{cell['shipped']}` / `{cell['derived']}`")
        lines.append(f"| {r['id']} | " + " | ".join(out) + " |")
    lines += ["", "## Distinct species", "", "| shipped name | plain SMILES | " + " | ".join(SPECIES_FIELDS) + " |",
              "|---|---|" + "---|" * len(SPECIES_FIELDS)]
    for key, s in sorted(species_rows.items(), key=lambda kv: kv[1]["shipped_name"]):
        out = []
        for f in SPECIES_FIELDS:
            c = s["fields"][f]
            out.append(("match (`%s`)" % c["derived"]) if c["match"] else f"`{c['shipped']}` / `{c['derived']}`")
        lines.append(f"| {s['shipped_name']} | `{s['plain_smiles']}` | " + " | ".join(out) + " |")
    total = sum(len(cols) for _ in rows)
    n_ok = sum(r["cells"][c]["match"] for r in rows for c in cols)
    sp_total = sum(len(s["fields"]) for s in species_rows.values())
    sp_ok = sum(c["match"] for s in species_rows.values() for c in s["fields"].values())
    lines += ["", f"Reaction-table cells: {n_ok}/{total} match ({len(rows)} reactions x {len(cols)} fields).",
              f"Species-table cells: {sp_ok}/{sp_total} match ({len(species_rows)} species x {len(SPECIES_FIELDS)} fields).",
              "state_class and default_route are not derived (out of scope) and not compared."]
    (RESULTS / "paper15_match.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if os.environ.get("CHAVIS_ADDON_WRITE_JSON") == "1":
        (RESULTS / "paper15_match.json").write_text(json.dumps(
            {"reactions": [{"id": r["id"], "cells": r["cells"], "derived": r["derived"]} for r in rows],
             "species": species_rows, "cells_match": n_ok, "cells_total": total,
             "species_cells_match": sp_ok, "species_cells_total": sp_total}, indent=1, ensure_ascii=False),
            encoding="utf-8")
    return n_ok, total, sp_ok, sp_total


class Paper15(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.species = compare()
        cls.n_ok, cls.total, cls.sp_ok, cls.sp_total = write_tables(cls.rows, cls.species)
        print(f"\nreaction cells {cls.n_ok}/{cls.total}, species cells {cls.sp_ok}/{cls.sp_total}")

    def test_counts(self):
        self.assertEqual(len(self.rows), 15)
        self.assertEqual(len(self.species), 15)  # distinct species by canonical plain SMILES
        self.assertEqual(len({s["shipped_name"] for s in self.species.values()}), 15)
        self.assertEqual(self.total, 15 * 9)  # 5 reaction fields + 4 species fields per reaction
        self.assertEqual(self.sp_total, 15 * 4)

    def test_no_mismatches(self):
        found = {(r["id"], c) for r in self.rows for c, cell in r["cells"].items() if not cell["match"]}
        self.assertEqual(found, set())
        self.assertEqual(self.n_ok, self.total)
        self.assertEqual(self.sp_ok, self.sp_total)

    def test_out_of_scope_fields_absent(self):
        for r in self.rows:
            self.assertNotIn("state_class", r["derived"]["reaction"])
            self.assertNotIn("default_route", r["derived"]["reaction"])

    def test_species_fields_all_match(self):
        bad = [(s["shipped_name"], f) for s in self.species.values() for f, c in s["fields"].items() if not c["match"]]
        self.assertEqual(bad, [])


class Rules(unittest.TestCase):
    def test_ambiguous_or_impossible_cut_raises(self):
        with self.assertRaises(ValueError):
            derive_reaction("C", ["[CH2]", "[H][H]"])  # no single bond gives these fragments
        with self.assertRaises(ValueError):
            derive_reaction("C", ["[CH3]", "[O]"])  # not atom balanced

    def test_two_radical_centres_raise(self):
        with self.assertRaises(ValueError):
            describe_species("[CH2][CH2]")

    def test_marker_in_input_raises(self):
        with self.assertRaises(ValueError):
            describe_species("ClC")

    def test_cli(self):
        import subprocess
        p = subprocess.run([sys.executable, "-m", "chavis_smiles_inputs", "--reactant", "C", "--products", "[CH3]", "[H]"],
                           cwd=ADDON, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(p.returncode, 0, p.stderr)
        doc = json.loads(p.stdout)
        self.assertEqual(doc["reaction"]["equation"], "CH4* -> CH3* + H*")
        self.assertEqual(len(doc["species"]), 3)


if __name__ == "__main__":
    unittest.main()
