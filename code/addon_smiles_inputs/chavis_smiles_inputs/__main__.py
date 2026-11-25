"""CLI: python -m chavis_smiles_inputs --reactant "C" --products "[CH3]" "[H]"

Association: give two reactants and one product, e.g. --reactant "[CH]" "[OH]" --products "[CH]O".
Prints the JSON entries; writes nothing.
"""
import argparse
import json
import sys

from .core import derive_reaction


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m chavis_smiles_inputs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reactant", nargs="+", required=True, help="plain SMILES of the reactant(s)")
    ap.add_argument("--products", nargs="+", required=True, help="plain SMILES of the product(s)")
    a = ap.parse_args(argv)
    try:
        out = derive_reaction(a.reactant if len(a.reactant) > 1 else a.reactant[0], a.products)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
