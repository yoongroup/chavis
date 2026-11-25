"""Plain SMILES -> CHAVIS species.json / reactions.json entries (add-on; does not import CHAVIS)."""
from .core import RULES, derive_reaction, describe_species

__all__ = ["RULES", "derive_reaction", "describe_species"]
__version__ = "0.1.0"
