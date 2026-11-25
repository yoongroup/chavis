from chavis.inputs import seed_geometry as _seed
from .species_recipes import SPECIES
align_vector = _seed.align_vector
surface_frame = _seed.surface_frame
site_at = _seed.site_at
place_on_snapshot = _seed.place_on_snapshot
cleavage = _seed.cleavage


def orientation_variants(species):
    return _seed.orientation_variants(species, SPECIES[species])


def orient(fragment, species, variant, normal=(0, 0, 1)):
    return _seed.orient(fragment, species, variant, normal, recipe=SPECIES[species])


__all__ = [
    'orientation_variants',
    'align_vector',
    'orient',
    'surface_frame',
    'site_at',
    'place_on_snapshot',
    'cleavage'
]
