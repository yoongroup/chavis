PIPELINE_VERSION = '0.7-parent-2fixed2free'
EMBED_SEED = 61453
EMBED_TRIES = 10
PLACEHOLDER = 'Cl'
SLAB_ELEMENT = 'Ni'
SLAB_SIZE = (3, 3, 4)
LATTICE_A = 3.52
VACUUM = 10.0
SITES = ('ontop', 'bridge', 'fcc', 'hcp')
CONTACT_SCALE = 1.0
UMA_MODEL = 'uma-s-1p2p1'
UMA_DEVICE = 'cpu'
UMA_COMPILE = False
GAS_TASK = 'omol'
GAS_FMAX = 0.02
GAS_STEPS = 300
SPIN_OVERRIDE = {'C': 3}
SURF_TASK = 'oc20'
SURF_FMAX = 0.05
SURF_STEPS = 200
SURF_FIXED_LAYERS = 2
DEGENERATE_EV = 0.02
MIGRATED_A = 1.0
DESORBED_A = 3.0


def provenance():
    return {
        k: v for k,
        v in globals().items() if k.isupper() and isinstance(v, (str, int, float, bool, tuple, dict))
    }
