from chavis.runtime.numeric_defaults import load_numeric_defaults
_NAMES = {
    'EMBED_SEED': 'embed_seed',
    'EMBED_TRIES': 'embed_tries',
    'PLACEHOLDER': 'placeholder',
    'SPIN_OVERRIDE': 'spin_override'
}


def __getattr__(name):
    if name not in _NAMES:
        raise AttributeError(name)
    return load_numeric_defaults()['inputs'][_NAMES[name]]
