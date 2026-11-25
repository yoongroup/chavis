from pathlib import Path
import sys


def ensure_chavis_import_path() -> Path:
    location = Path(__file__).resolve().parent
    if (location / 'chavis' / '__init__.py').is_file():
        source = location
    elif location.name == 'runtime_support' and location.parent.name == 'support':
        source = location.parents[1] / 'src'
    else:
        raise FileNotFoundError('Frozen worker lacks its chavis package closure')
    package = source / 'chavis' / '__init__.py'
    if not package.is_file():
        raise FileNotFoundError(package)
    value = str(source)
    if value not in sys.path:
        sys.path.insert(0, value)
    return source
