from chavis.inputs.fs_sites import direct_fs_seeds
from chavis.inputs.type2_methods import Type2Methods as _Type2Methods
from .search import PreparationIncomplete


class Type2Methods(_Type2Methods):
    _preparation_incomplete = PreparationIncomplete


__all__ = ['Type2Methods', 'direct_fs_seeds']
