import numpy as np
from ase.constraints import FixConstraint


class ReferenceConstraint(FixConstraint):

    def __init__(self, weights, point, axes):
        self.weights = {int(i): float(w) for i, w in weights.items()}
        self.point = np.asarray(point, float)
        self.axes = np.asarray(axes, float).reshape(-1, 3)
        if not self.weights or not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError('Reference weights must sum to one')
        if np.linalg.matrix_rank(self.axes) != len(self.axes):
            raise ValueError('Independent constraint axes required')

    def _matrix(self, atoms):
        b = np.zeros((len(self.axes), 3 * len(atoms)))
        for i, w in self.weights.items():
            b[:, 3 * i:3 * i + 3] = w * self.axes
        return b

    def adjust_positions(self, atoms, newpositions):
        b = self._matrix(atoms)
        error = b @ newpositions.ravel() - self.axes @ self.point
        newpositions[:] -= (b.T @ np.linalg.solve(b @ b.T, error)).reshape(-1, 3)

    def adjust_forces(self, atoms, forces):
        b = self._matrix(atoms)
        forces[:] -= (b.T @ np.linalg.solve(b @ b.T, b @ forces.ravel())).reshape(-1, 3)

    def get_removed_dof(self, atoms):
        return len(self.axes)

    def copy(self):
        return ReferenceConstraint(
            dict(self.weights),
            self.point.copy(),
            self.axes.copy()
        )

    def todict(self):
        return {
            'name': 'CampaignReferenceConstraint',
            'kwargs': {
                'weights': self.weights,
                'point': self.point.tolist(),
                'axes': self.axes.tolist()
            }
        }


def local_maxima(energies):
    return [i for i in range(1, len(energies) - 1) if energies[i] > energies[i - 1] and energies[i] > energies[i + 1]]


def scan_constraints_released(atoms):
    from ase.constraints import FixAtoms
    return all((isinstance(c, FixAtoms) for c in atoms.constraints))
