"""Calculator-free validation shared by the settings loader and adsorption CLI."""
from __future__ import annotations

import math


def parse_azimuth_step(value):
    """Parse 'auto' or an exact, positive angular grid covering one full turn."""
    if value == 'auto':
        return 'auto'
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ValueError('adsorption_azimuth_step_deg must be auto or a finite positive number') from None
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('adsorption_azimuth_step_deg must be auto or a finite positive number')
    count = 360.0 / value
    if not math.isfinite(count) or count < 1 or not math.isclose(count, round(count), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('adsorption_azimuth_step_deg must divide 360 degrees exactly')
    return float(value)
