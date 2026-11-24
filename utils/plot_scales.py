"""Data-dependent plot limits without clipping finite observations."""
import numpy as np


def adaptive_y(ax, values, scale='auto', padding=.08):
    """Fit all finite values, padding in axis coordinates; return the chosen scale.

    Auto uses linear axes for narrow ranges, log for positive data spanning two
    orders of magnitude, and symlog for signed/zero data with the same spread.
    Reference lines are deliberately excluded from the fit.
    """
    data = np.asarray(values, dtype=float).reshape(-1)
    data = data[np.isfinite(data)]
    if not data.size:
        return ax.get_yscale()
    nonzero = np.abs(data[data != 0])
    broad = bool(nonzero.size and nonzero.max() / nonzero.min() >= 100)
    if scale == 'auto':
        scale = ('log' if np.all(data > 0) else 'symlog') if broad else 'linear'
    if scale == 'log' and np.any(data <= 0):
        raise ValueError('Log scale requires strictly positive values; use symlog for zero or negative values')
    if scale == 'symlog':
        threshold = float(nonzero.min()) if nonzero.size else 1.0
        ax.set_yscale(scale, linthresh=threshold)
    else:
        ax.set_yscale(scale)
    transform = ax.yaxis.get_transform()
    lo, hi = transform.transform([data.min(), data.max()])
    span = hi - lo
    margin = padding * span if span else max(abs(lo) * .05, 1e-6)
    bounds = transform.inverted().transform([lo - margin, hi + margin])
    ax.set_ylim(*bounds)
    return scale
