"""Single-hue scalar-field palettes used by the paper figures."""

import numpy as np
from matplotlib.colors import ListedColormap, to_rgb

TEAL = "#644ba5"
FIELD_EDGE = "#111111"
ALPHA_FLOOR = 0.04
RAMP_SIZE = 256


def _lightness_over_white(color, alphas):
    hue = np.asarray(to_rgb(color), dtype=float)
    composite = 1.0 + np.asarray(alphas, dtype=float)[:, None] * (hue - 1.0)
    linear = np.where(
        composite <= 0.04045,
        composite / 12.92,
        ((composite + 0.055) / 1.055) ** 2.4,
    )
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    return np.where(
        luminance > (6.0 / 29.0) ** 3,
        116.0 * np.cbrt(luminance) - 16.0,
        luminance * (29.0 / 3.0) ** 3,
    )


def _perceptual_alphas(alpha_floor):
    """Interpolate opacities from alpha_floor to 1 with uniform CIE L* steps over white."""
    dense = np.linspace(0.0, 1.0, 2048)
    lightness = _lightness_over_white(TEAL, dense)
    low, high = np.interp([alpha_floor, 1.0], dense, lightness)
    targets = np.linspace(low, high, RAMP_SIZE)
    return np.interp(targets, lightness[::-1], dense[::-1])


def teal_alpha_cmap():
    """Return a single-hue colormap with opacity ALPHA_FLOOR..1 and uniform L* steps over white."""
    alphas = _perceptual_alphas(ALPHA_FLOOR)
    return ListedColormap(np.column_stack([np.tile(to_rgb(TEAL), (RAMP_SIZE, 1)), alphas]), name="teal_alpha")


def teal_opaque_cmap():
    """Return an opaque white-to-TEAL colormap with uniform CIE L* steps."""
    alphas = _perceptual_alphas(0.0)[:, None]
    rgb = 1.0 + alphas * (np.asarray(to_rgb(TEAL), dtype=float) - 1.0)
    return ListedColormap(np.column_stack([rgb, np.ones(RAMP_SIZE)]), name="teal_tint")
