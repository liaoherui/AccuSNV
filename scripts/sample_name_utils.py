"""Utilities for preserving user-provided sample names in output files."""

import numpy as np


def tree_display_sample_names(sample_names):
    """Return an independent array of the original sample names for tree outputs.

    dnapars receives separate, fixed-width internal identifiers, so user-provided
    names (including names that start with a digit) do not need to be rewritten.
    Object dtype prevents later assignments from silently truncating names.
    """
    return np.array(sample_names, dtype=object, copy=True)
