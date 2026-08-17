"""Sample names reach the tree outputs exactly as the user wrote them.

dnapars gets its own fixed-width identifiers, so nothing needs to rewrite a name that starts
with a digit, and nothing may truncate one.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from accusnv.preprocessing.utils import tree_display_sample_names


class TreeDisplaySampleNameTests(unittest.TestCase):
    def test_numeric_leading_names_are_preserved_exactly(self):
        original = np.array(["7029_3B_10", "7029_3B_02", "alpha"])

        display_names = tree_display_sample_names(original)

        self.assertEqual(display_names.tolist(), original.tolist())
        self.assertEqual(len(set(display_names)), len(display_names))
        self.assertEqual(display_names.dtype, np.dtype(object))

    def test_returned_names_do_not_alias_or_truncate_original_array(self):
        original = np.array(["7029_3B_10", "7029_3B_02"])

        display_names = tree_display_sample_names(original)
        display_names[0] = "a_name_longer_than_the_original_fixed_width"

        self.assertEqual(original.tolist(), ["7029_3B_10", "7029_3B_02"])
        self.assertEqual(display_names[0], "a_name_longer_than_the_original_fixed_width")


if __name__ == "__main__":
    unittest.main()
