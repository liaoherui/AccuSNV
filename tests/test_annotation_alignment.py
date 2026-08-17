"""parse_gff must return one dataframe per contig, in the order it was given the contig names.

Annotations are looked up by contig index downstream, so a contig the GFF says nothing about
still has to occupy its slot; skipping it would shift every later contig's annotations.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from accusnv.downstream.snv import parse_gff


class AnnotationAlignmentTests(unittest.TestCase):
    def test_contigs_without_gff_records_receive_empty_placeholders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_dir = Path(temp_dir)
            (reference_dir / "genome.gff").write_text("##gff-version 3\n")

            annotations = parse_gff(
                str(reference_dir),
                ["annotated_contig", "unannotated_contig"],
            )

            self.assertEqual(len(annotations), 2)
            self.assertEqual([annotation.shape for annotation in annotations], [(0, 0), (0, 0)])

    def test_a_reference_without_exactly_one_gff_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "gff"):
                parse_gff(temp_dir, ["contig"])

            (Path(temp_dir) / "genome.gff").write_text("##gff-version 3\n")
            (Path(temp_dir) / "extra.gff").write_text("##gff-version 3\n")
            with self.assertRaisesRegex(ValueError, "gff"):
                parse_gff(temp_dir, ["contig"])


if __name__ == "__main__":
    unittest.main()
