import tempfile
import unittest
from pathlib import Path

import accusnv_snakemake as cli


class InputValidationTests(unittest.TestCase):
    def test_findfastqfile_does_not_match_longer_prefixes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reads_dir = Path(temp_dir)
            expected_r1 = reads_dir / "7029_3B_1_1.fastq.gz"
            expected_r2 = reads_dir / "7029_3B_1_2.fastq.gz"
            expected_r1.write_text("")
            expected_r2.write_text("")
            (reads_dir / "7029_3B_10_1.fastq.gz").write_text("")
            (reads_dir / "7029_3B_10_2.fastq.gz").write_text("")

            found = cli.findfastqfile(str(reads_dir), "7029_3B_1", "7029_3B_1")

            self.assertEqual(found, [str(expected_r1), str(expected_r2)])

    def test_reference_validation_rejects_reserved_ref_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs_dir = root / "refs"
            ref_dir = refs_dir / "ref_clade1"
            ref_dir.mkdir(parents=True)
            (ref_dir / "genome.fasta").write_text(">chr\nACGT\n")
            sample_csv = root / "samples.csv"
            sample_csv.write_text(
                "Path,Sample,FileName,Reference,Group,Outgroup,Type\n"
                "/reads,sample,sample,ref_clade1,group,0,PE\n"
            )

            with self.assertRaisesRegex(ValueError, 'must not start with "ref_"'):
                cli._check_reference_inputs(str(sample_csv), str(refs_dir))

    def test_reference_validation_requires_genome_fasta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs_dir = root / "refs"
            (refs_dir / "clade1_ref").mkdir(parents=True)
            sample_csv = root / "samples.csv"
            sample_csv.write_text(
                "Path,Sample,FileName,Reference,Group,Outgroup,Type\n"
                "/reads,sample,sample,clade1_ref,group,0,PE\n"
            )

            with self.assertRaisesRegex(ValueError, "genome.fasta"):
                cli._check_reference_inputs(str(sample_csv), str(refs_dir))


if __name__ == "__main__":
    unittest.main()
