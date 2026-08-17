"""What the CLI accepts as a run: which FASTQs belong to a sample, and which references resolve.

Everything here runs before Snakemake starts, and a mistake at this point silently maps the
wrong reads, so each case is checked on the resolved sample sheet the workflow actually reads.
"""
import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from accusnv import cli
from accusnv.preprocessing.utils import resolve_fasta_path


class _Parser(argparse.ArgumentParser):
    """argparse exits the process on error; raise instead so the message can be asserted on."""
    def error(self, message):
        raise ValueError(message)


def resolve_samples(root, read_files, rows, reference="clade1_ref"):
    """Resolve a sample sheet the way a run does, and return it.

    `read_files` are created in <root>/reads (tests needing symlinks make them there first),
    and `rows` are (Sample, FileName, Group, Type) tuples all pointing at one reference.
    """
    reads = root / "reads"
    reads.mkdir(exist_ok=True)
    for name in read_files:
        (reads / name).write_text("")
    (root / "refs" / reference).mkdir(parents=True, exist_ok=True)
    (root / "refs" / reference / "genome.fasta").write_text(">chr\nACGT\n")
    (root / "out").mkdir(exist_ok=True)
    with open(root / "samples.csv", "w") as f:
        f.write("Path,Sample,FileName,Reference,Group,Outgroup,Type\n")
        for sample, filename, group, read_type in rows:
            f.write(f"{reads},{sample},{filename},{reference},{group},0,{read_type}\n")

    args = argparse.Namespace(
        input_sample_info=str(root / "samples.csv"), ref_dir=str(root / "refs"),
        output_dir=str(root / "out"),
        env="true",  # an env command is given, so no check for locally installed tools
        pipeline_file=None, config_file=None, downstream_only=False,
        skip_trees=False, skip_all_downstream=False, skip_samclip=False)
    return pd.read_csv(cli.check_inputs_and_dependencies(args, _Parser()))


class FastqMatchingTests(unittest.TestCase):
    def test_reads_of_a_longer_sample_name_are_not_matched(self):
        # 7029_3B_1* also globs 7029_3B_10's reads, so the match has to be anchored to what
        # follows the sample's FileName.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            samples = resolve_samples(
                root,
                ["7029_3B_1_1.fastq.gz", "7029_3B_1_2.fastq.gz",
                 "7029_3B_10_1.fastq.gz", "7029_3B_10_2.fastq.gz"],
                [("7029_3B_1", "7029_3B_1", "g1", "PE")])

            self.assertEqual(
                [samples.loc[0, "Read1_file_path"], samples.loc[0, "Read2_file_path"]],
                [str(root / "reads" / "7029_3B_1_1.fastq.gz"),
                 str(root / "reads" / "7029_3B_1_2.fastq.gz")])

    def test_symlinked_reads_are_matched_by_link_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source_reads"
            source_dir.mkdir()
            reads = root / "reads"
            reads.mkdir()
            for source_name, link_name in (("lane_A_R1.fastq.gz", "MOB_004_Vag_09_1.fastq.gz"),
                                           ("lane_A_R2.fastq.gz", "MOB_004_Vag_09_2.fastq.gz")):
                (source_dir / source_name).write_text("")
                (reads / link_name).symlink_to(source_dir / source_name)

            samples = resolve_samples(root, [], [("MOB_004_Vag_09", "MOB_004_Vag_09", "g1", "PE")])

            self.assertEqual(
                [samples.loc[0, "Read1_file_path"], samples.loc[0, "Read2_file_path"]],
                [str(reads / "MOB_004_Vag_09_1.fastq.gz"),
                 str(reads / "MOB_004_Vag_09_2.fastq.gz")])

    def test_files_sitting_next_to_the_reads_are_ignored(self):
        # A checksum or index beside a single-end read must not count as a second candidate.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            samples = resolve_samples(root, ["S1.fastq.gz", "S1.fastq.gz.md5", "S1.stats.txt"],
                                      [("S1", "S1", "g1", "SE")])

            self.assertEqual(samples.loc[0, "Read1_file_path"], str(root / "reads" / "S1.fastq.gz"))
            self.assertTrue(pd.isna(samples.loc[0, "Read2_file_path"]))

    def test_the_usual_read_markers_are_all_recognised(self):
        for r1, r2 in (("S1_1.fq", "S1_2.fq"), ("S1.R1.fastq", "S1.R2.fastq"),
                       ("S1-1.fq.gz", "S1-2.fq.gz"), ("S1_R1_001.fastq.gz", "S1_R2_001.fastq.gz")):
            with self.subTest(reads=r1), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                samples = resolve_samples(root, [r1, r2], [("S1", "S1", "g1", "PE")])

                self.assertEqual(
                    [samples.loc[0, "Read1_file_path"], samples.loc[0, "Read2_file_path"]],
                    [str(root / "reads" / r1), str(root / "reads" / r2)])

    def test_a_paired_end_sample_missing_its_reverse_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "forward and reverse"):
                resolve_samples(Path(temp_dir), ["S1_1.fastq.gz"], [("S1", "S1", "g1", "PE")])

    def test_a_sample_listed_twice_in_one_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "more than once in the same group"):
                resolve_samples(Path(temp_dir), ["S1_1.fastq.gz", "S1_2.fastq.gz"],
                                [("S1", "S1", "g1", "PE"), ("S1", "S1", "g1", "PE")])

    def test_one_sample_name_pointing_at_two_read_sets_is_rejected(self):
        # The same name in two groups is fine (an outgroup shared between them), but only if
        # every row names the same reads.
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "different FASTQ files"):
                resolve_samples(Path(temp_dir),
                                ["S1_1.fastq.gz", "S1_2.fastq.gz", "other_1.fastq.gz", "other_2.fastq.gz"],
                                [("S1", "S1", "g1", "PE"), ("S1", "other", "g2", "PE")])


class ReferenceTests(unittest.TestCase):
    def test_reference_directory_resolves_its_fasta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_dir = Path(temp_dir)
            (reference_dir / "scaffolds.fna.gz").write_text("")
            self.assertEqual(resolve_fasta_path(temp_dir), str(reference_dir / "scaffolds.fna.gz"))

            # genome.* wins when a directory holds more than one FASTA.
            (reference_dir / "genome.fasta").write_text(">chr\nACGT\n")
            self.assertEqual(resolve_fasta_path(temp_dir), str(reference_dir / "genome.fasta"))

    def test_a_reference_without_a_fasta_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "No reference FASTA"):
                resolve_fasta_path(temp_dir)

    def test_an_ambiguous_reference_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "one.fasta").write_text("")
            (Path(temp_dir) / "two.fasta").write_text("")
            with self.assertRaisesRegex(ValueError, "Multiple reference FASTA"):
                resolve_fasta_path(temp_dir)

    def test_reference_names_reserved_by_the_filename_scheme_are_rejected(self):
        # Mapping outputs are named <sample>_ref_<reference>, so a reference whose own name
        # contains that delimiter cannot be read back out of the filename.
        with tempfile.TemporaryDirectory() as temp_dir:
            for reference in ("ref_clade1", "clade_ref_1"):
                with self.assertRaisesRegex(ValueError, '"ref_"'):
                    resolve_samples(Path(temp_dir), ["S1_1.fastq.gz", "S1_2.fastq.gz"],
                                    [("S1", "S1", "g1", "PE")], reference=reference)


if __name__ == "__main__":
    unittest.main()
