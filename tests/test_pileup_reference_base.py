"""Where pileup2diversity gets its reference base, and what it does when that base is unusable.

samtools has been seen writing stray bytes into column 3 of the pileup, so the reference base is
now read from the FASTA instead. These check that the FASTA wins when column 3 disagrees, that a
non-ASCII byte no longer stops the read dead, and that a genuinely ambiguous reference base skips
its position rather than killing a run that is already hours in.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from accusnv.preprocessing.pileup2diversity import pileup2diversity

# Two contigs, so a position is only read correctly if it is looked up within its own contig.
CONTIGS = [("contig1", "AANAA"), ("contig2", "CCCCCGGGGGTTTTTAAAAA")]
CONTIG_START = {"contig1": 0, "contig2": 5}  # first row of each contig in the returned array
NTS = "ATCGatcg"  # columns 0-7 of that array, in order


def pileup_line(contig, position, reference_column, calls):
    """One pileup line, where reference_column is what samtools claims the reference base is."""
    depth = len(calls)
    tail_distances = ",".join(str(10 + i) for i in range(depth))
    fields = [contig.encode(), str(position).encode(), reference_column, str(depth).encode(),
              calls.encode(), b"I" * depth, b"]" * depth, tail_distances.encode()]
    return b"\t".join(fields) + b"\n"


def run_on(contigs, lines):
    """Write a toy reference and pileup to a temporary directory and run pileup2diversity."""
    with tempfile.TemporaryDirectory() as temp_dir:
        fasta = Path(temp_dir) / "genome.fasta"
        fasta.write_text("".join(f">{name}\n{seq}\n" for name, seq in contigs))
        pileup = Path(temp_dir) / "sample.pileup"
        pileup.write_bytes(b"".join(lines))
        return pileup2diversity(str(pileup), str(fasta), "toy")


def counts_at(data, contig, position):
    """The eight per-nucleotide read counts at one position, keyed by base."""
    row = data[CONTIG_START[contig] + position - 1]
    return {base: int(row[i]) for i, base in enumerate(NTS)}


class ReferenceBaseComesFromTheFasta(unittest.TestCase):
    def test_a_plausible_wrong_base_in_column_3_is_ignored(self):
        # contig2 position 2 is C. Column 3 claims A, which is a perfectly valid base, so nothing
        # downstream could notice. Reading absolute position 7 instead would give G, so this also
        # pins the lookup to the position within the contig rather than across the whole genome.
        data = run_on(CONTIGS, [pileup_line("contig2", 2, b"A", "....")])

        self.assertEqual(counts_at(data, "contig2", 2)["C"], 4)
        self.assertEqual(counts_at(data, "contig2", 2)["A"], 0)
        self.assertEqual(counts_at(data, "contig2", 2)["G"], 0)

    def test_a_non_ascii_byte_in_column_3_does_not_stop_the_read(self):
        # 0xc1 is the byte that used to raise UnicodeDecodeError before the line was even split.
        data = run_on(CONTIGS, [pileup_line("contig2", 7, b"\xc1", ",,,,"),
                                pileup_line("contig2", 8, b"G", "..")])

        self.assertEqual(counts_at(data, "contig2", 7)["g"], 4)  # ',' is a reverse-strand match
        self.assertEqual(counts_at(data, "contig2", 8)["G"], 2)  # the read carried on afterwards

    def test_ordinary_mixed_calls_are_still_counted_as_before(self):
        # contig2 position 12 is T: '.' is a forward match, ',' a reverse match, 'A' a real variant.
        data = run_on(CONTIGS, [pileup_line("contig2", 12, b"T", "..A,")])

        counts = counts_at(data, "contig2", 12)
        self.assertEqual(counts["T"], 2)
        self.assertEqual(counts["t"], 1)
        self.assertEqual(counts["A"], 1)


class AmbiguousReferenceBases(unittest.TestCase):
    def test_a_position_whose_reference_base_is_ambiguous_is_skipped(self):
        # contig1 position 3 is N, so '.' cannot be resolved to a base. That position is reported as
        # uncovered; the run continues and the position after it is counted normally.
        data = run_on(CONTIGS, [pileup_line("contig1", 3, b"N", "...."),
                                pileup_line("contig1", 4, b"A", "....")])

        self.assertEqual(sum(counts_at(data, "contig1", 3).values()), 0)
        self.assertEqual(counts_at(data, "contig1", 4)["A"], 4)

    def test_an_ambiguous_reference_base_with_no_matching_reads_is_harmless(self):
        # Reads that spell out their own base need no reference to resolve, so they still count.
        data = run_on(CONTIGS, [pileup_line("contig1", 3, b"N", "GGGG")])

        self.assertEqual(counts_at(data, "contig1", 3)["G"], 4)

    def test_a_reference_full_of_ambiguous_bases_still_fails(self):
        # A handful of unusable positions is bad data and gets skipped, but a reference that is
        # mostly unusable means the wrong reference genome, and that has to stop the run.
        contigs = [("all_ambiguous", "N" * 150)]
        lines = [pileup_line("all_ambiguous", p, b"N", "....") for p in range(1, 151)]

        with self.assertRaises(ValueError) as raised:
            run_on(contigs, lines)

        self.assertIn("reference genome", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
