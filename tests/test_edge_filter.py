"""Edge_filter: what it rejects, and the two ways a site escapes it.

A site is only rejected when it is both near a contig end and lopsided between the strands, so
these check each half in isolation as well as together, and that a rejected site is dropped even
when the CNN called it.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from accusnv.downstream.snv import dec_final_lab

CONTIG_STARTS = np.array([1, 1001])  # two 1000 bp contigs
GENOME_LENGTH = 2000


def edge_filter(positions, fwd, rev, contig_edge_bp=100, max_strand_imbalance=0.3):
    """The filter as accusnv.py applies it: distance to a contig end, and strand balance."""
    ends = np.append(CONTIG_STARTS[1:] - 1, GENOME_LENGTH)
    contig = np.searchsorted(CONTIG_STARTS, positions, 'right') - 1
    to_end = np.minimum(positions - CONTIG_STARTS[contig], ends[contig] - positions)
    quieter = np.minimum(fwd, rev) / np.maximum(fwd + rev, 1)
    return (to_end < contig_edge_bp) & (quieter < max_strand_imbalance)


class EdgeFilter(unittest.TestCase):

    def test_rejects_only_when_both_conditions_hold(self):
        # in order: near an end and one-sided; near an end but balanced; one-sided mid-contig;
        # neither. Only the first is the contig-end artifact.
        positions = np.array([1050, 1060, 1500, 1510])
        fwd = np.array([300, 150, 300, 150])
        rev = np.array([30, 150, 30, 150])
        np.testing.assert_array_equal(edge_filter(positions, fwd, rev),
                                      [True, False, False, False])

    def test_measures_distance_within_the_position_s_own_contig(self):
        # 1050 is 50 bp into contig 2, not 1049 bp into the genome
        one_sided = (np.array([300]), np.array([30]))
        self.assertTrue(edge_filter(np.array([1050]), *one_sided)[0])
        self.assertFalse(edge_filter(np.array([1500]), *one_sided)[0])

    def test_both_ends_of_a_contig_count(self):
        one_sided = (np.array([300, 300]), np.array([30, 30]))
        np.testing.assert_array_equal(edge_filter(np.array([1, 1000]), *one_sided), [True, True])

    def test_zero_turns_the_filter_off(self):
        self.assertFalse(edge_filter(np.array([1050]), np.array([300]), np.array([30]),
                                     contig_edge_bp=0)[0])

    def test_a_rejected_site_is_dropped_even_when_the_cnn_called_it(self):
        # the CNN and the filters both say yes; Edge_filter still wins, as Qual_filter does
        called = ['1', '1.0']
        self.assertEqual(dec_final_lab('1', called, '1', '0', 0.0, 0, False, 0.25, edge=0), '1')
        rejected = ['1', '1.0']
        self.assertEqual(dec_final_lab('1', rejected, '1', '0', 0.0, 0, False, 0.25, edge=1), '0')
        self.assertEqual(rejected, ['0', '0'])  # the model's call is rewritten to match


if __name__ == '__main__':
    unittest.main()
