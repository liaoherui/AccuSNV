"""Shared helpers for locating a reference genome and reading its stats.

Sample sheets and FASTQ paths are not handled here: the CLI resolves every read and reference
path up front and writes them into samples.csv, which the workflow reads directly."""
import os
import glob
import gzip

import numpy as np
from Bio import SeqIO


# Recognised reference FASTA extensions, most conventional first: a directory is searched for
# these in order, and a genome.<ext> wins over any other name.
FASTA_SUFFIXES = ('.fasta', '.fa', '.fna', '.fasta.gz', '.fa.gz', '.fna.gz')


def resolve_fasta_path(ref):
    '''Resolve a reference genome to its FASTA file.

    `ref` may be the FASTA file itself (any name or extension, optionally .gz) or a directory
    containing a single FASTA with one of FASTA_SUFFIXES (``genome.*`` is preferred when several
    are present). This lets the whole pipeline pass the located FASTA path around instead of
    assuming a fixed name.
    '''
    if os.path.isfile(ref):
        return ref
    cands = {f for suffix in FASTA_SUFFIXES for f in glob.glob(os.path.join(ref, '*' + suffix))}
    for suffix in FASTA_SUFFIXES:
        cand = os.path.join(ref, 'genome' + suffix)
        if cand in cands:
            return cand
    if len(cands) == 1:
        return cands.pop()
    raise ValueError(('No' if not cands else 'Multiple') + ' reference FASTA file(s) (' +
                     ', '.join(FASTA_SUFFIXES) + ') found in: ' + ref)


def ref_directory(ref):
    '''Directory that holds the reference (its GFF and annotation cache live alongside the FASTA).'''
    return os.path.dirname(ref) if os.path.isfile(ref) else ref


def read_fasta(ref):
    '''Read a reference FASTA (path or directory; .gz supported). Returns a SeqIO iterator.'''
    fasta = resolve_fasta_path(ref)
    if fasta.endswith('.gz'):
        return SeqIO.parse(gzip.open(fasta, "rt"), 'fasta')
    return SeqIO.parse(fasta, 'fasta')


def genomestats(REFGENOMEFOLDER):
    '''Parse genome to extract relevant stats

    Args:
        REFGENOMEFOLDER (str): Directory containing reference genome file.

    Returns:
        ChrStarts (arr): DESCRIPTION.
        Genomelength (arr): DESCRIPTION.
        ScafNames (arr): DESCRIPTION.

    '''

    refgenome = read_fasta(REFGENOMEFOLDER)

    Genomelength = 0
    ChrStarts = []
    ScafNames = []
    for record in refgenome:
        ChrStarts.append(Genomelength) # chr1 starts at 0 in analysis.m
        Genomelength = Genomelength + len(record)
        ScafNames.append(record.id)
    ChrStarts = np.asarray(ChrStarts,dtype=int)
    Genomelength = np.asarray(Genomelength,dtype=int)
    ScafNames = np.asarray(ScafNames,dtype=object)
    return ChrStarts,Genomelength,ScafNames


def p2chrpos(p, ChrStarts):
    '''Convert 1col list of pos to 2col array with chromosome and pos on chromosome

    Args:
        p (TYPE): DESCRIPTION.
        ChrStarts (TYPE): DESCRIPTION.

    Returns:
        chrpos (TYPE): DESCRIPTION.

    '''

    # get chr and pos-on-chr
    chromo = np.ones(len(p),dtype=int)
    if len(ChrStarts) > 1:
        for i in ChrStarts[1:]:
            chromo = chromo + (p > i) # when (p > i) evaluates 'true' lead to plus 1 in summation. > bcs ChrStarts start with 0...genomestats()
        positions = p - ChrStarts[chromo-1] # [chr-1] -1 due to 0based index
        chrpos = np.column_stack((chromo,positions))
    else:
        chrpos = np.column_stack((chromo,p))
    return chrpos


def tree_display_sample_names(sample_names):
    """Return an independent array of the original sample names for tree outputs.

    dnapars receives separate, fixed-width internal identifiers, so user-provided
    names (including names that start with a digit) do not need to be rewritten.
    Object dtype prevents later assignments from silently truncating names.
    """
    return np.array(sample_names, dtype=object, copy=True)
