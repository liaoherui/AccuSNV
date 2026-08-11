#!/usr/bin/env python3
"""Records the positions bcftools dropped before AccuSNV ever saw them, and why.

The candidate positions AccuSNV works from are whatever survives two gates in the mapping
stage: ``bcftools view -v snps -q <variant_min_af>``, which writes the variant VCF, and the
FQ cutoff in variants2positions. A position removed by either one leaves no trace in the SNV
tables, because it never becomes a candidate at all.

This script reads the two VCFs a sample already produced and reports the difference. It is a
pure observer: nothing consumes its output, so it cannot change which SNVs are called.
"""
import os
import gzip
import logging
import argparse

from accusnv import log as accusnv_log

log = logging.getLogger('accusnv')

COLUMNS = 'contig\tcontig_pos\tref\talt\tqual\tDP4\tFQ\tgenotype\tremoved_by\tsample\n'


def parse_vcf(path):
    '''Every single-base substitution record in a VCF, keyed by (contig, position).'''
    records = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            field = line.rstrip('\n').split('\t')
            ref, alt = field[3], field[4]
            if alt in ('.', '<*>') or len(ref) != 1 or len(alt) != len(ref) or ',' in alt:
                continue
            info = dict(x.split('=', 1) for x in field[7].split(';') if '=' in x)
            records[(field[0], int(field[1]))] = {
                'ref': ref, 'alt': alt, 'qual': field[5], 'info': info,
                'genotype': field[9].split(':')[0] if len(field) > 9 else '.',
            }
    return records


def write_rejects(strain_vcf, variant_vcf, max_fq, out_path, sample):
    '''One row per position that had a substitution called but did not become a candidate.'''
    strain = parse_vcf(strain_vcf)
    variant = parse_vcf(variant_vcf)
    log.debug('%s: %d substitution records in the strain VCF, %d in the variant VCF',
              sample, len(strain), len(variant))

    rows = []
    for key, rec in sorted(strain.items()):
        if key not in variant:
            # bcftools view -v snps -q <min_af> dropped it. -q compares the genotype-derived
            # allele frequency (AC/AN), so any call that is not homozygous alt scores 0.5.
            reason = 'variant_min_af'
        elif float(rec['info'].get('FQ', 0)) >= max_fq:
            reason = 'max_fq'
        else:
            continue        # this one became a candidate
        rows.append((key[0], key[1], rec['ref'], rec['alt'], rec['qual'],
                     rec['info'].get('DP4', '.'), rec['info'].get('FQ', '.'),
                     rec['genotype'], reason, sample))

    with open(out_path, 'w') as f:
        f.write(COLUMNS)
        for row in rows:
            f.write('\t'.join(str(x) for x in row) + '\n')

    by_af = sum(1 for r in rows if r[8] == 'variant_min_af')
    log.info('%s: %d positions had a substitution called but did not become candidates '
             '(%d dropped by variant_min_af, %d by max_fq)',
             sample, len(rows), by_af, len(rows) - by_af)


def combine(inputs, out_path, group):
    '''Merge the per-sample reject files into one table for the group.'''
    with open(out_path, 'w') as out:
        out.write(COLUMNS)
        positions = set()
        for path in inputs:
            if not os.path.exists(path):
                continue
            with open(path) as f:
                f.readline()
                for line in f:
                    out.write(line)
                    field = line.split('\t')
                    positions.add((field[0], field[1]))
    log.info('Group %s: %s distinct positions were dropped before candidate selection; '
             'they are listed in %s', group, f'{len(positions):,}', os.path.basename(out_path))


def main():
    p = argparse.ArgumentParser(prog='AccuSNV upstream rejects',
                                description='Record positions dropped before candidate selection.')
    p.add_argument('--strain', help='Path to the sample .strain.vcf.gz')
    p.add_argument('--variant', help='Path to the sample .variant.vcf.gz')
    p.add_argument('-q', '--max_fq', type=float, default=-30, help='FQ cutoff (default -30)')
    p.add_argument('-o', '--output', required=True, help='Where to write the table')
    p.add_argument('--sample', default='sample', help='Sample name, for the log')
    p.add_argument('--combine', nargs='*', help='Per-sample reject files to merge instead')
    p.add_argument('--group', default='group', help='Group name, for the log')
    accusnv_log.add_args(p)
    args = p.parse_args()
    accusnv_log.setup('upstream_rejects', args.log)

    if args.combine is not None:
        combine(args.combine, args.output, args.group)
    else:
        write_rejects(args.strain, args.variant, args.max_fq, args.output, args.sample)


if __name__ == '__main__':
    main()
