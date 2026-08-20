"""Stage 5b: the interactive SNV dashboard (``snv_dashboard.html``).

Reads the group's ``snv_table_unfiltered.tsv`` and ``_snv_state.npz`` from stage 2, plus the
parsimony tree from stage 4, and writes one self-contained HTML page holding every candidate position
AccuSNV considered, kept or not, why each was kept or dropped, its per-sample read counts, and the tree.
``templates/dashboard.html`` is the page; this module only assembles the data it embeds.

It also writes one detailed bar chart per SNV into ``per_snv_barcharts/`` (which the page links
to from its "Open full barchart" button, skipped when there are more than 1000 SNVs) and, for
small runs, the QC heatmaps in ``diagnostic_figures/``.
"""
import os
import json
import zlib
import base64
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from accusnv import log as accusnv_log
from accusnv.downstream import snv

log = logging.getLogger('accusnv')


TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')

# The per-position checks, in the order they run, as (table column, short name, explanation).
# The short names are worded as the problem the check finds, not the property it requires, because
# clicking one in the dashboard filters to the positions that failed it.
# Each cell is 1 (this check is what removed the variant), 0 (passed it) or -1 (an earlier
# check had already removed it, so this one never got to look).
# Several of these cutoffs are settable in pipeline.yaml, and the run that produced a given
# dashboard may not have used the shipped value, so every number below is labelled "default:".
# The column names in the first field are the literal table headers and must not be reworded.
CHECKS = [
    ('Qual_filter', 'Low variant quality',
     'Requires a bcftools FQ quality score above a cutoff (default: higher than 30). bcftools FQ '
     'scores are on a negative scale, so we take the negative of the score, meaning higher '
     'implies better quality.'),
    ('Cov_filter', 'Too few reads on both strands',
     'Requires a minimum number of forward reads and reverse reads covering this position '
     '(default: at least 5 forward reads and 5 reverse reads).'),
    ('MAF_filter', 'Mixed basecalls at site',
     'Requires most of the reads in a sample to be assigned to one base, rejecting highly mixed '
     'positions (default: at least 85%).'),
    ('Indel_filter', 'Only on reads with insertions/deletions',
     'Requires few of the reads covering this position to have contained an insertion or deletion '
     'anywhere in the read, which could make the surrounding alignment unreliable '
     '(default: fewer than 33%).'),
    ('MFAS_filter', 'Too few samples with coverage at site',
     'Requires at least a set percentage of samples to have a confident base call at this '
     'position, so there are enough samples to compare (default: no minimum).'),
    ('MMCP_filter', 'Low median depth across samples',
     'Requires a minimum median read depth across all samples (default: at least 5x).'),
    ('CPN_filter', 'Unusually high or low coverage across samples',
     'Requires the read depth to stay close to the whole-genome read depth, to avoid SNVs arising '
     'from mismapped repeat regions (default: less than 4x on average, and less than 7x for any '
     'one sample).'),
    ('Edge_filter', 'At a contig end, with reads on one strand only',
     'Rejects positions that are close to the ends of contigs (default <100 bp) and have strong '
     'strand bias (default: <30% of reads are one strand).'),
    ('Gap_filter', 'Unusual coverage compared to other samples',
     'Requires the samples carrying the variant not to drop far below their genome-wide median '
     'depth while the reference samples stay high, the signature of a gap or mismapping in the '
     'alignment (default: below 5% while reference samples stay above 20%).'),
]

# SNV type letter -> (badge wording, the sentence behind it).
EFFECTS = {
    'N': ['Non-synonymous', 'Changes protein amino acid sequence'],
    'S': ['Synonymous', 'Does not change protein amino acid sequence'],
    'P': ['Upstream of gene', 'The variant sits just before a gene (default: within 250 bases), where it may affect how much of that gene is made.'],
    'I': ['Intergenic', 'The variant is not inside a gene or just upstream of one.'],
    'U': ['Not determined', 'The gene here could not be translated, so the effect on the protein is unknown.'],
}


def pack(array, dtype):
    '''Base64 of a deflate stream of `array`, which the page unpacks with DecompressionStream.'''
    return base64.b64encode(zlib.compress(np.asarray(array, dtype).tobytes(), 6)).decode()


def removal_reasons(row, ambig_cutoff):
    '''
    Why this SNV is not in the final table, in plain words. Empty when the SNV was kept.

    The checks run in order and only the first one to remove a position reports 1, so at most
    one check is named here; the model's own verdict and the ambiguity cutoff are added on top.
    '''
    if row['Pred_label'] != 0:
        return []
    why = [help for column, _, help in CHECKS if row[column] == 1]
    # Fix_filter is not in CHECKS: positions no ingroup sample varies at are dropped before the
    # table is written, so it should never fire. Explain it rather than show a bare row if it does.
    if row['Fix_filter'] == 1:
        why.append('No sample carried a confident base differing from the inferred ancestor at '
                   'sufficient mutation quality, so there was no difference to report.')
    # Gap_filter covers two conditions; say which one applied rather than the gap wording.
    if row['Gap_filter'] == 1 and row['Gap_reason'] == 'no_variation':
        why = ['Fewer than two different bases were called across the samples here, so there '
               'was no difference to compare.' if 'align cleanly' in w or 'genome-wide median' in w
               else w for w in why]
    if not why and row['WideVariant_pred'] == 0 and str(row['CNN_pred_raw']) != '0':
        why.append('The WideVariant checks found no difference to report here.')
    if str(row['CNN_pred_raw']) == '0':
        why.append('The AccuSNV model judged the read evidence to be a sequencing or mapping '
                   'artifact rather than a real variant.')
    elif str(row['CNN_pred_raw']) == 'skip':
        why.append('The AccuSNV model could not score this position, so it was left out of the '
                   'model half of the decision.')
    if row['Fraction_ambiguous_samples'] > ambig_cutoff:
        why.append('%.0f%% of samples do not have reads that agree on a single base here.'
                   % (100 * row['Fraction_ambiguous_samples']))
    return why


def whole(value):
    '''Annotation columns carry their numbers as "1151.0"; the page shows them as 1151.'''
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def reference_name(ref_genome, state):
    '''
    A name a reader recognises. References are usually a genome.fasta inside a directory named
    after the organism, and the stage 1 state only kept the file's basename, so prefer the
    directory when one was passed in.
    '''
    if ref_genome:
        path = os.path.abspath(ref_genome)
        return os.path.basename(path if os.path.isdir(path) else os.path.dirname(path))
    return str(state['ref_genome_name'])


def write_barcharts(table, counts, samples, out_dir):
    '''One PNG per SNV position into out_dir/per_snv_barcharts: stacked ATCG counts on forward
    then reverse reads for every sample, the detailed chart the page links to. Skips positions
    whose chart already exists.'''
    fdir = out_dir + '/per_snv_barcharts'
    os.makedirs(fdir, exist_ok=True)
    for i in range(len(table)):
        p = int(table['genome_pos'].iloc[i])
        png = fdir + '/p_' + str(p) + '_bar_chart.png'
        if os.path.exists(png):
            continue
        # One group of three bars per sample (forward counts, reverse counts, then a zero gap).
        c = counts[i]
        rows = np.stack((c[0, :4], c[0, 4:]))
        for s in range(1, len(samples)):
            rows = np.concatenate((rows, np.stack((np.zeros(4), c[s, :4], c[s, 4:]))))
        fig = plt.figure(20); fig.clf()
        ax = fig.subplots(1)
        pd.DataFrame(rows).plot(kind='bar', stacked=True, width=1, ax=ax)
        ax.legend(['A', 'T', 'C', 'G']); ax.set_ylabel('counts')
        plt.xticks(ticks=range(0, len(samples) * 3, 3), labels=samples)
        plt.title('p=' + str(p) + ', type=' + table['mutation_type'].iloc[i])
        fig.tight_layout()
        fig.savefig(png, dpi=200)


def make_calls_qc_heatmaps( my_cmt, my_calls, save_plots_bool, dir_save_fig, show ):
    '''
    Make three heatmaps of calls, quals, and coverage (over SNV positions for all samples).

    NOTES
    -----

        1. This function is intended to be used to help evaluate SNV quality
        after filtering. It should be used in combination with
        write_per_snv_barcharts.

    @author: Arolyn Conwill

    '''

    # Check to make sure dimensions of candidate mutation table and calls objects are consistent
    if my_cmt.num_samples != my_calls.num_samples:
        raise Exception("Error! ")
    if my_cmt.num_pos != my_calls.num_pos:
        raise Exception("")
    if my_calls.num_pos > 300:
        raise Exception("Error! Too many SNV positions (n=" + str(my_calls.num_pos) + ") to plot.")
        # Note: A better alternative would be to plot a subset, e.g. every x
        # SNVs, to give an overview of SNV quality. #TODO

    # QC heatmap of calls
    masked_array_calls = np.ma.masked_where( my_calls.calls==0, my_calls.calls )
    # Plot
    fig1, ax1 = plt.subplots()
    im = ax1.imshow( masked_array_calls, vmin=1, vmax=4, cmap=mpl.colormaps['rainbow'] )
    # Labeling
    ax1.set_title("QC Heatmap: Calls")
    plt.xlabel('SNV position')
    plt.ylabel('sample')
    ax1.set_xticks(np.arange(len(my_calls.p)), labels=my_calls.p)
    ax1.set_yticks(np.arange(len(my_calls.sample_names)), labels=my_calls.sample_names)
    plt.setp(ax1.get_xticklabels(), rotation=90, ha="center")
    ax1.tick_params(axis='both', which='major', labelsize=6)
    # Colorbar
    ticks_for_colorbar = [ 1,2,3,4 ]
    tick_labels_for_colorbar = snv.ints2nts(ticks_for_colorbar)
    cbar = plt.colorbar(im, ticks=ticks_for_colorbar)
    cbar.ax.set_yticklabels(tick_labels_for_colorbar)
    cbar.ax.tick_params(labelsize=8)
    # #TODO: make colorbar discrete colors
    # Layout
    fig1.tight_layout()
    if show:
        plt.show()
    # Save figure
    if save_plots_bool:
        plt.savefig( dir_save_fig + "/snv_qc_heatmap_calls.png",dpi=400 )

    # QC heatmap of coverage
    masked_array_calls = np.ma.masked_where( my_cmt.coverage==0, my_cmt.coverage )
    # Plot
    fig2, ax2 = plt.subplots()
    im = ax2.imshow( masked_array_calls, vmin=0, vmax=np.max(my_cmt.coverage), cmap=mpl.colormaps['viridis'] )
    # Labeling
    ax2.set_title("QC Heatmap: Coverage")
    plt.xlabel('SNV position')
    plt.ylabel('sample')
    ax2.set_xticks(np.arange(len(my_calls.p)), labels=my_calls.p)
    ax2.set_yticks(np.arange(len(my_calls.sample_names)), labels=my_calls.sample_names)
    plt.setp(ax2.get_xticklabels(), rotation=90, ha="center")
    ax2.tick_params(axis='both', which='major', labelsize=6)
    # Colorbar
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('coverage', rotation=90)
    # Layout
    fig2.tight_layout()
    if show:
        plt.show()
    # Save figure
    if save_plots_bool:
        plt.savefig( dir_save_fig + "/snv_qc_heatmap_coverage.png" ,dpi=400)

    # QC heatmap of coverage
    masked_array_quals = np.ma.masked_where( my_cmt.quals<0, my_cmt.quals ) # no masking
    # Plot
    fig3, ax3 = plt.subplots()
    im = ax3.imshow( masked_array_quals, vmin=0, vmax=np.max(my_cmt.quals), cmap=mpl.colormaps['plasma'] )
    # Labeling
    ax3.set_title("QC Heatmap: Quals")
    plt.xlabel('SNV position')
    plt.ylabel('sample')
    ax3.set_xticks(np.arange(len(my_calls.p)), labels=my_calls.p)
    ax3.set_yticks(np.arange(len(my_calls.sample_names)), labels=my_calls.sample_names)
    plt.setp(ax3.get_xticklabels(), rotation=90, ha="center")
    ax3.tick_params(axis='both', which='major', labelsize=6)
    # Colorbar
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('quality (FQ)', rotation=90)
    # Layout
    fig3.tight_layout()
    if show:
        plt.show()
    # Save figure
    if save_plots_bool:
        plt.savefig( dir_save_fig + "/snv_qc_heatmap_quals.png", dpi=400)


def build_data(source, tree_path, group, ref_genome, out_dir, exclude_positions='',
               include_positions=''):
    '''Assemble everything the page embeds from the group's stage 2 and stage 4 outputs.'''
    table = pd.read_csv(source + '/snv_table_unfiltered.tsv', sep='\t', dtype=str).fillna('.')
    state = np.load(source + '/_snv_state.npz', allow_pickle=True)
    samples = [str(s) for s in state['cmt_sample_names']]

    # Fix_filter is not in CHECKS (the page does not show it) but removal_reasons still reads it.
    for column in ['Pred_label', 'CNN_pred', 'WideVariant_pred', 'Whether_recomb', 'Fix_filter'] + \
                  [column for column, _, _ in CHECKS]:
        table[column] = pd.to_numeric(table[column], errors='coerce').fillna(0).astype(int)
    for column in ['CNN_prob', 'Fraction_ambiguous_samples', 'quality']:
        table[column] = pd.to_numeric(table[column], errors='coerce').fillna(0.0)

    # Read counts, per-call quality and the calls that survived filtering, in table row order.
    # Counts are [A T C G on forward reads, then the same four on reverse reads].
    column_of = {p: i for i, p in enumerate(state['cmt_p'])}
    order = [column_of[p] for p in table['genome_pos'].astype(int)]
    counts = state['cmt_counts'][:, order, :].transpose(1, 0, 2)
    quals = state['cmt_quals'][:, order].T
    confident = np.array(list('NATCG'))[state['calls_filtered'][:, order].T]

    # Above 20 samples the calling stage demands cleaner agreement before it keeps a SNV the
    # model rejected; the dashboard explains removals with the same cutoff.
    ambig_cutoff = 0.1 if len(samples) > 20 else 0.25

    genes, snvs = {}, []
    for i, row in table.iterrows():
        gene = row['locus_tag'] if row['locus_tag'] not in ('.', 'nan', '') else None
        if gene and gene not in genes:
            genes[gene] = {'product': row['product'], 'protein': row['translation'],
                           'dna': row['sequence'], 'proteinId': row['protein_id'],
                           'strand': {'1.0': 'forward', '-1.0': 'reverse'}.get(row['strand'], '?')}
        calls = ''.join(row[s] if row[s] in 'ATCG' else 'N' for s in samples)
        # The page shows the alleles as one string, ancestor first, then each derived allele.
        alleles = row['ancestral_allele'] + row['derived_allele'].replace(',', '').replace('.', '')
        snvs.append({
            'p': int(row['genome_pos']), 'contig': row['contig'], 'cpos': whole(row['contig_pos']),
            'kept': int(row['Pred_label'] != 0), 'cnn': int(row['CNN_pred']),
            'wd': int(row['WideVariant_pred']), 'prob': round(float(row['CNN_prob']), 4),
            'checks': [int(row[column]) for column, _, _ in CHECKS],
            'recomb': int(row['Whether_recomb']), 'ambig': round(float(row['Fraction_ambiguous_samples']), 4),
            'why': removal_reasons(row, ambig_cutoff),
            'removedBy': row['Removed_by'], 'cnnRaw': str(row['CNN_pred_raw']),
            'probRaw': str(row['CNN_prob_raw']), 'gapReason': row['Gap_reason'],
            'effect': row['mutation_type'], 'change': row['aa_mutation'].replace(',', ', '),
            'anc': row['ancestral_allele'], 'alleles': alleles,
            'gene': gene, 'ntPos': whole(row['gene_nt_position']), 'aaPos': whole(row['aa_pos']),
            'codons': row['possible_codons_at_site'].split(),
            'orientation': 'reverse_complement' if str(row['strand']).startswith('-') else 'forward',
            'calls': calls, 'confident': ''.join(confident[i]),
            'nVariant': sum(c != row['ancestral_allele'] and c != 'N' for c in calls),
            'quality': round(float(row['quality']), 1),
            'depth': int(np.median(counts[i].sum(axis=1))),
        })

    # A bar chart per SNV, unless there are so many the render would dominate the run; the page
    # tells the user when they have been skipped. The QC heatmaps only cover small runs.
    made_barcharts = len(table) <= 1000
    if made_barcharts:
        write_barcharts(table, counts, samples, out_dir)
    if 0 < len(table) < 300:
        try:
            st = snv.rebuild_state(source + '/_snv_state.npz', ref_genome,
                                   snv.read_positions_file(exclude_positions),
                                   snv.read_positions_file(include_positions))
            figdir = out_dir + '/diagnostic_figures'
            os.makedirs(figdir, exist_ok=True)
            make_calls_qc_heatmaps(st.my_cmt_goodpos_all, st.my_calls_goodpos_all, True, figdir, False)
        except Exception as e:
            log.warning('Skipping the QC heatmaps for this group (%s). The dashboard itself is '
                        'unaffected.', e)

    tree = ''
    if tree_path and os.path.exists(tree_path) and os.path.getsize(tree_path) > 0:
        tree = open(tree_path).read().strip()

    return {
        'group': group, 'reference': reference_name(ref_genome, state), 'samples': samples,
        'outgroup': [bool(x) for x in state['cmt_in_outgroup']], 'tree': tree, 'barcharts': made_barcharts,
        'checks': [{'name': name, 'column': column, 'help': help} for column, name, help in CHECKS],
        'effects': EFFECTS, 'genes': genes, 'snvs': snvs,
        'counts': pack(np.clip(counts, 0, 65535), '<u2'),
        'quals': pack(np.clip(quals, 0, 65535), '<u2'),
    }


def write_dashboard(data, out_path):
    '''Fill templates/dashboard.html with the data and the bundled tree viewer.'''
    page = open(os.path.join(TEMPLATES, 'dashboard.html')).read()
    page = page.replace('"__DATA__"', json.dumps(data, separators=(',', ':')))
    page = page.replace('/*__PEARTREE__*/', open(os.path.join(TEMPLATES, 'peartree.bundle.min.js')).read())
    with open(out_path, 'w') as f:
        f.write(page)


def parse_args():
    p = argparse.ArgumentParser(prog='AccuSNV dashboard (stage 5b)')
    p.add_argument('-i', '--input_dir', dest='input_dir', required=True,
                   help="Group dir holding the SNV tables and _snv_state.npz from stage 2")
    p.add_argument('-o', '--output_dir', dest='output_dir', required=True,
                   help="Where to write snv_dashboard.html")
    p.add_argument('-g', '--group', default='', help="Group name shown in the page header")
    p.add_argument('-t', '--tree', default='', help="Newick tree to show in the page (optional)")
    p.add_argument('--exclude_positions', default='',
                   help="File of genome_pos values, one per line, to leave out of the SNV set.")
    p.add_argument('--include_positions', default='',
                   help="File of genome_pos values, one per line, to keep whatever the filters said.")
    accusnv_log.add_args(p)
    p.add_argument('-r', '--rer', dest='ref_genome', default='',
                   help="Reference genome FASTA (or its directory); only its name is shown (optional)")
    return p.parse_args()


def main():
    args = parse_args()
    accusnv_log.setup('dashboard', args.log)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = args.output_dir + '/snv_dashboard.html'
    group = args.group or os.path.basename(os.path.abspath(args.output_dir))

    table = args.input_dir + '/snv_table_unfiltered.tsv'
    if bool(np.load(args.input_dir + '/_snv_state.npz', allow_pickle=True)['fast_path']) \
            or not os.path.exists(table) or os.path.getsize(table) == 0:
        log.warning('Group %s: no SNV table to show, so the dashboard is written empty.', group)
        write_dashboard({'group': group, 'reference': '', 'samples': [], 'outgroup': [],
                         'tree': '', 'barcharts': False, 'checks': [], 'effects': EFFECTS,
                         'genes': {}, 'snvs': [], 'counts': '', 'quals': ''}, out_path)
        return

    data = build_data(args.input_dir, args.tree, group, args.ref_genome, args.output_dir,
                      args.exclude_positions, args.include_positions)
    write_dashboard(data, out_path)
    log.info('Group %s: dashboard written with %s SNVs across %d samples%s -- %s', group,
             f"{len(data['snvs']):,}", len(data['samples']),
             '' if data['tree'] else ' (no tree panel)', os.path.normpath(out_path))


if __name__ == '__main__':
    main()
