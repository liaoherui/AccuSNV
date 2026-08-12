"""Stage 1: run the CNN + WideVariant filters and select candidate SNV positions.

This is the only stage that runs the CNN (via ``cnn_pred.CNN_predict``). It writes
``_snv_state.npz`` (the hand-off the later stages read), the raw CNN scores
``snv_table_cnn_raw.tsv``, and ``snv_table_filtered_tmp.tsv``, which stage 2 folds into
``snv_table_unfiltered.tsv`` and then deletes. That table carries a row for every candidate
position, including the ones neither the CNN nor the filters called, so every removal has a
recorded reason (``Removed_by``). For very large inputs (more candidate positions
than ``--fast_mode_positions``) it takes a fast path that writes
``candidate_mutation_table_final.npz`` directly and signals the rest to skip.

Transcribed from new_snv_script.py lines ~411-1046.
"""
import os
import copy
import logging
import argparse

import numpy as np

from accusnv import log as accusnv_log
from accusnv.downstream import snv
from accusnv.accusnv import cnn_pred as cnn

log = logging.getLogger('accusnv')


def parse_args():
    p = argparse.ArgumentParser(prog='AccuSNV calling (stage 1)',
                                description='Run CNN + filters and select candidate SNVs.')
    p.add_argument('-i', '--input_mat', dest='input_mat', type=str, required=True, help="Input mutation table (npz)")
    p.add_argument('-c', '--input_cov', dest='input_cov', type=str, help="Input coverage table (npz)")
    p.add_argument('-s', '--min_cov_for_filter_sample', dest='min_cov_samp', type=str,
                   help="Filter out low-quality samples (default 45; 100 keeps all).")
    p.add_argument('-v', '--min_cov_for_filter_pos', dest='min_cov', type=str,
                   help="Min fwd/rev reads per call for the filter module (default 5).")
    p.add_argument('-e', '--excluse_samples', dest='exclude_samp', type=str,
                   help="Samples to exclude (e.g. -e S1,S2) or an SNV-count cutoff (e.g. -e 1000).")
    p.add_argument('-m', '--recomb', dest='recomb', type=str, default='1',
                   help="Run recombination detection (1) or not (0). Default: 1. Recombinant SNVs are "
                        "flagged (Whether_recomb) but kept in all tables; they are excluded from dN/dS, "
                        "tree building and dMRCA.")
    p.add_argument('-r', '--rer', dest='ref_genome', type=str, help="Reference genome FASTA (or its directory)")
    p.add_argument('-o', '--output_dir', dest='output_dir', type=str, help="The output dir")
    # Per-call SNV filter thresholds (surfaced to pipeline.yaml; defaults match prior hard-coded values).
    p.add_argument('--min_maf_call', dest='min_maf_call', type=float, default=0.85,
                   help="Minimum major-allele frequency for a confident call (default 0.85).")
    p.add_argument('--min_qual_call', dest='min_qual_call', type=int, default=30,
                   help="Minimum per-call quality (default 30).")
    p.add_argument('--max_indel_frac', dest='max_indel_frac', type=float, default=0.33,
                   help="Max fraction of reads supporting an indel before a call is dropped (default 0.33).")
    p.add_argument('--min_mut_qual', dest='min_mut_qual', type=float, default=1,
                   help="Minimum mutation quality to keep a SNV position (default 1).")
    # Across-sample position filters (MFAS_filter, MMCP_filter, CPN_filter in the SNV tables).
    p.add_argument('--max_frac_ambiguous_samples', dest='max_frac_ambiguous_samples', type=float, default=1,
                   help="Drop a position when more than this fraction of samples have no base call "
                        "(default 1, which never drops anything).")
    p.add_argument('--min_median_coverage_position', dest='min_median_coverage_position', type=float, default=5,
                   help="Drop a position whose median read depth across samples is below this (default 5).")
    p.add_argument('--max_mean_copynum', dest='max_mean_copynum', type=float, default=4,
                   help="Drop a position whose depth averages more than this many times the "
                        "genome-wide median depth (default 4).")
    p.add_argument('--max_max_copynum', dest='max_max_copynum', type=float, default=7,
                   help="Drop a position exceeding this multiple of the genome-wide median depth "
                        "in any single sample (default 7).")
    p.add_argument('--fast_mode_positions', dest='fast_mode_positions', type=int, default=100000,
                   help="Take the fast path above this many candidate SNV positions (default 100000).")
    # When the CNN rejects a SNV that the WideVariant filters accepted, the SNV is only rescued
    # if at most this fraction of samples have mixed read support (Fraction_ambiguous_samples).
    # Larger cohorts get the stricter cutoff.
    p.add_argument('--rebuild_sample_count', dest='rebuild_sample_count', type=int, default=20,
                   help="Cohort size above which the stricter ambiguity cutoff applies (default 20).")
    p.add_argument('--rebuild_cutoff_many', dest='rebuild_cutoff_many', type=float, default=0.1,
                   help="Ambiguity cutoff when sample count > rebuild_sample_count (default 0.1).")
    p.add_argument('--rebuild_cutoff_few', dest='rebuild_cutoff_few', type=float, default=0.25,
                   help="Ambiguity cutoff when sample count <= rebuild_sample_count (default 0.25).")
    p.add_argument('--recomb_distance', dest='recomb_distance', type=int, default=1000,
                   help="Max bp between SNV pairs tested for recombination (default 1000).")
    p.add_argument('--recomb_corr', dest='recomb_corr', type=float, default=0.75,
                   help="Allele-correlation threshold to flag recombinant positions (default 0.75).")
    p.add_argument('--group', default='group', help="Group name, for the log")
    accusnv_log.add_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    accusnv_log.setup('calling', args.log)
    group = args.group
    input_mat = args.input_mat
    input_cov = args.input_cov
    min_cov_samp = int(args.min_cov_samp) if args.min_cov_samp else 45
    min_cov_filt = int(args.min_cov) if args.min_cov else 5
    refg = args.ref_genome
    odir = args.output_dir
    exclude_samp = args.exclude_samp or ''
    if exclude_samp.lower() in ('null', 'none'):
        exclude_samp = ''      # pipeline.yaml's way of saying "exclude nothing"
    run_recomb = str(args.recomb) == '1'

    # Resolve which samples to exclude. A numeric cutoff means exclude any sample
    # with more than that many SNVs, otherwise it is a comma-separated name list.
    if snv.is_digit(exclude_samp):
        exclude_samp = int(exclude_samp)
    os.makedirs(odir, exist_ok=True)
    # The rule declares this as an output, so it has to exist even on the paths that return
    # early (no candidate positions, or fast mode). Rewritten with content once calling gets there.
    snv.write_invariant_positions([], None, None, odir)

    dcs = snv.check_snv(input_mat, odir)
    log.info('Group %s: SNV calling starting for %d samples', group, len(dcs))
    log.info('Group %s: rough SNV count per sample before filtering -- %s', group,
             ', '.join(f'{s} {n}' for s, n in sorted(dcs.items(), key=lambda kv: -kv[1])))
    if snv.is_digit(str(exclude_samp)):
        samples_to_exclude = [s for s in dcs if dcs[s] >= exclude_samp]
        log.info('Group %s: excluding any sample with %d or more rough SNVs', group, exclude_samp)
    else:
        if not exclude_samp == '':
            samples_to_exclude = exclude_samp.split(',')
        else:
            samples_to_exclude = [""]
    named = [s for s in samples_to_exclude if s]
    log.info('Group %s: %s', group,
             f'excluding {len(named)} sample(s) at the user\'s request: ' + ', '.join(named)
             if named else 'no samples excluded by name or SNV count')

    dir_output = odir
    data_file_cmt = input_mat
    data_file_cov = input_cov
    dir_ref_genome = refg
    ref_genome_name = snv.search_ref_name(refg)

    # ---- Run the CNN ----
    log.info('Group %s: scoring every candidate position with the AccuSNV CNN', group)
    cnn_pos, cnn_pred, cnn_prob, dgap, dgap_reason = cnn.CNN_predict(
        data_file_cmt, data_file_cov, odir, samples_to_exclude, min_cov_samp)
    log.info('Group %s: CNN scored %s positions and called %s of them real SNVs',
             group, f'{len(cnn_pos):,}', f'{int(np.sum(cnn_pred == 1)):,}')
    dlab = dict(zip(cnn_pos, cnn_pred))
    dprob = dict(zip(cnn_pos, cnn_prob))

    # Human-readable CNN result (just the AccuSNV CNN output; nothing downstream of it).
    with open(odir + '/snv_table_cnn_raw.tsv', 'w') as f:
        f.write('genome_pos\tCNN_pred\tCNN_prob\n')
        for pos, lab, prob in zip(cnn_pos, cnn_pred, cnn_prob):
            f.write(f'{pos}\t{lab}\t{prob}\n')

    # ---- Fast path for very large position sets ----
    if len(cnn_pos) > args.fast_mode_positions:
        [quals, p, counts, in_outgroup, sample_names, indel_counter] = \
            snv.read_candidate_mutation_table_npz(data_file_cmt)
        if not len(in_outgroup) == len(sample_names):
            in_outgroup = np.array([False] * len(sample_names))
        my_cmt = snv.cmt_data_object(sample_names, in_outgroup, p, counts, quals, indel_counter)
        log.warning('Group %s: %s candidate positions is more than --fast_mode_positions (%s), so '
                    'AccuSNV is switching to fast mode: the CNN calls are kept as they are and '
                    'annotation, dN/dS, trees and the dashboard are all skipped',
                    group, f'{len(cnn_pos):,}', f'{args.fast_mode_positions:,}')

        samples_to_exclude_bool = np.array([x in samples_to_exclude for x in sample_names])
        keep_p = np.isin(my_cmt.p, cnn_pos)
        my_cmt_zero_rebuild = copy.deepcopy(my_cmt)
        my_cmt_zero_rebuild.filter_positions(keep_p)
        label = np.array([dlab[pos] == 1 for pos in my_cmt_zero_rebuild.p])
        prob = np.array([dprob[pos] for pos in my_cmt_zero_rebuild.p])
        recomb = np.array([False] * len(my_cmt_zero_rebuild.p))
        quals_new = my_cmt_zero_rebuild.quals * -1
        new_cmt = {
            'sample_names': my_cmt_zero_rebuild.sample_names, 'p': my_cmt_zero_rebuild.p,
            'counts': my_cmt_zero_rebuild.counts, 'quals': quals_new,
            'in_outgroup': my_cmt_zero_rebuild.in_outgroup, 'indel_counter': my_cmt_zero_rebuild.indel_stats,
            'prob': prob, 'label': label, 'recomb': recomb, 'samples_exclude_bool': samples_to_exclude_bool,
        }
        np.savez_compressed(odir + '/candidate_mutation_table_final.npz', **new_cmt)
        np.savez_compressed(odir + '/_snv_state.npz', fast_path=True, too_many_positions=True)
        log.info('Group %s: fast mode finished -- %s SNVs written to '
                 'candidate_mutation_table_final.npz', group, f'{int(label.sum()):,}')
        return

    # ---- Build candidate mutation table, reference, coverage objects ----
    [quals, p, counts, in_outgroup, sample_names, indel_counter] = \
        snv.read_candidate_mutation_table_npz(data_file_cmt)
    if not len(in_outgroup) == len(sample_names):
        in_outgroup = np.array([False] * len(sample_names))
    my_cmt = snv.cmt_data_object(sample_names, in_outgroup, p, counts, quals, indel_counter)
    my_rg = snv.reference_genome_object(dir_ref_genome)
    my_cov = snv.cov_data_object(snv.read_cov_mat_npz(data_file_cov), my_cmt.sample_names,
                                 my_rg.genome_length, my_rg.contig_starts, my_rg.contig_names)
    log.info('Group %s: reference %s is %s bp in %d contig(s); %s candidate positions in %d samples '
             '(%d outgroup)', group, os.path.basename(snv.ref_directory(refg)),
             f'{my_rg.genome_length:,}', len(my_rg.contig_names),
             f'{len(my_cmt.p):,}', my_cmt.num_samples, int(np.sum(my_cmt.in_outgroup)))

    # ---- Exclude any samples flagged above ----
    my_cmt.filter_samples(~np.array([x in samples_to_exclude for x in my_cmt.sample_names]))
    my_cov.filter_samples(~np.array([x in samples_to_exclude for x in my_cov.sample_names]))
    my_cmt_zero = copy.deepcopy(my_cmt)
    dpt = {}  # per-position filter results, keyed by filter stage

    # No candidate positions at all: nothing to filter, plot, or annotate. This happens when
    # every bcftools candidate was a multi-allelic sequencing-error artifact (e.g. very high
    # error rate) and got dropped upstream. Write empty results and stop, rather than feed a
    # zero-width matrix into the filter/plot helpers (which raise on an empty axis).
    if len(my_cmt.p) == 0:
        log.warning('Group %s: no candidate SNV positions survived upstream filtering, so there is '
                    'nothing to call. Writing empty tables and skipping the rest of the analysis.', group)
        with open(odir + '/snv_table_filtered_tmp.tsv', 'w') as f:
            f.write(snv.TABLE_COLUMNS)
        empty = np.array([], dtype=int)
        np.savez_compressed(
            odir + '/candidate_mutation_table_final.npz',
            sample_names=my_cmt_zero.sample_names, p=empty, counts=my_cmt_zero.counts[:, :0, :],
            quals=my_cmt_zero.quals[:, :0], in_outgroup=my_cmt_zero.in_outgroup,
            indel_counter=my_cmt_zero.indel_stats[:, :0, :], prob=[], label=empty.astype(bool),
            recomb=empty.astype(bool),
            samples_exclude_bool=np.array([x in samples_to_exclude for x in sample_names]))
        np.savez_compressed(odir + '/_snv_state.npz', fast_path=True, too_many_positions=False)
        return

    # ---- Filter basecalls ----
    my_calls = snv.calls_object(my_cmt)

    filter_parameter_sample_across_sites = {
        'min_average_coverage_to_include_sample': 0,
        'max_frac_Ns_to_include_sample': 1,
    }
    filter_parameter_site_per_sample = {
        'min_major_nt_freq_for_call': args.min_maf_call,
        'min_cov_per_strand_for_call': min_cov_filt,
        'min_qual_for_call': args.min_qual_call,
        'max_frac_reads_supporting_indel': args.max_indel_frac,
    }
    filter_parameter_site_across_samples = {
        'max_fraction_ambiguous_samples': args.max_frac_ambiguous_samples,
        'min_median_coverage_position': args.min_median_coverage_position,
        'max_mean_copynum': args.max_mean_copynum,
        'max_max_copynum': args.max_max_copynum,
    }

    log.info('Group %s: filtering basecalls -- keeping calls with quality >= %s, >= %s forward and '
             'reverse reads, major allele at >= %.0f%% of reads, and under %.0f%% of reads supporting '
             'an indel', group, args.min_qual_call, min_cov_filt,
             100 * args.min_maf_call, 100 * args.max_indel_frac)

    # Drop low-coverage samples
    [low_cov_samples, goodsamples_coverage] = snv.filter_samples_by_coverage(
        my_cov.get_median_cov_of_chromosome(),
        filter_parameter_sample_across_sites['min_average_coverage_to_include_sample'],
        my_cov.sample_names, True, dir_output)
    my_cmt.filter_samples(goodsamples_coverage)
    my_cov.filter_samples(goodsamples_coverage)
    my_calls.filter_samples(goodsamples_coverage)

    # Keep unfiltered calls for ancestor inference and raw nucleotide output.
    my_calls_raw_for_ancestor = my_calls.copy()

    # Which positions the ingroup samples disagree at, judged on the unfiltered calls. It has to
    # be read before any filter runs: every filter works by removing the evidence for one allele,
    # so a position that a filter fired on is monomorphic afterwards and would be mistaken for one
    # of these. Positions the ingroup agrees on are differences from the reference genome, not
    # SNVs, and no filter downstream can make anything of them, so they are left out of the SNV
    # tables entirely rather than listed as rejected candidates (which they usually outnumber).
    variable_bool = snv.ingroup_variable(my_calls)
    log.info('Group %s: %s of %s candidate positions show no variation between the ingroup samples '
             '(they differ from the reference genome, but not from each other) and are left out of '
             'the SNV tables; they are listed in snv_table_invariant_positions.tsv', group,
             f'{int(np.sum(~variable_bool)):,}', f'{len(my_calls.p):,}')
    snv.write_invariant_positions(my_calls.p[~variable_bool], my_calls, my_rg, dir_output)

    # Quality
    my_calls_raw = copy.deepcopy(my_calls)
    my_calls.filter_calls_by_element(my_cmt.quals < filter_parameter_site_per_sample['min_qual_for_call'])
    tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T,
                                f'Qual_filter (call quality below {args.min_qual_call})')
    dpt['qual'] = dict(zip(my_calls.p, tokens))

    # Coverage (fwd/rev strands)
    my_calls_raw = copy.deepcopy(my_calls)
    my_calls.filter_calls_by_element(my_cmt.fwd_cov < filter_parameter_site_per_sample['min_cov_per_strand_for_call'])
    my_calls.filter_calls_by_element(my_cmt.rev_cov < filter_parameter_site_per_sample['min_cov_per_strand_for_call'])
    tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T,
                                f'Cov_filter (under {min_cov_filt} reads on either strand)')
    dpt['cov'] = dict(zip(my_calls.p, tokens))

    # Major allele frequency
    my_calls_raw = copy.deepcopy(my_calls)
    my_calls.filter_calls_by_element(my_cmt.major_nt_freq < filter_parameter_site_per_sample['min_major_nt_freq_for_call'])
    tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T,
                                f'MAF_filter (major allele under {args.min_maf_call:.0%} of reads)')
    dpt['maf'] = dict(zip(my_calls.p, tokens))

    # Indels
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_reads_supporting_indel = np.sum(my_cmt.indel_stats, axis=2) / my_cmt.coverage
        frac_reads_supporting_indel[~np.isfinite(frac_reads_supporting_indel)] = 0
    my_calls_raw = copy.deepcopy(my_calls)
    my_calls.filter_calls_by_element(frac_reads_supporting_indel > filter_parameter_site_per_sample['max_frac_reads_supporting_indel'])
    tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T,
                                f'Indel_filter (over {args.max_indel_frac:.0%} of reads supporting an indel)')
    dpt['indel'] = dict(zip(my_calls.p, tokens))


    # Positions that look iffy across samples
    my_calls_raw = copy.deepcopy(my_calls)
    my_calls.filter_calls_by_position(my_calls.get_frac_Ns_by_position() > filter_parameter_site_across_samples['max_fraction_ambiguous_samples'])
    tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T,
                                f'MFAS_filter (no call in over {args.max_frac_ambiguous_samples:.0%} of samples)')
    dpt['mfas'] = dict(zip(my_calls.p, tokens))

    snv.filter_histogram(my_calls.get_frac_Ns_by_position(),
                         filter_parameter_site_across_samples['max_fraction_ambiguous_samples'],
                         'Fraction Ns by position')

    my_calls_raw = copy.deepcopy(my_calls)
    my_calls.filter_calls_by_position(np.median(my_cmt.coverage, axis=0) < filter_parameter_site_across_samples['min_median_coverage_position'])
    tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T,
                                f'MMCP_filter (median depth across samples under {args.min_median_coverage_position}x)')
    dpt['mmcp'] = dict(zip(my_calls.p, tokens))

    my_calls_raw = copy.deepcopy(my_calls)
    copy_number_per_sample_per_pos = my_cmt.coverage / np.expand_dims(my_cov.get_median_cov_of_chromosome(), 1)
    copy_number_avg_per_pos = np.mean(copy_number_per_sample_per_pos, axis=0)
    copy_number_avg_per_pos[np.isnan(copy_number_avg_per_pos)] = 0
    my_calls.filter_calls_by_position(copy_number_avg_per_pos > filter_parameter_site_across_samples['max_mean_copynum'])
    copy_number_max_per_pos = np.max(copy_number_per_sample_per_pos, axis=0)
    copy_number_max_per_pos[np.isnan(copy_number_max_per_pos)] = 0
    my_calls.filter_calls_by_position(copy_number_max_per_pos > filter_parameter_site_across_samples['max_max_copynum'])
    tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T,
                                f'CPN_filter (depth over {args.max_mean_copynum}x the genome median on average, or {args.max_max_copynum}x in one sample)')
    dpt['cpn'] = dict(zip(my_calls.p, tokens))

    # Samples with too many ambiguous calls (histogram only; not applied to the table)
    pos_to_consider = my_calls.p[np.any(my_calls.calls, axis=0)]
    snv.filter_samples_by_ambiguous_basecalls(
        my_calls.get_frac_Ns_by_sample(pos_to_consider),
        filter_parameter_sample_across_sites['max_frac_Ns_to_include_sample'],
        my_calls.sample_names, my_calls.in_outgroup, True, dir_output)

    # ---- Infer ancestral alleles (from unfiltered calls) ----
    calls_ancestral, ancestor_source = snv.infer_ancestral_calls_from_raw_overlap(
        my_calls_raw_for_ancestor, my_rg, my_calls.p)

    # ---- Mutation quality ----
    calls_ingroup = my_calls.get_calls_in_sample_subset(np.logical_not(my_calls.in_outgroup))
    quals_ingroup = my_cmt.quals[np.logical_not(my_calls.in_outgroup), :]
    num_samples_ingroup = sum(np.logical_not(my_calls.in_outgroup))
    [mut_qual, mut_qual_samples] = snv.compute_mutation_quality(calls_ingroup, quals_ingroup)

    # ---- Recombination detection (on by default; -m 0 disables) ----
    # Recombinant positions are flagged only: they populate the Whether_recomb column / npz recomb
    # field and the recomb mask serialized below, but they are NOT removed from the WideVariant/CNN
    # SNV set or any table. Tree building and dMRCA exclude them via the mask in _snv_state.npz.
    if run_recomb:
        p_recombo, recombo_bool = snv.find_recombination_positions(
            my_calls, my_cmt, calls_ancestral, mut_qual, my_rg,
            args.recomb_distance, args.recomb_corr, True, dir_output)
        if len(p_recombo) > 0:
            with open(dir_output + '/snvs_from_recombo.csv', 'w') as f:
                for pr in p_recombo:
                    f.write(str(pr) + '\n')
    else:
        recombo_bool = np.array([False] * len(my_cmt.p))
    dpt['recomb'] = dict(zip(my_calls.p, recombo_bool))

    # ---- Determine high-quality SNV positions (WideVariant) ----
    filter_SNVs_not_N = (calls_ingroup != snv.nts2ints('N'))
    filter_SNVs_not_ancestral_allele = (calls_ingroup != np.tile(calls_ancestral, (num_samples_ingroup, 1)))
    filter_SNVs_quals_not_NaN = (np.tile(mut_qual, (num_samples_ingroup, 1)) >= args.min_mut_qual)
    fixedmutation = filter_SNVs_not_N & filter_SNVs_not_ancestral_allele & filter_SNVs_quals_not_NaN

    goodpos_bool = np.any(fixedmutation, axis=0)
    goodpos_idx = np.where(goodpos_bool)[0]
    tokens_final = snv.generate_tokens_last(tokens, goodpos_idx,
                                f'Fix_filter (no sample differs from the ancestral allele at mutation quality >= {args.min_mut_qual})')
    dpt['fix'] = dict(zip(my_calls.p, tokens_final))
    num_goodpos = len(goodpos_idx)
    log.info('Group %s: the WideVariant filters call %s SNVs', group, f'{num_goodpos:,}')

    # ---- Combine CNN + WideVariant ----
    goodpos_idx_cnn = cnn_pos[np.where(cnn_pred == 1)]
    goodpos_idx_wd = my_calls.p[goodpos_idx]
    all_p = np.sort(np.union1d(goodpos_idx_cnn, goodpos_idx_wd))
    log.info('Group %s: the CNN and the WideVariant filters agree on %s SNVs; %s were called only by '
             'the CNN and %s only by the filters', group,
             f'{len(np.intersect1d(goodpos_idx_cnn, goodpos_idx_wd)):,}',
             f'{len(np.setdiff1d(goodpos_idx_cnn, goodpos_idx_wd)):,}',
             f'{len(np.setdiff1d(goodpos_idx_wd, goodpos_idx_cnn)):,}')
    ambiguity_cutoff = (args.rebuild_cutoff_many if len(my_cmt_zero.sample_names) > args.rebuild_sample_count
                        else args.rebuild_cutoff_few)
    log.debug('Group %s: a filter-only SNV is kept if at most %s of samples have mixed read support '
              '(%d samples in this group)', group, ambiguity_cutoff, len(my_cmt_zero.sample_names))
    goodpos_bool, goodpos_bool_all = snv.generate_cnn_filter_table(
        all_p, goodpos_idx_wd, dpt, dlab, dprob, dir_output, my_cmt.p, dgap, my_cmt_zero, ambiguity_cutoff,
        dgap_reason, report_p=my_calls.p[variable_bool])
    goodpos_idx = np.where(goodpos_bool)[0]
    goodpos_idx_all = np.where(goodpos_bool_all)[0]
    num_goodpos = len(goodpos_idx)
    num_goodpos_all = len(goodpos_idx_all)
    log.info('Group %s: %s SNVs called in total by the CNN or the filters, %s of them by the CNN '
             'itself', group, f'{num_goodpos_all:,}', f'{num_goodpos:,}')

    # SNVs per sample: the number a user checks first, and the quickest way to spot a contaminated
    # or mislabelled isolate.
    calls_at_snvs = my_calls.calls[:, goodpos_idx_all]
    ancestral_at_snvs = np.tile(calls_ancestral[goodpos_idx_all], (my_calls.num_samples, 1))
    per_sample = np.sum((calls_at_snvs != ancestral_at_snvs) & (calls_at_snvs != 0), axis=1)
    log.info('Group %s: SNVs per sample -- %s', group,
             ', '.join(f'{name} {int(n)}' for name, n in zip(my_calls.sample_names, per_sample)))

    # Re-draw the too-many-Ns histogram over the final good positions (side-effect only;
    # the returned masks are unused). Overwrites the earlier snv_filter_sample_toomanyNs_hist.png.
    snv.filter_samples_by_ambiguous_basecalls(
        my_calls.get_frac_Ns_by_sample(goodpos_idx),
        filter_parameter_sample_across_sites['max_frac_Ns_to_include_sample'],
        my_calls.sample_names, my_calls.in_outgroup, True, dir_output)

    # ---- Build the AccuSNV result table (the CNN-stage deliverable / stopping point) ----
    # label/prob/recomb come from snv_table_filtered_tmp.tsv (cols 0/1/4/13); my_cmt_zero_rebuild
    # is the original input table reloaded.
    dk, dl, dr = {}, {}, {}
    with open(dir_output + '/snv_table_filtered_tmp.tsv') as f:
        f.readline()
        for line in f:
            ele = line.strip().split('\t')
            if len(ele) < 14:
                continue
            # The table also carries the candidates that neither the model nor the filters
            # called, so a reader can see why they went. They are not part of the SNV set.
            if ele[1] == '0' and ele[2] in ('0', 'skip') and ele[3] == '0':
                continue
            dk[int(ele[0])] = 0 if ele[4] == 'skip' else float(ele[4])
            if int(ele[1]) == 0:
                dl[int(ele[0])] = ''
            if int(ele[13]) == 1:
                dr[int(ele[0])] = int(ele[13])
    [q0, p0, c0, io0, sn0, ic0] = snv.read_candidate_mutation_table_npz(input_mat)
    if not len(io0) == len(sn0):
        io0 = np.array([False] * len(sn0))
    my_cmt_zero_rebuild = snv.cmt_data_object(sn0, io0, p0, c0, q0, ic0)
    samples_to_exclude_bool = np.array([x in samples_to_exclude for x in sn0])
    keep_p, prob, label, recomb = [], [], [], []
    for s in my_cmt_zero_rebuild.p:
        if s in dk:
            keep_p.append(True)
            prob.append(dk[s])
            label.append(s not in dl)
            recomb.append(s in dr)
        else:
            keep_p.append(False)
    my_cmt_zero_rebuild.filter_positions(np.array(keep_p))
    new_cmt = {
        'sample_names': my_cmt_zero_rebuild.sample_names, 'p': my_cmt_zero_rebuild.p,
        'counts': my_cmt_zero_rebuild.counts, 'quals': my_cmt_zero_rebuild.quals * -1,
        'in_outgroup': my_cmt_zero_rebuild.in_outgroup, 'indel_counter': my_cmt_zero_rebuild.indel_stats,
        'prob': prob, 'label': np.array(label), 'recomb': np.array(recomb),
        'samples_exclude_bool': samples_to_exclude_bool,
    }
    np.savez_compressed(dir_output + '/candidate_mutation_table_final.npz', **new_cmt)

    # ---- Serialize state for the downstream stages ----
    # With no good positions there is nothing for annotate/report/tree to do; reuse the
    # fast-path flag so they take their existing empty-output skip instead of indexing an
    # empty position array.
    np.savez_compressed(
        dir_output + '/_snv_state.npz',
        fast_path=(num_goodpos_all == 0), too_many_positions=False,
        cmt_sample_names=my_cmt.sample_names, cmt_in_outgroup=my_cmt.in_outgroup, cmt_p=my_cmt.p,
        cmt_counts=my_cmt.counts, cmt_quals=my_cmt.quals, cmt_indel_stats=my_cmt.indel_stats,
        calls_filtered=my_calls.calls, calls_ancestral=calls_ancestral, mut_qual=mut_qual,
        recomb_bool=recombo_bool,
        goodpos_bool_all=goodpos_bool_all, goodpos_idx_all=goodpos_idx_all,
        samples_to_exclude=np.array(samples_to_exclude, dtype=object),
        ref_genome_name=ref_genome_name, min_cov_filt=min_cov_filt,
        num_goodpos=num_goodpos, num_goodpos_all=num_goodpos_all,
    )
    log.info('Group %s: SNV calling finished; results handed to the annotation step in %s',
             group, dir_output)


if __name__ == '__main__':
    main()
