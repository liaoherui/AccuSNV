# -*- coding: utf-8 -*-
"""
---Gathers everything together for candidate_mutation_table---
NOTE: Still reads in many *.mat files etc. Further purging of matlab necessary!
Output:
# path_candidate_mutation_table: where to write
# candidate_mutation_table.mat, ex. results/candidate_mutation_table.mat
---
# Inputs (changed to argparse usage):
     path_to_p_file: where to find all_positions.mat
     path_to_sample_names_file: where to find text file with sample names
         (space delimited)
     path_to_outgroup_boolean_file: where to find text file with outgroup
         booleans (space delimited, 1=outgroup, 0=not)
    path_to_list_of_quals_files: where to find text file with list of
       quals.mat files for each sample (space delimited)
     path_to_list_of_diversity_files: where to find text file with list of
       diversity.mat files for each sample (space delimited)
# Output:
     path_candidate_mutation_table: where to write
     candidate_mutation_table.mat, ex. results/candidate_mutation_table.mat
# Note: Input paths may be absolute or relative to the current working directory.
## Version history
     This is adapted from TDL's build_mutation_table_master_smaller_file_size_backup.m
  #   Arolyn, 2018.12.19: This script was written as part of the transition to snakemake.
          It performs the part of the case step that gathers data for
         Quals and counts and saves candidate_mutation_table.mat
  #   Arolyn, 2019.02.12: Added another matlab variable that stores indel
          statistics called 'indel_counter'.
  #   Tami, 2019.12.12: Converted into python and also added ability save coverage data
  #   Felix: 2020.01-04: Continous Debugged and adapted script for streamlined Snakemake implementation.
  #                      Added argparse for proper argument parsing and optional coverage matrix build.
  #   Evan: 2022.02.07: Changed to work with fully python version of lab pipeline
  #   Arolyn, 2022.10.17:
        Fixed coverage matrix generation
        Changed inputs for coverage matrices from booleans to output filenames
        Changed coverage matrices to numpy arrays instead of scipy sparse matrices (because the matrices shouldn't be sparse)
  #   Arolyn, 2022.10.23:
        Fixed bug where indices of indel statistics were not correct (line 173): 8:10 -> 38:40
"""

''' load libraries '''
import numpy as np
import pickle
import logging
import os
import gzip

log = logging.getLogger('accusnv')

def build_candidate_mutation_table(path_to_p_file, SampleNames, outgroup_bool, quals_files, diversity_files,
         path_to_candidate_mutation_table, path_to_cov_mat_raw, path_to_cov_mat_norm,
         flag_cov_raw, flag_cov_norm, dim, group='group'):
    # p: positions on genome that are candidate SNPs
    with open(path_to_p_file, 'rb') as f:
        p = pickle.load(f)

    # SampleNames: list of names of all samples
    numSamples = len(SampleNames)  # save number of samples

    ## in_outgroup: booleans for whether or not each sample is in the outgroup
    in_outgroup = np.asarray([s == 1 for s in outgroup_bool], dtype=bool).reshape(1, len(outgroup_bool))

    log.info('Group %s: building the candidate mutation table -- read counts and call qualities for '
             '%d samples (%d outgroup) at all %s candidate positions',
             group, numSamples, int(in_outgroup.sum()), f'{len(p):,}')

    ## Quals: quality score (relating to sample purity) at each position for all samples
    # Make Quals
    Quals = np.zeros((len(p), numSamples), dtype='int')  # initialize
    for i in range(numSamples):
        log.debug('Group %s: reading call qualities for %s from %s', group, SampleNames[i], quals_files[i])
        with gzip.open(quals_files[i], "rb") as f:
            quals_temp = pickle.load(f)
        quals = quals_temp.flatten()
        Quals[:, i] = quals[p - 1]  # -1 convert position to index

    # Diversity data
    with gzip.open(diversity_files[0], 'rb') as f:
        data = np.array(pickle.load(f))
    size = np.shape(data)
    GenomeLength = size[0]

    # Make counts and coverage at the same time
    counts = np.zeros((dim, len(p), numSamples), dtype='uint')  # initialize
    all_coverage_per_bp = np.zeros((GenomeLength, numSamples), dtype='uint')
    indel_counter = np.zeros((2, len(p), numSamples), dtype='uint')


    for i in range(numSamples):
        log.debug('Group %s: reading read counts for %s from %s', group, SampleNames[i], diversity_files[i])
        with gzip.open(diversity_files[i], 'rb') as f:
            data = np.array(pickle.load(f))
        counts[:, :, i] = data[p - 1, 0:dim].T  # -1 convert position to index

        if flag_cov_raw:
            # 0:8 not 0:dim -- only the first 8 columns are read counts (see pileup2diversity)
            np.sum(data[:, 0:8], axis=1, out=all_coverage_per_bp[:, i])

        indel_counter[:, :, i] = data[p - 1, 38:40].T  # Num reads supporting indels and reads supporting deletions
        # -1 convert position to index


    # Normalize coverage by sample and then position; ignore /0 ; turn resulting inf to 0

    if flag_cov_norm:
        with np.errstate(divide='ignore', invalid='ignore'):
            # 1st normalization
            array_cov_norm = (all_coverage_per_bp - np.mean(all_coverage_per_bp, axis=1, keepdims=True)) / np.std(
                all_coverage_per_bp, axis=1,
                keepdims=True)  # ,keepdims=True maintains 2D array (second dim == 1), necessary for braodcasting
            array_cov_norm[~np.isfinite(array_cov_norm)] = 0

            # 2nd normalization
            array_cov_norm = (array_cov_norm - np.mean(array_cov_norm, axis=0, keepdims=True)) / np.std(array_cov_norm,
                                                                                                        axis=0,
                                                                                                        keepdims=True)  # ,keepdims=True maintains 2D array (second dim == 1), necessary for braodcasting
            array_cov_norm[~np.isfinite(array_cov_norm)] = 0

            # Scale and convert to int to save space
            array_cov_norm_scaled = (np.round(array_cov_norm, 3) * 1000).astype('int64')

    # Reshape & save matrices

    Quals = Quals.transpose() #quals: num_samples x num_pos
    p = p.transpose() #p: num_pos
    counts = counts.swapaxes(0,2) #counts: num_samples x num_pos x 8
    in_outgroup = in_outgroup.flatten() #in_outgroup: num_samples
    indel_counter = indel_counter.swapaxes(0,2) #indel_counter: num_samples x num_pos x 2


    outdir = os.path.dirname(path_to_candidate_mutation_table)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if flag_cov_raw:
        log.debug('Group %s: writing the genome-wide coverage matrix to %s', group, path_to_cov_mat_raw)
        all_coverage_per_bp = all_coverage_per_bp.transpose() #all_coverage_per_bp: num_samples x num_pos
        with open(path_to_cov_mat_raw, 'wb') as f:
            np.savez_compressed(path_to_cov_mat_raw, all_coverage_per_bp=all_coverage_per_bp)

    if flag_cov_norm:
        log.debug('Group %s: writing the normalized coverage matrix to %s', group, path_to_cov_mat_norm)
        array_cov_norm_scaled = array_cov_norm_scaled.transpose() #array_cov_norm_scaled: num_samples x num_pos
        with open(path_to_cov_mat_norm, 'wb') as f:
            np.savez_compressed(path_to_cov_mat_norm, array_cov_norm_scaled=array_cov_norm_scaled)

    CMT = {'sample_names': SampleNames,
           'p': p,
           'counts': counts,
           'quals': Quals,
           'in_outgroup': in_outgroup,
           'indel_counter': indel_counter,
           }

    file_path = path_to_candidate_mutation_table
    np.savez_compressed(file_path, **CMT)

    # Median depth over the candidate positions only: how much evidence the caller has to work with.
    depth = counts.sum(axis=2)
    log.info('Group %s: candidate mutation table written -- %d samples x %s positions, '
             'median depth %.1fx at those positions (range %dx-%dx across samples)',
             group, numSamples, f'{len(p):,}', np.median(depth) if depth.size else 0,
             depth.sum(axis=1).min() // max(len(p), 1) if depth.size else 0,
             depth.sum(axis=1).max() // max(len(p), 1) if depth.size else 0)
    log.debug('Group %s: wrote %s', group, path_to_candidate_mutation_table)