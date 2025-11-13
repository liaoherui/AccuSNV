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
# Note: All paths should be relative to pwd!
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

# %%Testing
# os.chdir("/Users/evanqu/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Evan/1-Projects/strainslicer/dev/build_cmt_py")
# path_to_p_file="6-case-temp/allpositions.pickle"
# path_to_sample_names_file="6-case-temp/string_sampleID_names.txt"
# path_to_outgroup_boolean_file="6-case-temp/positions/string_outgroup_bool.txt"
# path_to_list_of_quals_files="6-case-temp/string_qual_mat.txt"
# path_to_list_of_diversity_files="6-case-temp/string_diversity_mat.txt"
# path_to_candidate_mutation_table="test_candidate_mutation_table.pickle.gz"


# main(path_to_p_file, path_to_sample_names_file, path_to_outgroup_boolean_file, path_to_list_of_quals_files, path_to_list_of_diversity_files, path_to_candidate_mutation_table, cov_mat_raw, cov_mat_norm ):

# %%
''' load libraries '''
import numpy as np
import pickle
import scipy.io as sio
import os
import sys, argparse
import gzip
from scipy import sparse
from pathlib import Path

MEMMAP_THRESHOLD_BYTES = 2 * 1024 ** 3  # 2 GiB


def _needs_memmap(shape, dtype):
    size = np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
    return size >= MEMMAP_THRESHOLD_BYTES


def _allocate_array(base_dir, name, shape, dtype, use_memmap=True):
    """Allocate a potentially huge array without exhausting RAM."""

    if use_memmap and _needs_memmap(shape, dtype):
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f".{name}.memmap"
        arr = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
        arr.fill(0)
        return arr, path

    return np.zeros(shape, dtype=dtype), None


def _close_memmap(array, path):
    if path is None:
        return
    try:
        array.flush()
    finally:
        if hasattr(array, "_mmap"):
            array._mmap.close()
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

''' positional and optional argument parser'''

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description='''\
                            Gathers everything together for candidate_mutation_table.
                            Optional: Builds coverage matrix (optional w/ double-standardized matrix)
                               ''',
                                 epilog="Questions or comments? --> fkey@mit.edu")
parser.add_argument("-p", dest="allpositions", help="All positions p file (.pickle)", required=True, action='store')
parser.add_argument("-s", dest="sampleNames", help="File with sample names", required=True, action='store')
parser.add_argument("-g", dest="outgroupBool", help="String outgroup bool", required=True, action='store')
parser.add_argument("-q", dest="qualfiles", help="String qual matrix paths", required=True, action='store')
parser.add_argument("-d", dest="divfiles", help="String diversity paths", required=True, action='store')
parser.add_argument("-o", dest="candidate_mutation_table",
                    help="Output candidate mutation table. Py pickle structure (*.pickle.gz)", required=True,
                    action='store')
# parser.add_argument("-c", dest="get_cov", help="Set flag to build raw coverage matrix as sparse csr gzip numpy object (dirname+cov_raw_sparsecsr_mat.npz)",action="store_true", default=False)
# parser.add_argument("-n", dest="get_dbl_norm_cov", help="Set flag to build double normalized coverage matrix as sparse csr gzip numpy object (dirname+cov_norm_sparsecsr_mat.npz)",action="store_true", default=False)
parser.add_argument("-c", dest="cov_mat_raw", help="Output raw coverage matrix as sparse csr gzip numpy object (*.npz)",
                    action='store', default='none')
parser.add_argument("-n", dest="cov_mat_norm",
                    help="Output double normalized coverage matrix as sparse csr gzip numpy object (*.npz)",
                    action='store', default='none')
parser.add_argument("-t", dest="dim", help="Specify the number of statistics (default 8)", type=int, default=40)
args = parser.parse_args()

# %%
'''Functions'''


def main(path_to_p_file, path_to_sample_names_file, path_to_outgroup_boolean_file, path_to_list_of_quals_files,
         path_to_list_of_diversity_files, path_to_candidate_mutation_table, path_to_cov_mat_raw, path_to_cov_mat_norm,
         flag_cov_raw, flag_cov_norm, dim):
    pwd = os.getcwd()

    # p: positions on genome that are candidate SNPs
    print('Processing candidate SNP positions...')
    with open(path_to_p_file, 'rb') as f:
        p = pickle.load(f)
    print('Total number of positions: ' + str(len(p)))

    # SampleNames: list of names of all samples
    print('Processing sample names...')
    fname = pwd + '/' + path_to_sample_names_file
    with open(fname, 'r') as f:
        SampleNames = f.read().splitlines()
    numSamples = len(SampleNames)  # save number of samples
    print('Total number of samples: ' + str(numSamples))

    ## in_outgroup: booleans for whether or not each sample is in the outgroup
    print('Processing outgroup booleans...')

    fname = pwd + '/' + path_to_outgroup_boolean_file
    with open(fname, 'r') as f:
        in_outgroup_str = f.read().splitlines()
    
    in_outgroup = np.asarray([s == '1' for s in in_outgroup_str], dtype=bool).reshape(1, len(in_outgroup_str))

    ## Quals: quality score (relating to sample purity) at each position for all samples
    print('Gathering quality scores at each candidate position...')
    # Import list of directories for where to quals for each sample
    fname = pwd + '/' + path_to_list_of_quals_files
    with open(fname, 'r') as f:
        paths_to_quals_files = f.read().splitlines()
    # Make Quals
    scratch_dir = os.path.dirname(path_to_candidate_mutation_table) or "."

    Quals, quals_path = _allocate_array(
        scratch_dir,
        "candidate_quals",
        (numSamples, len(p)),
        np.int32,
    )
    for i in range(numSamples):
        print('Loading quals matrix for sample: ' + str(i))
        print('Filename: ' + paths_to_quals_files[i])
        with gzip.open(paths_to_quals_files[i], "rb") as f:
            quals_temp = pickle.load(f)
        quals = quals_temp.flatten()
        Quals[i, :] = quals[p - 1]  # -1 convert position to index

    ## counts: counts for each base from forward and reverse reads at each candidate position for all samples
    print('Gathering counts data at each candidate position...\n')

    # Import list of directories for where to diversity file for each sample
    fname = pwd + '/' + path_to_list_of_diversity_files
    with open(fname, 'r') as f:
        paths_to_diversity_files = f.read().splitlines()
    # Load in first diversity to get some stats
    with gzip.open(paths_to_diversity_files[1], 'rb') as f:
        data = np.array(pickle.load(f))
    size = np.shape(data)
    GenomeLength = size[0]

    # Make counts and coverage at the same time
    counts, counts_path = _allocate_array(
        scratch_dir,
        "candidate_counts",
        (numSamples, len(p), dim),
        np.uint32,
    )
    if flag_cov_raw or flag_cov_norm:
        all_coverage_per_bp, cov_path = _allocate_array(
            scratch_dir,
            "candidate_cov_raw",
            (GenomeLength, numSamples),
            np.uint32,
        )
    else:
        all_coverage_per_bp, cov_path = None, None
    indel_counter, indel_path = _allocate_array(
        scratch_dir,
        "candidate_indel",
        (numSamples, len(p), 2),
        np.uint32,
    )

    for i in range(numSamples):
        print('Loading counts matrix for sample: ' + str(i))
        print('Filename: ' + paths_to_diversity_files[i])
        with gzip.open(paths_to_diversity_files[i], 'rb') as f:
            data = np.array(pickle.load(f))
        counts[i, :, :] = data[p - 1, 0:dim]

        if all_coverage_per_bp is not None:
            np.sum(data[:, 0:dim], axis=1, out=all_coverage_per_bp[:, i])

        indel_counter[i, :, :] = data[p - 1, 38:40]  # Num reads supporting indels and reads supporting deletions
        # -1 convert position to index


    # Normalize coverage by sample and then position; ignore /0 ; turn resulting inf to 0

    cov_norm_memmap = None
    cov_norm_path = None
    if flag_cov_norm and all_coverage_per_bp is not None:
        chunk_rows = max(1, min(20000, max(1, int(MEMMAP_THRESHOLD_BYTES // max(1, numSamples * 8)))))
        row_mean = np.zeros(GenomeLength, dtype=np.float64)
        row_std = np.zeros(GenomeLength, dtype=np.float64)
        col_sum = np.zeros(numSamples, dtype=np.float64)
        col_sq_sum = np.zeros(numSamples, dtype=np.float64)

        for start in range(0, GenomeLength, chunk_rows):
            stop = min(start + chunk_rows, GenomeLength)
            chunk = np.asarray(all_coverage_per_bp[start:stop, :], dtype=np.float64)
            row_mean_chunk = chunk.mean(axis=1)
            row_std_chunk = chunk.std(axis=1)
            row_std_chunk = np.where(row_std_chunk > 0, row_std_chunk, 1.0)
            norm_chunk = (chunk - row_mean_chunk[:, None]) / row_std_chunk[:, None]
            norm_chunk[~np.isfinite(norm_chunk)] = 0
            col_sum += norm_chunk.sum(axis=0)
            col_sq_sum += np.square(norm_chunk).sum(axis=0)
            row_mean[start:stop] = row_mean_chunk
            row_std[start:stop] = row_std_chunk

        col_mean = col_sum / GenomeLength
        col_var = col_sq_sum / GenomeLength - col_mean ** 2
        col_var = np.where(col_var > 0, col_var, 1.0)
        col_std = np.sqrt(col_var)
        col_std = np.where(col_std > 0, col_std, 1.0)

        cov_norm_memmap, cov_norm_path = _allocate_array(
            scratch_dir,
            "candidate_cov_norm",
            (numSamples, GenomeLength),
            np.int32,
        )

        for start in range(0, GenomeLength, chunk_rows):
            stop = min(start + chunk_rows, GenomeLength)
            chunk = np.asarray(all_coverage_per_bp[start:stop, :], dtype=np.float64)
            row_mean_chunk = row_mean[start:stop]
            row_std_chunk = row_std[start:stop]
            norm_chunk = (chunk - row_mean_chunk[:, None]) / row_std_chunk[:, None]
            norm_chunk[~np.isfinite(norm_chunk)] = 0
            norm_chunk = (norm_chunk - col_mean) / col_std
            norm_chunk[~np.isfinite(norm_chunk)] = 0
            scaled = np.round(norm_chunk, 3) * 1000
            cov_norm_memmap[:, start:stop] = scaled.T.astype(np.int32)

    # Reshape & save matrices

    p = p.transpose() #p: num_pos
    in_outgroup = in_outgroup.flatten() #in_outgroup: num_samples
    SampleNames = SampleNames #sampleNames: num_samples


    outdir = os.path.dirname(path_to_candidate_mutation_table)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if flag_cov_raw and all_coverage_per_bp is not None:
        print("Saving " + path_to_cov_mat_raw)
        with open(path_to_cov_mat_raw, 'wb') as f:
            np.savez_compressed(path_to_cov_mat_raw, all_coverage_per_bp=all_coverage_per_bp.transpose())

    if flag_cov_norm and cov_norm_memmap is not None:
        print("Saving " + path_to_cov_mat_norm)
        with open(path_to_cov_mat_norm, 'wb') as f:
            np.savez_compressed(path_to_cov_mat_norm, array_cov_norm_scaled=cov_norm_memmap)

    CMT = {'sample_names': SampleNames,
           'p': p,
           'counts': counts,
           'quals': Quals,
           'in_outgroup': in_outgroup,
           'indel_counter': indel_counter,
           }

    file_path = path_to_candidate_mutation_table
    print("Saving " + path_to_candidate_mutation_table)
    np.savez_compressed(file_path, **CMT)

    print('DONE')

    _close_memmap(Quals, quals_path)
    _close_memmap(counts, counts_path)
    _close_memmap(indel_counter, indel_path)
    if all_coverage_per_bp is not None:
        _close_memmap(all_coverage_per_bp, cov_path)
    if cov_norm_memmap is not None:
        _close_memmap(cov_norm_memmap, cov_norm_path)


# %%
if __name__ == "__main__":
    path_to_p_file = args.allpositions
    path_to_sample_names_file = args.sampleNames
    path_to_outgroup_boolean_file = args.outgroupBool
    path_to_list_of_quals_files = args.qualfiles
    path_to_list_of_diversity_files = args.divfiles
    path_to_candidate_mutation_table = args.candidate_mutation_table
    path_to_cov_mat_raw = args.cov_mat_raw
    path_to_cov_mat_norm = args.cov_mat_norm
    dim=args.dim
    if path_to_cov_mat_raw == 'none':
        flag_cov_raw = False
    else:
        flag_cov_raw = True
    if path_to_cov_mat_norm == 'none':
        flag_cov_norm = False
    else:
        flag_cov_norm = True
    main(path_to_p_file, path_to_sample_names_file, path_to_outgroup_boolean_file, path_to_list_of_quals_files,
         path_to_list_of_diversity_files, path_to_candidate_mutation_table, path_to_cov_mat_raw, path_to_cov_mat_norm,
         flag_cov_raw, flag_cov_norm,dim)
