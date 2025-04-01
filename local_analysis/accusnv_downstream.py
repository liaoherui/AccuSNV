"""
This script takes candidate mutation table from snakemake pipeline as input and does a lot of downstream analysis
"""
import numpy as np
import pickle
import math
import pandas as pd
import sys
sys.path.insert(0, './miniscripts_for_dNdS')
import os
import re
import copy
import argparse
from scipy import stats
# Import Lieberman Lab SNV-calling python package
script_dir = os.path.dirname(os.path.abspath(__file__))
dir_py_scripts = script_dir+"/modules"
sys.path.insert(0, dir_py_scripts)
import snv_module_recoded_with_dNdS as snv

parser=argparse.ArgumentParser(prog='Downstream analysis module of AccuSNV',description='SNV calling tool for bacterial isolates using deep learning.')
parser.add_argument('-i','--input_mat',dest='input_mat',type=str,required=True,help="The input mutation table in npz file")
#parser.add_argument('-t','--input_report',dest='input_report',type=str,required=True,help="The input of the full annotation dataframe (snv_table_merge_all_mut_annotations.tsv) output by accusnv_snakemake")
parser.add_argument('-r','--ref_dir',dest='ref_dir',type=str,help="The dir of your reference genomes")
parser.add_argument('-c','--min_cov_for_call',dest='min_cov',type=str,help="For the fill-N module: on individual samples, calls must have at least this many fwd+rev reads. Default is 1.")
parser.add_argument('-q','--min_qual_for_call',dest='min_qual',type=str,help="For the fill-N module: on individual samples, calls must have at least this minimum quality score. Default is 30.")
parser.add_argument('-f','--min_freq_for_call',dest='min_freq',type=str,help="For the fill-N module: on individual samples, a call's major allele must have at least this freq. Default is 0.75.")
parser.add_argument('-o','--output_dir',dest='output_dir',type=str,help="The output dir")
args = parser.parse_args()

def set_para_int(invalue,expect):
    if not invalue:
        invalue=expect
    else:
        invalue=int(invalue)
    return invalue

def set_para_float(invalue,expect):
    if not invalue:
        invalue=expect
    else:
        invalue=float(invalue)
    return invalue

input_mat=args.input_mat
#input_report=args.input_report
ref_dir=args.ref_dir
fn_min_cov=args.min_cov
fn_min_qual=args.min_qual
fn_min_freq=args.min_freq
output_dir=args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fn_min_cov=set_para_int(fn_min_cov,1)
fn_min_qual=set_para_int(fn_min_qual,30)
fn_min_freq=set_para_float(fn_min_freq,0.75)


[quals,p,counts,in_outgroup,sample_names,indel_counter] = \
    snv.read_candidate_mutation_table_npz(input_mat)

my_cmt = snv.cmt_data_object( sample_names,
                             in_outgroup,
                             p,
                             counts,
                             quals,
                             indel_counter
                             )

with open(input_mat, 'rb') as f:
    cmt = np.load(f)
    prob = np.array(cmt['prob'])
    label = np.array(cmt['label'])
    recomb = np.array(cmt['recomb'])

# by default - will use label=1 positions
filt_pos= label
filt_pos2 = ~recomb
my_cmt.filter_positions(filt_pos)
my_cmt.filter_positions(filt_pos2)
my_calls = snv.calls_object( my_cmt )
my_rg = snv.reference_genome_object( ref_dir )
my_rg_annot = my_rg.annotations
#exit()

'''
Fill-N module: fill N for samples with a loose filter cutoff
'''
# Apply looser filters than before (want as many alleles as possible)
filter_parameter_calls_for_tree = {
                                    'min_cov_for_call' : fn_min_cov, # on individual samples, calls must have at least this many fwd+rev reads
                                    'min_qual_for_call' : fn_min_qual, # on individual samples, calls must have this minimum quality score
                                    'min_major_nt_freq_for_call' : fn_min_freq,  # on individual samples, a call's major allele must have at least this freq
                                    }



my_calls.filter_calls_by_element(
    my_cmt.coverage < filter_parameter_calls_for_tree['min_cov_for_call']
    ) # forward strand coverage too low

my_calls.filter_calls_by_element(
    my_cmt.quals < filter_parameter_calls_for_tree['min_qual_for_call']
    ) # quality too low

my_calls.filter_calls_by_element(
    my_cmt.major_nt_freq < filter_parameter_calls_for_tree['min_major_nt_freq_for_call']
    ) # major allele frequency too low
#print(my_calls.calls)
#exit()
###############################################

# Filtered calls for outgroup samples only
calls_outgroup = my_calls.get_calls_in_outgroup_only()
# Switch N's (0's) to NaNs
calls_outgroup_N_as_NaN = calls_outgroup.astype('float') # init ()
calls_outgroup_N_as_NaN[ calls_outgroup_N_as_NaN==0 ] = np.nan

# Infer ancestral allele as the most common allele among outgroup samples (could be N)
calls_ancestral = np.zeros( my_calls.num_pos, dtype='int') # init as N's
outgroup_pos_with_calls = np.any(calls_outgroup,axis=0) # positions where the outgroup has calls
calls_ancestral[outgroup_pos_with_calls] = stats.mode( calls_outgroup_N_as_NaN[:,outgroup_pos_with_calls], axis=0, nan_policy='omit' ).mode.squeeze()

#%% Compute mutation quality

# Grab filtered calls from ingroup samples only
calls_ingroup = my_calls.get_calls_in_sample_subset( np.logical_not( my_calls.in_outgroup ) )
quals_ingroup = my_cmt.quals[ np.logical_not( my_calls.in_outgroup ),: ]
num_samples_ingroup = sum( np.logical_not( my_calls.in_outgroup ) )
# Note: Here we are only looking for SNV differences among ingroup samples. If
# you also want to find SNV differences between the ingroup and the outgroup
# samples (eg mutations that have fixed across the ingroup), then you need to
# use calls and quals matrices that include outgroup samples.
#print(calls_ingroup,quals_ingroup)
#exit()
# Compute quality
[ mut_qual, mut_qual_samples ] = snv.compute_mutation_quality( calls_ingroup, quals_ingroup )
# note: returns NaN if there is only one type of non-N call

# Make a table (pandas dataframe) of SNV positions and relevant annotations
# # Pull alleles from reference genome across p
calls_reference = my_rg.get_ref_NTs_as_ints( p )

# # Update ancestral alleles
pos_to_update = ( calls_ancestral==0 )
calls_ancestral[ pos_to_update ] = calls_reference[ pos_to_update ]
goodpos_bool=arr = np.ones(len(my_cmt.p), dtype=bool)
goodpos_idx = np.where( goodpos_bool )[0]
calls_goodpos_all = my_calls.calls
calls_goodpos_ingroup_all = calls_goodpos_all[ np.logical_not( my_calls.in_outgroup ),: ]

# Filters
filter_SNVs_not_N = ( calls_ingroup != snv.nts2ints('N') ) # mutations must have a basecall (not N)
filter_SNVs_not_ancestral_allele = ( calls_ingroup != np.tile( calls_ancestral, (num_samples_ingroup,1) ) ) # mutations must differ from the ancestral allele

# Fixed mutations per sample per position
fixedmutation = \
    filter_SNVs_not_N \
    & filter_SNVs_not_ancestral_allele 

p_goodpos_all = my_calls.p
goodpos_idx_all = np.where( goodpos_bool)[0]

# Parameters
promotersize = 250; # how far upstream of the nearest gene to annotate something a promoter mutation (not used if no annotation)
mutations_annotated = snv.annotate_mutations( \
    my_rg, \
    p_goodpos_all, \
    np.tile( calls_ancestral[goodpos_idx], (my_cmt.num_samples,1) ), \
    calls_goodpos_ingroup_all, \
    my_cmt, \
    fixedmutation[:,goodpos_idx_all], \
    mut_qual[:,goodpos_bool].flatten(), \
    promotersize \
    )


# Choose subset of samples or positions to use in the tree by idx
samplestoplot = np.arange(my_cmt.num_samples) # default is to use all samples
num_goodpos_all = len(goodpos_idx_all)
goodpos4tree = np.arange(num_goodpos_all) # default is to use all positions
treesampleNamesLong = my_cmt.sample_names
calls_for_treei = my_calls.calls[np.ix_(samplestoplot, goodpos4tree)]  # numpy broadcasting of row_array and col_array requires np.ix_()
calls_for_tree = snv.ints2nts(calls_for_treei)  # NATCG translation
###############################################



output_tsv_filename = output_dir + '/snv_table_mutations_annotations.tsv'
snv.write_mutation_table_as_tsv( \
    my_cmt.p, \
    mut_qual[0, goodpos_idx_all], \
    my_cmt.sample_names, \
    mutations_annotated, \
    calls_for_tree, \
    treesampleNamesLong, \
    output_tsv_filename \
 \
    )

#exit()

"""
@author: Alyssa Mitchell, 2025.03.20

Local code block for integration with AccuSNV downstream analysis module (local) - dN/dS calculation
"""
dir_ref_genome = ref_dir
annotation_full = pd.read_csv(output_dir + '/snv_table_mutations_annotations.tsv',sep='\t')
output_directory = output_dir

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
#%%
# Mutation spectrum
NTs = np.array(['A', 'T', 'C', 'G', 'N'])
mutationmatrix, mut_observed, typecounts, prob_nonsyn = snv.mutation_spectrum_module(annotation_full, NTs)

# Notes:
# * Mutation spectrum tallied for each mutant alelle
# * Mutation type only tallied when there is only one mutant allele
# * Assumes all mutants are true de novo mutants
num_muts_for_empirical_spect = 100
if np.sum(mut_observed) >= num_muts_for_empirical_spect:
    # attempt to get mutation spectrum empirically if there were enough mutations
    mut_spec_prob = mut_observed / np.sum(mut_observed)
else:
    # otherwise assume a uniform mutation spectrum
    mut_spec_prob = np.ones(mut_observed.size) / mut_observed.size
    print('Warning! Assuming uniform mutation spectrum.')

# Expected N/S
probnonsyn_expected = snv.compute_expected_dnds(dir_ref_genome, mut_spec_prob)

# this is going to be a bit off from matlab bc of how it deals with alternate start codons
dnds_expected = probnonsyn_expected / (1 - probnonsyn_expected)  # N/S expected from neutral model

# compute observed N/S for fixed and diverse mutations
    # define gene_nums_of_interest if binning mutations for dN/dS analysis, rather than whole-genome dN/dS
p_nonsyn, CI_nonsyn, num_muts_N, num_muts_S = snv.compute_observed_dnds(annotation_full, gene_nums_of_interest=None)
dnds_observed = p_nonsyn / (1 - p_nonsyn)  # N/S observed
# note that in matlab version, binom(0,0) gives CI of [0,1] even when p=NaN. In python, both are NaN

# dN/dS
# relative to neutral model for this KEGG category
dNdS = dnds_observed / dnds_expected

CI_lower = (CI_nonsyn[0] / (1 - CI_nonsyn[0])) / dnds_expected
try:
    CI_upper = (CI_nonsyn[1] / (1 - CI_nonsyn[1])) / dnds_expected
except ZeroDivisionError:
    CI_upper = np.inf


print('dN/dS =', dNdS)

# save output as binary file using pickle
output_dict = {
    'dNdS': dNdS,
    'CI_lower': CI_lower,
    'CI_upper': CI_upper,
    'num_muts_N': num_muts_N,
    'num_muts_S': num_muts_S,
    'p_nonsyn': p_nonsyn,
    'probnonsyn_expected': probnonsyn_expected
}
'''
with open(output_directory+'/data_dNdS.pickle', 'wb') as f:
    pickle.dump(output_dict, f)
'''
np.savez_compressed(output_directory+'/data_dNdS.npz', **output_dict)
