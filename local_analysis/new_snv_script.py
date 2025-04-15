# -*- coding: utf-8 -*-
"""

SUMMARY:

    This script demonstrates how to go from a candidate mutation table and a 
    reference genome to a high-quality SNV table and a parsimony tree.

    This script assumes that your input is whole genome sequencing data from
    isolates that has been processed through the Lieberman Lab's standard 
    Snakemake pipeline.
    
    This script is intended to be run interactively rather than as a single 
    function to enable dynamic adjustment of filters to meet the particulars of 
    your sequencing runs, including sample depth and amount of 
    cross-contamination. In our experience, universal filters do not exist, so 
    manual inspection of filtering is essential to SNV calling.


WARNINGS!

    1. The SNV filters illustrated in this script may not be appropriate for 
    your dataset. It is critical that you use the built-in tools for manual 
    inspection of SNV quality. 
    
    2. This script uses the new version of the Lieberman Lab's SNV module, 
    which as of October 2022 is NOT compatible with old versions. If you add 
    functionality from previous versions, you must ensure that data structure
    and indexing are updated appropriately. See documentation for more 
    information on the standards implemented here.


VERSION HISTORY:

    YEAR.MONTH; Name: Add new revisions here!
    
    2022.10; Arolyn: Major overhaul of main script and python module. Not 
    compatible with previous versions. Introduced classes/methods for candidate 
    mutation table data, basecalls, and reference genomes. Implemented 
    consistency in which python packages are used (for example, all numerical
    arrays are now numpy arrays and heterogeneous arrays are now pandas 
    dataframes) and in indexing of genomes and nucleotides. Added many 
    functions for data visualization and quality control. 
    
    2022.04; Tami, Delphine, Laura: Lieberman Lab Hackathon
    
    Additional notes: This module is based on a previous MATLAB version.


@author: Lieberman Lab at MIT. Authors include: Tami Lieberman, Idan Yelin, 
Felix Key, Arolyn Conwill, A. Delphine Tripp, Evan Qu, Laura Markey, Chris Mancuso

"""


#%%#####################
## SET UP ENVIRONMENT ##
########################

# Import python packages
import sys
import os
import re
import copy
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import datetime, time
import matplotlib.pyplot as plt


# Some functions needed for subsequent steps
def search_ref_name(refg):
    pre=''
    #fname=''
    for filename in os.listdir(refg):
        if re.search('fa',filename) or re.search('fna',filename):
            pre=re.split('\.',filename)[0]

            break
    return pre
def remove_same(my_calls_in):
    keep_col = []
    for i in range(my_calls_in.calls.shape[1]):
        unique_nonzero_elements = np.unique(my_calls_in.calls[:, i][my_calls_in.calls[:, i] != 0])
        if len(unique_nonzero_elements) < 2:
            my_calls_in.calls[:, i] = 0
            keep_col.append(False)
        else:
            keep_col.append(True)
    keep_col = np.array(keep_col)
    return keep_col

def is_digit(input_string):
    return input_string.isdigit()

class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

def plot_snv_counts_gpt(data_dict, odir=None, title="SNV Counts by Sample", figsize=(10, 6),
                    color='#1f77b4', marker='o', markersize=100,
                    xlabel="Sample Name", ylabel="SNV Count", dpi=400):
    """
    Creates a scatter plot of SNV counts for different samples.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with sample names as keys and SNV counts as values
    odir : str
        Output directory path where the figure will be saved. If None, figure is not saved
    title : str
        Title of the plot
    figsize : tuple
        Figure size (width, height) in inches
    color : str
        Color of the markers
    marker : str
        Marker style
    markersize : int
        Size of the markers
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    dpi : int
        Resolution of the output image in dots per inch

    Returns:
    --------
    fig, ax : tuple
        Matplotlib figure and axis objects
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Extract sample names and counts
    samples = list(data_dict.keys())
    counts = list(data_dict.values())

    # Create x positions (0, 1, 2, ...)
    x_pos = np.arange(len(samples))

    # Create scatter plot
    ax.scatter(x_pos, counts, s=markersize, c=color, marker=marker, alpha=0.7)

    # Only show x-axis labels if sample count is 20 or less
    if len(samples) <= 20:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(samples, rotation=45, ha='right')
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    # Add gridlines for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Adjust y-axis to start from 0
    ymin, ymax = ax.get_ylim()
    ax.set_ylim([0, ymax * 1.1])  # Add 10% padding at the top

    # Add value labels for samples with counts greater than 1000
    ct=0
    for i, count in enumerate(counts):
        if count > 1000 and len(samples)>20:
            ax.annotate(samples[i],
                        (x_pos[i], counts[i]),
                        textcoords="offset points",
                        xytext=(0, -50),
                        ha='center', fontsize=10, color='red',rotation=90)
            ct+=1
            if ct>20:break


    # Adjust layout
    plt.tight_layout()

    fig2,ax2 = plt.subplots(figsize=(figsize[0], figsize[1] * 0.7))
    ax2.hist(counts, bins=20, color=color, alpha=0.7)
    ax2.set_xlabel(ylabel + " Distribution", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Histogram of SNV Counts")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()


    # Save the figure if output directory is provided
    if odir is not None:
        # Create the directory if it doesn't exist
        os.makedirs(odir, exist_ok=True)

        # Define the output file path
        output_path = os.path.join(odir, "snvs_per_sample.png")


        # Save the figure
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        fig2.savefig(os.path.join(odir, "snvs_histogram_per_sample.png"), dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {odir}")

    return fig, ax

# To check SNVs of each input sample
def check_snv(data_file_cmt,odir):
    [quals,p,counts,in_outgroup,sample_names,indel_counter] = \
    snv.read_candidate_mutation_table_npz(data_file_cmt)
    
    #print(in_outgroup)
    #exit()
    if not len(in_outgroup)==len(sample_names):
        in_outgroup=np.array([False] * len(sample_names))
    my_cmt = snv.cmt_data_object( sample_names,
                             in_outgroup,
                             p,
                             counts,
                             quals,
                             indel_counter
                             )
    #print(my_cmt.counts[:,:,:8])
    my_calls = snv.calls_object( my_cmt )
    keep_col = remove_same(my_calls)
    #print(keep_col)
    #exit()
    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)
    #print(my_calls.calls)
    def find_min_freq_elements(column):
        #print(column)
        values, counts = np.unique(column, return_counts=True)
        nonzero_mask = values != 0
        values = values[nonzero_mask]
        counts = counts[nonzero_mask]
        sorted_indices = np.argsort(-counts)
        #print(values,counts,sorted_indices)
        #exit()
        min_count = counts[sorted_indices[1]]
        #print(min_count)
        #print(values[counts == min_count])
        #exit()
        #print(min_count)
        #print(values[counts == min_count])
        #exit()
        res=values[counts == min_count]
        if len(res)>1:
            res=[res[0]]
        return res
    #print(my_calls.calls,my_calls.calls.shape)
    min_freq_elements = np.apply_along_axis(find_min_freq_elements, 0, my_calls.calls)[0]
    array=my_calls.calls
    row_match_counts = np.zeros(array.shape[0], dtype=int)
    for col_idx in range(array.shape[1]):
        col_values = array[:, col_idx]
        min_freq_vals = min_freq_elements[col_idx]

        matches = np.isin(col_values, min_freq_vals)
        row_match_counts += matches
    #print(min_freq_elements,row_match_counts)
    #exit()
    dcs=dict(zip(sample_names,row_match_counts))
    plot_snv_counts_gpt(dcs, odir)
    return dcs


# Import Lieberman Lab SNV-calling python package
script_dir = os.path.dirname(os.path.abspath(__file__))
dir_py_scripts = script_dir+"/modules"
sys.path.insert(0, dir_py_scripts)
import snv_module_recoded_with_dNdS as snv # SNV calling module
import build_SNP_Tree as bst
import CNN_pred as cnn


# Get timestamp
ts = time.time() 
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S') # used in output files

parser=argparse.ArgumentParser(prog='Local analysis module of AccuSNV',description='Apply filters and CNN to call SNVs for closely related bacterial isolates.')
parser.add_argument('-i','--input_mat',dest='input_mat',type=str,required=True,help="The input mutation table in npz file")
parser.add_argument('-c','--input_cov',dest='input_cov',type=str,help="The input coverage table in npz file")
parser.add_argument('-s','--min_cov_for_filter_sample',dest='min_cov_samp',type=str,help="Before running the CNN model, low-quality samples with more than 45%% of positions having zero aligned reads will be filtered out. (default \"-s 45\") You can adjust this threshold with this parameter; to include all samples, set \"-s 100\".")
parser.add_argument('-v','--min_cov_for_filter_pos',dest='min_cov',type=str,help="For the filter module: on individual samples, calls must have at least this many reads on the fwd/rev strands individually. If many samples have low coverage (e.g. <5), then you can set this parameter to smaller value. (e.g. -v 2). Default is 5.")
parser.add_argument('-e','--excluse_samples',dest='exclude_samp',type=str,help="The names of the samples you want to exclude (e.g. -e S1,S2,S3). If you specify a number, such as \"-e 1000\", any sample with more than 1,000 SNVs will be automatically excluded.")

parser.add_argument('-g','--generate_report',dest='generate_rep',type=str,help="If not generate html report and other related files, set to 0. (default: 1)")
parser.add_argument('-r','--rer',dest='ref_genome',type=str,help="The reference genome")
parser.add_argument('-o','--output_dir',dest='output_dir',type=str,help="The output dir")
#parser.add_argument('-o','--output_file',dest='output_file',type=str,help="The output file")
args=parser.parse_args()
input_mat=args.input_mat
input_cov=args.input_cov
min_cov_samp=args.min_cov_samp
min_cov_filt=args.min_cov
refg=args.ref_genome
odir=args.output_dir
greport=args.generate_rep
exclude_samp=args.exclude_samp
if not min_cov_filt:
    min_cov_filt=5
else:
    min_cov_filt=int(min_cov_filt)

if not min_cov_samp:
    min_cov_samp=45
else:
    min_cov_samp=int(min_cov_samp)

if not exclude_samp:
    exclude_samp=''

if is_digit(exclude_samp):
    exclude_samp=int(exclude_samp)
    
if not greport:
    greport=1
else:
    greport=int(greport)
#Build
if not os.path.exists(odir):
    os.makedirs(odir)



#%%#############################
## DATA IMPORT AND PROCESSING ##
################################


#%% Dataset info
dataset_name= 'Your-InputData'
data_file_cmt=input_mat
data_file_cov = input_cov
dir_ref_genome = refg
ref_genome_name = search_ref_name(refg)
#dir_ref_genome = refg+'/'+fname
#samples_to_exclude = ["P-15_O-Ec_S-STOOL_C-C1_D-4","P-15_O-Ec_S-STOOL_C-D3_D-4"] # option to exclude specific samples manually
#samples_to_exclude=["P-13_O-Kp_S-BLOOD_C-H1_D-0"]

dcs=check_snv(data_file_cmt,odir)
if is_digit(str(exclude_samp)):
    samples_to_exclude=[]
    for s in dcs:
        if dcs[s]>=exclude_samp:
            samples_to_exclude.append(s)

else:
    if not exclude_samp=='':
        tem=re.split(',',exclude_samp)
        samples_to_exclude=tem
    else:
        samples_to_exclude=[""]
print('Exclude samples are:',samples_to_exclude)
#exit()

#samples_to_exclude=["AHM_v0002_D02","AHM_v0002_D03","AHM_v0012_F12"]


# Make subdirectory for this dataset
#dir_output = 'output_elife-sau'
dir_output=odir
os.system( "mkdir " + dir_output );


################ Run CNN first and then combine the result of CNN and default filters ######

####### Run CNN ########
cnn_pos,cnn_pred,cnn_prob,dgap=cnn.CNN_predict(data_file_cmt,data_file_cov,odir,samples_to_exclude,min_cov_samp) # The label is predicted by CNN
dlab=dict(zip(cnn_pos,cnn_pred)) # pos -> label
dprob=dict(zip(cnn_pos,cnn_prob)) # pos -> probability
#######  Done  #########
#print(dgap)
#exit()

#%% Generate candidate mutation table object

# Import candidate mutation table data generated in Snakemake

# Use this version for updated candidate mutation table matrices
[quals,p,counts,in_outgroup,sample_names,indel_counter] = \
    snv.read_candidate_mutation_table_npz(data_file_cmt) 
#dx=np.where(p==832924)
#print(counts[:,dx,:])
if not len(in_outgroup)==len(sample_names):
    in_outgroup=np.array([False] * len(sample_names))
#print(in_outgroup)
# # Use this version for old candidate mutation table matrices
# [quals,p,counts,in_outgroup,sample_names,indel_counter] = \
#     snv.read_old_candidate_mutation_table_pickle_gzip( data_file_cmt ) 


# Create instance of candidate mutation table class
my_cmt = snv.cmt_data_object( sample_names, 
                             in_outgroup, 
                             p, 
                             counts, 
                             quals, 
                             indel_counter 
                             )
#print(counts[:,0,:8])

#%% Import reference genome information

# Create instance of reference genome class
#print(dir_ref_genome)
#exit()
my_rg = snv.reference_genome_object( dir_ref_genome )
#exit()
#print(p.shape)
contig_p=my_rg.p2contigpos(p)
#print(contig_p.shape)
#print(my_rg.contig_names)
pre_a=re.split('_',my_rg.contig_names[0])
pre='_'.join(pre_a[:-1])
d={}
x=0
for c in contig_p:
	#print(c)
	d[p[x]]=[int(c[0]),int(c[1])]
	x+=1
#print(len(d))	
#print(d)
#exit()
'''
f=open(odir+'/cnn_res.txt','r')
o=open(odir+'/cnn_res_contig.txt','w+')
line=f.readline()
o.write('Contig\tContig_pos\tPos_info\tPredicted_label\tProbability\tGap_Filt\n')
while True:
	line=f.readline().strip()
	if not line:break
	ele=line.split('\t')
	o.write(str(my_rg.contig_names[d[int(ele[0])][0]-1])+'\t'+str(d[int(ele[0])][1])+'\t'+line+'\n')
o.close()
'''
#print(contig_p.shape,my_rg.contig_names.shape)
#exit()
my_rg_annot = my_rg.annotations
#print(my_rg_annot)
#my_rg_annot_0 = my_rg_annot[0]

#exit()

#%% Process raw coverage matrix

# Create instance of simple coverage class (just a summary of coverage matrix data to save memory)
# my_cov = snv.cov_data_object_simple( snv.read_cov_mat_npz( covFile ), 
#                                     my_cmt.sample_names, 
#                                     my_rg.genome_length, 
#                                     my_rg.contig_starts, 
#                                     my_rg.contig_names 
#                                     )

# Create instance of full coverage matrix class (includes full coverage matrix as attribute)
my_cov = snv.cov_data_object( snv.read_cov_mat_npz( data_file_cov ), \
                             my_cmt.sample_names, \
                             my_rg.genome_length, \
                             my_rg.contig_starts, \
                             my_rg.contig_names \
                             )



#%% Exclude any samples listed above as bad

samples_to_exclude_bool = np.array( [x in samples_to_exclude for x in my_cmt.sample_names] )
my_cmt_zero_rebuild=copy.deepcopy(my_cmt)
my_cmt.filter_samples( ~samples_to_exclude_bool )
my_cov.filter_samples( ~samples_to_exclude_bool )
my_cmt_zero=copy.deepcopy(my_cmt)
#my_cmt_for_rebuild=copy.deepcopy(my_cmt)
#print(my_cov)
#exit()
######################
## Dicts For Table  ##
######################
# dpt is used to keep the result of filters
dpt={} # dict used to keep information of identified SNPs. e.g d->pos -> {"cov_filter": 1,"qual_filter":1,...... }
#### We can't cause there are multiple samples for each position
# dft is used to keep the value of fwd reads
#dft={} # e.g. pos -> {"cov":3,"qual":30,...}
# drt is used to keep the value of rev reads
#drt={} # e.g. pos -> {"cov":3,"qual":30,...}


#%%###################
## FILTER BASECALLS ##
######################


#%% FILTER BASECALLS

# Create instance of basecalls class for initial calls
my_calls = snv.calls_object( my_cmt )
#my_calls_raw = snv.calls_object( my_cmt )
#my_calls_raw_zero=copy.deepcopy(my_calls)
'''
x=my_calls_raw_zero.get_frac_Ns_by_position()
print(my_calls.p[-10:])
print(my_cmt.major_nt[:,-10:])
print(my_cmt.major_nt_freq[:,-10:])
#print(my_cmt.major_nt_freq.shape)
print(my_cmt.minor_nt[:,-10:])
print(my_cmt.minor_nt_freq[:,-10:])
#print(my_cmt.minor_nt_freq.shape)
#print(x.shape,x,my_calls.p)
exit()
'''


#%% Filter parameters

# Remove samples that are not high quality
filter_parameter_sample_across_sites = {
                                        'min_average_coverage_to_include_sample': 0, # remove samples that have low coverage # default: 10
                                        'max_frac_Ns_to_include_sample': 1 # remove samples that have too many undefined base (ie. N). # default: 0.2
                                        }

# Remove sites within samples that are not high quality
filter_parameter_site_per_sample = {
                                    'min_major_nt_freq_for_call' : 0.85, # on individual samples, a calls' major allele must have at least this freq
                                    'min_cov_per_strand_for_call' : min_cov_filt,  # on individual samples, calls must have at least this many reads on the fwd/rev strands individually
                                    'min_qual_for_call' : 30, # on individual samples, calls must have this minimum quality score
                                    'max_frac_reads_supporting_indel' : 0.33 # on individual samples, no more than this fraction of reads can support an indel at any given position
                                    }

# Remove sites across samples that are not high quality
filter_parameter_site_across_samples = {
                                        'max_fraction_ambigious_samples' : 1, # across samples per position, the fraction of samples that can have undefined bases
                                        'min_median_coverage_position' : 5, # across samples per position, the median coverage
                                        'max_mean_copynum' : 4, # mean copy number at a positions across all samples
                                        'max_max_copynum' : 7 # max maximum copynumber that a site can have across all samples
                                        }

original_stdout = sys.stdout
sys.stdout = Tee(odir+'/pipe_log.txt')
#%% Filter samples based on coverage

# Identify samples with low coverage and make a histogram
[ low_cov_samples, goodsamples_coverage ] = snv.filter_samples_by_coverage( 
    my_cov.get_median_cov_of_chromosome(), 
    filter_parameter_sample_across_sites['min_average_coverage_to_include_sample'], 
    my_cov.sample_names, 
    True, 
    dir_output 
    )
#print(low_cov_samples)
#exit()

# Filter candidate mutation table, coverage, and calls objects
my_cmt.filter_samples( goodsamples_coverage )
my_cov.filter_samples( goodsamples_coverage )
my_calls.filter_samples( goodsamples_coverage )


#%% Filter calls per position per sample
#print(my_cmt.p)
#print(my_calls.calls.T)
#print(my_cmt.fwd_cov.T)
#print(my_cmt.rev_cov.T)
#print(my_cmt)

#print('---------')
#print(my_calls.calls.T.shape)

# # Filter based on quality
# my_calls_raw=copy.deepcopy(my_calls)
# my_calls_qual=copy.deepcopy(my_calls)
# #my_cmt_qual=copy.deepcopy(my_cmt)
# my_calls_qual.filter_calls_by_element(
#     my_cmt.quals < filter_parameter_site_per_sample['min_qual_for_call']
#     ) # quality too low
#
# tokens = snv.token_generate(my_calls_raw.calls.T, my_calls_qual.calls.T, 'filter-qual')
# dpt['qual']=dict(zip(my_calls_qual.p,tokens))


# Filter based on quality
my_calls_raw=copy.deepcopy(my_calls)
my_calls.filter_calls_by_element(
    my_cmt.quals < filter_parameter_site_per_sample['min_qual_for_call']
    ) # quality too low
tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T, 'filter-qual')
dpt['qual']=dict(zip(my_calls.p,tokens))

# Filter based on coverage
my_calls_raw=copy.deepcopy(my_calls)

my_calls.filter_calls_by_element( 
    my_cmt.fwd_cov < filter_parameter_site_per_sample['min_cov_per_strand_for_call'] 
    ) # forward strand coverage too low

my_calls.filter_calls_by_element( 
    my_cmt.rev_cov < filter_parameter_site_per_sample['min_cov_per_strand_for_call'] 
    ) # reverse strand coverage too low
tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T, 'filter-coverage')
dpt['cov']=dict(zip(my_calls.p,tokens))
#print(my_cmt.quals.shape)
#exit()
#print(my_cmt.p,len(tokens))
#print(dpt)
#exit()
#print(my_calls.calls.T)
#print(my_calls_raw.calls.T)
#print(np.array_equal(my_calls.calls,my_calls_raw.calls) )
#unequal_mask = my_calls.calls != my_calls_raw.calls
#print(unequal_mask)
#exit()

#print('---------')

#tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T, 'filter-qual')
#dpt['qual']=dict(zip(my_calls.p,tokens))

#exit()
#print(my_calls.p)
#print(my_calls.calls.T)
#print(my_calls.calls.T.shape)
#exit()
# Filter based on major allele frequency
my_calls_raw=copy.deepcopy(my_calls)
my_calls.filter_calls_by_element( 
    my_cmt.major_nt_freq < filter_parameter_site_per_sample['min_major_nt_freq_for_call'] 
    ) # major allele frequency too low
tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T, 'filter-major allele freq')
dpt['maf']=dict(zip(my_calls.p,tokens))

#%% Filter positions with indels

with np.errstate(divide='ignore',invalid='ignore'):
    # compute number of reads supporting an indel
    frac_reads_supporting_indel = np.sum(my_cmt.indel_stats,axis=2)/my_cmt.coverage # sum reads supporting insertion plus reads supporting deletion
    frac_reads_supporting_indel[ ~np.isfinite(frac_reads_supporting_indel) ] = 0
    # note: this fraction can be above zero beacuse the number of reads supporting an indel includes a +/-3 bp window around a given position on the genome
my_calls_raw=copy.deepcopy(my_calls)
my_calls.filter_calls_by_element( 
    frac_reads_supporting_indel > filter_parameter_site_per_sample['max_frac_reads_supporting_indel'] 
    ) # too many reads supporting indels
tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T, 'filter-indel')
dpt['indel']=dict(zip(my_calls.p,tokens))

#%% Filter positions that look iffy across samples
my_calls_raw=copy.deepcopy(my_calls)
my_calls.filter_calls_by_position( 
    my_calls.get_frac_Ns_by_position() > filter_parameter_site_across_samples['max_fraction_ambigious_samples'] 
    ) # too many samples with ambiuguous calls at this position
tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T, 'filter-max_fraction_ambigious_samples')
dpt['mfas']=dict(zip(my_calls.p,tokens))

snv.filter_histogram( 
    my_calls.get_frac_Ns_by_position(), 
    filter_parameter_site_across_samples['max_fraction_ambigious_samples'], 
    'Fraction Ns by position'
    )

my_calls_raw=copy.deepcopy(my_calls)
my_calls.filter_calls_by_position( 
    np.median( my_cmt.coverage, axis=0 ) < filter_parameter_site_across_samples['min_median_coverage_position'] 
    ) # insufficient median coverage across samples at this position
tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T, 'filter-min_median_coverage_position')
dpt['mmcp']=dict(zip(my_calls.p,tokens))
# # Optional code to make a histogram
# snv.filter_histogram( 
#     np.median( my_cmt.coverage, axis=0 ), 
#     filter_parameter_site_across_samples['min_median_coverage_position'], 
#     'Median coverage by position'
#     )
my_calls_raw=copy.deepcopy(my_calls)
copy_number_per_sample_per_pos = my_cmt.coverage / np.expand_dims( my_cov.get_median_cov_of_chromosome(), 1) # compute copy number
copy_number_avg_per_pos = np.mean( copy_number_per_sample_per_pos, axis=0 ) # mean copy number at each position
copy_number_avg_per_pos[np.isnan(copy_number_avg_per_pos)]=0
my_calls.filter_calls_by_position( 
    copy_number_avg_per_pos > filter_parameter_site_across_samples['max_mean_copynum'] 
    ) # average copy number too high
copy_number_max_per_pos = np.max( copy_number_per_sample_per_pos, axis=0 ) # mean copy number at each position
copy_number_max_per_pos[np.isnan(copy_number_max_per_pos)]=0
my_calls.filter_calls_by_position( 
    copy_number_max_per_pos > filter_parameter_site_across_samples['max_max_copynum'] 
    ) # average copy number too high
tokens = snv.token_generate(my_calls_raw.calls.T, my_calls.calls.T, 'filter-copy number')
dpt['cpn']=dict(zip(my_calls.p,tokens))
#print(my_calls.p.shape)
#print(my_calls.p)
#exit()
# # Optional code to make a histogram
# snv.filter_histogram( 
#     copy_number_avg_per_pos, 
#     filter_parameter_site_across_samples['max_mean_copynum'] , 
#     'Average copy number by position'
#     )

#%% Filter samples that have too many ambiguous calls

# Identify samples with many ambiguous basecalls and make a histogram

pos_to_consider = my_calls.p[ np.any(  my_calls.calls, axis=0 ) ] # mask positions with no basecalls in any samples
[ samples_with_toomanyNs, goodsamples_nonambig ] = snv.filter_samples_by_ambiguous_basecalls( 
    my_calls.get_frac_Ns_by_sample( pos_to_consider ), 
    filter_parameter_sample_across_sites['max_frac_Ns_to_include_sample'], 
    my_calls.sample_names, 
    my_calls.in_outgroup, # does not filter outgroup samples!!!
    True, 
    dir_output 
    )
#print(len(pos_to_consider))
#print(my_calls.get_frac_Ns_by_sample( pos_to_consider ))
#print(my_calls.sample_names)
#print(samples_with_toomanyNs)
#exit()

#print(my_cmt.p.shape)
# Filter candidate mutation table, coverage, and calls objects
#my_cmt.filter_samples( goodsamples_nonambig )
#my_cov.filter_samples( goodsamples_nonambig )
#my_calls.filter_samples( goodsamples_nonambig )
#print(my_cmt.p.shape)
#exit()


#%%##########################
## INFER ANCESTRAL ALLELES ##
#############################


#%% Part 1: Get ancestral nucleotide from outgroup

# Ancestral alleles should be inferred from an outgroup.

# Filtered calls for outgroup samples only
calls_outgroup = my_calls.get_calls_in_outgroup_only()
# Switch N's (0's) to NaNs
calls_outgroup_N_as_NaN = calls_outgroup.astype('float') # init ()
calls_outgroup_N_as_NaN[ calls_outgroup_N_as_NaN==0 ] = np.nan

# Infer ancestral allele as the most common allele among outgroup samples (could be N)
calls_ancestral = np.zeros( my_calls.num_pos, dtype='int') # init as N's
outgroup_pos_with_calls = np.any(calls_outgroup,axis=0) # positions where the outgroup has calls
calls_ancestral[outgroup_pos_with_calls] = stats.mode( calls_outgroup_N_as_NaN[:,outgroup_pos_with_calls], axis=0, nan_policy='omit' ).mode.squeeze()

# Report number of ancestral alleles inferred from outgroup
print('Number of candidate SNVs with outgroup alleles: ' + str(sum(outgroup_pos_with_calls)) + '.')
print('Number of candidate SNVs missing outgroup alleles: ' + str(sum(calls_ancestral==0)) + '.')


#%% Part 2: Fill in any missing data with nucleotide from reference

# WARNING! Rely on this method with caution especially if the reference genome 
# was derived from one of your ingroup samples.

# # Pull alleles from reference genome across p
calls_reference = my_rg.get_ref_NTs_as_ints( p )

# # Update ancestral alleles
pos_to_update = ( calls_ancestral==0 )
calls_ancestral[ pos_to_update ] = calls_reference[ pos_to_update ]



#%%#########################
## IDENTIFY SNV POSITIONS ##
############################


#%% Compute mutation quality

# Grab filtered calls from ingroup samples only
calls_ingroup = my_calls.get_calls_in_sample_subset( np.logical_not( my_calls.in_outgroup ) )
quals_ingroup = my_cmt.quals[ np.logical_not( my_calls.in_outgroup ),: ]
num_samples_ingroup = sum( np.logical_not( my_calls.in_outgroup ) )
# Note: Here we are only looking for SNV differences among ingroup samples. If
# you also want to find SNV differences between the ingroup and the outgroup
# samples (eg mutations that have fixed across the ingroup), then you need to
# use calls and quals matrices that include outgroup samples.

# Compute quality
[ mut_qual, mut_qual_samples ] = snv.compute_mutation_quality( calls_ingroup, quals_ingroup ) 
# note: returns NaN if there is only one type of non-N call


#%% Identify suspected recombination positions

# Filter params
filter_parameter_recombination = {
                                    'distance_for_nonsnp' : 1000, # region in bp on either side of goodpos that is considered for recombination
                                    'corr_threshold_recombination' : 0.75 # minimum threshold for correlation
                                    }

# Find SNVs that are are likely in recombinant regions
[ p_recombo, recombo_bool ] = snv.find_recombination_positions( \
    my_calls, my_cmt, calls_ancestral, mut_qual, my_rg, \
    filter_parameter_recombination['distance_for_nonsnp'], \
    filter_parameter_recombination['corr_threshold_recombination'], \
    True, dir_output \
    )

#print(recombo_bool.shape,my_calls.p.shape)
dpt['recomb']=dict(zip(my_calls.p,recombo_bool))
#exit()
    
# Save positions with likely recombination
if len(p_recombo)>0:
    with open( dir_output + '/snvs_from_recombo.csv', 'w') as f:
        for p in p_recombo:
            f.write(str(p)+'\n')



#%% Determine which positions have high-quality SNVs

# Filters
filter_SNVs_not_N = ( calls_ingroup != snv.nts2ints('N') ) # mutations must have a basecall (not N)
filter_SNVs_not_ancestral_allele = ( calls_ingroup != np.tile( calls_ancestral, (num_samples_ingroup,1) ) ) # mutations must differ from the ancestral allele
filter_SNVs_quals_not_NaN = ( np.tile( mut_qual, (num_samples_ingroup,1) ) >= 1) # alleles must have strong support 
filter_SNVs_not_recombo = np.tile( np.logical_not(recombo_bool), (num_samples_ingroup,1) ) # mutations must not be due to suspected recombination

# Fixed mutations per sample per position
fixedmutation = \
    filter_SNVs_not_N \
    & filter_SNVs_not_ancestral_allele \
    & filter_SNVs_quals_not_NaN \
    & filter_SNVs_not_recombo # boolean

#print(fixedmutation.shape)
#exit()

hasmutation = np.any( fixedmutation, axis=0) # boolean over positions (true if at lest one sample has a mutation at this position)

goodpos_bool = np.any( fixedmutation, axis=0 )
#print(goodpos_bool.shape)
#exit()
goodpos_idx = np.where( goodpos_bool )[0]
#print(goodpos_idx)
tokens_final=snv.generate_tokens_last(tokens,goodpos_idx,'filter-fixedmutation')
dpt['fix']=dict(zip(my_calls.p,tokens_final))
#exit()
num_goodpos = len(goodpos_idx)
print('Num mutations identified by WideVariant: '+str(num_goodpos))

####### Combine CNN and WideVariant output and generate the SNV information table #######
goodpos_idx_cnn=cnn_pos[np.where(cnn_pred==1)]
#print(goodpos_idx_cnn)
#exit()
goodpos_idx_wd=my_calls.p[goodpos_idx]
all_p=np.sort(np.union1d(goodpos_idx_cnn,goodpos_idx_wd))
#all_p=my_calls.p[all_p]
#print(all_p)
#print(np.where(my_cmt.p==1095218))
#exit()
goodpos_bool,goodpos_bool_all=snv.generate_cnn_filter_table(all_p,goodpos_idx_wd,dpt,dlab,dprob,dir_output,my_cmt.p,dgap,my_cmt_zero,min_cov_filt)
goodpos_idx = np.where( goodpos_bool )[0]
#print(goodpos_bool.shape)
#exit()
#print(goodpos_bool,goodpos_bool.shape)
#print(goodpos_idx)
#exit()
goodpos_idx_all = np.where( goodpos_bool_all)[0]
num_goodpos = len(goodpos_idx)
num_goodpos_all = len(goodpos_idx_all)
print('Num mutations identified by CNN+WideVariant+Recomb_filt: '+str(num_goodpos_all))
print('Num mutations identified by CNN+Recomb_filt:'+str(num_goodpos))
if greport==0:
    print('Find -g 0. No report and extra files will be generated...')
    exit()
#pos_to_consider = my_calls.p[ np.any(  my_calls.calls, axis=0 ) ] # mask positions with no basecalls in any samples

[ samples_with_toomanyNs, goodsamples_nonambig ] = snv.filter_samples_by_ambiguous_basecalls(
    my_calls.get_frac_Ns_by_sample( goodpos_idx ),
    filter_parameter_sample_across_sites['max_frac_Ns_to_include_sample'],
    my_calls.sample_names,
    my_calls.in_outgroup, # does not filter outgroup samples!!!
    True,
    dir_output
    )
#print(my_calls.sample_names)
#print(my_calls.get_frac_Ns_by_sample( goodpos_idx ))
#print(samples_with_toomanyNs)
#exit()


#%% Make and annotate a SNV table

# Prepare data for SNV table

my_cmt_goodpos = my_cmt.copy()
my_cmt_goodpos.filter_positions( goodpos_bool )
my_cmt_goodpos_ingroup = my_cmt_goodpos.copy()
my_cmt_goodpos_ingroup.filter_samples( np.logical_not( my_cmt_goodpos_ingroup.in_outgroup ) )

my_calls_goodpos = my_calls.copy()
my_calls_goodpos.filter_positions( goodpos_bool )
calls_goodpos = my_calls_goodpos.calls
calls_goodpos_ingroup = calls_goodpos[ np.logical_not( my_calls_goodpos.in_outgroup ),: ]

p_goodpos = my_calls_goodpos.p

calls_ancestral_goodpos = calls_ancestral[ goodpos_bool ]

# Only for bar charts
my_cmt_goodpos_all = my_cmt.copy()
#print(np.where(my_cmt.p==1095218))
#print(goodpos_bool_all[np.where(my_cmt.p==1095218)[0]])
my_cmt_goodpos_all.filter_positions( goodpos_bool_all )
#print(my_cmt_goodpos_all.p)
#exit()
my_cmt_goodpos_ingroup_all = my_cmt_goodpos_all.copy()
my_cmt_goodpos_ingroup_all.filter_samples( np.logical_not( my_cmt_goodpos_ingroup_all.in_outgroup ) )

my_calls_goodpos_all= my_calls.copy()
my_calls_goodpos_all.filter_positions( goodpos_bool_all )
calls_goodpos_all = my_calls_goodpos_all.calls
calls_goodpos_ingroup_all = calls_goodpos_all[ np.logical_not( my_calls_goodpos_all.in_outgroup ),: ]
p_goodpos_all = my_calls_goodpos_all.p
calls_ancestral_goodpos_all = calls_ancestral[ goodpos_bool_all ]

# Generate SNV table

# Parameters
promotersize = 250; # how far upstream of the nearest gene to annotate something a promoter mutation (not used if no annotation)

# Make a table (pandas dataframe) of SNV positions and relevant annotations
mutations_annotated = snv.annotate_mutations( \
    my_rg, \
    p_goodpos_all, \
    np.tile( calls_ancestral[goodpos_idx_all], (my_cmt_goodpos_ingroup_all.num_samples,1) ), \
    calls_goodpos_ingroup_all, \
    my_cmt_goodpos_ingroup_all, \
    fixedmutation[:,goodpos_idx_all], \
    mut_qual[:,goodpos_bool_all].flatten(), \
    promotersize \
    ) 

#print(mutations_annotated)
#exit()
#%% SNV quality control plots

# Note: These data visualizations are intended to help you evaluate if your SNV
# filters are appropriate. Do not proceed to the next step until you are 
# convinced that your filters are strict enough to filter low-quality SNVs, but
# not so strict that good SNVs are eliminated as well. 


#print(p_goodpos_all)
#exit()
# Clickable bar charts for each SNV position

snv.plot_interactive_scatter_barplots( \
    p_goodpos_all, \
    mut_qual[0,goodpos_idx_all], \
    'pos', \
    'qual', \
    my_cmt_goodpos_all.sample_names, \
    mutations_annotated, \
    my_cmt_goodpos_all.counts,dir_output,False)

#exit()
# Heatmaps of basecalls, coverage, and quality over SNV positions
if 300>num_goodpos_all>0:
    snv.make_calls_qc_heatmaps( my_cmt_goodpos_all, my_calls_goodpos_all, True, dir_output,False )



#%%###########################
## PARSIMONY AND TREEMAKING ##
##############################

#%% Filter calls for tree

# Note: Here we are using looser filters than before

# Choose subset of samples or positions to use in the tree by idx
samplestoplot = np.arange(my_cmt_goodpos_all.num_samples) # default is to use all samples 
goodpos4tree = np.arange(num_goodpos_all) # default is to use all positions
#print(goo)

# Get calls for tree
my_calls_tree = snv.calls_object( my_cmt_goodpos_all ) # re-initialize calls

# Apply looser filters than before (want as many alleles as possible)
filter_parameter_calls_for_tree = {
                                    'min_cov_for_call' : 1, # on individual samples, calls must have at least this many fwd+rev reads
                                    'min_qual_for_call' : 30, # on individual samples, calls must have this minimum quality score
                                    'min_major_nt_freq_for_call' : 0.75,  # on individual samples, a call's major allele must have at least this freq
                                    }

my_calls_tree.filter_calls_by_element( 
    my_cmt_goodpos_all.coverage < filter_parameter_calls_for_tree['min_cov_for_call'] 
    ) # forward strand coverage too low

my_calls_tree.filter_calls_by_element( 
    my_cmt_goodpos_all.quals < filter_parameter_calls_for_tree['min_qual_for_call'] 
    ) # quality too low

my_calls_tree.filter_calls_by_element( 
    my_cmt_goodpos_all.major_nt_freq < filter_parameter_calls_for_tree['min_major_nt_freq_for_call'] 
    ) # major allele frequency too low

# Make QC plots again using calls for the tree
if 300>num_goodpos_all>0:
    snv.make_calls_qc_heatmaps( my_cmt_goodpos_all, my_calls_tree, False, dir_output, False )

# %% Make a tree

if num_goodpos_all > 0:

    # Get nucleotides of basecalls (ints to NTs)
    calls_for_treei = my_calls_tree.calls[
        np.ix_(samplestoplot, goodpos4tree)]  # numpy broadcasting of row_array and col_array requires np.ix_()
    calls_for_tree = snv.ints2nts(calls_for_treei)  # NATCG translation

    # Sample names for tree
    treesampleNamesLong = my_cmt_goodpos_all.sample_names
    for i, samplename in enumerate(treesampleNamesLong):
        if not samplename[0].isalpha():
            treesampleNamesLong[i] = 'S' + treesampleNamesLong[
                i]  # sample names are modified to make parsing easier downstream
    sampleNamesDnapars = ["{:010d}".format(i) for i in range(my_cmt_goodpos_all.num_samples)]

    # Add inferred ancestor and reference
    calls_ancestral_for_tree = np.expand_dims(snv.ints2nts(calls_ancestral_goodpos_all), axis=0)
    calls_reference_for_tree = np.expand_dims(my_rg.get_ref_NTs(my_calls_tree.p), axis=0)
    calls_for_tree_all = np.concatenate((calls_ancestral_for_tree, calls_reference_for_tree, calls_for_tree),
                                        axis=0)  # first column now outgroup_nts; outgroup_nts[:, None] to make ndims (2) same for both
    sampleNamesDnapars_all = np.append(['Sanc', 'Sref'], sampleNamesDnapars)
    treesampleNamesLong_all = np.append(['inferred_ancestor', 'reference_genome'], treesampleNamesLong)
    
    try:
        # Build tree
        snv.generate_tree( \
            calls_for_tree_all.transpose(), \
            treesampleNamesLong_all, \
            sampleNamesDnapars_all, \
            ref_genome_name, \
            dir_output, \
            "snv_tree_" + ref_genome_name, \
            buildTree='PS' \
            )
    except:
        print('#### error skip #####: something wrong in snv.generate_tree... skip...')



#%%###################
## SAVE DATA TABLES ##
######################


#%% Write SNV table to a tsv file

# Note: important to use calls for tree (rather than calls for finding SNVs)

# Make table
if num_goodpos>0:
    # This is the raw mutation table - contain positions identified by both CNN and WD, and are not recombinations
    output_tsv_filename = dir_output + '/' + 'snv_table_mutations_annotations_raw.tsv'
    snv.write_mutation_table_as_tsv( \
        p_goodpos_all, \
        mut_qual[0,goodpos_idx_all], \
        my_cmt_goodpos_all.sample_names, \
        mutations_annotated, \
        calls_for_tree, \
        treesampleNamesLong, \
        output_tsv_filename \
        
        )
    out_merge_tsv=dir_output+'/snv_table_merge_all_mut_annotations_draft.tsv'
    snv.merge_two_tables(dir_output+'/snv_table_cnn_plus_filter.txt',output_tsv_filename,out_merge_tsv)
    #exit()
    snv.generate_html_with_thumbnails(dir_output+'/snv_table_merge_all_mut_annotations_draft.tsv', dir_output+'/snv_table_with_charts_draft.html', dir_output+'/bar_charts')
    # Generate the tree for each identified SNPs
    try:
        bst.mutationtypes(dir_output+"/snv_tree_genome_latest.nwk.tree",dir_output+'/snv_table_merge_all_mut_annotations_draft.tsv',dir_output)
    except:
        print('#### error skip #####: something wrong in bst.mutationtypes... skip...')
    # # Contain all positions identified by CNN or WideVariant - even those false positions
    # snv.write_mutation_table_as_tsv( \
    #     p_goodpos_all, \
    #     mut_qual[0, goodpos_idx_all], \
    #     my_cmt_goodpos_all.sample_names, \
    #     mutations_annotated, \
    #     calls_for_tree, \
    #     treesampleNamesLong, \
    #     output_tsv_filename \
    #     )

# Rebuild output for new requirement - 2025-03-21 - Herui
'''
update - 1 - save candidate mutation table with only label '1' pos
update - 2 - regenerate output text and html report
update - 3 - add dN/dS output
'''
f=open(dir_output+'/snv_table_merge_all_mut_annotations_draft.tsv','r')
o=open(dir_output+'/snv_table_merge_all_mut_annotations_final.tsv','w+')
o2=open(dir_output+'/snv_table_merge_all_mut_annotations_label0.tsv','w+')
line=f.readline()
o.write(line)
o2.write(line)
dk={}
dl={}
dr={}
while True:
    line=f.readline().strip()
    if not line:break
    ele=line.split('\t')
    if int(ele[1])==0:
        #continue
        o2.write(line+'\n')
        dl[int(ele[0])]=''
        if ele[4]=='skip':
            dk[int(ele[0])]=0
        else:
            dk[int(ele[0])]=float(ele[4])
    else:
        o.write(line+'\n')
        if ele[4]=='skip':
            dk[int(ele[0])]=0
        else:
            dk[int(ele[0])]=float(ele[4])
    if int(ele[13])==1:
        dr[int(ele[0])]=int(ele[13])
o.close()
o2.close()
snv.generate_html_with_thumbnails(dir_output+'/snv_table_merge_all_mut_annotations_final.tsv', dir_output+'/snv_table_with_charts_final.html', dir_output+'/bar_charts')
if len(dl)>0:
    snv.generate_html_with_thumbnails(dir_output+'/snv_table_merge_all_mut_annotations_label0.tsv', dir_output+'/snv_table_with_charts_label0.html', dir_output+'/bar_charts')
keep_p=[]
prob=[]
label=[]
recomb=[]
for s in my_cmt_zero_rebuild.p:
    if s in dl:
        label.append(False)
    else:
        label.append(True)
    if s in dk:
        keep_p.append(True)
        prob.append(dk[s])
    else:
        keep_p.append(False)
    if s in dr:
        recomb.append(True)
    else:
        recomb.append(False)


keep_p=np.array(keep_p)
my_cmt_zero_rebuild.filter_positions(keep_p)
label=np.array(label)
recomb=np.array(recomb)
quals_new=my_cmt_zero_rebuild.quals* -1
new_cmt={'sample_names': my_cmt_zero_rebuild.sample_names,'p':my_cmt_zero_rebuild.p,'counts':my_cmt_zero_rebuild.counts,'quals':quals_new,'in_outgroup':my_cmt_zero_rebuild.in_outgroup,'indel_counter':my_cmt_zero_rebuild.indel_stats,'prob':prob,'label':label,'recomb':recomb,'samples_exclude_bool':samples_to_exclude_bool}
np.savez_compressed(dir_output+'/candidate_mutation_table_final.npz', **new_cmt)

# dN/dS part
sys.path.insert(0, './miniscripts_for_dNdS')
dir_ref_genome = refg
annotation_full = pd.read_csv(dir_output+'/snv_table_merge_all_mut_annotations_final.tsv',sep='\t')
output_directory = dir_output+'/dNdS_out'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
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

exit()



#%% Write table of tree distances to a csv file

if num_goodpos>0:
    
    # Compute number of SNVs separating each sample from the inferred ancestor
    fixedmutation_tree = ( calls_for_tree != calls_ancestral_for_tree ) & ( calls_for_tree != 'N' ) # boolean
    
    dist_to_anc_by_sample = np.sum( fixedmutation_tree, axis=1 )
    # Save to a file
    with open( dir_output + '/snv_table_tree_distances.csv', 'w') as f:
        f.write('sample_name,num_SNVs_to_ancestor\n')
        for i,name in enumerate(treesampleNamesLong):
            f.write(name + ',' + str(dist_to_anc_by_sample[i]) + '\n' )


#%% Write a table of summary stats

print( 'Number of samples in ingroup: ' + str(num_samples_ingroup) + '.') 
print( 'Number of good SNVs found: ' + str(num_goodpos) + '.') 
print('Number of good SNVs with outgroup alleles: ' + str(sum(calls_ancestral_goodpos>0)) + '.')
print('Number of good SNVs missing outgroup alleles: ' + str(sum(calls_ancestral_goodpos==0)) + '.')
if num_goodpos>0:
    dist_to_anc_by_sample_ingroup = dist_to_anc_by_sample[ np.logical_not(my_calls_tree.in_outgroup) ]
    dmrca_median = np.median( dist_to_anc_by_sample_ingroup )
    dmrca_min = np.min( dist_to_anc_by_sample_ingroup )
    dmrca_max = np.max( dist_to_anc_by_sample_ingroup )
else:
    dmrca_median = 0
    dmrca_min = 0
    dmrca_max = 0
print( 'Median dMRCA: ' + str(dmrca_median) + '.') 
print( 'Min dMRCA: ' + str(dmrca_min) + '.') 
print( 'Max dMRCA: ' + str(dmrca_max) + '.') 

with open( dir_output + '/snv_table_simple_stats.csv', 'w') as f:
    f.write('dataset,genome,num_samples_ingroup,num_snvs,num_snvs_with_outgroup_allele,dmrca_median,dmrca_min,dmrca_max,\n')
    f.write(dataset_name+','+ref_genome_name+','+str(num_samples_ingroup)+','+str(num_goodpos)+','+str(sum(calls_ancestral_goodpos>0))+','+str(dmrca_median)+','+str(dmrca_min)+','+str(dmrca_max)+',\n')



#%%##################
## CONTIG ANALYSIS ##
#####################

# This section examines contig coverage.


#%% Make a plot of presence/absence of each contig

my_cov.plot_heatmap_contig_copy_num( dir_output ,False)

# Record data in files
snv.write_generic_csv( my_cov.median_coverage_by_contig, my_cov.contig_names, my_cov.sample_names, dir_output+'/contig_table_median_coverage.csv' )
snv.write_generic_csv( my_cov.copy_number_by_contig, my_cov.contig_names, my_cov.sample_names, dir_output+'/contig_table_copy_number.csv' )


#%% Make plots of coverage traces of each contig

if type(my_cov) == snv.cov_data_object:
    for contig_num in np.linspace(1,my_cov.num_contigs,my_cov.num_contigs).astype(int):
        print('Generating copy number traces for contig ' + str(contig_num) + '...')
        my_cov.make_coverage_trace(contig_num,100,dir_output,False);
elif type(my_cov) == snv.cov_data_object_simple:
    print('Coverage matrix object type is cov_data_obj_simple, not cov_data_obj. Raw coverage matrix not available for copy number traces.')
