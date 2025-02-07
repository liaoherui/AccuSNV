#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

SUMMARY:

    This module contains the essential objects and methods for calling SNVs and 
    generating a parsimony tree from whole genome sequencing data that has been
    processed through the Lieberman Lab's standard Snakemake pipeline.
    

BEST PRACTICES FOR DATA STRUCTURES AND INDEXING:
    
    WARNING! This was not previously standardized across the Lieberman Lab code
    base. If you incorporate old code into this module for additional 
    functionality, you may need to update data types or indexing.

    This module implements the following standards:
    
    DATA TYPES:
        
        Numerical arrays are numpy arrays and employ numpy methods (not generic
        python methods).
        
        Heterogeneous arrays are pandas dataframes.
        
        Custom classes exist for:
            * Candidate mutation table data (cmt_data_object): collection of 
            associated numpy arrays.
            * Coverage data (cov_data_object_simple): simple summary of raw
            coverage data generated from raw coverage matrix.
            * Basecalls data (calls_object): tracks array of basecalls and 
            facilitates filtering.
            * Reference genome data (reference_genome_object): keeps track of 
            data relating to the reference genome (contigs, annotations, etc).
        
    INDEXING:
        
        Nucleotide indexing: 
            
            Defined in dictionaries nts2ints and ints2nts (need two 
            dictionaries because dictionaries are not 1 to 1 functions).
            
            N = 0
            A/T/C/G = 1/2/3/4

        Reference genome indexing: 
    
            * Positions on the reference genome are counted starting from 1 in 
            order to match the VCF files.
            * Contigs on the reference genome are also counted starting from 1. 
    
            (All python indexing still starts at 0. The above just refers to 
            how we are naming positions and contigs.)
        
        Data objects:
            
            Axis order in numerical matrices follows the following heirarchy:
                
                (earliest index)
                Sample
                Position on genome
                Other characteristics
                (latest index)
                
            Examples:
                
                Coverage array: 
                    Axis 0 = sample name
                    Axis 1 = position on genome
                    
                Counts array:
                    Axis 0 = sample name
                    Axis 1 = position on genome
                    Axis 2 = ATCGatcg read counts
                    
    DOCSTRINGS: 
        
        Docstrings follow the standard numpy template.


OTHER NOTES:
    
    Areas for potential improvement are marked with #TODO.
    
    
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
Felix Key, Arolyn Conwill, A. Delphine Tripp, Evan Qu, Laura Markey

'''

# IMPORT PYTHON PACKAGES

# General
import os
import glob
import pickle
import copy as cp # https://docs.python.org/3/library/copy.html
import gzip
import time,datetime
import subprocess
import warnings

# Numbers and data
import numpy as np
import pandas as pd
from collections import OrderedDict

# Bioinformatics
from Bio import SeqIO
from BCBio import GFF  # pip install bcbio-gff
from Bio.Seq import Seq
from Bio import Phylo
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor #NJ tree


# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib import rc
# import matplotlib.cm as cm
# import matplotlib.mlab as mlab
# from matplotlib.font_manager import FontProperties
import pylab as pl




###########################
# NUCLEOTIDE DICTIONARIES #
###########################

# Dictionaries to convert nucleotides to integers
NTs_to_int_dict = { 'N':0, 'n':0, 'A':1, 'a':1, 'T':2, 't':2, 'C':3, 'c':3, 'G':4, 'g':4 } # must convert lowercase NTs to ints
int_to_NTs_dict = { 0:'N', 1:'A', 2:'T', 3:'C', 4:'G' } # only need to convert ints back to uppercase NTs
# note: best practice is to define these here and never need to define these anywhere else (i.e. in main script)

# Functions to convert nucleotides to integers
def nts2ints( np_array_of_NTs ):
    ''' Converts NTs to integers according to dictionary NTs_to_int_dict. Defaults to 0 if key is missing. '''
    return np.vectorize(NTs_to_int_dict.get)(np_array_of_NTs,0) # vectorize much faster than looping; requires numpy array input

# Function to convert integers to nucleotides
def ints2nts( np_array_of_ints ):
    ''' Converts integers to NTs according to dictionary int_to_NTs_dict. '''
    return np.vectorize(int_to_NTs_dict.__getitem__)(np_array_of_ints) # vectorize much faster than looping; require numpy array input

# Function for getting reverse complement
NTs_complement_dict = { 'N':'N', 'A':'T', 'T':'A', 'C':'G', 'G':'C'} 
def nts_rev_comp( np_array_of_NTs ):
    ''' Gets reverse complement of sequence. Input must be uppercase letters. '''
    # Get complement and then reverse it
    NTs_complement = np.vectorize(NTs_complement_dict.get)(np_array_of_NTs,'N') # vectorize much faster than looping; require numpy array input
    NTs_reverse_complement = np.flip( NTs_complement )
    return NTs_reverse_complement

# List for looping through nucleotides (NOT for encoding nucleotides in data)
NTs_list_without_N = ['A','T','C','G']
NTs_list_without_N_to_idx_dict = { 'A':0, 'T':1, 'C':2, 'G':3 }


###############################################
# CANDIDATE MUTATION TABLE: CLASS DEFINITION  #
###############################################


# Import file

def read_candidate_mutation_table_npz(file_cmt_npz):
    '''
    Read candidate_mutation_table.pickle.gz.npz file (new version).
    
    NOTES
    -----

        New dimensions for candidate mutation table data:
            
            quals: num_samples x num_pos
            p: num_pos
            counts: num_samples x num_pos x 8
            in_outgroup: num_samples 
            sampleNames: num_samples
            indel_counter: num_samples x num_pos x 2
    
        For importing old candidate mutation tables, please use the function
        read_old_candidate_mtuation_table_pickle_gzip instead.

    '''
    
    # Read file
    with open(file_cmt_npz, 'rb') as f:
        cmt = np.load(f)
        sample_names = np.array(cmt['sample_names'])
        p = np.array(cmt['p'])
        counts = np.array(cmt['counts'])
        quals = (np.array(cmt['quals']) * -1)
        in_outgroup = np.array(cmt['in_outgroup'],dtype=bool).flatten()
        indel_counter = np.array(cmt['indel_counter'])

    # Return arrays
    return [ quals, p, counts, in_outgroup, sample_names, indel_counter ]

def read_candidate_mutation_table_pickle_gzip(file_cmt_pickle_gz):
    '''
    Read candidate_mutation_table.pickle.gz file (new version).
    
    NOTES
    -----

        New dimensions for candidate mutation table data:
            
            quals: num_samples x num_pos
            p: num_pos
            counts: num_samples x num_pos x 8
            in_outgroup: num_samples 
            sampleNames: num_samples
            indel_counter: num_samples x num_pos x 2
    
        For importing old candidate mutation tables, please use the function
        read_old_candidate_mtuation_table_pickle_gzip instead.

    '''
    
    # Read file
    with gzip.open(file_cmt_pickle_gz, 'rb') as f:
        cmt = pickle.load(f)
        # Object by object
        sample_names = np.array( cmt.get('sample_names') )
        p = np.array( cmt.get('p') )
        counts = np.array( cmt.get('counts') )
        quals = np.array( cmt.get('quals') ) * -1
        in_outgroup = np.array( cmt.get('in_outgroup') )
        indel_counter = np.array( cmt.get('indel_counter') )

    # Return arrays
    return [ quals, p, counts, in_outgroup, sample_names, indel_counter ]

def read_old_candidate_mutation_table_pickle_gzip(file_cmt_pickle_gz):
    '''
    Read candidate_mutation_table.pickle.gz file (old version). 
    
    NOTES
    -----
    
        Old dimensions for candidate mutation table data are updated:
            
            quals: num_pos x num_samples --> num_samples x num_pos
            p: num_pos --> no change
            counts: 8 x num_pos x num_samples --> num_samples x num_pos x 8
            in_outgroup: num_samples --> change to boolean
            sampleNames: num_samples --> no change
            indel_counter: 2 x num_pos x num_samples
        
        For importing new candidate mutation tables, please use the function
        read_candidate_mtuation_table_pickle_gzip instead.
    '''
    
    # Read file
    [ quals, p, counts, in_outgroup, sampleNames, indel_counter ] = read_candidate_mutation_table_pickle_gzip(file_cmt_pickle_gz)
    
    # Convert old array format into new array format
    new_sample_names = sampleNames # no change needed
    new_outgroup_bool = in_outgroup[0,0].split(' ')
    new_outgroup_bool = np.array( [int(i)>0 for i in new_outgroup_bool] )
    new_p = p # no change needed
    new_counts = np.array( counts ).astype(np.int_)
    new_counts = new_counts.swapaxes(0,2)
    new_quals = np.array( quals )
    new_quals = new_quals.swapaxes(0,1)
    new_indel_counter = np.array( indel_counter ).astype(np.int_)
    new_indel_counter = new_indel_counter.swapaxes(0,2)    
    
    # Return arrrays
    return [ new_quals, new_p, new_counts, new_outgroup_bool, new_sample_names, new_indel_counter ]


# Candidate mutation table class definition

class cmt_data_object:
    '''
    This class keeps track of candidate mutation table data.
        
    ARGUMENTS
    ---------
    
        sample_names_list: list of sample names; numpy array of strings
        
        in_outgroup_bool: list of booleans indicating which samples are 
        ougroups; numpy array of booleans
        
        positions_list: list of candidate SNV positions on genome; numpy array 
        of ints
        
        counts_array: array counting the number of forward and reverse reads 
        supporting each nucleotide; numpy array of ints; dimensions = (num 
        samples) x (num candidate SNV positions) x 8
        
        quals_array: array of basecall quality; numpy array of ints; dimensions
        = (num samples) x (num candidate SNV positions)
        
        indel_stats_array = array counting the number of reads supporting 
        indels (insertions or deletions); numpy array of ints; dimensions = 
        (num samples) x (num candidate SNV positions) x 2
        
        my_dataset_name = optional input for the name of a dataset; string
        
    ATTRIBUTES
    ----------
    
        dataset_name: name of dataset
        
        sample_names: array of sample names
        
        num_samples: number of samples
        
        in_outgroup: boolean array indicating if sample is an outgroup
        
        p: positions on genome where there are candidate SNVs
        
        num_pos: number of candidate SNV positions
        
        counts: number of reads supporting ATCGatcg (fwd/rev) for each sample 
        at each candidate SNV position
        
        quals: basecall quality for each sample at each candidate SNV position
        
        indel_stats: number of reads supporting insertions or deletions for
        each sample at each candidate SNV position
        
        coverage: read coverage for each sample at each candidate SNV position
        
        fwd_cov: read coverage from forward reads only for each sample at each
        candidate SNV position
        
        rev_cov: read coverage from reverse reads only for each sample at each
        candidate SNV position
        
        major_nt: most abundant basecall for each sample at each candidate SNV
        position
        
        major_nt: next most abundant basecall for each sample at each candidate 
        SNV position
        
        major_nt_freq = frequency of major NT allele for each sample at each 
        candidate SNV position
    
        minor_nt_freq = frequency of minor NT allele for each sample at each 
        candidate SNV position

    METHODS
    -------
    
        init: generates candidate mutation table object based on input arrays
        
        filter_samples: filters candidate mutation table along samples axis; 
        downsizes all attributes along sample axis based on boolean argument
        
        filter_positions: filters candidate mutation table along position axis; 
        downsizes all attributes along position axis based on boolean argument
        
        copy: returns a copy of itself
        
    RAISES
    ------
    
        Raises errors if inputs are not the correct data type or dimensions.

    NOTES
    -----

        ...

    @author: Arolyn Conwill
    
    '''       
    
    def __init__(self, sample_names_list, in_outgroup_bool, positions_list, counts_array, quals_array, indel_stats_array, my_dataset_name='My candidate mutation table' ): 
        ''' 
        Generates candidate mutation table object. 
        
        Checks that all arguments are the correct type and dimension. 
        * Input objects must be numpy arrays of the appropriate type.
        * Dimensions of arrays must confrom to: num_samples x num_pos x (third dimension where applicable).
        '''
        
        # Dataset name
        self.dataset_name = my_dataset_name
        
        # Sample names
        # sample_names
        try:
            if sample_names_list.dtype.type == np.str_:
                self.sample_names = sample_names_list # sample names
                self.num_samples = len( self.sample_names ) # compute number of samples
                print( "Number of samples in candidate mutation table: " + str(self.num_samples) + "." )
            else:
                raise Exception("Argument sample_names_list must be numpy array of strings.")
        except AttributeError: # no dtype
            raise Exception("Argument sample_names_list must be a numpy array.")
        # Outgroup boolean
        # in_outgroup
        try:
            if in_outgroup_bool.dtype.type == np.bool_:
                if in_outgroup_bool.shape == (self.num_samples,):
                    self.in_outgroup = in_outgroup_bool
                else:
                    raise Exception("Outgroup boolean array dimensions are " + str(in_outgroup_bool.shape) + ", but should be (" + str(self.num_samples) + ",)." )
            else:
                raise Exception("Argument in_outgroup_bool must be numpy array of booleans.")
        except AttributeError: # no dtype
            raise Exception("Argument in_outgroup_bool must be a numpy array.")
    
        # Candidate SNV positions
        # p
        try:
            if np.issubdtype(positions_list.dtype, np.integer):
                self.p = positions_list # candidate SNV positions on genome
                self.num_pos = len( self.p );
                print( "Number of genome positions in candidate mutation table: " + str(self.num_pos) + "." )
            else:
                raise Exception("Argument positions_list must be numpy array of integers.")
        except AttributeError: # no dtype
            raise Exception("Argument positions_list must be a numpy array.")
        
        # Candidate SNV statistics from snakemake step
        # counts
        try:
            if np.issubdtype(counts_array.dtype, np.integer):
                if counts_array.shape == ( self.num_samples,self.num_pos,8):
                    self.counts = counts_array
                else:
                    raise Exception("Counts array dimensions are " + str(counts_array.shape) + ", but should be (" + str(self.num_samples) + ", " + str(self.num_pos) + ", 8)." )
            else:
                    raise Exception("Argument counts_array must be numpy array of integers.")
        except AttributeError: # no dtype
            raise Exception("Argument counts_array must be a numpy array.")
        # quals
        try:
            if np.issubdtype(quals_array.dtype, np.integer):
                if quals_array.shape == ( self.num_samples,self.num_pos):
                    self.quals = quals_array
                else:
                    raise Exception("Quals array dimensions are " + str(quals_array.shape) + ", but should be (" + str(self.num_samples) + ", " + str(self.num_pos) + ")." )
            else:
                raise Exception("Argument quals_array must be numpy array of integers.")
        except AttributeError: # no dtype
            raise Exception("Argument quals_array must be a numpy array.")
        # indel_stats
        try:
            if np.issubdtype(indel_stats_array.dtype, np.integer):
                if indel_stats_array.shape == ( self.num_samples,self.num_pos,2):
                    self.indel_stats = indel_stats_array
                else:
                    raise Exception("Indel stats array dimensions are " + str(counts_array.shape) + ", but should be (" + str(self.num_samples) + ", " + str(self.num_pos) + ", 2)." )
            else:
                raise Exception("Argument indel_stats_array must be numpy array of integers.")
        except AttributeError:
            raise Exception("Argument indel_stats_array must be a numpy array.")
        
        # Compute coverage from candidate SNV counts array
        # total coverage
        self.coverage = np.zeros( (self.num_samples,self.num_pos), dtype='int') # init coverage array
        np.sum( self.counts, axis=2, out=self.coverage ) # compute with specified output
        # forward read coverage
        self.fwd_cov = np.zeros( (self.num_samples,self.num_pos), dtype='int') # init forward coverage array
        np.sum( self.counts[:,:,0:4], axis=2, out=self.fwd_cov ) # compute with specified output
        # reverse read coverage
        self.rev_cov = np.zeros( (self.num_samples,self.num_pos), dtype='int') # init reverse coverage array
        np.sum( self.counts[:,:,4:8], axis=2, out=self.rev_cov ) # compute with specified output
        
        # Compute major and minor allele identities and frequencies
        # major_nt, minor_nt, major_nt_freq, minor_nt_freq
        # examine number of reads supporting each nucleotide at each position in each sample
        counts_by_allele = self.counts[:,:,0:4] + self.counts[:,:,4:8] # flatten fwd and rev nucleotide counts
        # get major and minor nucleotide frequencies
        # note: minor allele frequency ignores cases where three or four alleles are present in the sample
        counts_sort = np.sort(counts_by_allele,axis=2) # sort number of reads for each nucleotide
        counts_major = np.squeeze( counts_sort[:,:,3:4], axis=2 ) # number of reads for most common nucleotide
        counts_minor = np.squeeze( counts_sort[:,:,2:3], axis=2 ) # number of reads for next most common nucleotide
        with np.errstate(divide='ignore',invalid='ignore'): # suppress warning for division by zero
            self.major_nt_freq = counts_major / self.coverage # add major allele frequency attribute
            self.minor_nt_freq = counts_minor / self.coverage # add minor allele frequency attribute
        self.major_nt_freq[ np.isnan(self.major_nt_freq) ] = 0 # set major allele frequency to zero to indicate there is no data; leave minor allele frequency as nan 
        # get major and minor nucleotide identities
        # note: if counts for all bases are zero, sort will not change the order, so the major alelle will always be the fourth nucleotide and the minor allele will always be the third nucleotide
        counts_argsort = np.argsort(counts_by_allele,axis=2) # sort idx of nucleotides by number of reads
        self.major_nt = 1 + np.squeeze( counts_argsort[:,:,3:4],axis=2 ) # add major alelle attribute # 3:4 necessary to maintain 3d structure # +1 necessary because 0=N and 1-4=ATCG 
        self.minor_nt = 1 + np.squeeze( counts_argsort[:,:,2:3],axis=2 ) # add minor allele attribute # 2:3 necessary to maintain 3d structure # +1 necessary because 0=N and 1-4=ATCG
        
        
    def filter_samples(self,samples_to_keep_bool):
        ''' Filters samples and updates all candidate mutation table attributes accordingly. '''
        try:
            if ( samples_to_keep_bool.dtype.type == np.bool_ ) and ( samples_to_keep_bool.size == self.num_samples ):
                # downsize attributes along samples dimension according to samples_to_keep_bool
                num_samples_old = self.num_samples # record original number of samples
                self.sample_names = self.sample_names[samples_to_keep_bool]
                self.num_samples = np.count_nonzero(samples_to_keep_bool)
                self.in_outgroup = self.in_outgroup[samples_to_keep_bool]
                self.counts = self.counts[samples_to_keep_bool,:,:]
                self.quals = self.quals[samples_to_keep_bool,:]
                self.indel_stats  = self.indel_stats[samples_to_keep_bool,:,:]
                self.coverage = self.coverage[samples_to_keep_bool,:]
                self.fwd_cov = self.fwd_cov[samples_to_keep_bool,:]
                self.rev_cov = self.rev_cov[samples_to_keep_bool,:]
                self.major_nt = self.major_nt[samples_to_keep_bool,:]
                self.minor_nt = self.minor_nt[samples_to_keep_bool,:]
                self.major_nt_freq = self.major_nt_freq[samples_to_keep_bool,:]
                self.minor_nt_freq = self.minor_nt_freq[samples_to_keep_bool,:]
                # print results
                print( "Number of samples in candidate mutation table reduced from " + str(num_samples_old) + " to " + str(self.num_samples) + "." )
            else:
                raise Exception("Argument samples_to_keep_bool must be a numpy array of booleans with size num_samples.")
        except AttributeError:
            raise Exception("Argument samples_to_keep_bool must be a numpy array.")

    
    def filter_positions(self,positions_to_keep_bool):
        ''' Filters positions and updates all candidate mutation table attributes accordingly. '''
        try:
            if ( positions_to_keep_bool.dtype.type == np.bool_ ) & ( positions_to_keep_bool.size == self.num_pos ):
                # downsize attributes along samples dimension according to positions_to_keep_bool
                num_pos_old = self.num_pos # record original number of positions
                self.p = self.p[positions_to_keep_bool]
                self.num_pos = np.count_nonzero(positions_to_keep_bool)
                self.counts = self.counts[:,positions_to_keep_bool,:]
                self.quals = self.quals[:,positions_to_keep_bool]
                self.indel_stats  = self.indel_stats[:,positions_to_keep_bool,:]
                self.coverage = self.coverage[:,positions_to_keep_bool]
                self.fwd_cov = self.fwd_cov[:,positions_to_keep_bool]
                self.rev_cov = self.rev_cov[:,positions_to_keep_bool]
                self.major_nt = self.major_nt[:,positions_to_keep_bool]
                self.minor_nt = self.minor_nt[:,positions_to_keep_bool]
                self.major_nt_freq = self.major_nt_freq[:,positions_to_keep_bool]
                self.minor_nt_freq = self.minor_nt_freq[:,positions_to_keep_bool]
                # print results
                print( "Number of positions in candidate mutation table reduced from " + str(num_pos_old) + " to " + str(self.num_pos) + "." )
            else:
                raise Exception("Argument positions_to_keep_bool must be a numpy array of booleans with size num_pos.")
        except AttributeError:
            raise Exception("Argument positions_to_keep_bool must be a numpy array.")
               
            
    def copy(self):
        ''' Makes a copy of candidate mutation table object. '''
        return cmt_data_object( self.sample_names, self.in_outgroup, self.p, self.counts, self.quals, self.indel_stats, self.dataset_name )



######################################
# COVERAGE MATRIX: CLASS DEFINITION  #
######################################


# Import file

def read_cov_mat_npz( raw_cov_mat_file ):
    '''Loads raw coverage matrix from file.'''
    
    # Reads from file
    with open(raw_cov_mat_file, 'rb') as f:
        raw_cov_mat_npz = np.load(f,allow_pickle=True)
        raw_cov_mat = raw_cov_mat_npz['all_coverage_per_bp']
    return raw_cov_mat

def read_cov_mat_gzip( raw_cov_mat_file ):
    '''Loads raw coverage matrix from old version of file.'''
    
    # Reads from file
    with gzip.open(raw_cov_mat_file, 'rb') as f:
        raw_cov_mat = pickle.load(f)
        
    return raw_cov_mat


# Coverage matrix class definitions

class cov_data_object_simple:
    '''
    Generates a simple coverage data object. This class tracks basic summary
    information about the raw coverage over each contig. 
    
    This class (cov_data_object_simple) is the superclass of cov_data_object.
    Use cov_data_object instead for any analyses requiring the full raw 
    coverage matrix. 
        
    ARGUMENTS
    ---------
    
        raw_cov_mat: raw coverage matrix
        
        sample_names: array of sample names
        
        genome_length: genome length of reference genome
        
        contig_starts: contig boundaries of reference genome
        
        contig_names = contig names of reference genome
        
    ATTRIBUTES
    ----------
    
        sample_names: array of sample names
        
        num_samples: number of samples
        
        contig_names: array of contig names
        
        num_contigs: number of contigs on reference genome
        
        median_cov_by_contig: median raw coverage for each sample across each 
        contig on the reference genome
        
        copy_number_by_contig: normalized raw coverage for each sample across
        each contig on the reference genome; normalized to the raw coverage
        across the longest contig (which we assume to be the longest 
        chromosomal contig)
        
    METHODS
    -------
    
        init: generates coverage matrix data object; stores summary data over 
        contigs (not whole coverage matrix) in order to conserve memory
        
        filter_samples: filters coverage data along samples axis; downsizes all 
        attributes along sample axis based on boolean argument

        get_median_cov_of_chromosome: returns median coverage over longest 
        contig; assumes the longest contig must be chromosomal
                
        plot_heatmap_contig_copy_num: creates a heatmap of copy number across 
        samples and across contigs

    NOTES
    -----
        
        1. See also class definition for cov_data_object which is a child class
        of cov_data_object_simple.
        
        2. This uses raw coverage matrix data. A normalized coverage matrix is 
        not needed for this data object. 
    
    @author: Arolyn Conwill

    '''       

    
    def __init__(self, raw_cov_mat, sample_names, genome_length, contig_starts, contig_names ): 
        ''' 
        Generates coverage matrix object.
        '''
        
        # Save basic info
        self.sample_names = sample_names
        self.num_samples = len( self.sample_names )
        print( "Number of samples in raw coverage matrix: " + str(self.num_samples) + "." )
        self.contig_names = contig_names
        self.num_contigs = len( self.contig_names )
        print( "Number of contigs in genome: " + str(self.num_contigs) + "." )
        self.genome_length = genome_length
        self.contig_starts = contig_starts
        self.contig_names = contig_names
        # Compute contig lengths
        if self.num_contigs > 1:
            contig_lengths = (self.contig_starts)[1:]-(self.contig_starts)[0:-1]+1
            contig_lengths =np.append( contig_lengths, self.genome_length-self.contig_starts[-1]+1 )
        else:
            contig_lengths = np.array( self.genome_length  )
        self.contig_lengths = contig_lengths 
        
        # Confirm dimensions of raw coverage matrix are correct
        if raw_cov_mat.shape != ( self.num_samples,self.genome_length):
            raise Exception("Raw coverage array dimensions are " + str(raw_cov_mat.shape) + ", but should be (" + str(self.num_samples) + ", " + str(self.genome_length) + ")." )
        
        # Compute median coverage per contig per sample
        self.median_coverage_by_contig = np.zeros( ( self.num_samples, self.num_contigs ) )
        for idx,c_name in enumerate(contig_names):
            c_start = contig_starts[idx]-1
            if idx<self.num_contigs-1:
                c_end = contig_starts[idx+1]
            else:
                c_end = genome_length
            np.median( raw_cov_mat[:,c_start:c_end], axis=1, out=self.median_coverage_by_contig[:,idx] )
            
        # Compute copy number per contig per sample
        self.copy_number_by_contig = np.zeros( ( self.num_samples, self.num_contigs ) ) # init
        idx_chromosomal_contig = np.argmax( self.contig_lengths ) # find longest contig
        np.divide( self.median_coverage_by_contig, np.expand_dims( self.median_coverage_by_contig[:,idx_chromosomal_contig],1 ), \
                  out=self.copy_number_by_contig ) # normalize by sample to coverage over longest contig (which we assume is chromosomal)
            
    
    
    def filter_samples(self,samples_to_keep_bool):
        ''' Filters samples and updates all coverage data objects attributes accordingly. '''
        try:
            if ( samples_to_keep_bool.dtype.type == np.bool_ ) and ( samples_to_keep_bool.size == self.num_samples ):
                # downsize attributes along samples dimension according to samples_to_keep_bool
                num_samples_old = self.num_samples # record original number of samples
                # downsize attributes along samples dimension according to samples_to_keep_bool
                self.sample_names = self.sample_names[samples_to_keep_bool]
                self.num_samples = len( self.sample_names )
                self.median_coverage_by_contig = self.median_coverage_by_contig[samples_to_keep_bool,:]
                # print results
                print( "Number of samples in coverage object reduced from " + str(num_samples_old) + " to " + str(self.num_samples) + "." )
            else:
                raise Exception("Argument samples_to_keep_bool must be a numpy array of booleans with size num_samples.")
        except AttributeError:
            raise Exception("Argument samples_to_keep_bool must be a numpy array.")


    def get_median_cov_of_chromosome(self):
        ''' Grab median coverage of longest contig (which we assume is the longest chromosomal contig). '''
        idx_chromosomal_contig = np.argmax( self.contig_lengths ) # find longest  contig
        return self.median_coverage_by_contig[:,idx_chromosomal_contig]
    
    
    def plot_heatmap_contig_copy_num( self, dir_save_fig, copynum_too_low_cutoff=0.5 ):
        ''' Creates a heatmap of copy number across samples and across contigs'''
        
        # Possible improvements:
        #TODO: Incorporate contig length into heatmap column width??
        #TODO: Set max number of contigs (for cases where there are tons of contigs)??
        
        # Make masked array where we are masking copy numbers that are too low
        masked_array = np.ma.masked_where( self.copy_number_by_contig<copynum_too_low_cutoff, self.copy_number_by_contig )
        
        # Heatmap appearance
        copynum_high_cutoff = 12
        
        # Plot heatmap of copy number by sample for each contig
        plt.clf() # reset plot axes
        fig, ax = plt.subplots()
        im = ax.imshow( masked_array, vmin=copynum_too_low_cutoff, vmax=copynum_high_cutoff ) 
        
        # Labeling
        ax.set_title("Copy number by sample by contig")
        plt.xlabel('contig')
        plt.ylabel('sample')
        ax.set_xticks(np.arange(len(self.contig_names)), labels=np.arange(1,self.num_contigs+1,1))
        ax.set_yticks(np.arange(len(self.sample_names)), labels=self.sample_names)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center",rotation_mode="anchor")
        ax.tick_params(axis='both', which='major', labelsize=6)
               
        # Colorbar
        ticks_for_colorbar = [ copynum_too_low_cutoff, 1, 3, 6, 9, copynum_high_cutoff ]
        tick_labels_for_colorbar = [ '$\leq$' + str(copynum_too_low_cutoff), '1', '3', '6', '9', '$\geq$' + str(copynum_high_cutoff) ]
        cbar = plt.colorbar(im, ticks=ticks_for_colorbar)
        cbar.ax.set_yticklabels(tick_labels_for_colorbar)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('copy number', rotation=90)
        
        fig.tight_layout()
        plt.show()

        # Save figure
        plt.savefig( dir_save_fig + "/contig_fig_copynum_heatmap.png" )
        

class cov_data_object( cov_data_object_simple ):
    '''
    Generates a coverage data object. This class tracks raw coverage over all 
    positions on the genome.

    This class (cov_data_object) is a subclass of cov_data_object_simple, 
    and thus inherits cov_data_object_simple's attributes and methods. 
    
    ARGUMENTS
    ---------
    
        raw_cov_mat: raw coverage matrix
        
        sample_names: array of sample names
        
        genome_length: genome length of reference genome
        
        contig_starts: contig boundaries of reference genome
        
        contig_names = contig names of reference genome
        
    ATTRIBUTES
    ----------
    
        sample_names: array of sample names
        
        num_samples: number of samples
        
        contig_names: array of contig names
        
        num_contigs: number of contigs on reference genome
        
        median_cov_by_contig: median raw coverage for each sample across each 
        contig on the reference genome
        
        copy_number_by_contig: normalized raw coverage for each sample across
        each contig on the reference genome; normalized to the raw coverage
        across the longest contig (which we assume to be the longest 
        chromosomal contig)
        
        raw_coverage: raw coverage for each sample across all positions on the
        reference genome
        
    METHODS
    -------
    
        init: generates coverage matrix data object; stores whole raw coverage
        matrix
        
        filter_samples: filters coverage data along samples axis; downsizes all 
        attributes along sample axis based on boolean argument

        get_median_cov_of_chromosome: returns median coverage over longest 
        contig; assumes the longest contig must be chromosomal
        
        make_cov_trace: makes a plot of raw coverage traces for all samples 
        over a given contig
                
    NOTES
    -----
        
        1. This uses raw coverage matrix data. A normalized coverage matrix is 
        not needed for this data object. 
    
    @author: Arolyn Conwill

    '''       

    def __init__( self, raw_cov_mat, sample_names, genome_length, contig_starts, contig_names ): 
        ''' 
        Generates full coverage matrix object.
        '''
        
        # Inherits attributes of class cov_data_object_simple
        cov_data_object_simple.__init__( self, raw_cov_mat, sample_names, genome_length, contig_starts, contig_names)
        
        # Also saves complete raw coverage matrix as attribute
        self.raw_cov_mat = raw_cov_mat


    def filter_samples( self, samples_to_keep_bool ):
       ''' Filters samples and updates all coverage data objects attributes accordingly. '''
       # Same method as superclass, but also downsizes attribute raw_cov_mat
       super().filter_samples( samples_to_keep_bool )
       self.raw_cov_mat = self.raw_cov_mat[samples_to_keep_bool,:]
 
           
           
    def make_coverage_trace( self, contig_number, num_cov_bins, dir_save_fig ):
        ''' Makes a plot of copy number traces for all samples over a given contig. '''
        
        # Coverage bins
        contig_idx = contig_number-1
        contig_start = self.contig_starts[contig_idx]
        if contig_number<self.num_contigs:
            contig_end = self.contig_starts[contig_idx+1]
        else:
            contig_end = self.genome_length
        cov_bins = np.linspace( contig_start, contig_end, num=num_cov_bins+1 ).astype(int)
        cov_bin_centers = cov_bins[:-1] + np.diff(cov_bins)/2
        
        # Compute mean coverage across contig for each sample, binned into num_cov_bins bins
        cov_trace_data = np.zeros( ( self.num_samples, num_cov_bins ) )
        for idx in np.arange(num_cov_bins):
            bin_start_idx = cov_bins[idx]-1
            bin_end_idx = cov_bins[idx+1]-1
            cov_trace_data[:,idx] = np.mean( self.raw_cov_mat[:,bin_start_idx:bin_end_idx], axis=1 ) / self.get_median_cov_of_chromosome()
        
        # Make a plot
        plt.clf() # reset plot axes
        fig, ax = plt.subplots()
        # Add traces
        for idx in np.arange(self.num_samples):
            ax.plot( cov_bin_centers, cov_trace_data[idx,:], 'k', alpha=0.1 )
        plt.xlim( xmin=np.min(cov_bins), xmax=np.max(cov_bins) )
        plt.ylim( ymin=0, ymax=np.maximum(1.,1.1*np.max(cov_trace_data)) )
        # Add horizontal line at copy number 1
        plt.axhline(y=1, color='b', linestyle='-')
        # Add labels
        plt.xlabel('position on genome')
        plt.ylabel('copy number')
        plt.title('contig ' + str(contig_number))
        # Display plot
        plt.show()
        
        # Save figure
        plt.savefig( dir_save_fig + "/contig_fig_trace_contig-" + str(contig_number) + ".png" )

        return cov_trace_data
        


###########################
# CALLS: CLASS DEFINITION #
###########################


# Calls object class definition 

class calls_object:
    '''
    This object holds basecalls which are generated from major_nt of a 
    cmt_data_object and can subsequently be filtered using the methods below.
    
    This object is separate from cmt_data_object as it is common to re-generate
    and filter calls for different objectives (stricter filters for identifying
    SNV positions vs looser filters for basecalls for a parsimony tree).
    
    Initialized attributes are taken directly from the candidate mutation 
    table object and can then be filtered using the class methods.

        
    ARGUMENTS
    ---------
    
        candidate_mutation_table: cmt_data_object
        
    ATTRIBUTES
    ----------
            
        calls: basecalls for each sample across candidate SNV positions
        
        sample_names: array of sample names
        
        num_samples: number of samples
        
        p: candidate SNV positions on the reference genome
        
        num_pos: number of candidate SNV positions
        
        in_outgroup: array of booleans indicating if each sample is an outgroup
        sample
            
    METHODS
    -------

        copy: return a copy of calls object
        
        filter_samples: remove bad samples by downsizing array attributes
        
        filter_positions: remove bad positions by downsizing array attributes
        
        get_frac_Ns_by_position: compute fraction of samples called as Ns at 
        each position
        
        get_frac_Ns_by_sample: compute fraction of positions called as Ns in 
        each sample
        
        filter_calls_by_element: filter individual calls based on boolean input
    
        filter_calls_by_sample: filter calls in bad samples
        
        filter_calls_by_position: filter calls in bad positions
        
        get_NTs: return array of NTs (as strings) from calls

        get_calls_in_outgroup_only: return array of calls in outgroup samples 
        only
        
        get_NTs_in_outgroup_only: return array of NTs in outgroup samples only
        
        get_calls_in_sample_subset: return array of calls in outgroup samples 
        only
        
        get_calls_in_position_subset: return array of calls in outgroup samples 
        only
        
        get_calls_in_sample_and_position_subset: return array of calls in 
        outgroup samples only.
        
    NOTES
    -----
    
        ...

    @author: Arolyn Conwill
    
    '''       
    
    def __init__( self, candidate_mutation_table ):
        ''' Initialize calls from major_nt attribute of candidate_mutation_table. '''
        if type(candidate_mutation_table) == cmt_data_object:
            self.calls = candidate_mutation_table.major_nt
            self.sample_names = candidate_mutation_table.sample_names
            self.num_samples = candidate_mutation_table.num_samples
            self.p = candidate_mutation_table.p
            self.num_pos = candidate_mutation_table.num_pos
            self.in_outgroup = candidate_mutation_table.in_outgroup
            print( "Number of samples in calls object: " + str(self.num_samples) + "." )
            print( "Number of positions in calls object: " + str(self.num_pos) + "." )
        else:
            raise Exception("Argument candidate_mutation_table must belong to class cmt_data_object.")

    def copy( self ):
        ''' Makes a copy of calls object. '''
        return cp.deepcopy(self) 
    
    # For downsizing array attributes
    
    def filter_samples(self,samples_to_keep_bool):
        ''' Filters samples and updates all calls object attributes accordingly. '''
        try:
            if ( samples_to_keep_bool.dtype.type == np.bool_ ) and ( samples_to_keep_bool.size == self.num_samples ):
                # downsize attributes along samples dimension according to samples_to_keep_bool
                num_samples_old = self.num_samples # record original number of samples
                # downsize attributes along samples dimension according to samples_to_keep_bool
                self.sample_names = self.sample_names[samples_to_keep_bool]
                self.num_samples = len( self.sample_names )
                self.in_outgroup = self.in_outgroup[samples_to_keep_bool]
                self.calls = self.calls[samples_to_keep_bool,:]
                # print results
                print( "Number of samples in calls object reduced from " + str(num_samples_old) + " to " + str(self.num_samples) + "." )
            else:
                raise Exception("Argument samples_to_keep_bool must be a numpy array of booleans with size num_samples.")
        except AttributeError:
            raise Exception("Argument samples_to_keep_bool must be a numpy array.")
            
    def filter_positions(self,positions_to_keep_bool):
        ''' Filters positions and updates all calls object attributes accordingly. '''
        try:
            if ( positions_to_keep_bool.dtype.type == np.bool_ ) & ( positions_to_keep_bool.size == self.num_pos ):
                # downsize attributes along samples dimension according to positions_to_keep_bool
                num_pos_old = self.num_pos # record original number of positions
                self.p = self.p[positions_to_keep_bool]
                self.num_pos = np.count_nonzero(positions_to_keep_bool)
                self.calls = self.calls[:,positions_to_keep_bool]
                # print results
                print( "Number of positions in calls object reduced from " + str(num_pos_old) + " to " + str(self.num_pos) + "." )
            else:
                raise Exception("Argument positions_to_keep_bool must be a numpy array of booleans with size num_pos.")
        except AttributeError:
            raise Exception("Argument positions_to_keep_bool must be a numpy array.")
                       
    # For querying number of ambiguous basecalls
    
    def get_frac_Ns_by_position( self ):
        ''' Compute fraction of samples called as Ns at each position. '''
        return 1 - np.count_nonzero( self.calls, axis=0 )/self.num_samples

    def get_frac_Ns_by_sample( self, pos_to_consider=[] ):
        ''' 
        Compute fraction of positions called as Ns in each sample. 
        Optional input to only mask certain candidate SNV positions 
        (default is to consider all positions).
        '''
        pos_to_consider_bool = np.isin( self.p, pos_to_consider )
        num_pos_to_consider = np.count_nonzero(pos_to_consider_bool)
        return 1 - np.count_nonzero( self.calls[:,pos_to_consider_bool], axis=1 )/num_pos_to_consider

    # For filtering basecalls (set bad ones to N)
    
    def filter_calls_by_element( self, calls_to_filter ):
        ''' Filter calls based on boolean input. '''
        self.calls[calls_to_filter] = NTs_to_int_dict['N'] # set to N
    
    def filter_calls_by_sample( self, samples_to_filter ):
        ''' Filter calls in bad samples. '''
        self.calls[samples_to_filter,:] = NTs_to_int_dict['N'] # set all calls in bad samples to N by broadcasting boolean argument

    def filter_calls_by_position( self, positions_to_filter ):
        ''' Filter calls in bad positions. '''
        self.calls[:,positions_to_filter] = NTs_to_int_dict['N'] # set all calls in bad positions to N by broadcasting boolean argument

    # For querying basecalls
    
    def get_NTs( self ):
        ''' Return array of NTs from calls. '''
        return ints2nts( self.calls )
    
    def get_calls_in_outgroup_only( self ):
        ''' Return array of calls in outgroup samples only. '''
        return self.calls[self.in_outgroup,:]
    
    def get_NTs_in_outgroup_only( self ):
        ''' Return array of NTs in outgroup samples only. '''
        return nts2ints( self.get_calls_in_outgroup_samples() )

    def get_calls_in_sample_subset( self, sample_bool ):
        ''' Return array of calls in outgroup samples only. '''
        return self.calls[sample_bool,:]

    def get_calls_in_position_subset( self, position_bool ):
        ''' Return array of calls in outgroup samples only. '''
        return self.calls[:,position_bool]

    def get_calls_in_sample_and_position_subset( self, sample_bool, position_bool ):
        ''' Return array of calls in outgroup samples only. '''
        return self.calls[sample_bool,position_bool]



######################################
# REFERENCE GENOME: CLASS DEFINITION #
######################################


# Reference genome class definition 

class reference_genome_object:
    '''
    This object holds information about a referenge genome. Its sole input is a 
    reference genome directory which must contain a fasta file and a genbank 
    file.
        
    ARGUMENTS
    ---------

        dir_ref_genome: path to directory with reference genome; must contain 
        file called "genome.fasta" as well as a GFF file "*.gff"

    ATTRIBUTES
    ----------

        dir_reference_genome: remember source of reference genome

        contig_starts: positions on reference genome contig boundaries 
        (specifically where each contig starts)
        
        contig_names: names of contigs (array of strings)
        
        genome_length: length fo genome
        
        annotations: pandas dataframe with genome annotations, read from GFF
        
        locus_tags: tags each position on the genome which gene is present 
        there; ingergenic = 0.5
        
        contig_tags: tags each position on the genome with the contig number
        
        
    METHODS
    -------
    
        p2contigpos: converts position on genome (1...genome_length) to two-
        element positions (contig_num pos_on_contig)
        
        contigpos2p: converts two-element positions (contig_num pos_on_contig) 
        to position on genome (1...genome_length)
        
        get_ref_NTs: gets nucleotides of reference genome (as strings) at 
        positions provided as method argument
        
        get_ref_NTs_as_ints: gets nucleotides of reference genome (as ints) at 
        positions provided as method argument
    
        
    @author: Arolyn Conwill

    '''       
    
    def __init__( self, dir_ref_genome ):
        ''' Initializes reference genome object from fasta file and gff file. '''
        
        # Remember reference genome directory
        self.dir_ref_genome = dir_ref_genome
        
        # Read in FASTA file
        [ self.contig_starts, self.contig_names, self.genome_length, self.genome_seq ] = get_genome_stats_from_fasta( dir_ref_genome )
        
        # Read in gff
        if len(glob.glob(dir_ref_genome + '/*.gff*'))!=1:
            self.annotations = [] # allows creation of reference genome object even if annotations do not exist
            print("WARNING! Reference genome object initialized with empty annotations because either no gff file was found or multple gff files were found!")
        else:
            self.annotations = parse_gff( dir_ref_genome, self.contig_names )
        
        # Tag all positions with information about coding sequences
        [ self.locus_tagnumbers, self.cds_indices ] = tag_all_genomic_positions( self.annotations, self.genome_length, self.contig_starts ) 


    def p2contigpos( self, p ):
        ''' Converts positions on genome to positions on contig. '''
        idx_of_contig = np.ones(len(p),dtype=int) # init assuming all positions are on first contig
        if len(self.contig_starts) > 1: # if multiple contigs exist
            for next_contig_start in self.contig_starts[1:]:
                idx_of_contig = idx_of_contig + (p >= next_contig_start) # note: (p > i) adds 1 if true and adds 0 if false
            idx_on_contig = p - self.contig_starts[idx_of_contig-1] + 1
            contigpos = np.column_stack((idx_of_contig,idx_on_contig))
        else:
            contigpos = np.column_stack((idx_of_contig,p))
        return contigpos

    def contigpos2p( self, contigpos ):
        ''' Converts positions on contig to positions on genome. '''
        idx_of_contig = contigpos[:,0]
        idx_on_contig = contigpos[:,1]
        p = self.contig_starts[idx_of_contig-1] + idx_on_contig -1
        return p
    
    def get_ref_NTs( self, p ):
        ''' Gets nucleotide identity (character) for requested positions on reference genome. '''
        # Check that positions were provided as p not contigpos
        if p.ndim==2:
            raise Exception("Error! Argument p should not be two-dimensional. Convert contigpos to p using method contigpos2p.")
        # Get reference nucleotides as characters
        refnt = self.genome_seq[p-1]
        return refnt
        
    def get_ref_NTs_as_ints( self, p ):
        ''' Gets nucleotide identity (integer) for requested positions on reference genome. '''
        # Check that positions were provided as p not contigpos
        if p.ndim==2:
            raise Exception("Error! Argument p should not be two-dimensional. Convert contigpos to p using method contigpos2p.")
        # Get reference nucleotides as integers
        return nts2ints( self.get_ref_NTs( p ) )



# Reference genome funcitons
# # Functions that support initialization of class reference_genome_object or
# # that can be used independently. Some of these are messy and should be 
# # re-written to be more readable (parse_gff).

def get_genome_stats_from_fasta( dir_ref_genome ):
    '''
    This function get basic genome stats from a fasta file. 
    
    Note: Positions on the genome are indexed starting with 1 (to match 
    positions in vcfs).
    '''
    
    # Read fasta file
    ref_genome = SeqIO.parse( dir_ref_genome + '/genome.fasta', 'fasta' )
    genome_length = 0 # init
    contig_starts = [] # init
    contig_names = [] # init
    genome_seq = '' # init

    for record in ref_genome: # loop through contigs
        contig_starts.append(genome_length+1)
        contig_names.append(record.id)
        genome_length = genome_length + len(record)
        genome_seq = genome_seq + str(record.seq).upper()

    # Turn into numpy arrays
    contig_starts = np.asarray( contig_starts, dtype=np.int_ )
    genome_length = np.asarray( genome_length, dtype=np.int_ )
    contig_names = np.asarray( contig_names, dtype=object )
    genome_seq = np.array( list(genome_seq) )
    
    return [ contig_starts, contig_names, genome_length, genome_seq ]


def parse_gff( dir_ref_genome, contig_names, ortholog_info_series=pd.Series(dtype='float64') ):
    '''
    This function reads genome annotations from a gff file.
    
    NOTES
    -----
    
        1. Fails if more than one gff file exists.
        
        2. No data is always reported as '.'.
        
        3. If column contains multiple entries, they are separated by ';'.
        
        4. More info on gff parsing: https://biopython.org/wiki/GFF_Parsing
        
      
    @author: Felix Key
    '''

    # Possible improvements:
            
        # 1. Annotations function is picky about GFF format. In the future it 
        # us worth changing this function so it works with a broader set of GFF 
        # sources. #TODO
        
        # 2. Only read gff if dataframe does not already exist. #TODO


    # # Print warning regarding "phase" field in GFF
    # print("""
    #       ! ! ! Warning (from Arolyn) ! ! !
    #       This GFF parser function assumes that the "phase" of a coding 
    #       sequence (CDS) is '0', i.e. that there are no extra bases before the
    #       start codon that need to be truncated before translation. This is 
    #       consistent with prokka annotations which always report a phase of 0
    #       in the GFFs. This is also necessary for RAST annotations which report
    #       the phase relative to the contig, not the CDS. However, I do not know
    #       why an older version of this function used the "phase" field from the
    #       GFF. It is possible that this field is necessary to correctly
    #       translate some amino acid sequences. If this is the case with your
    #       GFF, you can uncomment the section that uses the phase as reported
    #       in the GFF. A good reality check would be to look at the dataframe in
    #       the 'annotations' attribute of your reference genome and see if the 
    #       start and stop codons are in reasonable places. 
    #       """)
    
    # Find gff file:
    gff_file = glob.glob(dir_ref_genome + '/*.gff*')
    
    if len(gff_file) != 1:
        raise ValueError('Either no file or more than 1 *gff file found in ' + dir_ref_genome)
    print( gff_file  )
    
    # Check gff file available fields:
    examiner = GFF.GFFExaminer()
    with open(gff_file[0]) as gff_handle:
        possible_limits = examiner.available_limits(gff_handle) # available_limits function gives a summary of feature attributes along with counts for the number of times they appear in the file
    # Make a list of all attributes in gff_type except gene and region 
    limits = dict(gff_type = [i[0] for i in possible_limits['gff_type'].keys() if i[0] != 'gene' and i[0] != 'region'] )
    
    # Read gff file: 
    
    list_of_dataframes = [] # init # each element is a pandas dataframe with all annotations for the contig; contigs are ordered according to contig_names
    tagnumber_counter = 0 # init # unique numerical identifier for all features across all contigs
   
    for contig in contig_names: # loop over contig annotations according to order in contig_names
        with open(gff_file[0]) as gff_handle:
            for rec in GFF.parse(gff_handle, limit_info=limits): # loop over every contig, but only grab attributes specified by [limits] to save memory
                if rec.id == contig:
                    # if contig has any feature build list of dicts and append to list_of_dataframes, else append empty dataframe
                    if len(rec.features) > 0:
                        # test if seq object part of gff (prokka-based yes, but NCBI-based no >> then load ref genome.fasta)
                        if len(rec.seq) == rec.seq.count('?'):
                            for seq_record in SeqIO.parse(dir_ref_genome+"/genome.fasta", "fasta"):
                                if seq_record.id == rec.id:
                                    rec.seq = seq_record.seq
                            if len(rec.seq) == rec.seq.count('?'): # test if succesful
                                print('Warning: No reference genome found that matches chromosome:' + rec.id)
                        print(rec.id)
                        lod_genes = [] # list-of-dictionary; easy to convert to pandas dataframe
                        for gene_feature in rec.features:
                            
                            gene_dict = {}
                            tagnumber_counter += 1
                            
                            gene_dict['type'] = gene_feature.type
                            gene_dict['locustag'] = gene_feature.id

                            # add ortholog info if locustag (eg. repeat region has none)
                            if gene_feature.id != "" and gene_feature.type == 'CDS' and not ortholog_info_series.empty:
                                gene_dict['orthologtag'] = ortholog_info_series[ortholog_info_series.str.findall(gene_feature.id).str.len() == 1].index[0]

                            if 'gene' in gene_feature.qualifiers.keys():
                                gene_dict['gene'] = ";".join(gene_feature.qualifiers['gene'])
                            else:
                                gene_dict['gene'] = "." # add "." instead of []

                            if gene_dict['type'] == "CDS" or gene_dict['type'] == "gene":
                                gene_dict['tagnumber'] = tagnumber_counter
                            else:
                                gene_dict['tagnumber'] = 0
                            
                            if 'product' in gene_feature.qualifiers.keys():
                                gene_dict['product'] = ";".join(gene_feature.qualifiers['product'])
                            elif 'Name' in gene_feature.qualifiers.keys(): # Arolyn, 2022.10: RAST output has protein in "Names" field
                                gene_dict['product'] = ";".join(gene_feature.qualifiers['Name'])
                            else:
                                gene_dict['product'] = "."

                            if 'protein_id' in gene_feature.qualifiers.keys():
                                gene_dict['protein_id'] = gene_feature.qualifiers['protein_id']
                            else:
                                gene_dict['protein_id'] = "."

                            if "Dbxref" in gene_feature.qualifiers.keys(): # for prokka annotations
                                gene_dict['db_xref'] = ";".join(gene_feature.qualifiers['Dbxref'])
                            elif "ID" in gene_feature.qualifiers.keys(): # for RAST annotations
                                gene_dict['db_xref'] = ";".join(gene_feature.qualifiers['ID'])
                            else:
                                gene_dict['db_xref'] = "."

                            if 'Ontology_term' in gene_feature.qualifiers.keys(): 
                                gene_dict['ontology_term'] = gene_feature.qualifiers['Ontology_term']
                            else:
                                gene_dict['ontology_term'] = '.'

                            if "note" in gene_feature.qualifiers.keys():
                                gene_dict['note'] = ";".join(gene_feature.qualifiers['note'])
                            elif "Note" in gene_feature.qualifiers.keys():
                                gene_dict['note'] = ";".join(gene_feature.qualifiers['Note'])
                            else:
                                gene_dict['note'] = "."

                            gene_dict['indices'] = [gene_feature.location.start.position+1,gene_feature.location.end.position] # Arolyn, 2022.10: +1 for indexing on first pos only
                            gene_dict['loc1'] = gene_feature.location.start.position+1 # Arolyn, 2022.10: +1 for indexing since default is 0-based
                            gene_dict['loc2'] = gene_feature.location.end.position # Arolyn, 2022.10: final pos is already correct

                            gene_dict['strand'] = gene_feature.location.strand 
                            dna_seq = rec.seq[gene_feature.location.start:gene_feature.location.end]
                            if gene_dict['strand'] == 1:
                                gene_dict['sequence'] = dna_seq
                            elif gene_dict['strand'] == -1:
                                gene_dict['sequence'] = dna_seq.reverse_complement()
                            else:
                                gene_dict['sequence'] = dna_seq # eg. repeat region

                            # # Use this section if you need to use the 'phase' field of the GFF in order to translate proteins correctly
                            # if 'phase' in gene_feature.qualifiers.keys():
                            #     gene_dict['codon_start'] = int(gene_feature.qualifiers['phase'][0])
                            # else:
                            #     gene_dict['codon_start'] = "."
                            # if isinstance( gene_dict['codon_start'] , int):
                            #     sequence2translate = gene_dict['sequence'][gene_dict['codon_start']:]
                            #     gene_dict['translation'] = sequence2translate.translate(table="Bacterial") # bacterial genetic code GTG is a valid start codon, and while it does normally encode Valine, if used as a start codon it should be translated as methionine. http://biopython.org/DIST/docs/tutorial/Tutorial.html#sec:translation
                            # elif gene_dict['type'] == "CDS":
                            #     sequence2translate = gene_dict['sequence']
                            #     gene_dict['translation'] = sequence2translate.translate(table="Bacterial")
                            # else:
                            #     gene_dict['translation'] = "." # all non-CDS (RNA's or repeat regions) not translated (as those are sometimes also off-frame)
                            # Use this section if you want to ignore the 'phase' field of the GFF in order to translate proteins correctly
                            if gene_dict['type'] == "CDS":
                                sequence2translate = gene_dict['sequence']
                                gene_dict['translation'] = sequence2translate.translate(table="Bacterial")
                            else:
                                gene_dict['translation'] = "." # all non-CDS (RNA's or repeat regions) not translated (as those are sometimes also off-frame)

                            lod_genes.append(gene_dict)

                        # make pandas dataframe
                        df_sort = pd.DataFrame(lod_genes)
                        df_sort = df_sort.sort_values(by=['loc1']) # sort pandas dataframe (annotation not necessarily sorted)
                        list_of_dataframes.append(df_sort)
                    else:
                        list_of_dataframes.append( pd.DataFrame([{}]) )
    # Save annotations file in dir_ref_genome
    afile = open(dir_ref_genome+"/annotation_genes.pandas.py.pk1", 'wb')
    pickle.dump(list_of_dataframes, afile)
    afile.close()
    return list_of_dataframes



def tag_all_genomic_positions( anno_genes_ls, genome_length, contig_starts ):
    ''' 
    Tag all genomic positions with:
        * locus_tagnumbers: unique identifier for each CDS in the genome; 0.5 
        indicates intergenic; tRNA is 0 (inherited from 'tagnumber' field of 
        annotations dataframe--see function parse_gff)
        * cds_indices: indexes each CDS uniquely on a given contig; intergenic 
        regions are 0.5+preivous_cds_idx; indexes tRNAs like genes
    
    WARNING! This does not handle cases where there are overlapping coding 
    regions well. In this case, the tag representing the earlier coding region 
    gets overwritten by tag(s) representing the later coding regions. #TODO
    
    CHANGES
    -------
    
        * Arolyn, 2022.10: added comments and made compatible with new indexing

    @author: Felix Key
    '''

    # Initialize
    locus_tagnumbers = np.ones(genome_length,dtype=float)*0.5 # CDS tag ('tagnumber' from annotations dataframe) that is unique across all contigs; intergenic = 0.5; tRNA = 0 (since gff parser function sets tagnumber to zero for non-CDS annotations)
    cds_indices = np.ones(genome_length,dtype=float)*0.5 # CDS tag that is unique on a given contig only; intragenic = previous_idx+0.5

    # Loop through annotation tables for each contig
    for i,this_contig_df in enumerate(anno_genes_ls): 
        
        if this_contig_df.empty: # skip contigs any coding sequence annotations
            continue

        # Get info from annotation dataframe
        gene_tagnumbers = this_contig_df[['tagnumber']].values.flatten() 
        gene_starts = this_contig_df[['loc1']].values.flatten() + contig_starts[i] - 1 # genome position indexing starts at 1
        gene_ends = this_contig_df[['loc2']].values.flatten() + contig_starts[i] - 1 # genome position indexing starts at 1
        
        # Mark positions across all genes except for the last one on the contig
        for j in range(len(gene_starts)-1):
            locus_tagnumbers[ (gene_starts[j]-1):gene_ends[j] ] = gene_tagnumbers[j] # populate locus_tagnumbers across this gene
            cds_indices[ (gene_starts[j]-1):gene_ends[j] ] = j+1; # populate cds_indices across this gene
            cds_indices[ (gene_ends[j]):(gene_starts[j+1]-1) ] = j+1+0.5 # populate cds_indices between this gene and the next gene
        
        # Mark positions for last gene on the contig
        locus_tagnumbers[ (gene_starts[-1]-1):gene_ends[-1] ] = gene_tagnumbers[-1] # populate locus_tagnumbers across the last gene
        cds_indices[ (gene_starts[-1]-1):gene_ends[-1] ] = len(gene_tagnumbers) # populate cds_indices across the last gene
        
        # Mark remaining positions on contig 
        if ((i+1) < len(contig_starts)):
            cds_indices[ gene_ends[-1]:contig_starts[i+1]-1 ] = len(gene_tagnumbers) + 0.5 # populate cds_indices after the last gene until the end of the contig
        else: # last contig
            cds_indices[ gene_ends[-1]:genome_length ] = len(gene_tagnumbers) + 0.5 # populate cds_indices after the last gene until the end of the contig (same as the end of the genome)

    return [ locus_tagnumbers, cds_indices ]



############################
# SNV FILTERING: FUNCTIONS #
############################


def filter_histogram( filter_value, filter_cutoff, filter_name, save_bool=False, dir_save_fig=os.getcwd(), fig_file_name='snv_filter_histogram.png' ):
    '''
    Make a generic histogram to evaluate filter cutoff.
        
    ARGUMENTS
    ---------
    
        filter_value: quantity that is being used for filtering (numerical 
        array)
        
        filter_cutoff: threshold for filtering (numerical value)
        
        filter_name: description of filter_value (string)
        
        save_bool: whether or not to save a plot
        
        dir_save_fig: directory in which to save the figure
        
        fig_file_name: file name of figure (string)
        
    @author: Arolyn Conwill

    '''
    
    # Make a histogram
    plt.clf() # reset plot axes
    my_bins = np.linspace( np.min(filter_value), np.max(filter_value), 50 )
    n, bins, patches = plt.hist(x=filter_value, bins=my_bins, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(filter_name)
    # Add a line at filter cutoff
    plt.axvline(x = filter_cutoff, color = 'r')
       
    if save_bool: # save plot
       plt.savefig( dir_save_fig + "/" + fig_file_name )
   

def filter_samples_by_coverage( median_cov_by_sample, min_average_coverage_to_include_sample, sample_names, plot_bool=False, dir_save_fig=os.getcwd() ):
    '''
    Filters samples based on median coverage. Option to make a histogram.
        
    ARGUMENTS
    ---------
    
        median_cov_by_sample: median coverage across genome by sample
        
        min_average_coverage_to_include_sample: raw coverage cutoff for 
        including a sample
        
        sample_names: array of sample names
        
        plot_bool: whether or not to generate a histogram of median coverage 
        by sample
        
        dir_save_fig: directory in which to save the figure
        
        
    RETURNS
    -------
    
        sampleNames_lowcov: names of low coverage samples
        
        bool_goodsamples: boolean of goodsamples that passed filters, indexed 
        according to input sample_names
        
    NOTES
    -----
    
        1. median_cov_by_sample is intended to be the median coverage across 
        the whole chromosome. It is acceptable to provide the median coverage
        across the longest contig in an assembly (assumed to be the longest
        chromosomal contig). It is NOT recommended to use the median coverage
        across candidate mutation positions, since candidate SNVs may not be 
        representative (e.g. may be enriched for mobile elements or regions 
        that are repeated throughout the genome).
    
    @author: Arolyn Conwill

    '''
    
    # Filter 
    bool_goodsamples = median_cov_by_sample>=min_average_coverage_to_include_sample
    sampleNames_lowcov = sample_names[~bool_goodsamples]
    
    # Make a plot
    if plot_bool:
        
        # Make a histogram
        plt.clf() # reset plot axes
        maxcov=median_cov_by_sample.max()
        maxcovbin=np.ceil(maxcov/10)*10+10
        my_bins = np.arange(0,int(maxcovbin),5)
        n, bins, patches = plt.hist(x=median_cov_by_sample, bins=my_bins, color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Median coverage')
        plt.ylabel('Number of samples')
        plt.title('Median coverage across samples')
        # Set a clean upper y-axis limit.
        plt.ylim( ymin=0, ymax=(np.ceil(n.max())/10)*10+2 )
        plt.xlim( xmin=0, xmax=maxcovbin )
        # Add a line at filter cutoff
        plt.axvline(x = min_average_coverage_to_include_sample, color = 'r')
        
        # Save plot
        plt.savefig( dir_save_fig + "/snv_filter_sample_coverage_hist.png" )
    
    return [ sampleNames_lowcov, bool_goodsamples ]


def filter_samples_by_ambiguous_basecalls( frac_ambig_basecalls_by_sample, max_frac_Ns_to_include_sample, sample_names, in_outgroup_bool, plot_bool=False, dir_save_fig=os.getcwd() ):
    '''
    Filters samples based on the number of ambiguous basecalls across
    candidate SNV positions. Cannot filter outgroup samples.
        
    ARGUMENTS
    ---------
    
        frac_ambig_basecalls_by_sample: fraction of ambiguous basecalls across
        candidate SNV positions
        
        max_frac_Ns_to_include_sample: maximum allowable fraction of positions 
        with ambiguous basecalls (Ns)
        
        sample_names: array of sample names
        
        in_outgroup: boolean array indicating which samples are outgroup samples
        
        plot_bool: whether or not to generate a histogram of fraction of 
        ambiguous basecalls
        
        dir_save_fig: directory in which to save the figure
        
        
    RETURNS
    -------
    
        sampleNames_toomanyNs: names of samples with too many ambiguous basecalls
        
        bool_goodsamples: boolean of goodsamples that passed filters, indexed 
        according to input sample_names
    
    @author: Arolyn Conwill

    '''
    
    # Filter 
    bool_goodsamples = ( frac_ambig_basecalls_by_sample <= max_frac_Ns_to_include_sample ) \
        | in_outgroup_bool # cannot filter outgroup samples
    sampleNames_toomanyNs = sample_names[~bool_goodsamples]
    
    # Make a plot
    if plot_bool:
        
        # Make a histogram
        plt.clf() # reset plot axes
        my_bins = np.linspace(0,1, num=21, endpoint=True)
        n, bins, patches = plt.hist(x=frac_ambig_basecalls_by_sample[~in_outgroup_bool], bins=my_bins, color='#0504aa', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Fraction ambiguous basecalls (Ns)')
        plt.ylabel('Number of samples')
        plt.title('Fraction ambiguous basecalls (Ns) across samples')
        # Set a clean upper y-axis limit.
        plt.ylim( ymin=0, ymax=(np.ceil(n.max())/10)*10+2 )
        plt.xlim( xmin=0, xmax=1 )
        # Add a line at filter cutoff
        plt.axvline(x = max_frac_Ns_to_include_sample, color = 'r')
        
        # Save plot
        plt.savefig( dir_save_fig + "/snv_filter_sample_toomanyNs_hist.png" )
    
    return [ sampleNames_toomanyNs, bool_goodsamples ]



def compute_mutation_quality( Calls, Quals ):
    '''
    This functions aims at providing a FQ value for every SNP position.
    
    Method: Across all pairwise different allele calls, it reports the best FQ 
    value among the minimum FQ values per pair.
        
    ARGUMENTS
    ---------
    
        Calls: filtered basecalls by sample by position
        
        Quals: FQ values by sample by position
        
    RETURNS
    -------
    
        MutQual: quality score (FQ) for each SNV position
        
        MutQualIsolates: indices of isolate pairs for each SNV position from 
        which MutQual was obtained

    NOTES
    -----
    
        1. This function is slow for many SNVs.
    
    '''
    
    Calls = Calls.transpose()
    Quals = Quals.transpose()
    
    [Nmuts, NStrain] = Calls.shape ;
    MutQual = np.zeros((Nmuts,1)) ;
    MutQualIsolates = np.zeros((Nmuts,2));
    
    idx_for_N = NTs_to_int_dict['N']

    # generate template index array to sort out strains gave rise to reported FQ values
    s_template=np.zeros( (len(Calls[0,:]),len(Calls[0,:])) ,dtype=object)
    for i in range(s_template.shape[0]):
        for j in range(s_template.shape[1]):
            s_template[i,j] = str(i)+"_"+str(j)

    for k in range(Nmuts):
        if len(np.unique(np.append(Calls[k,:], idx_for_N))) <= 2: # if there is only one type of non-N (4) call, skip this location
            MutQual[k] = np.nan ;
            MutQualIsolates[k,:] = 0;
        else:
            c = Calls[k,:] ; c1 = np.tile(c,(c.shape[0],1)); c2 = c1.transpose() # extract all alleles for pos k and build 2d matrix and a transposed version to make pairwise comparison
            q = Quals[k,:] ; q1 = np.tile(q,(q.shape[0],1)); q2 = q1.transpose() # -"-
            g = np.all((c1 != c2 , c1 != idx_for_N , c2 != idx_for_N) ,axis=0 )  # no data ==4; boolean matrix identifying find pairs of samples where calls disagree (and are not N) at this position
            #positive_pos = find(g); # numpy has no find; only numpy where, which does not flatten 2d array that way
            # get MutQual + logical index for where this occurred
            MutQual[k] = np.max(np.minimum(q1[g],q2[g])) # np.max(np.minimum(q1[g],q2[g])) gives lower qual for each disagreeing pair of calls, we then find the best of these; NOTE: np.max > max value in array; np.maximum max element when comparing two arryas
            MutQualIndex = np.argmax(np.minimum(q1[g],q2[g])) # return index of first encountered maximum!
            # get strain ID of reorted pair (sample number)
            s = s_template
            strainPairIdx = s[g][MutQualIndex]
            MutQualIsolates[k,:] = [strainPairIdx.split("_")[0], strainPairIdx.split("_")[1]]
            
    MutQual = MutQual.transpose()
    MutQualIsolates = MutQualIsolates.transpose()

    return [MutQual,MutQualIsolates]


def find_recombination_positions( my_calls, my_cmt, calls_ancestral, mut_qual, my_rg, distance_for_nonsnp, corr_threshold_recombination, save_plots_bool=False, dir_save_fig=os.getcwd() ):
    '''
    Finds mutations suspected to arise from recombination (not SNVs) by 
    detecting pairs of preliminary SNVs that have correlated mutant allele 
    frequencies.
    '''
    
    # Make array of ancestral nucleotides that has num_samples_ingroup rows
    num_samples_ingroup = sum( np.logical_not( my_calls.in_outgroup ) )
    calls_ancestral_tiled = np.tile( calls_ancestral, (num_samples_ingroup,1) )

    # Compute mutant allele frequency
    # Major alelle
    major_nt_ingroup = my_cmt.major_nt[ np.logical_not( my_cmt.in_outgroup ), : ]
    major_nt_freq_ingroup = my_cmt.major_nt_freq[ np.logical_not( my_cmt.in_outgroup ), : ]
    major_nt_freq_ingroup[np.isnan(major_nt_freq_ingroup)]=0 # set nan values to 0
    # Minor allele
    minor_nt_ingroup = my_cmt.minor_nt[ np.logical_not( my_cmt.in_outgroup ), : ]
    minor_nt_freq_ingroup = my_cmt.minor_nt_freq[ np.logical_not( my_cmt.in_outgroup ), : ]
    minor_nt_freq_ingroup[np.isnan(minor_nt_freq_ingroup)]=0 # set nan values to 0
    # Mutant allele frequency: sum major allele frequencies and minor allele frequencies when they don't match the ancestral allele
    major_nt_mut_freq = major_nt_freq_ingroup
    major_nt_mut_freq[ np.where( major_nt_ingroup == calls_ancestral_tiled) ] = 0
    minor_nt_mut_freq = minor_nt_freq_ingroup
    minor_nt_mut_freq[ np.where( minor_nt_ingroup == calls_ancestral_tiled) ] = 0
    mutant_allele_freq = major_nt_mut_freq + minor_nt_mut_freq
    
    # Find preliminary SNV positions to test for recombination
    calls_ingroup = my_calls.get_calls_in_sample_subset( np.logical_not( my_calls.in_outgroup ) )
    filter_SNVs_not_N = ( calls_ingroup != nts2ints('N') ) # mutations must have a basecall (not N)
    filter_SNVs_not_ancestral_allele = ( calls_ingroup != np.tile( calls_ancestral, (num_samples_ingroup,1) ) ) # mutations must differ from the ancestral allele
    filter_SNVs_quals_not_NaN = ( np.tile( mut_qual, (num_samples_ingroup,1) ) >= 1) # alleles must have strong support
    fixedmutation = filter_SNVs_not_N & filter_SNVs_not_ancestral_allele & filter_SNVs_quals_not_NaN # boolean    
    goodpos_bool = np.any( fixedmutation, axis=0 )
    goodpos_idx = np.where( goodpos_bool )[0]
    num_goodpos = len(goodpos_idx)
    p = my_calls.p # extract candidate mutation positions
    p_goodpos = p[goodpos_idx] # extract preliminary SNV positions
    
    # Downsize mutant allele frequency to goodpos only
    mutant_allele_freq_goodpos = mutant_allele_freq[ :,goodpos_idx ]

    # Find recombination regions 
    # #TODO: this is slow
    nonsnp = np.zeros(0,dtype='int') # init
    for i in range(num_goodpos):
        p_snv = p[goodpos_idx[i]]
        # Find nearby preliminary SNVs
        region = np.array(np.where( \
                                   ( p_goodpos > p_snv - distance_for_nonsnp ) \
                                   & ( p_goodpos < p_snv + distance_for_nonsnp ) \
                                   ) ).flatten()
        # Check if pairs are correlated
        if len(region)>1: 
            r = mutant_allele_freq_goodpos[:,region] # dimension = num samples in ingroup x num positions in region
            corrmatrix = np.corrcoef(r.transpose()) # dimension = num positions in region x num positions in region
            [a,b] = np.where( corrmatrix > corr_threshold_recombination )
            nonsnp = np.concatenate(( nonsnp, region[a[np.where(a!=b)]] ))

    # Get unique positions
    nonsnp=np.unique(nonsnp) # indexed in goodpos
    p_nonsnp = p_goodpos[ nonsnp ]
    p_keep = np.setdiff1d( p_goodpos, p_nonsnp )
    nonsnp_bool = np.isin( p, p_nonsnp )
    
    # Make a plot
    plt.clf() # reset plot axes
    # Add blue lines for good SNV positions
    line_blue = plt.axvline(x = -1e6, color = 'b', label = 'SNV' ) # for legend handle only; outside of xlim  
    for pos in p_keep:
        plt.axvline(x = pos, color = 'b')    
    # Add red lines for recombination positions
    line_red = plt.axvline(x = -1e6, color = 'r', label = 'recombo' ) # for legend handle only; outside of xlim  
    for pos in p_nonsnp:
        plt.axvline(x = pos, color = 'r')    
    # Labels
    plt.title('recombination position filtering')
    plt.xlim( xmin=1, xmax=my_rg.genome_length )
    plt.xlabel('position on genome')
    plt.yticks([])
    plt.legend(handles=[line_blue, line_red])
    
    # Save figure
    if save_plots_bool:
        plt.savefig( dir_save_fig + "/snv_filter_recombo.png" )
    
    # Print results
    print("Number of recombination positions found: " + str(sum(nonsnp_bool)) + " of " + str(len(p_goodpos)) + " preliminary SNVs.")

    return p_nonsnp, nonsnp_bool
    


###########################################
# EVALUATING SNV QUALITY: FUNCTIONS/PLOTS #
###########################################


def plot_interactive_scatter_barplots(xcoor,ycoor,xlabel,ylabel,samplenames,annotation_mutations,countdata):
    '''
    Generates clickable scatter plot (ylabel vs xlabel).
    Upon cick, generates bar chart showing basecalls on fwd/rev reads across
    samples for that SNV position.
    
    These plots are intended for manual inspection of SNV quality.
        
    ARGUMENTS
    ---------
    
        xcoor: values to plot on x axis of scatter plot; recommended to use SNV
        position on genome
        
        ycoor: values to plot on y axis of scatter plot; recommended to use 
        mutation quality
        
        xlabel: x axis label for scatter plot
        
        ylabel: y axis label for scatter plot
        
        samplenames: array of sample names
        
        annotation_mutations: annotation table of SNV positions
        
        countdata: ATCGatcg (fwd/rev) read counts for each sample at each SNV
        position; this will be plotted for a given SNV position in the bar 
        chart
        
    NOTES
    -----

        1. Signatures of high-quality SNVs include:
            * high read coverage over the SNV position
            * consistent basecalls in forward and reverse reads
            * consitent number of reads aligned in the forward and reverse directions

        2. Signatures of questionable SNVs include:
            * impure alleles (alignment issues or contamination issues)
            * differences in the basecalls in forward vs reverse reads (alignment issues)
            * differences in the number of forward vs reverse reads (alignment issues)
            * lots of SNVs close together on the genome (likely recombination)
            * and more!

    @author: Tami Lieberman
        
    '''
    
    nsample=pl.size(countdata,axis=0)
    
    def onpick(event):
        i=event.ind[0]
        p_of_mut=annotation_mutations._get_value(i,'p')
        type_of_mut=annotation_mutations._get_value(i,'type')
        print("Index of selected mutation: " + str(i) )
        
        drilldownfig =plt.figure(20)
        drilldownfig.clf()
        
        formated_count_data=np.stack((countdata[0,i,:4], countdata[0,i,4:])) # initialize
        for s in range(1,nsample): # 0 to nsample not 1 to nsample
            new=np.stack((np.zeros((4)),countdata[s,i,:4], countdata[s,i,4:])) # put zeros in between samples
            formated_count_data= np.concatenate((formated_count_data,new)) # add next sample
              
        
        axsub = drilldownfig.subplots(1)
        stackedbar = pd.DataFrame(formated_count_data )
        stackedbar.plot(kind='bar',stacked=True, width=1, ax=axsub)
        pl.legend(['A','T','C','G'])
        axsub.set_ylabel('counts')
        pl.xticks(ticks=range(0,nsample*3,3), labels=samplenames) 
        pl.title('p='+str(p_of_mut)+', type='+type_of_mut)
        drilldownfig.tight_layout()
        drilldownfig.show()
       
        
    fig, axs = plt.subplots()
    axs.scatter(xcoor,  ycoor, alpha=0.6, s=12, picker=True)
        
    plt.xlabel(xlabel,fontsize=16)
    plt.ylabel(ylabel,fontsize=16)
    fig.canvas.mpl_connect('pick_event', onpick)
    
    return fig


def make_calls_qc_heatmaps( my_cmt, my_calls, save_plots_bool, dir_save_fig ):
    ''' 
    Make three heatmaps of calls, quals, and coverage (over SNV positions for all samples). 
    
    NOTES
    -----
    
        1. This function is intended to be used to help evaluate SNV quality 
        after filtering. It should be used in combination with 
        plot_interactive_scatter_barplots.
    
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
    # plt.figure()
    # plt.clf() # reset plot axes
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
    tick_labels_for_colorbar = ints2nts(ticks_for_colorbar)
    cbar = plt.colorbar(im, ticks=ticks_for_colorbar)
    cbar.ax.set_yticklabels(tick_labels_for_colorbar)
    cbar.ax.tick_params(labelsize=8)
    # #TODO: make colorbar discrete colors
    # Layout
    fig1.tight_layout()
    plt.show()
    # Save figure
    if save_plots_bool:
        plt.savefig( dir_save_fig + "/snv_qc_heatmap_calls.png" )
    
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
    plt.show()
    # Save figure
    if save_plots_bool:
        plt.savefig( dir_save_fig + "/snv_qc_heatmap_coverage.png" )
    
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
    plt.show()
    # Save figure
    if save_plots_bool:
        plt.savefig( dir_save_fig + "/snv_qc_heatmap_quals.png" )
    
    return [ fig1, fig2, fig3 ]
    


########################
# SNV TABLE: FUNCTIONS #
########################


def annotate_mutations( my_rg , p_gp , ancnti_gp , calls_gp , my_cmt_gp , fixedmutation_gp , mutQual_gp, promotersize ):
    '''
    Makes a dataframe with annotations for each SNV position. 
        
    ARGUMENTS
    ---------
    
        my_rg: reference genome object
        
        p_gp: SNV positions on genome (should already be filtered)
        
        ancnti_gp: ancestral nucleotide at SNV positions
        
        calls_gp: bascalls across samples for each SNV position
        
        my_cmt_gp: candidate mutation table data object across SNV positions
        
        fixedmutation_gp: whether or not each sample has a mutation across SNV
        positions
        
        mutQual_gp: mutation quality (from compute_mutation_quality) for each
        SNV position
        
        promoter_size: size of promoter region in bp
                
    RETURNS
    -------

        dataframe_mut: pandas dataframe with information on annotations for 
        each SNV
        
    NOTES
    -----
    
        1. Version history: This function was based on MATLAB 
        annotate_mutations_gb and append_annotations.
    
    '''       

    # Get information from candidate mutation table object    
    maNT_gp = my_cmt_gp.major_nt
    minorNT_gp = my_cmt_gp.minor_nt
    
    # Get info from reference genome object
    annotation_genes = my_rg.annotations
    locus_tags = my_rg.locus_tagnumbers
    cds_tags = my_rg.cds_indices

    # Get info at mutation positions
    contigpos_gp = my_rg.p2contigpos( p_gp )  
    mut_locus_tags = locus_tags[p_gp-1]
    mut_cds_indices = cds_tags[p_gp-1]    
    
    # Initialize list of dictionaries, where each entry will correspond to one SNV position
    lod_mutAnno = [] 
    
    # Loop through SNV positions
    for i,pos in enumerate(p_gp):

        mut_annotations = {} # initialize dictionary for this SNV
        
        # Get information about SNV position on genome
        mut_annotations['p'] = pos
        mut_annotations['contig_idx'] = contigpos_gp[i][0]
        mut_annotations['contig_pos'] = contigpos_gp[i][1]
        mut_annotations['quals'] = mutQual_gp[i]
        mut_annotations['gene_num_global'] = mut_locus_tags[i] # from locus_tagnumbers
        mut_annotations['gene_num'] = mut_cds_indices[i] # from cds_indices

        # Get info on annotations at this SNV position
        contig_idx = contigpos_gp[i][0]
        if mut_cds_indices[i] == int(mut_cds_indices[i]): # Intragenic
        
            # Get annotations from row in dataframe 
            p_anno = annotation_genes[contig_idx-1].iloc[int(mut_cds_indices[i])-1] # first -1 bcs contigs indexed starting at 1; second -1 bcs cds_indices indexed at 1 but dataframe rows indexed at zero
            
            mut_annotations['locustag'] = p_anno.loc['locustag']
            mut_annotations['product'] = p_anno.loc['product']
            mut_annotations['protein_id'] = p_anno.loc['protein_id']
            mut_annotations['gene'] = p_anno.loc['gene']
            mut_annotations['ontology'] = p_anno.loc['ontology_term']
            mut_annotations['note'] = p_anno.loc['note']
            mut_annotations['strand'] = p_anno.loc['strand']
            mut_annotations['loc1'] = p_anno.loc['loc1'] # first position of gene
            mut_annotations['loc2'] = p_anno.loc['loc2'] # last position of gene (inclusive)
            mut_annotations['sequence'] = p_anno.loc['sequence']
            mut_annotations['translation'] = p_anno.loc['translation']
            
            # Compute position within gene, taking into account strandedness
            if p_anno.loc['strand'] == 1: # fwd
                mut_annotations['nt_pos'] = mut_annotations['contig_pos'] - mut_annotations['loc1'] + 1 # positions on genome indexed at 1 (pos, loc1)
            elif p_anno.loc['strand'] == -1: # rev
                mut_annotations['nt_pos'] = mut_annotations['loc2'] - mut_annotations['contig_pos'] + 1 # positions on genome indexed at 1 (pos, loc2)
            else: # no strandedness
                mut_annotations['nt_pos'] = "." # can happen eg. Crispr
            if mut_annotations['nt_pos'] == ".": # observed with special 'type's like crispr annotated (rare!); leads to no AA inference
                aan = 9999999
            else:
                aan = int( (mut_annotations['nt_pos']-1 )/3 ) + 1 # get codon that harbours mutation (codons indexed starting at 1)
                ncn = mut_annotations['nt_pos'] - ((aan-1)*3) # get nucleotide within codon triplet (1, 2, or 3)
                mut_annotations['aa_pos'] = aan
            
            # Determine how a mutation *could* change the codon
            codons_ls = [] # init
            aa_ls = [] # init
            if len(mut_annotations['sequence']) >= (aan*3) and mut_annotations['translation'] != ".": # only for proteins
                codon = mut_annotations['sequence'][aan*3-2-1:aan*3] # -1 bcs sequence is indexed at 0 but positional argument aan is not
                codon = [ n for n in codon ] # turn seq.object to list (bcs seq object does not allow reassignment)
                for nt in NTs_list_without_N: # test all four nucleotide options for resulting AA change
                    if p_anno.loc['strand'] == 1:
                        codon[ncn-1] = nt
                    else:
                        codon[ncn-1] = NTs_complement_dict[nt]
                    codonSeqO = Seq( "".join(codon))
                    codons_ls.append(codonSeqO)
                    aa_ls.append(codonSeqO.translate())
            mut_annotations['codons'] = codons_ls
            mut_annotations['AA'] = [aa[0] for aa in aa_ls] # turn amino acids into string
            mut_annotations['NonSyn'] = len(set(mut_annotations['AA']))>1 # indicates if a nonsynonymous mutation is *possible*, not necessarily if it happened
            
            # Determine if any observed mutations *did* change the codon
            mut_annotations['AA_gt'] = '' # init # for filling in amino acids corresponding to observed mutations relative to the ancestor
            if len(mut_annotations['AA']) < 4:
                mut_annotations['type'] = 'U' # unknown mutation type
            # Fill in ancestral allele
            if np.unique(ancnti_gp[:,i])[0] != NTs_to_int_dict['N']: # only if ancestral allele is known; [0] ok bcs ancnti_gp should by definition never have > 1 unique element per row
                mut_annotations['anc'] = int_to_NTs_dict[np.unique(ancnti_gp[:,i])[0]]
                mut_annotations['nts'] = mut_annotations['anc'] # add ancestral allele to observed NTs; will add other NTs later
                if len(mut_annotations['AA']) == 4:
                    mut_annotations['AA_gt'] = mut_annotations['AA_gt'] + mut_annotations['AA'][ NTs_list_without_N_to_idx_dict[int_to_NTs_dict[np.unique(ancnti_gp[:,i])[0]]] ] # the AA list order corresponds to NTs list! 
            else: # ancestral allele not known, so cannot evaluate if mutations changed the codon
                mut_annotations['nts'] = "."
                mut_annotations['anc'] = "."
            # Fill in derived genotype(s) across all samples
            for j,callidx in enumerate(calls_gp[:,i]): # loop across samples
                if fixedmutation_gp[j,i] == True : # fixed mutations only
                    if calls_gp[j,i] != NTs_to_int_dict['N']:
                        mut_annotations['nts'] = mut_annotations['nts'] + int_to_NTs_dict[calls_gp[j,i]]
                        if len(mut_annotations['AA']) == 4:
                            mut_annotations['AA_gt'] = mut_annotations['AA_gt'] + mut_annotations['AA'][ NTs_list_without_N_to_idx_dict[int_to_NTs_dict[maNT_gp[j,i]]] ]
                    elif calls_gp[j,i] == -1: # if diverse (calls not a mutation), add minor and major call
                        mut_annotations['AA_gt'] = mut_annotations['AA_gt'] + mut_annotations['AA'][ NTs_list_without_N_to_idx_dict[int_to_NTs_dict[maNT_gp[j,i]]] ]
                        mut_annotations['AA_gt'] = mut_annotations['AA_gt'] + mut_annotations['AA'][ NTs_list_without_N_to_idx_dict[int_to_NTs_dict[minorNT_gp[j,i]]] ]
            if len(mut_annotations['AA']) == 4:
                mut_annotations['type'] = 'S' # initialize as S; eventually overwritten below if N
            # Remove duplicates
            mut_annotations['nts'] = ''.join(OrderedDict.fromkeys( mut_annotations['nts'] ).keys()) # get only unique nucleotides and keep order
            mut_annotations['AA_gt'] = ''.join(OrderedDict.fromkeys( mut_annotations['AA_gt'] ).keys()) # get only unique AAs and keep order
            # Record if nonsynonymous mutation
            if len(mut_annotations['AA_gt'])>1:
                mut_annotations['type'] = 'N'
            # Record all observed mutations across all isolates; E.g. K134Y, W47A, etc.
            if len(mut_annotations['AA_gt'])>1:
                mut_annotations['muts'] = []
                ancAA = mut_annotations['AA_gt'][0]
                derAAs = mut_annotations['AA_gt'][1:]
                for j,derAA in enumerate(derAAs):
                    mut_annotations['muts'].append( ancAA+str(mut_annotations['aa_pos'])+derAA )
            else:
                mut_annotations['muts'] = "."
    
        else: # Intergenic
        
            # Get info for gene prior to SNP (if any)
            if int(mut_cds_indices[i])>0: 
                p_anno = annotation_genes[contig_idx-1].iloc[int(mut_cds_indices[i])-1] # first -1 bcs contigs indexed starting at 1; second -1 bcs cds_indices indexed at 1 but dataframe rows indexed at zero
                mut_annotations['gene1'] = p_anno.loc['gene']
                mut_annotations['locustag1'] = p_anno.loc['locustag']
                mut_annotations['product1'] = p_anno.loc['product']
                mut_annotations['distance1'] = mut_annotations['contig_pos'] - p_anno.loc['loc2'] # how far back the previous gene is
                if p_anno.loc['strand'] == -1:
                    mut_annotations['distance1'] = mut_annotations['distance1'] * -1
            
            # Get info for gene after SNP (if any)
            if int(mut_cds_indices[i]+0.5) <= annotation_genes[contig_idx-1].shape[0] and annotation_genes[contig_idx-1].shape[1] != 0: 
                p_anno = annotation_genes[contig_idx-1].iloc[int(mut_cds_indices[i])] # first -1 bcs contigs indexed starting at 1; second -1 bcs cds_indices indexed at 1 but dataframe rows indexed at zero
                mut_annotations['gene2'] = p_anno.loc['gene']
                mut_annotations['locustag2'] = p_anno.loc['locustag']
                mut_annotations['product2'] = p_anno.loc['product']
                mut_annotations['distance2'] = p_anno.loc['loc1'] - mut_annotations['contig_pos'] # how far ahead the next gene is
                if p_anno.loc['strand'] == 1: # second conditional to evade empty chr (?????)
                    mut_annotations['distance2'] = mut_annotations['distance2'] * -1
            
            # Determine mutation type
            if ( 'distance1' in mut_annotations and mut_annotations['distance1'] > (-1*promotersize) and mut_annotations['distance1'] < 0) or ( 'distance2' in mut_annotations and mut_annotations['distance2'] > (-1*promotersize) and mut_annotations['distance2'] < 0):
                mut_annotations['type'] = 'P'
            else:
                mut_annotations['type'] = 'I'
            
            # Get ancestral allele (repeat of intragenic code)
            if np.unique(ancnti_gp[:,i])[0] != NTs_to_int_dict['N']: # only if ancestral allele is known; [0] ok bcs ancnti_gp should by definition never have > 1 unique element per row
                mut_annotations['anc'] = int_to_NTs_dict[np.unique(ancnti_gp[:,i])[0]] 
                mut_annotations['nts'] = mut_annotations['anc']
            else:
                mut_annotations['nts'] = "."
                mut_annotations['anc'] = "."
            # Extract derived genotype(s) across all samples
            for j,callidx in enumerate(calls_gp[:,i]): # loop across samples 
                if calls_gp[j,i] != NTs_to_int_dict['N']:
                    mut_annotations['nts'] = mut_annotations['nts'] + int_to_NTs_dict[calls_gp[j,i]]
            # Remove duplicates
            mut_annotations['nts'] = ''.join(OrderedDict.fromkeys( mut_annotations['nts'] ).keys()) # get only unique nucleotides and keep order
            
        # Append dictionary for this SNV to list of dictionaries
        lod_mutAnno.append(mut_annotations)
    
    # Turn list of dictionaries into a pandas dataframe
    dataframe_mut = pd.DataFrame(lod_mutAnno)
    
    return dataframe_mut



#########################
# TREEMAKING: FUNCTIONS #
#########################


def generate_tree(calls_for_tree,treeSampleNamesLong,sampleNamesDnapars,refgenome,dir_output,filetag,buildTree=False,writeDnaparsAlignment=False):
    '''
    Creates a parsimony tree (calling dnapars) with provided basecalls at SNV
    positions. 
    '''       
    
    # Write alignment file (as fasta)
    # calc NJ or Parsimonous tree or None
    # writeDnaparsAlignment==True for writing dnapars input for usage on cluster
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

    # write alignment fasta,read alignment
    write_calls_sampleName_to_fasta(calls_for_tree,treeSampleNamesLong,timestamp) #timestamp.fa
    if writeDnaparsAlignment:
        # change tip labels and write phylip
        write_calls_sampleName_to_fasta(calls_for_tree,sampleNamesDnapars,timestamp+"_"+filetag+"_dnapars") #timestamp_dnapars.fa > for dnapars...deleted later
        # turn fa to phylip and delete fasta with short tip labels
        aln = AlignIO.read(timestamp+"_"+filetag+"_dnapars.fa", 'fasta')
        AlignIO.write(aln, timestamp+"_"+filetag+".phylip", "phylip")
        subprocess.run(["rm -f " + timestamp+"_"+filetag+"_dnapars.fa"],shell=True)
        print("Written file: " + timestamp+"_"+filetag+".phylip")
        # write parameter file
        with open(timestamp+"_"+filetag+"_options.txt",'w') as file:
            file.write(timestamp+"_"+filetag+".phylip"+"\n")
            file.write("f"+"\n")
            file.write(timestamp+"_"+filetag+"_out.dnapars"+"\n")
            file.write("5"+"\n")
            file.write("V"+"\n")
            file.write("1"+"\n")
            file.write("y"+"\n")
            file.write("f"+"\n")
            file.write(timestamp+"_"+filetag+".tree"+"\n"+"\n")

    if buildTree=='PS':
        # write phylip file with dnaparse compatible 10c samplenames
        write_calls_sampleName_to_fasta(calls_for_tree,sampleNamesDnapars,timestamp+"_dnapars") #timestamp_dnapars.fa > for dnapars...deleted later
        # turn fa to phylip and delete fasta with short tip labels
        aln = AlignIO.read(timestamp+"_dnapars.fa", 'fasta')
        AlignIO.write(aln, timestamp+".phylip", "phylip")
        subprocess.run(["rm -f " + timestamp+"_dnapars.fa"],shell=True)

        # find dnapars executable
        dnapars_path = glob.glob('dnapars')
        path_extension = "../"
        backstop = 0
        while len(dnapars_path) == 0 and backstop <= 5:
            dnapars_path = glob.glob(path_extension+'dnapars')
            path_extension = path_extension + "../"
            backstop = backstop + 1
        if len(dnapars_path) == 0:
            raise ValueError('dnapars executable could not be located.')
        elif dnapars_path[0]=='dnapars':
            dnapars_path[0] = './dnapars'
        # write parameter file
        with open(timestamp+"_options.txt",'w') as file:
            file.write(timestamp+".phylip"+"\n")
            file.write("f"+"\n")
            file.write(timestamp+"_out.dnapars"+"\n")
            file.write("5"+"\n")
            file.write("V"+"\n")
            file.write("1"+"\n")
            file.write("y"+"\n")
            file.write("f"+"\n")
            file.write(timestamp+".tree"+"\n"+"\n")

        # run dnapars
        print("Build parsimony tree...")
        #print( dnapars_path[0] + " < " + timestamp+"_options.txt > " + timestamp+"_dnapars.log")
        subprocess.run([ "touch outtree"  ],shell=True)
        subprocess.run([ dnapars_path[0] + " < " + timestamp+"_options.txt > " + timestamp+"_dnapars.log"  ],shell=True)
        #print('done')
        # re-write tree with new long tip labels
        tree = Phylo.read(timestamp+".tree", "newick")
        for leaf in tree.get_terminals():
            # print(leaf.name)
            idx = np.where(sampleNamesDnapars==leaf.name)
            if len(idx[0]) > 1:
                warnings.warn("Warning: dnapars 10c limit leads to ambigous re-naming for "+leaf.name)
                idx = idx[0][0] #np.where returns: tuple with array with index
            else:
                idx = idx[0][0] #np.where returns: tuple with array with index
            leaf.name = treeSampleNamesLong[idx]
        Phylo.write(tree, timestamp+".tree", 'nexus')
        Phylo.write(tree, dir_output+"/"+filetag+"_latest.nwk.tree", 'newick')

        # clean up
        # subprocess.run(["rm -f " + timestamp+".phylip " + timestamp+"_options.txt " + timestamp+"_dnapars.log"],shell=True)

    elif buildTree == 'NJ':
        ## biopython tree build
        print("Build NJ tree...")
        # build starting tree (NJ)
        aln = AlignIO.read(timestamp+'.fa', 'fasta')
        calculator = DistanceCalculator('identity')
        constructor = DistanceTreeConstructor(calculator, 'nj')
        treeNJ = constructor.build_tree(aln)
        # Phylo.draw(treeNJ)
        Phylo.write(treeNJ,timestamp+"_NJ.tree","nexus")
        # build parsimonous tree
        #scorer = ParsimonyScorer()
        #searcher = NNITreeSearcher(scorer)
        #constructor = ParsimonyTreeConstructor(searcher, treeNJ)
        #treePS = constructor.build_tree(aln)
        #Phylo.write(treePS,timestamp+"_PS.tree","nexus")
    return timestamp


# Functions that support treemaking

def annotate_sampleNames(samplenames,locations_long_names,patients_sbj,visits_sbj,locations_sbj):
    # extend sample name with patient/visit/location identifier
    extendend_sampleNames = np.copy(samplenames)
    for i,name in enumerate(extendend_sampleNames):
        extendend_sampleNames[i] = "S"+patients_sbj[i]+"_V"+str(visits_sbj[i])+"_"+locations_long_names[locations_sbj[i]]+"_"+name
    return extendend_sampleNames

def write_calls_sampleName_to_fasta(calls_for_tree,treeSampleNames,timestamp):
    fa_file = open(timestamp+".fa", "w")
    for i,name in enumerate(treeSampleNames):
        nucl_string = "".join(list(calls_for_tree[:,i]))
        fa_file.write(">" + name + "\n" + nucl_string + "\n")
    fa_file.close()

def build_table_for_tree_labeling(p_chr_table, treeSampleNamesLong, calls_for_tree, patient=""):
    # make new folder ('tree_counting'), wipe all previous content, add table
    if patient != "":
        subprocess.run(["rm -fr tree_counting/subject"+patient+" ; mkdir tree_counting/subject"+patient ],shell=True)
        with open("tree_counting/subject"+patient+"/for_tree_labeling.csv",'w') as csv_file:
            csv_file.write(",".join(np.append(np.array(['chr','pos']),treeSampleNamesLong))+"\n") # write header
            for i in range(p_chr_table.shape[0]):
                csv_file.write(",".join(np.append( np.array([str(p_chr_table[i,0]),str(p_chr_table[i,1])]) ,calls_for_tree[i,]))+"\n")
    else:
        subprocess.run(["rm -fr tree_counting/ ; mkdir tree_counting/ " ],shell=True)
    # build table
    with open("tree_counting/for_tree_labeling.csv",'w') as csv_file:
        csv_file.write(",".join(np.append(np.array(['chr','pos']),treeSampleNamesLong))+"\n") # write header
        for i in range(p_chr_table.shape[0]):
            csv_file.write(",".join(np.append( np.array([str(p_chr_table[i,0]),str(p_chr_table[i,1])]) ,calls_for_tree[i,]))+"\n")



############################
# SAVING TABLES: FUNCTIONS #
############################

def write_generic_csv( data, column_labels, row_labels, csv_filename ):
    '''
    Generic function for writing a CSV given a two-dimensional data array along
    with column labels and row labels
    ''' 
    
    with open( csv_filename, 'w') as f:
        # header row
        f.write(',')
        for c_label in column_labels:
            f.write(c_label)
            f.write(',')
        f.write('\n')
        # one line for each row
        for i,r_label in enumerate(row_labels):
            f.write(r_label)
            f.write(',')
            for j,c_label in enumerate(column_labels):
                f.write(str(data[i,j]))
                f.write(',')
            f.write('\n')


def write_mutation_table_as_tsv( mut_positions, mut_quality, sampleNames, annotation_mutations, calls_for_tree, names_for_tree, tsv_filename ):
    '''
    Writes a TSV file given an annotated SNV table.
    '''
    
    with open( tsv_filename, 'w') as f:
        # header
        f.write('genome_pos')
        f.write('\t')
        f.write('contig_idx')
        f.write('\t')
        f.write('contig_pos')
        f.write('\t')
        f.write('gene_num')
        f.write('\t')
        f.write('gene_num_global')
        f.write('\t')
        f.write('quality')
        f.write('\t')
        f.write('product')
        f.write('\t')
        f.write('protein_id')
        f.write('\t')
        f.write('ontology')
        f.write('\t')
        f.write('locustag')
        f.write('\t')
        f.write('strand')
        f.write('\t')
        f.write('loc1')
        f.write('\t')
        f.write('loc2')
        f.write('\t')
        f.write('sequence')
        f.write('\t')
        f.write('translation')
        f.write('\t')
        f.write('nt_pos')
        f.write('\t')
        f.write('aa_pos')
        f.write('\t')
        f.write('codons')
        f.write('\t')
        f.write('AA')
        f.write('\t')
        f.write('anc')
        f.write('\t')
        f.write('nts')
        f.write('\t')
        f.write('muts')
        f.write('\t')
        f.write('type')
        for name in names_for_tree:
            f.write('\t')
            f.write(name)
        f.write('\t\n')
        # one line for each position
        for i,pos in enumerate(mut_positions):
            #print(i)
            f.write( str(pos) )
            f.write('\t')
            f.write( str(annotation_mutations._get_value(i,'contig_idx')) )
            f.write('\t')
            f.write( str(annotation_mutations._get_value(i,'contig_pos')) )
            f.write('\t')
            f.write( str(annotation_mutations._get_value(i,'gene_num_global')) )
            f.write('\t')
            f.write( str(annotation_mutations._get_value(i,'gene_num')) )
            f.write('\t')
            f.write( str(mut_quality[i]) )
            f.write('\t')
            next_product = annotation_mutations._get_value(i,'product')
            if type(next_product)!=float:
                f.write( next_product )
            else:
                f.write( str(next_product) )
            f.write('\t')
            next_protein_id = annotation_mutations._get_value(i,'protein_id')
            if type(next_protein_id)!=float:
                f.write( next_protein_id )
            else:
                f.write( str(next_protein_id) )
            f.write('\t')
            f.write( str(annotation_mutations._get_value(i,'ontology')) )
            f.write('\t')
            next_locustag= annotation_mutations._get_value(i,'locustag')
            if type(next_locustag)!=float:
                f.write( next_locustag )
            else:
                f.write( str(next_locustag) )
            f.write('\t')
            f.write( str(annotation_mutations._get_value(i,'strand')) )
            f.write('\t')
            f.write( str(annotation_mutations._get_value(i,'loc1')) )
            f.write('\t')
            f.write( str(annotation_mutations._get_value(i,'loc2')) )
            f.write('\t')
            next_seq = annotation_mutations._get_value(i,'sequence')
            if type(next_seq)!=float: # value is nan if sequence does not exist
                f.write( str(next_seq) )
            f.write('\t')
            next_translation = annotation_mutations._get_value(i,'translation')
            if type(next_translation)!=float:
                f.write( str(next_translation) )
            f.write('\t')
            f.write( str(annotation_mutations._get_value(i,'nt_pos')) )
            f.write('\t')
            f.write( str(annotation_mutations._get_value(i,'aa_pos')) )
            f.write('\t')
            next_codons = annotation_mutations._get_value(i,'codons')
            if type(next_codons)!=float:
                for codon in next_codons:
                    f.write( str(codon)+' ' )
            f.write('\t')
            next_AA = annotation_mutations._get_value(i,'AA')
            if type(next_AA)!=float:
                for AA in next_AA:
                    f.write( AA+' ' )
            f.write('\t')
            f.write( annotation_mutations._get_value(i,'anc') )
            f.write('\t')
            f.write( annotation_mutations._get_value(i,'nts') )
            f.write('\t')
            next_muts = annotation_mutations._get_value(i,'muts')
            if type(next_muts)==list:
                for mut in next_muts:
                    f.write( mut + ',')
            else:
                f.write( '.' )
            f.write('\t')
            f.write( annotation_mutations._get_value(i,'type') )
            # Add basecalls for all samples
            for j,name in enumerate(names_for_tree):
                f.write('\t')
                f.write(calls_for_tree[j,i])
            f.write('\t\n')
    
