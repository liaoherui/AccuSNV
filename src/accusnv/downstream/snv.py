"""Shared kernel for the downstream stages: CMT/calls/coverage data structures, npz IO,
nucleotide<->int helpers, reference genome + GFF parsing, the filter cascade, and the
small cross-stage helpers (ancestral-allele inference, state rebuild)."""
import os
import re
import sys
import glob
import logging
import gzip
import copy as cp
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from Bio import SeqIO
from BCBio import GFF

from accusnv.preprocessing.utils import tree_display_sample_names, read_fasta, resolve_fasta_path, ref_directory

log = logging.getLogger('accusnv')


# Nucleotide indexing: N=0, A/T/C/G=1/2/3/4
NTs_to_int_dict = {'N':0,'n':0,'A':1,'a':1,'T':2,'t':2,'C':3,'c':3,'G':4,'g':4}
int_to_NTs_dict = {0:'N',1:'A',2:'T',3:'C',4:'G'}
NTs_complement_dict = {'N':'N','A':'T','T':'A','C':'G','G':'C'}
NTs_list_without_N = ['A','T','C','G']
NTs_list_without_N_to_idx_dict = {'A':0,'T':1,'C':2,'G':3}


def nts2ints( np_array_of_NTs ):
    ''' Converts NTs to integers according to dictionary NTs_to_int_dict. Defaults to 0 if key is missing. '''
    return np.vectorize(NTs_to_int_dict.get)(np_array_of_NTs,0) # vectorize much faster than looping; requires numpy array input


def ints2nts( np_array_of_ints ):
    ''' Converts integers to NTs according to dictionary int_to_NTs_dict. '''
    return np.vectorize(int_to_NTs_dict.__getitem__)(np_array_of_ints) # vectorize much faster than looping; require numpy array input


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
        counts = np.array(cmt['counts'][:,:,:8])
        quals = (np.array(cmt['quals']) * -1)
        in_outgroup = np.array(cmt['in_outgroup'],dtype=bool).flatten()
        try:
            indel_counter = np.array(cmt['indel_counter'])
        except:
            indel_counter=np.zeros((quals.shape[0],quals.shape[1],2))

    # Return arrays
    return [ quals, p, counts, in_outgroup, sample_names, indel_counter ]


def read_cov_mat_npz( raw_cov_mat_file ):
    '''Loads raw coverage matrix from file.'''
    
    # Reads from file
    with open(raw_cov_mat_file, 'rb') as f:
        raw_cov_mat_npz = np.load(f,allow_pickle=True)
        raw_cov_mat = raw_cov_mat_npz['all_coverage_per_bp']
    return raw_cov_mat


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
        
        minor_nt: next most abundant basecall for each sample at each candidate 
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
                log.debug('Candidate mutation table holds %d samples', self.num_samples)
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
                log.debug('Candidate mutation table holds %d candidate positions', self.num_pos)
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
        counts_major = np.squeeze( counts_sort[:,:,3:4], axis=2 ) # number of reads for most common nucleotide
        counts_minor = np.squeeze( counts_sort[:,:,2:3], axis=2 ) # number of reads for next most common nucleotide
        # 2024-12-28 - Add by Herui - check super large fp pos
        self.counts_major=counts_major
        self.counts_minor=counts_minor
        self.major_nt = 1 + np.squeeze( counts_argsort[:,:,3:4],axis=2 ) # add major alelle attribute # 3:4 necessary to maintain 3d structure # +1 necessary because 0=N and 1-4=ATCG
        self.minor_nt = 1 + np.squeeze( counts_argsort[:,:,2:3],axis=2 ) # add minor allele attribute # 2:3 necessary to maintain 3d structure # +1 necessary because 0=N and 1-4=ATCG
        x = np.sum(self.counts[:, :, 0:8], axis=2)
        self.major_nt[x == 0] = 0
        self.minor_nt[x == 0] = 0
        
        #### 2024-08-22 - Update - Add by Herui - Add fwd major nt and rev major nt for minor-mix cases
        counts_argsort_fwd = np.argsort(self.counts[:,:,0:4] , axis=2)
        counts_argsort_rev = np.argsort(self.counts[:, :, 4:8], axis=2)
        
        self.major_nt_fwd = 1 + np.squeeze(counts_argsort_fwd[:, :, 3:4], axis=2) # Note, it can be 0
        self.major_nt_rev = 1 + np.squeeze(counts_argsort_rev[:, :, 3:4], axis=2)  # Note, it can be 0
        self.minor_nt_fwd = 1 + np.squeeze(counts_argsort_fwd[:, :, 2:3], axis=2)
        self.minor_nt_rev = 1 + np.squeeze(counts_argsort_rev[:, :, 2:3], axis=2)
        ##### calculate the fwd and rev freq
        counts_sort_fwd = np.sort(self.counts[:,:,0:4], axis=2)  # sort number of reads for each nucleotide
        
        counts_sort_rev = np.sort(self.counts[:, :, 4:8], axis=2)  # sort number of reads for each nucleotide
        counts_major_fwd = np.squeeze(counts_sort_fwd[:, :, 3:4], axis=2)  # number of reads for most common nucleotide
        counts_major_rev = np.squeeze(counts_sort_rev[:, :, 3:4], axis=2)  # number of reads for most common nucleotide
        counts_minor_fwd = np.squeeze(counts_sort_fwd[:, :, 2:3], axis=2)  # number of reads for next most common nucleotide
        counts_minor_rev = np.squeeze(counts_sort_rev[:, :, 2:3], axis=2)  # number of reads for next most common nucleotide
        self.counts_major_max=np.maximum(counts_major_fwd,counts_major_rev)
        # Compare fwd and rev major counts, take the bigger one
        self.counts_minor_max=np.maximum(counts_minor_fwd,counts_minor_rev)
        
        cov_fwd=np.sum(counts_sort_fwd,axis=2)
        cov_rev = np.sum(counts_sort_rev, axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):  # suppress warning for division by zero
            self.major_nt_freq_fwd = counts_major_fwd / cov_fwd  # add major allele frequency attribute
            self.major_nt_freq_rev = counts_major_rev / cov_rev  # add major allele frequency attribute
        self.major_nt_freq_fwd=np.nan_to_num(self.major_nt_freq_fwd, nan=0.0)
        self.major_nt_freq_rev = np.nan_to_num(self.major_nt_freq_rev, nan=0.0)
        ####### Set some elements to 0
        x=np.sum(self.counts[:,:,0:4],axis=2)
        self.major_nt_fwd[x==0]=0
        self.minor_nt_fwd[x==0]=0
        self.minor_nt_fwd[self.major_nt_freq_fwd==1]=0

        x = np.sum(self.counts[:, :, 4:8], axis=2)
        self.major_nt_rev[x == 0] = 0
        self.minor_nt_rev[x == 0] = 0
        self.minor_nt_rev[self.major_nt_freq_rev == 1] = 0








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
                self.major_nt_fwd = self.major_nt_fwd[samples_to_keep_bool,:]
                self.major_nt_rev = self.major_nt_rev[samples_to_keep_bool,:]
                self.minor_nt = self.minor_nt[samples_to_keep_bool,:]
                self.major_nt_freq = self.major_nt_freq[samples_to_keep_bool,:]
                self.minor_nt_freq = self.minor_nt_freq[samples_to_keep_bool,:]
                self.counts_major = self.counts_major[samples_to_keep_bool, :]
                self.counts_minor = self.counts_minor[samples_to_keep_bool, :]
                self.counts_major_max=self.counts_major_max[samples_to_keep_bool, :]
                self.counts_minor_max=self.counts_minor_max[samples_to_keep_bool, :]
                log.debug('Candidate mutation table samples: %d -> %d', num_samples_old, self.num_samples)
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
                self.major_nt_fwd=self.major_nt_fwd[:,positions_to_keep_bool]
                self.major_nt_rev=self.major_nt_rev[:,positions_to_keep_bool]
                self.minor_nt_fwd = self.minor_nt_fwd[:,positions_to_keep_bool]
                self.minor_nt_rev=self.minor_nt_rev[:,positions_to_keep_bool]
                self.major_nt_freq_fwd=self.major_nt_freq_fwd[:,positions_to_keep_bool]
                self.major_nt_freq_rev = self.major_nt_freq_rev[:, positions_to_keep_bool]
                self.counts_major = self.counts_major[:,positions_to_keep_bool]
                self.counts_minor = self.counts_minor[ :,positions_to_keep_bool]
                self.counts_major_max=self.counts_major_max[:,positions_to_keep_bool]
                self.counts_minor_max=self.counts_minor_max[:,positions_to_keep_bool]
                log.debug('Candidate mutation table positions: %d -> %d', num_pos_old, self.num_pos)
            else:
                raise Exception("Argument positions_to_keep_bool must be a numpy array of booleans with size num_pos.")
        except AttributeError:
            raise Exception("Argument positions_to_keep_bool must be a numpy array.")
               
            
    def copy(self):
        ''' Makes a copy of candidate mutation table object. '''
        return cmt_data_object( self.sample_names, self.in_outgroup, self.p, self.counts, self.quals, self.indel_stats, self.dataset_name )


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

        filter_calls_by_position: filter calls in bad positions

        get_calls_in_sample_subset: return array of calls in the given samples

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
            log.debug('Basecall matrix: %d samples x %d positions', self.num_samples, self.num_pos)
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
                log.debug('Basecall matrix samples: %d -> %d', num_samples_old, self.num_samples)
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
                log.debug('Basecall matrix positions: %d -> %d', num_pos_old, self.num_pos)
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
    
    def filter_calls_by_position( self, positions_to_filter ):
        ''' Filter calls in bad positions. '''
        self.calls[:,positions_to_filter] = NTs_to_int_dict['N'] # set all calls in bad positions to N by broadcasting boolean argument

    # For querying basecalls

    def get_calls_in_sample_subset( self, sample_bool ):
        ''' Return array of calls in the given samples only. '''
        return self.calls[sample_bool,:]


class cov_data_object:
    '''
    Tracks summary coverage over each contig of the reference genome.

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

        median_coverage_by_contig: median raw coverage for each sample across
        each contig on the reference genome

    METHODS
    -------

        init: generates coverage matrix data object; stores summary data over
        contigs (not the whole coverage matrix) in order to conserve memory

        filter_samples: filters coverage data along samples axis; downsizes all
        attributes along sample axis based on boolean argument

        get_median_cov_of_chromosome: returns median coverage over longest
        contig; assumes the longest contig must be chromosomal

    @author: Arolyn Conwill

    '''

    def __init__(self, raw_cov_mat, sample_names, genome_length, contig_starts, contig_names ):
        '''
        Generates coverage matrix object.
        '''

        # Save basic info
        self.sample_names = sample_names
        self.num_samples = len( self.sample_names )
        log.debug('Coverage matrix holds %d samples', self.num_samples)
        self.contig_names = contig_names
        self.num_contigs = len( self.contig_names )
        log.debug('Reference genome has %d contig(s)', self.num_contigs)
        self.genome_length = genome_length
        self.contig_starts = contig_starts
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
        for idx in range(self.num_contigs):
            c_start = contig_starts[idx]-1
            if idx<self.num_contigs-1:
                c_end = contig_starts[idx+1]
            else:
                c_end = genome_length
            np.median( raw_cov_mat[:,c_start:c_end], axis=1, out=self.median_coverage_by_contig[:,idx] )


    def filter_samples(self,samples_to_keep_bool):
        ''' Filters samples and updates all coverage data objects attributes accordingly. '''
        try:
            if ( samples_to_keep_bool.dtype.type == np.bool_ ) and ( samples_to_keep_bool.size == self.num_samples ):
                # downsize attributes along samples dimension according to samples_to_keep_bool
                num_samples_old = self.num_samples # record original number of samples
                self.sample_names = self.sample_names[samples_to_keep_bool]
                self.num_samples = len( self.sample_names )
                self.median_coverage_by_contig = self.median_coverage_by_contig[samples_to_keep_bool,:]
                log.debug('Coverage matrix samples: %d -> %d', num_samples_old, self.num_samples)
            else:
                raise Exception("Argument samples_to_keep_bool must be a numpy array of booleans with size num_samples.")
        except AttributeError:
            raise Exception("Argument samples_to_keep_bool must be a numpy array.")


    def get_median_cov_of_chromosome(self):
        ''' Grab median coverage of longest contig (which we assume is the longest chromosomal contig). '''
        idx_chromosomal_contig = np.argmax( self.contig_lengths ) # find longest  contig
        return self.median_coverage_by_contig[:,idx_chromosomal_contig]



class reference_genome_object:
    '''
    This object holds information about a reference genome. Its input is either the
    reference FASTA file (any name, optionally .gz) or the directory containing it; the
    directory must also hold a GFF annotation file ("*.gff").

    ARGUMENTS
    ---------

        dir_ref_genome: path to the reference FASTA (any name) or to the directory that
        contains it alongside a GFF file "*.gff"

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
        
        # Remember reference genome directory (the GFF + annotation cache live here). The
        # incoming value may be the FASTA path itself or its directory; normalize to the dir.
        self.dir_ref_genome = ref_directory( dir_ref_genome )

        # Read in FASTA file (resolved by name from the FASTA path or its directory)
        [ self.contig_starts, self.contig_names, self.genome_length, self.genome_seq ] = get_genome_stats_from_fasta( dir_ref_genome )

        # Read in gff
        if len(glob.glob(self.dir_ref_genome + '/*.gff*'))!=1:
            self.annotations = [] # allows creation of reference genome object even if annotations do not exist
            log.warning('No single GFF file found in %s, so the reference has no gene annotations. '
                        'Every SNV will be reported as intergenic.', self.dir_ref_genome)
        else:
            self.annotations = parse_gff( self.dir_ref_genome, self.contig_names )
        
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


def get_genome_stats_from_fasta( dir_ref_genome ):
    '''
    This function get basic genome stats from a fasta file. 
    
    Note: Positions on the genome are indexed starting with 1 (to match 
    positions in vcfs).
    '''
    
    # Read fasta file (any name / .gz; resolved from a FASTA path or its directory)
    ref_genome = read_fasta( dir_ref_genome )
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


def _seq_is_undefined(seq):
    """True if a GFF record carries no real sequence. Biopython >=1.80 raises
    UndefinedSequenceError on .count() for such records (older versions returned '?'*len)."""
    try:
        return len(seq) == seq.count('?')
    except Exception:
        return True


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
    log.debug('Reading gene annotations from %s', gff_file[0])

    # Update to support gzip file
    def open_gff(filename):
        if filename.endswith('.gz'):
            # 'rt' 表示以文本模式读取（GFF 解析器需要字符串而不是字节流）
            return gzip.open(filename, 'rt')
        else:
            return open(filename, 'r')
    
    # Check gff file available fields:
    examiner = GFF.GFFExaminer()
    with open_gff(gff_file[0]) as gff_handle:
        possible_limits = examiner.available_limits(gff_handle) # available_limits function gives a summary of feature attributes along with counts for the number of times they appear in the file
    # Make a list of all attributes in gff_type except gene and region 
    limits = dict(gff_type = [i[0] for i in possible_limits['gff_type'].keys() if i[0] != 'gene' and i[0] != 'region'] )
    
    # Read gff file: 
    
    list_of_dataframes = [] # init # each element is a pandas dataframe with all annotations for the contig; contigs are ordered according to contig_names
    tagnumber_counter = 0 # init # unique numerical identifier for all features across all contigs
    num_contigs_without_annotations = 0
   
    for contig in contig_names: # loop over contig annotations according to order in contig_names
        annotation_found = False
        with open(gff_file[0]) as gff_handle:
            for rec in GFF.parse(gff_handle, limit_info=limits): # loop over every contig, but only grab attributes specified by [limits] to save memory
  
                if rec.id == contig:
                    annotation_found = True
                    # if contig has any feature build list of dicts and append to list_of_dataframes, else append empty dataframe
                    if len(rec.features) > 0:
                        # test if seq object part of gff (prokka-based yes, but NCBI-based no >> then load ref genome.fasta)

                        if _seq_is_undefined(rec.seq):
                            for seq_record in read_fasta(dir_ref_genome):
                                if seq_record.id == rec.id:
                                    rec.seq = seq_record.seq
                            if _seq_is_undefined(rec.seq): # test if succesful
                                log.warning('Contig %s in the GFF has no matching sequence in the '
                                            'reference FASTA; its genes cannot be translated', rec.id)
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

                            # Helper function to extract position -- robust to Biopython version changes (AHM 2025.07.11)
                            def get_position(pos_obj):
                                if hasattr(pos_obj, 'position'):
                                    return pos_obj.position
                                else:
                                    return int(pos_obj)
                            
                            gene_dict['indices'] = [get_position(gene_feature.location.start)+1, get_position(gene_feature.location.end)]
                            gene_dict['loc1'] = get_position(gene_feature.location.start)+1 
                            gene_dict['loc2'] = get_position(gene_feature.location.end) 
                            
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
                        list_of_dataframes.append(pd.DataFrame())
                        num_contigs_without_annotations += 1
                    break
        if not annotation_found:
            # Keep annotations aligned one-to-one with FASTA contigs, even when
            # a contig has no feature records in the GFF.
            list_of_dataframes.append(pd.DataFrame())
            num_contigs_without_annotations += 1

    if num_contigs_without_annotations > 0:
        log.warning('%d reference contig(s) have no genes in the GFF; SNVs there will be reported '
                    'as intergenic', num_contigs_without_annotations)

    log.info('Loaded %s genes from the reference annotations across %d contig(s)',
             f'{sum(len(df) for df in list_of_dataframes):,}', len(list_of_dataframes))

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


def report_filter(pre, res):
    '''Log what one filter did. res holds -1 for positions that were already invariant or dropped
    by an earlier filter, a positive value for positions this filter dropped, and 0 for the rest.'''
    already, dropped = int(np.sum(res == -1)), int(np.sum(res > 0))
    log.info('%s: %s positions still varied between samples going in, this filter dropped %s of '
             'them, leaving %s', pre, f'{len(res) - already:,}', f'{dropped:,}',
             f'{len(res) - already - dropped:,}')


def token_generate(inmatrix_raw, inmatrix_new,pre):
    if inmatrix_raw.shape[0] == 0:
        # No candidate positions (e.g. every candidate was a high-error-rate artifact that
        # filtered out). apply_along_axis rejects a zero-length axis, so short-circuit.
        log.info('%s: no candidate positions to filter', pre)
        return np.array([], dtype=int)
    unique_counts_raw = np.apply_along_axis(lambda row: len(np.unique(row[row != 0])), axis=1, arr=inmatrix_raw)
    unique_counts_raw[unique_counts_raw ==1]=0
    unique_counts_raw[unique_counts_raw >1] = 2
    unique_counts_new = np.apply_along_axis(lambda row: len(np.unique(row[row != 0])), axis=1, arr=inmatrix_new)
    unique_counts_new[unique_counts_new ==1]=0
    unique_counts_new[unique_counts_new >1] = 4

    res=unique_counts_new-unique_counts_raw
    res[res==0]=-1
    res[res==4]=0
    res[res==2]=0
    res[res==-2]=1
    report_filter(pre, res)
    return res


def generate_tokens_last(tokens,goodpos_idx,pre):
    rep=np.where(tokens==0)[0] #remain pos after all filters

    filt= np.setdiff1d(rep, goodpos_idx)
    # A position an earlier filter already removed reads -1 for this one, as for every other
    # filter. Carrying the earlier filter's own 1 through would credit this filter with a
    # removal it did not make.
    res=np.where(tokens==0,0,-1)
    res[filt]=1
    report_filter(pre, res)
    return res


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


def process_arrays(arr1, arr2,arr3,arr4, sample_num,arr5):
    col_data_nonzero = [arr1[:, col][arr1[:, col] != 0] for col in range(arr1.shape[1])]
    column_modes = [np.unique(col)[0] if len(np.unique(col)) == 1 else ( 1 if len(col)==0 else np.argmax(np.bincount(col))) for col in col_data_nonzero]
    column_second_nonzero_modes = [sorted(set(col[col != 0]), key=lambda x: np.count_nonzero(col[col != 0] == x))[-2] if len(set(col[col != 0])) > 1 else 0 for col in col_data_nonzero]
    scount = np.sum(arr1 == column_modes, axis=0)
    mask = arr1 != np.array(column_modes) # minor sample
    mask2= arr1 == np.array(column_modes) # major sample
    mask3= arr5==np.array(column_second_nonzero_modes)

    arr3[~mask]=0 # minor sample - major count
    arr4[~mask2]=0 # major sample - minor count
    arr4[~mask3]=0
    minors_major= np.max(arr3, axis=0)
    majors_minor=np.max(arr4, axis=0)
    majors_minor[majors_minor > 2] += 2
    check_minors_majorm=minors_major<majors_minor
    arr2[mask] = 0
    result = np.sum((arr2 > 0) & (arr2 < 0.95), axis=0)

    return result/scount,check_minors_majorm


def cal_freq_amb_samples(all_p,my_cmt):
    keep_col=[]
    for p in my_cmt.p:
        if p in all_p:
            keep_col.append(True)
        else:
            keep_col.append(False)
    keep_col=np.array(keep_col)
    my_cmt.filter_positions(keep_col)
    freq_arr,check_arr=process_arrays(my_cmt.major_nt,my_cmt.major_nt_freq,my_cmt.counts_major_max,my_cmt.counts_minor_max,my_cmt.major_nt.shape[0],my_cmt.minor_nt)
    freq_d={}
    check_d={}
    c=0
    for p in my_cmt.p:
        freq_d[p]=freq_arr[c]
        check_d[p]=check_arr[c]
        c+=1
    return freq_d,check_d


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
    log.info('Recombination: %s of %s preliminary SNVs sit in a likely recombinant block. They stay '
             'in the SNV tables flagged as Whether_recomb, but are left out of dN/dS, tree building '
             'and dMRCA.', f'{int(sum(nonsnp_bool)):,}', f'{len(p_goodpos):,}')

    return p_nonsnp, nonsnp_bool


def dec_final_lab(cnn,warr,wd,gap,freq,qual,check,cutoff):
    if str(qual)=='1':
        warr[0]='0'
        warr[1]='0'
        return '0'
    if cnn=='1' and wd=='1':
        return '1'
    if cnn=='1' and wd=='0':
        return '1'
    if cnn=='0' or cnn=='skip':
        if wd=='0':
            return '0'
        else:

            if gap=='1' or freq>cutoff or check:
                return '0'
            else:
                warr[0]='1'
                
                if not re.search('s',warr[1]):
                    warr[1]=str(1-float(warr[1]))
                else:
                    warr[1]='1.0'
                return '1'


# The per-position filters, in the order they run, as (dpt key, name used in Removed_by).
FILTER_ORDER = [('qual','Qual_filter'), ('cov','Cov_filter'), ('maf','MAF_filter'),
                ('indel','Indel_filter'), ('mfas','MFAS_filter'), ('mmcp','MMCP_filter'),
                ('cpn','CPN_filter'), ('fix','Fix_filter')]

TABLE_COLUMNS = ('genome_pos\tPred_label\tCNN_pred\tWideVariant_pred\tCNN_prob\tQual_filter\t'
                 'Cov_filter\tMAF_filter\tIndel_filter\tMFAS_filter\tMMCP_filter\tCPN_filter\t'
                 'Fix_filter\tWhether_recomb\tFraction_ambiguous_samples\tGap_filter\t'
                 'CNN_pred_raw\tCNN_prob_raw\tGap_reason\tRemoved_by\n')


def removed_by(fl,dpt,p,gf,cnn_l,filt_l):
    '''The one stage that removed this position, for the Removed_by column. Only the first
    filter to fire reports 1 (later ones report -1), so this reads them in the order they ran.'''
    if fl=='1':
        return 'kept'
    for key,name in FILTER_ORDER:
        if dpt[key][p]==1:
            return name
    if gf=='1':
        return 'Gap_filter'
    if cnn_l=='skip':
        return 'not_scored_by_CNN'
    if filt_l=='1':
        return 'CNN_rescue_declined'
    return 'CNN'


def generate_cnn_filter_table(all_p,filt_res,dpt,dlab,dprob,dir_output,cmt_p,dgap,my_cmt,cutoff,dgap_reason=None):
    o=open(dir_output+'/snv_table_filtered_tmp.tsv','w+')
    o.write(TABLE_COLUMNS)
    return_bool=[]
    return_bool_all=[]
    drb={}
    drba={}
    filt={}
    written=set()
    dgap_reason=dgap_reason or {}
    # Every candidate position gets a row, so freq/check are needed for all of them. The
    # calculation is per-position, so the values for all_p are the same either way.
    freq_d,check_d=cal_freq_amb_samples(cmt_p,my_cmt)
    for p in all_p:
        drba[p] = ''
        if p not in dlab:
            cnn_l='skip'
            cnn_p='skip'
        else:
            cnn_l=str(dlab[p])
            cnn_p=str(dprob[p])
        warr=[cnn_l,cnn_p]
        if p in filt_res:
            filt_l='1'
        else:
            filt_l='0'
        # Whether_recomb is a pure flag: it no longer gates which positions are kept
        # (drb) or the final label (dec_final_lab); recombinant SNVs stay in the tables.
        recomb='1' if dpt['recomb'][p]==True else '0'
        if p in dlab:
            if dlab[p]==1:
                drb[p]=''
        if p not in dgap:
            gf='0'
        else:
            gf=dgap[p]
        freq=freq_d[p]
        check=check_d[p]
        fl=dec_final_lab(cnn_l,warr,filt_l,gf,freq,dpt['qual'][p],check,cutoff)
        freq="%.6f" % freq
        if re.search('skip',str(warr[0])):
            tem_warr=0
        else:
            tem_warr=warr[0]
        if int(fl)==0 and int(tem_warr)==0 and int(filt_l)==0:
            # Removed by both the model and the filters. This still counts as removed for the
            # SNV set (filt), but the row is written so the reason is visible.
            filt[p]=''
        o.write(str(p)+'\t'+fl+'\t'+warr[0]+'\t'+filt_l+'\t'+warr[1]+'\t'+str(dpt['qual'][p])+'\t'+str(dpt['cov'][p])+'\t'+str(dpt['maf'][p])+'\t'+str(dpt['indel'][p])+'\t'+str(dpt['mfas'][p])+'\t'+str(dpt['mmcp'][p])+'\t'+str(dpt['cpn'][p])+'\t'+str(dpt['fix'][p])+'\t'+recomb+'\t'+str(freq)+'\t'+gf+'\t'+cnn_l+'\t'+cnn_p+'\t'+dgap_reason.get(p,'.')+'\t'+removed_by(fl,dpt,p,gf,cnn_l,filt_l)+'\n')
        written.add(p)
    # Candidates neither the model nor the filters called. They were never considered above,
    # so report them here with the filter verdicts that were already computed for them.
    for p in cmt_p:
        if p in written:
            continue
        cnn_l=str(dlab[p]) if p in dlab else 'skip'
        cnn_p=str(dprob[p]) if p in dprob else 'skip'
        gf=dgap.get(p,'0')
        recomb='1' if dpt['recomb'][p]==True else '0'
        o.write(str(p)+'\t0\t'+cnn_l+'\t0\t'+cnn_p+'\t'+str(dpt['qual'][p])+'\t'+str(dpt['cov'][p])+'\t'+str(dpt['maf'][p])+'\t'+str(dpt['indel'][p])+'\t'+str(dpt['mfas'][p])+'\t'+str(dpt['mmcp'][p])+'\t'+str(dpt['cpn'][p])+'\t'+str(dpt['fix'][p])+'\t'+recomb+'\t'+"%.6f" % freq_d[p]+'\t'+gf+'\t'+cnn_l+'\t'+cnn_p+'\t'+dgap_reason.get(p,'.')+'\t'+removed_by('0',dpt,p,gf,cnn_l,'0')+'\n')
    o.close()
    for p in cmt_p:
        if p in filt:
            return_bool.append(False)
            return_bool_all.append(False)
            continue
        if p in drb:
            return_bool.append(True)
        else:
            return_bool.append(False)
        if p in drba:
            return_bool_all.append(True)
        else:
            return_bool_all.append(False)
    return np.array(return_bool),np.array(return_bool_all)


def search_ref_name(refg):
    """Return the reference name (FASTA basename without extension).

    ``refg`` may be the FASTA file itself (any name) or a directory containing it.
    """
    return re.split(r'\.', os.path.basename(resolve_fasta_path(refg)))[0]


def remove_same(my_calls_in):
    """Zero out (and flag) positions where <2 distinct non-zero calls exist across samples."""
    keep_col = []
    for i in range(my_calls_in.calls.shape[1]):
        unique_nonzero_elements = np.unique(my_calls_in.calls[:, i][my_calls_in.calls[:, i] != 0])
        if len(unique_nonzero_elements) < 2:
            my_calls_in.calls[:, i] = 0
            keep_col.append(False)
        else:
            keep_col.append(True)
    # dtype is explicit: an empty list would otherwise give a float array, which
    # filter_positions rejects (this is what a cohort with no candidate SNVs produces).
    return np.array(keep_col, dtype=bool)


def is_digit(input_string):
    return input_string.isdigit()


def plot_snv_counts_gpt(data_dict, odir=None, title="SNV Counts by Sample", figsize=(10, 6),
                        color='#1f77b4', marker='o', markersize=100,
                        xlabel="Sample Name", ylabel="SNV Count", dpi=400):
    """Scatter + histogram of per-sample SNV counts; writes png/tsv summaries to ``odir``."""
    fig, ax = plt.subplots(figsize=figsize)

    samples = list(data_dict.keys())
    counts = list(data_dict.values())
    x_pos = np.arange(len(samples))

    ax.scatter(x_pos, counts, s=markersize, c=color, marker=marker, alpha=0.7)

    # Only show x-axis labels if sample count is 20 or less
    if len(samples) <= 20:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(samples, rotation=45, ha='right')
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Adjust y-axis to start from 0
    ymin, ymax = ax.get_ylim()
    ax.set_ylim([0, ymax * 1.1])

    # Label samples with counts greater than 1000 (only on the dense layout)
    ct = 0
    for i, count in enumerate(counts):
        if count > 1000 and len(samples) > 20:
            ax.annotate(samples[i], (x_pos[i], counts[i]), textcoords="offset points",
                        xytext=(0, -50), ha='center', fontsize=10, color='red', rotation=90)
            ct += 1
            if ct > 20:
                break

    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(figsize[0], figsize[1] * 0.7))
    ax2.hist(counts, bins=20, color=color, alpha=0.7)
    ax2.set_xlabel(ylabel + " Distribution", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Histogram of SNV Counts")
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(figsize[0], figsize[1] * 0.7))
    ax3.hist(np.array(counts)[np.array(counts) <= 1000], bins=500, color=color, alpha=0.7)
    ax3.vlines(x=100, ymin=0, ymax=len(np.array(counts)[np.array(counts) <= 1000]), color='red')
    ax3.set_xlabel(ylabel + " Distribution", fontsize=12)
    ax3.set_ylabel("Frequency", fontsize=12)
    ax3.set_title("Histogram of SNV Counts (Zoomed)")
    ax3.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if odir is not None:
        os.makedirs(odir, exist_ok=True)
        fig.savefig(os.path.join(odir, "snvs_per_sample.png"), dpi=dpi, bbox_inches='tight')
        fig2.savefig(os.path.join(odir, "snvs_histogram_per_sample.png"), dpi=dpi, bbox_inches='tight')
        fig3.savefig(os.path.join(odir, "ZOOMED_snvs_histogram_per_sample.png"), dpi=dpi, bbox_inches='tight')
        log.debug('Wrote SNV count figure to %s', odir)
        with open(os.path.join(odir, "snvs_per_sample.tsv"), 'w') as f:
            f.write('sample\tsnv_count\n')
            for sample, count in data_dict.items():
                f.write(f'{sample}\t{count}\n')

    return fig, ax


def check_snv(data_file_cmt, odir):
    """Count, per sample, the positions carrying the minor (second-most-frequent) allele;
    plot them and return {sample_name: count}."""
    [quals, p, counts, in_outgroup, sample_names, indel_counter] = \
        read_candidate_mutation_table_npz(data_file_cmt)

    if not len(in_outgroup) == len(sample_names):
        in_outgroup = np.array([False] * len(sample_names))
    my_cmt = cmt_data_object(sample_names, in_outgroup, p, counts, quals, indel_counter)
    my_calls = calls_object(my_cmt)
    keep_col = remove_same(my_calls)
    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)

    def find_min_freq_elements(column):
        values, counts = np.unique(column, return_counts=True)
        nonzero_mask = values != 0
        values = values[nonzero_mask]
        counts = counts[nonzero_mask]
        sorted_indices = np.argsort(-counts)
        min_count = counts[sorted_indices[1]]
        res = values[counts == min_count]
        if len(res) > 1:
            res = [res[0]]
        return res

    if my_calls.calls.shape[1] == 0:
        # No position varies between samples (a clonal cohort, or a single sample). There is
        # nothing to count, and apply_along_axis rejects a zero-width array.
        dcs = dict(zip(sample_names, [0] * len(sample_names)))
        plot_snv_counts_gpt(dcs, odir)
        return dcs

    min_freq_elements = np.apply_along_axis(find_min_freq_elements, 0, my_calls.calls)[0]
    array = my_calls.calls
    row_match_counts = np.zeros(array.shape[0], dtype=int)
    for col_idx in range(array.shape[1]):
        matches = np.isin(array[:, col_idx], min_freq_elements[col_idx])
        row_match_counts += matches
    dcs = dict(zip(sample_names, row_match_counts))
    plot_snv_counts_gpt(dcs, odir)
    return dcs


def read_positions_file(path):
    """Read an --exclude_positions or --include_positions file: one genome_pos per line, as
    numbered in the SNV tables. Blank lines, anything after a '#', and any extra columns are
    ignored, so a block pasted from snv_table_final.tsv works as long as genome_pos is the
    first column."""
    if not path:
        return np.array([], dtype=int)
    positions = []
    for line in open(path):
        line = line.split('#')[0].split()
        if line and line[0].lower() != 'genome_pos':   # skip a pasted header
            positions.append(int(line[0]))
    return np.array(positions, dtype=int)


def rebuild_state(state_path, refg, exclude_positions=(), include_positions=()):
    """Reconstruct the in-memory objects the annotate and report stages share, from the
    ``_snv_state.npz`` that stage 1 wrote plus the reference dir. Returns a namespace.

    Mirrors new_snv_script.py lines 1085-1099 (goodpos-all objects) and 1218-1270 (tree calls).
    """
    st = np.load(state_path, allow_pickle=True)
    my_cmt = cmt_data_object(st['cmt_sample_names'], st['cmt_in_outgroup'], st['cmt_p'],
                                 st['cmt_counts'], st['cmt_quals'], st['cmt_indel_stats'])
    my_calls = calls_object(my_cmt)
    my_calls.calls = st['calls_filtered']            # restore the fully-filtered calls
    my_calls_raw_for_ancestor = calls_object(my_cmt)  # == the unfiltered line-761 object
    my_rg = reference_genome_object(refg)

    # The user's own additions and removals are applied here, the one place the annotation, tree
    # and dashboard stages all build from, so every stage sees the same set. Excluding wins where
    # a position is named in both files. The counts are recomputed rather than read back from the
    # state file for the same reason.
    forced = np.isin(st['cmt_p'], include_positions)
    goodpos_bool_all = (st['goodpos_bool_all'] | forced) & ~np.isin(st['cmt_p'], exclude_positions)
    calls_ancestral = st['calls_ancestral']
    mut_qual = st['mut_qual']
    goodpos_idx_all = np.where(goodpos_bool_all)[0]
    num_goodpos_all = len(goodpos_idx_all)
    added = int(np.count_nonzero(forced & ~st['goodpos_bool_all']))
    dropped = int(st['num_goodpos_all']) + added - num_goodpos_all
    if added or dropped:
        log.info('Including %s and excluding %s SNV position(s) at the user\'s request; %s left',
                 f'{added:,}', f'{dropped:,}', f'{num_goodpos_all:,}')
    # A position with no row in the candidate mutation table has no read evidence to report, so
    # it cannot be forced in; say so rather than leaving the user to notice it never appeared.
    missing = np.setdiff1d(np.asarray(include_positions, dtype=int), st['cmt_p'])
    if missing.size:
        log.warning('%s --include_positions position(s) are not candidate positions in this run, '
                    'so they cannot be added: %s', f'{missing.size:,}',
                    ', '.join(str(p) for p in missing[:10]) + (' ...' if missing.size > 10 else ''))
    # Recombination flag over the goodpos_all positions (same order as p_goodpos_all /
    # calls_for_tree columns). Tree building and dMRCA use ~recomb_goodpos_all to exclude
    # recombinant SNVs; the annotation tables keep every position (recombinant ones flagged).
    recomb_goodpos_all = st['recomb_bool'][goodpos_bool_all]

    # goodpos-all objects (bar charts / annotation / tables)
    my_cmt_goodpos_all = my_cmt.copy()
    my_cmt_goodpos_all.filter_positions(goodpos_bool_all)
    my_calls_goodpos_all = my_calls.copy()
    my_calls_goodpos_all.filter_positions(goodpos_bool_all)
    p_goodpos_all = my_calls_goodpos_all.p
    calls_ancestral_goodpos_all = calls_ancestral[goodpos_bool_all]

    # Calls for the tree (looser filters than the SNV-calling step)
    my_calls_tree = calls_object(my_cmt_goodpos_all)
    my_calls_tree.filter_calls_by_element(my_cmt_goodpos_all.coverage < 1)
    my_calls_tree.filter_calls_by_element(my_cmt_goodpos_all.quals < 30)
    my_calls_tree.filter_calls_by_element(my_cmt_goodpos_all.major_nt_freq < 0.75)

    # Both call arrays already cover exactly the goodpos_all positions, so they are used whole.
    calls_for_tree = calls_for_tree_raw = treesampleNamesLong = None
    if num_goodpos_all > 0:
        calls_for_tree = ints2nts(my_calls_tree.calls)
        my_calls_raw_goodpos_all = my_calls_raw_for_ancestor.copy()
        my_calls_raw_goodpos_all.filter_positions(goodpos_bool_all)
        calls_for_tree_raw = ints2nts(my_calls_raw_goodpos_all.calls)
        treesampleNamesLong = tree_display_sample_names(my_cmt_goodpos_all.sample_names)

    return SimpleNamespace(
        my_cmt=my_cmt, my_calls_raw_for_ancestor=my_calls_raw_for_ancestor,
        my_rg=my_rg, goodpos_bool_all=goodpos_bool_all,
        mut_qual=mut_qual, goodpos_idx_all=goodpos_idx_all,
        num_goodpos_all=num_goodpos_all, ref_genome_name=str(st['ref_genome_name']),
        my_cmt_goodpos_all=my_cmt_goodpos_all,
        my_calls_goodpos_all=my_calls_goodpos_all, p_goodpos_all=p_goodpos_all,
        calls_ancestral_goodpos_all=calls_ancestral_goodpos_all, my_calls_tree=my_calls_tree,
        calls_for_tree=calls_for_tree, calls_for_tree_raw=calls_for_tree_raw,
        treesampleNamesLong=treesampleNamesLong,
        recomb_goodpos_all=recomb_goodpos_all,
    )


def infer_ancestral_calls_from_raw_overlap(calls_for_ancestor, reference_genome, positions=None):
    """Infer ancestral nucleotides from raw ingroup/outgroup overlap.

    Unique ingroup/outgroup allele overlaps are accepted as known ancestors. Sites with no
    overlap use the unique major ingroup allele when available. Ambiguous sites are filled
    from the outgroup sample that best matches the uniquely inferred overlap sites. Remaining
    missing sites fall back to the reference genome.
    """
    if positions is None:
        positions = calls_for_ancestor.p
    positions = np.asarray(positions)
    pos_to_idx = {int(pos): idx for idx, pos in enumerate(calls_for_ancestor.p)}
    position_indices = np.asarray([pos_to_idx[int(pos)] for pos in positions])

    calls = calls_for_ancestor.calls[:, position_indices]
    ingroup_calls = calls[np.logical_not(calls_for_ancestor.in_outgroup), :]
    outgroup_calls = calls[calls_for_ancestor.in_outgroup, :]
    outgroup_names = calls_for_ancestor.sample_names[calls_for_ancestor.in_outgroup]

    idx_for_N = NTs_to_int_dict['N']
    valid_nts = set(NTs_to_int_dict[nt] for nt in NTs_list_without_N)
    calls_ancestral = np.zeros(len(positions), dtype='int')
    ancestor_source = np.array(['unknown'] * len(positions), dtype=object)

    for pos_idx in range(len(positions)):
        ingroup_alleles = set(int(nt) for nt in ingroup_calls[:, pos_idx] if int(nt) in valid_nts)
        outgroup_alleles = set(int(nt) for nt in outgroup_calls[:, pos_idx] if int(nt) in valid_nts)
        overlap = ingroup_alleles & outgroup_alleles
        if len(overlap) == 1:
            calls_ancestral[pos_idx] = overlap.pop()
            ancestor_source[pos_idx] = 'overlap_unique'
        elif len(overlap) > 1:
            ancestor_source[pos_idx] = 'overlap_ambiguous'
        else:
            ingroup_valid_calls = [int(nt) for nt in ingroup_calls[:, pos_idx] if int(nt) in valid_nts]
            if len(ingroup_valid_calls) == 0:
                ancestor_source[pos_idx] = 'no_overlap_no_ingroup_call'
            else:
                ingroup_nts, ingroup_nt_counts = np.unique(ingroup_valid_calls, return_counts=True)
                max_ingroup_nt_count = np.max(ingroup_nt_counts)
                ingroup_major_nts = ingroup_nts[ingroup_nt_counts == max_ingroup_nt_count]
                if len(ingroup_major_nts) == 1:
                    calls_ancestral[pos_idx] = int(ingroup_major_nts[0])
                    ancestor_source[pos_idx] = 'ingroup_major_no_overlap'
                else:
                    ancestor_source[pos_idx] = 'no_overlap_ingroup_tie'

    known_ancestor_bool = ancestor_source == 'overlap_unique'
    if outgroup_calls.shape[0] > 0 and np.any(known_ancestor_bool):
        known_ancestors = calls_ancestral[known_ancestor_bool]
        outgroup_known_calls = outgroup_calls[:, known_ancestor_bool]
        outgroup_match_counts = np.sum(outgroup_known_calls == known_ancestors, axis=1)
        outgroup_nonN_counts = np.sum(outgroup_known_calls != idx_for_N, axis=1)
        outgroup_order = sorted(
            range(outgroup_calls.shape[0]),
            key=lambda idx: (-outgroup_match_counts[idx], -outgroup_nonN_counts[idx], idx)
        )
        best_outgroup_name = outgroup_names[outgroup_order[0]]
    else:
        outgroup_order = []
        best_outgroup_name = 'none'

    unresolved_bool = calls_ancestral == idx_for_N
    for pos_idx in np.where(unresolved_bool)[0]:
        for outgroup_idx in outgroup_order:
            outgroup_nt = int(outgroup_calls[outgroup_idx, pos_idx])
            if outgroup_nt in valid_nts:
                calls_ancestral[pos_idx] = outgroup_nt
                ancestor_source[pos_idx] = 'best_outgroup_sample'
                break

    unresolved_bool = calls_ancestral == idx_for_N
    if np.any(unresolved_bool):
        calls_reference = reference_genome.get_ref_NTs_as_ints(positions)
        calls_ancestral[unresolved_bool] = calls_reference[unresolved_bool]
        ancestor_source[unresolved_bool] = 'reference'

    log.info('Ancestral alleles inferred for %s positions: %s from an unambiguous ingroup/outgroup '
             'overlap, %s from the ingroup major allele where there was no overlap, %s from the '
             'closest outgroup sample (%s), %s fell back to the reference base',
             f'{len(ancestor_source):,}',
             f"{int(np.sum(ancestor_source == 'overlap_unique')):,}",
             f"{int(np.sum(ancestor_source == 'ingroup_major_no_overlap')):,}",
             f"{int(np.sum(ancestor_source == 'best_outgroup_sample')):,}", best_outgroup_name,
             f"{int(np.sum(ancestor_source == 'reference')):,}")
    return calls_ancestral, ancestor_source
