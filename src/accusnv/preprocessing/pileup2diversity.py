#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:14:06 2022

@author: evanqu
"""

import numpy as np
import gzip
import logging
import argparse
from accusnv import log as accusnv_log
from accusnv.preprocessing import utils as ghf
import pickle

log = logging.getLogger('accusnv')

#%% Version history
#2022.02.08: Evan: Direct translation from pileup_to_diversity_matrix_snakemake.m
#2022.10.18, Arolyn: Now works when reference genome has lowercase letters or ambiguous letters
#2022.10.23, Arolyn: Updated comments on 40 statistics to have python indexing (0-39) as opposed to matlab indexing (1-40)

#%%Some notes

# This function saves the following 40 statistics for each position on the
# reference genome for the sample being analyzed:
# [A T C G a t c g Aq ... gq Am .... gm  At .... gt Ps Pb Pm Pftd Prtd E I D]
# List of statistics by index:
# [0-3] A is the number of forward reads supporting A
# [4-7] a is the number of reverse reads supporting A
# [8-15] Aq is the average phred qualities of all A's
# [16-23] Am is the average mapping qualities of all A's
# [24-31] At is the average tail distance of all A's
# [32] -- STATISTIC NOT COMPUTED OR RECORDED IN THIS VERSION -- 
# Ps is the p value for strand bias (fishers test)
# [33] -- STATISTIC NOT COMPUTED OR RECORDED IN THIS VERSION -- 
# Pb is the p value for the base qualities being the same for the two
# different types of calls (1st major, 2nd major nt, either strand) (ttest)
# [34] -- STATISTIC NOT COMPUTED OR RECORDED IN THIS VERSION -- 
# Pm is the p value for the mapping qualities being the same for the two
# different types of calls (ttest)
# [35] -- STATISTIC NOT COMPUTED OR RECORDED IN THIS VERSION -- 
# Pftd is the p value for the tail distantces on the forward strand
# being the same for the two different types of calls (ttest)
# [36] -- STATISTIC NOT COMPUTED OR RECORDED IN THIS VERSION -- 
# Pftd is the p value for the tail distantces on the reverse strand
# being the same for the two  different types of calls (ttest)
# [37] E is number of calls at ends of a read -- STATISTIC NOT COMPUTED OR RECORDED IN THIS VERSION -- 
# [38] I is number of reads supporting insertions in the +/- (indelregion) bp region
# [39] D is number of reads supporting deletions in the +/- (indelregion) bp region

# ChrStarts: is an array holding the indices in the position dimension
# corresponding to the start of a new chromsome.

#%%
def pileup2diversity(input_pileup, path_to_ref, sample='sample'):
    """Grabs relevant allele info from mpileupfile and stores as a nice array

    Args:
        input_pileup (str): Path to input pileup file.
        path_to_ref (str): Path to reference genome file
        sample (str): Sample name, for the log.

    """
    #Set parameters
    Phred_offset=33 #mpileup is told that the reads are in fastq format and 
                    #corrects so its always at Phred+33 when the mpileup comes out
    nts='ATCGatcg'
    nts_dict={'A':0,'T':1,'C':2,'G':3,'a':4,'t':5,'c':6,'g':7}
    num_fields=40
    indelregion=3 #region surrounding each p where indels recorded 
    #get reference genome + position information
    chr_starts,genome_length,scaf_names = ghf.genomestats(path_to_ref)
    ref_seq = {r.id: str(r.seq) for r in ghf.read_fasta(path_to_ref)}

    #init
    data = np.zeros((genome_length,num_fields)) #format [[A T C G  a t c g],[...]]
        
    #Read in mpileup file
    log.debug('%s: counting reads per base at every position of the %s bp reference',
              sample, f'{genome_length:,}')
    log.debug('%s: reading %s', sample, input_pileup)
    # latin-1, not the default utf-8: the pileup is really bytes, and the call and quality columns
    # are read back out as bytes below. Any stray non-ASCII byte would otherwise stop the read dead.
    mpileup = open(input_pileup, encoding='latin-1')

    #####
    loading_bar=0
    ambiguous=0 # positions skipped because the reference base there was not A, T, C or G
    max_ambiguous=max(100, genome_length//10000) # a few of these is bad data, many is a bad reference

    for line in mpileup:

        loading_bar+=1
        if loading_bar % 500000 == 0:
            log.debug('%s: %s pileup lines read so far', sample, f'{loading_bar:,}')

        lineinfo = line.strip().split('\t')
        
        #holds info for each position before storing in data
        temp = np.zeros((num_fields))
        
        chromo = lineinfo[0]
        #position (absolute)
        if len(chr_starts) == 1:
            position=int(lineinfo[1])
        else:
            if chromo not in scaf_names:
                raise ValueError("Scaffold name in pileup file not found in reference")
            position=int(chr_starts[np.where(scaf_names == chromo)[0][0]]) + int(lineinfo[1])
            #chr_starts starts at 0
        
        #ref allele, taken from the reference FASTA and not from column 3 of the pileup: samtools has
        #been seen writing stray bytes into that column, and the FASTA is already loaded here.
        ref_str = ref_seq[chromo][int(lineinfo[1])-1] # lineinfo[1] counts from 1 within its contig
        ref = nts_dict[ref_str] % 4 if ref_str in nts_dict else -1 # -1 = ambiguous reference base
        
        #calls info (.copy(): frombuffer is read-only and calls is edited in place below)
        calls=np.frombuffer(lineinfo[4].encode('latin-1'), dtype=np.int8).copy() #to ASCII

        #qual info
        bq=np.frombuffer(lineinfo[5].encode('latin-1'), dtype=np.int8) # base quality, BAQ corrected, ASCII
        mq=np.frombuffer(lineinfo[6].encode('latin-1'), dtype=np.int8) # mapping quality, ASCII
        # distance from tail, comma-sep ints; '*' at zero-coverage positions -> empty
        # (np.fromstring text mode is removed/raises in numpy 2.x, so parse explicitly)
        td=np.array(lineinfo[7].split(','), dtype=int) if lineinfo[7] not in ('*', '') else np.array([], dtype=int)
        
        #find starts of reads ('^' in mpileup)
        startsk=np.where(calls==94)[0]
        for k in startsk:
            calls[k:k+2]=-1
            #remove mapping character, 
            #absolutely required because the next chr could be $
        
        #find ends of reads ('$' in mpileup)
        endsk=np.where(calls==36)[0]
        calls[endsk]=-1
        
        #find indels + calls from reads supporting indels ('+-')
        indelk = np.where((calls==43) | (calls==45))[0]
        for k in indelk:
            if (calls[k+2] >=48) and (calls[k+2] < 58): #2 digit indel (size > 9 and < 100)
                indelsize=int(chr(calls[k+1]) + chr(calls[k+2])) 
                indeld=2
            else: #1 digit indel (size <= 9)
                indelsize=int(chr(calls[k+1]))
                indeld=1
            #record that indel was found in +/- indelregion nearby
            #indexing is slightly different here from matlab version
            if calls[k]==45: #deletion
                if (position-indelregion-1 >= 0) and (position+indelsize+indelregion-1 < genome_length): # if in middle of contig
                    #must store directly into data as it affects lines earlier and later
                    data[position-indelregion-1:position+indelsize+indelregion-1,39]+=1
                elif position-indelregion >= 0: # if at end of contig
                    data[position-indelregion-1:,39]+=1
                else: # if at beginning of contig
                    data[:position+indelsize+indelregion-1,39]+=1
            else: #insertion
                #insertion isn't indexed on the chromosome, no need for complex stuff
                if (position-indelregion-1 >= 0) and (position+indelregion-1 < genome_length): # if in middle of contig
                    data[position-indelregion-1:position+indelregion-1,38]+=1
                elif position-indelregion >= 0: # if at end of contig
                    data[position-indelregion-1:,38]+=1
                else: # if at beginning of contig
                    data[:position+indelregion-1,38]+=1 # indelsize->indelregion 2022.10.24 Evan and Arolyn

            #remove indel info from counting
            calls[k:(k+1+indeld+indelsize)] = -1 #don't remove base that precedes an indel
        
        #replace reference matches (.,) with their actual calls
        if ref >=0: # when reference allele is not ambiguous
            calls[np.where(calls==46)[0]]=ord(nts[ref]) #'.'
            calls[np.where(calls==44)[0]]=ord(nts[ref+4]) #','
        elif np.any(calls==46) | np.any(calls==44): # reads match a reference base we cannot resolve
            ambiguous+=1
            if ambiguous > max_ambiguous:
                raise ValueError(f'{sample}: more than {max_ambiguous} positions have reads matching '
                                 f'a reference base that is not A, T, C or G. Is this the right '
                                 f'reference genome for this sample?')
            continue # leave this position at zero, as though it had no coverage

        #index reads for finding scores
        simplecalls=calls[np.where(calls>0)[0]]
        #simplecalls is a tform of calls where each calls position
        #corresponds to its position in bq, mq, td
        
        #count how many of each nt and average scores
        for nt in range(8):
            nt_count=np.count_nonzero(simplecalls == ord(nts[nt]))
            if nt_count > 0:
                temp[nt]=nt_count
                try:
                    temp[nt+8]=round(np.sum(bq[simplecalls == ord(nts[nt])])/temp[nt])-Phred_offset
                    temp[nt+16]=round(np.sum(mq[simplecalls == ord(nts[nt])])/temp[nt])-33
                    temp[nt+24]=round(np.sum(td[simplecalls == ord(nts[nt])])/temp[nt])
                except:
                    log.warning('%s: could not average base/mapping/tail-distance scores at position '
                                '%s; leaving them at zero', sample, lineinfo[1])

        
        #-1 is needed to turn 1-indexed positions to python 0-indexed
        data[position-1,:38]=temp[:38]
        
    #######
    mpileup.close()

    if ambiguous:
        log.warning('%s: skipped %s position(s) where reads matched a reference base that was not '
                    'A, T, C or G; they are reported as having no coverage', sample, f'{ambiguous:,}')

    #calc coverage: columns 0-7 hold the read counts; 8-39 hold average quality scores,
    #tail distances and indel counts, which are not reads and must not be summed in.
    coverage=np.sum(data[:,:8],1)

    # Coverage breadth and depth are the first thing to check when a sample calls no SNVs.
    covered = np.count_nonzero(coverage)
    log.debug('%s: %s pileup lines read in total', sample, f'{loading_bar:,}')
    log.info('%s: %.1f%% of the genome covered by at least one read, median depth %.1fx '
             '(mean %.1fx) over the covered part',
             sample, 100 * covered / genome_length,
             np.median(coverage[coverage > 0]) if covered else 0,
             coverage.sum() / covered if covered else 0)
    if covered / genome_length < 0.5:
        log.warning('%s: less than half the reference genome is covered; expect few or no SNV calls '
                    'from this sample', sample)

    return data

#%%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', dest='input', type=str, help='Path to input pileup',required=True)
    parser.add_argument('-r', dest='ref', type=str, help='Path to reference genome FASTA (or its directory)',required=True)
    parser.add_argument('-o', dest='output', type=str, help='Path to output diversity file', required=True)
    parser.add_argument('--sample', default='sample', help='Sample name, for the log')
    accusnv_log.add_args(parser)

    args = parser.parse_args()
    accusnv_log.setup('pileup2diversity', args.log)

    diversity_arr = pileup2diversity(args.input,args.ref,args.sample)

    with gzip.open(args.output, 'wb') as f:
        pickle.dump(diversity_arr,f)
