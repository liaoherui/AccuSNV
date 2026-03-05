#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:14:06 2022

@author: evanqu
"""

import numpy as np
import sys
import gzip
import argparse
import gus_helper_functions as ghf
import pickle

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
def pileup2diversity(input_pileup, path_to_ref):
    """Grabs relevant allele info from mpileupfile and stores as a nice array 

    Args:
        input_pileup (str): Path to input pileup file.
        path_to_ref (str): Path to reference genome file
        
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
    
    #init
    data = np.zeros((genome_length,num_fields)) #format [[A T C G  a t c g],[...]]
        
    #Read in mpileup file
    print(f"Reading input file: {input_pileup}")
    mpileup = open(input_pileup)
    
    #####
    loading_bar=0
    
    for line in mpileup:

        loading_bar+=1
        if loading_bar % 50000 == 0:
            print('.')

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
            position=int(chr_starts[np.where(chromo==scaf_names)]) + int(lineinfo[1])
            #chr_starts starts at 0
        
        #ref allele
        ref_str = lineinfo[2]; # reference allele from pileup (usually A T C or G but sometimes a different symbol if nucleotide is ambiguous)
        if ref_str in nts_dict.keys():
            ref=nts_dict[ref_str] # convert to 0123
            if ref >= 4:
                ref = ref - 4
        else:
            ref=-1 # for cases where reference base is ambiguous
        
        #calls info
        calls=np.fromstring(lineinfo[4], dtype=np.int8) #to ASCII
        
        #qual info
        bq=np.fromstring(lineinfo[5], dtype=np.int8) # base quality, BAQ corrected, ASCII
        mq=np.fromstring(lineinfo[6], dtype=np.int8) # mapping quality, ASCII
        td=np.fromstring(lineinfo[7], dtype=int, sep=',') # distance from tail, comma sep
        
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
        else: # added 2022.09.23 by Arolyn: for cases where reference allele is ambiguous, confirm there are no ,'s or .'s
            if np.any(calls==46) | np.any(calls==44):
                print( 'Line from mpileup: ' + line )
                raise ValueError('Error! Calls at this position allegedly match reference allele even though reference allele was ambiguous.')

        #index reads for finding scores
        simplecalls=calls[np.where(calls>0)[0]]
        #simplecalls is a tform of calls where each calls position
        #corresponds to its position in bq, mq, td
        
        #count how many of each nt and average scores
        for nt in range(8):
            nt_count=np.count_nonzero(simplecalls == ord(nts[nt]))
            if nt_count > 0:
                temp[nt]=nt_count
                # tem = (simplecalls == ord(nts[nt]))
                # if len(bq) < len(tem):
                #     print(lineinfo[1],' skips due to dimension problem.')
                #     continue
                try:
                    temp[nt+8]=round(np.sum(bq[simplecalls == ord(nts[nt])])/temp[nt])-Phred_offset
                    temp[nt+16]=round(np.sum(mq[simplecalls == ord(nts[nt])])/temp[nt])-33
                    temp[nt+24]=round(np.sum(td[simplecalls == ord(nts[nt])])/temp[nt])
                except:
                    print(lineinfo[1],' error at line 186-188! Skip!')

        
        #-1 is needed to turn 1-indexed positions to python 0-indexed
        data[position-1,:38]=temp[:38]
        
    #######
    mpileup.close()
    
    #calc coverage
    coverage=np.sum(data,1)
    
    return data, coverage

#%%
if __name__ == "__main__":
    
    SCRIPTS_DIR="scripts"
    sys.path.insert(0, SCRIPTS_DIR)
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', dest='input', type=str, help='Path to input pileup',required=True)
    parser.add_argument('-r', dest='ref', type=str, help='Path to reference genome',required=True)
    parser.add_argument('-o', dest='output', type=str, help='Path to output diversity file', required=True)
    parser.add_argument('-c', dest='coverage', type=str, help='Path to coverage file', required=True)
    
    args = parser.parse_args()
    
    diversity_arr, coverage_arr = pileup2diversity(args.input,args.ref)
    
    with gzip.open(args.output, 'wb') as f:
        pickle.dump(diversity_arr,f)
    
    if args.coverage:
        with gzip.open(args.coverage, 'wb') as f:
            pickle.dump(coverage_arr,f)
