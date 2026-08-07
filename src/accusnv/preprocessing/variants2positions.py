#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:36:31 2022

@author: evanqu
"""
import numpy as np
import gzip
import pickle
import logging
import argparse
from accusnv import log as accusnv_log
from accusnv.preprocessing import utils as ghf

log = logging.getLogger('accusnv')

#%%
def generate_positions_single_sample(path_to_variant_vcf,path_to_output_positions,maxFQ,REFGENOMEDIRECTORY,sample):
    '''Python version of generate_positions_single_sample_snakemake.m

    Only ingroup samples reach here: the workflow asks for a positions file for those alone,
    because outgroup samples do not contribute candidate positions.

    Args:
        path_to_variant_vcf (str): Path to .variant.vcf.gz file.
        path_to_output_positions (str): Output path to positions file (.pickle.gz)
        maxFQ (int): Purity threshold for including position.
        REFGENOMEDIRECTORY (str): Path to reference genome directory.
        sample (str): Sample name, for the log.

    Returns:
        None.

    '''
    log.debug('%s: looking for candidate variant positions in the bcftools calls', sample)
    log.debug('%s: reading %s (keeping single-base substitutions with FQ below %d)',
              sample, path_to_variant_vcf, int(maxFQ))

    [chr_starts,genome_length,scaf_names] = ghf.genomestats(REFGENOMEDIRECTORY)

    # Initialize boolean vector for positions to include as candidate SNPs that
    # vary from the reference genome
    include = np.zeros((genome_length,1))

    file = gzip.open(path_to_variant_vcf,'rt')

    num_records = num_simple = 0
    for line in file:
        if not line.startswith("#"):
            num_records += 1
            lineinfo = line.strip().split('\t')

            chromo=lineinfo[0]
            position_on_chr=lineinfo[1] #1-indexed
            
            if len(chr_starts) == 1:
                position=int(lineinfo[1])
            else:
                if chromo not in scaf_names:
                    raise ValueError("Scaffold name in vcf file not found in reference")
                position=int(chr_starts[np.where(scaf_names == chromo)[0][0]]) + int(position_on_chr)
                #chr_starts begins at 0
                
            alt=lineinfo[4]
            ref=lineinfo[3]
            
            #only consider for simple calls (not indel, not ambiguous)
            if (alt) and ("," not in alt) and (len(alt) == len(ref)) and (len(ref)==1):
                num_simple += 1
                #find and parse quality score
                xt = lineinfo[7]
                xtinfo = xt.split(';')
                entrywithFQ=[x for x in xtinfo if x.startswith('FQ')][0]
                fq=entrywithFQ[entrywithFQ.index("=")+1:]
                
                if float(fq) < maxFQ: #better than maxFQ
                    include[position-1]=1
                    #-1 converts position (1-indexed) to index
    #+1 converts index back to position for p2chrpos
    Var_positions=ghf.p2chrpos(np.nonzero(include)[0]+1,chr_starts)

    #save
    with gzip.open(path_to_output_positions,"wb") as f:
        pickle.dump(Var_positions,f)

    log.debug('%s: %d variant records in the VCF, %d of them single-base substitutions',
              sample, num_records, num_simple)
    log.info('%s: %d candidate variant positions kept (%.4f%% of the %s bp genome)',
             sample, len(Var_positions), 100 * len(Var_positions) / genome_length, f'{genome_length:,}')

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', type=str, help='Path to input variant vcf',required=True)
    parser.add_argument('-r', type=str, help='Path to reference genome FASTA (or its directory)',required=True)
    parser.add_argument('-o', type=str, help='Path to output positions file', required=True)
    parser.add_argument('-q', type=int, help='MaxFQ threshold', required=True)
    parser.add_argument('--sample', default='sample', help='Sample name, for the log')
    accusnv_log.add_args(parser)

    args = parser.parse_args()
    accusnv_log.setup('variants2positions', args.log)

    generate_positions_single_sample(args.i,args.o,args.q,args.r,args.sample)

    
