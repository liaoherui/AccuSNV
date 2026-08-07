#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:13:33 2022

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

def vcf_to_quals_snakemake(path_to_vcf_file,output_path_to_quals,REFGENOMEDIRECTORY,sample):
    '''Python version of vcf_to_quals_snakemake.py
    Given a vcf file with one file per line, grabs FQ score for each positions. Ignores lines corresponding to indels

    Args:
        path_to_vcf_file (str): Path to .vcf file.
        output_path_to_quals (str): Path to output quals file
        REFGENOMEDIRECTORY (str): Path to reference genome directory.
        sample (str): Sample name, for the log.

    Returns:
        None.

    '''
    [chr_starts,genome_length,scaf_names] = ghf.genomestats(REFGENOMEDIRECTORY)

    #initialize vector to record quals
    quals = np.zeros((genome_length,1), dtype=int)

    log.debug('%s: reading FQ call-quality scores for all %s genome positions', sample, f'{genome_length:,}')
    log.debug('%s: reading %s', sample, path_to_vcf_file)
    file = gzip.open(path_to_vcf_file,'rt') #load in file
    
    for line in file:
        if not line.startswith("#"):
            lineinfo = line.strip().split('\t')
            
            #Note: not coding the loading bar in the matlab script
            
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
                #find and parse quality score
                xt = lineinfo[7]
                xtinfo = xt.split(';')
                entrywithFQ=[x for x in xtinfo if x.startswith('FQ')][0]
                fq=float(entrywithFQ[entrywithFQ.index("=")+1:])
                
                #If already a position wiht a stronger FQ here, don;t include this
                #More negative is stronger
                if fq < quals[position-1]:
                    quals[position-1]=round(fq) 
                        #python int(fq) will by default round down, round matches matlab behavior
                        #-1 important to convert position (1-indexed) to python index
    
    #save
    with gzip.open(output_path_to_quals,"wb") as f:
        pickle.dump(quals,f)

    scored = int(np.sum(quals < 0))
    log.info('%s: %s positions have a call-quality score (%.1f%% of the genome); best FQ %d',
             sample, f'{scored:,}', 100 * scored / genome_length, int(quals.min()))
    log.debug('%s: wrote %s', sample, output_path_to_quals)

    return

#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', type=str, help='Path to input vcf file',required=True)
    parser.add_argument('-r', type=str, help='Path to reference genome FASTA (or its directory)',required=True)
    parser.add_argument('-o', type=str, help='Path to output quals file (.pickle.gz)', required=True)
    parser.add_argument('--sample', default='sample', help='Sample name, for the log')
    accusnv_log.add_args(parser)

    args = parser.parse_args()
    accusnv_log.setup('vcf2quals', args.log)

    vcf_to_quals_snakemake(args.i,args.o,args.r,args.sample)
