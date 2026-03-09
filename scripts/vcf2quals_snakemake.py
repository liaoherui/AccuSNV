#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:13:33 2022

@author: evanqu
"""
import numpy as np
import gzip
import sys
import pickle
import argparse
import gus_helper_functions as ghf

def vcf_to_quals_snakemake(path_to_vcf_file,output_path_to_quals,REFGENOMEDIRECTORY):
    '''Python version of vcf_to_quals_snakemake.py
    Given a vcf file with one file per line, grabs FQ score for each positions. Ignores lines corresponding to indels

    Args:
        path_to_vcf_file (str): Path to .vcf file.
        output_path_to_quals (str): Path to output quals file
        REFGENOMEDIRECTORY (str): Path to reference genome directory.

    Returns:
        None.
        
    '''
    [chr_starts,genome_length,scaf_names] = ghf.genomestats(REFGENOMEDIRECTORY)
    
    #initialize vector to record quals
    quals = np.zeros((genome_length,1), dtype=int)
    
    print(f"Loaded: {path_to_vcf_file}")
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
                position=int(chr_starts[np.where(chromo==scaf_names)]) + int(position_on_chr)
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
        
    print(f"Saved: {output_path_to_quals}")
    
    return

#%%
if __name__ == '__main__':
    
    SCRIPTS_DIR="scripts"
    sys.path.insert(0, SCRIPTS_DIR)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', type=str, help='Path to input vcf file',required=True)
    parser.add_argument('-r', type=str, help='Path to reference genome directory',required=True)
    parser.add_argument('-o', type=str, help='Path to output quals file (.pickle.gz)', required=True)
    
    args = parser.parse_args()
    
    vcf_to_quals_snakemake(args.i,args.o,args.r)
