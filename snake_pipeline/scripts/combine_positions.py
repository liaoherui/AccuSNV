#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 20:00:41 2022

@author: evanqu
"""
import numpy as np
import pickle
import argparse
import gzip
import gus_helper_functions as ghf

#%%
def chrpos2index(chrpos,chr_starts):
    '''Python version of chrpos2index.m

    Args:
        chrpos (arr): px2 array of position and chromsome idx.
        chr_starts (arr): Vector of chromosome starts (begins at 0).

    Returns:
        p (arr): Vector of position indexes.

    '''
    #if np.size(chrpos,0) < np.size(chrpos,1):
    #   chrpos=chrpos.T
    #    print('Reversed orientation of chrpos')
        
    if len(chr_starts) == 1:
        p=chrpos[:,1]
    else:
        p=chr_starts[chrpos[:,0]-1]+chrpos[:,1]

    return p

def generate_positions_snakemake(positions_files_list, REFGENOMEDIRECTORY):
    '''Python version of generate_positions_snakemake.m
    
    Args:
        paths_to_input_p_files (list): List of input positions files.
        REFGENOMEDIRECTORY (str): Path to reference genome.

    Returns:
        combined_pos (arr): Vector of variable positions across samples.

    '''
    
    [chr_starts,genome_length,scaf_names] = ghf.genomestats(REFGENOMEDIRECTORY)
    
    # initialize vector to count occurrances of variants across samples
    timesvariant = np.zeros((genome_length,1))
    
    for i in range(len(positions_files_list)):
        #load in positions array for sample
        with gzip.open(positions_files_list[i].rstrip('\n'),"rb") as f:
            positions=pickle.load(f)
        if positions.shape[0]==0:continue
        #if len(positions)>2:

        x=chrpos2index(positions,chr_starts)
        #if 307924 in positions:
        #    print(positions_files_list[i],x)            
        timesvariant[x]=timesvariant[x]+1
    
    #print(timesvariant[307924])
    #Keep positions that vary from the reference in at least one sample but
    #that don't vary from the reference in ALL samples
    #combined_pos = np.where((timesvariant > 0) & (timesvariant < len(positions_files_list)))[0]
    combined_pos = np.where((timesvariant > 0))[0]
    #print(307924 in combined_pos)
    
    return combined_pos
    

def combine_positions(path_to_positions_files, path_to_output_p_file, path_to_outgroup_boolean_file, REFGENOMEDIRECTORY):
    
    #in_outgroup: booleans giving whether or not each sample is in the outgroup
    print("Processing outgroup booleans...")
    in_outgroup=[]
    with open(path_to_outgroup_boolean_file) as file:
        for line in file:
            in_outgroup.append(line)
    #Bool of samples to include
    include = [not i for i in in_outgroup]
    
    #Get positions on reference genome
    [chr_starts,genome_length,scaf_names] = ghf.genomestats(REFGENOMEDIRECTORY)
    
    #Find positions with at least 1 fixed mutation relative to reference genome
    print('\n\nFinding positions with at least 1 fixed mutation...\n')
    
    positions_files_ls=[]
    with open(path_to_positions_files) as file:
        for line in file:
            positions_files_ls.append(line)
            
    # print(f"\nIngroup paths used to generate positions: {positions_files_ls[include]}")
    print(include)
    
    cp = generate_positions_snakemake(positions_files_ls,REFGENOMEDIRECTORY)
    print(f"Found {len(cp)} positions where provided vcfs called a fixed variant in at least one in-group sample \n")

    #Todo: Add candidate positions manually
    #Combine different types of positions
    allp=cp
    
    print("Saving list of all positions...")
    with open(path_to_output_p_file,"wb") as wf:
        pickle.dump(allp,wf)
    
    return

#%%
if __name__ == '__main__':
    
    # SCRIPTS_DIR="scripts"
    # sys.path.insert(0, SCRIPTS_DIR)
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', type=str, help='Path to List of input positions files',required=True)
    parser.add_argument('-r', type=str, help='Path to reference genome directory',required=True)
    parser.add_argument('-o', type=str, help='Path to output positions file', required=True)
    parser.add_argument('-b', type=str, help='Outgroup boolean', required=True)
    
    args = parser.parse_args()
    
    print()

    combine_positions(args.i,args.o,args.b,args.r)
