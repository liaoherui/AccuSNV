#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing helper scripts for GUS
"""
from Bio import SeqIO
from itertools import compress
import os
import re
import numpy as np
import pickle
import csv
import glob
import subprocess
import gzip

def read_samples_CSV(spls):
    hdr_check = ['Path','Sample','FileName','Reference','Group','Outgroup','Type']
    switch = "on"
    file = open(spls, 'r')
    #initialize lists
    list_path = []; list_splID = []; list_fileN = []; list_refG = []
    list_group=[]; list_outgroup=[]; list_seqtype=[]
    for line in file:
        line = line.strip('\n').split(',')
        # Test Header. Note: Even when header wrong code continues (w/ warning), but first line not read.
        if switch == "on":
            if (line == hdr_check):
                print("Passed CSV header check")
            else:
                Warning("CSV did NOT pass header check! Code continues, but first line ignored")
            switch = "off"
            continue
        # build lists
        list_path.append(line[0].strip())
        list_splID.append(line[1].strip())
        list_fileN.append(line[2].strip())
        list_refG.append(line[3].strip())
        list_group.append(line[4].strip())
        list_outgroup.append(line[5].strip())
        list_seqtype.append(line[6].strip())

    return [list_path,list_splID,list_fileN,list_refG,list_group,list_outgroup,list_seqtype]


def split_samplesCSV(PATH_ls,SAMPLE_ls,FILENAME_ls,REF_Genome_ls,GROUP_ls,OUTGROUP_ls,TYPE_ls,outdir):
    '''Take info from samples.csv, concat by sample name & save each line as sample_info.csv in data/{sampleID}'''
    
    #Loop through unique samples
    for sample in set(SAMPLE_ls):
        # Concat info for this sample
        sample_info_bool=[s==sample for s in SAMPLE_ls]
        sample_paths=list(compress(PATH_ls,sample_info_bool))
        sample_filenames=list(compress(FILENAME_ls,sample_info_bool))
        sample_references=list(compress(REF_Genome_ls,sample_info_bool))
        sample_groups=list(compress(GROUP_ls,sample_info_bool))
        sample_outgroups=list(compress(OUTGROUP_ls,sample_info_bool))
        sample_type = list(compress(TYPE_ls, sample_info_bool))
        
        sample_info_csv=list(zip(sample_paths,[sample]*sum(sample_info_bool),sample_filenames,sample_references,sample_groups,sample_outgroups,sample_type))
        
        # make data directory for this sample if it doesn't already exist
        if not(os.path.isdir(outdir+'/data/' + sample)):
            os.makedirs(outdir+'/data/' + sample, exist_ok=True)
        # check to see if this mini csv with sample info already exists
        if os.path.isfile(outdir+'/data/' + sample + '/sample_info.csv'):
            # if so, read file
            old_file = open(outdir+'/data/' + sample + '/sample_info.csv','r')
            old_info_read = csv.reader(old_file)
            old_info = list(map(tuple, old_info_read))
            old_file.close()
            
            # check to see if the existing file is consistent with samples.csv
            if not(old_info == sample_info_csv):
                # if not, remove the old file and save sample info in a new file
                # print('Information file must be updated.')
                os.remove(outdir+'/data/' + sample + '/sample_info.csv')
                with open(outdir+'/data/' + sample + '/sample_info.csv', "w") as f:
                    writer = csv.writer(f)
                    for row in sample_info_csv:
                        writer.writerow(row)
        else: # if mini csv with sample info does not already exist
            # save sample info in mini csv
            with open(outdir+'/data/' + sample + '/sample_info.csv', "w") as f:
                writer = csv.writer(f)
                for row in sample_info_csv:
                    writer.writerow(row)



def findfastqfile(dr, smple, filename):
    ##### Add by Herui - 20240921 - Modified function based on the codes from Evan
    # Given the input path and filename, will return the fastq file (include SE, PE, different suffixs) Will gzip the file automatically.
    file_suffixs = ['.fastq.gz', '.fq.gz', '.fastq', '.fq',
                    '_001.fastq.gz', '_001.fq.gz', '_001.fastq', '_001.fq']
    # Check whether the file is the soft link
    target_f=[]
    for f in glob.glob(f"{dr}/*"):
        if not os.path.islink(f):
            target_f.append(f)
        else:
            link_abs_path = os.path.abspath(f)
            link_name = os.path.basename(f)
            target_f.append(link_abs_path+'/'+link_name)
            
    #print('target...',target_f)
    #exit()
    # Search for filename as a prefix
    files_F = [f for f in target_f if re.search(f"{filename}_?.*?R1({'|'.join(file_suffixs)})", f)]
    files_R = [f for f in target_f if re.search(f"{filename}_?.*?R2({'|'.join(file_suffixs)})", f)]
    #print(dr,smple,filename,glob.glob(f"{dr}/*"))
    # Search for filename as a directory
    file_F = []
    file_R = []
    if os.path.isdir(f"{dr}/{filename}"):
        target_f = []
        for f in glob.glob(f"{dr}/{filename}/*"):
            if not os.path.islink(f):
                target_f.append(f)
            else:
                link_abs_path = os.path.abspath(f)
                link_name = os.path.basename(f)
                target_f.append(link_abs_path+'/'+link_name)
                
        files_F = files_F + [f"{filename}/{f}" for f in target_f
                             if re.search(f"{filename}/.*_?.*?R1({'|'.join(file_suffixs)})", f)]
        files_R = files_R + [f"{filename}/{f}" for f in target_f
                             if re.search(f"{filename}/.*_?.*?R2({'|'.join(file_suffixs)})", f)]
    #print(files_F,files_R)
    if len(files_F) == 0 and len(files_R) == 0:
        # Can be single-end reads and no "1" or "2" ID in the filename
        #print(f'No file found in {dr} for sample {smple} with prefix {filename}! Go single-end checking!')
        files_F = [f for f in target_f if re.search(f"{filename}.*_?.*({'|'.join(file_suffixs)})", f)]

        if os.path.isdir(f"{dr}/{filename}"):
            files_F = files_F + [f"{filename}/{f}" for f in target_f if
                                 re.search(f"{filename}/.*_?.*({'|'.join(file_suffixs)})", f)]
        if len(files_F) == 0:
            raise ValueError(f'No file found in {dr} for sample {smple} with prefix {filename}')
        else:
            file_F = files_F[0]
            if not file_F.endswith('.gz'):
                subprocess.run("gzip " + file_F, shell=True)
                file_F += '.gz'
        # print(files_F)

    elif len(files_F) > 1 or len(files_R) > 1:
        # print(",".join(files_F))
        # print(",".join(files_R))
        raise ValueError(f'More than 1 matching files found in {dr} for sample {smple} with prefix {filename}:\n \
                         {",".join(files_F)}\n \
                         {",".join(files_R)}')

    elif len(files_F) == 1 and len(files_R) == 1:
        file_F = files_F[0]
        file_R = files_R[0]

        ## Zip fastq files if they aren't already zipped
        if not file_F.endswith('.gz'):
            subprocess.run("gzip " + file_F, shell=True)
            file_F += '.gz'
        if not file_R.endswith('.gz'):
            subprocess.run("gzip " + file_R, shell=True)
            file_R += '.gz'
    elif len(files_F) == 1 or len(files_R) == 1:
        if len(files_F) == 1:
            file_F = files_F[0]
            if not file_F.endswith('.gz'):
                subprocess.run("gzip " + file_F, shell=True)
                file_F += '.gz'
        if len(files_R) == 1:
            file_T = files_R[0]
            if not file_R.endswith('.gz'):
                subprocess.run("gzip " + file_R, shell=True)
                file_R += '.gz'
    if file_R==[]:
        file_R=''
    return [file_F, file_R]

def findfastqfile_old(dr,ID,filename):
    fwd=[]
    rev=[]
    #search for filename as directory first
    potentialhits_forward=glob.glob(dr + '/' + filename +'/*1.fastq.gz')
    potentialhits_reverse=glob.glob(dr + '/' + filename +'/*2.fastq.gz')
    if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
        fwd=potentialhits_forward[0]
        rev=potentialhits_reverse[0]
    #then search for filename as file.gz
    elif len(potentialhits_forward)==0 and len(potentialhits_reverse)==0:
        potentialhits_forward=glob.glob(dr + '/' + filename +'*1.fastq.gz')
        potentialhits_reverse=glob.glob(dr + '/' + filename +'*2.fastq.gz')
        if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
            fwd=potentialhits_forward[0]
            rev=potentialhits_reverse[0]
        #then search as unzipped file
        elif len(potentialhits_forward)==0 and len(potentialhits_reverse)==0:
            potentialhits_forward=glob.glob(dr + '/' + filename +'*1.fastq')
            potentialhits_reverse=glob.glob(dr + '/' + filename +'*2.fastq')
            if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
                subprocess.run("gzip " + potentialhits_forward[0], shell=True)  
                subprocess.run("gzip " + potentialhits_reverse[0], shell=True)
                fwd=potentialhits_forward[0]+'.gz'
                rev=potentialhits_reverse[0]+'.gz'
            else:
                foldername=glob.glob(dr + '/' + filename + '*')
                if foldername and os.path.isdir(foldername[0]):
                    foldername=foldername[0]
                    potentialhits_forward=glob.glob(foldername + '/*' + filename + '*1*.fastq.gz')
                    potentialhits_reverse=glob.glob(foldername + '/*' + filename + '*2*.fastq.gz')
                    if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
                        fwd=potentialhits_forward[0]
                        rev=potentialhits_reverse[0]
                    elif len(potentialhits_forward)==0 and len(potentialhits_reverse)==0:
                        print(foldername + '/*' + filename + '*2*.fastq.gz')
                        potentialhits_forward=glob.glob(foldername +  '/*' + filename + '*1*.fastq')
                        potentialhits_reverse=glob.glob(foldername + '/*' + filename + '*2*.fastq')
                        if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
                            subprocess.run("gzip " + potentialhits_forward[0], shell=True)  
                            subprocess.run("gzip " + potentialhits_reverse[0], shell=True)
                            fwd=potentialhits_forward[0]+'.gz'
                            rev=potentialhits_reverse[0]+'.gz'
    if not(fwd) or not(rev):
        raise ValueError('Either no file or more than 1 file found in ' + dr + 'for ' + ID)
    ##zip fastq files if they aren't already zipped
    subprocess.run("gzip " + fwd, shell=True)   
    subprocess.run("gzip " + rev, shell=True)   
    return [fwd, rev]


def read_sample_info_CSV(path_to_sample_info_csv):
    with open(path_to_sample_info_csv, 'r') as f:
        this_sample_info = f.readline()  # only one line to read
    this_sample_info = this_sample_info.strip('#').split(',')
    path = this_sample_info[0]  # remember python indexing starts at 0
    paths = path.split(' ')
    sample = this_sample_info[1]
    filename = this_sample_info[2]
    reference = this_sample_info[3]
    stype = this_sample_info[-1]
    suff=findfastqfile(path,sample,sample)
    #print(suff)
    suff[0]=os.path.basename(suff[0])
    suff[1]=os.path.basename(suff[1])
    #print(suff)
    #exit()

    return paths, sample, reference, filename, stype,suff
#Jonathan new code
# def findfastqfile(dr,ID,filename):
#     fwd=[]
#     rev=[]
#     potentialhits_forward=glob.glob(dr + '/' + filename +'/*1.fastq.gz')
#     potentialhits_reverse=glob.glob(dr + '/' + filename +'/*2.fastq.gz')
#     if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
#         fwd=potentialhits_forward[0]
#         rev=potentialhits_reverse[0]
#     elif len(potentialhits_forward)==0 and len(potentialhits_reverse)==0:
#         potentialhits_forward=glob.glob(dr + '/' + filename +'*1.fastq.gz')
#         potentialhits_reverse=glob.glob(dr + '/' + filename +'*2.fastq.gz')
#         if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
#             fwd=potentialhits_forward[0]
#             rev=potentialhits_reverse[0]
#         elif len(potentialhits_forward)==0 and len(potentialhits_reverse)==0:
#             potentialhits_forward=glob.glob(dr + '/' + filename +'*1.fastq')
#             potentialhits_reverse=glob.glob(dr + '/' + filename +'*2.fastq')
#             if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
#                 subprocess.run("gzip " + potentialhits_forward[0], shell=True)
#     # wildcards: sampleID=20221
#     # resources: mem_mb=1000, disk_mb=1000, tmpdir=/tmp
#                 subprocess.run("gzip " + potentialhits_reverse[0], shell=True)
#                 fwd=potentialhits_forward[0]+'.gz'
#                 rev=potentialhits_reverse[0]+'.gz'
#             else:
#                 foldername=glob.glob(dr + '/' + filename + '*')
#                 if foldername and os.path.isdir(foldername[0]):
#                     foldername=foldername[0]
#                     potentialhits_forward=glob.glob(foldername + '/*' + filename + '*1*.fastq.gz')
#                     potentialhits_reverse=glob.glob(foldername + '/*' + filename + '*2*.fastq.gz')
#                     if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
#                         fwd=potentialhits_forward[0]
#                         rev=potentialhits_reverse[0]
#                     elif len(potentialhits_forward)==0 and len(potentialhits_reverse)==0:
#                         print(foldername + '/*' + filename + '*2*.fastq.gz')
#                         potentialhits_forward=glob.glob(foldername +  '/*' + filename + '*1*.fastq')
#                         potentialhits_reverse=glob.glob(foldername + '/*' + filename + '*2*.fastq')
#                         if len(potentialhits_forward)==1 and len(potentialhits_reverse)==1:
#                             subprocess.run("gzip " + potentialhits_forward[0], shell=True)
#                             subprocess.run("gzip " + potentialhits_reverse[0], shell=True)
#                             fwd=potentialhits_forward[0]+'.gz'
#                             rev=potentialhits_reverse[0]+'.gz'
#     if not(fwd) or not(rev):
#         raise ValueError('Either no file or more than 1 file found in ' + dr + 'for ' + ID)
#     ##make read only --- addition 20220415 by Jonathan (jdg)
#     subprocess.run("chmod 444 " + fwd, shell=True)
#     subprocess.run("chmod 444 " + rev, shell=True)
#     ##zip fastq files if they aren't already zipped
#     subprocess.run("gzip " + fwd, shell=True)
#     subprocess.run("gzip " + rev, shell=True)
#     return [fwd, rev]

def makelink(path,sample,filename,output_dir):
    #When sample is run on a single lane
    #File name can be either a COMPLETE directory name or a file name in batch(called path in this fx)
    [fwd_file, rev_file]=findfastqfile(path,sample, filename)
    # subprocess.run('ln -s -T ' + fwd_file + ' data/' + sample + '/R1.fq.gz', shell=Trueq)    
    # subprocess.run('ln -s -T ' + rev_file + ' data/' + sample + '/R2.fq.gz', shell=True)    
    
    subprocess.run(f"ln -s -T {fwd_file} {output_dir}/{sample}/R1.fq.gz", shell=True)    
    subprocess.run(f"ln -s -T {rev_file} {output_dir}/{sample}/R2.fq.gz", shell=True)    


def cp_append_files(paths,sample,filename,output_dir):
    #When sample is run on multiple lanes with same barcode
    fwd_list=''
    rev_list=''
    for path in paths:
        #Provider name can be either a COMPLETE directory name or a file name in batch(called path in this fx)
        [fwd_file, rev_file]=findfastqfile(path,sample, filename)
        fwd_list=fwd_list+ ' ' + fwd_file
        rev_list=rev_list+ ' ' + rev_file
        print(rev_list)
        print(fwd_list)
    subprocess.run("zcat " + fwd_list + ' | gzip > ' + output_dir + '/' +  sample + '/R1.fq.gz', shell=True)
    subprocess.run("zcat " + rev_list + ' | gzip > ' + output_dir + '/' +  sample + '/R2.fq.gz', shell=True)
    

def read_fasta(REFGENOME_DIR): 
    '''Reads in fasta file. If directory is given, reads in dir/genome.fasta
    Args:
        REFGENOME_DIR (str): Path to reference genome.

    Returns: SeqIO object for reference genome.
    '''
    fasta_file = glob.glob(REFGENOME_DIR + '/genome.fasta')
    if len(fasta_file) != 1:
        fasta_file_gz = glob.glob(REFGENOME_DIR + '/genome.fasta.gz')
        if len(fasta_file_gz) != 1:
            raise ValueError('Either no genome.fasta(.gz) or more than 1 genome.fasta(.gz) file found in ' + REFGENOME_DIR)
        else: # genome.fasta.gz
            refgenome = SeqIO.parse(gzip.open(fasta_file_gz[0], "rt"),'fasta')
    else: # genome.fasta
        refgenome = SeqIO.parse(fasta_file[0],'fasta')
    
    return refgenome

def genomestats(REFGENOMEFOLDER):
    '''Parse genome to extract relevant stats

    Args:
        REFGENOMEFOLDER (str): Directory containing reference genome file.

    Returns:
        ChrStarts (arr): DESCRIPTION.
        Genomelength (arr): DESCRIPTION.
        ScafNames (arr): DESCRIPTION.

    '''

    refgenome = read_fasta(REFGENOMEFOLDER)
    
    Genomelength = 0
    ChrStarts = []
    ScafNames = []
    for record in refgenome:
        ChrStarts.append(Genomelength) # chr1 starts at 0 in analysis.m
        Genomelength = Genomelength + len(record)
        ScafNames.append(record.id)
    # close file
    #refgenome.close() # biopy update SeqIO has no close attribute anymore.
    # turn to np.arrys!
    ChrStarts = np.asarray(ChrStarts,dtype=int)
    Genomelength = np.asarray(Genomelength,dtype=int)
    ScafNames = np.asarray(ScafNames,dtype=object)
    return ChrStarts,Genomelength,ScafNames

def p2chrpos(p, ChrStarts):
    '''Convert 1col list of pos to 2col array with chromosome and pos on chromosome

    Args:
        p (TYPE): DESCRIPTION.
        ChrStarts (TYPE): DESCRIPTION.

    Returns:
        chrpos (TYPE): DESCRIPTION.

    '''
        
    # get chr and pos-on-chr
    chromo = np.ones(len(p),dtype=int)
    if len(ChrStarts) > 1:
        for i in ChrStarts[1:]:
            chromo = chromo + (p > i) # when (p > i) evaluates 'true' lead to plus 1 in summation. > bcs ChrStarts start with 0...genomestats()
        positions = p - ChrStarts[chromo-1] # [chr-1] -1 due to 0based index
        chrpos = np.column_stack((chromo,positions))
    else:
        chrpos = np.column_stack((chromo,p))
    return chrpos


def p2chrpos_add_base(p, ChrStarts, base_type):
    '''Convert 1col list of pos to 2col array with chromosome and pos on chromosome

    Args:
        p (TYPE): DESCRIPTION.
        ChrStarts (TYPE): DESCRIPTION.

    Returns:
        chrpos (TYPE): DESCRIPTION.

    '''

    # get chr and pos-on-chr
    chromo = np.ones(len(p), dtype=int)
    if len(ChrStarts) > 1:
        for i in ChrStarts[1:]:
            chromo = chromo + (
                        p > i)  # when (p > i) evaluates 'true' lead to plus 1 in summation. > bcs ChrStarts start with 0...genomestats()
        positions = p - ChrStarts[chromo - 1]  # [chr-1] -1 due to 0based index
        chrpos = np.column_stack((chromo, positions,base_type))
    else:
        chrpos = np.column_stack((chromo, p,base_type))
    return chrpos

# def get_clade_wildcards(cladeID):
#     is_clade = [int(i == cladeID) for i in GROUP_ls]
#     sampleID_clade = list(compress(SAMPLE_ls,is_clade))
#     reference_clade = list(compress(REF_Genome_ls,is_clade))
#     outgroup_clade = list(compress(OUTGROUP_ls,is_clade))
#     return sampleID_clade,reference_clade,outgroup_clade
    
# def get_sampleID_names(wildcards):  
#     sampleID_clade,_,_ = get_clade_wildcards(wildcards.cladeID)
#     return sampleID_clade

# def get_outgroup_bool(wildcards):  
#     _,_,outgroup_clade = get_clade_wildcards(wildcards.cladeID)
#     return outgroup_clade

# def get_positions_prep(wildcards):
#     sampleID_clade,reference_clade,outgroup_clade = get_clade_wildcards(wildcards.cladeID)
#     mat_positions_prep=expand("Case/temp/{sampleID}_ref_{reference}_outgroup{outgroup}_positions.pickle",zip,sampleID=sampleID_clade, reference=reference_clade, outgroup=outgroup_clade)
#     return mat_positions_prep

# def get_diversity(wildcards):
#     sampleID_clade,reference_clade,outgroup_clade = get_clade_wildcards(wildcards.cladeID)
#     diversity_mat = expand("Mapping/diversity/{sampleID}_ref_{reference}_outgroup{outgroup}.diversity.pickle.gz",zip,sampleID=sampleID_clade, reference=reference_clade, outgroup=outgroup_clade)
#     return diversity_mat   

# def get_quals(wildcards):
#     sampleID_clade,reference_clade,outgroup_clade = get_clade_wildcards(wildcards.cladeID)
#     quals_mat = expand("Mapping/quals/{sampleID}_ref_{reference}_outgroup{outgroup}.quals.pickle.gz",zip,sampleID=sampleID_clade, reference=reference_clade, outgroup=outgroup_clade)
#     return quals_mat 

# def get_ref_genome(wildcards):
#     sampleID_clade,reference_clade,outgroup_clade = get_clade_wildcards(wildcards.cladeID)
#     return reference_clade
