#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:39:04 2019

@author: fmk
"""




import argparse,subprocess,string,random
import pandas as pd



''' positional and optional argument parser'''

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description='''\
    Infer orthologs across two or more prokka-based annotations, and returns overview table for all genes.
    Homology is inferred using CD-HIT and annotations need to be in fasta format (nucleotide (*.ffn) or amino acid (*.faa))
    CD-HIT: %identity optional. Fixed: -s 0.9, ie. shorter sequences need to be 	at least 90% length of the representative of the cluster.
                                ''',
                                epilog="Questions or comments? --> fkey@mit.edu")
parser.add_argument("-f", dest="file_sample_annotation", help="2-col TSV file with subject-identifier and annotation file path.",type=argparse.FileType('r'),required=True)
parser.add_argument('-p', dest="percentIdentity", action="store", default='0.98', help="Percent identity cd-hit. Default: 0.98")
parser.add_argument('-o', dest="outpath", action="store", help="Output path.",required=True)
parser.add_argument("-c", dest="cdhit", help="Path to CD-HIT executable", action="store",default="cd-hit")
parser.add_argument('-m', dest="memory", action="store", default='8000', help="mem for CDHIT; default 8G")

args = parser.parse_args()



''' FUNCTIONS'''

def fix_path(path):
    # make sure path has trailing "/"
    if path[-1] != "/":
        path = path + "/"
        return path
    else:
        return path
    

def read_merge_sample_annotation_file(file_sample_annotation):
    # get list of annotation file paths, tuple/dict of subjectID and prokka-assigned gene-tag (use first line)
    subj_tag_dict = {}
    subprocess.run(['mkdir','-p',outpath],check=True)
    with open(outpath+"merged_annotation.fa", 'w') as outfile:
#        with open(file_sample_annotation,'r') as infile:
        for line in file_sample_annotation:
            line = line.strip().split('\t')
            print('1')
            print(line)
            # read annotation file: extract prokka-gene-tag (from 1st line) and merge all annotation files into one
            with open(line[1]) as annofile:
                # link subjectID with prokka gene tag
                first_line = annofile.readline()
                if first_line.startswith('>'):
                    prokka_gene_tag = first_line.strip().split(' ')[0].split('>')[1].split('_')[0]
                    subj_tag_dict[ line[0] ] = prokka_gene_tag
                else:
                    raise ValueError('Annotation file does not start with ">": '+first_line+' in '+line[1])
                # write annotation files into merged outfile
                outfile.write(first_line) # necessary bcs already read
                outfile.write(annofile.read())
        file_sample_annotation.close()
    return subj_tag_dict
        
        
def read_cdhit_cluster(cdhit_clstr,percentIdentity,prokka_tag_list):
    # read cdhit results and build for each cluster entry in dict
    ctr = 1
    ctr_cluster_within_subject = 0
    rdm_tag = ''.join(random.choice(string.ascii_lowercase) for x in range(4)) # added to cluster-tags to avoid confusion w/ different runs
    saab_cluster_genes = {}
#    value_default = ['NA' for i in prokka_tag_list]
    with open(cdhit_clstr) as infile:
        for line in infile:
            if line.startswith('>'):                
                cluster_tag = "SAAB_" + "%05d" % ctr + "_pid" + percentIdentity + "_" + rdm_tag
                saab_cluster_genes[cluster_tag] = ['NA' for i in prokka_tag_list]
                ctr += 1
            else:
                line = line.strip().split('\t')[1].split(" ")[1].split('.')[0].split('>')[1] # remodel string so it matches prokka-gene_id eg. "0       10155aa, >JKPBNODD_00001... *"
                subject_identifier_prokka = line.split('_')[0]             
                subject_index = prokka_tag_list.index( subject_identifier_prokka )
                if saab_cluster_genes[cluster_tag][subject_index] == 'NA':
                    saab_cluster_genes[cluster_tag][subject_index] = line
                else:
                    saab_cluster_genes[cluster_tag][subject_index] = saab_cluster_genes[cluster_tag][subject_index] + "," + line
                    ctr_cluster_within_subject += 1
    if ctr_cluster_within_subject > 0:
        print('NOTE: ' + str(ctr_cluster_within_subject) + " occasions where a gene cluster had >1 gene from the same individual assigned.")
    return saab_cluster_genes



''' MAIN '''
# TEST Vars

if __name__ == "__main__":
    # assign argparse arguments
    file_sample_annotation = args.file_sample_annotation
#    annopath = fix_path(args.annopath) # fix path to annotation has trailing "/"
    
    outpath = fix_path(args.outpath)
#    filetype = args.filetype
    cdhit_executable = args.cdhit
    percentIdentity = args.percentIdentity
    mem = args.memory
    # get concatenated annotation file (output: merged_annotation.fa) and dict[subject]=prokka-tag
    subj_tag_dict = read_merge_sample_annotation_file(file_sample_annotation)
    subject_list_ord = list(subj_tag_dict.keys())
    prokkaTag_list_ord = [ subj_tag_dict[k] for k in subject_list_ord ]
    
    # cd-hit
    command_cdhit = cdhit_executable + " -s 0.9 -M " + mem  + " -c " + percentIdentity + " -i " + outpath + "merged_annotation.fa" + " -o " + outpath+"cdhit_results"
    subprocess.run(command_cdhit,shell=True)

    # read-in cdhit results: dict[SAAB_XXXXX_pidZZZ_YYY]=[geneX,geneY,geneZ]
    cdhit_res_dict = read_cdhit_cluster(outpath+"cdhit_results.clstr",percentIdentity,prokkaTag_list_ord)
    
    # build table of gene annotation
    cdhit_res_df = pd.DataFrame.from_dict(cdhit_res_dict,orient='index',columns=subject_list_ord)
    
    # write cdhit res
    cdhit_res_df.to_csv(outpath+'annotation_orthologs.tsv',sep="\t")
    