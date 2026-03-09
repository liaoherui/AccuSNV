Change log (2026 Feb-Mar):

1. put all files under one folder and only keep the files nessesary (make the repo as clean as possible).

2. make bwa -t (threads) parameters available to be adjusted by users, instead of hardcoded.

3. Add samclip parameter when use bwa as the aligner.

4. Add bowtie2 parameter for alternative aligner.

5. Run the snakemake module locally even without Slurm system (e.g. on the MacOS laptop without Slurm).

6. Fix Gzip parsing function in parse_gff func in snv_module_recoded_with_dNdS.py.

7. Fix the issues with splitting of "path_to_raw_data" and "fn" in Snakefile

8.  

TODO list:

top-1: Fix problems when input too many SNVs and isolates. If there are too many SNVs or isolates in the input data, then the code may occur issues.

1. take bam file as input, request from Selena (or maybe other people also) [see chats on Slack] 
2. generate tree file (which format? need to confirm) for downstream analysis or other custom analysis? -> suggestions from Evan
3. check recombination modules -> Cause Selena found there may be some issues with that part, and she is working on that [see chats on Slack]
4. check dN/dS calculation for multi-allelic sites -> multi-allelic sites may lead to wrong call, found by Alyssa. Need to check and see whether there is a bug there. [see chats on Slack]
5. test the annotation file from batak (reported issues about this case)
6. look more into the tree building process, see if there are any issues
