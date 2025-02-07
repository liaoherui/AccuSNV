# Local analysis


## Overview

This is a tutorial for local analysis portion of Lieberman Lab's SNV-calling pipeline. The core function of the local analysis script is to filter the candidate SNVs that come out of the snakemake pipeline (mapping and case steps) and generate a parsimony tree indicating how samples are related to each other.

The inputs to the local analysis script include: a reference genome (fasta file) with annotations (gff file), a candidate mutation table (generated in the snakemake pipeline case step), and a coverage matrix (also generated in the snakemake pipeline case step). 

It is imparative to set appropriate filters for your dataset, and you should not assume that the filters and filter thresholds in this tutorial can be applied to your data. Please make use of the interactive diagnostic tools built into the script to evaluate your filter strategy. 


## How to run the local analysis script, tutorial

Here are instructions for [how to run](readme_local_run.md) the local analysis script. This includes some details on false positive SNPs and real positive SNPs beyond what is included below, make sure to read both. 


## Caveats

### False-positive and false-negative SNVs

False-positive SNVs can arise from:
* contaminated samples 
* poor alignments to the reference genome
* indels, recombination, and mobile elements (true genetic variants but not SNVs)
* generally loose filters

False-negative SNVs can arise from:
* insufficient read coverage
* missing gene content in the reference genome
* overly aggressive filters

...and more reasons not listed here!


### Reference genome considerations

The reference genome is the genome to which you align reads from your samples. Ideally, your reference genome is closely related to your samples. 

If your reference genome is not closely related to your samples, then you may miss SNVs on genes that are not present in your reference genome or you may have issues with reads from your samples aligning poorly to the reference. 

Building a co-assembly from your samples is a good way to ensure you have a closely related reference genome. However, you can still run into issues with read alignments on areas where the assembly was challenging, and you can run into additional problems if the reads going into your assembly were not pure (contamination can give rise to extra contigs representing gene content that is not present truly present in your samples). In the case that the reads going into the co-assembly contained low-level contamination, it may be worthwhile to filter the contigs in the assembly before attempting to call SNVs.


### Ancestral alleles and outgroups

The only reliable way to root your phylogeny is by using outgroup to infer ancestral alleles. Some methods for finding outgroups include:
* Querying public databases for closely related genomes of the same species
* Using other samples in your dataset that are closely related but independent from your samples of interest

Words of caution: 
* You should not assume your reference genome is an outgroup, particularly if you assembled it from your samples (then it is definitively an ingroup). 
* It is difficult to determine the ancestral allele of a nucleotide that is under strong selection across many environments (e.g. a point mutation conferring antibiotic resistance).


## Wishlist

Here is a [wishlist](readme_local_wishlist.md) of features to add to the current version of the local analysis script.


## Table of contents

[Main Lieberman Lab pipeline README](../README.md)
* [Snakemake pipeline](readme_snake_main.md)
	* [How to run the snakemake pipeline](readme_snake_run.md)
	* [Technical details about the snakemake pipeline](readme_snake_rules.md)
	* [Wishlist for snakemake pipeline upgrades](readme_snake_wishlist.md)
	* [Helpful hints for using the command line](readme_snake_basics.md)
* [Local analysis](readme_local_main.md)
	* [How to run the local analysis script](readme_local_run.md)
	* [Wishlist for local analysis upgrades](readme_local_wishlist.md)
	* [Python best practices](readme_local_best.md)
