"""Stage 4: compute dN/dS from the annotated final table.

Reads ``snv_table_final.tsv`` + the reference from the group dir; writes
``dNdS_out/data_dNdS.npz``, the same numbers as ``dnds_genomewide.tsv``, and the per-gene
breakdown ``dnds_per_gene.tsv`` to the analysis dir. Includes the mutation-spectrum /
expected-vs-observed dN/dS library functions inline.

SNVs flagged as recombinant are excluded, so this covers the same SNV set as the dMRCA tables.

Transcribed from new_snv_script.py lines ~1464-1525.
"""
import os
import logging
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from Bio.Seq import Seq

from accusnv import log as accusnv_log
from accusnv.downstream import snv
from accusnv.preprocessing.utils import ref_directory, genomestats

log = logging.getLogger('accusnv')



def compute_expected_dnds(all_genes, mut_spec_prob, gene_nums_of_interest=None):
    '''
    @author: Alyssa Mitchell, 2025.03.20

    Parameters
    ----------
    mut_spec_prob : output of mutation_spectrum_module, 6-entry vector of observed mutation types, normalized to sum
    gene_nums_of_interest : optional
        The default is None. Used to provide flexibility to calculate null probability of nonsynonymous muts in given gene(s) rather than entire genome.
        This refers to the global gene number (does not restart when multiple contigs)

    Returns
    -------
    probnonsyn : Expected value probability of nonsynonymous muts, given genome characteristics and assumption of neutral mutation.
        Compared to observed values to calculate dN/dS.

    '''
    def codon_composition_table(bases, codons):
        # Initialize table
        table = np.zeros((len(codons), len(bases)))
        # Loop through all codons
        for i in range(len(codons)):
            # Loop through all bases
            for j in range(len(bases)):
                # Count the number of times a base occurs in a given codon
                # Then update the count in the appropriate position in the table
                table[i, j] = codons[i].count(bases[j])
        return table
    
    def prob_nonsyn_codon_mutation(codon, mut):
        # Convert the codon to a Seq object
        codon_seq = Seq(codon)
        
        aa0 = codon_seq.translate(table=11)  # amino acid before mutation
        
        # Find the positions on the codon at which mutation could occur
        possible_muts = [i for i, nt in enumerate(codon) if nt == mut[0]]
        
        if not possible_muts:  # if the mutation cannot occur on this codon
            probability = None
        else:  # if the mutation can occur on this codon
            tally = 0  # initialize variable to count mutations that are nonsynonymous
            for i in possible_muts:  # loop through possible mutations
                new_codon = list(codon)  # create a mutable copy of the codon
                new_codon[i] = mut[1]  # perform mutation
                aaf = ''.join(new_codon)
                if aa0 != Seq(aaf).translate(table=11):  # if amino acid changed
                    tally += 1  # then add one to the tally
            probability = tally / len(possible_muts)  # fraction of mutations that were nonsynonymous
        
        return probability

    def codon_mutation_table(allmuts, allcodons):
        table = np.zeros((len(allcodons), len(allmuts)))
        for i in range(len(allcodons)):
            for j in range(len(allmuts)):
                table[i, j] = prob_nonsyn_codon_mutation(allcodons[i], allmuts[j])
        return table
    
    def codons_in_genome(all_genes, all_codons, subset_gene_nums=None):
        # Get structure representing genes of interest
        if subset_gene_nums is not None:  # if genes of interest are provided
            # Gene numbers are the global tagnumbers, and do not restart between contigs.
            my_genes = all_genes[all_genes['tagnumber'].isin(subset_gene_nums)]
        else:  # otherwise use all coding regions
            # Remove tRNAs and rRNAs
            my_genes = all_genes[all_genes['tagnumber'] > 0] # doesn't consider tRNA and rRNA (=0)
        
        # Tally codon occurrences over all proteins in genome
        codon_counts = np.zeros(len(all_codons))  # for storing tally of codons
        for _,gene in my_genes.iterrows():
            next_sequence = np.char.upper(gene['sequence'])
            next_codon_counts = Counter(["".join(next_sequence[i:i+3]) for i in range(0, len(next_sequence), 3)])  # tally codons in sequence
            for codon, count in next_codon_counts.items():
                if codon in all_codons:
                    codon_index = all_codons.index(codon)  # find the index of the codon in the list of all codons
                    codon_counts[codon_index] += count  # update tally of codons
                        
        codon_counts /= np.sum(codon_counts) # Renormalize to probability of each codon over the whole genome
        return codon_counts
    
    # Define DNA bases and possible mutations
    allbases = ['A', 'T', 'G', 'C']
    allmuts = ['AT', 'AG', 'AC', 'TA', 'TG', 'TC', 'GA', 'GT', 'GC', 'CA', 'CT', 'CG']
    mut_types_names = ['AT/TA', 'AC/TG', 'AG/TC', 'GC/CG', 'GT/CA', 'GA/CT']
    
    # Define codons
    allcodons = [ # pretty non-ideal way to do this but apparently codon table is diff in matlab and python?!
    'AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA',
    'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC',
    'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG',
    'CGT', 'CTA', 'CTC', 'CTG', 'CTT', 'GAA', 'GAC', 'GAG', 'GAT',
    'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA',
    'GTC', 'GTG', 'GTT', 'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC',
    'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG',
    'TTT'
    ]
    # Generate table of codon composition by base
    codoncompositiontable = codon_composition_table(allbases, allcodons)
    # Generate table of probabilities of nonsynonymous mutations
    codonnonsyntable = codon_mutation_table(allmuts, allcodons)

    mutationalspectrum = np.zeros(len(allmuts))
    for i, mut in enumerate(allmuts):
        next_mut = mut[:2]  # Extract the first two characters
        next_index = next((i for i, mut_type in enumerate(mut_types_names) if next_mut in mut_type.split('/')), None)
        mutationalspectrum[i] = mut_spec_prob[next_index] / 2
        # Calculate codon distribution in reference genome
        
    if gene_nums_of_interest is not None:
        codondistribution = codons_in_genome(all_genes, allcodons, gene_nums_of_interest)
    else:
        codondistribution = codons_in_genome(all_genes, allcodons)
    
    # Calculate probability of nonsynonymous mutation
    probnonsyn = 0
    for thismutindex in range(len(allmuts)): # loop through all possible mutations
        thismutation = allmuts[thismutindex] # The mutation being considered
        probthismutation = mutationalspectrum[thismutindex] # Probability that this mutation occurs
        
        #Find the codons that can undergo this mutation
        thisbase = thismutation[0] # base that gets mutated; ex. A
        thisbaseindex = allbases.index(thisbase) # ex. A is indexed in position 0
        thisbasecodonoccurrences = codoncompositiontable[:, thisbaseindex] # ex. how many A's in each codon
        
        # Indices of codons that have the relevant initial base
        thismutcodons = np.where(thisbasecodonoccurrences > 0)[0] # ex. AAT has A's but GGC does not
        
        # Probability that this mutation occurs on a given relevant codon
        # Take into account base composition of codons
        probmutoncodonbases = thisbasecodonoccurrences[thismutcodons]
        # Take into account codon abundance on reference genome
        probcodonongenome = codondistribution[thismutcodons]
        # Combine these two probabilities
        probmutoncodon = probmutoncodonbases * probcodonongenome
        # Renormalize (sum = 1 over all relevant codons)
        probmutoncodon = probmutoncodon / np.sum(probmutoncodon)
        
        # Probability that this mutation is nonsynonymous at each relevant codon
        thismutnonsynoncodon = codonnonsyntable[:, thismutindex]
        probmutnonsynoncodon = thismutnonsynoncodon[thismutcodons]
        
        # Overall probability that this mutation is nonsynonymous over all possible codons
        probmutnonsyn = probthismutation * sum(probmutnonsynoncodon*probmutoncodon)
        
        # Add contribution of this mutation to the total probability of a nonsynonymous mutation (Bayesian prior)
        probnonsyn = probnonsyn + probmutnonsyn
        
    return probnonsyn


def div_matrix2_6types(matrix):
    """
    Converts a 4x4 mutation matrix to a 6-entry vector of mutation types.
    
    Parameters:
    matrix - 4x4 mutation matrix
    
    Returns:
    x - 6-entry vector of mutation types
    
    Mutation types:
    1: AT, TA
    2: AC, TG
    3: AG, TC
    4: GC, CG
    5: GT, CA
    6: GA, CT
    """
    
    # For 2D matrix (most common case in this context)
    x = np.zeros(6)
    
    # Compress into 6 elements
    x[0] = matrix[0, 1] + matrix[1, 0] # AT, TA
    x[1] = matrix[0, 2] + matrix[1, 3] # AC, TG
    x[2] = matrix[0, 3] + matrix[1, 2] # AG, TC
    x[3] = matrix[3, 2] + matrix[2, 3] # GC, CG
    x[4] = matrix[3, 1] + matrix[2, 0] # GT, CA
    x[5] = matrix[3, 0] + matrix[2, 1] # GA, CT
    
    return x


def mutation_spectrum_module(annotation_full, NTs):
    """
    Analyzes mutation spectrum from annotation data.
    
    Parameters:
    annotation_full - DF with columns: ancestral_allele, derived_allele, type, pos
    NTs - Array of nucleotides
    
    Returns:
    mutationmatrix - 4x4 matrix of mutation counts (ATCG -> ATCG)
    mut_observed - 6-entry vector of mutation types
    typecounts - 4-entry vector of mutation type counts (NSPI)
    prob_nonsyn - Probability of non-synonymous mutations
    """
    
    # Initialize vectors
    mutationmatrix = np.zeros((4, 4))  # ATCG -> ATCG matrix; gathered into mut_observed (6 entry vector)
    typecounts = np.zeros(4)  # NSPI - used to calculate prob_nonsyn
    
    # Count mutations
    for _,pos in annotation_full.iterrows():  # loop through positions at which there is a SNP
        anc = pos['ancestral_allele'] # ancestor NT at this position
        derived = [nt for nt in str(pos['derived_allele']).split(',') if nt in NTs] # NTs seen in samples

        # Check if ancestor is known
        if anc in NTs:
            # Convert to numbers
            anc_idx = int(np.where(NTs == anc)[0][0])

            # Find indices of the derived alleles in NTs (they already exclude the ancestor)
            new_indices = [int(np.where(NTs == nt)[0][0]) for nt in derived]
            # Remove N from new
            new_indices = [idx for idx in new_indices if idx != 4]

            if len(new_indices) == 0:  # if nothing left
                log.debug('Position %s is in the table but has no mutation to count', pos['genome_pos'])
            elif len(new_indices) == 1:  # if one mutation
                # Update mutation matrix
                mutationmatrix[anc_idx, new_indices[0]] += 1
                
                # Count type (use annotation_full)
                if pos['mutation_type'] == 'N':  # nonsyn mut
                    typecounts[0] += 1
                elif pos['mutation_type'] == 'S':  # syn mut
                    typecounts[1] += 1
                elif pos['mutation_type'] == 'P':  # promoter mut
                    typecounts[2] += 1
                elif pos['mutation_type'] == 'I':  # intergenic mut
                    typecounts[3] += 1
                elif pos['mutation_type'] == 'U':  # annotation could not infer an amino-acid change
                    log.debug('Position %s has no assigned mutation type and is not counted in dN/dS',
                              pos['genome_pos'])
                else:
                    log.warning('Skipping position %s: unrecognized mutation type %r',
                                pos['genome_pos'], pos['mutation_type'])
            elif len(new_indices) > 1:  # if more than one mutation
                log.debug('Position %s has more than one mutant allele and is not counted in dN/dS',
                          pos['genome_pos'])
                
                # Update mutation matrix
                for new_idx in new_indices:  # once for each mutation
                    mutationmatrix[anc_idx, new_idx] += 1
        else:
            log.debug('Position %s has no inferred ancestral allele and is not counted in dN/dS',
                      pos['genome_pos'])
    
    # Count six types of mutations
    mut_observed = div_matrix2_6types(mutationmatrix)  # tally of mutations, NOT normalised
    
    # Calculate fraction of nonsynonymous mutations (only N and S)
    if typecounts[0] + typecounts[1] > 0:
        prob_nonsyn = typecounts[0] / (typecounts[0] + typecounts[1])
    else:
        prob_nonsyn = 0
    
    return mutationmatrix, mut_observed, typecounts, prob_nonsyn


def binofit(x, n, alpha=0.05):
    """
    MATLAB-like binofit function that estimates binomial parameter and confidence interval.
    
    Parameters:
    x - Number of successes
    n - Number of trials
    alpha - Significance level (default 0.05 for 95% confidence interval)
    
    Returns:
    p - Maximum likelihood estimate of probability
    ci - Tuple containing the confidence interval
    """
    # Maximum likelihood estimate
    p = x / n if n > 0 else 0
    
    # Clopper-Pearson interval (equivalent to MATLAB's binofit)
    if n > 0:
        if x == 0:
            ci_low = 0
            ci_high = 1 - (alpha/2)**(1/n)
        elif x == n:
            ci_low = (alpha/2)**(1/n)
            ci_high = 1
        else:
            ci_low = stats.beta.ppf(alpha/2, x, n-x+1)
            ci_high = stats.beta.ppf(1-alpha/2, x+1, n-x)
        ci = (ci_low, ci_high)
    else:
        ci = (0, 0)
    
    return p, ci


def compute_observed_dnds(annotation_full, gene_nums_of_interest=None):
    """
    Computes the observed probability of a nonsynonymous mutation on a
    protein coding region based on all the mutations provided in annotation_full.
    
    Parameters:
    annotation_full - List of objects with attributes: gene_num_global, type
    gene_nums_of_interest - Optional list of gene numbers to consider
    
    Returns:
    p_nonsyn - Probability of nonsynonymous mutation
    CI_nonsyn - Confidence interval for p_nonsyn
    num_muts_N - Number of nonsynonymous mutations
    num_muts_S - Number of synonymous mutations
    """
    
    # Number of mutations
    num_muts_total = len(annotation_full)
    
    # Which mutations to consider
    muts_genenums = np.array(annotation_full['gene_num_global'])
    
    if gene_nums_of_interest is not None:  # consider specific genes if provided
        muts_filter = np.isin(muts_genenums, gene_nums_of_interest)
    else:  # otherwise consider all genes
        muts_filter = muts_genenums > 0
    
    # Tally number of nonsynonymous vs synonymous mutations
    muts_types = np.array(annotation_full['mutation_type'])
    num_muts_N = np.sum(muts_types[muts_filter] == 'N')
    num_muts_S = np.sum(muts_types[muts_filter] == 'S')
    
    # Compute probability of a nonsynonymous mutation with 95% CIs
    x = num_muts_N
    n = num_muts_N + num_muts_S
    alpha = 0.05
    
    # Use binofit function to calculate p_nonsyn and confidence interval
    p_nonsyn, CI_nonsyn = binofit(x, n, alpha)
    
    return p_nonsyn, CI_nonsyn, num_muts_N, num_muts_S


def parse_args():
    p = argparse.ArgumentParser(prog='AccuSNV dN/dS (stage 4)')
    p.add_argument('-r', '--rer', dest='ref_genome', type=str, required=True, help="Reference genome FASTA (or its directory)")
    p.add_argument('-i', '--input_dir', dest='input_dir', type=str, required=True, help="Group dir holding the SNV tables from stage 2")
    p.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True, help="Where to write the dN/dS results")
    p.add_argument('--group', default='group', help="Group name, for the log")
    accusnv_log.add_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    accusnv_log.setup('dnds', args.log)
    group = args.group
    # The GFF sits beside the FASTA; accept either the reference FASTA path or its directory.
    dir_ref_genome = ref_directory(args.ref_genome)
    final_tsv = args.input_dir + '/snv_table_final.tsv'
    # Nothing to compute when the calling stage took its fast path, or when it left no SNVs
    # (the table is then written empty). On the >100000-position fast path nothing is annotated
    # at all and snv_table_final.tsv holds the raw CNN scores instead, which have none of the
    # columns below, so the flag has to be checked rather than the table.
    fast_path = bool(np.load(args.input_dir + '/_snv_state.npz', allow_pickle=True)['fast_path'])
    if fast_path or not os.path.exists(final_tsv) or os.path.getsize(final_tsv) == 0:
        log.warning('Group %s: no annotated SNV table, so dN/dS cannot be computed. Writing an '
                    'empty result.', group)
        # Still create the declared output so the workflow rule completes; an all-zero record
        # marks "no dN/dS computed" without pretending a value exists.
        output_directory = args.output_dir + '/dNdS_out'
        os.makedirs(output_directory, exist_ok=True)
        np.savez_compressed(output_directory + '/data_dNdS.npz',
                            dNdS=np.nan, CI_lower=np.nan, CI_upper=np.nan,
                            num_muts_N=0, num_muts_S=0, p_nonsyn=np.nan, probnonsyn_expected=np.nan)
        return

    annotation_full = pd.read_csv(final_tsv, sep='\t')
    output_directory = args.output_dir + '/dNdS_out'
    os.makedirs(output_directory, exist_ok=True)

    # Recombinant SNVs are flagged but kept in the tables. They are not independent mutations,
    # so counting them would bias dN/dS; drop them here, as tree building and dMRCA already do.
    # With recombination detection off (--skip_recombination) nothing is ever flagged, so
    # nothing is dropped -- that is how a user opts into counting them.
    num_recomb = int((annotation_full['Whether_recomb'] == 1).sum())
    log.info('Group %s: computing genome-wide dN/dS from %s SNVs (%s recombinant SNVs left out)',
             group, f'{len(annotation_full) - num_recomb:,}', f'{num_recomb:,}')
    annotation_full = annotation_full[annotation_full['Whether_recomb'] == 0]

    # Parse the reference annotations once and reuse the frame for the genome-wide number and
    # every per-gene one below.
    all_genes = pd.concat(snv.parse_gff(dir_ref_genome,
                                        genomestats(dir_ref_genome)[2])).reset_index()

    # Mutation spectrum (tallied per mutant allele; only single-allele sites contribute a type)
    NTs = np.array(['A', 'T', 'C', 'G', 'N'])
    mutationmatrix, mut_observed, typecounts, prob_nonsyn = mutation_spectrum_module(annotation_full, NTs)
    num_muts_for_empirical_spect = 100
    if np.sum(mut_observed) >= num_muts_for_empirical_spect:
        mut_spec_prob = mut_observed / np.sum(mut_observed)
    else:
        mut_spec_prob = np.ones(mut_observed.size) / mut_observed.size
        log.warning('Group %s: only %d mutations observed, fewer than the %d needed to estimate the '
                    'mutation spectrum from the data, so a uniform spectrum is assumed. Treat dN/dS '
                    'as approximate.', group, int(np.sum(mut_observed)), num_muts_for_empirical_spect)

    # Expected vs observed N/S relative to a neutral model
    probnonsyn_expected = compute_expected_dnds(all_genes, mut_spec_prob)
    dnds_expected = probnonsyn_expected / (1 - probnonsyn_expected)
    p_nonsyn, CI_nonsyn, num_muts_N, num_muts_S = compute_observed_dnds(annotation_full, gene_nums_of_interest=None)
    dnds_observed = p_nonsyn / (1 - p_nonsyn)
    dNdS = dnds_observed / dnds_expected

    CI_lower = (CI_nonsyn[0] / (1 - CI_nonsyn[0])) / dnds_expected
    try:
        CI_upper = (CI_nonsyn[1] / (1 - CI_nonsyn[1])) / dnds_expected
    except ZeroDivisionError:
        CI_upper = np.inf

    log.info('Group %s: dN/dS = %.4f (95%% CI %.4f-%.4f) from %d nonsynonymous and %d synonymous '
             'mutations; a neutral genome of this composition would give %.4f nonsynonymous',
             group, dNdS, CI_lower, CI_upper, num_muts_N, num_muts_S, probnonsyn_expected)
    output_dict = {
        'dNdS': dNdS, 'CI_lower': CI_lower, 'CI_upper': CI_upper,
        'num_muts_N': num_muts_N, 'num_muts_S': num_muts_S,
        'p_nonsyn': p_nonsyn, 'probnonsyn_expected': probnonsyn_expected,
    }
    np.savez_compressed(output_directory + '/data_dNdS.npz', **output_dict)
    pd.DataFrame([output_dict]).to_csv(output_directory + '/dnds_genomewide.tsv', sep='\t', index=False)

    # Per-gene breakdown, over the genes carrying at least one SNV. A gene with only
    # nonsynonymous mutations has p_nonsyn == 1 and so an infinite dN/dS; with a handful of
    # mutations per gene these ratios are noisy, and are meant for eyeballing, not testing.
    odds = lambda p: p / (1 - p) if p < 1 else np.inf
    per_gene = []
    # tRNA/rRNA genes are numbered 0, and an intergenic position sometimes still carries a
    # neighbouring gene's number, so select on the annotation instead: a row with a locus_tag
    # sits in a gene. The gene start/stop arrive as text because intergenic rows write them as '.'.
    genic = annotation_full[(annotation_full['gene_num_global'] >= 1) & (annotation_full['locus_tag'] != '.')]
    for gene_num, rows in genic.groupby('gene_num_global'):
        gene_num = int(gene_num)
        p_gene, _, gene_muts_N, gene_muts_S = compute_observed_dnds(annotation_full, [gene_num])
        expected_gene = compute_expected_dnds(all_genes, mut_spec_prob, [gene_num])
        first = rows.iloc[0]
        per_gene.append({
            'gene_num_global': gene_num, 'locus_tag': first['locus_tag'], 'product': first['product'],
            'gene_length': int(float(first['gene_contig_stop_pos']) - float(first['gene_contig_start_pos']) + 1),
            'num_muts_N': gene_muts_N, 'num_muts_S': gene_muts_S,
            'p_nonsyn': p_gene, 'probnonsyn_expected': expected_gene,
            'dNdS': odds(p_gene) / odds(expected_gene),
        })
    pd.DataFrame(per_gene).to_csv(output_directory + '/dnds_per_gene.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
