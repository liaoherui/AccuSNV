"""Stage 2: annotate candidate mutations against the reference, merge the annotations with
the CNN+filter table into the unfiltered table, and split those by CNN label into the final
tables. Includes the mutation-annotation library functions inline.

Reads ``_snv_state.npz`` and ``snv_table_filtered_tmp.tsv`` (both from stage 1) and the
reference dir; writes ``snv_table_unfiltered.tsv`` and ``snv_table_final.tsv``.

The tmp table is kept rather than deleted, so that this stage can be run again on an existing
output directory (``--downstream_only``). It is not a declared input of the Snakemake rule,
because the >100k fast path does not produce one.

Transcribed from new_snv_script.py lines ~1104-1164, 1304-1338, and ~1365-1416.
"""
import os
import re
import shutil
import logging
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
from Bio.Seq import Seq

from accusnv import log as accusnv_log
from accusnv.downstream import snv

log = logging.getLogger('accusnv')

# The columns that lead the final table, ahead of the CNN and filter verdicts.
LEADING_COLUMNS = ['genome_pos', 'contig', 'contig_pos',
                   'ancestral_allele', 'derived_allele', 'reference_allele',
                   'gene_nucleotide_mutation', 'gene_nt_position', 'aa_mutation', 'mutation_type',
                   'protein_id', 'locus_tag', 'product']


def annotate_mutations( my_rg , p_gp , ancnti_gp , calls_gp , my_cmt_gp , fixedmutation_gp , mutQual_gp, promotersize ):
    '''
    Makes a dataframe with annotations for each SNV position. 
        
    ARGUMENTS
    ---------
    
        my_rg: reference genome object
        
        p_gp: SNV positions on genome (should already be filtered)
        
        ancnti_gp: ancestral nucleotide at SNV positions
        
        calls_gp: bascalls across samples for each SNV position
        
        my_cmt_gp: candidate mutation table data object across SNV positions
        
        fixedmutation_gp: whether or not each sample has a mutation across SNV
        positions
        
        mutQual_gp: mutation quality (from compute_mutation_quality) for each
        SNV position
        
        promoter_size: size of promoter region in bp
                
    RETURNS
    -------

        dataframe_mut: pandas dataframe with information on annotations for 
        each SNV
        
    NOTES
    -----
    
        1. Version history: This function was based on MATLAB 
        annotate_mutations_gb and append_annotations.
    
    '''       

    # Get information from candidate mutation table object    
    maNT_gp = my_cmt_gp.major_nt
    minorNT_gp = my_cmt_gp.minor_nt
    
    # Get info from reference genome object
    annotation_genes = my_rg.annotations
    locus_tags = my_rg.locus_tagnumbers
    cds_tags = my_rg.cds_indices

    # Get info at mutation positions
    contigpos_gp = my_rg.p2contigpos( p_gp )  
    mut_locus_tags = locus_tags[p_gp-1]
    mut_cds_indices = cds_tags[p_gp-1]
    mut_ref_nts = my_rg.get_ref_NTs( np.asarray(p_gp) ) # the reference base at each SNV position

    # Initialize list of dictionaries, where each entry will correspond to one SNV position
    lod_mutAnno = [] 
    
    # Loop through SNV positions
    for i,pos in enumerate(p_gp):

        mut_annotations = {} # initialize dictionary for this SNV
        
        # Get information about SNV position on genome
        mut_annotations['p'] = pos
        mut_annotations['contig'] = my_rg.contig_names[contigpos_gp[i][0]-1] # contigs are indexed starting at 1
        mut_annotations['contig_pos'] = contigpos_gp[i][1]
        mut_annotations['quals'] = mutQual_gp[i]
        mut_annotations['reference_allele'] = mut_ref_nts[i]
        mut_annotations['gene_num_global'] = mut_locus_tags[i] # from locus_tagnumbers
        mut_annotations['gene_num'] = mut_cds_indices[i] # from cds_indices

        # Get info on annotations at this SNV position
        contig_idx = contigpos_gp[i][0]
        if mut_cds_indices[i] == int(mut_cds_indices[i]): # Intragenic
        
            # Get annotations from row in dataframe 
            p_anno = annotation_genes[contig_idx-1].iloc[int(mut_cds_indices[i])-1] # first -1 bcs contigs indexed starting at 1; second -1 bcs cds_indices indexed at 1 but dataframe rows indexed at zero
            if re.search('gene',p_anno.loc['locustag']):
                p_anno.loc['locustag']=re.sub('gene','pseudogene',p_anno.loc['locustag'])
            mut_annotations['locus_tag'] = p_anno.loc['locustag']
            mut_annotations['product'] = p_anno.loc['product']
            mut_annotations['protein_id'] = p_anno.loc['protein_id']
            mut_annotations['gene'] = p_anno.loc['gene']
            mut_annotations['ontology'] = p_anno.loc['ontology_term']
            mut_annotations['note'] = p_anno.loc['note']
            mut_annotations['strand'] = p_anno.loc['strand']
            mut_annotations['gene_contig_start_pos'] = p_anno.loc['loc1'] # leftmost position of gene on the contig
            mut_annotations['gene_contig_stop_pos'] = p_anno.loc['loc2'] # rightmost position of gene on the contig (inclusive)
            mut_annotations['sequence'] = p_anno.loc['sequence']
            mut_annotations['translation'] = p_anno.loc['translation']
            
            # Where the SNV sits in the gene sequence as reported in the 'sequence' column
            # (reverse-complemented for a -1 strand gene), counting from 1. A forward gene
            # is read from loc1, a reverse gene from loc2, so they count from opposite ends.
            if p_anno.loc['strand'] == 1: # fwd
                mut_annotations['gene_nt_position'] = mut_annotations['contig_pos'] - mut_annotations['gene_contig_start_pos'] + 1
            elif p_anno.loc['strand'] == -1: # rev
                mut_annotations['gene_nt_position'] = mut_annotations['gene_contig_stop_pos'] - mut_annotations['contig_pos'] + 1
            else: # no strandedness
                mut_annotations['gene_nt_position'] = "." # can happen eg. Crispr
            if mut_annotations['gene_nt_position'] == ".": # observed with special 'type's like crispr annotated (rare!); leads to no AA inference
                aan = 9999999
            else:
                aan = int( (mut_annotations['gene_nt_position']-1 )/3 ) + 1 # get codon that harbours mutation (codons indexed starting at 1)
                ncn = mut_annotations['gene_nt_position'] - ((aan-1)*3) # get nucleotide within codon triplet (1, 2, or 3)
                mut_annotations['aa_pos'] = aan
            
            # Determine how a mutation *could* change the codon
            codons_ls = [] # init
            aa_ls = [] # init
            if len(mut_annotations['sequence']) >= (aan*3) and mut_annotations['translation'] != ".": # only for proteins
                codon = mut_annotations['sequence'][aan*3-2-1:aan*3] # -1 bcs sequence is indexed at 0 but positional argument aan is not
                codon = [ n for n in codon ] # turn seq.object to list (bcs seq object does not allow reassignment)
                for nt in snv.NTs_list_without_N: # test all four nucleotide options for resulting AA change
                    if p_anno.loc['strand'] == 1:
                        codon[ncn-1] = nt
                    else:
                        codon[ncn-1] = snv.NTs_complement_dict[nt]
                    codonSeqO = Seq( "".join(codon))
                    codons_ls.append(codonSeqO)
                    aa_ls.append(codonSeqO.translate())
            mut_annotations['codons'] = codons_ls
            mut_annotations['AA'] = [aa[0] for aa in aa_ls] # turn amino acids into string
            mut_annotations['NonSyn'] = len(set(mut_annotations['AA']))>1 # indicates if a nonsynonymous mutation is *possible*, not necessarily if it happened
            
            # AA[] is indexed by the genome-forward nucleotide on both strands: the loop above
            # already complemented each candidate base before placing it in a reverse-strand
            # codon, so AA[i] is the amino acid produced by forward base NTs_list_without_N[i].
            # Complementing again here would be a second complement, which is where the
            # original AccuSNV reports an ancestral AA that contradicts its own translation.
            nts_for_aa_idx_dict = {nt: nt for nt in snv.NTs_list_without_N}

            # Determine if any observed mutations *did* change the codon
            mut_annotations['AA_gt'] = '' # init # for filling in amino acids corresponding to observed mutations relative to the ancestor
            if len(mut_annotations['AA']) < 4:
                mut_annotations['type'] = 'U' # unknown mutation type
            # Fill in ancestral allele
            anc_is_known = (np.unique(ancnti_gp[:,i])[0] != snv.NTs_to_int_dict['N'])
            if anc_is_known: # only if ancestral allele is known; [0] ok bcs ancnti_gp should by definition never have > 1 unique element per row
                mut_annotations['anc'] = snv.int_to_NTs_dict[np.unique(ancnti_gp[:,i])[0]]
                mut_annotations['nts'] = mut_annotations['anc'] # add ancestral allele to observed NTs; will add other NTs later
                if len(mut_annotations['AA']) == 4:
                    anc_nt = snv.int_to_NTs_dict[np.unique(ancnti_gp[:,i])[0]]
                    aa_lookup_nt = nts_for_aa_idx_dict[anc_nt]
                    mut_annotations['AA_gt'] = mut_annotations['AA_gt'] + mut_annotations['AA'][ snv.NTs_list_without_N_to_idx_dict[aa_lookup_nt] ] # the AA list order corresponds to NTs list after strand-aware mapping
            else: # ancestral allele not known, so cannot evaluate if mutations changed the codon
                mut_annotations['nts'] = "."
                mut_annotations['anc'] = "."
            # Fill in derived genotype(s) across all samples
            for j,callidx in enumerate(calls_gp[:,i]): # loop across samples
                if fixedmutation_gp[j,i] == True : # fixed mutations only
                    if calls_gp[j,i] == -1: # if diverse (calls not a mutation), add minor and major call
                        ma_nt = snv.int_to_NTs_dict[maNT_gp[j,i]]
                        minor_nt = snv.int_to_NTs_dict[minorNT_gp[j,i]]
                        mut_annotations['nts'] = mut_annotations['nts'] + ma_nt + minor_nt
                        if len(mut_annotations['AA']) == 4:
                            ma_aa_lookup_nt = nts_for_aa_idx_dict[ma_nt]
                            minor_aa_lookup_nt = nts_for_aa_idx_dict[minor_nt]
                            mut_annotations['AA_gt'] = mut_annotations['AA_gt'] + mut_annotations['AA'][ snv.NTs_list_without_N_to_idx_dict[ma_aa_lookup_nt] ]
                            mut_annotations['AA_gt'] = mut_annotations['AA_gt'] + mut_annotations['AA'][ snv.NTs_list_without_N_to_idx_dict[minor_aa_lookup_nt] ]
                    elif calls_gp[j,i] != snv.NTs_to_int_dict['N']:
                        mut_annotations['nts'] = mut_annotations['nts'] + snv.int_to_NTs_dict[calls_gp[j,i]]
                        if len(mut_annotations['AA']) == 4:
                            called_nt = snv.int_to_NTs_dict[calls_gp[j,i]]
                            aa_lookup_nt = nts_for_aa_idx_dict[called_nt]
                            mut_annotations['AA_gt'] = mut_annotations['AA_gt'] + mut_annotations['AA'][ snv.NTs_list_without_N_to_idx_dict[aa_lookup_nt] ]
            if len(mut_annotations['AA']) == 4:
                mut_annotations['type'] = 'S' # initialize as S; eventually overwritten below if N
            # Remove duplicates
            mut_annotations['nts'] = ''.join(OrderedDict.fromkeys( mut_annotations['nts'] ).keys()) # get only unique nucleotides and keep order
            mut_annotations['AA_gt'] = ''.join(OrderedDict.fromkeys( mut_annotations['AA_gt'] ).keys()) # get only unique AAs and keep order
            # Record mutation type and amino-acid changes.
            # If ancestral allele is unknown, keep this mutation as unknown.
            if not anc_is_known:
                mut_annotations['type'] = 'U'
                mut_annotations['muts'] = "."
            elif len(mut_annotations['AA_gt'])>1:
                mut_annotations['type'] = 'N'
                mut_annotations['muts'] = []
                ancAA = mut_annotations['AA_gt'][0]
                derAAs = mut_annotations['AA_gt'][1:]
                for j,derAA in enumerate(derAAs):
                    mut_annotations['muts'].append( ancAA+str(mut_annotations['aa_pos'])+derAA )
            else:
                mut_annotations['muts'] = "."
    
        else: # Intergenic
        
            # Get info for gene prior to SNP (if any)
            if int(mut_cds_indices[i])>0: 
                p_anno = annotation_genes[contig_idx-1].iloc[int(mut_cds_indices[i])-1] # first -1 bcs contigs indexed starting at 1; second -1 bcs cds_indices indexed at 1 but dataframe rows indexed at zero
                mut_annotations['gene1'] = p_anno.loc['gene']
                mut_annotations['locustag1'] = p_anno.loc['locustag']
                mut_annotations['product1'] = p_anno.loc['product']
                mut_annotations['distance1'] = mut_annotations['contig_pos'] - p_anno.loc['loc2'] # how far back the previous gene is
                if p_anno.loc['strand'] == -1:
                    mut_annotations['distance1'] = mut_annotations['distance1'] * -1
            
            # Get info for gene after SNP (if any)
            if int(mut_cds_indices[i]+0.5) <= annotation_genes[contig_idx-1].shape[0] and annotation_genes[contig_idx-1].shape[1] != 0: 
                p_anno = annotation_genes[contig_idx-1].iloc[int(mut_cds_indices[i])] # first -1 bcs contigs indexed starting at 1; second -1 bcs cds_indices indexed at 1 but dataframe rows indexed at zero
                mut_annotations['gene2'] = p_anno.loc['gene']
                mut_annotations['locustag2'] = p_anno.loc['locustag']
                mut_annotations['product2'] = p_anno.loc['product']
                mut_annotations['distance2'] = p_anno.loc['loc1'] - mut_annotations['contig_pos'] # how far ahead the next gene is
                if p_anno.loc['strand'] == 1: # second conditional to evade empty chr (?????)
                    mut_annotations['distance2'] = mut_annotations['distance2'] * -1
            
            # Determine mutation type
            if ( 'distance1' in mut_annotations and mut_annotations['distance1'] > (-1*promotersize) and mut_annotations['distance1'] < 0) or ( 'distance2' in mut_annotations and mut_annotations['distance2'] > (-1*promotersize) and mut_annotations['distance2'] < 0):
                mut_annotations['type'] = 'P'
            else:
                mut_annotations['type'] = 'I'
            
            # Get ancestral allele (repeat of intragenic code)
            if np.unique(ancnti_gp[:,i])[0] != snv.NTs_to_int_dict['N']: # only if ancestral allele is known; [0] ok bcs ancnti_gp should by definition never have > 1 unique element per row
                mut_annotations['anc'] = snv.int_to_NTs_dict[np.unique(ancnti_gp[:,i])[0]] 
                mut_annotations['nts'] = mut_annotations['anc']
            else:
                mut_annotations['nts'] = "."
                mut_annotations['anc'] = "."
            # Extract derived genotype(s) across all samples
            for j,callidx in enumerate(calls_gp[:,i]): # loop across samples 
                if calls_gp[j,i] != snv.NTs_to_int_dict['N']:
                    mut_annotations['nts'] = mut_annotations['nts'] + snv.int_to_NTs_dict[calls_gp[j,i]]
            # Remove duplicates
            mut_annotations['nts'] = ''.join(OrderedDict.fromkeys( mut_annotations['nts'] ).keys()) # get only unique nucleotides and keep order
            
        # Split 'nts' (ancestor first, then every allele seen in a sample) into its own columns,
        # and write the same change along the gene, e.g. A120T, complemented on a -1 strand gene.
        anc = mut_annotations['anc']
        derived = [nt for nt in mut_annotations['nts'] if nt != anc and nt in snv.NTs_list_without_N]
        mut_annotations['ancestral_allele'] = anc
        mut_annotations['derived_allele'] = ','.join(derived) if derived else '.'
        if derived and anc != '.' and mut_annotations.get('gene_nt_position', '.') != '.':
            flip = (lambda nt: nt) if mut_annotations['strand'] == 1 else snv.NTs_complement_dict.get
            gene_pos = int(mut_annotations['gene_nt_position'])
            mut_annotations['gene_nucleotide_mutation'] = ','.join(
                flip(anc) + str(gene_pos) + flip(nt) for nt in derived)

        # Append dictionary for this SNV to list of dictionaries
        lod_mutAnno.append(mut_annotations)
    
    # Turn list of dictionaries into a pandas dataframe
    dataframe_mut = pd.DataFrame(lod_mutAnno)
    
    return dataframe_mut


def build_mutation_table( mut_positions, mut_quality, annotation_mutations, calls_for_tree, names_for_tree ):
    '''
    Builds the annotated SNV table (all columns as strings, for merge_two_tables).

    Annotations that do not apply to a position (e.g. codons at an intergenic SNV) become
    '.'. The codons, AA and muts columns hold Python lists, which are flattened to strings
    here.
    '''

    # annotate_mutations leaves out a column entirely when no SNV of that kind occurred
    # (an all-intergenic run has no codons), so reindex to guarantee every column exists.
    anno = annotation_mutations.reindex(columns=[
        'contig', 'contig_pos', 'gene_num', 'gene_num_global', 'product', 'protein_id',
        'ontology', 'locus_tag', 'strand', 'gene_contig_start_pos', 'gene_contig_stop_pos',
        'gene_nt_position', 'gene_nucleotide_mutation', 'aa_pos', 'codons', 'AA',
        'ancestral_allele', 'derived_allele', 'reference_allele', 'muts', 'type',
        'sequence', 'translation'])

    def flatten(column, separator, default):
        '''Join a column of lists into a column of strings, one string per position.'''
        return [separator.join(str(item) for item in value) if type(value) == list else default
                for value in anno[column]]

    def whole(column):
        '''Write whole numbers as 120, not 120.0 (a missing value makes the column a float).'''
        return [str(int(float(value))) if str(value) not in ('.', 'nan', '') else '.'
                for value in anno[column]]

    table = pd.DataFrame({
        'genome_pos': mut_positions,
        'contig': anno['contig'],
        'contig_pos': anno['contig_pos'],
        'ancestral_allele': anno['ancestral_allele'],
        'derived_allele': anno['derived_allele'],
        'reference_allele': anno['reference_allele'],
        'gene_nucleotide_mutation': anno['gene_nucleotide_mutation'],
        'gene_nt_position': whole('gene_nt_position'),
        'aa_mutation': flatten('muts', ',', '.'),
        'mutation_type': anno['type'],
        'protein_id': [value[0] if type(value) == list else value for value in anno['protein_id']],
        'locus_tag': anno['locus_tag'],
        'product': anno['product'],
        'gene_num_global': anno['gene_num_global'],
        'gene_num_contig': anno['gene_num'],
        'quality': mut_quality,
        'ontology': anno['ontology'],
        'strand': anno['strand'],
        'gene_contig_start_pos': whole('gene_contig_start_pos'),
        'gene_contig_stop_pos': whole('gene_contig_stop_pos'),
        'aa_pos': whole('aa_pos'),
        'possible_codons_at_site': flatten('codons', ' ', ''),
        'possible_AAs_at_site': flatten('AA', ' ', ''),
        **{name: calls_for_tree[j] for j, name in enumerate(names_for_tree)},
        'sequence': anno['sequence'],
        'translation': anno['translation'],
    })
    # Everything is stringified here, and the CNN table is read as text below, so that the
    # merge cannot reformat values (a missing value would otherwise turn an integer column
    # into floats, and 5 would be written back out as 5.0).
    return table.fillna('.').astype(str)


def merge_two_tables(in_cnn_table, annotations, out_merge_tsv, exclude_positions=(),
                     include_positions=()):
    '''
    Merges the CNN+filter table with an annotation table on genome_pos. Annotated positions
    come first, then any positions only the CNN table has. Excluded positions are dropped here
    as well as from the SNV set, so that they leave the tables entirely; included ones have
    their verdict overridden to 1, which is what carries them into the final table. The other
    columns are left alone, so the table still shows what the model and the filters decided.
    '''
    cnn = pd.read_csv(in_cnn_table, sep='\t', dtype=str)
    cnn = cnn[~cnn['genome_pos'].astype(int).isin(exclude_positions)]
    cnn.loc[cnn['genome_pos'].astype(int).isin(include_positions), 'Pred_label'] = '1'
    merged = annotations.merge(cnn, on='genome_pos', how='outer')

    # A position --include_positions forced in that the calling stage never scored (invariant
    # across the ingroup and rejected by both the model and the filters) has no row in the CNN
    # table, so the merge leaves its verdicts empty. These are the two the later stages read as
    # numbers: the user kept it, and nothing flagged it as recombinant.
    merged['Pred_label'] = merged['Pred_label'].fillna('1')
    merged['Whether_recomb'] = merged['Whether_recomb'].fillna('0')

    # merge does not preserve the row order of either table, so impose it here.
    order = annotations['genome_pos'].tolist()
    order += [pos for pos in cnn['genome_pos'] if pos not in set(order)]
    merged = merged.set_index('genome_pos').loc[order].reset_index()

    # Where the SNV is, what gene it hits and what changed all lead the table; then the CNN
    # and filter verdicts; then the rest of the annotation detail.
    # Positions the CNN table has but the annotations do not leave the annotation columns
    # empty. Do not fill them with '.': downstream/dnds.py reads gene_num_global from this file
    # and compares it numerically, which a '.' would turn into a string comparison and crash.
    leading = [column for column in LEADING_COLUMNS if column in merged.columns]
    columns = leading + [column for column in cnn.columns if column not in leading] \
                      + [column for column in annotations.columns if column not in leading]
    merged[columns].to_csv(out_merge_tsv, sep='\t', index=False)


def write_final_table(dir_output):
    '''
    Writes the final table: the draft rows the CNN accepted (Pred_label != 0).

    The draft is absent whenever there was nothing to annotate (the fast path, or a group
    with no SNVs); the final table is then written empty, so it always exists.
    '''
    draft_path = dir_output + '/snv_table_unfiltered.tsv'
    final_path = dir_output + '/snv_table_final.tsv'
    if not os.path.exists(draft_path) or os.path.getsize(draft_path) == 0:
        open(final_path, 'w').close()
        return
    draft = pd.read_csv(draft_path, sep='\t', dtype=str)
    final = draft[draft['Pred_label'].astype(int) != 0]
    final.to_csv(final_path, sep='\t', index=False)
    log.info('snv_table_unfiltered.tsv holds all %s candidate positions with the stage that '
             'removed each one in Removed_by; snv_table_final.tsv holds the %s that passed',
             f'{len(draft):,}', f'{len(final):,}')


def annotate_main(args):
    group = args.group
    dir_output = args.output_dir
    state_path = dir_output + '/_snv_state.npz'
    state = np.load(state_path, allow_pickle=True)
    if bool(state['fast_path']):
        log.warning('Group %s: this run took the fast path (or found no SNVs), so there is nothing '
                    'to annotate. Writing the tables empty.', group)
        # The rule declares snv_table_unfiltered.tsv as an output, so it must exist even when there
        # is nothing to annotate (the >100k fast path, or a run with no candidate positions).
        open(dir_output + '/snv_table_unfiltered.tsv', 'w').close()
        if bool(state['too_many_positions']):
            # The CNN scores are everything this run computed, so they stand in as the final
            # table and reach the user via the copy the rule makes. Note the columns are the
            # cnn_raw ones (genome_pos, CNN_pred, CNN_prob), not the usual annotated set.
            shutil.copy(dir_output + '/snv_table_cnn_raw.tsv', dir_output + '/snv_table_final.tsv')
        else:
            write_final_table(dir_output)
        return

    # The filter verdicts to merge in come from the calling stage. Output directories written
    # before AccuSNV 1.1 deleted this file once it had been merged, so they cannot be re-annotated.
    filter_table = dir_output + '/snv_table_filtered_tmp.tsv'
    if not os.path.exists(filter_table):
        log.error('Group %s: %s is missing, so the CNN and filter verdicts cannot be merged in. '
                  'Re-run the calling stage for this group.', group, filter_table)
        raise SystemExit(1)

    excluded = snv.read_positions_file(args.exclude_positions)
    included = snv.read_positions_file(args.include_positions)
    st = snv.rebuild_state(state_path, args.ref_genome, excluded, included)

    # The CNN can accept a candidate table yet leave zero good positions (e.g. a single
    # candidate that then filters out). There is nothing to annotate, and the ancestral-allele
    # step rejects an empty position array, so write empty tables and stop.
    if not np.any(st.goodpos_bool_all):
        log.warning('Group %s: no SNV positions survived filtering, so there is nothing to annotate. '
                    'Writing the tables empty.', group)
        open(dir_output + '/snv_table_unfiltered.tsv', 'w').close()
        write_final_table(dir_output)
        return

    log.info('Group %s: annotating %s SNVs against the reference genes (a SNV up to %d bp upstream '
             'of a gene is reported as promoter)', group, f'{int(np.sum(st.goodpos_bool_all)):,}', 250)

    # ---- Annotate mutations (rebuild calls with looser, downstream-style thresholds) ----
    promotersize = 250
    my_cmt = st.my_cmt.copy()
    my_cmt.filter_positions(st.goodpos_bool_all)
    my_calls = snv.calls_object(my_cmt)
    my_calls.filter_calls_by_element(my_cmt.coverage < 1)
    my_calls.filter_calls_by_element(my_cmt.quals < 30)
    my_calls.filter_calls_by_element(my_cmt.major_nt_freq < args.maf)
    my_calls.filter_calls_by_element(my_cmt.fwd_cov < 1)
    my_calls.filter_calls_by_element(my_cmt.rev_cov < 1)

    calls_ancestral = snv.infer_ancestral_calls_from_raw_overlap(
        st.my_calls_raw_for_ancestor, st.my_rg, my_cmt.p)[0]

    calls_ingroup = my_calls.get_calls_in_sample_subset(np.logical_not(my_calls.in_outgroup))
    quals_ingroup = my_cmt.quals[np.logical_not(my_calls.in_outgroup), :]
    num_samples_ingroup = sum(np.logical_not(my_calls.in_outgroup))
    mut_qual = snv.compute_mutation_quality(calls_ingroup, quals_ingroup)[0]

    filter_SNVs_not_N = (calls_ingroup != snv.nts2ints('N'))
    filter_SNVs_not_ancestral_allele = (calls_ingroup != np.tile(calls_ancestral, (num_samples_ingroup, 1)))
    fixedmutation = filter_SNVs_not_N & filter_SNVs_not_ancestral_allele

    my_cmt_ingroup = my_cmt.copy()
    my_cmt_ingroup.filter_samples(np.logical_not(my_cmt_ingroup.in_outgroup))

    mutations_annotated = annotate_mutations(
        st.my_rg, st.p_goodpos_all,
        np.tile(calls_ancestral, (my_cmt_ingroup.num_samples, 1)),
        calls_ingroup, my_cmt_ingroup, fixedmutation,
        mut_qual.flatten(), promotersize)

    # ---- Write the SNV annotation tables ----
    # The sample columns carry the unfiltered nucleotide calls: a call that failed the
    # per-call filters still shows the base that was observed, not an N.
    annotations = build_mutation_table(st.p_goodpos_all, st.mut_qual[0, st.goodpos_idx_all],
                                       mutations_annotated,
                                       st.calls_for_tree_raw, st.treesampleNamesLong)
    merge_two_tables(filter_table, annotations, dir_output + '/snv_table_unfiltered.tsv',
                     excluded, included)

    write_final_table(dir_output)

    # What kind of mutations these are is the headline result of the whole run.
    kinds = mutations_annotated['type'].value_counts() if 'type' in mutations_annotated else {}
    named = {'N': 'nonsynonymous', 'S': 'synonymous', 'P': 'promoter', 'I': 'intergenic', 'U': 'undefined'}
    log.info('Group %s: annotation finished -- %s', group,
             ', '.join(f'{int(n):,} {named.get(k, k)}' for k, n in kinds.items()) or 'no mutation types assigned')


def parse_args():
    p = argparse.ArgumentParser(prog='AccuSNV annotate (stage 2)')
    p.add_argument('-r', '--rer', dest='ref_genome', type=str, required=True, help="Reference genome FASTA (or its directory)")
    p.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True, help="Group output dir")
    p.add_argument('--maf', dest='maf', type=float, default=0.75,
                   help="Major-allele frequency cutoff when rebuilding calls for annotation (default 0.75).")
    p.add_argument('--exclude_positions', default='',
                   help="File of genome_pos values, one per line, to leave out of the SNV set.")
    p.add_argument('--include_positions', default='',
                   help="File of genome_pos values, one per line, to keep whatever the filters said.")
    p.add_argument('--group', default='group', help="Group name, for the log")
    accusnv_log.add_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    accusnv_log.setup('annotate', args.log)
    annotate_main(args)


if __name__ == '__main__':
    main()
