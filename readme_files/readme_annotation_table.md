# Explaination to two core output tables

## 1. Final SNV report table

`snv_table_final.tsv` is the main final SNV report table from the Snakemake pipeline. A copy is
written at the top level of the output directory as `group_<group>_snv_table_final.tsv`.
It is produced by merging:

- CNN + filter summary table, and
- mutation annotation table,

then keeping final positive calls in the final file. The same merge with every candidate position
kept, including the rejected ones, is written as `snv_table_unfiltered.tsv`.

It looks like this (the `sequence` and `translation` columns are truncated here for readability,
and the third row is an intergenic position):


```
genome_pos	contig	contig_pos	ancestral_allele	derived_allele	reference_allele	gene_nucleotide_mutation	gene_nt_position	aa_mutation	mutation_type	protein_id	locus_tag	product	Pred_label	CNN_pred	WideVariant_pred	CNN_prob	Qual_filter	Cov_filter	MAF_filter	Indel_filter	MFAS_filter	MMCP_filter	CPN_filter	Fix_filter	Whether_recomb	Fraction_ambiguous_samples	Gap_filter	gene_num_global	gene_num_contig	quality	ontology	strand	gene_contig_start_pos	gene_contig_stop_pos	aa_pos	possible_codons_at_site	possible_AAs_at_site	strain1	strain2	strain3	strain4	sequence	translation
11476	NC_018707.1	11476	G	T	G	C692A	692	S231*	N	WP_041446291.1	cds-WP_041446291.1	ATP-binding cassette domain-containing protein	1	1	1	0.99997425	0	0	0	0	0	0	0	0	0	0.000000	0	32.0	14.0	54.0	.	-1.0	10263	12167	231	TTA TAA TGA TCA	L * * S	G	G	G	T	ATGTGCATGGACTGCTCCGG...(truncated)	MCMDCSGLGYVDGIDLQELI...(truncated)
257244	NC_018707.1	257244	A	C	A	T1074G	1074	.	S	WP_002514990.1	cds-WP_002514990.1	phosphoglucomutase (alpha-D-glucose-1,6-bisphosphate-dependent)	1	1	0	0.9998721	0	1	-1	-1	-1	-1	-1	-1	0	0.000000	0	273.0	265.0	.	['GO:0005975', 'GO:0004614']	-1.0	256686	258317	358	GCT GCA GCG GCC	A A A A	A	A	C	A	ATGGCTCATGAACGTGCTGG...(truncated)	MAHERAGKPAQESDLIDVDA...(truncated)
321762	NC_018707.1	321762	C	T	C	.	.	.	P	.	.	.	1	1	1	0.9999999	0	0	0	0	0	0	0	0	0	0.000000	0	0.5	321.5	63.0	.	.	.	.	.			C	C	T	C	.	.
......
```

### Column meaning

(1) Position and gene columns (these lead the table)

- `genome_pos`: Identified SNV position to the reference genome, counting straight through all contigs.
- `contig`, `contig_pos`: name of the reference contig and the position on that contig.

(2) Allele and mutation columns

All three allele columns are reported in genome-forward orientation, so they can be compared with each other directly.

- `ancestral_allele`: the inferred ancestral base. With outgroup samples this comes from the ingroup/outgroup allele overlap; without them it is the majority base across the ingroup samples, which need not match the reference.
- `derived_allele`: every other base seen in a sample at this position, comma-separated when more than one.
- `reference_allele`: the base the reference genome carries here.
- `gene_nucleotide_mutation`: the same change written along the gene, e.g. `G256T`, ancestral base first and derived base last. On a reverse-strand gene both bases are complemented, so this column matches the `sequence` column rather than the three columns above. Comma-separated when there is more than one derived allele. `.` when the SNV is intergenic.
- `gene_nt_position`: 1-based position of the changed base within the `sequence` column. Use `sequence[gene_nt_position - 1]` to index it, on either strand. This is the number that appears inside `gene_nucleotide_mutation`.
- `aa_mutation`: the observed amino-acid change or changes, e.g. `V86F`, ancestral amino acid first. Comma-separated when there is more than one. `.` when there is none, which includes synonymous and intergenic positions.
- `mutation_type`: `N` nonsynonymous, `S` synonymous, `P` promoter, `I` intergenic, `U` unknown because the ancestral allele could not be inferred.
- `protein_id`, `locus_tag`, `product`: gene and protein annotation for the gene the SNV sits in. `.` when the SNV is intergenic.

(3) Prediction/filter columns

Each filter column is a token: `1` means this filter is the one that removed the evidence for a variant at this position, `0` means the position was still variable after the filter ran, and `-1` means the position had already stopped varying before the filter was reached.

- `Pred_label`: final predicted label (typically 1 = true, 0 = false).
- `CNN_pred`: CNN class output （1 = true, 0 = false).
- `WideVariant_pred`: WideVariant (rule-based) prediction summary （1 = true, 0 = false).
- `CNN_prob`: CNN confidence score.
- `Qual_filter`: per-call quality filter token (default cutoff: FQ quality above 30).
- `Cov_filter`: per-call coverage filter token (default: at least 5 forward and 5 reverse reads).
- `MAF_filter`: major-allele-frequency filter token (default: at least 0.85 of reads on one base).
- `Indel_filter`: indel-support filter token (default: fewer than 0.33 of reads supporting an indel).
- `MFAS_filter`: fraction-ambiguous-samples filter token (default: no minimum).
- `MMCP_filter`: median-coverage-position filter token (default: median depth at least 5x across samples).
- `CPN_filter`: copy-number filter token (default: below 4x the genome-wide median on average, and below 7x in any one sample).
- `Fix_filter`: fixed-mutation criterion token.
- `Whether_recomb`: recombination flag (1 yes, 0 no).
- `Fraction_ambiguous_samples`: fraction of ambiguous calls at this site across samples.
- `Gap_filter`: gap/region filter token.
- `CNN_pred_raw`: the CNN's own class output, before the final decision rewrote `CNN_pred`.
  Use this column, not `CNN_pred`, when you want to know what the model actually said.
  `skip` means the CNN did not score this position.
- `CNN_prob_raw`: the CNN's own confidence score, matching `CNN_pred_raw`.
- `Gap_reason`: which of the two conditions behind `Gap_filter` applied -- `gap` (reads around
  this position do not align cleanly) or `no_variation` (fewer than two distinct base calls
  across samples, so there was nothing to compare).
- `Removed_by`: the single stage that removed this position, or `kept`. This is the column to
  read first. The filters run in order and only the first one to fire reports `1`, so this
  names that filter. Values: `kept`, any of the nine filter names, `not_scored_by_CNN`,
  `CNN_rescue_declined` (the filters called it, the model did not, and it did not meet the bar
  for the benefit of the doubt), or `CNN`.

The cutoffs above are the defaults and are set in `pipeline.yaml`.

### `CNN_pred` vs `CNN_pred_raw`

`CNN_pred` and `CNN_prob` are rewritten by the final decision step in two cases: when the
quality filter removes a position (both are set to 0) and when the rule-based filters overrule
a model rejection (`CNN_pred` is set to 1 and `CNN_prob` becomes `1 - probability`). The
`_raw` columns always hold what the model actually output.

Practical note: these columns explain why a site was kept or rejected. These filters are included in the output to facilitate checking and validation of the identified SNVs.

(4) Remaining annotation columns

- `gene_num_global`: gene number that is unique across the whole genome and does not restart between contigs. This is the number the dN/dS output uses.
- `gene_num_contig`: index of the gene within its own contig, restarting at 1 on each contig. A value ending in `.5` means the position is intergenic, between that gene and the next.
- `quality`: per-site mutation quality used in the annotation table.
- `ontology`, `strand`: gene annotation fields. `strand` is `1.0` for a forward gene and `-1.0` for a reverse one.
- `gene_contig_start_pos`, `gene_contig_stop_pos`: first and last base of the gene on the contig, inclusive. The start is always the leftmost position on the reference, so on a reverse-strand gene the gene is read from `gene_contig_stop_pos` back toward `gene_contig_start_pos`.
- `aa_pos`: 1-based codon number of the affected amino acid within `translation`.
- `possible_codons_at_site`: the four codons this position would produce if the changed base were A, T, C, then G, in that order. The bases are already complemented for reverse-strand genes, so the codons read in gene orientation while their order follows the genome-forward base.
- `possible_AAs_at_site`: the amino acid each of those four codons encodes, in the same order. When all four are identical, no change at this position can alter the protein.
- sample genotype columns (one column per sample -> per-sample nucleotide call).
- `sequence`, `translation`: the gene coding sequence and its translated amino-acid sequence. On a reverse-strand gene, `sequence` is the reverse complement of the reference interval, so it reads in gene orientation.

## 2. Per-position prediction/filter summary table

`snv_table_cnn_raw.tsv` is the CNN + filter summary table, written to
`2-SNV-filtering/group_<group>/`. It looks like this:



```
genome_pos	Pred_label	CNN_pred	WideVariant_pred	CNN_prob	Qual_filter	Cov_filter	MAF_filter	Indel_filter	MFAS_filter	MMCP_filter	CPN_filter	Fix_filter	Whether_recomb	Fraction_ambiguous_samples	Gap_filter
3989	1	1	1	0.99992144	0	0	0	0	0	0	0	0	0	0.000000	0
13554	1	1	1	0.9999056	0	0	0	0	0	0	0	0	0	0.000000	0
42123	1	1	0	0.9999666	0	1	-1	-1	-1	-1	-1	-1	0	0.000000	0
......
```

Same as `(3) Prediction/filter columns` in `snv_table_final.tsv`. This file does not include
annotation information for each SNV.

Practical note: When the number of candidate SNVs exceeds 100,000, the program switches to fast
mode and outputs only `snv_table_cnn_raw.tsv` by default for computational efficiency. That
threshold is `fast_mode_positions` in `pipeline.yaml`.
