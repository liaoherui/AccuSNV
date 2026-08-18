# The final SNV table

`group_<group>_snv_table_final.tsv` is the primary output of AccuSNV that you will likely want to work with. This TSV file has one row per filtered SNV position that AccuSNV called positive, with a considerable amount of information and detail for that SNV.

`snv_table_unfiltered.tsv` has the same columns, but it also contains the rows which AccuSNV decided are not valid SNVs (`Pred_label = 0`). Everything described below applies to both.

The information in this table can be divided into four general categories: SNV position and impact, how and why AccuSNV called it a SNV, the specific annotations, and the base pair at that site found in each sample.

:::{tip}
You can load this table in R or Python for your analysis. A Python example is below:

```python
import pandas as pd
snvs = pd.read_csv('group_pe_test_snv_table_final.tsv', sep='\t')
snvs[['genome_pos', 'product', 'aa_mutation', 'mutation_type', 'CNN_prob']]
```

:::

**Missing values**:  a `.` in the table means the column doesn't apply. For example, intergenic SNVs have no `product`, `aa_pos` or `sequence`, so all three are recorded as `.` for intergenic SNVs.

## SNV position and impact

These columns start the table because they are some of the fields you will most likely be interested in:

| Column                     | Example                                          | Meaning                                                                                                                                                                                                              |
| -------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `genome_pos`               | `11476`                                          | The position on the genome, counting through multiple concatenated contigs from 1. This is a global identifier for SNVs used throughout AccuSNV.                                                                     |
| `contig`                   | `NC_018707.1`                                    | The contig the SNV is on.                                                                                                                                                                                            |
| `contig_pos`               | `11476`                                          | The SNV position within that contig, 1-based. Will be equal to `genome_pos` with a single-contig reference.                                                                                                          |
| `ancestral_allele`         | `G`                                              | The inferred ancestral base, which is *not* necessarily the reference base.                                                                                                                                          |
| `derived_allele`           | `T`                                              | Every other base seen in any sample (comma-separated if more than one).                                                                                                                                              |
| `reference_allele`         | `G`                                              | The base of the reference genome at this site.                                                                                                                                                                       |
| `gene_nucleotide_mutation` | `C692A`                                          | The nucleotide mutation with respect to the gene, if the SNV is in a gene: ancestral base, position in the gene, and derived base. Comma-separated for more than one derived allele and will be `.` when intergenic. |
| `gene_nt_position`         | `692`                                            | The position of the SNV within a gene, if it is found within one.                                                                                                                                                    |
| `aa_mutation`              | `S231*`                                          | The amino-acid change, if the SNV changes the protein sequence: ancestral amino acid, codon number, and derived amino acid. `*` represents a stop codon. `.` for synonymous and intergenic positions.                |
| `mutation_type`            | `N`                                              | `N` for nonsynonymous, `S` for synonymous, `P` for promoter, `I` for intergenic, `U` for undetermined.                                                                                                               |
| `protein_id`               | `WP_041446291.1`                                 | Protein accession from the annotation that the SNV falls in. `.` when intergenic.                                                                                                                                    |
| `locus_tag`                | `cds-WP_041446291.1`                             | Locus tag from the annotation. `.` when intergenic.                                                                                                                                                                  |
| `product`                  | `ATP-binding cassette domain-containing protein` | Product description from the annotation of the gene the SNV is found in. `.` when intergenic.                                                                                                                        |

### Gene coordinate systems

The allele genotype columns (`ancestral_allele`, `derived_allele`, `reference_allele`) are reported in **genome-forward** orientation, so they can be compared with each other directly.

But `gene_nucleotide_mutation` is reported in **gene** orientation. On a reverse-strand gene both bases are complemented, which is why the first row of the test data reads `G` to `T` in the allele columns, but `C692A` in the gene column. 

The gene `sequence` is reported in the gene orientation. Therefore, when a gene is on the forward strand:

```
gene_nt_position = contig_pos − gene_contig_start_pos + 1
sequence[gene_nt_position − 1] = ancestral_allele
```

And when a gene is on the reverse strand, the gene is counted from its last base on the contig, and the base is complemented:

```
gene_nt_position = gene_contig_stop_pos − contig_pos + 1
sequence[gene_nt_position − 1] = complement(ancestral_allele)
```

### Mutation types

**N** (nonsynonymous): The change alters the amino acid sequence.

**S** (synonymous): The position is inside a coding sequence, but the change does not alter the amino acid.

**P** (promoter): Not in a gene, but <250 bp upstream of a gene, where it may affect expression.

**I** (intergenic): Not inside a gene and not in the 250 bp window upstream of one.

**U** (undetermined): The position is inside an annotated feature, but the effect is unknown because the gene is not a coding gene (e.g., in rRNA). Positions typed `U` are excluded from dN/dS.

## SNV calling filters

More details on how all of these filters work can be found on the page describing [How SNVs are called](filters.md). The nine filter columns are *not* pass/fail flags. `0` means the position passed the check, `1` means the position failed the check, and `-1` means the position had already been ruled out before that check. 

| Column                       | Description                                                                                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `Pred_label`                 | The final AccuSNV call `1` means this is a SNV. `snv_table_final.tsv` contains only `1` rows while `snv_table_unfiltered` has both `0` and `1`. |
| `CNN_pred`                   | The AccuSNV CNN's call, `1` or `0`, or `skip` when the position was one the CNN could not score. Not a probability, just its yes/no.            |
| `WideVariant_pred`           | Whether the rule-based WideVariant filters called a variant  (`1` or `0`).                                                                      |
| `CNN_prob`                   | The network's probability, between 0 and 1, or `skip` alongside a `skip` in `CNN_pred`.                                                          |
| `Qual_filter`                | SNV bcftools quality (**default**: better than 30).                                                                                             |
| `Cov_filter`                 | Whether the site has reads on both strands (**default**: at least 5x per strand).                                                               |
| `MAF_filter`                 | Whether the site's reads within a sample agree (**default**: at least 85%).                                                                     |
| `Indel_filter`               | Whether too many reads at the site contain an indel (**default**: fewer than 33%).                                                              |
| `MFAS_filter`                | Whether too few samples have a base call at the site (**default**: no minimum).                                                                 |
| `MMCP_filter`                | Whether the site is below minimum median depth across samples (**default**: at least 5x).                                                       |
| `CPN_filter`                 | Whether there is abnormally high coverage at the site (**default**: under 4x the genome median on average, 7x in any sample).                   |
| `Fix_filter`                 | Whether any sample differs from the inferred ancestor with at least `min_mut_qual` at the site.                                                 |
| `Gap_filter`                 | Whether samples with the alternative allele have unusually low or high coverage compared to samples with the reference allele at the site.      |
| `Whether_recomb`             | `1` if this SNV is part of a potential recombined tract. See [Recombination](recombination.md).                                                 |
| `Fraction_ambiguous_samples` | Whether enough samples have clonal read support (not mixed) at the site.                                                                        |
| `CNN_pred_raw`               | The network's call before AccuSNV rewrote `CNN_pred`. Identical to `CNN_pred` unless a rewrite happened.                                        |
| `CNN_prob_raw`               | The network's probability before AccuSNV rewrote `CNN_prob`.                                                                                    |
| `Gap_reason`                 | Why the CNN never scored the site: `gap` for an alignment gap beside it, `no_variation` when every remaining sample had the same base. `.` otherwise. |
| `Removed_by`                 | The one stage that dropped the site, or `kept`. See below.                                                                                      |

### When `CNN_pred` is rewritten

`CNN_pred` and `CNN_prob` are the network's verdict after AccuSNV has reconciled it with the rule-based filters, so they are not always what the network produced. There are two rewrites:

* `Qual_filter` failed. The site is dropped whatever the network said, and `CNN_pred` and `CNN_prob` are both set to `0`.
* The filters called the site, the network did not, and the read evidence was clean enough to override it. `CNN_pred` becomes `1` and `CNN_prob` becomes one minus the original probability (or `1.0` if the network never scored the site).

`CNN_pred_raw` and `CNN_prob_raw` hold the values from before either rewrite, so comparing the two pairs tells you which sites AccuSNV overruled. `snv_table_cnn_raw.tsv` holds the same numbers for the sites the network scored.

### `Removed_by` values

| Value | Meaning |
| ----- | ------- |
| `kept` | `Pred_label` is 1. |
| a filter name, e.g. `Cov_filter` | The first of the nine checks to fail. Only the first reports `1`, so this is the one that did the removing. |
| `Gap_filter` | The CNN could not score the site. `Gap_reason` says why. |
| `not_scored_by_CNN` | The CNN did not score the site and no filter had failed. |
| `CNN_rescue_declined` | The filters called it and the network did not, but the read evidence was too muddy to override the network. |
| `CNN` | Only the network rejected it. |

## Detailed SNV annotations

| Column                    | Example           | Meaning                                                                                                                                                                                                                  |
| ------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `gene_num_global`         | `32.0`            | The gene number across the whole genome (not restarting between contigs). `0.5` means the position is intergenic; `0` means it is in a non-coding feature such as a tRNA or rRNA gene.                                   |
| `gene_num_contig`         | `14.0`            | The gene number per contig, restarting at 1 on each contig.                                                                                                                                                              |
| `quality`                 | `54.0`            | The 'best' pair mutation quality for this position: across all pairs of ingroup samples with different alleles, takes the lower quality for each pair, then the highest pair quality. `.` when it could not be computed. |
| `ontology`                |                   | Any annotation ontology from the annotation file.                                                                                                                                                                        |
| `strand`                  | `-1.0`            | `1.0` for a gene on the forward strand, `-1.0` for the reverse. `.` when intergenic.                                                                                                                                     |
| `gene_contig_start_pos`   | `10263`           | The first base of the gene on the contig.                                                                                                                                                                                |
| `gene_contig_stop_pos`    | `12167`           | Last base of the gene on the contig.                                                                                                                                                                                     |
| `aa_pos`                  | `231`             | The codon number of the affected amino acid within `translation`, counting from 1.                                                                                                                                       |
| `possible_codons_at_site` | `TTA TAA TGA TCA` | The four codons this position *could* produce if the changed base were A, T, C, then G, in that order.                                                                                                                   |
| `possible_AAs_at_site`    | `L * * S`         | The amino acid each of those four possible codons encodes, in the same order.                                                                                                                                            |
| `sequence`                | `ATGTGCATG...`    | The gene's coding sequence. Reverse-complemented for a reverse-strand gene, so it always reads in gene orientation.                                                                                                      |
| `translation`             | `MCMDCSGLGY...`   | The protein sequence for the gene.                                                                                                                                                                                       |

## Sample allele columns

There is also one column per sample, named as in your sample sheet, showing the base that AccuSNV finds at the SNV position in that sample. These calls are rebuilt with looser thresholds than the ones used to decide whether the position is a SNV: a base call needs a quality of 30, at least one read on each strand, and a major-allele frequency of 0.75 (set by `annotate_min_major_allele_freq`). These columns tell you what was observed at a SNV the overall data has already determined to be real, so the per-sample thresholds can be relaxed. If a sample still does not pass them, it gets an `N` here.

### Raw CNN probabilities for SNVs

The file `2-SNV-filtering/group_<group>/snv_table_cnn_raw.tsv` is the CNN's raw output, and has three columns:

```text
genome_pos  CNN_pred  CNN_prob
11476       1         0.999974250793457
96058       1         0.9999998807907104
```

This is the best place to identify the raw CNN probability each site received, as these probabilities are later processed by AccuSNV and combined with the rule-based filters.

In fast mode, with >100,000 candidate positions by default, this file is the main result and is copied to `snv_table_final.tsv` in place of the usual annotated table.
