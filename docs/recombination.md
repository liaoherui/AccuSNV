# Recombinant SNV flagging

A clonal phylogeny assumes that each SNV arose independently, as do many inferences of evolutionary change (e.g., parallel mutations). However, homologous recombination of a tract of donor DNA can introduce multiple SNVs non-independently in the same locus. AccuSNV looks for the signature of such tracts and flags the SNVs that may arise from them. This detection is on by default; however, it is a conservative approach - it may flag SNVs that are real - and so you may sometimes wish to turn it off.

Recombinant SNV tracts are identified as SNVs that:

1. Occur within a close distance of each other, set by `recomb_distance_bp` in `pipeline.yaml` (1000 bp by default).
2. Are almost always found in the same set of isolates, set by `recomb_corr_threshold` in `pipeline.yaml` (a Pearson correlation cutoff, 0.75 by default).

```{figure} _static/figures/recombination_concept.svg
:alt: Two pairs of nearby SNV positions, one varying independently across samples and one varying together
:width: 100%

Only the second pair meets both criteria, so only it is flagged.
```

This is not intended to capture all SNVs that are possibly recombinant - only *recombinant tracts*. Identifying all possible recombinant positions across a phylogeny would require more advanced phylogenetic inference.

## Interpretation

Flagged recombinant positions **stay in the SNV tables**, and are labeled with the column `Whether_recomb = 1`. However, these positions are by default removed from the analyses that assume independence of SNVs:

| Analysis                                          | Recombinant SNVs |
| ------------------------------------------------- | ---------------- |
| `snv_table_final.tsv`, `snv_table_unfiltered.tsv` | kept, flagged    |
| dN/dS                                             | excluded         |
| Parsimony tree                                    | excluded         |
| dMRCA distances                                   | excluded         |

Passing `--skip_recombination` turns off detection of recombinant SNVs, and they are therefore counted in all analyses. This is all-or-nothing: there is no way to re-admit individual flagged sites. `--include_positions` does not do it, because it only sets `Pred_label` to 1, while dN/dS, tree building and dMRCA all select on `Whether_recomb`, which it leaves alone.

## Recombination diagnostic figure

AccuSNV creates the file `2-SNV-filtering/group_<group>/snv_filter_recombo.png`, which shows SNV positions along the reference genome (contigs are concatenated in order), demonstrating the loci.

```{figure} _static/figures/recombination_positions.png
:alt: Genome-wide plot with blue lines for SNVs and red lines for flagged recombinant positions
:width: 100%

A simulated cohort containing three recombined blocks. The dense red band near position 330,000
is one imported tract; the scattered red lines elsewhere are pairs that happened to correlate.
```

