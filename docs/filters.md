# SNV filtering and calling

In the AccuSNV pipeline, each candidate SNV is assessed in two ways: the probability from the neural network (Above >0.5), and whether it passes a set of nine filters inherited from the WideVariant pipeline. Results from both are reported in the SNV tables for every position so you can identify the reasoning behind a final call.

## Overall SNV filtering logic

**1. If the variant-quality check rejected the position, it is *not* a SNV.** 

First, a SNV candidate position must have a bcftools "FQ" quality score higher than 30 (inverted from the bcftools negative format) by default. This is the `Qual_filter`.

**2. If the CNN calls a position as positive, it is a SNV.** 

Next, if a SNV candidate position receives a CNN probability >0.5, then it is considered a confident SNV position.

**3. If the WideVariant filters call a SNV that the CNN does not, then it must pass one additional rule**.

In this case, the decision is made based on whether or not too many of the samples in your set have mixed read support at the site (both genotypes observed). This is the `Fraction_ambiguous_samples` parameter, which by default must be <25% of samples if you have fewer than 20 samples, and <10% of samples if you have more than 20 samples.

If a SNV site passes all WideVariant filters and has few ambiguous samples, it can override the AccuSNV CNN. 

**4. If neither the CNN nor the WideVariant filters call a SNV, it is *not* a SNV.** 

 

## The WideVariant filters

| Filter         | Requirements                                                                                                                        | Default cutoff                                                                 | Parameter                             |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------- |
| `Qual_filter`  | A high enough bcftools FQ score at the site. (its sign is flipped from bcftools convention, so higher is better)                    | above 30                                                                       | `call_min_qual`                       |
| `Cov_filter`   | Enough reads  at the site on **both** strands.                                                                                      | at least 5 forward and 5 reverse                                               | `min_cov_filt`                        |
| `MAF_filter`   | Most of a sample's reads agree on one base. Rejects positions where a sample's reads are mixed.                                     | at least 85%                                                                   | `call_min_major_allele_freq`          |
| `Indel_filter` | Few of the supporting reads contain an insertion or deletion anywhere in the read, which could make the local alignment unreliable. | fewer than 33%                                                                 | `call_max_indel_frac`                 |
| `MFAS_filter`  | Enough samples have a confident base call.                                                                                          | *no minimum by default*, can be modified.                                      | `max_frac_ambiguous_samples`          |
| `MMCP_filter`  | The median read depth across *all samples* is high enough.                                                                          | at least 5x                                                                    | `min_median_coverage_position`        |
| `CPN_filter`   | Read depth is similar to the genome-wide depth, to avoid repeat regions.                                                            | under 4x the genome median on average, under 7x in any one sample              | `max_mean_copynum`, `max_max_copynum` |
| `Fix_filter`   | At least one sample carries a confident base that differs from the inferred ancestor.                                               | mutation quality at least 1                                                    | `min_mut_qual`                        |
| `Gap_filter`   | Variant samples must have similar depths at the site compared to the genome wide average, and to non-variant samples.               | variant samples below 5% of their median depth while the others stay above 20% | not adjustable                        |

The first four are per sample: they are required for a sample to nominate a SNV at that position. The next five are per-position: they test a position's data across every sample at once.

## Inferring the ancestral allele

AccuSNV infers the ancestral allele at a site for each group using the following approach:

1. **If the outgroup has one allele observed in the ingroup, that allele is the ancestor.**** This is basic outgroup rooting and should be the way most ancestral alleles are resolved in your data.
2. **If there is no outgroup, then the ingroup's most common allele is assumed ancestral.** 

   This is kind of a basic heuristic, and it is probably best to run with an outgroup if you want to properly root your tree.
3. The reference genome is not the ancestor and AccuSNV does not treat it as one. The `reference_allele` column is reported separately from `ancestral_allele`, and they differ whenever the whole ingroup has diverged from the reference at a position.

## Sample-level checks

There are also two checks that can eliminate a sample from the results.

The first is **zero-coverage fraction**. A sample with no read coverage at more than `min_cov_samp` percent (45 by default) of the candidate positions is dropped from the output.

The second is **manual exclusion**. `exclude_samp` parameter can either be used to list samples names to drop manually, or if it is an integer, it will represent a SNV count above which a sample is dropped. This second approach can automatically exclude samples that are highly divergent from the rest of your set. 

:::{warning}
`CNN_pred` and `CNN_prob` are rewritten in the final tables to reflect the reconciled verdict of the pipeline. Sites that WideVariant calls will have `CNN_pred` set to `1` and `CNN_prob` to one minus the original probability (honestly I acknowledge this is very hacky). The unmodified CNN values are still kept in `CNN_pred_raw` and `CNN_prob_raw`, and in `snv_table_cnn_raw.tsv`.
:::
