# Output files

This page explains all of the output files created by AccuSNV when run by default. These all go into the `-o` output directory as specified by the user./

## The output directory structure

Here is a complete look at all of the files that AccuSNV will generate for you upon a successful run:

```text
my_output/
├── group_pe_test_snv_table_final.tsv          
├── group_pe_test_snv_table_unfiltered.tsv
├── group_pe_test_snv_dashboard.html           
├── group_pe_test_snv_tree_final.nwk.tree
├── group_pe_test_snv_table_invariant_positions.tsv
├── group_pe_test_snv_table_rejected_upstream.tsv
├── samples.csv
├── accusnv.log
├── accusnv.full.log
├── accusnv.snakemake.log
├── configs/
│   ├── config.yaml
│   └── pipeline.yaml
├── logs/                   
├── 1-Mapping/
│   ├── alignment/          
│   │   └── trimmed_filtered_reads/   
│   ├── vcf/                
│   ├── quals/              
│   └── diversity/          
├── 2-SNV-filtering/
│   ├── raw_tables/
│   │   ├── group_pe_test_candidate_mutation_table.npz
│   │   ├── group_pe_test_coverage_matrix_raw.npz
│   │   ├── group_pe_test_coverage_matrix_norm.npz
│   │   └── group_pe_test_allpositions.pickle
│   └── group_pe_test/
│       ├── snv_table_final.tsv
│       ├── snv_table_unfiltered.tsv
│       ├── snv_table_cnn_raw.tsv
│       ├── snv_table_invariant_positions.tsv
│       ├── candidate_mutation_table_final.npz
│       ├── snvs_per_sample.tsv
│       ├── snvs_per_sample.png
│       ├── snvs_histogram_per_sample.png
│       ├── ZOOMED_snvs_histogram_per_sample.png
│       ├── snvs_from_recombo.csv 
│       ├── snv_filter_recombo.png
│       ├── snv_filter_sample_coverage_hist.png
│       ├── snv_filter_sample_toomanyNs_hist.png
│       ├── snv_table_filtered_tmp.tsv
│       └── _snv_state.npz
└── 3-Analysis/
    └── group_pe_test/
        ├── snv_dashboard.html
        ├── dNdS_out/
        │   ├── data_dNdS.npz
        │   ├── dnds_genomewide.tsv
        │   └── dnds_per_gene.tsv
        ├── phylogeny/
        │   ├── snv_tree_final.nwk.tree
        │   ├── snv_table_tree_distances.tsv
        │   ├── snv_table_simple_stats.tsv
        │   ├── alignment.fa
        │   ├── alignment.phylip
        │   ├── tree.nexus
        │   ├── dnapars.log
        │   ├── dnapars_options.txt
        │   ├── dnapars_report.txt
        │   ├── snvChart.csv         (only with --build_snv_trees)
        │   └── snv_trees/           (only with --build_snv_trees)
        ├── diagnostic_figures/
        │   ├── snv_qc_heatmap_calls.png
        │   ├── snv_qc_heatmap_coverage.png
        │   └── snv_qc_heatmap_quals.png
        └── per_snv_barcharts/
            ├── p_11476_bar_chart.png
            └── ...
```

When you have more than one group in your sample sheet, each of the primary files appears once per group, and `2-SNV-filtering/` and `3-Analysis/` will have one subdirectory per group.

`samples.csv` is a copy of the sample sheet you passed with `-i`, rewritten with the FASTQ and reference paths AccuSNV resolved for each sample.  The workflow actually reads this file, so it is the file to check when there might be issues with a sample being matched correctly to its data.

## The primary results files

| File                                              | Description                                                                                                               |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `group_<group>_snv_table_final.tsv`               | SNV calls with all annotates. One row per position. See [The final SNV table](snv_table.md).                              |
| `group_<group>_snv_dashboard.html`                | An interactive HTML dashboard for browsing your SNVs and their metadata.                                                  |
| `group_<group>_snv_table_unfiltered.tsv`          | The SNV table including the rejected candidate positions, so you can see which SNV sites were dropped and why.            |
| `group_<group>_snv_tree_final.nwk.tree`           | The maximum-parsimony tree in Newick format. See [Phylogeny and dMRCA](phylogeny.md).                                     |
| `group_<group>_snv_table_invariant_positions.tsv` | Candidate positions where every ingroup sample agrees: any differences from the reference that all of your samples share. |
| `group_<group>_snv_table_rejected_upstream.tsv`   | The most raw set of SNVs, including SNVs that were immediately dropped by bcftools filtering on FQ quality.               |

The SNV tables and the invariant positions are also copied in `2-SNV-filtering/group_<group>/`, the dashboard in `3-Analysis/group_<group>/`, and the tree is copied in `3-Analysis/group_<group>/phylogeny/`.

## SNV calling stage files: `2-SNV-filtering/group_<group>/`

| File                                 | Description                                                                                                                                                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `snv_table_cnn_raw.tsv`              | The network's raw output: `genome_pos`, `CNN_pred` (0 or 1) and `CNN_prob`, one row per position it scored, for raw and unmodified probabilities from the AccuSNV CNN for each candidate site.                           |
| `candidate_mutation_table_final.npz` | Numpy version of the candidate mutation table. You can load it with `numpy.load(path, allow_pickle=True)`.                                                                                                               |
| `snvs_per_sample.tsv`                | Counts of the number of SNVs in each sample, computed before filtering. Two columns: `sample`, `snv_count`.                                                                                                              |
| `snv_table_invariant_positions.tsv`  | Sites that all in-group samples differ from the reference on.                                                                                                                                                            |
| `snvs_from_recombo.csv`              | The `genome_pos` of positions flagged as potential recombinant, one per line.                                                                                                                                            |
| `snv_table_filtered_tmp.tsv`         | The SNV filter results before annotation merges them in. Kept mainly so that `--downstream_only` can re-annotate.                                                                                                        |
| `_snv_state.npz`                     | An internal file for passing data between stages: read counts, qualities, filtered calls, the inferred ancestor, the recombination mask. The dashboard and the tree builder read this. No need to edit or read directly. |

### QC figures from SNV calling

`snvs_per_sample.png`: The per-sample SNV count as a scatter plot. Any sample with more than 1000 SNVs is labeled in red.

`snvs_histogram_per_sample.png` and `ZOOMED_snvs_histogram_per_sample.png`: The histograms of those counts. The zoomed version includes only samples with <1000 SNVs and draws a line at 100 SNVs.

`snv_filter_sample_coverage_hist.png`: Median depth per sample. 

```{figure} _static/figures/sample_coverage_hist.png
:alt: Histogram of median per-sample coverage for four test isolates, all near 15x
:width: 80%

The four test isolates.
```

`snv_filter_sample_toomanyNs_hist.png`: The fraction of SNV positions where each sample has no confident base call. 

`snv_filter_recombo.png`: SNV positions along the genome. They are colored red when flagged as a potential recombinant. See [the recombination diagnostic figure](recombination.md#recombination-diagnostic-figure).

## The evolutionary analyses: `3-Analysis/group_<group>/`

| File                                                                                                     | Description                                                                                                                    |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `dNdS_out/dnds_genomewide.tsv`                                                                           | `dNdS` calculated for the whole genome. See [dN/dS](dnds.md).                                                                  |
| `dNdS_out/dnds_per_gene.tsv`                                                                             | `dNdS` calculated for each gene. See [dN/dS](dnds.md).                                                                         |
| `dNdS_out/data_dNdS.npz`                                                                                 | `dNdS` as a numpy file. See [dN/dS](dnds.md).                                                                                  |
| `phylogeny/snv_tree_final.nwk.tree`                                                                      | The tree in Newick format.                                                                                                     |
| `phylogeny/tree.nexus`                                                                                   | The tree in NEXUS format.                                                                                                      |
| `phylogeny/snv_table_tree_distances.tsv`                                                                 | Distances to the inferred ancestor for each sample. Columns are: `sample_name`, `num_SNVs_to_ancestor`.                        |
| `phylogeny/snv_table_simple_stats.tsv`                                                                   | Summary values for a group: sample count, SNV count, and the median, minimum and maximum dMRCA.                                |
| `phylogeny/alignment.fa`, `alignment.phylip`, `dnapars.log`, `dnapars_options.txt`, `dnapars_report.txt` | Inputs and outputs from `dnarpars` during tree building. Worth reading if the tree looks wrong.                                |
| `phylogeny/snv_trees/`                                                                                   | One NEXUS tree per SNV, tips colored by base call. Only made with `--build_snv_trees`.                                         |
| `phylogeny/snvChart.csv`                                                                                 | The per-sample base calls used for the snv_trees. Only made with `--build_snv_trees`.                                          |
| `per_snv_barcharts/p_<pos>_bar_chart.png`                                                                | Read counts at every position showing samples by base and by forward/reverse reads. Written for runs with 1000 SNVs or fewer.  |
| `diagnostic_figures/snv_qc_heatmap_*.png`                                                                | Three heatmaps of samples by SNV position: base calls, coverage, call quality. Only written for runs with fewer than 300 SNVs. |

The bar charts and the heatmaps are made by the report stage, so `--skip_report` skips them too even though they are separate files.

### SNV bar charts

These barcharts are output for each position, and are also available in the interactive HTML dashboard. There are two bars per sample: forward reads then reverse reads, colored by base.

```{figure} _static/figures/barchart_p11476.png
:alt: Stacked bar chart of read counts by base for four samples at position 11476
:width: 85%
```

### QC heatmaps

These heatmaps show coverage and SNV quality across positions.

```{figure} _static/figures/qc_heatmap_calls.png
:alt: Heatmap of base calls for four samples across ten SNV positions
:width: 100%

`snv_qc_heatmap_calls.png`
```

```{figure} _static/figures/qc_heatmap_coverage.png
:alt: Heatmap of read depth for four samples across ten SNV positions
:width: 100%

`snv_qc_heatmap_coverage.png`
```

.

## Intermediate files

The intermediate files created by AccuSNV are likely rarely needed, but can be inspected when a run has an issue  or when you want to reuse the data in another analysis.

`1-Mapping/alignment/`: deduplicated, sorted, indexed BAM per sample (`*_aligned.sorted.bam` and `.bai`) the duplicate statistics from `samtools markdup` (`*.bam.stats.txt`). 

`1-Mapping/alignment/trimmed_filtered_reads/`:  `*_R1_trimmed.fastq.gz` and `*_R2_trimmed.fastq.gz` from `cutadapt`,  `*_R1_filtered.fastq.gz` and `*_R2_filtered.fastq.gz` from `sickle`. `*_unpaired.fastq.gz` contains reads without high quality pairs, but these are not used by the pipeline.

`1-Mapping/vcf/`: The whole-genome `strain.vcf.gz` (all genomic positions) and SNP-only `variant.vcf.gz`. `*.upstream_rejects.tsv` is each sample's rejected SNVs.

`1-Mapping/quals/`: `*.quals.pickle.gz` is the `bcftools` FQ score at every position of the genome, and `*.positions.pickle` is the candidate SNV positions for each sample.

`1-Mapping/diversity/`: `*.diversity.pickle.gz` has 40 statistics for every position of the genome: read counts per base per strand, average base quality, mapping quality and tail distance per base, and indel support. 

`*.coverage.pickle.gz`  is the **per-position depth pulled out of the same pileup**.

`2-SNV-filtering/raw_tables/`: The candidate mutation table, and `*_allpositions.pickle`, the merged list of candidate positions.

## AccuSNV logs

AccuSNV produces three log files with varying levels of detail.

`accusnv.log`: This is the main log file. It contains one line per step for each sample, and any warnings and errors. Failed jobs are logged in this file too. You can read this log first if something goes wrong.

`accusnv.full.log`: This log also contains the raw output of `bwa`, `samtools`, `bcftools`, `cutadapt` and `sickle` for each sample. 

`logs/`: Individual job logs that are concatenated into the two files above. They are named like `<step>-<sample or group>.log` and `.full.log`. This output is created primarily by snakemake and these results should always end up in the full logs as well, but they are preserved for completeness.

`accusnv.snakemake.log`: This logs the raw output of Snakemake, including every job with the resources requested, and the snakemake errors for anything that failed. 

## Possible empty outputs

Some situations can produce empty files:

* **A clonal group with no SNVs:** the tables will be empty, the tree file will be empty, and the dashboard will be empty.
* **Fast mode**, above 100,000 candidate positions by default: the run writes `snv_table_cnn_raw.tsv` and `candidate_mutation_table_final.npz` and stops. `snv_table_final.tsv` is a copy of the raw scores rather than the full annotated columns, and nothing downstream runs.
