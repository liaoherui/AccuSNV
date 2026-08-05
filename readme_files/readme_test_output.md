# Output files of AccuSNV

##  Main output of Snakemake pipeline

The headline results of the **Quick Test** are written at the top level of the output directory,
one set per sample group. Everything behind them is under `2-SNV-filtering/group_<group>/` and
`3-Analysis/group_<group>/`. The folder structure should look like this:

```
tree cae_pe_test_snakemake

|-- group_pe_test_snv_table_final.tsv
|-- group_pe_test_snv_table_unfiltered.tsv
|-- group_pe_test_snv_dashboard.html
|-- group_pe_test_snv_tree_final.nwk.tree
|-- samples.csv
|-- accusnv.log
|-- accusnv.full.log
|-- accusnv.snakemake.log
|-- configs
|   |-- config.yaml
|   `-- pipeline.yaml
|-- 1-Mapping
|   |-- alignment
|   |-- diversity
|   |-- quals
|   `-- vcf
|-- 2-SNV-filtering
|   |-- group_pe_test
|   |   |-- candidate_mutation_table_final.npz
|   |   |-- snv_table_final.tsv
|   |   |-- snv_table_unfiltered.tsv
|   |   |-- snv_table_cnn_raw.tsv
|   |   |-- snvs_per_sample.tsv
|   |   |-- snvs_per_sample.png
|   |   |-- snvs_histogram_per_sample.png
|   |   |-- ZOOMED_snvs_histogram_per_sample.png
|   |   |-- snv_filter_recombo.png
|   |   |-- snv_filter_sample_coverage_hist.png
|   |   |-- snv_filter_sample_toomanyNs_hist.png
|   |   `-- _snv_state.npz
|   `-- raw_tables
|       |-- group_pe_test_candidate_mutation_table.npz
|       |-- group_pe_test_coverage_matrix_raw.npz
|       |-- group_pe_test_coverage_matrix_norm.npz
|       `-- group_pe_test_allpositions.pickle
`-- 3-Analysis
    `-- group_pe_test
        |-- snv_dashboard.html
        |-- dNdS_out
        |   |-- data_dNdS.npz
        |   |-- dnds_genomewide.tsv
        |   `-- dnds_per_gene.tsv
        |-- phylogeny
        |   |-- snv_tree_final.nwk.tree
        |   |-- snv_table_tree_distances.tsv
        |   |-- snv_table_simple_stats.tsv
        |   |-- alignment.fa
        |   |-- alignment.phylip
        |   |-- tree.nexus
        |   |-- dnapars.log
        |   |-- dnapars_options.txt
        |   `-- dnapars_report.txt
        |-- diagnostic_figures
        |   |-- snv_qc_heatmap_calls.png
        |   |-- snv_qc_heatmap_coverage.png
        |   `-- snv_qc_heatmap_quals.png
        |-- per_snv_barcharts
        |   |-- p_1041058_bar_chart.png
        |   |-- p_1054786_bar_chart.png
        |   |-- p_10866_bar_chart.png
        |   |-- ......
        `-- snv_trees
            |-- p_1041058_1.tree
            |-- p_1054786_1.tree
            |-- ......
```

### Core files:

| File or Folder |  Description |
| ---  | --- | 
| `group_<group>_snv_table_final.tsv`  | Final SNV report table (recommended primary text result for interpretation). More details, including explanations of the columns in this file, can be found [here](readme_annotation_table.md). The same file is also written to `2-SNV-filtering/group_<group>/snv_table_final.tsv`.
| `2-SNV-filtering/group_<group>/snv_table_cnn_raw.tsv` | Per-position prediction/filter summary table (CNN output + rule-based filters (from WideVariant)). Note that this file does not include annotation information for each SNV.
| `group_<group>_snv_dashboard.html`  | Interactive final HTML report (recommended to view). One self-contained page; nothing else needs to be installed and it works offline.
| `2-SNV-filtering/group_<group>/candidate_mutation_table_final.npz`  | Final machine-readable SNV matrix for downstream analysis. Contains arrays such as sample names, genomic positions, counts, quality values, prediction labels/probabilities, and recombination flags.

For final SNV calling results, please use:

`group_<group>_snv_table_final.tsv` as the primary human-readable SNV result table.

`candidate_mutation_table_final.npz` as the machine-readable final result for any downstream analysis or re-analysis.

### Other files (include QC figures):

All paths below are relative to the group folder they sit in.

| File or Folder |  Description |
| ---  | --- | 
| `2-SNV-filtering/group_<group>/snv_table_unfiltered.tsv`  | Every candidate position either caller called, with the CNN and filter breakdown for each, including the positions that were rejected. A copy is written at the top level as `group_<group>_snv_table_unfiltered.tsv`.
| `2-SNV-filtering/group_<group>/snvs_per_sample.tsv`  | Per-sample SNV count table (simple summary counts).
| `2-SNV-filtering/group_<group>/snvs_per_sample.png`  | Per-sample SNV count plot.
| `2-SNV-filtering/group_<group>/snvs_histogram_per_sample.png`  | Histogram of SNV counts across samples.
| `2-SNV-filtering/group_<group>/ZOOMED_snvs_histogram_per_sample.png`  | Histogram of SNV counts across samples. (Zoomed histogram version for easier viewing of the main range)
| `2-SNV-filtering/group_<group>/snv_filter_recombo.png`  | Recombination filtering plot (visual mark of retained vs suspected recombination-associated positions).
| `2-SNV-filtering/group_<group>/snv_filter_sample_coverage_hist.png`  | Sample-level coverage histogram with cutoff line (used to identify low-coverage samples).
| `2-SNV-filtering/group_<group>/snv_filter_sample_toomanyNs_hist.png`  | Sample-level ambiguous-call (N fraction) histogram with cutoff line.
| `2-SNV-filtering/group_<group>/_snv_state.npz`  | Internal calling state (read counts, qualities, filtered calls) that the dashboard reads to show the raw data behind each call.
| `2-SNV-filtering/raw_tables/`  | Group-level candidate mutation table and coverage matrices produced before the CNN stage.
| `3-Analysis/group_<group>/dNdS_out/data_dNdS.npz`  | Saved dN/dS result bundle (e.g., dNdS, confidence interval bounds, mutation counts).
| `3-Analysis/group_<group>/dNdS_out/dnds_genomewide.tsv`  | The same genome-wide dN/dS numbers as text.
| `3-Analysis/group_<group>/dNdS_out/dnds_per_gene.tsv`  | Per-gene dN/dS breakdown, one row per gene with mutations.
| `3-Analysis/group_<group>/phylogeny/snv_tree_final.nwk.tree`  | Final genome-wide SNV phylogeny in Newick format. A copy is written at the top level as `group_<group>_snv_tree_final.nwk.tree`.
| `3-Analysis/group_<group>/phylogeny/snv_table_tree_distances.tsv`  | Per-sample SNP distances to ancestor.
| `3-Analysis/group_<group>/phylogeny/snv_table_simple_stats.tsv`  | Summary statistics for the SNV set used to build the tree.
| `3-Analysis/group_<group>/phylogeny/`  | Also holds the dnapars working files: `alignment.fa`, `alignment.phylip`, `tree.nexus`, `dnapars.log`, `dnapars_options.txt` and `dnapars_report.txt`.
| `3-Analysis/group_<group>/diagnostic_figures/snv_qc_heatmap_calls.png`  | Heatmap of per-sample calls across SNV positions.
| `3-Analysis/group_<group>/diagnostic_figures/snv_qc_heatmap_coverage.png`  | Heatmap of per-sample coverage across SNV positions.
| `3-Analysis/group_<group>/diagnostic_figures/snv_qc_heatmap_quals.png`  | Heatmap of per-sample quality across SNV positions (quality axis labeled as FQ-derived quality).
| `3-Analysis/group_<group>/per_snv_barcharts/`  | One per-SNV bar chart (`p_<genome_pos>_bar_chart.png`) showing base support across samples at that position. Same format as those (e.g. Fig.1) described in the paper. Linked from the dashboard.
| `3-Analysis/group_<group>/snv_trees/`  | Per-SNV tree files (`p_<pos>_<n>.tree`), where `n` >= 2 flags a homoplasic (parallel) mutation. Written only when you pass `--build_snv_trees`.
| `accusnv.log`  | One line per step per sample saying what it did and what came out of it, plus every warning and error. This is the one to read first.
| `accusnv.full.log`  | The same, plus the per-sample detail and everything bwa, samtools, bcftools, cutadapt and sickle printed.
| `accusnv.snakemake.log`  | Everything Snakemake printed, verbatim. A failed job is already summarised in `accusnv.log` with its reason and resource tier; this is the surrounding detail.
| `configs/config.yaml`, `configs/pipeline.yaml`  | The two config files generated for this run, with defaults filled in. Copy, edit and pass back with `-c` or `-p` to change them.

##  Re-running the downstream analyses

Passing `--downstream_only` re-runs only the evolutionary analyses from an existing output
directory, without repeating alignment and SNV calling:

```
accusnv --downstream_only -i <samples.csv> -r <reference_genomes_dir> -o <output_dir>
```

It reads the SNV tables and `candidate_mutation_table_final.npz` already in
`2-SNV-filtering/group_<group>/` and rewrites everything under `3-Analysis/`. The output files are
the ones listed above; no separate downstream output folder is created.

This exists to let users re-analyze the same final call set quickly with different downstream
choices (for example, recombination exclusion or repeated dN/dS runs), without re-running the
full alignment and calling workflow on HPC. Those choices are set in `pipeline.yaml`.

##  Other output of Snakemake pipeline

`1-Mapping/` <BR>

Mapping-stage outputs generated from read alignment and pileup/VCF processing.
It contains intermediate files used to build the candidate mutation tables in `2-SNV-filtering/raw_tables`.

`1-Mapping/alignment/` <BR>

Per-sample alignment files (SAM/BAM/BAM index and related mapping intermediates), whichever aligner was used.

`1-Mapping/vcf/` <BR>
Per-sample pileup and VCF outputs (raw and SNP-filtered), produced from mapped BAM files.

`1-Mapping/quals/` <BR>
Per-sample quality matrices (*.quals.pickle.gz) derived from VCF and used later in SNV filtering/calling.

`1-Mapping/diversity/` <BR>
Per-sample diversity matrices (*.diversity.pickle.gz) used by the candidate mutation table step. Each
holds 40 per-position statistics: read counts, base and mapping qualities, tail distances and indel counts.

`2-SNV-filtering/raw_tables/` <BR>
Group-level candidate mutation tables (`group_<group>_candidate_mutation_table.npz`), the position
list, and the genome-wide coverage matrices, used as direct input to the CNN calling stage.

`2-SNV-filtering/group_<group>/` <BR>
The SNV calls for one group: the final tables, the CNN and filter verdicts, the filtering QC figures,
and `candidate_mutation_table_final.npz`.

`3-Analysis/group_<group>/` <BR>
The evolutionary analyses for one group: dN/dS, the parsimony tree and dMRCA distances, the QC
heatmaps, the per-SNV bar charts, and the interactive dashboard.

`configs/` <BR>
Run-specific configuration folder generated in your output directory. Contains the generated
`config.yaml` (Snakemake execution settings and run paths) and `pipeline.yaml` (pipeline parameters).
`config.yaml` is the file users typically edit for cluster resource tuning.

`samples.csv` <BR>
A copy of the sample sheet used for this run, with the read file paths and reference FASTA path
resolved, written at the top level of the output directory.
