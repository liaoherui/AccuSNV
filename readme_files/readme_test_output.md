# Output files of AccuSNV


##  Main output of Snakemake pipeline

The main output files of the **Quick Test** (Snakemake pipeline) are in the `cae_pe_test_snakemake/3-AccuSNV/group_pe_test` folder. The folder structure should look like this:

```
tree cae_pe_test_snakemake/3-AccuSNV/group_pe_test/

|-- ZOOMED_snvs_histogram_per_sample.png
|-- bar_charts
|   |-- p_1041058_bar_chart.png
|   |-- p_1054786_bar_chart.png
|   |-- p_10866_bar_chart.png
|   |-- p_1154874_bar_chart.png
|   |-- ......
|-- candidate_mutation_table_final.npz
|-- dNdS_out
|   `-- data_dNdS.npz
|-- pipe_log.txt
|-- snpChart.csv
|-- snp_trees
|   |-- p_1041058_1.tree
|   |-- p_1054786_1.tree
|   |-- p_10866_1.tree
|   |-- p_1154874_1.tree
|   |-- ......
|-- snv_cov_scatter.png
|-- snv_filter_recombo.png
|-- snv_filter_sample_coverage_hist.png
|-- snv_filter_sample_toomanyNs_hist.png
|-- snv_qc_heatmap_calls.png
|-- snv_qc_heatmap_coverage.png
|-- snv_qc_heatmap_quals.png
|-- snv_table_cnn_plus_filter.txt
|-- snv_table_merge_all_mut_annotations_draft.tsv
|-- snv_table_merge_all_mut_annotations_final.tsv
|-- snv_table_merge_all_mut_annotations_label0.tsv
|-- snv_table_mutations_annotations_raw.tsv
|-- snv_table_with_charts_draft.html
|-- snv_table_with_charts_final.html
|-- snv_tree_genome_latest.nwk.tree
|-- snvs_histogram_per_sample.png
|-- snvs_per_sample.csv
`-- snvs_per_sample.png
```
### Core files:

| File or Folder |  Description |
| ---  | --- | 
| `snv_table_merge_all_mut_annotations_final.tsv`  | Final merged SNV report table (recommended primary text result for interpretation). More details, including explanations of the columns in this file, can be found here.
| `snv_table_cnn_plus_filter.txt` | Per-position prediction/filter summary table (CNN output + rule-based filters (from WideVariant)). Note that this file does not include annotation information for each SNV.
| `snv_table_with_charts_final.html`  | Interactive final HTML report for the final merged table (recommended to view). Keep `bar_charts/` in the same output folder so image links work.
| `candidate_mutation_table_final.npz`  | Final machine-readable SNV matrix for downstream analysis. Contains arrays such as sample names, genomic positions, counts, quality values, prediction labels/probabilities, and recombination flags. This is the main input for `accusnv_downstream`.

For final SNV calling results, please use:

`snv_table_merge_all_mut_annotations_final.tsv` as the primary human-readable SNV result table (final filtered/merged report).

`candidate_mutation_table_final.npz` as the machine-readable final result for any downstream analysis or re-analysis.

### Other files (include QC figures):

| File or Folder |  Description |
| ---  | --- | 
| `ZOOMED_snvs_histogram_per_sample.png`  | Histogram of SNV counts across samples. (Zoomed histogram version for easier viewing of the main range)
| `bar_charts`  |  One per-SNV bar chart (`p_<genome_pos>_bar_chart.png`) showing base support across samples at that position. Same format as those (e.g. Fig.1) described in the paper.
| `data_dNdS.npz`  | Saved dN/dS result bundle (e.g., dNdS, confidence interval bounds, mutation counts).
| `pipe_log.txt`  |  Full pipeline runtime log captured from script stdout/stderr-like messages; first place to inspect for warnings/errors and run statistics.
| `snpChart.csv`  | Intermediate chart file used as input for per-SNP tree annotation/export workflow.
| `snp_trees`  | Per-SNV tree files (`p_<pos>_<n>.tree`) derived from the main tree and SNV chart.
| `snv_cov_scatter.png`  | SNV position vs quality scatter plot used for SNV QC overview.
| `snv_filter_recombo.png`  | Recombination filtering plot (visual mark of retained vs suspected recombination-associated positions).
| `snv_filter_sample_coverage_hist.png`  | Sample-level coverage histogram with cutoff line (used to identify low-coverage samples).
| `snv_filter_sample_toomanyNs_hist.png`  | Sample-level ambiguous-call (N fraction) histogram with cutoff line.
| `snv_qc_heatmap_calls.png`  | Heatmap of per-sample calls across SNV positions.
| `snv_qc_heatmap_coverage.png`  | Heatmap of per-sample coverage across SNV positions
| `snv_qc_heatmap_quals.png`  | Heatmap of per-sample quality across SNV positions (quality axis labeled as FQ-derived quality).
| `snv_table_merge_all_mut_annotations_draft.tsv`  | Draft merged table combining annotation table + CNN/filter table.
| `snv_table_merge_all_mut_annotations_label0.tsv`  | Subset table for positions with `Pred_label = 0` (useful for reviewing filtered/negative calls).
| `snv_table_mutations_annotations_raw.tsv`  | Raw annotated SNV table before final label split/cleanup (contains positions passing core SNV logic and annotation fields).
| `snv_table_with_charts_draft.html`  | Interactive draft HTML report for the draft merged table; links each SNV to its bar chart image.
| `snv_tree_genome_latest.nwk.tree`  | Final genome-wide SNV phylogeny in Newick format.
| `snvs_histogram_per_sample.png`  | Histogram of SNV counts across samples.
| `snvs_per_sample.csv`  | Per-sample SNV count table (simple summary counts).
| `snvs_per_sample.png`  | Per-sample SNV count plot.


##  Output of Downstream analysis


The main output files of the **Quick Test** (Downstream analysis) are in the `cae_accusnv_pe_downstream` folder. The folder structure should look like this:

```
cae_accusnv_pe_downstream
|-- bar_charts
|   |-- p_1041058_bar_chart.png
|   |-- p_1054786_bar_chart.png
|   |-- p_10866_bar_chart.png
|   |-- p_1154874_bar_chart.png
|   |-- ......
|-- data_dNdS.npz
|-- snpChart.csv
|-- snp_trees
|   |-- p_1041058_1.tree
|   |-- p_1054786_1.tree
|   |-- p_10866_1.tree
|   |-- p_1154874_1.tree
|   |-- ......
|-- snv_cov_scatter.png
|-- snv_table_cnn_plus_filter.txt
|-- snv_table_mutations_annotations.tsv
|-- snv_table_with_charts_final.html
|-- snv_table_with_filters.tsv
`-- snv_tree_genome_latest.nwk.tree
```

| File or Folder |  Description |
| ---  | --- | 
| `bar_charts`  | Per-SNV bar charts linked by the HTML report.
| `data_dNdS.npz`  | dN/dS results (dNdS, CI, N/S mutation counts, etc.).
| `snpChart.csv`  | Intermediate SNP chart used for per-SNP tree export.
| `snp_trees`  | Per-SNP tree files for mutation-by-mutation tree context review.
| `snv_cov_scatter.png`  | Position vs quality scatter plot for downstream SNV QC view.
| `snv_table_cnn_plus_filter.txt`  | Prediction/filter summary table used in downstream merge step
| `snv_table_mutations_annotations.tsv`  | Main annotated SNV table produced by downstream analysis.
| `snv_table_with_charts_final.html`  | Final interactive downstream HTML report with bar-chart links.
| `snv_table_with_filters.tsv`  | Merged table combining annotations + filter/prediction fields (recommended downstream text table for review).
| `snv_tree_genome_latest.nwk.tree`  | Newick tree for the downstream run context.

You may ask: Given that there is so much overlap between the Snakemake pipeline and the results of downstream analysis, why is the downstream analysis step still necessary? Because:

The downstream step exists to let users re-analyze the same final NPZ quickly with different downstream choices (for example, recombination exclusion, position exclusion, and downstream reporting), without re-running the full alignment/calling workflow on HPC.

So, you can use downstream when you want fast iterative biological interpretation on the final callset (e.g., alternate filtering choices, recombination handling, or repeated dN/dS runs) without rerunning the full pipeline.

##  Other output of Snakemake pipeline


```
tree cae_pe_test_snakemake

|-- 1-Mapping
|   |-- bwa
|   |-- diversity
|   |-- quals
|   `-- vcf
|-- 2-Case
|   |-- candidate_mutation_table
|   `-- temp
|-- 3-AccuSNV
|-- conf
|   `-- config.yaml
|-- data
|   |-- strain1
|   |-- strain2
|   |-- strain3
|   `-- strain4
|-- link
`-- temp
```
`1-Mapping/` <BR>

Mapping-stage outputs generated from read alignment and pileup/VCF processing.
It contains intermediate files used to build candidate mutation tables in `2-Case`.

`1-Mapping/bwa/` or `1-Mapping/bowtie2/` <BR>

Per-sample alignment files (SAM/BAM/BAM index and related mapping intermediates).

`1-Mapping/vcf/` <BR>
Per-sample pileup and VCF outputs (raw and SNP-filtered), produced from mapped BAM files.

`1-Mapping/quals/` <BR>
Per-sample quality matrices (*.quals.pickle.gz) derived from VCF and used later in SNV filtering/calling.

`1-Mapping/diversity/` <BR>
Per-sample diversity and coverage matrices (*.diversity.pickle.gz, *.coverage.pickle.gz) used by the candidate mutation table step.

`2-Case/` <BR>
Candidate-SNV discovery stage before final AccuSNV output.
This stage aggregates per-sample mapping statistics into group-level candidate mutation tables.

`2-Case/candidate_mutation_table/` <BR>
Group-level candidate mutation tables (`group_<cladeID>_candidate_mutation_table.npz`) and optional coverage matrices, used as direct input to the final AccuSNV stage. (These files are the input to `new_snv_script.py`)

`2-Case/temp/` <BR>
Temporary per-sample/per-group files (position lists, string lists, prep files) created during candidate-SNV aggregation.

`3-AccuSNV/` <BR>
Final AccuSNV result folder (per group), including final SNV tables/HTML/plots/tree and `candidate_mutation_table_final.npz` for downstream analysis

`conf/` <BR>
Run-specific configuration folder generated in your output directory.
Contains the generated config.yaml used by Snakemake/profile execution.

`conf/config.yaml` <BR>
Main run config file (resources, profile behavior, experiment linkage).
This is the file users typically edit for cluster resource tuning.

`data/` <BR>
Per-sample working folders used by Snakemake execution (e.g., filtered read files and sample metadata files).

`data/strain1`, `data/strain2`, `data/strain3`, `data/strain4` <BR>
One folder per sample.
Each sample folder stores its own sample_info.csv and processed files used by downstream rules.

`link/` <BR>
Symlink staging folder for normalized input file names (e.g., <sample>_1.fastq.gz, <sample>_2.fastq.gz) created by accusnv_snakemake.py.
This makes input handling consistent regardless of original raw file naming/location

`temp/` <BR>
Temporary setup/utility files generated during initialization (e.g., rebuilt sample CSV, temporary experiment config files).
These are helper files for orchestrating the run and are not final biological outputs.
