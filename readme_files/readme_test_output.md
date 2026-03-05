# Output files of AccuSNV


##  Output of Snakemake pipeline

The main output files of the Quick Test (Snakemake pipeline) are in the cae_pe_test_snakemake/3-AccuSNV/group_pe_test folder. The folder structure should look like this:

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

##  Output of Downstream analysis

The main output files of the Quick Test (Downstream analysis) are in the cae_accusnv_pe_downstream folder. The folder structure should look like this:

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