"""AccuSNV downstream analysis: the stages run after accusnv.accusnv's candidate calling.

snv.py is the shared kernel (data structures, reference/GFF parsing, filters, state
rebuild) used by every stage below it:
annotate (stage 2) -> recombination / dnds / generate_dashboard / tree_building
Each stage is runnable as ``python -m accusnv.downstream.<stage>`` and chained by Snakemake.
"""
