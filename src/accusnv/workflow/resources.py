"""resources.py - memory and runtime for AccuSNV jobs.
Every job starts in the first (smallest) tier. If it OOMs or times out, Snakemake resubmits it
with a higher `attempt`, which moves it to the next tier up.
"""

TIERS = {
    'create_mapping_index':     [(2000, 60), (8000, 60), (320000, 240)],
    'mapping':                  [(8000, 120), (32000, 120), (64000, 240)], ## medium amount of RAM
    'cutadapt':                 [(2000, 30), (8000, 60), (32000, 240)],
    'sickle':                   [(2000, 30), (8000, 60), (32000, 240)],
    'sam2bam':                  [(2000, 30), (4000, 60), (32000, 240)],
    'mpileup2vcf':              [(1000, 30), (4000, 60), (64000, 240)],
    'vcf2quals':                [(2000, 30), (4000, 60), (64000, 240)],
    'variants2positions':       [(1000, 30), (4000, 60), (64000, 240)],
    'pileup2diversity':         [(8000, 30), (32000, 60), (128000, 240)],
    'combine_positions':        [(1000, 30), (4000, 60), (32000, 240)],
    'candidate_mutation_table': [(32000, 30), (64000, 60), (128000, 240)], ## can use a LOT of RAM
    'calling_accusnv':          [(16000, 30), (64000, 60), (128000, 240)], ## can use a decent amount of RAM
    'annotate_snvs':            [(4000, 30), (16000, 60), (64000, 240)],
    'dnds':                     [(2000, 30), (16000, 60), (64000, 240)],
    'build_tree':               [(2000, 30), (16000, 60), (64000, 240)],
    'report_html':              [(2000, 30), (4000, 60), (32000, 240)],
}


def pick(rule, attempt=1):
    """Return (mem_mb, runtime_min) for this attempt: tier 1 first, then bump up each retry."""
    tiers = TIERS[rule]
    return tiers[min(attempt - 1, len(tiers) - 1)]


def mem(rule, attempt=1):
    return pick(rule, attempt)[0]


def runtime(rule, attempt=1):
    return pick(rule, attempt)[1]
