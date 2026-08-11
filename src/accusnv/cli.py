#!/usr/bin/env python3
"""AccuSNV pipeline launcher and argparse.

This file parses inputs, writes configs, and runs Snakemake.

  accusnv -m {dryrun,slurm,local} -i samples.csv -r ./ref_dir/ -o out/ [-c config.yaml] [-p pipeline.yaml]

Two config files are required for a run:
  1. config.yaml   - Snakemake execution settings (executor, jobs, resources, partition) plus
                    the run paths (sample_table, outdir).
  2. pipeline.yaml - pipeline parameters the workflow reads (aligner, cutoffs, filters).

These can be auto-generated with defaults under <out_dir>/configs, and then edited, or you can pass your own 
with -c and -p. Edit those files to change any pipeline or execution parameter.
"""

import os
import re
import sys
import glob
import yaml
import shutil
import getpass
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from importlib.resources import files
from accusnv import __version__

from accusnv import log as accusnv_log
from accusnv.workflow import resources as tiers
from accusnv.preprocessing.utils import resolve_fasta_path

log = logging.getLogger('accusnv')

SNAKEFILE = str(files("accusnv").joinpath("workflow", "Snakefile"))
FLAG_KEYS = ['skip_recombination', 'skip_report', 'skip_dnds', 'skip_trees', 'skip_samclip',
         'build_snv_trees', 'skip_all_downstream', 'downstream_only']

# Dependencies every run needs. dnapars is checked separately for tree building.
TOOL_DEPS = ['snakemake', 'samtools', 'bcftools', 'cutadapt', 'sickle', 'tabix']

################# CONFIG FILES #################

def fail(run_parser, message):
    """Record the problem in the log, then let argparse print it and exit."""
    log.error(message)
    run_parser.error(message)


def guess_slurm_account():
    """The SLURM account to submit under, or None to leave it to Snakemake.

    Snakemake's SLURM plugin guesses the account from your recent jobs, but it hands sacct a shell
    pipe as arguments, which older versions of sacct reject ("sacct: invalid option -- '1'"). So we
    first run the plugin's own command: if it works the plugin can look after itself, and only if it
    fails do we repeat the lookup properly and take the most recent account sacct reports.
    """
    user = getpass.getuser()
    try:
        if subprocess.run(['sacct', '-nu', user, '-o', 'Account%256', '|', 'tail', '-1'],
                          capture_output=True, timeout=60).returncode == 0:
            return None
        out = subprocess.run(['sacct', '-nu', user, '-o', 'Account%256'],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    accounts = [a for a in (line.strip() for line in out.splitlines())
                if a not in ('', 'none', '(null)')]
    return accounts[-1] if accounts else None


def describe_failure(block, started):
    """One line for the AccuSNV log about a failed job: which one, why, and at which tier.

    `block` is the fields Snakemake printed under 'Error in rule ...', `started` the ones it
    printed when the job began, which is where the wildcards and the requested resources are.
    """
    who = block['rule'] + (f" ({started['wildcards']})" if started.get('wildcards') else '')
    # On SLURM the executor plugin reports the state as its failure message; a local run has none.
    status = re.search(r"SLURM status is: '(\w+)'", block.get('message', ''))
    why = {'TIMEOUT': 'ran out of time', 'OUT_OF_MEMORY': 'ran out of memory'}.get(
        status.group(1) if status else '', 'failed')

    asked = dict(re.findall(r'(\w+)=(\d+)', started.get('resources', '')))
    tier_list = tiers.TIERS.get(block['rule'], [])
    tier = (int(asked.get('mem_mb', 0)), int(asked.get('runtime', 0)))
    if tier not in tier_list:
        return f'{who} {why}'
    n = tier_list.index(tier) + 1
    at = f'{who} {why} at tier {n} of {len(tier_list)} ({tier[0]} MB, {tier[1]} min)'
    return at if n < len(tier_list) else (at + ', the largest tier there is, so a retry asks for '
                                          'exactly the same again. Raise it in resources.py')


def run_snakemake(cmd, snakemake_log):
    """Run Snakemake, copying its output to the terminal and to accusnv.snakemake.log, and
    summarising each job failure in the AccuSNV log. Returns Snakemake's exit code."""
    block, failed, started = None, False, {}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(snakemake_log, 'a') as tee:
        for line in proc.stdout:
            sys.stdout.write(line)
            tee.write(line)
            # Snakemake prints one block per job, headed 'rule x:' (or 'localrule x:', or
            # 'Error in rule x:') and followed by indented 'key: value' lines. Anything
            # unindented ends the block. A retry reprints the header, so the resources we
            # remember for a job are always the ones the failing attempt asked for.
            field = re.match(r'\s+(\w+):\s*(.*?)\s*$', line) if block else None
            if field:
                block[field.group(1)] = field.group(2)
                continue
            if block and failed:
                log.error(describe_failure(block, started.get(block.get('jobid'), {})))
            elif block and 'jobid' in block:
                started[block['jobid']] = block
            header = re.match(r'(?:local)?(?:rule|checkpoint) (\w+):|Error in rule (\w+):', line)
            block = {'rule': header.group(1) or header.group(2)} if header else None
            failed = bool(header and header.group(2))
    return proc.wait()


def pipeline_params(args):
    """Load the pipeline parameter yaml for this run
    (the user's -p file or the packaged default template)."""
    if args.pipeline_file and os.path.exists(args.pipeline_file):
        with open(args.pipeline_file) as f:
            return yaml.safe_load(f) or {}
    return yaml.safe_load(files("accusnv").joinpath("templates", "pipeline.yaml").read_text(encoding="utf-8")) or {}


def create_configs(args, run_parser, conf_dir):
    """Write the runtime copies of pipeline.yaml + config.yaml into <out>/configs.
    pipeline.yaml carries the workflow parameters
    config.yaml is the Snakemake execution profile and
    also carries the run inputs (sample_table / outdir)
    under the 'config:' block, which Snakemake merges into the workflow config."""

    ## Find or create pipeline file
    if args.pipeline_file:
        if not os.path.exists(args.pipeline_file):
            fail(run_parser, f'--pipeline_file not found: {args.pipeline_file}')
        pipeline_file = os.path.abspath(args.pipeline_file)
        log.info('Reading pipeline parameters from %s', pipeline_file)
    else:
        pipeline_file = os.path.join(conf_dir, "pipeline.yaml")
        with open(pipeline_file, 'w') as f:
            f.write(files("accusnv").joinpath("templates", 'pipeline.yaml').read_text(encoding="utf-8"))
        log.info('No pipeline.yaml given; wrote one with the default parameters to %s', pipeline_file)

    ## Find or create config file
    if args.config_file:
        if not os.path.exists(args.config_file):
            fail(run_parser, f'--config_file not found: {args.config_file}')
        config_file = os.path.abspath(args.config_file)
        log.info('Reading Snakemake execution settings from %s', config_file)
    else:
        config_file = os.path.join(conf_dir, "config.yaml")
        with open(config_file, 'w') as f:
            f.write(files("accusnv").joinpath("templates", 'config.yaml').read_text(encoding="utf-8"))
        log.info('No config.yaml given; wrote one with the default execution settings to %s', config_file)

    ## pipeline.yaml: apply the --skip_* / --build_* toggles (everything else is edited in-file)
    with open(pipeline_file) as f:
        pipeline_config = yaml.safe_load(f) or {}

    for flag in FLAG_KEYS:
        if getattr(args, flag):
            log.info('Command line sets %s', flag)
            pipeline_config[flag] = True

    # Both position lists travel as absolute paths, because the workflow jobs run from the output
    # directory rather than from where the user typed the command.
    for flag, verb in (('exclude_positions', 'Leaving out'), ('include_positions', 'Keeping')):
        path = getattr(args, flag)
        if path:
            if not os.path.exists(path):
                fail(run_parser, f'--{flag} not found: {path}')
            pipeline_config[flag] = os.path.abspath(path)
            log.info('%s the SNV positions listed in %s (each group logs how many it applied)',
                     verb, path)

    # One core setting for the whole run: it is what cutadapt and the aligner are given, what
    # SLURM is asked for as cpus-per-task for those two rules, and the budget of a local run.
    cores = args.cores or pipeline_config.get('cores') or 4
    pipeline_config['cores'] = cores

    pipeline_file_new = os.path.join(conf_dir, "pipeline.yaml")
    log.debug('Writing the pipeline parameters this run will use to %s', pipeline_file_new)
    with open(pipeline_file_new, 'w') as f:
        yaml.safe_dump(pipeline_config, f, sort_keys=False, default_flow_style=False)

    ## config.yaml: point at pipeline.yaml, inject the run paths, apply the run mode
    with open(config_file) as f:
        config = yaml.safe_load(f) or {}

    # Snakemake reads `config:` (a list of KEY=VALUE) as --config and merges it into the
    # workflow config dict, so the run paths live in config.yaml yet reach the Snakefile.
    existing = {}
    for item in (config.get('config') or []):
        if '=' in str(item):
            k, v = str(item).split('=', 1)
            existing[k] = v
    # -e overrides; if not passed, keep whatever env is already in the config.yaml. The unfilled
    # template placeholder ("[passed by accusnv -e]") counts as unset -> empty.
    existing_env = existing.get('env', '')
    if str(existing_env).strip().startswith('[passed by accusnv'):
        existing_env = ''

    run_params = {
        'sample_table': os.path.join(os.path.abspath(args.output_dir), 'samples.csv'),
        'outdir':       os.path.abspath(args.output_dir),
        'env':          args.env if args.env is not None else existing_env,
    }
    config['config'] = [f'{k}={v}' for k, v in run_params.items()]

    if args.partition and args.mode != 'slurm':
        log.warning('Ignoring --partition %r: it only applies in slurm mode', args.partition)

    if args.mode == 'slurm':
        config['executor'] = 'slurm'
        config.setdefault('jobs', 100)
        config.pop('cores', None)
        # --partition sets slurm_partition in default-resources; if omitted, none is set and
        # sbatch runs with the cluster's default partitions. slurm_account is only filled in if the
        # config does not already give one, i.e. if the user has not uncommented that line.
        resources = {}
        for item in (config.get('default-resources') or []):
            if '=' in str(item):
                k, v = str(item).split('=', 1)
                resources[k] = v
        if args.partition:
            resources['slurm_partition'] = f'"{args.partition}"'
        if 'slurm_account' not in resources:
            account = guess_slurm_account()
            if account:
                log.info('Submitting under SLURM account %s, guessed from your recent jobs', account)
                resources['slurm_account'] = f'"{account}"'
        config['default-resources'] = [f'{k}={v}' for k, v in resources.items()]

    else:
        # local/dryrun: the template ships the SLURM executor, so switch it back and drop
        # the SLURM-only default resources, which the local executor does not understand.
        config['executor'] = 'local'
        config.pop('jobs', None)
        # Snakemake caps every rule's threads at this, so it has to match the cores the
        # cutadapt and mapping rules ask for or they would quietly run narrower.
        config['cores'] = cores
        config['default-resources'] = [item for item in (config.get('default-resources') or [])
                                       if not str(item).startswith('slurm_')]
        if args.mode == 'dryrun':
            config['dryrun'] = True

    # unfortunate naming, but this is what snakemake calls it
    config['configfile'] = os.path.abspath(pipeline_file_new)

    config_file_new = os.path.join(conf_dir, "config.yaml")
    log.debug('Writing the execution settings this run will use to %s', config_file_new)
    with open(config_file_new, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

    return pipeline_file_new, config_file_new

################# CHECK INPUTS AND DEPENDENCIES #################

def check_inputs_and_dependencies(args, run_parser):
    ''' Validates the existence of all input files and dependencies before run'''

    ## aligner lives in pipeline.yaml
    aligner = pipeline_params(args).get('aligner') or 'bwa'

    env = None
    if args.env is not None:
        env = args.env
    if args.config_file and os.path.exists(args.config_file):
        cfg = yaml.safe_load(open(args.config_file).read())
        for item in (cfg.get('config') or []):
            if str(item).startswith('env='):
                if str(item).split("=")[1] != '':
                    env = str(item).split('=')[1]

    ## check for dependencies
    dependencies = list(TOOL_DEPS) + [aligner]
    # Only a parsimony tree needs dnapars. skip_trees and skip_all_downstream are also command-line
    # flags; use_nj_tree is set in pipeline.yaml only.
    params = pipeline_params(args)
    flag_on = lambda flag: getattr(args, flag, False) or params.get(flag, False)
    if not (flag_on('skip_trees') or flag_on('skip_all_downstream') or flag_on('use_nj_tree')):
        dependencies.append('dnapars')
    # samclip only runs on bwa alignments, and only when it has not been turned off.
    if aligner == 'bwa' and not flag_on('skip_samclip'):
        dependencies.append('samclip')

    if not env:
        missing_tools = []
        found_tools = []

        for dependency in dependencies:
            if shutil.which(dependency) is None:
                missing_tools.append(dependency)
            else:
                found_tools.append(dependency)

        if len(missing_tools) > 0:
            fail(run_parser, "Error: Could not find dependencies: {}".format(", ".join(missing_tools)))
        else:
            log.info('Found every required tool: %s', ', '.join(found_tools))
    else:
        log.info('Not checking for tools: every job runs %r first, so they are expected to be '
                 'available there', env)

    ## check input metadata
    if not args.input_sample_info:
        fail(run_parser, 'Error: -i/--input_sample_info is required')
    if not args.ref_dir:
        fail(run_parser, 'Error: -r/--ref_dir is required')

    if not os.path.exists(args.input_sample_info):
        fail(run_parser, "Error: Could not find sample_file: {}".format(args.input_sample_info))

    sample_file = pd.read_csv(args.input_sample_info)

    ## a sample may be listed more than once (e.g. as the outgroup of several groups), but only
    ## once per group, and every row of it must point at the same reads.
    for sample, rows in sample_file.groupby('Sample'):
        if rows['Group'].duplicated().any():
            fail(run_parser, "Error: sample {} is listed more than once in the same group".format(sample))
        if len(rows[['FileName', 'Path', 'Type']].drop_duplicates()) > 1:
            fail(run_parser, "Error: sample {} is listed more than once with different FASTQ files or Type. Give the rows different sample names if they are different samples.".format(sample))

    ## check fastqs
    ## read sample file
    fwd_paths = []
    rev_paths = []

    for index, row in sample_file.iterrows():
        # Only the reads themselves. The glob also picks up sidecars sitting next to them, and
        # those must not count towards the SE ambiguity check below; matching on the extension
        # rather than a substring keeps out the ones that embed a read name (S1.fastq.gz.md5).
        fastqs = [f for f in glob.glob(row['Path'].rstrip("/") + "/" + row['FileName'] + "*")
                  if f.endswith(('.fastq', '.fq', '.fastq.gz', '.fq.gz'))]
        fwd, rev = None, None

        for fastq in fastqs:
            if row['Type'] == 'PE' or row['Type'] == 'pe':
                if '_R1.f' in fastq or '.1.f' in fastq or '_1.f' in fastq:
                    if not fwd:
                        fwd = fastq
                    else:
                        fail(run_parser, "Error: two possible forward fastq files found for sample {}: {}".format(row['FileName'], fastqs))
                elif '_R2.f' in fastq or '.2.f' in fastq or '_2.f' in fastq:
                    if not rev:
                        rev = fastq
                    else:
                        fail(run_parser, "Error: two possible reverse fastq files found for sample {}: {}".format(row['FileName'], fastqs))
            elif row['Type'] == 'SE' or row['Type'] == 'se':
                if len(fastqs) > 1:
                    fail(run_parser, "Error: for SE sample {}, multiple possible FASTQs found: {}".format(row['FileName'], fastqs))
                fwd = fastq

        # Store absolute paths so the workflow resolves them regardless of its --directory.
        if row['Type'] == 'SE' or row['Type'] == 'se':
            if not fwd:
                fail(run_parser, "Error: could not find a single FASTQ file for sample: {} in directory {}: {}".format(row['FileName'], row['Path'], fastqs))
            fwd_paths.append(os.path.abspath(fwd))
            rev_paths.append('')                       # SE: no reverse read
        else:
            if not fwd or not rev:
                fail(run_parser, "Error: could not find both a forward and reverse read for sample {} in directory {}: {}".format(row['FileName'], row['Path'], fastqs))
            fwd_paths.append(os.path.abspath(fwd))
            rev_paths.append(os.path.abspath(rev))

    sample_file['Read1_file_path'] = fwd_paths
    sample_file['Read2_file_path'] = rev_paths


    ## check refs
    # resolve_fasta_path is what the workflow stages use too, so a reference resolves to the
    # same file here and downstream: any of .fasta/.fa/.fna (optionally .gz), genome.* first.
    fastas = []
    for index, row in sample_file.iterrows():
        ref_dir = os.path.join(args.ref_dir, row['Reference'])
        try:
            fastas.append(os.path.abspath(resolve_fasta_path(ref_dir)))
        except ValueError as e:
            fail(run_parser, "Error: {}".format(e))

    sample_file['Reference_FASTA'] = fastas

    # save sample file (index=False so the workflow reads clean, named columns)
    new_sample_file = os.path.join(args.output_dir, 'samples.csv')
    log.info('Found the FASTQs and reference genome for all %d samples across %d group(s); '
             'references: %s', len(sample_file), sample_file['Group'].nunique(),
             ', '.join(sorted(sample_file['Reference'].unique())))
    log.debug('Wrote the resolved sample sheet to %s', new_sample_file)
    sample_file.to_csv(new_sample_file, index=False)


    ## check if downstream only
    if args.downstream_only:
        for group in sample_file['Group'].unique():
            # _snv_state.npz is what the annotation and analysis stages actually load; the
            # calling stage that writes it is not re-run in this mode.
            group_dir = os.path.join(args.output_dir, '2-SNV-filtering', 'group_' + str(group))
            state_file = os.path.join(group_dir, '_snv_state.npz')
            if not os.path.exists(state_file):
                fail(run_parser, "Error: You are running in --downstream_only mode, but the SNV calling output for group {} was not found: {}".format(group, state_file))
            # Annotation also needs the filter verdicts. Checking here rather than letting the
            # rule fail matters: a failed rule makes Snakemake delete the tables it was
            # rewriting, so the output directory would lose the results it already had.
            filter_table = os.path.join(group_dir, 'snv_table_filtered_tmp.tsv')
            state = np.load(state_file, allow_pickle=True)
            if not bool(state['fast_path']) and not os.path.exists(filter_table):
                fail(run_parser, "Error: You are running in --downstream_only mode, but {} is missing, so group {} cannot be re-annotated. Output directories written by AccuSNV before 1.1 deleted this file. Re-run without --downstream_only.".format(filter_table, group))

    return new_sample_file

################# TERMINAL COLORS #################

_SECTION_COLORS = {
    'Inputs and Outputs': 149,    
    'Config': 168,                
    'Output options': 222,        
    'options': 250,               
    'positional arguments': 159,
}

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')

def _color_on():
    """Styles only when writing to a terminal with color."""
    return sys.stdout.isatty() and os.environ.get('NO_COLOR') is None and os.environ.get('TERM') != 'dumb'

def _bold(s):
    """Bold `s` on a color-capable terminal, else return it unchanged."""
    return f'\033[1m{s}\033[0m' if _color_on() else s

def _colorize(text):
    """Adds pretty colors to help."""

    if not _color_on():
        return text
    cur, out = None, []
    for ln in text.split('\n'):
        body = ln.lstrip(' ')
        indent = ln[:len(ln) - len(body)]
        plain = _ANSI_RE.sub('', body)                            # match on text minus any ANSI
        if not indent and plain.endswith(':') and plain[:-1] in _SECTION_COLORS:
            cur = _SECTION_COLORS[plain[:-1]]                      # heading: switch color
            out.append(f"{indent}\033[1;38;5;{cur}m{plain}\033[0m")  # bold + colored
        elif cur is not None and body and len(indent) <= 4:        # an entry line (small indent)
            gap = body.find('  ')                                  # split entry name from its help
            name, rest = (body, '') if gap < 0 else (body[:gap], body[gap:])
            out.append(f"{indent}\033[38;5;{cur}m{name}\033[0m{rest}")
        else:
            out.append(ln)                                         # usage, description, wrapped help
    return '\n'.join(out)

################# ARGUMENT PARSING #################

class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keep multi-line descriptions raw, and keep each option's help on the same
    line as the flag (don't wrap it to the next line for longer flags)."""
    def __init__(self, *args, **kwargs):
        kwargs['max_help_position'] = 40
        super().__init__(*args, **kwargs)

class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose help drops the lone -h `options:` block and is colorized."""
    def format_help(self):
        text = super().format_help()
        text = re.sub(r'  -h, --help[^\n]*\n', '', text)
        return _colorize(text)

def build_parser():
    """Build the (single, flat) `accusnv` argument parser."""

    parser = _Parser(
        prog='accusnv',
        formatter_class=_HelpFormatter,
        description=f'{_bold("AccuSNV")} v{__version__}\n'
                    'High-accuracy SNV calling for bacterial isolates using deep learning.\n')
    parser.add_argument('--version', action='version', version=f'AccuSNV {__version__}',
                        help='Show version and exit')

    inputs = parser.add_argument_group('Inputs and Outputs')
    inputs.add_argument('-i', '--input_sample_info', metavar='CSV', help='Input sample CSV (required)')
    inputs.add_argument('-r', '--ref_dir', metavar='DIR', help='Reference genomes dir (required)')
    inputs.add_argument('-o', '--output_dir', metavar='DIR', default='accusnv_output',
                      help='Output dir (default: accusnv_output)')
    inputs.add_argument('--exclude_positions', metavar='FILE', default=None,
                      help='File of genome_pos values, one per line, to leave out of the SNV set \
                      (also written to pipeline.yaml). Combine with --downstream_only to change \
                      the list without re-calling SNVs')
    inputs.add_argument('--include_positions', metavar='FILE', default=None,
                      help='File of genome_pos values, one per line, to keep in the SNV set \
                      whatever the model and filters decided. Excluding wins if a position is in \
                      both files')

    configs = parser.add_argument_group('Config')
    configs.add_argument('-c', '--config_file', metavar='FILE', default=None,
                       help='Execution settings + run paths (config.yaml; default: autogenerated in out_dir)')
    configs.add_argument('-p', '--pipeline_file', metavar='FILE', default=None,
                       help='Pipeline params (pipeline.yaml; default: autogenerated in out_dir)')
    configs.add_argument('-m', '--mode', choices=['dryrun', 'slurm', 'local'], default='local',
                      help='Run mode (default: local)')
    configs.add_argument('-j', '--cores', metavar='N', type=int, default=None,
                      help='Cores per cutadapt/mapping job, and the total cores a local run may \
                      use (default: the cores value in pipeline.yaml, 4)')
    configs.add_argument('-sp', '--partition', metavar='PARTITIONS', default=None,
                      help='SLURM partition(s) to submit to, comma-separated list for >1 (default: not specified)')
    configs.add_argument('--skip_samclip', action='store_true',
                      help='Do not pipe bwa alignments through samclip, which drops soft-clipped \
                      reads (on by default with bwa; samclip is never used with bowtie2)')
    configs.add_argument('-e', '--env', metavar='CMD', default=None,
                      help='Environment activation command, e.g. \'conda activate accusnv\' \
                      (default: inherit current environment)')

    outputs = parser.add_argument_group('Output options')
    outputs.add_argument('--skip_all_downstream', action='store_true',
                       help='Skip all downstream evolutionary analyses and report generation')
    outputs.add_argument('--skip_report', action='store_true', help='Skip generating the HTML report')
    outputs.add_argument('--skip_recombination', action='store_true',
                       help='Skip recombination detection. On by default: recombinant SNVs are flagged and kept \
                       in the SNV tables, but left out of dN/dS, tree building and dMRCA. Pass this to stop \
                       flagging them, which counts them everywhere')
    outputs.add_argument('--skip_dnds', action='store_true', help='Skip dN/dS calculations')
    outputs.add_argument('--skip_trees', action='store_true', help='Skip parsimony tree building (dnapars)')
    outputs.add_argument('--build_snv_trees', action='store_true',
                       help='Write one NEXUS tree per SNV, tips coloured by basecall, for viewing in FigTree (default: off)')
    outputs.add_argument('--downstream_only', action='store_true',
                       help='Run only downstream analyses (Assumes AccuSNV SNV tables and output directory already exist)')

    # Build usage from every action except those in the outputs group
    visible = [a for a in parser._actions if a.container is not outputs]
    fmt = parser._get_formatter()
    fmt.add_usage(None, visible, parser._mutually_exclusive_groups, prefix='')
    parser.usage = fmt.format_help().strip() + ' [output options]'
    return parser

################# MAIN PIPELINE #################
def run_pipeline(args, run_parser):

    ## Load output dir
    out_dir = os.path.abspath(args.output_dir)
    existed = os.path.isdir(out_dir)
    if args.downstream_only and not existed:
        # Nothing to re-run, and nowhere to write a log saying so.
        run_parser.error("Error: could not find existing directory for downstream_only run: {}".format(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    # Open this process's own log files before anything else that can fail, so the reason is
    # recorded. Every workflow job writes its own pair alongside them; merge() at the end of the
    # run gathers them all into summary_log and full_log.
    params = pipeline_params(args)
    summary_log, full_log = accusnv_log.paths(out_dir, params)
    accusnv_log.setup('accusnv', accusnv_log.stem(out_dir, 'accusnv'))
    log.info('AccuSNV starting in %s mode, writing to %s%s', args.mode, out_dir,
             ' (which already exists)' if existed else '')
    if args.downstream_only:
        log.info('Re-running only the downstream analyses on the existing output directory')

    ## Check for inputs and dependencies
    new_sample_file = check_inputs_and_dependencies(args, run_parser)

    ## Create and update config files with cli params
    conf_dir = os.path.join(out_dir, 'configs')
    os.makedirs(conf_dir, exist_ok=True)
    pipeline_file, config_file = create_configs(args, run_parser, conf_dir)

    ### Run snakemake pipeline
    cmd = ['snakemake', '--snakefile', SNAKEFILE, '--workflow-profile', conf_dir, '--directory', out_dir]
    if args.downstream_only:
        # Annotation's input is marked ancient() in this mode so the calling stage is not re-run,
        # which also stops Snakemake re-running annotation itself. Ask for it explicitly, or it
        # reports nothing to be done and any edited cutoffs are ignored. dN/dS, the tree and the
        # dashboard then follow from annotation's rewritten tables.
        cmd += ['--forcerun', 'annotate_snvs']
    snakemake_log = os.path.join(os.path.dirname(summary_log), 'accusnv.snakemake.log')
    log.info('Handing the run to Snakemake; everything it prints is kept in %s', snakemake_log)
    log.debug('Snakemake command: %s', ' '.join(cmd))
    # Everything this process says from here on is the wrap-up, so it goes to a second pair of
    # files. They are written after the jobs are done and so land at the end of the merged logs,
    # rather than back in the block holding the message the run opened with.
    accusnv_log.setup('accusnv', accusnv_log.stem(out_dir, 'accusnv', 'finished'))
    try:
        results = run_snakemake(cmd, snakemake_log)
    except Exception as e:
        log.error('Could not start Snakemake: %s', e)
        results = 1
    else:
        if results != 0:
            log.error('Pipeline FAILED (Snakemake exit code %d). The failed jobs are named above; '
                      'the whole Snakemake run is in %s and what the tools printed is in %s. Each '
                      'job also has its own log in %s', results, snakemake_log, full_log,
                      os.path.join(out_dir, 'logs'))
            if args.mode == 'slurm':
                log.error('If SLURM refused the jobs over the account (no account given, invalid '
                          'account), uncomment the slurm_account line under default-resources in %s '
                          'and set it to an account you can submit under.', config_file)
        else:
            log.info('Pipeline completed successfully. Summary: %s  Full detail: %s', summary_log, full_log)

    # Last, so that this process's own messages end up in the merged files as well.
    accusnv_log.merge(out_dir, params)
    return results


def main():
    parser = build_parser()
    argv = sys.argv[1:]

    ## show help if no args or -h
    if not argv or argv[0] in ('-h', '--help'):
        print(parser.format_help())
        return 0

    args = parser.parse_args(argv)
    return run_pipeline(args, parser)


if __name__ == '__main__':
	sys.exit(main())
