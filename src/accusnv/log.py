"""The log files AccuSNV writes for a run.

Every job writes its own pair of files in the ``logs`` directory of the output directory, named
for the rule and the sample or group it ran on. Snakemake only ever runs one job per rule and
sample, so each file has exactly one writer; jobs sharing a file would garble each other's output
locally and can lose lines outright on a cluster filesystem. merge() then concatenates them into
the run's two log files at the top of the output directory (see log_file / full_log_file in
pipeline.yaml). ``accusnv.log`` holds the milestones, warnings and errors: what ran, what went
into it and what came out. ``accusnv.full.log`` holds all of that plus the per-sample detail and
the output of bwa/samtools/bcftools.

An entry point calls setup() once with its step name and the stem of its own two files. Library
modules just do ``log = logging.getLogger('accusnv')`` and pick up whatever the entry point
configured.
"""

import glob
import logging
import os
import re
import sys
import time


def setup(step, stem):
    """Point this step's messages at its own two log files and the terminal.

    info() and above reach all three; debug() reaches only <stem>.full.log. An uncaught exception
    is logged with its traceback, so a crash is recorded in the files and not just on screen.
    """
    log = logging.getLogger('accusnv')
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter(f'%(asctime)s  %(levelname)-7s [{step}] %(message)s', '%Y-%m-%d %H:%M:%S')
    if stem:
        # Absolute, because the tree step chdirs into its own directory to run dnapars.
        stem = os.path.abspath(stem)
        os.makedirs(os.path.dirname(stem), exist_ok=True)
    files = ((stem + '.full.log', logging.DEBUG), (stem + '.log', logging.INFO)) if stem else ()
    for path, level in files + ((None, logging.INFO),):
        # delay, so a step that logs nothing at this level leaves no empty file behind. Appending,
        # so that a job retried at a higher resource tier keeps the failed attempt above it.
        handler = logging.FileHandler(path, delay=True) if path else logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(fmt)
        log.addHandler(handler)
    sys.excepthook = lambda *exc: log.error('%s failed', step, exc_info=exc)
    return log


def stem(outdir, step, tag=None):
    """Where one job writes its two log files: <outdir>/logs/<step>-<tag>.log and .full.log.
    The tag names the sample or group the job ran on, as in that job's output filenames."""
    return os.path.join(outdir, 'logs', f'{step}-{tag}' if tag else step)


def paths(outdir, params):
    """Where this run's two merged log files go, from log_file / full_log_file in pipeline.yaml.
    A relative path (the default) sits at the top level of the output directory."""
    return tuple(p if os.path.isabs(p) else os.path.join(outdir, p)
                 for p in (params.get('log_file', 'accusnv.log'),
                           params.get('full_log_file', 'accusnv.full.log')))


def merge(outdir, params):
    """Concatenate the per-job logs into the run's two log files, oldest job first.

    Whole files rather than single lines, so each job's messages and the tool output they describe
    stay together. The per-job files are left where they are, so one failed sample can still be
    read on its own, and the merge can be run again at any point in the run.
    """
    summary_path, full_path = paths(outdir, params)
    found = glob.glob(os.path.join(outdir, 'logs', '*.log'))
    full = [f for f in found if f.endswith('.full.log')]
    for merged, files, named in ((summary_path, [f for f in found if f not in full], False),
                                 (full_path, full, True)):
        with open(merged, 'w') as out:
            for path in sorted(files, key=_started):
                if named:
                    # Tool output carries nothing of its own saying which job it came from.
                    out.write('--- %s ---\n' % re.sub(r'\.full\.log$', '', os.path.basename(path)))
                out.write(open(path).read())


def _started(path):
    """When the job that wrote this file started, for ordering the merge: the timestamp its first
    line begins with, or the file's modification time if it holds only untimestamped tool output.
    Jobs that started within the same second are then ordered by when they last wrote."""
    mtime = os.path.getmtime(path)
    first = open(path).readline()[:19]
    if not re.match(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d', first):
        first = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
    return first, mtime


def add_args(parser):
    """Add this step's log stem to its argument parser. The workflow always passes it."""
    parser.add_argument('--log', default=None, help='Stem for this step\'s two log files')


if __name__ == '__main__':
    # One log line from a workflow rule that otherwise only runs external tools:
    #   python -m accusnv.log <log_stem> <message>
    # The step is the part of the stem's name before the sample or group the job ran on.
    log_stem, message = sys.argv[1:3]
    setup(os.path.basename(log_stem).split('-')[0], log_stem).info(message)
