"""The two log files AccuSNV writes for a run.

``accusnv.log`` holds the milestones, warnings and errors: what ran, what went into it and what
came out. ``accusnv.full.log`` holds all of that plus the per-sample detail and the output of
bwa/samtools/bcftools. Both live in the output directory (see log_file / full_log_file in
pipeline.yaml) and are appended to, so every step of a run lands in one place in the order it
happened, the same way whether the run was local or on SLURM.

An entry point calls setup() once with its step name. Library modules just do
``log = logging.getLogger('accusnv')`` and pick up whatever the entry point configured.
"""

import logging
import os
import sys


def setup(step, summary_path, full_path):
    """Point this step's messages at the two log files and the terminal.

    info() and above reach all three; debug() reaches only the full log. An uncaught exception
    is logged with its traceback, so a crash is recorded in the files and not just on screen.
    """
    log = logging.getLogger('accusnv')
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter(f'%(asctime)s  %(levelname)-7s [{step}] %(message)s', '%Y-%m-%d %H:%M:%S')
    for path, level in ((full_path, logging.DEBUG), (summary_path, logging.INFO), (None, logging.INFO)):
        if path:
            # Absolute, because the tree step chdirs into its own directory to run dnapars.
            path = os.path.abspath(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = logging.FileHandler(path) if path else logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(fmt)
        log.addHandler(handler)
    sys.excepthook = lambda *exc: log.error('%s failed', step, exc_info=exc)
    return log


def paths(outdir, params):
    """Where this run's two log files go, from log_file / full_log_file in pipeline.yaml.
    A relative path (the default) sits at the top level of the output directory."""
    return tuple(p if os.path.isabs(p) else os.path.join(outdir, p)
                 for p in (params.get('log_file', 'accusnv.log'),
                           params.get('full_log_file', 'accusnv.full.log')))


def add_args(parser):
    """Add the two log paths to a step's argument parser. The workflow always passes them."""
    parser.add_argument('--log', default=None, help='Summary log file')
    parser.add_argument('--full_log', default=None, help='Full log file')


if __name__ == '__main__':
    # One log line from a workflow rule that otherwise only runs external tools:
    #   python -m accusnv.log <step> <summary_log> <full_log> <message>
    step, summary, full, message = sys.argv[1:5]
    setup(step, summary, full).info(message)
