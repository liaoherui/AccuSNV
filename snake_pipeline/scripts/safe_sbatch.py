#!/usr/bin/env python3
"""Thin wrapper around ``sbatch`` that retries when submission limits are hit.

The ``QOSMaxSubmitJobPerUserLimit`` error surfaces on clusters when the
combination of running and pending jobs exceeds the QoS allowance.  Snakemake
stops immediately on that non-zero exit code, interrupting the workflow.

This wrapper catches that specific error, waits for a configurable delay, and
retries the submission.  Optional environment variables allow users to tune the
behaviour:

``ACCUSNV_SUBMIT_LIMIT``
    Upper bound on total jobs (``squeue`` rows) to allow before attempting a
    submission.  When present the script waits until the user's queue drops
    below this number before calling ``sbatch``.

``ACCUSNV_SUBMIT_RETRY_DELAY``
    Seconds to sleep before retrying a submission after the QoS limit is
    encountered.  Defaults to 60 seconds.

``ACCUSNV_SUBMIT_MAX_RETRIES``
    Maximum number of retries before giving up.  Defaults to 0 which means to
    retry indefinitely.  Positive numbers cap the number of retries.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Iterable, List


QOS_LIMIT_TOKEN = "QOSMaxSubmitJobPerUserLimit"
DEFAULT_RETRY_DELAY = 60


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _current_job_count(user: str | None) -> int:
    """Return the number of jobs owned by ``user`` according to ``squeue``."""

    if not user:
        return 0

    proc = subprocess.run(
        ["squeue", "--noheader", "-u", user],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        # If we cannot query the queue, fall back to not rate-limiting.
        return 0

    # Each non-empty line corresponds to one job.
    return sum(1 for line in proc.stdout.splitlines() if line.strip())


def _wait_for_queue_drop(limit: int, user: str | None, delay: int) -> None:
    """Sleep until the number of queued jobs is below ``limit``."""

    if limit <= 0:
        return

    while True:
        count = _current_job_count(user)
        if count < limit:
            return
        time.sleep(delay)


def _run_sbatch(argv: Iterable[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sbatch", *argv],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main(args: List[str]) -> int:
    retry_delay = _parse_int(os.environ.get("ACCUSNV_SUBMIT_RETRY_DELAY")) or DEFAULT_RETRY_DELAY
    submit_limit = _parse_int(os.environ.get("ACCUSNV_SUBMIT_LIMIT"))
    max_retries = _parse_int(os.environ.get("ACCUSNV_SUBMIT_MAX_RETRIES"))

    user = os.environ.get("SLURM_JOB_USER") or os.environ.get("USER")

    attempts = 0
    while True:
        if submit_limit is not None:
            _wait_for_queue_drop(submit_limit, user, retry_delay)

        proc = _run_sbatch(args)
        if proc.returncode == 0:
            sys.stdout.write(proc.stdout)
            if proc.stderr:
                sys.stderr.write(proc.stderr)
            return 0

        # Forward non-QoS errors immediately.
        if QOS_LIMIT_TOKEN not in proc.stderr:
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            return proc.returncode

        attempts += 1
        if max_retries and attempts > max_retries:
            sys.stderr.write(proc.stderr)
            return proc.returncode

        queue_note = ""
        if user:
            queued_jobs = _current_job_count(user)
            if queued_jobs:
                queue_note = (
                    f" Current queue size for {user!s}: {queued_jobs} job(s)."
                    " Slurm counts both running and pending jobs across"
                    " all partitions/QoS associations when evaluating submit limits."
                )
        sys.stderr.write(
            f"[safe_sbatch] Encountered {QOS_LIMIT_TOKEN}; retrying in {retry_delay} seconds..."
            f"{queue_note}\n"
        )
        time.sleep(retry_delay)


def entrypoint() -> int:
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(entrypoint())
