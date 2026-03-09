#!/usr/bin/env python3
"""
Submit this clustering script for sbatch to snakemake
"""

import os
import sys
import warnings
import subprocess


jobid = sys.argv[-1]

if "Submitted" in jobid:
    jobid = jobid.split('batch job ')[1]

state = subprocess.run(['sacct','-j',jobid,'-X','--format=State'],stdout=subprocess.PIPE).stdout.decode('utf-8')
state = state.split('\n')[2].strip()

map_state={"PENDING":'running',
           "RUNNING":'running', 
           "SUSPENDED":'running', 
           "REQUEUED":'running',
           "CANCELLED":'failed', 
           "CANCELLED+":'failed',
           "COMPLETING":'running', 
           "COMPLETED":'success', 
           "CONFIGURING":'running', 
           "FAILED":'failed',
           "TIMEOUT":'failed',
           "PREEMPTED":'failed',
           "NODE_FAIL":'failed',
           "REVOKED":'failed',
           "SPECIAL_EXIT":'failed',
           "OUT_OF_ME+":'failed',
           "":'running'}


print(map_state[state])
