# Helpful hints for using the command line

This page covers the command-line basics of:
* [Bash](#bash-basics) - command-line shell on Unix operating systems
* [Slurm](#Slurm-basics) - workload manager for Linux clusters
* [Conda](#Conda-basics) - package and environment management system
* [Snakemake](#Snakemake-basics) - workflow management system designed for bioinformatics pipelines


## Bash basics

Bash is the command line shell on Unix operating systems. There are many bash resources available online (like [this one](https://wizardzines.com/zines/bite-size-command-line/)). Below is a list of simple bash commands that will help you interact with a computer via the terminal.


### Getting more info on commands

To see the manual file on a given command, you can type `man COMMAND_NAME` into the terminal. This will tell you what the command does and what its options are.

### Navigating to different directories

`pwd` - print working directory
`cd` - change directory 
`ls` - list directory contents

### Viewing files

`less` - view contents of a text file
`head` - view the start of a file 
`tail` - view the end of a file
`cat` - read a file, create a file, concatenate files

Note that there are additional commands for viewing compressed files (e.g. `zcat` insetad of `cat`).

#### Searching files

`grep` - search text for patterns

### Manipulating files and directories

`mv` - move/rename a file/directory
`cp` - copy a file/directory 

`mkdir` - make a directory
`rmdir` - remove a directory
`rm` - remove a file (`rm -r` remove a directory)

`chmod` - set file/directory permissions (e.g. make a file executable)

### Writing and editing to files

`echo` - print text to terminal window
`touch` - create a file or update the timestamp on an existing file

### Manipulating text files

`vim` - [vim](https://www.vim.org) is a powerful text editor

`awk` - for manipulating columns of data
`sed` - for replacing text in a file

`diff` - for comparing two text files

### Miscellaneous

`|` (pipe) - takes the standard output of one command and passes it as the input to another
`>` - redirect stadout
`>>` - append to a file (or create a file if it doesn't already exist; in contrast `>` will overwrite a file if it already exists)

`man` - print manual page for a command



## Slurm basics

Slurm is a workload manager for linux clusters. Full documentation is available [here](https://slurm.schedmd.com/documentation.html). Below are some examples to get you started.

### Batching jobs

`sbatch` - submit a batch script to Slurm

There are MANY `sbatch` options (see `man sbatch`) that you should use to specify how many resources your job needs (e.g. how many CPUs, how much memory, how much time, etc.).

### Interactive sessions

`srun --pty -t 0-0:30:0 -n 1 -p PARTITION_NAME /bin/bash` - request a 30 minute interactive session on partition PARTITION_NAME.

### Determining cluster and job status

`sinfo` - view information about Slurm cluster nodes and partitions
`squeue` - view information about jobs in the Slurm scheduling queue
`sacct` - display job accounting data

Here is an example of how you could see print a table of all of your jobs that failed recently:
`sacct --format="JobID,JobName%30,State,nodelist,partition,cluster,elapsed,start,end,exitcode" | grep FAILED`.

### Cancelling jobs

`scancel` - cancel job(s)

Here is how you would cancel an individual job: `scancel JOBID`. 
Here is how you would cancel all of your current jobs: `scancel -u YOUR_USERNAME`.
Here is how you would cancel all of your jobs with JOB_STR in the name: `squeue | grep USERNAME | grep JOB_STR | awk '{print $1}' | xargs scancel`



## Conda basics

Conda is a package management system and environment management system. Full documentation is available [here](https://slurm.schedmd.com/documentation.html).

### Conda cheatsheet

The official Conda cheatsheet is available [here](https://conda.io/projects/conda/en/latest/user-guide/cheatsheet.html).



## Snakemake basics

Snakemake is a workflow management system designed for bioinformatics pipelines. 
* A brief intro video is available [here](https://www.youtube.com/watch?v=UOKxta3061g)
* Full documentation is available [here](https://snakemake.readthedocs.io/en/stable/).

Below are some tips on running Snakemake through the command line.

### Running a Snakemake pipeline

For reproducibility, we recommend running Snakemake through the files provided (`myjob.slurm` and `snakemakeslurm.sh`). We do not recommend running Snakemake directly from the command line. For more information, see [How to run the snakemake pipeline](readme_snake_run.md).

If your pipeline gets unexpectedly aborted or cancelled, you may need to unlock your Snakemake directory (`snakemake --unlock -s Snakefile.py`). Do NOT attempt this unless you are certain Snakemake is no longer running.

### Dry runs

We strongly recommend running a dryrun (`snakemake -s Snakefile.py -n -p -c1`) before batching your Snakemake pipeline to the cluster. This will print a list of jobs that Snakemake is planning to execute. 

### Visualizing a Snakemake pipeline

Snakemake can build a pdf of a flowchart representing the Snakemake pipeline (`snakemake --forceall -dag | dot -Tpdf > dag.pdf`). 



## Table of contents

[Main Lieberman Lab pipeline README](../README.md)
* [Snakemake pipeline](readme_snake_main.md)
	* [How to run the snakemake pipeline](readme_snake_run.md)
	* [Technical details about the snakemake pipeline](readme_snake_rules.md)
	* [Wishlist for snakemake pipeline upgrades](readme_snake_wishlist.md)
	* [Helpful hints for using the command line](readme_snake_basics.md)
* [Local analysis](readme_local_main.md)
	* [How to run the local analysis script](readme_local_run.md)
	* [Wishlist for local analysis upgrades](readme_local_wishlist.md)
	* [Python best practices](readme_local_best.md)
