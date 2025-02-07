# Wishlist for local analysis upgrades


## Modules with additional functionality

In the future, we should add the following modules to the local analysis script:
* Adaptive evolution analysis: look for parallel evolution and compute dN/dS (N/S compared to neutral model)
	* Status: Arolyn has a nice version in Matlab, but it hasn't been coded in python yet.
* Gain/loss analysis: use coverage matrices to find genes that are missing from some samples in the lineage
	* Status: Arolyn and Felix developed an amplification/deletion python script which could be incorporated into this version. Arolyn also made a gain/loss script in matlab which could be re-coded in python. 
* Additional treemaking: implement treecounting script
	* Status: This was implemented in previous python analysis script (should be relatively easy to incorporate).


## Changes to existing functions

* Rewrite functions that generate genome annotations from scratch (they are messy)


## Environment/python version

* Upgrade to Python 3.11 (for [faster CPython](https://docs.python.org/3/whatsnew/3.11.html#whatsnew311-faster-cpython))


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
