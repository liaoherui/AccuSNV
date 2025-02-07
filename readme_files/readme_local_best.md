# Python best practices

This is a summary of the best practices implemented in the local analysis scripts. 


## Python objects

### Numerical arrays

Numerical arrays are stored as [numpy](https://numpy.org) arrays of various types (integers, floats, booleans), and not as regular python arrays. 

Numpy arrays have a number of advantages, including:
* Low memory load: Regular python lists allow elements to be of different types, which means that the type/class of each element has to be stored individually, whereas numpy arrays must have elements that are all the same type (eg integers), so they only have to store the type once.
* Fast and efficient computation: Numpy has functions called [ufuncs](https://numpy.org/doc/stable/user/basics.ufuncs.html#ufuncs-basics), which are vectorized and support [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).

### Data tables

Data tables (e.g. reference genome annotations) are stored as [pandas](https://pandas.pydata.org) [dataframes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). 

Dataframes have a number of advantages, including:
* Labeled axes (e.g. column names can be used like a dictionary)
* Ability to support heterogeneous data (e.g. integers in one column, strings in another column)


## Indexing

### Reference genome

The following standards have been implemented for counting positions and contigs on the reference genome:

* Positions on the reference genome are counted starting from 1 in order to match the VCF files.
* Contigs on the reference genome are also counted starting from 1. 

These standards are implemented in the reference genome class.

(All python indexing still starts at 0. The above just refers to how we are naming positions and contigs.)

### Basecalls and nucleotides

Basecalls are stored in numerical arrays where: 

* 0 = N (ambiguous basecall)
* 1/2/3/4 = A/T/C/G

Python dictionaries ```NTs_to_int_dict``` and ```int_to_NTs_dict``` define the mapping between integers and nucleotides. This mapping should never be hard-coded.

Storing Ns as 0s is efficient because we can utilize built-in functions like ```numpy.count_nonzero```.

### Data objects in candidate mutation table

Indexing of arrays is standardized according to: 
* Index 0 = sample
* Index 1 = position on genome
* Index 2 = another characteristic (if applicable)


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
