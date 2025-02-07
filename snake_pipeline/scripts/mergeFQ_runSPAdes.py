# read clade files with sample IDs (input)
# re-build fa file names 
# merge per clade and bgzip
# run spades on newly generated file (output)

# 2019.09.09, Arolyn: changed subdirectory sturcture...
# 2019.09.09, Arolyn: running spades via module add on c3ddb, not from Felix's home directory

import argparse,sys
import subprocess # needed for system calls

''' FUNCTIONS'''

def build_sample_file_list(file):
	''' read clade file with sample IDs and transform to files (based on known structure) '''
	# with open(file, "r") as ins:
	ls_fwd = []
	ls_rev = []
	for line in file:
		#line = line.rstrip('\n')
		ls_fwd.append("1-data_processed/"+line+"/filt1.fq.gz")
		ls_rev.append("1-data_processed/"+line+"/filt2.fq.gz")
	return [ls_fwd,ls_rev]

def merge_fq(ls_sample_file,clade,reads2grab=None):
	''' take all sample-files and merge them using system cat call (rev & fwd) '''
	input_file1_string = ' '.join(ls_sample_file[0])
	input_file2_string = ' '.join(ls_sample_file[1])
	ls1 = ls_sample_file[0]
	ls2 = ls_sample_file[1]
	outfile1 = "0-tmp/in1_spades_"+clade+".fq.gz"
	outfile2 = "0-tmp/in2_spades_"+clade+".fq.gz"
	if reads2grab:
		lines2grab = int(reads2grab)*4
	else:
		reads_smallest_fq = subprocess.run('zgrep -c "^+$" ' + input_file1_string + " | cut -d':' -f 2 | sort -n | head -n 1",shell=True, stdout=subprocess.PIPE)
		lines2grab = int(reads_smallest_fq.stdout)*4
		
	for i in range(len(ls1)):
		subprocess.run("gzip -cd " + ls1[i] + " | head -q -n " + str(lines2grab) + " | gzip >> " + outfile1,shell=True)
		subprocess.run("gzip -cd " + ls2[i] + " | head -q -n " + str(lines2grab) + " | gzip >> " + outfile2,shell=True)
	return [outfile1,outfile2]

def run_spades(ls_merge_fq, clade, threads):
	''' run spades. output to 3-spades/clade_[]/ '''
	outfolder = "Assembly/3-spades/clade_"+clade+"/"
	subprocess.run("mkdir -p " + outfolder , shell=True)
	subprocess.run("spades.py --phred-offset 33 --careful -t " + str(threads) + " -1 " + ls_merge_fq[0] + " -2 " + ls_merge_fq[1] + " -o " + outfolder , shell=True)


''' MAIN '''


if __name__ == "__main__":

	
	''' positional and optional argument parser'''
	
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
									 description='''\
		Py Script integrated in de-novo genome assembly snakemake routine.
		Step1: Merge previously created and kraken-validated fq.gz files (each contains 250k)
		Step2: Run SPAdes in careful mode.
		Note: Input file has to follow naming scheme in order to allow the script to extract the clade identifier (see def run_spades : [folder]/[samplesPerSubject]/samples[cladeID]_[string.txt])
									''',
									epilog="Questions or comments? --> fkey@mit.edu")
	parser.add_argument("-i", dest='input', help="Input file per clade including sample-IDs validated by kraken", type=argparse.FileType('rt'))
	parser.add_argument('-t', dest='threads',help="Number of threads",type=int,default=1)
	parser.add_argument('-s', dest='cladeid',help="Clade ID (Snakemake wildcard!)",type=str)
	args = parser.parse_args()
	

	# infile = '3-spades/samplesPerSubject/clade10_samples.txt' # from cammand line
	infile = args.input
	file_names = build_sample_file_list(infile)
	cladeID = args.cladeid 
	outfileLs = merge_fq(file_names,cladeID)
	run_spades(outfileLs,cladeID,threads)
	sys.exit()

