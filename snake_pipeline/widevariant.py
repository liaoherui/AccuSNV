import re
import os
import sys
import argparse
import glob
import uuid
import subprocess
### Herui - 2024-09

usage="WideVariant - SNP calling pipeline."
script_dir=os.path.split(os.path.abspath(__file__))[0]

def findfastqfile(dr, smple, filename):
    ##### Add by Herui - 20240919 - Modified function based on the codes from Evan
    # Given the input path and filename, will return the fastq file (include SE, PE, different suffixs) Will gzip the file automatically.
    file_suffixs = ['.fastq.gz', '.fq.gz', '.fastq', '.fq',
                    '_001.fastq.gz', '_001.fq.gz', '_001.fastq', '_001.fq']
    # Check whether the file is the soft link
    target_f=[]
    for f in glob.glob(f"{dr}/*"):
        if not os.path.islink(f):
            target_f.append(f)
    #print('target...',target_f)
    #exit()
    # Search for filename as a prefix
    files_F = [f for f in target_f if re.search(f"{filename}_?.*?R?1({'|'.join(file_suffixs)})", f)]
    files_R = [f for f in target_f if re.search(f"{filename}_?.*?R?2({'|'.join(file_suffixs)})", f)]
    #print(dr,smple,filename,glob.glob(f"{dr}/*"))
    # Search for filename as a directory
    file_F = []
    file_R = []
    if os.path.isdir(f"{dr}/{filename}"):
        target_f = []
        for f in glob.glob(f"{dr}/{filename}/*"):
            if not os.path.islink(f):
                target_f.append(f)
        files_F = files_F + [f"{filename}/{f}" for f in target_f
                             if re.search(f"{filename}/.*_?.*?R?1({'|'.join(file_suffixs)})", f)]
        files_R = files_R + [f"{filename}/{f}" for f in target_f
                             if re.search(f"{filename}/.*_?.*?R?2({'|'.join(file_suffixs)})", f)]
    #print(files_F,files_R)
    if len(files_F) == 0 and len(files_R) == 0:
        # Can be single-end reads and no "1" or "2" ID in the filename
        print(f'No file found in {dr} for sample {smple} with prefix {filename}! Go single-end checking!')
        files_F = [f for f in target_f if re.search(f"{filename}.*_?.*({'|'.join(file_suffixs)})", f)]

        if os.path.isdir(f"{dr}/{filename}"):
            files_F = files_F + [f"{filename}/{f}" for f in target_f if
                                 re.search(f"{filename}/.*_?.*({'|'.join(file_suffixs)})", f)]
        if len(files_F) == 0:
            raise ValueError(f'No file found in {dr} for sample {smple} with prefix {filename}')
        else:
            file_F = files_F[0]
            if not file_F.endswith('.gz'):
                subprocess.run("gzip " + file_F, shell=True)
                file_F += '.gz'
        # print(files_F)

    elif len(files_F) > 1 or len(files_R) > 1:
        # print(",".join(files_F))
        # print(",".join(files_R))
        raise ValueError(f'More than 1 matching files found in {dr} for sample {smple} with prefix {filename}:\n \
                         {",".join(files_F)}\n \
                         {",".join(files_R)}')

    elif len(files_F) == 1 and len(files_R) == 1:
        file_F = files_F[0]
        file_R = files_R[0]

        ## Zip fastq files if they aren't already zipped
        if not file_F.endswith('.gz'):
            subprocess.run("gzip " + file_F, shell=True)
            file_F += '.gz'
        if not file_R.endswith('.gz'):
            subprocess.run("gzip " + file_R, shell=True)
            file_R += '.gz'
    elif len(files_F) == 1 or len(files_R) == 1:
        if len(files_F) == 1:
            file_F = files_F[0]
            if not file_F.endswith('.gz'):
                subprocess.run("gzip " + file_F, shell=True)
                file_F += '.gz'
        if len(files_R) == 1:
            file_T = files_R[0]
            if not file_R.endswith('.gz'):
                subprocess.run("gzip " + file_R, shell=True)
                file_R += '.gz'
    if file_R==[]:
        file_R=''
    return [file_F, file_R]

def pre_check_type(infile):
	f=open(infile,'r')
	line=f.readline()
	d={}
	while True:
		line=f.readline().strip()
		if not line:break
		ele=re.split(',',line)
		d[ele[-1]]=''
	if len(d)==1:
		print('There are only SE or PE reads in the input. Use single mode!')
		return False
	else:
		print('There are both SE and PE reads in the input. Use mix mode!')
		return True

def process_input_sfile(infile,uid):
# This function will choose Snakefile according to the input type and set the soft link for input file
	# Generate new sample file for input file
	f=open(infile,'r')
	file_prefix = os.path.splitext(os.path.basename(infile))[0]
	pre_check=pre_check_type(infile)
	#print(file_prefix)
	#exit()
	o=open(file_prefix+'_rebuild_'+uid+'.csv','w+')
	line=f.readline()
	o.write(line)
	d={} # used to check whether the sequencing type contains 1. only SE? 2. only PE? 3. Both SE and PE.
	while True:
		line=f.readline().strip()
		if not line:break
		ele=re.split(',',line)
		files=findfastqfile(ele[0], ele[1], ele[2])
		if os.path.exists('data/'+ele[1]):
			os.system('rm -rf data/'+ele[1])
		abspath=os.path.abspath(ele[0])
		if pre_check:
			newpath1=abspath+'/'+ele[2]+'_1.fastq.gz'
			newpath2 = abspath +'/'+ ele[2] + '_2.fastq.gz'
			newpath1s=abspath+'/'+ele[2]+'_1.fastq.gz'
		else:
			newpath1=abspath+'/'+ele[2]+'_1.fastq.gz'
			newpath2 = abspath +'/'+ ele[2] + '_2.fastq.gz'
		if ele[-1]=='SE':
			if pre_check:
				newpath=newpath1s
			else:
				newpath=newpath1
			if not os.path.exists(newpath):
				print('ln -s '+os.path.abspath(files[0])+' '+newpath)
				subprocess.run('ln -s '+os.path.abspath(files[0])+' '+newpath, shell=True)
			else:
				print(newpath1+' exists! Skip data links.')
		else:
			if not os.path.exists(newpath1):
				print('ln -s ' + os.path.abspath(files[0]) + ' ' + newpath1)
				subprocess.run('ln -s ' + os.path.abspath(files[0]) + ' ' + newpath1, shell=True)
			else:
				print(newpath1+' exists! Skip data links.')
			if not os.path.exists(newpath2):
				print('ln -s ' + os.path.abspath(files[1]) + ' ' + newpath2)
				subprocess.run('ln -s ' + os.path.abspath(files[1]) + ' ' + newpath2, shell=True)
			else:
				print(newpath2+' exists! Skip data links.')
		ele[0]=abspath+'/'
		o.write(','.join(ele)+'\n')
		d[ele[-1]]=''
	o.close()
	print('New sample info file:',file_prefix+'_rebuild_'+uid+'.csv',' is generated for you. Please use it for the Snakemake pipeline!')
	return file_prefix+'_rebuild_'+uid+'.csv'
	# choose Snakefile according to the input type
	'''
	if len(d)==1:
		if 'SE' in d:
			os.system('cp '+script_dir+'/Snakefile-3cases/Snakefile_se Snakefile')
		else:
			os.system('cp ' + script_dir + '/Snakefile-3cases/Snakefile_pe Snakefile')
	else:
		os.system('cp ' + script_dir + '/Snakefile-3cases/Snakefile_se_pe Snakefile')
	'''

def build_dir(indir):
    if not os.path.exists(indir):
        os.makedirs(indir)

def reset_exp_file(infile,outdir,uid,sfile,ref_dir):
	f=open(infile,'r')
	tfile='exp_'+uid+'.yaml'
	o=open(tfile,'w+')
	while True:
		line=f.readline().strip()
		if not line:break
		if re.search('outdir',line):
			o.write('outdir: '+outdir+'\n')
		elif re.search('sample_table',line):
			o.write('sample_table: '+sfile+'\n')
		elif re.search('ref_genome_directory',line) and not ref_dir=='':
			ref_dir=os.path.abspath(ref_dir)
			o.write('ref_genome_directory: '+ref_dir+'\n')
		else:
			o.write(line+'\n')
	o.close()
	os.system(' mv '+tfile+' '+infile)
	

def main():

	pwd=os.getcwd()
	# Get para
	parser=argparse.ArgumentParser(prog='WideVariant',description=usage)
	parser.add_argument('-i','--input_sample_info',dest='input_sp',type=str,required=True,help="The dir of input sample info file --- Required")
	#parser.add_argument('-j','--input_fastq_2',dest='input_fq2',type=str,help="The dir of input fastq data (for pair-end data).")
	parser.add_argument('-r','--ref_dir',dest='ref_dir',type=str,help="The dir of your reference genomes")
	parser.add_argument('-o','--output_dir',dest='out_dir',type=str,help='Output dir (default: current dir/wd_out_(uid), uid is generated randomly)') # uid=uuid.uuid1().hex
	args = parser.parse_args()
	input_file=args.input_sp
	out_dir=args.out_dir
	ref_dir=args.ref_dir

	uid=uuid.uuid1().hex
	if not out_dir:
		out_dir = pwd+'/wd_out_'+uid
	if not ref_dir:
		ref_dir=''
    #build_dir(out_dir )
	sfile=process_input_sfile(input_file,uid)
	reset_exp_file(script_dir+'/experiment_info.yaml',out_dir,uid,sfile,ref_dir)

	


if __name__=='__main__':
	sys.exit(main())
