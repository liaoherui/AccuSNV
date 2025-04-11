import re
import os
import sys
import argparse
import glob
import uuid
import subprocess
### Herui - 2024-09

usage="AccuSNV - SNV calling tool for bacterial isolates using deep learning."
script_dir=os.path.split(os.path.abspath(__file__))[0]

def build_dir(indir):
    if not os.path.exists(indir):
        os.makedirs(indir)

def findfastqfile(dr, smple, filename):
    ##### Add by Herui - 20240919 - Modified function based on the codes from Evan
    # Given the input path and filename, will return the fastq file (include SE, PE, different suffixs) Will gzip the file automatically.
    file_suffixs = ['.fastq.gz', '.fq.gz', '.fastq', '.fq',
                    '_001.fastq.gz', '_001.fq.gz', '_001.fastq', '_001.fq']
    # Check whether the file is the soft link
    target_f=[]
    for f in glob.glob(f"{dr}/*"):
        #print(f,dr)
        if not os.path.islink(f):
            target_f.append(f)

    #print('target...',target_f)
    #exit()
    # Search for filename as a prefix
    files_F = [f for f in target_f if re.search(f"{filename}_?.*?R?1({'|'.join(file_suffixs)})", f)]
    files_R = [f for f in target_f if re.search(f"{filename}_?.*?R?2({'|'.join(file_suffixs)})", f)]
    #print(files_F)
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
    #exit()
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
        #print(",".join(files_F))
        #print(",".join(files_R))
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
            file_R = files_R[0]
            if not file_R.endswith('.gz'):
                subprocess.run("gzip " + file_R, shell=True)
                file_R += '.gz'
    if file_R==[]:
        file_R=''
    if file_F==[]:
        file_F=''
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








def process_input_sfile(infile,out_dir,uid,tem_dir):
	# This function will choose Snakefile according to the input type and set the soft link for input file
	# Generate new sample file for input file
	f=open(infile,'r')
	file_prefix = os.path.splitext(os.path.basename(infile))[0]
	pre_check=pre_check_type(infile)
	#print(file_prefix)
	#exit()
	o=open(tem_dir+'/'+file_prefix+'_rebuild_'+uid+'_tem.csv','w+')
	line=f.readline()
	o.write(line)
	d={} # used to check whether the sequencing type contains 1. only SE? 2. only PE? 3. Both SE and PE.
	while True:
		line=f.readline().strip()
		if not line:break
		ele=re.split(',',line)
		files=findfastqfile(ele[0], ele[1], ele[2])

		abspath=os.path.abspath(ele[0])
		abspath_link=os.path.abspath(out_dir+'/link/')
		if pre_check:
			newpath1=out_dir+'/link/'+ele[1]+'_1.fastq.gz'
			newpath2 = out_dir +'/link/'+ ele[1] + '_2.fastq.gz'
			newpath1s= out_dir +'/link/'+ele[1]+'_1.fastq.gz'
		else:
			newpath1=out_dir+'/link/'+ele[1]+'_1.fastq.gz'
			newpath2 = out_dir +'/link/'+ ele[1] + '_2.fastq.gz'
		if ele[-1]=='SE':
			if pre_check:
				newpath=newpath1s
			else:
				newpath=newpath1
			if not os.path.exists(newpath):
				if not files[0]=='':
					print('ln -s '+os.path.abspath(files[0])+' '+newpath)
					subprocess.run('ln -s '+os.path.abspath(files[0])+' '+newpath, shell=True)
				else:
					print('ln -s ' + os.path.abspath(files[1]) + ' ' + newpath)
					subprocess.run('ln -s ' + os.path.abspath(files[1]) + ' ' + newpath, shell=True)
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
		ele[0]=abspath_link+'/'
		o.write(','.join(ele)+'\n')
		d[ele[-1]]=''
	o.close()
	print('New sample info file:',tem_dir+'/'+file_prefix+'_rebuild_'+uid+'_tem.csv',' is generated for you. Please use it for the Snakemake pipeline!')
	return tem_dir+'/'+file_prefix+'_rebuild_'+uid+'_tem.csv'
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




def copy_config_files(sfile,uid,odir,all_p,idle_p,tf_slurm,tem_dir):
	cond=odir+'/conf'
	build_dir(cond)
	#os.system('cp config.yaml slurm_status_script.py '+cond)
	nexp=tem_dir+'/experiment_info_'+uid+'_tem.yaml'
	os.system('cp experiment_info.yaml ' + nexp)
	o=open(cond+'/config.yaml','w+')
	if tf_slurm == 1:
		os.system('cp config_files/config_no_slurm.yaml ./config.yaml')
	f=open('config.yaml','r')
	while True:
		line=f.readline()
		if not line:break
		if re.search('configfile',line):
			line=re.sub('\./experiment_info.yaml','./'+nexp,line)
		if re.search('partition=\"',line):
			ele=re.split('\"',line)
			if len(idle_p)<3:
				line=ele[0]+"\""+','.join(all_p)+"\"\n"
			else:
				line = ele[0] + "\"" + ','.join(idle_p) + "\"\n"
		o.write(line)
	o.close()
	o2=open(script_dir+'/scripts/dry_run.sh','w+')
	o2.write('snakemake -np  --profile '+cond+'\n')
	o3=open(script_dir+'/scripts/run_snakemake.slurm','w+')
	o3.write('#!/bin/bash\n')
	o3.write('#SBATCH --job-name widevariant.main\n')
	if len(idle_p)<3:
		o3.write('#SBATCH -n 1\n#SBATCH -p '+','.join(all_p)+'\n')
	else:
		o3.write('#SBATCH -n 1\n#SBATCH -p ' + ','.join(idle_p) + '\n')
	o3.write('#SBATCH --time=10:00:00\n')
	o3.write('#SBATCH --mem=50GB\n')
	o3.write('#SBATCH -o mainout.txt\n')
	o3.write('#SBATCH -e mainerr.txt\n')
	o3.write('#SBATCH --mail-user=YOUR_EMAIL_HERE\n')
	o3.write('#SBATCH --mail-type=ALL\n')
	o3.write('# Activate conda environment (may need to change name of env)\n')
	o3.write('#source activate snakemake\n')
	o3.write('snakemake  --profile '+cond+'\n')
	o3.write('# Print "Done!!!" at end of main log file\n')
	o3.write('echo Done!!!\n')


	
	

def reset_exp_file(infile,outdir,uid,sfile,ref_dir,all_p,idle_p,tf_slurm,tem_dir,min_cov_filt,min_cov_samp,exclude_samp,greport):
	f=open(infile,'r')
	tfile=tem_dir+'/exp_'+uid+'_tem.yaml'
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
		elif re.search('min_cov_samp',line):
			o.write('min_cov_samp: \"'+str(min_cov_samp)+'\"\n')
		elif re.search('min_cov_filt',line):
			o.write('min_cov_filt: \"'+str(min_cov_filt)+'\"\n')
		elif re.search('exclude_samp',line):
			o.write('exclude_samp: \"'+str(exclude_samp)+'\"\n')
		elif re.search('greport',line):
			o.write('greport: \"'+str(greport)+'\"\n')
		else:
			o.write(line+'\n')
	o.close()
	os.system(' mv '+tfile+' '+infile)
	copy_config_files(sfile,uid,outdir,all_p,idle_p,tf_slurm,tem_dir)


def is_partition_valid(partition_name):

	try:
		result = subprocess.run(
			["sinfo", "-p", partition_name],
			capture_output=True,
			text=True,
			check=False
		)

		return (result.returncode == 0 and
				len(result.stdout.split('\n'))>1)
	except Exception:
		return False



def parse_time_limit(time_limit):

    days, hours, minutes = 0, 0, 0
    parts = time_limit.split("-")

    if len(parts) == 2:
        days = int(parts[0])
        time_part = parts[1]
    else:
        time_part = parts[0]

    time_parts = list(map(int, time_part.split(":")))

    if len(time_parts) == 3:
        hours, minutes = time_parts[:2]
    elif len(time_parts) == 2:
        minutes = time_parts[0]

    return days * 24 * 60 + hours * 60 + minutes

def is_partition_time_limit_exceed(partition):
    try:

        time_limit = subprocess.check_output(
            f"sinfo -h -o '%l' -p {partition}", shell=True, text=True
        ).strip()

        return parse_time_limit(time_limit) >= 300
    except Exception:
        return False

def snakefile_modify(infile,cp_env):
	o=open('sk_tem','w+')
	f=open(infile,'r')
	lines=f.read().split('\n')
	for line in lines:
		if not re.search('source',line):
			o.write(line+'\n')
		else:
			#ele=re.split('accusnv_sub',line)
			if not re.search('source accusnv_sub',line):
				o.write(line+'\n')
			else:
				out=re.sub('accusnv_sub',cp_env,line)
				#out=ele[0]+res
				o.write(out+'\n')
	o.close()
	os.system('mv sk_tem Snakefile')

'''
def snakefile_modify_accusnv(infile,min_cov_filt,min_cov_samp,exclude_samp,greport):
	o = open('sk_tem', 'w+')
	f=open(infile,'r')
	lines = f.read().split('\n')
	for line in lines:
		if not re.search('new_snv_script',line):
			o.write(line + '\n')
		else:
			if exclude_samp=='':
				out=re.sub('-s 45 -v 5','-s '+str(min_cov_samp)+' -v '+str(min_cov_filt)+' -g '+str(greport),line)
			else:
				out = re.sub('-s 45 -v 5', '-s ' + str(min_cov_samp) + ' -v ' + str(min_cov_filt) + ' -g ' + str(greport)+' -e '+str(exclude_samp), line)
			o.write(out + '\n')
	o.close()
	os.system('mv sk_tem Snakefile')
'''

def main():

	pwd=os.getcwd()
	# Get para
	parser=argparse.ArgumentParser(prog='AccuSNV',description=usage)
	parser.add_argument('-i','--input_sample_info',dest='input_sp',type=str,required=True,help="The dir of input sample info file --- Required")
	parser.add_argument('-t','--turn_off_slurm',dest='tf_slurm',type=int,help="If set to 1, the SLURM system will not be used for automatic job submission. Instead, all jobs will run locally or on a single node. (Default: 0)")
	parser.add_argument('-c','--conda_prebuilt_env',dest='cp_env',type=str,help="The absolute dir of your pre-built conda env. e.g. /path/snake_pipeline/accusnv_sub")
	parser.add_argument('-r','--ref_dir',dest='ref_dir',type=str,help="The dir of your reference genomes")
	#### AccuSNV - CNN-filter part params
	parser.add_argument('-s','--min_cov_for_filter_sample',dest='min_cov_samp',type=str,help="Before running the CNN model, low-quality samples with more than 45%% of positions having zero aligned reads will be filtered out. (default \"-s 45\") You can adjust this threshold with this parameter; to include all samples, set \"-s 100\".")
	parser.add_argument('-v','--min_cov_for_filter_pos',dest='min_cov',type=str,help="For the filter module: on individual samples, calls must have at least this many reads on the fwd/rev strands individually. If many samples have low coverage (e.g. <5), then you can set this parameter to smaller value. (e.g. -v 2). Default is 5.")
	parser.add_argument('-e','--excluse_samples',dest='exclude_samp',type=str,help="The names of the samples you want to exclude (e.g. -e S1,S2,S3). If you specify a number, such as \"-e 1000\", any sample with more than 1,000 SNVs will be automatically excluded.")
	parser.add_argument('-g','--generate_report',dest='generate_rep',type=str,help="If not generate html report and other related files, set to 0. (default: 1)")
	####
	parser.add_argument('-o','--output_dir',dest='out_dir',type=str,help='Output dir (default: current dir/wd_out_(uid), uid is generated randomly)') # uid=uuid.uuid1().hex
	args = parser.parse_args()
	input_file=args.input_sp
	out_dir=args.out_dir
	ref_dir=args.ref_dir
	tf_slurm = args.tf_slurm
	cp_env=args.cp_env
	# CNN-filter params
	min_cov_samp = args.min_cov_samp
	min_cov_filt = args.min_cov
	greport = args.generate_rep
	exclude_samp = args.exclude_samp


	uid=uuid.uuid1().hex
	if not out_dir:
		out_dir = pwd+'/wd_out_'+uid
	if not ref_dir:
		ref_dir=''
	if not cp_env:
		cp_env=''
	if not tf_slurm:
		tf_slurm=0
	if not min_cov_filt:
		min_cov_filt = 5
	else:
		min_cov_filt = int(min_cov_filt)

	if not min_cov_samp:
		min_cov_samp = 45
	else:
		min_cov_samp = int(min_cov_samp)

	if not exclude_samp:
		exclude_samp = 'null'


	if not greport:
		greport = 1
	else:
		greport = int(greport)

	#snakefile_modify_accusnv('Snakefile',min_cov_filt,min_cov_samp,exclude_samp,greport)

	if not cp_env=='':
		cp_env=os.path.abspath(cp_env)
		os.system('cp Snakefiles_diff_options/Snakefile ./')
		snakefile_modify('Snakefile', cp_env)



	tem_dir=out_dir+'/temp'
	data_dir = out_dir + '/link'
	build_dir(tem_dir)
	build_dir(data_dir)
	all_p_raw = subprocess.check_output("sinfo -h -o '%P' | sort -u", shell=True, text=True).split()
	all_p=[]
	for s in all_p_raw:
		if is_partition_valid(s) and is_partition_time_limit_exceed(s):
			all_p.append(s)
	#all_p=','.join(all_p)

	idle_p_raw = subprocess.check_output("sinfo -h -o '%P %T' | awk '$2 == \"idle\" {print $1}'", shell=True, text=True).split()
	idle_p=[]
	for s in idle_p_raw:
		if s in all_p:
			idle_p.append(s)
	#idle_p = ','.join(idle_p)
	#out_dir=os.path.abspath(out_dir)
	#print(all_p,idle_p)
	#exit()
	sfile=process_input_sfile(input_file,out_dir,uid,tem_dir)
	reset_exp_file(script_dir+'/experiment_info.yaml',out_dir,uid,sfile,ref_dir,all_p,idle_p,tf_slurm,tem_dir,min_cov_filt,min_cov_samp,exclude_samp,greport)

	


if __name__=='__main__':
	sys.exit(main())
