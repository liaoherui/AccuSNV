"""TDL Nov 2011 edited from Nature Genetics project
edited August 2013 to just generate .tree files

"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
from pylab import *
import random
from matplotlib.font_manager import FontProperties
from scipy import stats
from datetime import datetime, date, time
from matplotlib import rc
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord



rc('font', **{'sans-serif':'Arial'})


def mutation_count(inputtree, lca):
	#print(inputtree,lca)
	#exit()
	emptycount=0
	ambiguous=['R','Y','M','K','S','W']
	ACGT=['A','C','T','G']

	tree=''
	for i,c in enumerate(inputtree):
		if c in ['(', ',', ')']:
			tree+=c
		elif c =='*':
			tree+=inputtree[i+1]
	#print(tree)
	#exit()
	#remove all question marks and ambiguities from the tree

	#simplify tree while counting mutations. give preference to simplifying first.
	simplified=1
	mutlist={('A','C'):0,('G','C'):0,('T','C'):0,('A','G'):0,('C','G'):0,('T','G'):0, ('A','T'):0,('G','T'):0,('C','T'):0, ('T','A'):0,('G','A'):0,('C','A'):0}
	dcount={}
	all_val_mut=[]
	for i in tree:
		if i=='N' or i==',' or i==')' or i=='(':continue
		if i not in dcount:
			dcount[i]=1
		else:
			dcount[i]+=1
		all_val_mut.append(i)
	res=sorted(dcount.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
	#print(res)
	#print(all_val_mut)
	anc=res[0][0]

	for i in res:
		if i[0] ==anc:continue
		mutlist[anc,i[0]]+=i[1]

	
	muts=0
	for pair in mutlist.keys():
		muts+=mutlist[pair]
	
	if muts==0:
		print(tree)
		#pass
	
	return muts, mutlist


def load_chr(chart):
	f=open(chart,'r')
	line=f.readline()
	chromosomes=[]
	d={}
	while True:
		line=f.readline().strip()
		if not line:break
		ele=re.split(',',line)
		if ele[0] not in d:
			d[ele[0]]=''
			chromosomes.append(ele[0])
	return chromosomes


def convert_csv(chart_raw,mode,out):
	# if mode == 1 -> used in the Snakemake pipeline, mode==2 -> used in the downstream analysis module
	chart = out + '/snpChart.csv'
	df = pd.read_csv(chart_raw, sep='\t')
	c=0
	raw_pos=[]
	keep_col=['contig_idx','contig_pos']
	#print(df.columns)
	#exit()
	for col in df.columns:
		if c == 0:
			c += 1
			raw_pos = np.array(df[col])
			continue
		if mode==1:
			if c < 34:
				# if col=='contig_idx' or col=='contig_pos':
				# 	keep_col.append(col)
				c+=1
				continue
		else:
			if c < 19:
				c+=1
				continue

		if re.search('sequence', col): continue
		if re.search('transl', col): continue
		if re.search('Unnamed', col): continue
		keep_col.append(col)
	#print(keep_col)
	#exit()
	subset=df[keep_col]
	subset.rename(columns={'contig_pos': 'pos'}, inplace=True)
	subset.rename(columns={'contig_idx': 'chr'}, inplace=True)
	subset = subset.applymap(lambda x: x.replace(',', '') if isinstance(x, str) else x)
	subset.to_csv(chart, index=False)

	return chart,raw_pos
		


def _load_chart_lookup(chart):
	"""Load SNP table once and build lookup for fast per-position access."""
	with open(chart, 'r') as fh:
		rows = [line.strip().split(',') for line in fh if line.strip()]
	if not rows:
		return [], {}, []

	header = rows[0]
	locations = [idx for idx, name in enumerate(header) if 'pos' not in name and 'chr' not in name]
	lookup = {}
	for row in rows[1:]:
		if len(row) < 2:
			continue
		lookup[(row[0], row[1])] = row
	return header, lookup, locations


def _available_bar_chart_positions(out):
	bar_dir = out + '/bar_charts'
	if not os.path.isdir(bar_dir):
		return None
	positions = set()
	for fn in os.listdir(bar_dir):
		m = re.match(r'^p_(.+)_bar_chart\.png$', fn)
		if m:
			positions.add(m.group(1))
	return positions


def mutationtypes(tree, chart_raw, mode, out, max_trees=None, only_positions_with_bar_chart=True):

	chart, raw_pos = convert_csv(chart_raw, mode, out)
	tout = out + '/snp_trees'
	if not os.path.exists(tout):
		os.makedirs(tout)

	header, chart_lookup, locations = _load_chart_lookup(chart)
	if not header:
		return

	bar_chart_positions = _available_bar_chart_positions(out) if only_positions_with_bar_chart else None

	with open(chart, 'r') as f:
		lines = f.readlines()[1:]

	generated = 0
	for pidx, line in enumerate(lines):
		l = line.strip().split(',')
		if len(l) < 5:
			print(l)
			continue

		chromosome = l[0]
		pos = l[1]
		raw_position = str(raw_pos[pidx])

		if bar_chart_positions is not None and raw_position not in bar_chart_positions:
			continue

		if max_trees is not None and generated >= int(max_trees):
			break

		# use first strain as lca
		lca = l[4]

		# Build tree annotation directly from lookup (no repeated full file scan)
		row = chart_lookup.get((chromosome, pos))
		if row is None:
			continue
		with open('temptree.txt', 'w') as fo:
			for idx in locations:
				if idx < len(row) and len(row[idx]) > 0:
					fo.write(header[idx] + '\t' + row[idx] + '\n')
				else:
					fo.write(header[idx] + '\t?\n')
		newtree = nameswap(tree, 'temptree.txt')

		a, mutlist = mutation_count(newtree, lca)

		if a == 0:
			print('NO MUTS:')
			continue

		outfile = tout + '/p_' + raw_position + '_' + str(a) + '.tree'
		if os.path.exists(outfile):
			# Align with bar_charts generation behavior: skip if already present
			continue

		with open(outfile, 'w') as f1:
			with open('tempnexus.txt', 'r') as tf:
				f1.writelines(tf.readlines())
		generated += 1

	for temp_file in ('temptree.txt', 'tempnexus.txt'):
		if os.path.exists(temp_file):
			os.remove(temp_file)



def annotateSNP(tree, dict, chromosome,pos):
	#chromosomes=['NZ_CH482380.1','NZ_CH482381.1','NZ_CH482382.1']

	f=open(dict).readlines()
	fo=open('temptree.txt','w')
	
	header=f[0].strip().split(',')
	locations=[]
	
	for n,i in enumerate(header):
		if not re.search('pos',i) and not re.search('chr',i) :
			locations.append(n)
	#print(locations)
	#exit()

	for line in f:
		l=line.strip('\n').split(',')
		#print(l,chromosome,pos)
		#exit()
		if l[0]==chromosome and l[1]==pos:
			for i in locations:
				#print(len(l),i)
				#exit()
				if len(l) > i and len(l[i])>0:
					fo.write(header[i]+'\t'+l[i]+'\n')
				else:
					fo.write(header[i]+'\t?\n')
				if i > len(l):
					print(line, l, i)
			break
	#exit()
	fo.close()
	newtree = nameswap(tree,'temptree.txt')
	return newtree



def nameswap(tree, dictionary):

	fraw=open(dictionary).readlines()
	f=[]
	for r in fraw:
		if not re.search('muts\t',r) and not re.search('type\t',r):
			f.append(r)
	#print(f,len(f))
	#exit()
	numStrains=len(f)
	dict={}
	annotation={}
	newname={}
	
	#print header for tempnexus
	fo=open('tempnexus.txt','w')
	fo.write('#NEXUS\nbegin taxa;\n\tdimensions ntax='+str(numStrains)+';\n\ttaxlabels\n')			
	colors={'A':'[&!color=#-16776961]', 'C':'[&!color=#-16725916]','G':'[&!color=#-3670016]','T':'[&!color=#-3618816]','?':'[&!color=#-16777216]','N':'[&!color=#-16777216]'}
	ambiguous=['R','Y','M','K','S','W']
	for each in ambiguous:
		colors[each]='[&!color=#-16777216]'
				
	#get annotations
	f=open(dictionary, 'r').readlines()
	for line in f:
		if not line.startswith('#'):
			l=line.split()
			annotation[l[0]]=l[1]
	
	
	#combine names and annotations	
	for i in annotation.keys():
		newname[i]=i+ '--*'+ annotation[i]
		#print(newname)
		#exit()
		#if i in dict.keys():
			#newname[i]=dict[i]+ '--*'+ annotation[i] #for dating newname[i]=dict[i]+ '--*'+ annotation[i]
		#else:
			#newname[i]= i + '--*'+ annotation[i] #for reference, etc.
	#print(newname)
	#exit()
	#print(dictionary,annotation.keys())
	#exit()
	#make new tree
	f=open(tree,'r').readlines()
	#print(tree)
	
	newtree=''
	
	for line in f:
		line=line.strip()
		#print(line)
		line=re.sub(',reference_genome:0.00000,inferred_ancestor:0.00000','',line)
		i=0
		#print(line)
		# Regular expression to extract key:value pairs
		pattern = r"([\w\-]+:\d+\.\d+)"

		# Find all matches in the string
		matches = re.findall(pattern, line)
		#print(matches)
		#exit()

		# Create an empty list to store the replaced patterns
		replaced_patterns = []
		#print(matches)
		nline=line
		# Replace keys using the dictionary and store the results
		for match in matches:
			key, value = match.split(":")
			#print(key,value)
			if key=='reference_genome' or key=='inferred_ancestor' or newname[key][-1] not in colors:
				print(key)
				continue
			if key in newname:
				# Replace the key with the corresponding value from the dictionary
				new_key = newname[key]
				replaced_patterns.append(f"{new_key}:{value}")
				#new = line.replace(key, new_key)
				#newtree += new
				fo.write('\t\'' + new_key + '\'' + colors[new_key[-1]] + '\n')

				nline=re.sub(key+':',newname[key]+':',nline)

		#print(replaced_patterns)
		#print(replaced_patterns)
		#exit()

		newtree+=nline
		#print(newtree)
		#exit()
		'''
		for key, value in newname.items():
			if key in line:
				new = line.replace(key, value)
				fo.write('\t\'' + new + '\'' + colors[value[-1]] + '\n')
		
		while i < len(line):
			if line[i] not in ['T']: #T for TL...
				newtree+=line[i]
				i+=1
			else:
				oldname=line[i:i+5]
				i=i+5
				if oldname in newname.keys():
					new=newname[oldname]
				else:
					new=oldname
				print(new)
				exit()
				if new[-1]=='N':
					new=new[:-1]+'?'

				print(new)
				newtree+=new
				fo.write('\t\''+new+'\''+colors[new[-1]]+'\n') #write down new colors to nexus file
		'''
	#exit()
	#write rest of nexus file
	fo.write(';\nend\n\nbegin trees;\n\ttree tree_1=[&R] ')
	for line in newtree:
		#line=re.sub(',reference_genome:0.00000,inferred_ancestor:0.00000','',line)
		#print(line)
		fo.write(line)
	#exit()
	fo.write('end;\n\nbegin figtree;\n')
	fo.write('\tset tipLabels.fontSize=10;\n')
	fo.write('\tset rectilinearLayout.curvature=7000;\n')
	fo.write('\tset layout.expansion=700;\n')
	fo.write('end;')
	fo.close()
	#exit()
	return newtree	







# if __name__ == "__main__":
# 		print('WARNING: Uses first strain as LCA')
# 		#out='tree_test'
# 		dir_output='../kcp_science_local_new'
#
# 		mutationtypes(dir_output+"/snv_tree_genome_latest.nwk.tree",dir_output+'/snv_table_merge_all_mut_annotations.tsv',dir_output)
# 		print('Done.')


		
		

