"""Phylogenetic code.
Everything is written to a phylogeny/ dir under the output dir, because dnapars writes
to fixed filenames in the working directory and so needs one of its own.
Outputs:
1. snv_tree_final.nwk.tree: the parsimony tree, from dnapars
2. snv_trees/*.tree: one tree per SNV, tips coloured by basecall (off by default),
3. snv_table_tree_distances.tsv: the dMRCA tree-distance table.
4. snv_table_simple_stats.tsv
5. the dnapars scratch files (alignment.phylip, dnapars.log, dnapars_report.txt, tree.nexus)

Both dMRCA tables cover the SNVs in snv_table_final.tsv that are not flagged as recombinant;
the tree itself still uses every called position except the recombinant ones.

Input
_snv_state.npz
Reference FASTA
snv_table_unfiltered.tsv (per-SNV trees) and snv_table_final.tsv (the dMRCA SNV set)

Transcribed from 'new_snv_script.py' in the AccuSNV ~v1 repo,
lines ~1255-1296, 1342-1347 and 1535-1571.
"""
import os
import re
import logging
import argparse
import traceback
import subprocess

import numpy as np
import pandas as pd
from matplotlib import rc
from Bio import Phylo
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

from accusnv import log as accusnv_log
from accusnv.downstream import snv

log = logging.getLogger('accusnv')

rc('font', **{'sans-serif':'Arial'})


def generate_tree(calls_for_tree,treeSampleNamesLong,sampleNamesDnapars,out_tree,use_nj=False):
    '''
    Creates a parsimony tree (calling dnapars) or, with use_nj, a neighbor-joining tree
    (Biopython, no external tool) from the provided basecalls at SNV positions. Both write
    the same two files: tree.nexus and out_tree (newick), tips labelled with the full sample
    names. Runs in the current directory, which the caller sets to the group's phylogeny/
    dir: dnapars always writes its report and tree to the fixed names 'outfile' and
    'outtree', so it needs a directory of its own.
    '''
    write_calls_sampleName_to_fasta(calls_for_tree,treeSampleNamesLong,"alignment")

    if use_nj:
        log.info('Building a neighbor-joining tree from %d samples x %d SNV positions',
                 len(treeSampleNamesLong), calls_for_tree.shape[0])
        tree = DistanceTreeConstructor(DistanceCalculator('identity'), 'nj').build_tree(
            AlignIO.read("alignment.fa", 'fasta'))
        # Match the shape of the dnapars tree: no internal node labels, and no negative branch
        # lengths (NJ produces them, and the per-SNV tree annotation cannot parse them).
        for clade in tree.find_clades():
            if not clade.is_terminal():
                clade.name = None
            clade.branch_length = max(clade.branch_length or 0.0, 0.0)
    else:
        # dnapars truncates tip labels to 10 characters, so it gets the numbered names instead;
        # they are swapped back for the real sample names once the tree is built.
        write_calls_sampleName_to_fasta(calls_for_tree,sampleNamesDnapars,"alignment_dnapars")
        AlignIO.write(AlignIO.read("alignment_dnapars.fa", 'fasta'), "alignment.phylip", "phylip")
        os.remove("alignment_dnapars.fa")

        # Keystrokes for the interactive dnapars menu: the input file, the settings the original
        # pipeline used, then y to accept them.
        with open("dnapars_options.txt",'w') as file:
            file.write("alignment.phylip\n5\nV\n1\ny\n\n")

        log.info('Building a maximum-parsimony tree with dnapars from %d samples x %d SNV positions',
                 len(treeSampleNamesLong), calls_for_tree.shape[0])
        # dnapars prompts for a new filename if these already exist, which desyncs the keystrokes
        # above and aborts the run; a previous crash is the only way to find them here.
        for leftover in ("outfile", "outtree"):
            if os.path.exists(leftover):
                os.remove(leftover)
        subprocess.run(["dnapars < dnapars_options.txt > dnapars.log"],shell=True,check=True)
        os.rename("outfile", "dnapars_report.txt")

        # re-write tree with new long tip labels
        tree = Phylo.read("outtree", "newick")
        for leaf in tree.get_terminals():
            idx = np.where(sampleNamesDnapars==leaf.name)[0]
            if len(idx) > 1:
                log.warning('Tree tip %s matches more than one sample: dnapars truncates names to 10 '
                            'characters, so two of your samples collide there', leaf.name)
            leaf.name = treeSampleNamesLong[idx[0]]
        os.remove("outtree")

    Phylo.write(tree, "tree.nexus", 'nexus')
    Phylo.write(tree, out_tree, 'newick')


def write_calls_sampleName_to_fasta(calls_for_tree,treeSampleNames,filename):
    fa_file = open(filename+".fa", "w")
    for i,name in enumerate(treeSampleNames):
        nucl_string = "".join(list(calls_for_tree[:,i]))
        fa_file.write(">" + name + "\n" + nucl_string + "\n")
    fa_file.close()


def mutation_count(inputtree):
	tree=''
	for i,c in enumerate(inputtree):
		if c in ['(', ',', ')']:
			tree+=c
		elif c =='*':
			tree+=inputtree[i+1]

	mutlist={('A','C'):0,('G','C'):0,('T','C'):0,('A','G'):0,('C','G'):0,('T','G'):0, ('A','T'):0,('G','T'):0,('C','T'):0, ('T','A'):0,('G','A'):0,('C','A'):0}
	dcount={}
	for i in tree:
		# Anything that is not a real base (N, ?, ambiguity codes, the brackets and commas
		# carried over from the tree) is missing data and does not count as a mutation.
		if i not in 'ACGT':continue
		if i not in dcount:
			dcount[i]=1
		else:
			dcount[i]+=1
	res=sorted(dcount.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
	anc=res[0][0]

	for i in res:
		if i[0] ==anc:continue
		mutlist[anc,i[0]]+=i[1]

	muts=0
	for pair in mutlist.keys():
		muts+=mutlist[pair]

	if muts==0:
		log.debug('No mutations on this per-SNV tree: %s', tree)

	return muts, mutlist


def convert_csv(chart_raw,out):
	chart = out + '/snvChart.csv'
	df = pd.read_csv(chart_raw, sep='\t')
	# The per-sample basecalls are the columns between 'possible_AAs_at_site' and 'sequence';
	# everything else is annotation or a CNN/filter verdict, which the chart does not show.
	columns = df.columns.tolist()
	raw_pos = np.array(df['genome_pos'])
	samples = columns[columns.index('possible_AAs_at_site') + 1 : columns.index('sequence')]
	subset = df[['contig', 'contig_pos'] + samples]
	subset = subset.rename(columns={'contig': 'chr', 'contig_pos': 'pos'})
	subset = subset.map(lambda x: x.replace(',', '') if isinstance(x, str) else x)
	subset.to_csv(chart, index=False)

	return chart,raw_pos


def mutationtypes(tree, chart_raw,out,skip_positions=None):

	chart,raw_pos=convert_csv(chart_raw,out)

	tout=out+'/snv_trees'
	if not os.path.exists(tout):
		os.makedirs(tout)

	f=open(chart,'r').readlines()

	pidx=0
	for i, line in enumerate(f[1:]):

		# Skip recombinant SNVs: they are flagged in the tables but excluded from tree building.
		if skip_positions and int(raw_pos[pidx]) in skip_positions:
			pidx+=1
			continue

		l=line.strip().split(',')
		if len(l) < 5:
			log.debug('Skipping short row in the SNV chart: %s', l)
			pidx+=1
			continue
		chromosome=l[0]
		pos=l[1]

		#count mutations
		newtree = annotateSNP(tree, chart, chromosome, pos)

		a, mutlist= mutation_count(newtree)

		if a==0:
			log.debug('No mutations mapped onto the tree for %s position %s; no per-SNV tree written',
			          chromosome, pos)
		#save trees
		else:
			f1=open(tout+'/p_'+str(raw_pos[pidx])+'_'+str(a)+'.tree','w')
			t=open('tempnexus.txt','r').readlines()
			for line in t:
				f1.write(line)
		pidx+=1
	os.system('rm temptree.txt')
	os.system('rm tempnexus.txt')


def annotateSNP(tree, dict, chromosome,pos):
	f=open(dict).readlines()
	fo=open('temptree.txt','w')

	header=f[0].strip().split(',')
	locations=[]

	for n,i in enumerate(header):
		if not re.search('pos',i) and not re.search('chr',i) :
			locations.append(n)

	for line in f:
		l=line.strip('\n').split(',')
		if l[0]==chromosome and l[1]==pos:
			for i in locations:
				if len(l) > i and len(l[i])>0:
					fo.write(header[i]+'\t'+l[i]+'\n')
				else:
					fo.write(header[i]+'\t?\n')
				if i > len(l):
					log.debug('SNV chart row is shorter than its header at column %d: %s', i, line)
			break
	fo.close()
	newtree = nameswap(tree,'temptree.txt')
	return newtree


def nameswap(tree, dictionary):

	fraw=open(dictionary).readlines()
	f=[]
	for r in fraw:
		if not re.search('muts\t',r) and not re.search('type\t',r):
			f.append(r)
	numStrains=len(f)
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

	#make new tree
	f=open(tree,'r').readlines()

	newtree=''

	for line in f:
		line=line.strip()
		line=re.sub(',reference_genome:0.00000,inferred_ancestor:0.00000','',line)
		# Regular expression to extract key:value pairs
		pattern = r"([\w\-]+:\d+\.\d+)"

		# Find all matches in the string
		matches = re.findall(pattern, line)

		nline=line
		# Replace keys using the dictionary and store the results
		for match in matches:
			key, value = match.split(":")
			if key=='reference_genome' or key=='inferred_ancestor' or newname[key][-1] not in colors:
				log.debug('Leaving tip %s uncoloured in the per-SNV tree', key)
				continue
			if key in newname:
				# Replace the key with the corresponding value from the dictionary
				new_key = newname[key]
				fo.write('\t\'' + new_key + '\'' + colors[new_key[-1]] + '\n')

				nline=re.sub(key+':',newname[key]+':',nline)

		newtree+=nline

	#write rest of nexus file
	fo.write(';\nend\n\nbegin trees;\n\ttree tree_1=[&R] ')
	for line in newtree:
		fo.write(line)
	fo.write('end;\n\nbegin figtree;\n')
	fo.write('\tset tipLabels.fontSize=10;\n')
	fo.write('\tset rectilinearLayout.curvature=7000;\n')
	fo.write('\tset layout.expansion=700;\n')
	fo.write('end;')
	fo.close()
	return newtree


def parse_args():
    p = argparse.ArgumentParser(prog='Script for building trees of AccuSNV SNVs, using dnapars.')
    p.add_argument('-r', '--rer', dest='ref_genome', type=str, required=True, help="Reference genome FASTA (or its directory)")
    p.add_argument('-i', '--input_dir', dest='input_dir', type=str, required=True, help="Group dir holding the SNV tables.")
    p.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True, help="Where to write the trees and distance tables")
    p.add_argument('-g', '--group', type=str, default='', help="Group ID, recorded in the stats table (default: the output dir name)")
    p.add_argument('--skip_trees', action='store_true',
                   help="Skip building the parsimony tree (dnapars). dMRCA tables still run.")
    p.add_argument('--use_nj_tree', action='store_true',
                   help="Build a neighbor-joining tree (Biopython) instead of a parsimony tree (dnapars).")
    p.add_argument('--exclude_positions', default='',
                   help="File of genome_pos values, one per line, to leave out of the SNV set.")
    p.add_argument('--include_positions', default='',
                   help="File of genome_pos values, one per line, to keep whatever the filters said.")
    accusnv_log.add_args(p)
    p.add_argument('--build_snv_trees', action='store_true',
                   help="Write one annotated NEXUS tree per SNV, tips coloured by basecall (for FigTree).")
    return p.parse_args()


def main():
    args = parse_args()
    accusnv_log.setup('tree', args.log)
    # Absolute, because everything below runs from the phylogeny dir.
    source = os.path.abspath(args.input_dir)
    dataset = args.group or os.path.basename(os.path.abspath(args.output_dir))
    d = os.path.abspath(args.output_dir) + '/phylogeny'
    os.makedirs(d, exist_ok=True)
    os.chdir(d)
    # A group with no SNVs gets no tree, but the pipeline declares this file as an output of
    # the stage, so create an empty file at first.
    tree_file = d + '/snv_tree_final.nwk.tree'
    open(tree_file, 'w').close()

    if bool(np.load(source + '/_snv_state.npz', allow_pickle=True)['fast_path']):
        log.warning('Group %s: this run took the fast path (or found no SNVs), so no tree is built.',
                    dataset)
        return

    st = snv.rebuild_state(source + '/_snv_state.npz', args.ref_genome,
                           snv.read_positions_file(args.exclude_positions),
                           snv.read_positions_file(args.include_positions))
    # Recombinant SNVs are flagged in the tables but excluded from all phylogenetics by default
    keep = np.logical_not(st.recomb_goodpos_all)

    # Create Parsimony tree (dnapars) 
    if st.num_goodpos_all > 0 and not args.skip_trees:
        sampleNamesDnapars = ["{:010d}".format(i) for i in range(st.my_cmt_goodpos_all.num_samples)]
        calls_ancestral_for_tree = np.expand_dims(snv.ints2nts(st.calls_ancestral_goodpos_all), axis=0)
        calls_reference_for_tree = np.expand_dims(st.my_rg.get_ref_NTs(st.my_calls_tree.p), axis=0)
        calls_for_tree_all = np.concatenate(
            (calls_ancestral_for_tree, calls_reference_for_tree, st.calls_for_tree), axis=0)
        calls_for_tree_all = calls_for_tree_all[:, keep]  # drop recombinant positions from the alignment
        sampleNamesDnapars_all = np.append(['Sanc', 'Sref'], sampleNamesDnapars)
        treesampleNamesLong_all = np.append(['inferred_ancestor', 'reference_genome'], st.treesampleNamesLong)
        # Deliberately not caught: a failed tree is the whole point of this stage, so it should
        # fail the pipeline rather than quietly leave the empty file above in place.
        generate_tree(calls_for_tree_all.transpose(), treesampleNamesLong_all,
                          sampleNamesDnapars_all, tree_file, use_nj=args.use_nj_tree)

    # ---- Per-SNV trees: one NEXUS file per SNV in snv_trees, tips coloured by basecall for
    # FigTree. Needs the parsimony tree, so it is off whenever the tree is skipped. ----
    run_snv_trees = args.build_snv_trees and not args.skip_trees
    recomb_positions = set(int(x) for x in st.p_goodpos_all[st.recomb_goodpos_all])
    unfiltered = source + '/snv_table_unfiltered.tsv'
    if run_snv_trees and os.path.exists(unfiltered):
        # The table also lists candidates neither the model nor the filters called, so that
        # their removal has a recorded reason. Those are not SNVs and get no tree.
        table = pd.read_csv(unfiltered, sep='\t', dtype=str)
        uncalled = table[(table['Pred_label'] == '0') & (table['CNN_pred'].isin(['0', 'skip']))
                         & (table['WideVariant_pred'] == '0')]
        recomb_positions |= set(uncalled['genome_pos'].astype(int))
        try:
            mutationtypes(tree_file, unfiltered, d, skip_positions=recomb_positions)
        except Exception as e:
            log.warning('Group %s: could not build the per-SNV trees (%s). The main tree is '
                        'unaffected.', dataset, e)
            log.debug('Per-SNV tree traceback:\n%s', traceback.format_exc())
        finally:
            # mutationtypes deletes these on success; remove leftovers if it stopped early.
            for scratch in ('temptree.txt', 'tempnexus.txt'):
                if os.path.exists(scratch):
                    os.remove(scratch)

    # ---- dMRCA / tree distances (the SNVs in snv_table_final.tsv, minus the recombinant ones) ----
    # Distance = number of non-N tree positions where a sample differs from the inferred ancestor.
    # The final table is the only place the Pred_label verdict is recorded, so read it back.
    final = source + '/snv_table_final.tsv'
    keep_dmrca = np.zeros(len(st.p_goodpos_all), dtype=bool)
    if os.path.exists(final) and os.path.getsize(final) > 0:
        rows = pd.read_csv(final, sep='\t', usecols=['genome_pos', 'Whether_recomb'])
        keep_dmrca = np.isin(st.p_goodpos_all, rows.loc[rows['Whether_recomb'] == 0, 'genome_pos'])

    if st.num_goodpos_all > 0 and st.treesampleNamesLong is not None:
        cft = st.calls_for_tree[:, keep_dmrca]
        anc = snv.ints2nts(st.calls_ancestral_goodpos_all[keep_dmrca])[None, :]
        fixedmutation_tree = (cft != anc) & (cft != 'N')
        dist_to_anc_by_sample = np.sum(fixedmutation_tree, axis=1)
        with open(d + '/snv_table_tree_distances.tsv', 'w') as f:
            f.write('sample_name\tnum_SNVs_to_ancestor\n')
            for i, name in enumerate(st.treesampleNamesLong):
                f.write(str(name) + '\t' + str(int(dist_to_anc_by_sample[i])) + '\n')

        ingroup = np.logical_not(st.my_calls_tree.in_outgroup)
        dist_ingroup = dist_to_anc_by_sample[ingroup]
        if dist_ingroup.size > 0:
            dmrca_median = float(np.median(dist_ingroup))
            dmrca_min = int(np.min(dist_ingroup))
            dmrca_max = int(np.max(dist_ingroup))
        else:
            dmrca_median = dmrca_min = dmrca_max = 0
        num_samples_ingroup = int(np.sum(ingroup))
        num_snvs = int(np.sum(keep_dmrca))
        num_snvs_with_outgroup_allele = int(np.sum(st.calls_ancestral_goodpos_all[keep_dmrca] > 0))
        log.info('Group %s: %d ingroup samples sit a median of %.1f SNVs from the inferred common '
                 'ancestor (range %d-%d) over %d SNVs', dataset, num_samples_ingroup, dmrca_median,
                 dmrca_min, dmrca_max, num_snvs)
        with open(d + '/snv_table_simple_stats.tsv', 'w') as f:
            f.write('dataset\tgenome\tnum_samples_ingroup\tnum_snvs\tnum_snvs_with_outgroup_allele\t'
                    'dmrca_median\tdmrca_min\tdmrca_max\n')
            f.write(dataset + '\t' + str(st.ref_genome_name) + '\t' + str(num_samples_ingroup) + '\t' +
                    str(num_snvs) + '\t' + str(num_snvs_with_outgroup_allele) + '\t' +
                    str(dmrca_median) + '\t' + str(dmrca_min) + '\t' + str(dmrca_max) + '\n')


if __name__ == '__main__':
    main()
