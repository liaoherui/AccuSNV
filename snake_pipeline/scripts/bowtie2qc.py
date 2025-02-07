'''
Inputs:
    paths_to_samples

Outputs:
    alignment_stats.csv

Change Log:
    2022.10.13, Arolyn: changed hardcoded directory structure to be compatible with GUS (though ideally this should not be hardcoded in the first place)
'''
import math
import os
import shutil
import argparse
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description='''\
                            Generates QC stats on the bowtie2 rule
                               ''',
                               epilog="Questions or comments? --> aholton@mit.edu")
parser.add_argument("-s", dest="samples", help="csv file with samples",required=True,action='store')
parser.add_argument("-r", dest="refGenome", help="name of reference genome as string",required=True,action='store')
parser.add_argument("-d", dest="currentDirectory", help="current directory of Snakefile", required=True, action="store")
parser.add_argument("-o", dest="outfileString", help="name of outfile without file extension", required=True, action="store")
args = parser.parse_args()

def main(path_to_samples, reference_genome, current_directory, out_file_string):

    cwd = current_directory
    home_dir = cwd.split("/")[-1]

    # make copy of samples.csv to add to
    print("Copying samples csv...")
    shutil.copyfile(path_to_samples, cwd + "/" + out_file_string + ".csv")

    # make new columns in csv for the alignment stats
    alignment_stats = pd.read_csv(cwd + "/" + out_file_string + ".csv",index_col=False) # index_col=False needed so that Path does not become row name
    #alignment_stats.iloc[0]
    alignment_stats = alignment_stats.astype(str)
    alignment_stats["Number Aligned Once"] = ""
    alignment_stats["Percent Aligned Once"] = ""
    alignment_stats["Percent Overall Alignment"] = ""

    # grab relevant info from log files
    os.system("grep exactly 1-Mapping/bowtie2/bowtie2_*_ref_" + reference_genome + ".txt" " >> " + cwd + "/" + out_file_string + ".txt")
    os.system("grep overall 1-Mapping/bowtie2/bowtie2_*_ref_" + reference_genome + ".txt" " >> " + cwd + "/" + out_file_string + ".txt") 

    sample_id_to_number_once  = {}        
    sample_id_to_percent_once = {}
    sample_id_to_overall      = {}

    stats_file = open(cwd + "/" + out_file_string + ".txt")
    for line in stats_file:
        split_on_rule_name = line.split("bowtie2_")
        split_on_ref = split_on_rule_name[1].split("_ref_")
        sample_id = split_on_ref[0]
        split_on_colon = split_on_ref[1].split(":")
        stat_info = split_on_colon[1]
        if "exactly" in stat_info:
            info = stat_info.split()
            number = info[0]
            percent = info[1].replace("(", "").replace(")","").replace("%", "")
            sample_id_to_number_once[sample_id] = number
            sample_id_to_percent_once[sample_id] = percent
        elif "overall" in stat_info:
            info = stat_info.split()
            percent = info[0].replace("%", "")  
            sample_id_to_overall[sample_id] = percent

    for idx, row in alignment_stats.iterrows():
        sample_id_from_csv = row["Sample"]
        ref_id_from_csv = row["Reference"]
        if reference_genome in ref_id_from_csv:
            if sample_id_from_csv in sample_id_to_number_once:
                row["Number Aligned Once"] = sample_id_to_number_once[sample_id_from_csv]
                row["Percent Aligned Once"] = sample_id_to_percent_once[sample_id_from_csv]
                row["Percent Overall Alignment"] = sample_id_to_overall[sample_id_from_csv]
            else:
                row["Number Aligned Once"] = "-1"
                row["Percent Aligned Once"] = "-1"
                row["Percent Overall Alignment"] = "-1"
    # Remove rows where samples weren't aligned to this reference genome
    alignment_stats = alignment_stats[(alignment_stats["Number Aligned Once"] != "")]


    # for each stat, make a histogram across the samples
    number_samples = alignment_stats.shape[0]
    number_bins = math.ceil(math.sqrt(number_samples))
    # number aligned once histogram
    number_aligned_once = alignment_stats["Number Aligned Once"].tolist()
    number_aligned_once = [int(number) for number in number_aligned_once]
    number_aligned_once = np.array(number_aligned_once)
    hist, bin_edges = np.histogram(number_aligned_once, bins=number_bins)
    plt.hist(number_aligned_once, bins=bin_edges, edgecolor='white', linewidth=1.0)
    plt.title("Number of Paired Reads That Aligned Once")
    plt.xlabel("Number of Paired Reads")
    plt.ylabel("Number of Samples")
    plt.savefig(cwd + "/" + out_file_string + "_number_aligned_once_histogram.png")

    # percent aligned once histogram
    percent_aligned_once = alignment_stats["Percent Aligned Once"].tolist()
    percent_aligned_once = [float(percent) for percent in percent_aligned_once]
    percent_aligned_once = np.array(percent_aligned_once)
    hist, bin_edges = np.histogram(percent_aligned_once, bins=number_bins)
    plt.clf()
    plt.hist(percent_aligned_once, bins=bin_edges, edgecolor='white', linewidth=1.0)
    plt.title("Percent of Paired Reads That Aligned Once")
    plt.xlabel("Percent of Paired Reads")
    plt.ylabel("Number of Samples")
    plt.savefig(cwd + "/" + out_file_string + "_percent_aligned_once_histogram.png")

    # overall alignment histogram
    overall_alignment = alignment_stats["Percent Overall Alignment"].tolist()
    overall_alignment = [float(percent) for percent in overall_alignment]
    overall_alignment = np.array(overall_alignment)
    hist, bin_edges = np.histogram(overall_alignment, bins=number_bins)
    plt.clf()
    plt.hist(overall_alignment, bins=bin_edges, edgecolor='white', linewidth=1.0)
    plt.title("Overall Alignment Percent")
    plt.xlabel("Percent of Paired Reads")
    plt.ylabel("Number of Samples")
    plt.savefig(cwd + "/" + out_file_string + "_overall_alignment_histogram.png")

    # save the csv file in the bowtie2 step folder
    path_to_save_csv = cwd + "/" + out_file_string + ".csv" 
    alignment_stats.to_csv(path_to_save_csv) # alignment_stats.to_csv(path_to_save_csv, index = False)
    print("Done with bowtie2 QC")

if __name__ == "__main__":
    path_to_samples=args.samples
    reference_genome=args.refGenome
    current_directory=args.currentDirectory
    out_file_string=args.outfileString
    main(path_to_samples, reference_genome, current_directory, out_file_string)
