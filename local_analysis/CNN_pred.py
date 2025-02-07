import os
import re
import sys
#from sklearn.model_selection import StratifiedKFold
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix, classification_report
script_dir = os.path.dirname(os.path.abspath(__file__))
dir_py_scripts = script_dir+"/modules"
sys.path.insert(0, dir_py_scripts)
import snv_module_recoded_new as snv # SNV calling module

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Make major, minor, other_1, other_2 as the channel
class CNNModel(nn.Module):
    def __init__(self, n_channels, num_classes=1):
        super(CNNModel, self).__init__()

        # 1x1 Convolution to capture channel information
        # self.conv1x1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=(3,4), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=(1, 0))
        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=(1, 0))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Input shape: (batch_size, sample_num, features, n_channels->ATGC)
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)  # Rearrange to (batch_size, n_channels, x_dim, y_dim)
        # print(x.shape)
        # x = torch.relu(self.conv1x1(x))
        x = torch.relu(self.conv1(x))
        # print(x.shape)
        # exit()
        x = torch.relu(self.conv2(x))
        # print(x.shape)
        x = torch.relu(self.conv3(x))
        # print(x.shape)
        # exit()

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        # exit()
        return x

def cal_med_cov_for_given_array(x):
    nx=x.transpose(1, 0, 2)
    data=x.reshape(nx.shape[0],nx.shape[1]*nx.shape[2])
    non_zero_data = [row[row != 0] for row in data]
    median_values = np.array([np.median(row) if len(row) > 0 else 0 for row in non_zero_data])
    return median_values

def get_the_new_order(matrix):
    # Define the elements to check
    elements = np.array([1, 2, 3, 4])
    # Count the occurrences of each element along the rows
    counts = np.array([(matrix == e).sum(axis=1) for e in elements]).T
    # Sort elements by their counts in descending order
    sorted_indices = np.argsort(-counts, axis=1)
     # Get the sorted elements based on counts
    sorted_elements = np.take_along_axis(np.tile(elements, (matrix.shape[0], 1)), sorted_indices, axis=1)
    return sorted_elements

# Reorder the channel (ATCG to Max-min-3rd-4th)
def reorder_norm(combined_array, my_cmt):
    major_nt = my_cmt.major_nt.T
    order_base = get_the_new_order(
        major_nt)  # each row refers to one position, the 4 elements refer to the base index of "major", "minor", "other_1", "other_2"

    order_base -= 1
    reordered_array = np.take_along_axis(combined_array, order_base[:, np.newaxis, np.newaxis, :], axis=-1)
    ############ Order finished ##################
    first_two_rows = reordered_array[:, :, :2, :]
    sum_first_two = np.sum(first_two_rows, axis=(2, 3), keepdims=True)
    sum_first_two_fur = np.sum(sum_first_two, axis=1)
    exp_sum_first_two_fur = np.repeat(sum_first_two_fur, repeats=sum_first_two.shape[1], axis=1)
    exp_sum_first_two_fur = np.expand_dims(exp_sum_first_two_fur, axis=3)
    expanded_result = np.repeat(exp_sum_first_two_fur, repeats=4, axis=-1)
    last_row = np.max(reordered_array[:, :, -1:, :], axis=3, keepdims=True)
    expanded_last_row = np.repeat(last_row, repeats=4, axis=-1)
    normalized_first_two = reordered_array[:, :, :2, :] / expanded_result
    # Divide by the elements in the last row
    new_first_two = reordered_array[:, :, :2, :] / expanded_last_row
    new_first_two[new_first_two > 10] = 10
    normalized_first_two = np.nan_to_num(normalized_first_two, nan=0)
    new_first_two = np.nan_to_num(new_first_two, nan=0)
    new_array = reordered_array[:, :, :-1, :]
    final_array = np.concatenate([normalized_first_two, new_first_two, new_array], axis=2)
    return final_array

def find_sm_top_x(arr,x, mc):

    smallest_indices = np.argsort(arr.T, axis=1)[:, :x]
    # Create a boolean array initialized to False
    mask = np.zeros_like(arr.T, dtype=bool)
    # Use advanced indexing to set the smallest 3 elements to True
    np.put_along_axis(mask, smallest_indices, True, axis=1)
    mask=mask.T
    # Compute the median of each row
    medians = np.median(arr, axis=1)
    # Reshape the medians to align with the shape of the array
    medians = medians[:, np.newaxis]
    # Create a boolean mask for elements differing by more than 10 from the median
    mask2 = np.abs(arr - medians) > mc
    fmask=mask & mask2
    
    return fmask



def find_sm_top_x_test(arr, call_arr):
    #print(call_arr[40])
    #print(arr.T[40])
    #exit()
    #smallest_indices = np.argsort(arr.T, axis=1)[:, :3]
    masked_matrix = np.where(call_arr == 0, np.nan, call_arr)
    def most_frequent_nonzero(row):
        unique_elements, counts = np.unique(row[~np.isnan(row)], return_counts=True)
        if len(unique_elements) > 0:
            return unique_elements[np.argmax(counts)]
        else:
            return np.nan
    def least_frequent_nonzero(row):
        unique_elements, counts = np.unique(row[~np.isnan(row)], return_counts=True)
        if len(unique_elements) > 0:
            return unique_elements[np.argmin(counts)]
        else:
            return np.nan

    most_frequent_elements = np.apply_along_axis(most_frequent_nonzero, axis=1, arr=masked_matrix)
    least_frequent_elements = np.apply_along_axis(least_frequent_nonzero, axis=1, arr=masked_matrix)
    bool_max = call_arr == most_frequent_elements[:, np.newaxis]
    bool_min = call_arr == least_frequent_elements[:, np.newaxis]
    #print(arr.shape,bool_min.shape)
    #exit()
    c1 = arr.copy().T
    c2 = arr.copy().T
    #print(c1.shape,bool_min.shape)
    #exit()
    #print(c1[0], bool_max[0])
    #print(c2[0],bool_min[0])
    #exit()
    c1[~bool_max] = 0
    c2[~bool_min] = 0
    #print(c2[0])
    #exit()

    def median_no_zeros(row):
        non_zero_row = row[row != 0]  # remove 0
        if len(non_zero_row) == 0:  # if all 0, return 0
            return 0
        return np.median(non_zero_row)
    # Compute the median of each row
    #medians = np.apply_along_axis(median_no_zeros, axis=1, arr=c2)

    def mean_no_zeros(row):
        non_zero_row = row[row != 0]  # remove 0
        if len(non_zero_row) == 0:  # if all 0, return 0
            return 0
        return np.mean(non_zero_row)
    def std_no_zeros(row):
        non_zero_row = row[row != 0]  # remove 0
        if len(non_zero_row) == 0:  # if all 0, return 0
            return 0
        return np.std(non_zero_row)

    def iqr_no_zeros(row):
        non_zero_row = row[row != 0]  # 去掉0的元素
        if len(non_zero_row) == 0:  # 如果全是0，返回0
            return 0
        q75, q25 = np.percentile(non_zero_row, [75, 25])  # 计算75%分位数和25%分位数
        return 0.5*(q75 - q25)
    #mean_c1=np.apply_along_axis(mean_no_zeros,axis=1,arr=c1)
    median_c1 = np.apply_along_axis(median_no_zeros, axis=1, arr=c1)
    iqr_c1=np.apply_along_axis(iqr_no_zeros, axis=1, arr=arr)
    median_c1 = median_c1[:, np.newaxis]
    median_c1= np.tile(median_c1, (1, bool_min.shape[1]))
    #print(median_c1.shape,bool_min.shape)
    #print(median_c1[25])
    #exit()
    median_c1[~bool_min]=0
    #print(median_c1[25])
    #exit()
    iqr_c1=iqr_c1[:,np.newaxis]
    #print(median_c1.shape,c2.shape)
    res=median_c1-c2
    fmask = res > iqr_c1.T
    fmask=fmask.T

    return fmask

def remove_lp(combined_array,inp,my_cmt,my_calls, median_cov ):
    raw_p=len(inp)
    #print(np.where(inp==864972))
    #print(inp.shape,combined_array[40])
    #exit()
    ######### Further Scan bad pos, eg: potential FPs caused by Low-Depth samples
    #print(my_cmt.p)
    keep_col = []
    for pos in my_cmt.p:
        if pos not in inp:
            keep_col.append(False)
        else:
            keep_col.append(True)
    keep_col = np.array(keep_col)
    #print(keep_col.shape)
    #print(my_cmt.p.shape)
    #print(my_calls.p.shape)
    #exit()

    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)
    #exit()
    my_calls.filter_calls_by_element(
        my_cmt.fwd_cov < 1
    )

    my_calls.filter_calls_by_element(
        my_cmt.rev_cov < 1
    )

    
    my_calls.filter_calls_by_element(
        my_cmt.major_nt_freq < 0.7
    )
    #print(median_cov.shape,median_cov)
    #exit()
    if np.median(median_cov)>9:
        my_calls.filter_calls_by_element(
            ((my_cmt.rev_cov == 1 ) & (my_cmt.fwd_cov==1)) | ((my_cmt.rev_cov == 1 ) & (my_cmt.fwd_cov<4)) | ((my_cmt.rev_cov < 4 ) & (my_cmt.fwd_cov==1))
        )

    #print(combined_array.shape)
    #exit()

    if np.median(median_cov) > 20 and combined_array.shape[1]>50:
        #print(median_cov,median_cov.shape)
        #exit()
        my_calls.filter_calls_by_element(
            my_cmt.fwd_cov < 4
        )

        my_calls.filter_calls_by_element(
            my_cmt.rev_cov < 4
        )




    keep_col = remove_same(my_calls)
    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)

    keep_col_arr=[]
    for s in inp:
        if s in my_calls.p:
            keep_col_arr.append(True)
        else:
            keep_col_arr.append(False)
    keep_col_arr=np.array(keep_col_arr)
    inp=inp[keep_col_arr]
    combined_array=combined_array[keep_col_arr]
    #### Filter low gap pos
    med_cov_ratio_fwd=my_cmt.fwd_cov/median_cov[:, np.newaxis]
    med_cov_ratio_rev = my_cmt.rev_cov / median_cov[:, np.newaxis]
    med_cov_ratio_fwd = np.nan_to_num(med_cov_ratio_fwd, nan=0)
    med_cov_ratio_rev = np.nan_to_num(med_cov_ratio_rev, nan=0)
    #print(combined_array[40][21])
    #print(med_cov_ratio_fwd[40][21])
    #print(med_cov_ratio_rev[40][21])
    #print(combined_array.shape,my_cmt.fwd_cov.shape)
    #exit()
    # Old methods: Use loose filtering
    mask1 = find_sm_top_x(my_cmt.fwd_cov, 3, 20)
    mask2 = find_sm_top_x(my_cmt.rev_cov, 3, 20)
    if combined_array.shape[1]<25:
        mask1=find_sm_top_x(my_cmt.fwd_cov, 3, 20)
        mask2=find_sm_top_x(my_cmt.rev_cov, 3, 20)
    else:
        mask1 = find_sm_top_x(my_cmt.fwd_cov, 5, 20)
        mask2 = find_sm_top_x(my_cmt.rev_cov, 5, 20)
    ####### old done #########################
    #print(mask1.shape)
    #print(my_calls.calls.shape)
    #exit()
    #print(np.where(my_cmt.p==864972))
    #exit()
    # New methods: Test IQR-based filtering
    #mask1 = find_sm_top_x_test(my_cmt.fwd_cov,my_calls.calls.T)
    #mask2 = find_sm_top_x_test(my_cmt.rev_cov,my_calls.calls.T)
    #exit()
    #print(med_cov_ratio_fwd[:,25])
    ##print(med_cov_ratio_fwd)
    #print(mask1[:,25])
    #exit()


    mask1 = mask1 & (med_cov_ratio_fwd< 0.1)
    mask2 = mask2 & (med_cov_ratio_rev < 0.1)
    mask=mask1 & mask2
    my_calls.filter_calls_by_element(
        mask
    )
    #print(my_calls.p, len(my_calls.p))
    keep_col = remove_same(my_calls)
    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)
    #print(my_calls.p, len(my_calls.p))
    #exit()
    keep_col_arr = []
    for s in inp:
        if s in my_calls.p:
            keep_col_arr.append(True)
        else:
            keep_col_arr.append(False)
    keep_col_arr = np.array(keep_col_arr)
    inp = inp[keep_col_arr]
    combined_array = combined_array[keep_col_arr]

    print('There are ',raw_p-len(inp),' pos filtered. Keep ',len(inp),' positions.')
    #print(inp)
    #exit()
    return combined_array,inp


# filter low-quality samples and low-quality positions
def remove_low_quality_samples(inarray,thred,inpos):
    #print(inpos)
    #print(inarray.shape)
    trans = np.transpose(inarray, (1, 0, 2, 3))
    raw_sample=trans.shape[0]
    sum1=np.sum(trans,axis=3)
    sum2=np.sum(sum1,axis=2)
    # Count the number of zeros in each row
    zero_count = np.sum(sum2 == 0, axis=1)

    # Calculate the percentage of zeros
    percent_zeros = (zero_count / sum2.shape[1]) * 100
    trans=trans[percent_zeros<thred]
    sum1_new = np.sum(trans, axis=3)
    new_sample=trans.shape[0]
    trans = np.transpose(trans, (1, 0, 2, 3))
    print('Remove ',raw_sample-new_sample,' low-quality samples!')
    raw_pos=trans.shape[0]
    ########
    tem = trans[:, :, 4:6, :]
    #print(tem[6,:])
    ########## filter the position with the same type of base
    if raw_sample-new_sample>0:
        check=tem[:,:,:,1]==0
        check=np.sum(check,axis=1)
        check = np.sum(check, axis=1)
        check_2 = tem[:, :, :, 0] != 0
        # check 0-lines
        all_zero=tem==0
        az=np.sum(all_zero,axis=-1)
        zl=az==4
        find_minor=tem[:, :, :, 1] - tem[:, :, :, 0]
        find_minor=find_minor>0
        find_minor=~find_minor
        check_2=(check_2 | zl ) & find_minor

        check_2 = np.sum(check_2, axis=1)
        check_2 = np.sum(check_2, axis=1)
        keep_col_1=check!=trans.shape[1]*2

        keep_col_2=check_2==trans.shape[1]*2
        keep_col_2=~keep_col_2

        keep_col=keep_col_1 & keep_col_2

        trans=trans[keep_col]
        inpos=inpos[keep_col]
        
        unqiue_pos=trans.shape[0]
        print('Remove ',raw_pos-unqiue_pos,' same positions!')
        #print(inpos_out)
        # c=0
        # t=0
        # for i in inpos:
        #     if i not in inpos_out and inlab[c]==1:
        #         print(i)
        #         t+=1
        #         if t>9:
        #             exit()
        #     c+=1

        #print(np.where(inpos_out==676632))
        #exit()
        #exit()
        #else:
        #inpos_out=inpos
        #outlab=inlab

    ##########  filter low-quality positions
    ## - new try
    tem=trans[:, :, 4:6, :]

    ###### C31 refers to the minor sample
    c1=trans[:,:,0,1]>0
    c2=trans[:,:,1,1]>0
    c31=c1 | c2
    c31_b= c1 & c2 # pure minor

    ###### C32 -> Check whether both rev and fwd >0
    c1 = np.sum(tem[:,:,0,:],axis=-1)>0
    c2 = np.sum(tem[:, :, 1, :],axis=-1)>0
    c32= c1 & c2

    ###### C33 -> Both fwd and rev are pure (only 1 type of non-zero base)
    c3=np.sum(tem>0,axis=-1)==1
    c33=np.sum(c3,axis=-1)==2

    ###### C34 -> fwd and rew has different bases
    c1=np.argmax(tem[:,:,0,:],axis=-1)
    c2 = np.argmax(tem[:, :, 1, :], axis=-1)
    c34=c1!=c2
    call=c31 & c32 & c33 & c34
    pure_minor = c31_b & c32 & c33
    fc1=np.sum(call,axis=-1)>0
    fc2 = np.sum(pure_minor, axis=-1)<2
    ### Stat how many pure minor samples
    keep_col=fc1 & fc2
    keep_col=~keep_col
    raw_pos=len(keep_col)
    trans=trans[keep_col]
    inpos=inpos[keep_col]
    #inpos_out = inpos_out[keep_col]

    #outlab = outlab[keep_col]
    new_pos=trans.shape[0]
    print('Remove ',raw_pos-new_pos,' low-quality positions! Finally remains ',new_pos,' positions!')

    return trans,inpos

def trans_shape(indata):
    return np.transpose(indata, (1, 0, 2, 3))

def remove_same(my_calls_in):
	keep_col = []
	for i in range(my_calls_in.calls.shape[1]):
		unique_nonzero_elements = np.unique(my_calls_in.calls[:, i][my_calls_in.calls[:, i] != 0])
		if len(unique_nonzero_elements) < 2:
			my_calls_in.calls[:, i] = 0
			keep_col.append(False)
		else:
			keep_col.append(True)
	keep_col = np.array(keep_col)
	return keep_col

def load_p(infile):
    f=open(infile,'r')
    line=f.readline()
    d={}
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split('\t')
        d[int(ele[0])]=''
    return d

def data_transform(infile,incov,fig_odir):

    # infile='../../../Scan_FP_TP_for_CNN/Cae_files/npz_files/Lineage_10c/candidate_mutation_table_cae_Lineage_10c.npz'
    [quals, p, counts, in_outgroup, sample_names, indel_counter] = \
        snv.read_candidate_mutation_table_npz(infile)

    #print(quals.shape,indel_counter.shape,indel_counter.shape)
    #exit()

    ########  remove outgroup samples
    quals = quals[~in_outgroup]
    counts = counts[~in_outgroup]
    sample_names = sample_names[~in_outgroup]
    indel_counter = indel_counter[~in_outgroup]
    raw_cov_mat = snv.read_cov_mat_npz(incov)
    raw_cov_mat = raw_cov_mat[~in_outgroup]
    in_outgroup=in_outgroup[~in_outgroup]
    
    my_cmt = snv.cmt_data_object(sample_names,
                                 in_outgroup,
                                 p,
                                 counts,
                                 quals,
                                 indel_counter
                                 )
    my_calls = snv.calls_object(my_cmt)

    keep_col = remove_same(my_calls)

    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)
    
    quals = quals[:,keep_col]
    counts = counts[:,keep_col,:]
    p=p[keep_col]
    #sample_names = sample_names[in_outgroup]
    indel_counter=indel_counter[:,keep_col,:]
    median_cov = np.median(raw_cov_mat, axis=1)
    #print(median_cov.shape)
    #exit()
 
    ###### Single pos fig check

    #check=0
    '''
    for i in range(len(p)):
        # only for checking 13b
        #widp=load_p('../../../../../../Downloads/clabsi_project/analysis/output_new/Klebsiella_pneumoniae_P-13_b/snv_table_mutations_annotations.tsv')
        #if p[i] not in widp:continue
        #check+=1
        # done for 13 b
        snv.single_point_plot_herui_add_median(my_cmt.counts, i, str(p[i]), my_cmt.sample_names, 'Unknown-'+fig_odir , 'Col-pos-check-single/'+fig_odir, median_cov)
        snv.single_point_plot_herui(my_cmt.counts, i, str(p[i]), my_cmt.sample_names, 'Unknown-' + fig_odir,'Col-pos-check-single/' + fig_odir+'_no_median')
        #exit()
    #print(check)
    '''

    indata_32 = counts
    indel = indel_counter
    qual = quals
    indel= np.sum(indel, axis=-1)

    expanded_array = np.repeat(indel[:, :, np.newaxis], 4, axis=2)
    expanded_array_2 = np.repeat(qual[:, :, np.newaxis], 4, axis=2)
    med_ext = np.repeat(median_cov[:, np.newaxis], 4, axis=1)
    med_arr = np.tile(med_ext, (counts.shape[1], 1, 1))

    new_data = indata_32.reshape(indata_32.shape[0], indata_32.shape[1], 2, 4)
    new_data=trans_shape(new_data)
    
    indel_arr_final = np.expand_dims(expanded_array, axis=2)
    indel_arr_final=trans_shape(indel_arr_final)
    qual_arr_final = np.expand_dims(expanded_array_2, axis=2)
    qual_arr_final=trans_shape(qual_arr_final)
    med_arr_final = np.expand_dims(med_arr, axis=2)
    combined_array = np.concatenate((new_data, qual_arr_final, indel_arr_final, med_arr_final), axis=2)
    check_idx = 0
    c1 = (combined_array[..., :] == 0)
    x1 = (np.sum(c1[:, :, :2, :], axis=-2) == 2)
    mx = x1
    mxe = np.repeat(mx[:, :, np.newaxis, :], 5, axis=2)
    combined_array[mxe] = 0
    ####### Reorder the columns and normalize & split the count info
    '''
    keep_col = []
    # print(my_cmt.p)
    # print(inpos)
    # exit()
    for p in my_cmt.p:
        if diff_pos:
            if p + 1 not in inpos:
                keep_col.append(False)
            else:
                keep_col.append(True)
        else:
            if p not in inpos:
                keep_col.append(False)
            else:
                keep_col.append(True)
    keep_col = np.array(keep_col)
    my_cmt.filter_positions(keep_col)
    for p in inpos:
        if diff_pos:
            if p - 1 not in my_cmt.p:
                print('Pos not consistent! Exit!')
                exit()
        else:
            if p not in my_cmt.p:
                print('Pos not consistent! Exit!')
                exit()
    '''
    

    combined_array = reorder_norm(combined_array, my_cmt)
    #### Remove low quality samples
    #print(combined_array.shape,len(p))
    #exit()
    combined_array,p=remove_low_quality_samples(combined_array, 45,p)
    #print(np.where(p==864972))
    #exit()
	#### Remove bad positions
    combined_array,p=remove_lp(combined_array,p,my_cmt,my_calls,median_cov )
    

    return combined_array,p


def load_test_name(infile):
    dt={}
    f=open(infile,'r')
    while True:
        line=f.readline().strip()
        if not line:break
        dt[line]=''
    return dt

def CNN_predict(data_file_cmt,data_file_cov,out):
    if not os.path.exists(out):
        os.makedirs(out)
    setup_seed(1234)
    # dr=load_test_name('../39features-train.txt')
    # dt=load_test_name('../39features-test.txt') # test dict
    #indir='CNN_select_10features_mask_balance_no_Sep'
    #indir='CNN_select_10features_mask_balance_align_slides'
    ########## Kcp paper ###########
    #indir='CNN_select_40features_science_kcp' # these datasets have correct labels now!!!
    # indir='../../../Other_datasets/ScienceTM_KCp/CNN_select_features_target_10features_kcp'
    # in_npz='../../../Other_datasets/ScienceTM_KCp/npz_files_from_server'
    ########## elife-Sau paper ###########
    # indir='../../../Other_datasets/eLife-Sau-2022/CNN_select_features_target_10features_sau'
    # in_npz='../../../Other_datasets/eLife-Sau-2022/npz_files_from_server'
    ########## PNAS-Sau paper ###########
    # indir='../../../Other_datasets/PNAS-Sau-2014/CNN_select_features_target_10features_sau'
    # in_npz='../../../Other_datasets/PNAS-Sau-2014/npz_files_from_server'
    incount = data_file_cmt
    incov = data_file_cov
    '''
    for filename in os.listdir(in_npz_dir):
        # if re.search('DS',filename):continue
        # data = np.load(indir + '/' + filename + '/cnn_select.npz')
        # pre=re.sub('_Bfg','',filename)
        # pre=re.sub('_Cae','',pre)
        # pre = re.sub('_Sep', '', pre)
        # ind=in_npz+'/'+pre
        if re.search('mutation',filename):
            incount=in_npz_dir+'/'+filename
            pre=re.sub('_candidate_mutation_table.npz','',filename)
            pre=re.sub('group_','',pre)
        if re.search('coverage_matrix_raw',filename):
            incov=in_npz_dir+'/'+filename
    '''
    if not os.path.exists(incount) or not os.path.exists(incov):
        print('Mutation table or coverage matrix is not available! Please check! Exit.')
        exit()
    ########## From Collaborator data - Pseudomonas_aeruginosa_P-12 #######
    #mut='/Users/liaoherui/Downloads/clabsi_project/data/candidate_mutation_tables/group_29_candidate_mutation_table.pickle.gz'
    mut=incount
    #cov='/Users/liaoherui/Downloads/clabsi_project/data/candidate_mutation_tables/group_29_coverage_matrix_raw.pickle.gz'
    cov=incov
    #pre='Ecoli-P33'
    #pre='Kpn-P4'
    #pre='Pae-P14'
    fig_odir=out
    # indir='CNN_select_features'
    #train_datasets=[]
    test_datasets=[]
    #dsize=[]

    info=[]
    #print('Test data :'+pre)
    odata,pos=data_transform(mut,cov,fig_odir)
    #odata=odata[np.where(pos==864972)]
    #odata=odata[:,20:23,:,:]
    #print(odata,odata.shape)
    #exit()
    #print(odata.shape,pos)
    #exit()
    #print(odata[5])
    #exit()
    print('Transformed data shape:',odata.shape)
    #test_datasets.append((data['x'][:,:,8:].astype(np.float64), data['label']))
    nlab=np.zeros(odata.shape[0])
    #test_datasets.append((odata))
    test_datasets.append((odata,nlab))
    #odata[2][5][2][1]=1
    #print(odata.shape)
    #exit()
    #exit()
    #test_datasets.append((data['x'][:, :, index].astype(np.float64), data['label']))
    for p in pos:
        info.append(str(p))
    #dsize.append(len(nlab))
    #return info

    info = np.array(info)
    #dsize=np.array(dsize)
    #exit()
    #train_datasets=np.delete(datasets, selected_indices, axis=0)
    #train_datasets=datasets[keep]
    #test_datasets = datasets[selected_indices]

    # #### stat datasets info
    # def stat(datasets,pre):
    #     p=0
    #     n=0
    #     for i,(data,label) in enumerate(datasets):
    #         n+=np.count_nonzero(label == 0)
    #         p+=np.count_nonzero(label == 1)
    #     print(pre,' dataset has ', len(datasets),' lineages, ',p,' true SNPs, ',n,' false SNPs, total', n+p,' SNPs',flush=True)
    #
    # #stat(train_datasets,'Training')
    # stat(test_datasets,'Test')
    # #exit()


    #Create DataLoaders
    def create_dataloader(datasets):
        dataloader=[]
        for i, (data,label)  in enumerate(datasets):
            dataset = CustomDataset(data, label)
            #dataset2 = CustomDataset(data2, labels2)
            dataloader_tem = DataLoader(dataset, batch_size=512, shuffle=False)
            #dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=True)
            dataloader.append(dataloader_tem)
        return dataloader

    # #train_loader=create_dataloader(train_datasets)
    test_loader=create_dataloader(test_datasets)


    # Training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('The device you are using is: ',device,flush=True)
    #model = CNNModel_2(n_channels=11).to(device)
    model= CNNModel(n_channels=4).to(device)

    model.load_state_dict(torch.load(script_dir+'/CNN_models/checkpoint_best_3conv.pt'))
    
    # #weight = torch.tensor([0.01, 0.99])
    # #criterion = nn.BCEWithLogitsLoss(weight=weight)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # early_stopping = EarlyStopping(patience=20, verbose=True)

    # valid_losses=[]
    # num_epochs = 500
    #o=open('check_res_cnn_39features_multichannel_science_kcp_with_reorder_split_m2_basechannel_large_remove.txt','w+')
    #o=open('check_res_cnn_39features_multichannel_elife_sau_with_reorder_split_m2_basechannel_large_remove.txt','w+')
    #o=open('check_res_cnn_39features_multichannel_pnas_sau_with_reorder_split_m2_basechannel_large_remove.txt','w+')
    #o=open('check_res_cnn_39features_multichannel_elife_sau_with_mask.txt','w+')
    o=open(out+'/cnn_res.txt','w+')
    o.write('Pos_info\tPredicted_label\tProbability\n')

    #print('Train')
    model.eval()

    predictions = []
    y_test=[]
    with torch.no_grad():
        for loader in test_loader:
            for inputs,label in loader:
                #print(inputs)
                #exit()
                inputs=inputs.to(device)
                #inputs = torch.from_numpy(np.float32(inputs)).to(device)
                # print(inputs[20])
                # exit()
                outputs = model(inputs)
                # print(model(inputs))
                # exit()
                predictions.extend(outputs.cpu().numpy().flatten())
    #loss = criterion(outputs, y_test)
    # Convert predictions to binary labels
    prob=np.array(predictions)
    predictions = (np.array(predictions) > 0.5).astype(int)
    #print(predictions,prob)
    #exit()
    # print(predictions)
    y_pred = predictions
    c=0
    #print('Predicted results:',np.count_nonzero(y_pred),' true SNPs, ',len(y_pred)-np.count_nonzero(y_pred),' false SNPs.')
    for s in y_pred:
        o.write(info[c]+'\t'+str(s)+'\t'+str(prob[c])+'\n')
        c+=1
    # Return all reamining positions and CNN's predicted probabilities array
    return pos,y_pred,prob
    # accuracy = accuracy_score(y_test, y_pred)
    # # Calculate precision
    # precision = precision_score(y_test, y_pred)
    #
    # # Calculate recall (sensitivity)
    # recall = recall_score(y_test, y_pred)
    # # Calculate F1-score
    # f1 = f1_score(y_test, y_pred)
    # #roc_auc = roc_auc_score(y_test, y_pred)
    # print('Test dataset accuracy is:', accuracy, ' precision:', precision, ' recall:', recall, ' f1-score:', f1, flush=True)
    # print('Test dataset accuracy is:', accuracy, ' precision:', precision, ' recall:', recall, ' f1-score:', f1, flush=True,file=o)
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # class_report = classification_report(y_test, y_pred)
    # print('Confusion matrix:', conf_matrix, '\nClassification report:', class_report)
    #
    #
    # #print(new_data.shape,len(new_data))
    # #torch.save(model.state_dict(),'cae_7000plus_cnn_af.pt')

# Herui's test
#pos,pred,prob=CNN_predict('../npz_of_Test_data/cae_pe/candidate_mutation_table','cae_pe_local')
#print(pos,pred,prob)
