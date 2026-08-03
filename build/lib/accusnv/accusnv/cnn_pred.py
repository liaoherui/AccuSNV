import os
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from importlib.resources import files
from accusnv.downstream import snv

log = logging.getLogger('accusnv')

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
        x = x.permute(0, 3, 1, 2)  # Rearrange to (batch_size, n_channels, x_dim, y_dim)
        # x = torch.relu(self.conv1x1(x))
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x

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

def cal_major_freq_in_call(arr1):
    col_data_nonzero = [arr1[:, col][arr1[:, col] != 0] for col in range(arr1.shape[1])]
    column_modes = [np.unique(col)[0] if len(np.unique(col)) == 1 else (1 if len(col) == 0 else np.argmax(np.bincount(col))) for col in col_data_nonzero]
    scount = np.sum(arr1 == column_modes, axis=0)
    return scount,arr1==column_modes




def scan_continue_gap_revise(arr):
    arr.sort()
    B = {}

    i, j = 0, 1
    n = len(arr)
    while i < n:
        j=i+1
        while j < n and arr[j] - arr[i] <= 100:
            if arr[j] - arr[i] > 0:
                if arr[i] not in B:
                    B[arr[i]]={arr[j]:''}
                else:
                    B[arr[i]][arr[j]]=''
                if arr[j] not in B:
                    B[arr[j]]={arr[i]:''}
                else:
                    B[arr[j]][arr[i]]=''

            j += 1
        i += 1
        if j <= i:
            j = i + 1
    res=[]
    for p in B:
        if len(B[p])>2:
            res.append(p)
    return res

def zscore_variant(large_array,small_array):
    mean_diff = np.mean(large_array) - np.mean(small_array)
    std_err_diff = np.std(large_array, ddof=1)
    z_score = mean_diff / std_err_diff
    p_value=1 - stats.norm.cdf(abs(z_score))
    return z_score,p_value

def compare_arrays_ttest(large_array_raw, small_array_raw):
    large_array=[x for x in large_array_raw if x != 0]
    large_array=np.array(large_array)

    small_array=[x for x in small_array_raw if x != 0]
    n_small = len(small_array)
    if n_small == 0:
        return "Empty array! Please check."

    elif n_small == 1:
        target_value = small_array[0]
        _, p_value = stats.ttest_1samp(large_array, popmean=target_value)
        # new p-cdf with z-score variant
        _,p_cdf=zscore_variant(large_array,small_array)
        return p_value,p_cdf

    else:
        _, p_value = stats.ttest_ind(large_array, small_array, equal_var=False)
        _,p_cdf=zscore_variant(large_array,small_array)
        return p_value,p_cdf

def check_mm(arr1,counts_major):
    col_data_nonzero = [arr1[:, col][arr1[:, col] != 0] for col in range(arr1.shape[1])]
    column_modes = [sorted([(x, i) for i, x in enumerate(np.bincount(col)) if x != 0], reverse=True)[1][1] if len(np.unique(col)) >= 2 and len([(x, i) for i, x in enumerate(np.bincount(col)) if x != 0]) > 1 else (1 if len(col) == 0 else 0) for col in col_data_nonzero]
    scount = np.sum(arr1 == column_modes, axis=0)
    only_minor= arr1 == column_modes
    check_minor_more_than_1=np.repeat([scount>1], arr1.shape[0], axis=0)
    cmj_copy=copy.deepcopy(counts_major)
    cmj_copy[~only_minor]=0
    min_v=[min(col[col != 0]) if len(col[col != 0]) > 0 else 0 for col in cmj_copy.T]
    copy_minv=np.repeat([min_v],arr1.shape[0],axis=0)
    x=counts_major-copy_minv
    x[only_minor]=-1

    minor_max = check_minor_more_than_1 & (x<0)

    return minor_max,arr1==column_modes

def process_arrays(arr1, arr2):
    col_data_nonzero = [arr1[:, col][arr1[:, col] != 0] for col in range(arr1.shape[1])]
    column_modes = [np.unique(col)[0] if len(np.unique(col)) == 1 else ( 1 if len(col)==0 else np.argmax(np.bincount(col))) for col in col_data_nonzero]
    scount = np.sum(arr1 == column_modes, axis=0)
    mask = arr1 != np.array(column_modes)
    arr2[mask] = 0
    result = np.sum((arr2 > 0) & (arr2 < 1), axis=0)

    return result/scount

def cal_freq_amb_samples(all_p,my_cmt):
    keep_col=[]
    for p in my_cmt.p:
        if p in all_p:
            keep_col.append(True)
        else:
            keep_col.append(False)
    keep_col=np.array(keep_col, dtype=bool)
    my_cmt.filter_positions(keep_col)
    freq_arr=process_arrays(my_cmt.major_nt,my_cmt.major_nt_freq)
    freq_d={}
    c=0
    for p in my_cmt.p:
        freq_d[p]=freq_arr[c]
        c+=1
    return freq_d,freq_arr

def remove_lp(combined_array,inp,my_cmt,my_calls, median_cov ):
    raw_p=len(inp)
    ######### Further Scan bad pos, eg: potential FPs caused by Low-Depth samples
    keep_col = []
    for pos in my_cmt.p:
        if pos not in inp:
            keep_col.append(False)
        else:
            keep_col.append(True)
    keep_col = np.array(keep_col, dtype=bool)

    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)
    my_calls.filter_calls_by_element(
        my_cmt.fwd_cov < 1
    )

    my_calls.filter_calls_by_element(
        my_cmt.rev_cov < 1
    )
    my_calls_check=copy.deepcopy(my_calls)
    scount_before,b=cal_major_freq_in_call(my_calls_check.calls)
    my_calls_check.filter_calls_by_element(
            (my_cmt.fwd_cov < 11) & (my_cmt.rev_cov < 11)
    )
    scount_after,b=cal_major_freq_in_call(my_calls_check.calls)
    fratio=scount_after/scount_before
    count=np.sum(fratio > 0.5)
    stat=count/len(fratio)
    my_cmt_tem=copy.deepcopy(my_cmt)
    _,freq_arr=cal_freq_amb_samples(my_calls.p,my_cmt_tem)
    freq_check=np.repeat([freq_arr>0],my_calls.calls.shape[0],axis=0) 
    if count>10:
        my_calls.filter_calls_by_element(
            (my_cmt.fwd_cov < 3) & (my_cmt.rev_cov < 3) & freq_check
        )
    else:
        if stat>=0.1:
            my_calls.filter_calls_by_element(
                    (my_cmt.fwd_cov < 3) & (my_cmt.rev_cov < 3) & freq_check
            )
    # if fwd type != rev type -> filter
    my_calls.filter_calls_by_element(
            (my_cmt.major_nt_fwd != my_cmt.major_nt_rev) & ((my_cmt.major_nt_fwd>0)&(my_cmt.major_nt_rev>0))
    )

    my_calls.filter_calls_by_element(
        my_cmt.major_nt_freq < 0.7
    )
    if np.median(median_cov)>9:
        my_calls.filter_calls_by_element(
            ((my_cmt.rev_cov == 1 ) & (my_cmt.fwd_cov==1)) | ((my_cmt.rev_cov == 1 ) & (my_cmt.fwd_cov<4)) | ((my_cmt.rev_cov < 4 ) & (my_cmt.fwd_cov==1))
        )


    if np.median(median_cov) > 20 and combined_array.shape[1]>50:
        my_calls.filter_calls_by_element(
            my_cmt.fwd_cov < 4
        )

        my_calls.filter_calls_by_element(
            my_cmt.rev_cov < 4
        )




    keep_col = snv.remove_same(my_calls)
    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)

    keep_col_arr=[]
    for s in inp:
        if s in my_calls.p:
            keep_col_arr.append(True)
        else:
            keep_col_arr.append(False)
    keep_col_arr=np.array(keep_col_arr, dtype=bool)
    inp=inp[keep_col_arr]
    combined_array=combined_array[keep_col_arr]
    #### Filter low gap pos
    rawp=my_cmt.p # used to check positions removed by gap filter
    # Add super big fp filt in gap filt
    a=np.array(my_cmt.counts_major, dtype=np.int64)
    b=np.array(my_cmt.counts_minor, dtype=np.int64)
    minor_max,mbool=check_mm(my_calls.calls,a) # Whether the major  counts all < the min minor count & # of minor > 1
    my_calls.filter_calls_by_element(
        (my_cmt.counts_minor>30) & ((a-b*3)>0) & (my_cmt.counts_major>250) & minor_max
    )

    # Add gap rule: if all minor sample <=10 and one part must <=5, >=85% major sample not satisfy this rule and >=20, then should be gap fp
    my_calls_tem=copy.deepcopy(my_calls)
    _,b=cal_major_freq_in_call(my_calls_tem.calls)
    count_combine_ratio=my_cmt.counts_major/median_cov[:, np.newaxis]
    c3=copy.deepcopy(count_combine_ratio)
    count_combine_ratio[~b]=0
    c3[~mbool]=0
    p_arr_ratio=[]
    p_arr_ratio_cdf=[]
    tem=[]
    for i in range(count_combine_ratio.shape[1]):
        c_arr=count_combine_ratio[:,i]
        d_arr=c3[:,i]
        c1=[x for x in c_arr if x != 0]
        d1=[x for x in d_arr if x != 0]
        tem.append([c1,d1])
        p,p_cdf=compare_arrays_ttest(c_arr,d_arr) # Ratio
        p_arr_ratio.append(p)
        p_arr_ratio_cdf.append(p_cdf)
    gap_candidate=[]
    tem_p=[]
    for i in range(len(p_arr_ratio)):
        if p_arr_ratio_cdf[i] <0.01:
            tem[i][0]=np.array(tem[i][0])
            if max(tem[i][1])<min(tem[i][0]) and max(tem[i][1])<0.05:
                if max(tem[i][0])>0.2:
                    gap_candidate.append(my_cmt.p[i])
        if p_arr_ratio_cdf[i] <0.01 or p_arr_ratio[i] <0.05:
            if np.max(tem[i][1])<0.1 and len(tem[i][1])<=3 and max(tem[i][0])>0.2:
                tem_p.append(my_cmt.p[i])
            tem[i][0]=np.array(tem[i][0])
    gap_pos_add=scan_continue_gap_revise(tem_p)
    gap_pos=gap_candidate

    for p in gap_pos_add:
        if p not in gap_pos:
            gap_pos.append(p)

    log.debug('%d positions sit next to a coverage gap and are marked in Gap_filter', len(gap_pos))

    keep_col_raw = snv.remove_same(my_calls)
    keep_col=[]
    c=0
    for k in keep_col_raw:
        if my_calls.p[c] in gap_pos:
            keep_col.append(False)
        else:
            keep_col.append(k)
        c+=1
    keep_col=np.array(keep_col, dtype=bool)
    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)
    pfgap=[p for p in rawp if p not in my_cmt.p] # position filterd due to gap
    keep_col_arr = []
    for s in inp:
        if s in my_calls.p:
            keep_col_arr.append(True)
        else:
            keep_col_arr.append(False)
    keep_col_arr = np.array(keep_col_arr, dtype=bool)
    inp = inp[keep_col_arr]
    combined_array = combined_array[keep_col_arr]

    log.debug('CNN input: dropped %d of %d positions that the CNN cannot score, keeping %d',
              raw_p - len(inp), raw_p, len(inp))
    log.debug('Positions not scored by the CNN: %s', [p for p in rawp if p not in inp])
    return combined_array,inp,pfgap,rawp


# filter low-quality samples and low-quality positions
def remove_low_quality_samples(inarray,thred,inpos):
    trans = np.transpose(inarray, (1, 0, 2, 3))
    raw_sample=trans.shape[0]
    sum1=np.sum(trans,axis=3)
    sum2=np.sum(sum1,axis=2)
    # Count the number of zeros in each row
    zero_count = np.sum(sum2 == 0, axis=1)

    # Calculate the percentage of zeros
    percent_zeros = (zero_count / sum2.shape[1]) * 100
    trans=trans[percent_zeros<thred]
    new_sample=trans.shape[0]
    trans = np.transpose(trans, (1, 0, 2, 3))
    if raw_sample - new_sample > 0:
        log.warning('Dropping %d of %d samples from the CNN input: more than %s%% of their genome has '
                    'no coverage', raw_sample - new_sample, raw_sample, thred)
    raw_pos=trans.shape[0]
    tem = trans[:, :, 4:6, :]
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
        log.debug('Dropped %d positions where every remaining sample has the same base',
                  raw_pos - unqiue_pos)

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
    new_pos=trans.shape[0]
    log.debug('Dropped %d low-quality positions; %d positions go to the CNN', raw_pos - new_pos, new_pos)

    return trans,inpos

def trans_shape(indata):
    return np.transpose(indata, (1, 0, 2, 3))

def remove_exclude_samples(samples_to_exclude_bool,quals,counts,sample_names,indel_counter,raw_cov_mat):
    in_outgroup=samples_to_exclude_bool
    quals = quals[~in_outgroup]
    counts = counts[~in_outgroup]
    sample_names = sample_names[~in_outgroup]
    indel_counter = indel_counter[~in_outgroup]
    raw_cov_mat = raw_cov_mat[~in_outgroup]
    in_outgroup=in_outgroup[~in_outgroup]
    return quals,counts,sample_names,indel_counter,raw_cov_mat,in_outgroup


def data_transform(infile,incov,samples_to_exclude,min_cov_samp):

    [quals, p, counts, in_outgroup, sample_names, indel_counter] = \
        snv.read_candidate_mutation_table_npz(infile)

    if not len(in_outgroup)==len(sample_names):
        in_outgroup=np.array([False] * len(sample_names))

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
    samples_to_exclude_bool = np.array( [x in samples_to_exclude for x in my_cmt.sample_names] )
    my_cmt.filter_samples( ~samples_to_exclude_bool )
    quals,counts,sample_names,indel_counter,raw_cov_mat,in_outgroup=remove_exclude_samples(samples_to_exclude_bool,quals,counts,sample_names,indel_counter,raw_cov_mat)

    my_calls = snv.calls_object(my_cmt)

    keep_col = snv.remove_same(my_calls)

    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)
    
    quals = quals[:,keep_col]
    counts = counts[:,keep_col,:]
    p=p[keep_col]
    indel_counter=indel_counter[:,keep_col,:]
    median_cov = np.median(raw_cov_mat, axis=1)

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
    c1 = (combined_array[..., :] == 0)
    x1 = (np.sum(c1[:, :, :2, :], axis=-2) == 2)
    mx = x1
    mxe = np.repeat(mx[:, :, np.newaxis, :], 5, axis=2)
    combined_array[mxe] = 0
    ####### Reorder the columns and normalize & split the count info
    combined_array = reorder_norm(combined_array, my_cmt)
    #### Remove low quality samples
    if not min_cov_samp==100:
        combined_array,p=remove_low_quality_samples(combined_array, min_cov_samp,p)
    #### Remove bad positions
    combined_array,p,pfgap,praw=remove_lp(combined_array,p,my_cmt,my_calls,median_cov )
    dgap={}
    for s in praw:
        if s not in pfgap:
            dgap[s]='0'
        else:
            dgap[s]='1'
    

    return combined_array,p,dgap


def CNN_predict(data_file_cmt,data_file_cov,out,samples_to_exclude,min_cov_samp):
    if not os.path.exists(out):
        os.makedirs(out)
    setup_seed(1234)
    incount = data_file_cmt
    incov = data_file_cov
    if not os.path.exists(incount) or not os.path.exists(incov):
        log.error('Cannot run the CNN: the candidate mutation table (%s) or the coverage matrix (%s) '
                  'is missing', incount, incov)
        exit()
    mut=incount
    cov=incov
    test_datasets=[]

    odata,pos,dgap=data_transform(mut,cov,samples_to_exclude,min_cov_samp)
    log.debug('CNN input tensor shape (positions, samples, channels, window): %s', odata.shape)
    if len(pos)==0:
        # No candidate position survived filtering (e.g. a clonal cohort with no SNVs).
        # There is nothing for the CNN to score, and torch cannot take an empty batch.
        log.warning('No candidate positions are left for the CNN to score. This is expected for a '
                    'clonal set of isolates with no SNVs between them.')
        return np.array([],dtype=int),np.array([],dtype=int),np.array([],dtype=float),dgap
    nlab=np.zeros(odata.shape[0])
    test_datasets.append((odata,nlab))

    #Create DataLoaders
    def create_dataloader(datasets):
        dataloader=[]
        for i, (data,label)  in enumerate(datasets):
            dataset = CustomDataset(data, label)
            dataloader_tem = DataLoader(dataset, batch_size=512, shuffle=False)
            dataloader.append(dataloader_tem)
        return dataloader

    test_loader=create_dataloader(test_datasets)

    # Training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info('Running the CNN on %s', 'GPU (cuda)' if device == 'cuda' else 'CPU')
    model= CNNModel(n_channels=4).to(device)

    model.load_state_dict(torch.load(str(files("accusnv").joinpath("models", "checkpoint_best_3conv.pt"))))

    model.eval()

    predictions = []
    with torch.no_grad():
        for loader in test_loader:
            for inputs,label in loader:
                inputs=inputs.to(device)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy().flatten())
    # Convert predictions to binary labels
    prob=np.array(predictions)
    predictions = (np.array(predictions) > 0.5).astype(int)
    y_pred = predictions
    # Return all reamining positions and CNN's predicted probabilities array
    return pos,y_pred,prob,dgap
