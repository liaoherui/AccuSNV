import os
import re
import sys
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.power import TTestPower
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

def cal_major_freq_in_call(arr1):
    #print(arr1.shape)
    #print(arr1)
    col_data_nonzero = [arr1[:, col][arr1[:, col] != 0] for col in range(arr1.shape[1])]
    '''
    for col in col_data_nonzero:
        print(col,np.bincount(col))
        print(np.argmax(np.bincount(col)))
    '''
    column_modes = [np.unique(col)[0] if len(np.unique(col)) == 1 else (1 if len(col) == 0 else np.argmax(np.bincount(col))) for col in col_data_nonzero]
    #exit()
    #print(column_modes)
    scount = np.sum(arr1 == column_modes, axis=0)
    #print(arr1==column_modes)
    #exit()
    return scount,arr1==column_modes



def compare_arrays_nonparametric(large_array_raw, small_array_raw):

    large_array=[x for x in large_array_raw if x != 0]
    #sorted_arr = np.sort(arr)
    #n = len(sorted_arr)
    #k = int(np.ceil(n * 0.25))
    #large_array = sorted_arr[:k]

    small_array=[x for x in small_array_raw if x != 0]
    #print(large_array,small_array)
    if len(small_array) == 0:
        return "Empty arrays! Please check."
        
    elif len(small_array) == 1:
        statistic, p_value = stats.wilcoxon(large_array - small_array[0])
    else:
        statistic, p_value = stats.mannwhitneyu(large_array, small_array)
    return p_value


def scan_continue_gap(arr):
    arr.sort()
    B = set()
    i, j = 0, 1
    n = len(arr)
    while i < n:
        while j < n and arr[j] - arr[i] <= 100:
            if arr[j] - arr[i] > 0:
                B.add(arr[i])
                B.add(arr[j])
            j += 1
        i += 1
        if j <= i:
            j = i + 1

    B = list(B)
    return B

def scan_continue_gap_revise(arr):
    arr.sort()
    #print(arr)
    B = {}

    i, j = 0, 1
    n = len(arr)
    #print(arr)
    while i < n:
        j=i+1
        #print('i,n:',i,',',n,'j,n:',j,',',n,'arr[j]-arr[i]:',arr[j],'-',arr[i])
        while j < n and arr[j] - arr[i] <= 100:
            #print(arr[j],arr[i])
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
            #print(i,j)
        i += 1
        if j <= i:
            j = i + 1
    res=[]
    #print(B)
    #print(B[1899148])
    '''
    arr_B=sorted(list(B.keys()))
    for key in B[arr_B[0]]:
        for key2 in B[arr_B[0]]:
            if not key==key2:
                B[key][key2]=''
    print(B)
    '''
    #print(check_arr)
    #exit()
    for p in B:
        if len(B[p])>2:
            res.append(p)
        '''
        else:
            check=0
            if len(B[p])==0:continue
            for s in B[p]:
                if s in check_arr:
                    check+=1
            if check==len(B[p]):
                res.append(p)
        # old code
        if len(B[p])>1:
            if len(B[p])>2:
                res.append(p)
            else:
                check=0
                for s in B[p]:
                    if s in check_arr:
                        check+=1
                if check==len(B[p]):
                    res.append(p)
        elif len(B[p])==1:
            check=1
            for e in B[p]:
                if e in check_arr and p in check_arr:
                    check=0
            if check==0:
                res.append(p)
        '''
    return res

def zscore_variant(large_array,small_array):
    mean_diff = np.mean(large_array) - np.mean(small_array)
    std_err_diff = np.std(large_array, ddof=1)
    z_score = mean_diff / std_err_diff
    #p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    p_value=1 - stats.norm.cdf(abs(z_score))
    #print(mean_diff,z_score,std_err_diff)
    return z_score,p_value

def compare_arrays_ttest(large_array_raw, small_array_raw):
    #print(large_array,small_array)
    large_array=[x for x in large_array_raw if x != 0]
    #print('-> raw_array:',large_array)
    #mu=np.mean(large_array)
    #sigma = np.std(large_array)
    #alpha=0.01
    #threshold=norm.ppf(alpha, loc=mu, scale=sigma) 
    #print('-> threshold is ',threshold)
    large_array=np.array(large_array)
    #large_array_cdf=large_array[large_array<=threshold]
    #print('-> new_array:',large_array_cdf)
    '''
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    k = int(np.ceil(n * 0.25))
    large_array = sorted_arr[:k]
    '''

    small_array=[x for x in small_array_raw if x != 0]
    #print(large_array,small_array)
    #exit()
    n_small = len(small_array)
    normc=1
    if n_small == 0:
        return "Empty array! Please check."
        exit()
    
    elif n_small == 1:
        target_value = small_array[0]
        t_stat, p_value = stats.ttest_1samp(large_array, popmean=target_value)
        # new p-cdf with z-score variant
        z_score,p_cdf=zscore_variant(large_array,small_array)
        '''
        if not (len(large_array_cdf)==0 or len(large_array_cdf)==1):
            t_stat, p_cdf = stats.ttest_1samp(large_array_cdf, popmean=target_value)
        else:
            normc=0
            p_cdf=p_value
        '''
        #print('-> 2 p-values: ',p_value,p_cdf)
        if np.isnan(p_cdf):
            normc=0
        return p_value,p_cdf
    
    else:
        t_stat, p_value = stats.ttest_ind(large_array, small_array, equal_var=False)
        z_score,p_cdf=zscore_variant(large_array,small_array)
        '''
        if not (len(large_array_cdf)==0 or len(large_array_cdf)==1):
            t_stat, p_cdf = stats.ttest_ind(large_array_cdf,small_array,equal_var=False)
        else:
            normc=0
            p_cdf=p_value
        '''
        #print('-> 2 p-values: ',p_value,p_cdf)
        if np.isnan(p_cdf):
            normc=0
        return p_value,p_cdf

def check_mm(arr1,counts_major):
    #print(arr1[:,:10])
    col_data_nonzero = [arr1[:, col][arr1[:, col] != 0] for col in range(arr1.shape[1])]
    #print(col_data_nonzero)
    #exit()
    '''
    column_modes=[]
    for col in col_data_nonzero:
        #print(col,np.bincount(col),np.unique(col))
        if len(np.unique(col)) >= 2
            nonzero_pairs = sorted([(x, i) for i, x in enumerate(np.bincount(col)) if x != 0], reverse=True)
            minor=nonzero_pairs[1][1] if len(nonzero_pairs) > 1 else -1
            
        else:
            if len(col) == 0:
                minor=1
            else:
                minor=0
        column_modes.append(minor)
        #print(res)
        exit()
    '''
    column_modes = [sorted([(x, i) for i, x in enumerate(np.bincount(col)) if x != 0], reverse=True)[1][1] if len(np.unique(col)) >= 2 and len([(x, i) for i, x in enumerate(np.bincount(col)) if x != 0]) > 1 else (1 if len(col) == 0 else 0) for col in col_data_nonzero]
    #print(column_modes)
    #exit()
    #column_modes = [np.unique(col)[1] if len(np.unique(col)) >= 2 else (1 if len(col) == 0 else 0) for col in col_data_nonzero]
    scount = np.sum(arr1 == column_modes, axis=0)
    only_minor= arr1 == column_modes
    #print(only_minor.shape,counts_major.shape)
    #exit()
    #print(scount)
    check_minor_more_than_1=np.repeat([scount>1], arr1.shape[0], axis=0)
    cmj_copy=copy.deepcopy(counts_major)
    cmj_copy[~only_minor]=0
    #print(counts_major)
    min_v=[min(col[col != 0]) if len(col[col != 0]) > 0 else 0 for col in cmj_copy.T]
    #print(min_v)
    #exit()
    c,b=cal_major_freq_in_call(arr1)
    copy_minv=np.repeat([min_v],arr1.shape[0],axis=0)
    #print('cminv:',copy_minv)
    #print('cmajor:',counts_major)
    x=counts_major-copy_minv
    x[only_minor]=-1
    #print('cmj-cmi:',x)

    #print(counts_major,arr1)
    #print(arr1.shape,check_minor_more_than_1.shape)
    #exit()
    ##print(column_modes)
    #print(scount)
    #print(arr1)
    #c,b=cal_major_freq_in_call(arr1)
    #minor_max = check_minor_more_than_1
    minor_max = check_minor_more_than_1 & (x<0)
    #exit()
    #print(minor_max)
    #print(arr1)
    #exit()
    
    return minor_max,arr1==column_modes

def process_arrays(arr1, arr2, sample_num):
    col_data_nonzero = [arr1[:, col][arr1[:, col] != 0] for col in range(arr1.shape[1])]
    column_modes = [np.unique(col)[0] if len(np.unique(col)) == 1 else ( 1 if len(col)==0 else np.argmax(np.bincount(col))) for col in col_data_nonzero]
    #print(arr2)
    scount = np.sum(arr1 == column_modes, axis=0)
    #print(scount)
    mask = arr1 != np.array(column_modes)
    arr2[mask] = 0
    result = np.sum((arr2 > 0) & (arr2 < 1), axis=0)
    #print(result/scount)
    #exit()

    return result/scount

def cal_freq_amb_samples(all_p,my_cmt):
    #print(len(all_p))
    #print(len(my_cmt.p))
    #exit()
    keep_col=[]
    for p in my_cmt.p:
        if p in all_p:
            keep_col.append(True)
        else:
            keep_col.append(False)
    keep_col=np.array(keep_col)
    #print(len(my_cmt.p))
    #print(len(keep_col))
    #print(keep_col)
    #exit()
    my_cmt.filter_positions(keep_col)
    #exit()
    freq_arr=process_arrays(my_cmt.major_nt,my_cmt.major_nt_freq,my_cmt.major_nt.shape[0])
    ##exit()
    freq_d={}
    c=0
    for p in my_cmt.p:
        freq_d[p]=freq_arr[c]
        c+=1
    return freq_d,freq_arr

def remove_lp(combined_array,inp,my_cmt,my_calls, median_cov ):
    raw_p=len(inp)
    rawp=inp
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
    #print(my_cmt.p)
    #exit()
    my_calls.filter_calls_by_element(
        my_cmt.fwd_cov < 1
    )

    my_calls.filter_calls_by_element(
        my_cmt.rev_cov < 1
    )
    my_calls_check=copy.deepcopy(my_calls)
    scount_before,b=cal_major_freq_in_call(my_calls_check.calls)
    #print(scount_before[:100])
    #exit()
    my_calls_check.filter_calls_by_element(
            (my_cmt.fwd_cov < 11) & (my_cmt.rev_cov < 11)
    )
    scount_after,b=cal_major_freq_in_call(my_calls_check.calls)
    #print(scount_after[:100])
    fratio=scount_after/scount_before
    #print(fratio)
    count=np.sum(fratio > 0.5)
    #print(count)
    stat=count/len(fratio)
    #print(stat)
    #exit()
    my_cmt_tem=copy.deepcopy(my_cmt)
    freq_p,freq_arr=cal_freq_amb_samples(my_calls.p,my_cmt_tem)
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
    #print(np.where(my_cmt.p==1161288))
    my_calls.filter_calls_by_element(
            (my_cmt.major_nt_fwd != my_cmt.major_nt_rev) & ((my_cmt.major_nt_fwd>0)&(my_cmt.major_nt_rev>0))
    )
    '''
    # new rule - large fwd/rev and very samll rev/fwd - >70% sample fwd>4*rev or rev>4*fwd, and fwd or rev <10
    my_calls_check=copy.deepcopy(my_calls)
    cbefore=np.sum(my_calls_check.calls != 0, axis=0) 
    my_calls_check.filter_calls_by_element(
            ((my_cmt.fwd_cov - 4*my_cmt.rev_cov >0 ) | (my_cmt.rev_cov - 4*my_cmt.fwd_cov > 0)) & ((my_cmt.fwd_cov < 10) | (my_cmt.rev_cov < 10))
            )
    cafter=np.sum(my_calls_check.calls != 0, axis=0)
    
    diff_ratio=1-cafter/cbefore
    keep_col=[]
    if my_calls_check.calls.shape[0]>10:
        cut=0.6
    else:
        cut=0.7
    for d in diff_ratio:
        if d>cut:
            keep_col.append(False)
        else:
            keep_col.append(True)
    keep_col=np.array(keep_col)
    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)
    '''

    #print(diff_ratio,my_calls_check.p)
    #exit()

    my_calls.filter_calls_by_element(
        my_cmt.major_nt_freq < 0.7
    )
    #print(my_cmt.p[:10])
    # print(my_cmt.counts_major.shape)
    # print(my_cmt.counts_minor.shape)
    # na=my_cmt.counts_major-5*my_cmt.counts_minor
    # test=my_cmt.major_nt_freq < 0.7
    # t2= my_cmt.counts_minor>20
    #a=np.array(my_cmt.counts_major, dtype=np.int64)
    #b=np.array(my_cmt.counts_minor, dtype=np.int64)
    #print(my_calls.calls[:,np.where(my_cmt.p==1052499)])
    # exit()
    ### Super big fp pos filter - 2024-12-18
    #rawp=my_c.p
    #my_calls.filter_calls_by_element(
    #    (my_cmt.counts_minor>30) & ((a-b*3)>0) & (my_cmt.counts_major>250)
    #)
    #print(my_cmt.counts_major[:,np.where(my_cmt.p==1052499)]-(my_cmt.counts_minor[:,np.where(my_cmt.p==1052499)]*10))
    #print((a-b*10)[:,np.where(my_cmt.p==1052499)])
    #exit()
    #print('filter_low_quality_pos...')
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
    #print(keep_col)
    #exit()
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
    #print(my_cmt.p)
    #exit()
    #### Filter low gap pos
    #print('before-filt-gap:',my_cmt.p)
    rawp=my_cmt.p # used to check positions removed by gap filter
    #exit()
    '''
    # new rule - large fwd/rev and very samll rev/fwd - >70% sample fwd>4*rev or rev>4*fwd, and fwd or rev <10  & freq must >0
    my_calls_check=copy.deepcopy(my_calls)
    if my_calls_check.calls.shape[0]>10:
        cut=0
    else:
        cut=0.2
    freq_p,freq_arr=cal_freq_amb_samples(my_calls_check.p,my_cmt)
    freq_check=np.repeat([freq_arr>cut],my_calls_check.calls.shape[0],axis=0) 

    cbefore=np.sum(my_calls_check.calls != 0, axis=0)
    my_calls_check.filter_calls_by_element(
            ((my_cmt.fwd_cov - 4*my_cmt.rev_cov >0 ) | (my_cmt.rev_cov - 4*my_cmt.fwd_cov > 0)) & ((my_cmt.fwd_cov < 10) | (my_cmt.rev_cov < 10)) & freq_check
            )
    cafter=np.sum(my_calls_check.calls != 0, axis=0)

    diff_ratio=1-cafter/cbefore
    keep_col=[]
    if my_calls_check.calls.shape[0]>10:
        cut=0.6
    else:
        cut=0.7
    for d in diff_ratio:
        if d>cut:
            keep_col.append(False)
        else:
            keep_col.append(True)
    keep_col=np.array(keep_col)
    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)
    ########################### new rule done
    '''
    med_cov_ratio_fwd=my_cmt.fwd_cov/median_cov[:, np.newaxis]
    med_cov_ratio_rev = my_cmt.rev_cov / median_cov[:, np.newaxis]
    med_cov_ratio_fwd = np.nan_to_num(med_cov_ratio_fwd, nan=0)
    med_cov_ratio_rev = np.nan_to_num(med_cov_ratio_rev, nan=0)
    # Add super big fp filt in gap filt
    a=np.array(my_cmt.counts_major, dtype=np.int64)
    b=np.array(my_cmt.counts_minor, dtype=np.int64)
    #print(a.shape)
    #print(my_calls.calls[:,np.where(my_cmt.p==1052499)])
    # exit()
    #rawp=my_c.p
    #print(my_calls.p)
    minor_max,mbool=check_mm(my_calls.calls,a) # Whether the major  counts all < the min minor count & # of minor > 1
    my_calls.filter_calls_by_element(
        (my_cmt.counts_minor>30) & ((a-b*3)>0) & (my_cmt.counts_major>250) & minor_max
    )
    #print(my_cmt.fwd_cov)
    #print((my_cmt.counts_minor>30) & ((a-b*3)>0) & (my_cmt.counts_major>250))
    #print(((my_cmt.fwd_cov>200) | (my_cmt.rev_cov>200)))

    #my_calls.filter_calls_by_element(
    #    (my_cmt.counts_minor>30) & ((a-b*3)>0) & (my_cmt.counts_major>250)
    #)
    #print(my_cmt.p)
    #print(my_cmt.fwd_cov.shape)
    #print(my_cmt.fwd_cov)
    #print(my_cmt.fwd_cov<200)
    #print(my_calls.p)
    #print(combined_array[40][21])
    #print(med_cov_ratio_fwd[40][21])
    #print(med_cov_ratio_rev[40][21])
    #print(combined_array.shape,my_cmt.fwd_cov.shape)
    #exit()
    # Old methods: Use loose filtering
    #mask1 = find_sm_top_x(my_cmt.fwd_cov, 3, 20)
    #mask2 = find_sm_top_x(my_cmt.rev_cov, 3, 20)
    # old gap rule  - 2025-02-03
    '''
    if combined_array.shape[1]<25:
        mask1=find_sm_top_x(my_cmt.fwd_cov, 3, 20)
        mask2=find_sm_top_x(my_cmt.rev_cov, 3, 20)
    else:
        mask1 = find_sm_top_x(my_cmt.fwd_cov, 5, 20)
        mask2 = find_sm_top_x(my_cmt.rev_cov, 5, 20)
    '''

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

    # old gap rule - 2025-02-03
    '''
    mask1 = mask1 & (med_cov_ratio_fwd< 0.1)
    mask2 = mask2 & (med_cov_ratio_rev < 0.1)
    mask=mask1 & mask2
    my_calls.filter_calls_by_element(
        mask
    )
    '''


    # Add gap rule: if all minor sample <=10 and one part must <=5, >=85% major sample not satisfy this rule and >=20, then should be gap fp
    my_calls_tem=copy.deepcopy(my_calls)
    #print(1899148 in my_calls_tem.p)
    rawp_tem=my_calls_tem.p
    scount_before,b=cal_major_freq_in_call(my_calls_tem.calls)
    #count_combine=my_cmt.counts_major
    count_combine_ratio=my_cmt.counts_major/median_cov[:, np.newaxis]
    #print(my_cmt.counts_major,my_cmt.counts_major[:,temp])
    #print(temp)
    #exit()
    #print(temp,my_cmt.counts_major,my_cmt.counts_major.shape,b.shape,mbool.shape)
    #exit()
    #c2=copy.deepcopy(count_combine)
    c3=copy.deepcopy(count_combine_ratio)
    #count_combine[~b]=0
    count_combine_ratio[~b]=0
    #c2[~mbool]=0
    c3[~mbool]=0
    p_arr=[]
    p_arr_ratio=[]
    p_arr_ratio_cdf=[]
    #normc=[]
    tem=[]
    tem2=[]
    #p_non=[]
    for i in range(count_combine_ratio.shape[1]):
        #a_arr=count_combine[:,i]
        #b_arr=c2[:,i]
        c_arr=count_combine_ratio[:,i]
        d_arr=c3[:,i]
        #a1=[x for x in a_arr if x != 0]
        #b1=[x for x in b_arr if x != 0]
        c1=[x for x in c_arr if x != 0]
        d1=[x for x in d_arr if x != 0]
        tem.append([c1,d1])
        #tem2.append([a1,b1])
        #p2=compare_arrays_ttest(a_arr,b_arr) # Count
        p,p_cdf=compare_arrays_ttest(c_arr,d_arr) # Ratio
        #p,effect_size, power=compare_groups(c_arr, d_arr)
        #p2=compare_arrays_nonparametric(c_arr,d_arr) # Ratio
        #p_non.append(p2)
        #p2=compare_arrays_nonparametric(a_arr,b_arr) # Count
        #p_arr.append(p2)
        p_arr_ratio.append(p)
        #tem2.append([effect_size,power])
        p_arr_ratio_cdf.append(p_cdf)
        #normc.append(norm_check)
    #print(p_arr,p_arr_ratio)
    gap_candidate=[]
    tem_p=[]
    #tpos=[3950318,2403312,78143,960515,1011874,624313,4129099]
    #label=[0,0,0,1,1,1,1]
    #dtl=dict(zip(tpos,label))
    #o=open('position_vector_to_Tami/p11.txt','w+')
    #for p in range(len(p_arr_ratio))
    ct=0
    for i in range(len(p_arr_ratio)):
        '''
        if my_cmt.p[i] in tpos:
            o.write('positions\tnorm_depth_major_isolates\tnorm_depth_minor_isolates\tlabel\n')
            o.write(str(my_cmt.p[i])+'\t'+','.join(map(str, tem[i][0]))+'\t'+','.join(map(str, tem[i][1]))+'\t'+str(dtl[my_cmt.p[i]])+'\n')
        '''   

        if p_arr_ratio_cdf[i] <0.01:
            tem[i][0]=np.array(tem[i][0])
            if max(tem[i][1])<min(tem[i][0]) and max(tem[i][1])<0.05:
                #raw: if max(tem[i][0])>0.1 and len(tem[i][0][tem[i][0]<0.2])/len(tem[i][0])<0.5:
                if max(tem[i][0])>0.2:
                    gap_candidate.append(my_cmt.p[i])
                '''
                # old rule
                if normc[i]==1:
                    #if len(tem[i][0][tem[i][0]<0.1])/len(tem[i][0])<0.5:
                    if max(tem[i][0])>0.1:
                        gap_candidate.append(my_cmt.p[i])
                else:
                    #print('')
                    #if len(tem[i][0][tem[i][0]<0.1])/len(tem[i][0])<0.3:
                    #print(tem[i][0])
                    if len(tem[i][0][tem[i][0]>0.1])/len(tem[i][0])>=0.7:
                        gap_candidate.append(my_cmt.p[i])
                '''
            '''
            elif min(tem[i][1])>max(tem[i][0]) and min(tem[i][1])>0.05 and max(tem[i][0])<0.05:
                gap_candidate.append(my_cmt.p[i])
            '''
        if p_arr_ratio_cdf[i] <0.01 or p_arr_ratio[i] <0.05:
            if np.max(tem[i][1])<0.1 and len(tem[i][1])<=3 and max(tem[i][0])>0.2:
                tem_p.append(my_cmt.p[i])
            tem[i][0]=np.array(tem[i][0])
            #print('pos:',my_cmt.p[i],'\n','p-value:',p_arr_ratio[i],'\np-value-z-score:',p_arr_ratio_cdf[i],'\nnorm_major_mean:',np.mean(tem[i][0]),'\nnorm_major_median:',np.median(tem[i][0]),'\n# of major <0.2',np.sum(tem[i][0]<0.2),'\nthese elements are:',tem[i][0][tem[i][0]<0.2],'\nnorm_minor_arr:',tem[i][1])
        #print(my_cmt.p[i],tem[i][1])
        '''
        if p_arr_ratio[i]<0.05 and np.max(tem[i][1])<0.1 :
            p=1
            #print(my_cmt.p[i],p_arr_ratio[i],p_arr[i],tem[i][0],tem[i][1])
            gap_candidate.append(my_cmt.p[i])
            #gap_pos.append()
            tem[i][0]=np.array(tem[i][0])
            # old one with # '\nnorm_major_arr:',tem[i][0]
            print('pos:',my_cmt.p[i],'\n','p-value:',p_arr_ratio[i],'\nnorm_major_mean:',np.mean(tem[i][0]),'\nnorm_major_median:',np.median(tem[i][0]),'\n# of major <0.1',np.sum(tem[i][0]<0.1),'\nthese elements are:',tem[i][0][tem[i][0]<0.1],'\nnorm_minor_arr:',tem[i][1])
            if np.sum(tem[i][0]<0.1)>1:
                if np.min(tem[i][0])<np.max(tem[i][1]) or np.min(tem[i][0])-np.max(tem[i][1])<0.01 or np.sum(tem[i][0]<0.1)>0.2*len(tem[i][0]):
                    print('\n')
                else:
                    p=compare_arrays_ttest(tem[i][0][tem[i][0]<0.1],tem[i][1]) 
                    print('\np-value-major-s0.1-minor:',p)
                    print('\nnorm_major_s0.1_median:',np.median(tem[i][0][tem[i][0]<0.1]))
                    if p<0.05:
                        tem_p.append(my_cmt.p[i])
            else:
                if np.sum(tem[i][0]<0.1)==1:
                    #print(tem[i][0][tem[i][0]<0.1][0])
                    if  np.min(tem[i][1])>0.05:
                        print('\n')
                    elif not tem[i][0][tem[i][0]<0.1][0]<0.09:
                        tem_p.append(my_cmt.p[i])
                    elif np.max(tem[i][1])<0.02 and np.min(tem[i][0])-np.max(tem[i][1])>0.05:
                        tem_p.append(my_cmt.p[i])
                else:
                    if np.min(tem[i][1])<0.05:
                        tem_p.append(my_cmt.p[i])
            #if np.sum(tem[i][0]<0.1)==0 or (np.sum(tem[i][0]<0.1)==1 and tem[i][0][0]>0.05) or (p<0.05 and np.median(tem[i][1])<np.median(tem[i][0][tem[i][0]<0.1])):
                #tem_p.append(my_cmt.p[i])
        
            #print(p_arr[i],tem2[i][0],tem2[i][1])
        '''
    #print(gap_candidate)
    #exit()
    gap_pos_add=scan_continue_gap_revise(tem_p)
    #gap_pos_add=[]
    gap_pos=gap_candidate
    
    for p in gap_pos_add:
        if p not in gap_pos:
            gap_pos.append(p)
    
    print('gap_pos:\n',gap_pos)
    #print(p_arr[36],p_arr_ratio[36])
    #exit()
    #print(my_cmt.p[0],my_cmt.p[36])

    #exit()
    #print(scount_before)
    #print('before:',my_calls_tem.calls)
    #exit()
    # old gap rule - 2025-02-03
    '''
    my_calls_tem.filter_calls_by_element(
        ((my_cmt.fwd_cov <= 10) & (my_cmt.rev_cov <= 10)) & ((my_cmt.fwd_cov <= 5) | (my_cmt.rev_cov <= 5))
    )

    #print(my_cmt.major_nt,my_cmt.major_nt.shape)
    #print(my_cmt.fwd_cov,my_cmt.fwd_cov.shape)
    #exit()

    my_calls_tem.filter_calls_by_element(
            (my_cmt.fwd_cov <= 20) & (my_cmt.rev_cov <= 20) & b
    )
    
    scount_after,b=cal_major_freq_in_call(my_calls_tem.calls)
    keep_col_tem = remove_same(my_calls_tem)
    #print(keep_col_tem)
    if len(keep_col_tem)==0:
        print('No SNPs detected! Exit.')
        exit()
    #exit()
    my_calls_tem.filter_positions(keep_col_tem)
    gap_ratio=scount_after/scount_before
    dtem=dict(zip(rawp_tem,gap_ratio))
    #print(dtem)
    #print(scount_after)
    #print('after:',my_calls_tem.calls)
    #print(my_calls_tem.p,gap_ratio)
    gap_pos=[]
    for p in rawp_tem:
        if p not in my_calls_tem.p:
            if dtem[p]>=0.85:
                gap_pos.append(p)
    #exit()
    #print(gap_pos)
    #exit()
    

    '''
    #print(my_calls.p, len(my_calls.p))
    keep_col_raw = remove_same(my_calls)
    keep_col=[]
    c=0
    for k in keep_col_raw:
        if my_calls.p[c] in gap_pos:
            keep_col.append(False)
        else:
            keep_col.append(k)
        c+=1
    keep_col=np.array(keep_col)
    my_cmt.filter_positions(keep_col)
    my_calls.filter_positions(keep_col)
    #print('after-filt-gap:',my_cmt.p)
    pfgap=[p for p in rawp if p not in my_cmt.p] # position filterd due to gap
    #print(len(pfgap))
    #exit()
    #print(my_calls.p, len(my_calls.p))
    #print(pfgap)
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
    print([p for p in rawp if p not in inp])
    #exit()
    return combined_array,inp,pfgap,rawp


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

def remove_exclude_samples(samples_to_exclude_bool,quals,counts,sample_names,indel_counter,raw_cov_mat,in_outgroup):
    in_outgroup=samples_to_exclude_bool
    quals = quals[~in_outgroup]
    counts = counts[~in_outgroup]
    sample_names = sample_names[~in_outgroup]
    indel_counter = indel_counter[~in_outgroup]
    raw_cov_mat = raw_cov_mat[~in_outgroup]
    in_outgroup=in_outgroup[~in_outgroup]
    return quals,counts,sample_names,indel_counter,raw_cov_mat,in_outgroup


def data_transform(infile,incov,fig_odir,samples_to_exclude,min_cov_samp):

    # infile='../../../Scan_FP_TP_for_CNN/Cae_files/npz_files/Lineage_10c/candidate_mutation_table_cae_Lineage_10c.npz'
    [quals, p, counts, in_outgroup, sample_names, indel_counter] = \
        snv.read_candidate_mutation_table_npz(infile)

    #print(in_outgroup,in_outgroup.shape)
    if not len(in_outgroup)==len(sample_names):
        in_outgroup=np.array([False] * len(sample_names))
    # Only for P15
    #in_outgroup[2]=True
    #in_outgroup[13]=True
    #in_outgroup[13]=True

    ########  remove outgroup samples
    quals = quals[~in_outgroup]
    counts = counts[~in_outgroup]
    sample_names = sample_names[~in_outgroup]
    indel_counter = indel_counter[~in_outgroup]
    raw_cov_mat = snv.read_cov_mat_npz(incov)
    raw_cov_mat = raw_cov_mat[~in_outgroup]
    in_outgroup=in_outgroup[~in_outgroup]
    #in_outgroup=np.array([False] * len(sample_names))
    #in_outgroup[2]=True
    #in_outgroup[13]=True
    
    my_cmt = snv.cmt_data_object(sample_names,
                                 in_outgroup,
                                 p,
                                 counts,
                                 quals,
                                 indel_counter
                                 )
    samples_to_exclude_bool = np.array( [x in samples_to_exclude for x in my_cmt.sample_names] )
    my_cmt.filter_samples( ~samples_to_exclude_bool )
    quals,counts,sample_names,indel_counter,raw_cov_mat,in_outgroup=remove_exclude_samples(samples_to_exclude_bool,quals,counts,sample_names,indel_counter,raw_cov_mat,in_outgroup)

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
    #print(p,len(p))
    #exit()
    if not min_cov_samp==100:
        combined_array,p=remove_low_quality_samples(combined_array, min_cov_samp,p)
    #print(np.where(p==864972))
    #exit()
    #### Remove bad positions
    #pfgap=[]
    combined_array,p,pfgap,praw=remove_lp(combined_array,p,my_cmt,my_calls,median_cov )
    dgap={}
    #praw=p
    for s in praw:
        if s not in pfgap:
            dgap[s]='0'
        else:
            dgap[s]='1'
    

    return combined_array,p,dgap


def load_test_name(infile):
    dt={}
    f=open(infile,'r')
    while True:
        line=f.readline().strip()
        if not line:break
        dt[line]=''
    return dt

def CNN_predict(data_file_cmt,data_file_cov,out,samples_to_exclude,min_cov_samp):
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
    odata,pos,dgap=data_transform(mut,cov,fig_odir,samples_to_exclude,min_cov_samp)
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
    #o=open(out+'/cnn_res.txt','w+')
    #o.write('Pos_info\tPredicted_label\tProbability\tGap_filt\n')

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
    #c=0
    #print('Predicted results:',np.count_nonzero(y_pred),' true SNPs, ',len(y_pred)-np.count_nonzero(y_pred),' false SNPs.')
    '''
    for s in y_pred:
        o.write(info[c]+'\t'+str(s)+'\t'+str(prob[c])+'\t'+dgap[int(info[c])]+'\n')
        c+=1
    '''
    # Return all reamining positions and CNN's predicted probabilities array
    return pos,y_pred,prob,dgap
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
