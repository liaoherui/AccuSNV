import os
import re
import sys
from sklearn.model_selection import StratifiedKFold
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
script_dir = os.path.dirname(os.path.abspath(__file__))
dir_py_scripts = script_dir+"/modules"
sys.path.insert(0, dir_py_scripts)
import snv_module_recoded_herui_analysis_39f as snv # SNV calling module

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='Saved_models_2025/checkpoint_best_39features_3conv_mask_balance_1x1_3x3-2x2_multichannel_reorder_split_m2_basechannel_mask.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 1x1 - 3x3 - 2x2
class CNNModel_1(nn.Module):
    def __init__(self, n_channels, num_classes=1):
        super(CNNModel_1, self).__init__()

        # 1x1 Convolution to capture channel information
        # original output channel dim 32
        #self.conv1x1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=1)
        self.conv1x1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=1)
        #print()
        #self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=(1, 0))
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=(1, 0))
        #self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=(1, 0))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Input shape: (batch_size, x_dim, y_dim, n_channels)
        print(x[:3])
        print('----')
        x = x.permute(0, 2, 1, 3)  # Rearrange to (batch_size, n_channels, x_dim, y_dim)
        #print(x.shape)
        x = torch.relu(self.conv1x1(x))
        print(x[:3])
        exit()
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        #x = torch.relu(self.conv3(x))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
# 3x3 - 2x2
class CNNModel_2(nn.Module):
    def __init__(self, n_channels, num_classes=1):
        super(CNNModel_2, self).__init__()

        # 1x1 Convolution to capture channel information
        #self.conv1x1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=(1, 0))
        #self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=(1, 0))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Input shape: (batch_size, x_dim, y_dim, n_channels)
        x = x.permute(0, 2, 1, 3)  # Rearrange to (batch_size, n_channels, x_dim, y_dim)

        #x = torch.relu(self.conv1x1(x))
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        #x = torch.relu(self.conv3(x))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Make major, minor, other_1, other_2 as the channel
class CNNModel_2_new(nn.Module):
    def __init__(self, n_channels, num_classes=1):
        super(CNNModel_2_new, self).__init__()

        # 1x1 Convolution to capture channel information
        #self.conv1x1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=(3,4), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=(1, 0))
        #self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=(1, 0))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Input shape: (batch_size, sample_num, features, n_channels->ATGC)
        #print(x.shape)
        x = x.permute(0, 3, 1, 2)  # Rearrange to (batch_size, n_channels, x_dim, y_dim)
        #print(x.shape)
        #x = torch.relu(self.conv1x1(x))
        x = torch.relu(self.conv1(x))
        #print(x.shape)
        #exit()
        x = torch.relu(self.conv2(x))
        #print(x.shape)
        x = torch.relu(self.conv3(x))
        #print(x.shape)
        #exit()

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        #exit()
        return x


#### three dimensional kernel test
class CNNModel_2_3d(nn.Module):
    def __init__(self, n_channels, num_classes=1):
        super(CNNModel_2_3d, self).__init__()

        # 3D Convolutions
        self.conv1 = nn.Conv3d(in_channels=n_channels, out_channels=64, kernel_size=(3, 3, 4), padding=(1, 1, 2))
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(2, 2, 3), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 1, 2), padding=(0, 0, 1))

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        # Input shape: (batch_size, n_channels, height, width)
        print(x.shape)
        exit()
        x = torch.relu(self.conv1(x))
        print(x.shape)
        exit()
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# 3x3 - 2x2 - 1x1
class CNNModel_3(nn.Module):
    def __init__(self, n_channels, num_classes=1):
        super(CNNModel_3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=(1, 0))

        # 1x1 Convolution to capture channel information
        self.conv1x1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 64)  # Adjusted input size to match output of conv1x1
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Input shape: (batch_size, x_dim, y_dim, n_channels)
        x = x.permute(0, 2, 1, 3)  # Rearrange to (batch_size, n_channels, x_dim, y_dim)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv1x1(x))  # Apply 1x1 convolution after spatial convolutions

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

def cal_med_cov_for_given_array(x):
    nx=x.transpose(1, 0, 2)
    data=x.reshape(nx.shape[0],nx.shape[1]*nx.shape[2])
    non_zero_data = [row[row != 0] for row in data]


    median_values = np.array([np.median(row) if len(row) > 0 else 0 for row in non_zero_data])

    return median_values


def norm_array_min_max(array):
    min_val = np.min(array, axis=2)
    max_val = np.max(array, axis=2)
    #print(min_val[0])
    #print(max_val[0])
    #exit()
    expanded_min= np.repeat(min_val[:, :, np.newaxis, :], 10, axis=2)
    expanded_max = np.repeat(max_val[:, :, np.newaxis, :], 10, axis=2)
    #print(expanded_min[0])
    #print(expanded_max[0])
    #exit()

    # Apply min-max normalization
    normalized_array = (array - expanded_min) / (expanded_max - expanded_min)
    normalized_array=np.nan_to_num(normalized_array, nan=0)
    return normalized_array


def get_the_new_order(matrix):
    # Define the elements to check
    elements = np.array([1, 2, 3, 4])

    # Count the occurrences of each element along the rows
    counts = np.array([(matrix == e).sum(axis=1) for e in elements]).T
    #print(counts[:3])
    #exit()

    # Sort elements by their counts in descending order
    sorted_indices = np.argsort(-counts, axis=1)


    # Get the sorted elements based on counts
    sorted_elements = np.take_along_axis(np.tile(elements, (matrix.shape[0], 1)), sorted_indices, axis=1)
    # print(sorted_elements[:3])
    # exit()
    # # Identify the unique elements and their positions
    # unique_sorted_elements = np.unique(sorted_elements, axis=1)
    #
    # # Ensure all elements [1, 2, 3, 4] are in each row by appending missing ones
    # missing_elements = np.setdiff1d(elements, unique_sorted_elements)
    # missing_elements_array = np.tile(missing_elements, (matrix.shape[0], 1))
    # final_result = np.hstack((unique_sorted_elements, missing_elements_array))
    return sorted_elements

def reorder_norm(combined_array,my_cmt):
    major_nt = my_cmt.major_nt.T
    #print(major_nt[:20])
    #print(combined_array[0])
    #print('-------------')
    order_base=get_the_new_order(major_nt) # each row refers to one position, the 4 elements refer to the base index of "major", "minor", "other_1", "other_2"
    #print(order_base[0])
    order_base-=1
    reordered_array = np.take_along_axis(combined_array, order_base[:, np.newaxis, np.newaxis, :], axis=-1)
    ############ Order finished #################
    #print('-------------')
    #print(reordered_array[:2])
    ##### first try
    first_two_rows = reordered_array[:, :, :2, :]
    #print(first_two_rows.shape)
    #exit()
    #print(first_two_rows[:2])
    #last_row = reordered_array[:, :, -1:, :]
    sum_first_two = np.sum(first_two_rows, axis=(2, 3), keepdims=True)
    sum_first_two_fur=np.sum(sum_first_two,axis=1)
    exp_sum_first_two_fur=np.repeat(sum_first_two_fur, repeats=sum_first_two.shape[1], axis=1)
    exp_sum_first_two_fur = np.expand_dims(exp_sum_first_two_fur, axis=3)
    #print(sum_first_two[:3])
    #print(exp_sum_first_two_fur[:3])
    #print(sum_first_two.shape,exp_sum_first_two_fur.shape)
    #exit()
    # # Step 1: Sum across the rows for each block
    # sum_block_1 = np.sum(sum_first_two[0], keepdims=True)
    # sum_block_2 = np.sum(sum_first_two[1], keepdims=True)
    #
    # # Step 2: Broadcast the sum across the rows within each block
    # result_block_1 = np.tile(sum_block_1, (sum_first_two.shape[1], 1, 1))
    # result_block_2 = np.tile(sum_block_2, (sum_first_two.shape[1], 1, 1))

    # Step 3: Stack the two blocks together
    # result = np.stack([result_block_1, result_block_2])
    # print(result.shape)
    #exit()
    expanded_result = np.repeat(exp_sum_first_two_fur, repeats=4, axis=-1)
    #print(expanded_result[:2])
    #print(expanded_result.shape)
    #exit()

    ##### Second try
    # Step 1: Sum across all rows and columns
    #sum_all = np.sum(reordered_array[:, :, :, :], axis=(2, 3), keepdims=True)
    #sum_first_two = np.tile(sum_all, (1, 1, 2, 1))
    last_row = np.max(reordered_array[:, :, -1:, :], axis=3, keepdims=True)
    expanded_last_row = np.repeat(last_row, repeats=4, axis=-1)
    #print(np.where(expanded_last_row==0))
    #print(expanded_last_row[6])
    #print(reordered_array[6])
    #exit()
    #print(expanded_result.shape,expanded_last_row.shape)
    #exit()
    normalized_first_two = reordered_array[:, :, :2, :] / expanded_result

    # Step 5: Divide by the elements in the last row
    new_first_two = reordered_array[:, :, :2, :] / expanded_last_row
    new_first_two[new_first_two>10]=10
    normalized_first_two = np.nan_to_num(normalized_first_two, nan=0)
    new_first_two = np.nan_to_num(new_first_two, nan=0)
    #print(last_row[:2])
    #print(reordered_array[:2] )
    #print(normalized_first_two[:2])
    #print(new_first_two[:2])
    #exit()
    #new_array = reordered_array[:, :, 6:8, :]
    #new_array= reordered_array[:,:,:-1,:]
    #print(reordered_array[0, :, :, :])
    new_array = reordered_array[:, :, :-1, :]
    final_array = np.concatenate([normalized_first_two, new_first_two,new_array], axis=2)
    #print(reordered_array[:2])
    #print(final_array[0])
    #exit()
    #print(final_array.shape)
    #exit()
    # major_nt_fwd = my_cmt.major_nt_fwd
    # major_nt_rev = my_cmt.major_nt_rev
    # minor_nt_fwd = my_cmt.minor_nt_fwd
    # minor_nt_rev = my_cmt.minor_nt_rev
    # #print(combined_array.shape)
    # print(my_cmt.counts[:,4,:])
    # print(major_nt_fwd[:,4])
    # print(major_nt_rev[:, 4])
    # print(minor_nt_fwd[:,4])
    # print(minor_nt_rev[:, 4])
    #print(final_array.shape)
    #exit()
    return final_array

    
    

def data_transform(indata,inpos,infile,diff_pos):

    #infile='../../../Scan_FP_TP_for_CNN/Cae_files/npz_files/Lineage_10c/candidate_mutation_table_cae_Lineage_10c.npz'
    [quals, p, counts, in_outgroup, sample_names, indel_counter] = \
        snv.read_candidate_mutation_table_npz(infile)
    #print(quals.shape)
    #print(counts.shape)
    #exit()
    #print()
    #exit()
    my_cmt = snv.cmt_data_object(sample_names,
                                 in_outgroup,
                                 p,
                                 counts,
                                 quals,
                                 indel_counter
                                 )


    median_cov = np.median(my_cmt.coverage, axis=1).T
    #print(median_cov.shape)
    #print(indata[])
    #exit()
    #print(median_cov.shape,median_cov)
    #exit()
    #print(np.where(my_cmt.p==43654))
    #print(mdcov,mdcov.shape)
    #exit()
    #print(indata[0,:,:])
    #print(indata.shape)
    #exit()
    indata_32=indata[:,:,:8]
    '''
    if len(sample_names)<5:
        print(indata_32[0].T)
        print(p[0])
    '''
    indel=indata[:,:,-4]
    qual=indata[:,:,-5]
    #median_cov=cal_med_cov_for_given_array(indata[:,:,:8])
    #print(median_cov,median_cov.shape)
    #exit()
    #print(indel,indel.shape)
    expanded_array = np.repeat(indel[:, :, np.newaxis], 4, axis=2)
    expanded_array_2 = np.repeat(qual[:, :, np.newaxis], 4, axis=2)
    #print(expanded_array,expanded_array.shape)
    med_ext=np.repeat(median_cov[:, np.newaxis], 4, axis=1)
    med_arr= np.tile(med_ext, (indata.shape[0], 1, 1))

    new_data=indata_32.reshape(indata_32.shape[0],indata_32.shape[1],2,4)
    #print(expanded_array.shape, med_arr.shape,new_data.shape)
    indel_arr_final=np.expand_dims(expanded_array, axis=2)
    qual_arr_final=np.expand_dims(expanded_array_2, axis=2)
    med_arr_final = np.expand_dims(med_arr, axis=2)

    #print(expanded_array[0],indel_arr_final[0])
    #print(med_arr[0], med_arr_final[0])
    combined_array = np.concatenate((new_data, qual_arr_final, indel_arr_final, med_arr_final), axis=2)
    #print(combined_array[0],combined_array.shape)
    #exit()
    # remove the median coverage element if only this element is non-zero
    #print(np.all(combined_array[..., :] == 0, axis=-3)[0])
    check_idx=0
    c1=(combined_array[..., :] == 0)
    #print(c1)
    #print(c1.shape)
    
    #
    x1=(np.sum(c1[:,:,:2,:],axis=-2)==2)
    #print(x1[0])
    #exit()
    #x1=(np.sum(c1,axis=-2)==4) | (np.sum(c1,axis=-2)==3) # 9 and 8 refers to channel number
    #x1=(np.sum(c1,axis=-2)==3)
    #x2 = (combined_array[:, :, -1, :] != 0)
    #print(x1[0])
    #print(x2[0])
    #exit()
    #mx=x1 & x2
    mx=x1
    mxe = np.repeat(mx[:, :, np.newaxis, :], 5, axis=2)
    #print(np.sum(c1,axis=-2)[0])
    # print(combined_array[check_idx])
    # print(x1[check_idx])
    # print(x2[check_idx])
    # print(mx[check_idx])

    #print(mxe[0])
    #print(mx.shape)
    #print(mxe.shape)
    #exit()

    combined_array[mxe] = 0
    #print(combined_array[0], combined_array.shape)
    #exit()
    #print(combined_array[check_idx])
    #exit()

    #print(combined_array.shape)
    #exit()
    #print(combined_array[0])
    #######  Min-Max-Normalization
    #combined_array=norm_array_min_max(combined_array)
    ####### Reorder the columns and normalize & split the count info
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
    keep_col=np.array(keep_col)
    my_cmt.filter_positions(keep_col)
    for p in inpos:
        if diff_pos:
            if p-1 not in my_cmt.p:
                print('Pos not consistent! Exit!')
                exit()
        else:
            if p not in my_cmt.p:
                print('Pos not consistent! Exit!')
                exit()
    #print(np.where(my_cmt.p==43654))
    #exit()
    #print(my_cmt.counts.shape)
    #exit()
    ## munnal test data for clustering
    # tem_out=[]
    # def convert(inarray,x,y,z,w):
    #     tem_array=inarray.copy()
    #
    #     return
    # print(combined_array[0])
    # tem_out.append(convert(combined_array[0],))
    # #print(inpos[:10])
    # exit()

    combined_array=reorder_norm(combined_array,my_cmt)

    #print(combined_array[:2])
    #exit()
    '''
    if len(sample_names) < 5:
        print(combined_array[0].T)
        exit()
    '''

    return combined_array


def load_test_name(infile):
    dt={}
    f=open(infile,'r')
    while True:
        line=f.readline().strip()
        if not line:break
        ele=line.split()
        ele[-1]=ele[-1].strip()
        dt[ele[-1]]=''
    return dt

setup_seed(1234)
dtest=load_test_name('test_data_run_slides.txt')
#indir='CNN_select_10features_mask_balance_no_Sep'
indir='../CNN_select_10features_mask_balance_align_slides'
#indir='../CNN_select_40features_based_on_laptop_fix_label' # these datasets have correct labels now!!!
#in_npz='../../../More_features_10to39/npz_files_40features_from_server'
in_npz='all_npz_cae_bfg_jacob'

# indir='CNN_select_features'
train_datasets=[]
test_datasets=[]
j=[]
a=[]
jc=[]
js=[]

dsize=[]
#index=[0,1,2,3,4,5,6,7,13,14,15,16,17]
d={'a':0,'b':0,'c':0,'d':0}
d2={'a':[],'b':[],'c':[],'d':[]}

ot=open('2025_val_data_for_test_other_tools.txt','w+')
for filename in os.listdir(indir):
    if re.search('DS',filename):continue
    #if filename not in dtest:continue # 2025-01-16-test
    #if re.search('Lineage', filename):continue

    diff_pos=False
    data = np.load(indir + '/' + filename + '/cnn_select.npz')
    pre=re.sub('_Bfg','',filename)
    pre=re.sub('_Cae','',pre)
    pre = re.sub('_Sep', '', pre)
    ind=in_npz+'/'+pre
    infile=''
    for s in os.listdir(ind):
        if not re.search('mutation',s):continue
        infile=ind+'/'+s
    if infile=='':
        print(filename,' doesn\'t have corresponding npz file. Please check!')
    print(pre)
    #if not re.search('2c',filename):continue
    #continue
    #print(filename)
    #print(data['label'])
    #print(data['pos'])
    #exit()
    #continue
    odata = data_transform(data['x'], data['pos'], infile, diff_pos)
    print('Check Sample num:',data['x'].shape[1])
    if re.search('Lineage',filename):
        d['b']+=data['x'].shape[1]
        d2['b'].append(data['x'].shape[1])
        a.append((odata, data['label']))
    if re.search('L',filename) and not re.search('Lineage',filename):
        d['a'] += data['x'].shape[1]
        d2['a'].append(data['x'].shape[1])
        j.append((odata, data['label']))
    if re.search('cacnes_clade',filename):
        d['c'] += data['x'].shape[1]
        d2['c'].append(data['x'].shape[1])
        jc.append((odata, data['label']))
    if re.search('sepi_clade',filename):
        d['d'] += data['x'].shape[1]
        d2['d'].append(data['x'].shape[1])
        js.append((odata, data['label']))

    #continue
    if filename in dtest:
        stat=np.count_nonzero(data['label'] == 1)
        print('Test data :'+filename,', True SNPs:',stat)
        #odata=data_transform(data['x'],data['pos'],infile,diff_pos)
        #test_datasets.append((data['x'][:,:,8:].astype(np.float64), data['label']))
        test_datasets.append((odata,data['label']))
        #print(odata.shape,data['pos'].shape)
        #print(data['pos'])
        for p in data['pos']:
            ot.write(filename+'\t'+str(p)+'\n')
        #test_datasets.append((data['x'][:, :, index].astype(np.float64), data['label']))
        #exit()
    else:
        #odata = data_transform(data['x'],data['pos'],infile,diff_pos)
        #train_datasets.append((data['x'][:, :, 8:].astype(np.float64), data['label']))
        train_datasets.append((odata, data['label']))
        #train_datasets.append((data['x'][:,:,index].astype(np.float64), data['label']))
    #print(data['pos'][0])
    #print(odata[0])
    #if not odata[:,:,:,3].==0:
    #print(odata,pre)
    #exit()
    #exit()
    #print(odata[0])
    #exit()
    #print(odata[0])
    dsize.append(len(data['label']))
    #if len(train_datasets) > 0 and len(test_datasets) > 0: break
    #exit()
#dsize=np.array(dsize)
#exit()
#train_datasets=np.delete(datasets, selected_indices, axis=0)
#train_datasets=datasets[keep]
#test_datasets = datasets[selected_indices]
#print(d)
'''
pyplot.figure(figsize=(6, 3), dpi=400)
pyplot.hist(d2['a'], bins=100, alpha=0.5, label='Jay_2019')
pyplot.hist(d2['b'], bins=100, alpha=0.5, label='Arolyn_2022')
pyplot.hist(d2['c'], bins=100, alpha=0.5, label='Jacob_2024_Cae')
pyplot.hist(d2['d'], bins=100, alpha=0.5, label='Jacob_2024_Sep')
pyplot.legend(loc='upper right')
#pyplot.xlabel("# of isolates")
#pyplot.ylabel("# of lineages")
#pyplot.xlab("# of isolates")
pyplot.savefig("training_data_hist.png",dpi=400)
'''
#exit()
#### stat datasets info
def stat(datasets,pre):
    p=0
    n=0
    for i,(data,label) in enumerate(datasets):
        n+=np.count_nonzero(label == 0)
        p+=np.count_nonzero(label == 1)
    print(pre,' dataset has ', len(datasets),' lineages, ',p,' true SNPs, ',n,' false SNPs, total', n+p,' SNPs',flush=True)

# Save traing and test datasets to local files
#np.savez('training_datasets_large_all.npz', x=train_datasets)
#np.savez('test_datasets_large_all.npz', x=test_datasets)
stat(train_datasets,'Training')
stat(test_datasets,'Test')
stat(j,'Jay_2019')
stat(a,'Arolyn_2022')
stat(jc,'Jacob_2024_Cae')
stat(js,'Jacob_2024_Sep')
exit()


# Create DataLoaders
def create_dataloader(datasets):
    dataloader=[]
    for i, (data,label)  in enumerate(datasets):
        dataset = CustomDataset(data, label)
        #dataset2 = CustomDataset(data2, labels2)
        dataloader_tem = DataLoader(dataset, batch_size=512, shuffle=True)
        #dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=True)
        dataloader.append(dataloader_tem)
    return dataloader

train_loader=create_dataloader(train_datasets)
test_loader=create_dataloader(test_datasets)

# Applying the trained model to new data
# new_data = data_test['x'].astype(np.float64)  # 50 samples, 100 variants, 16 features
# new_dataset = CustomDataset(new_data, np.zeros(len(new_data)))  # Dummy labels since we only want to get predictions
# new_dataloader = DataLoader(new_dataset, batch_size=32, shuffle=False)
# print('Cae_Lineage_4a, input array:',new_data.shape)
# count_ones_l1 = np.count_nonzero(data_test['label'] == 1)
# # Count the number of 0s
# count_zeros_l1 = np.count_nonzero(data_test['label'] == 0)
# print('There are ',count_ones_l1,' true SNPs and ',count_zeros_l1,' false SNPs in Cae_Lineage_4a')
#
exit()
# new_data = data_test2['x'].astype(np.float64)  # 50 samples, 100 variants, 16 features
# new_dataset = CustomDataset(new_data, np.zeros(len(new_data)))  # Dummy labels since we only want to get predictions
# new_dataloader2 = DataLoader(new_dataset, batch_size=32, shuffle=False)
# print('Cae_Lineage_1c, input_array:',new_data.shape)
# count_ones_l1 = np.count_nonzero(data_test2['label'] == 1)
# # Count the number of 0s
# count_zeros_l1 = np.count_nonzero(data_test2['label'] == 0)
# print('There are ',count_ones_l1,' true SNPs and ',count_zeros_l1,' false SNPs in Cae_Lineage_1c')
#exit()


# Training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('The device you are using is: ',device,flush=True)
model = CNNModel_2_new(n_channels=4).to(device)
#model = CNNModel_1(n_channels=13).to(device)
criterion = nn.BCELoss()
#weight = torch.tensor([0.01, 0.99])
#criterion = nn.BCEWithLogitsLoss(weight=weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)
early_stopping = EarlyStopping(patience=20, verbose=True)

valid_losses=[]
num_epochs = 114
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    predictions = []
    y_train = []
    for dataloader in train_loader:
        for inputs, targets in dataloader:
            #print(inputs.shape)
            #exit()
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predictions.extend(outputs.cpu().detach().numpy().flatten())
            y_train.extend(targets.numpy())
    y_pred = (np.array(predictions) > 0.5).astype(int)
    #print(y_train)
    #print(y_pred)
    #exit()
    accuracy = accuracy_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / (len(dataloader)):.4f}, Train acc: {accuracy:.4f}, Train F1: {f1:.4f}',flush=True)

    #print('Train')
    model.eval()

    predictions = []
    y_test=[]
    with torch.no_grad():
        for loader in test_loader:
            for inputs, label in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                test_loss = criterion(outputs, label.to(device).unsqueeze(1))
                valid_losses.append(test_loss.item())
                predictions.extend(outputs.cpu().numpy().flatten())
                y_test.extend(label)
    #loss = criterion(outputs, y_test)
    # Convert predictions to binary labels
    predictions = (np.array(predictions) > 0.5).astype(int)
    # print(predictions)
    y_pred = predictions


    accuracy = accuracy_score(y_test, y_pred)
    # Calculate precision
    precision = precision_score(y_test, y_pred)

    # Calculate recall (sensitivity)
    recall = recall_score(y_test, y_pred)
    # Calculate F1-score
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    valid_loss = np.average(valid_losses)
    print('Validation loss:',valid_loss,', test dataset accuracy is:', accuracy, ' precision:', precision, ' recall:', recall, ' f1-score:', f1, ' AUC:', roc_auc,flush=True)
    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print("Early stopping!!!")
        break
#exit()


#print(new_data.shape,len(new_data))
#torch.save(model.state_dict(),'cae_7000plus_cnn_af.pt')
