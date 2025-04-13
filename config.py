import argparse
import numpy as np

# # data config

import os
# HOME_PATH = os.environ.get('GEMINI_DATA_IN1') # 数据根目录
# OUTPUT_PATH = os.environ.get('GEMINI_DATA_OUT')
# os.makedirs(OUTPUT_PATH, exist_ok=True)
HOME_PATH = 'D:/ecgprogram/ecgdata' # 数据根目录
OUTPUT_PATH = r'D:\ecgprogram\ecgdata\output'

DATASET_NAMES = ['mit-bih'] # 数据集名称列表
FREQ = {
    "mit-bih": 360
} # 数据集的采样频率
R_LIST = np.array(['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 
                   'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 
                   'f', 'Q', '?']) # R点标注
AAMI = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N', 'n': 'N',  'B': 'N','/': 'N','f': 'N',
    'a': 'S', 'J': 'S', 'S': 'S', 'A': 'S',  
    'r': 'V', 'V': 'V', "E": 'V',
    'F': 'F', 
    'Q': 'Q', '?': 'Q'
} # AAMI 5类标签
CATEGORIES = ['N', 'S', 'V', 'F', 'Q'] # 5类类别列表

# model config
SEG_LENGTH = 360 ########

# training config
TRAIN_TEST_RATIO = 0.2 # 测试集在数据集中所占的比例
N_EPOCHS = 30
BATCH_SIZE = 256 ###########
LR= 1e-3 ###########
DROPOUT = 0.1 #########

# TimesBlockConfig
TOP_K = 3 ##########
D_FF = 64 ###########
N_KERNELS = 6 ########
D_MODEL = 32 ##########
N_LAYERS = 2 ###########
  

def get_config():
    parser = argparse.ArgumentParser(description='ECG Program Configuration')
    ### 不改变
    parser.add_argument('--home_path', type=str, default=HOME_PATH, help='Data root directory')
    parser.add_argument('--output_path', type=str, default=OUTPUT_PATH, help='Output root directory')
    parser.add_argument('--n_epochs', type=int, default=N_EPOCHS, help='Number of training epochs')
    parser.add_argument('--dataset_names', type=list, default=DATASET_NAMES, help='Dataset names')
    parser.add_argument('--freq', type=dict, default=FREQ, help='Sampling frequency of datasets')
    parser.add_argument('--r_list', type=np.array, default=R_LIST, help='R point annotation')
    parser.add_argument('--aami', type=dict, default=AAMI, help='AAMI 5 categories')
    parser.add_argument('--categories', type=list, default=CATEGORIES, help='5 categories')
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--use-gpu", type=bool, default=False, help="Use GPU")
    parser.add_argument("--wavelet", type=str, default='bior3.5', help="Type of wavelet basis")
    
    
    # 不重点调优参数
    parser.add_argument('--lr', type=float, default=LR, help='Learning rate')
    parser.add_argument('--drop_out', type=float, default=DROPOUT, help='Dropout rate')
    parser.add_argument('--train_test_ratio', type=float, default=TRAIN_TEST_RATIO, help='Proportion of test set in the dataset')
    
    # 重点调优参数
    parser.add_argument('--n_layers', type=int, default=N_LAYERS, help='Num of TimesBlock')
    parser.add_argument('--top_k', type=int, default=TOP_K, help='Top K value')
    parser.add_argument('--d_ff', type=int, default=D_FF, help='Feed-forward dimension')
    parser.add_argument('--num_kernels', type=int, default=N_KERNELS, help='Number of kernels')
    parser.add_argument('--d_model', type=int, default=D_MODEL, help='D-model of TimesNet')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--seg_len', type=int, default=SEG_LENGTH, help='Length of each segment')
    
    args, _ = parser.parse_known_args()
    return vars(args)