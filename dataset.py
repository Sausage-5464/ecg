import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
from utils import ndarray_to_one_hot_tensor
import pandas as pd
from config import get_config

config = get_config()

## I/O
# 获取数据集中的所有记录名
def get_records_name(path):
    with open(path, 'r') as file:
        records = file.read().splitlines()
    return records

# 读取物理信号和标注
def rdr_and_ann(record_name, dataset_name,home_path=config['home_path']):
    # 读取物理信号
    signals, _ = wfdb.rdsamp(home_path + '/' + dataset_name + '/' + record_name, channels=[0])
    
    signal_ann = wfdb.rdann(home_path + '/' + dataset_name + '/' + record_name, 'atr',summarize_labels=True)
    
    return signals, signal_ann

## segematation
def segment_dataset(dataset_name, home_path=config['home_path']):
    width=config['freq'][dataset_name]
    ann_summary = pd.DataFrame()
    segments = []
    labels = []
    record_names = get_records_name(home_path+ '/'+ dataset_name + '/RECORDS')

    for record in record_names:
        signal, signal_ann = rdr_and_ann(record, dataset_name, home_path=home_path)
        
        indices = [i for i, label in enumerate(signal_ann.symbol) if label in config['r_list']]

        for idx in indices[1:]:  # 从第二个idx开始分割，忽略第一个样本
            start = max(0, signal_ann.sample[idx] - width // 2)
            end = start + width
            if end <= len(signal):
                seg = signal[start:end]
                segments.append(seg)
                labels.append(config['aami'][signal_ann.symbol[idx]])  # 使用AAMI中的五类标签
        
        ann_summary = pd.concat([ann_summary, signal_ann.contained_labels], axis=0)
        
        ann_summary = ann_summary[ann_summary['symbol'].isin(config['r_list'])]
        ann_summary = ann_summary.groupby(['label_store', 'symbol', 'description']).agg({'n_occurrences': 'sum'}).reset_index().sort_values(by='n_occurrences', ascending=False)
        label_summary = pd.DataFrame({'label': labels})
    label_summary = label_summary.groupby('label').size().reset_index(name='n_occurrences').sort_values(by='n_occurrences', ascending=False)
    
    
    return {
        "dataset_name": dataset_name,
        "segments": np.array(segments),
        "labels": np.array(labels),
        "ann_summary": ann_summary,
        "label_summary": label_summary
        }

def clean(segs, method='delete'):
    data = torch.from_numpy(np.concatenate([seg['segments'] for seg in segs])).float().squeeze()
    labels = torch.cat([ndarray_to_one_hot_tensor(seg['labels']) for seg in segs]).float()
    has_nan = torch.isnan(data).any(dim=1)
    
    if method == 'replace':
        data = torch.nan_to_num(data, nan=0.0)
    elif method == 'delete':
        # 获取不包含 NaN 的行的索引
        valid_indices = ~has_nan
        # 根据有效索引筛选数据
        data = data[valid_indices]
        labels = labels[valid_indices]
    return data, labels

def randomdownsample(data,labels,seed):
    n_class_index = 0
    n_class_mask = labels[:, n_class_index] == 1  # 获取“N”类样本的掩码
    n_class_indices = torch.nonzero(n_class_mask).squeeze()  # 获取“N”类样本的索引

    # 2. 找到其他类样本的索引
    other_class_indices = torch.nonzero(~n_class_mask).squeeze()

    # 3. 计算其他类样本的总数
    num_other_classes = len(other_class_indices)

    # 4. 从“N”类中随机采样相同数量的样本
    torch.manual_seed(seed)
    sampled_indices = torch.randperm(len(n_class_indices))[:num_other_classes]  # 随机选择
    sampled_n_class_indices = n_class_indices[sampled_indices]

    # 5. 合并降采样后的“N”类样本和其他类样本
    downsampled_indices = torch.cat([sampled_n_class_indices, other_class_indices])

    # 6. 根据索引提取降采样后的 X 和 y
    X_downsampled = data[downsampled_indices]
    y_downsampled = labels[downsampled_indices]

    # 打印结果
    print("降采样后的 X 形状:", X_downsampled.shape)
    print("降采样后的 y 形状:", y_downsampled.shape)
    return X_downsampled,y_downsampled

## packing
class ECGDataset(Dataset):
    def __init__(self, data, labels):
        # 检查数据中是否存在 nan
        self.data = data.unsqueeze(-1)
        print('type of data:',type(self.data))
        print('shape of data:',self.data.shape)
        self.labels = labels
        print('type of labels:',type(self.labels))
        print('shape of labels:',self.labels.shape)
        
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

