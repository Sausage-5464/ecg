import numpy as np 
import matplotlib.pyplot as plt
import torch
from config import get_config
import pywt
import itertools

## data I/O
config = get_config()

# 独热编码
def ndarray_to_one_hot_tensor(str_ndarray, str_to_category=config['aami'], categories=config['categories']):
    # 创建类别到索引的映射
    category_to_index = {category: index for index, category in enumerate(categories)}

    # 使用列表推导式和 NumPy 的 eye 函数生成独热编码
    one_hot_encodings = np.array([np.eye(len(categories))[category_to_index[str_to_category[string]]] for string in str_ndarray])

    # 将 numpy 数组转换为 torch.Tensor
    one_hot_tensor = torch.tensor(one_hot_encodings, dtype=torch.float32)

    return one_hot_tensor


## TimesBlock
    
def FFT_for_Period(x, k):
    # xf shape [B, T, C], denoting the amplitude of frequency(T) given the datapiece at B,N
    xf = torch.fft.rfft(x, dim=1) 

    # find period by amplitudes: here we assume that the periodic features are basically constant
    # in different batch and channel, so we mean out these two dimensions, getting a list frequency_list with shape[T] 
    # each element at pos t of frequency_list denotes the overall amplitude at frequency (t)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0

    #by torch.topk(),we can get the biggest k elements of frequency_list, and its positions(i.e. the k-main frequencies in top_list)
    _, top_list = torch.topk(frequency_list, k)

    #Returns a new Tensor 'top_list', detached from the current graph.
    #The result will never require gradient.Convert to a numpy instance
    top_list = top_list.detach().cpu().numpy()
     
    #period:a list of shape [top_k], recording the periods of mean frequencies respectively
    period = x.shape[1] // top_list

    #Here,the 2nd item returned has a shape of [B, top_k],representing the biggest top_k amplitudes 
    # for each piece of data, with N features being averaged.
    return period, abs(xf).mean(-1)[:, top_list] 

## WaveBlock
def Wavelet_for_Scales(x, top_k=3, wavelet_type='db4'):
    """
    使用小波变换分析时间序列，返回最重要的top_k个尺度(近似周期)
    
    参数:
        x: 输入序列 [B, T, N]
        top_k: 返回的重要尺度数量 (默认3)
        wavelet_type: 小波基类型 (默认'db4')，可选：
            - 'bior3.5' (推荐心律失常分类)
            - 'db4'/'db6' (平衡效率与精度)
            - 'sym8' (高精度需求)
            - 'coif3' (微弱特征检测)
    
    返回:
        scale_list: 重要尺度列表 [top_k]
        energy_weight: 各尺度能量占比 [B, top_k]
    """
    B, T, N = x.shape
    device = x.device
    
    # 验证小波基有效性
    try:
        wavelet = pywt.Wavelet(wavelet_type)
    except ValueError:
        raise ValueError(f"不支持的的小波基: {wavelet_type}。可用选项: {pywt.wavelist(kind='discrete')}")
    
    # 计算最大分解层数（至少保留2个点）
    max_level = min(
        pywt.dwt_max_level(T, wavelet.dec_len),
        int(np.log2(T)) - 1  
    )
    if max_level < 1:
        return [T], torch.ones(B, 1, device=device)  # 序列过短时返回原始长度
    
    # 初始化能量统计
    scale_energies = torch.zeros(B, max_level, device=device)
    
    # 批量处理优化（仍需要循环，但更高效）
    for b, n in itertools.product(range(B), range(N)):
        seq_np = x[b, :, n].cpu().numpy()
        try:
            coeffs = pywt.wavedec(seq_np, wavelet, level=max_level)
            for level, coeff in enumerate(coeffs[1:]):  # 跳过近似系数
                scale_energies[b, level] += torch.sum(torch.from_numpy(coeff).to(device)**2)
        except:
            # 处理可能的分解失败（如全零序列）
            scale_energies[b, :] = 1.0 / max_level  # 均匀分布能量
    
    # 计算能量占比（带平滑）
    total_energy = torch.sum(scale_energies, dim=1, keepdim=True).clamp_min(1e-8)
    energy_ratio = scale_energies / total_energy
    
    # 获取top_k尺度（考虑至少1个周期的合理性）
    _, top_indices = torch.topk(energy_ratio.mean(0), min(top_k, max_level))
    scale_list = [min(2**(i+1), T//2) for i in top_indices]  # 限制不超过半周期
    
    return scale_list, energy_ratio[:, top_indices]

# visualization
def draw_signal(signal, loc=None, title="ECG Signal"):
    """
    绘制心电图信号。

    参数:
    signal (numpy.ndarray): 输入的心电图信号。
    loc (tuple, optional): 要标注的点的坐标，格式为 (x, y)，默认为 None。
    title (str): 图表标题，默认为 "ECG Signal"。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(signal, label="ECG Signal")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    if loc is not None:
        x, y = loc
        if 0 <= x < len(signal):
            plt.scatter(x, y, marker="*", color="red", label="Annotation")

    plt.show()


def draw_bar(df, title="Bar Chart of n_occurrences"):
    """
    绘制条形图，按照数值从大到小排列，色彩丰富。

    参数:
    df (pandas.DataFrame): 输入的数据框，其中包含数值列和类别列。
    num_col (str): 数值列的列名，默认为 'num'。
    class_col (str): 类别列的列名，默认为 'class'。
    title (str): 图表标题，默认为 "Bar Chart"。
    """
    # 按数值列从大到小排序
    df_sorted = df.sort_values(by='n_occurrences', ascending=False)

    # 绘制条形图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df_sorted['symbol'], df_sorted['n_occurrences'], color=plt.cm.tab20.colors)
    plt.title(title)
    plt.xlabel('Symbol')
    plt.ylabel('n_occurrences')

    # 为每个条形图添加数值标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()