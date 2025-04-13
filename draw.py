import dataset
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from scipy.signal import stft
import pywt
from scipy.fft import fft, fftfreq

mit = dataset.segment_dataset("mit-bih")

segs = mit["segments"]
labels = mit["labels"]

myfont = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 根据你的系统路径修改字体路径，这里以黑体为例
plt.rcParams['font.family'] = myfont.get_name()
plt.rcParams['axes.unicode_minus'] = False

# # 遍历不同的标号
# unique_labels = np.unique(labels)
# for label in unique_labels:
#     # 找到第一个对应标号的时间序列的索引
#     idx = np.where(labels == label)[0][0]
#     selected_time_series = segs[idx, :, 0]

#     # 绘制折线图
#     plt.figure()
#     plt.plot(selected_time_series)
#     plt.title(f'标号为 {label} 的心电图示例')
#     plt.xlabel('序号')
#     plt.ylabel('mV')
#     plt.show()
  
first_index = np.where(labels == 'N')[0][0]
N_series = segs[first_index]
print(N_series.shape)

# Perform Fourier Transform
N_series_fft = fft(N_series[:, 0])
freqs = fftfreq(len(N_series_fft), d=1/360)  # Assuming a sampling rate of 360 Hz

plt.figure()
plt.plot(freqs[:len(freqs)//2], np.abs(N_series_fft[:len(freqs)//2]))
plt.title('傅里叶变换结果')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅值')
plt.show()

# Perform Short-Time Fourier Transform (STFT)
f, t, Zxx = stft(N_series[:, 0], fs=360, nperseg=128)  # Adjust nperseg as needed

plt.figure()
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('短时傅里叶变换结果')
plt.xlabel('时间 (s)')
plt.ylabel('频率 (Hz)')
plt.colorbar(label='幅值')
plt.show()

# Perform Wavelet Transform
coeffs, freqs = pywt.cwt(N_series[:, 0], scales=np.arange(1, 128), wavelet='mexh')  # Adjust scales and wavelet as needed

plt.figure()
plt.imshow(np.abs(coeffs), extent=[0, len(N_series[:, 0])/360, 1, 128], cmap='jet', aspect='auto', origin='lower')
plt.title('小波变换结果')
plt.xlabel('时间 (s)')
plt.ylabel('尺度')
plt.colorbar(label='幅值')
plt.show()
