import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 根据你的系统路径修改字体路径，这里以黑体为例
plt.rcParams['font.family'] = myfont.get_name()
# 窗函数参数
L = 64  # 窗长度
n = np.arange(L)

# 定义窗函数
windows = {
    "矩形窗": np.ones(L),
    "汉宁窗": signal.windows.hann(L),
    "海明窗": signal.windows.hamming(L),
    "布莱克曼窗": signal.windows.blackman(L),
    "高斯窗 (σ=0.3)": signal.windows.gaussian(L, std=0.3*L/2),
    "平顶窗": signal.windows.flattop(L)
}

# 绘制时域窗函数
plt.figure(figsize=(12, 6))
for i, (name, win) in enumerate(windows.items(), 1):
    plt.subplot(2, 3, i)
    plt.plot(win, 'b-', linewidth=2)
    plt.title(name)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制频域响应（对数尺度）
plt.figure(figsize=(12, 6))
for i, (name, win) in enumerate(windows.items(), 1):
    freq, response = signal.freqz(win, fs=2*np.pi)
    plt.semilogy(freq, np.abs(response), label=name, linewidth=2)
plt.title('窗函数频域响应（主瓣与旁瓣）')
plt.xlabel('Normalized Frequency [rad/sample]')
plt.ylabel('Magnitude (dB)')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim(0, np.pi)
plt.ylim(1e-4, 1.1)
plt.show()