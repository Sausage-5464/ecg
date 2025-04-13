import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal

# 1. 生成示例信号
fs = 1000  # 采样频率
t = np.linspace(0, 1, fs, endpoint=False)
x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# 2. 连续小波变换（CWT）
widths = np.arange(1, 100)  # 小波宽度范围
cwtmatr, freqs = pywt.cwt(x, widths, 'morl')  # 使用Morlet小波

# 显示频谱图
plt.figure(figsize=(10, 4))
plt.imshow(np.abs(cwtmatr), extent=[0, 1, 1, 100], cmap='jet', aspect='auto')
plt.colorbar(label='Magnitude')
plt.ylabel('Scale')
plt.xlabel('Time')
plt.title('CWT Spectrum')
plt.show()

# 3. 模拟Inception分析（这里简化为频谱图的滤波处理）
# 例如：对频谱图进行平滑或特征增强
from scipy.ndimage import gaussian_filter
processed_cwt = gaussian_filter(np.abs(cwtmatr), sigma=1)

# 4. 小波逆变换（ICWT）重构信号
# 注意：PyWavelets没有直接提供ICWT，这里使用近似重构
reconstructed_signal = np.sum(processed_cwt, axis=0)  # 简化为各尺度分量求和

# 归一化重构信号（因为CWT/ICWT通常需要精确的标定）
reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal)) * np.max(np.abs(x))

# 5. 残差连接
residual = x - reconstructed_signal
final_output = reconstructed_signal + residual  # 理论上应接近原始信号

# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, x, label='Original Signal')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, reconstructed_signal, label='Reconstructed Signal', color='orange')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, final_output, label='Final Output (with Residual)', color='green')
plt.legend()

plt.tight_layout()
plt.show()