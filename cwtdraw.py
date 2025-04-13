import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

myfont = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 根据你的系统路径修改字体路径，这里以黑体为例
plt.rcParams['font.family'] = myfont.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 生成信号（含高频脉冲和低频振荡）
t = np.linspace(0, 1, 1000, endpoint=False)
x = np.sin(2 * np.pi * 10 * t)  # 低频
x[500:520] += np.random.normal(0, 1, 20)  # 高频脉冲

# 计算CWT
scales = np.arange(1, 128)
coef, freqs = pywt.cwt(x, scales, 'morl')  # Morlet小波

plt.figure(figsize=(10, 4))
plt.plot(t, x, 'b-', linewidth=1.5, label='Original Signal')
plt.axvspan(0.5, 0.52, color='red', alpha=0.3, label='High-frequency Impulse')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Original Signal with Impulse and Noise')
plt.legend()
plt.grid(True)
plt.show()

# 绘制时频图
plt.imshow(np.abs(coef), extent=[0, 1, 1, 128], cmap='jet', aspect='auto')
plt.colorbar(label='Magnitude')
plt.xlabel('Time [s]')
plt.ylabel('Scale')
plt.title('CWT时频分析（Morlet小波）')
plt.show()

# DWT分解与重构
coeffs = pywt.wavedec(x, 'db4', level=4)  # 4层分解，Daubechies4小波
cA4, cD4, cD3, cD2, cD1 = coeffs  # 各层系数

# 重构信号
x_reconstructed = pywt.waverec(coeffs, 'db4')

# 绘制分解结果
plt.figure(figsize=(10, 6))
for i, coeff in enumerate(coeffs):
    plt.subplot(len(coeffs), 1, i+1)
    plt.plot(coeff)
    plt.title(f'Level {4-i}: {"Approx" if i==0 else "Detail"}')
plt.tight_layout()
plt.show()