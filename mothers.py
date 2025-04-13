import numpy as np
import matplotlib.pyplot as plt
import pywt

# 设置小波类型
wavelets = ['db6', 'bior3.5', 'coif3', 'sym4']
titles = ['Daubechies 6 (db6)', 'Biorthogonal 3.5 (bior3.5)', 
          'Coiflet 3 (coif3)', 'Symlet 4 (sym4)']

# 创建画布
plt.figure(figsize=(14, 10))

for i, (wavelet, title) in enumerate(zip(wavelets, titles), 1):
    # 计算尺度函数和小波函数
    if wavelet.startswith('bior'):
        phi, psi, phi_r, psi_r, x = pywt.Wavelet(wavelet).wavefun(level=5)
    else:
        phi, psi, x = pywt.Wavelet(wavelet).wavefun(level=5)
    
    # 绘制尺度函数
    plt.subplot(4, 2, 2*i-1)
    plt.plot(x, phi, 'b', linewidth=1.5)
    plt.title(f'{title} - Scaling Function')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制小波函数
    plt.subplot(4, 2, 2*i)
    plt.plot(x, psi, 'r', linewidth=1.5)
    plt.title(f'{title} - Wavelet Function')
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('wavelets_compare.pdf', bbox_inches='tight', dpi=300)
plt.show()