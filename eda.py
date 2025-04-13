import dataset
from config import get_config
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
myfont = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 根据你的系统路径修改字体路径，这里以黑体为例
plt.rcParams['font.family'] = myfont.get_name()

config = get_config()

seg_result = dataset.segment_dataset("mit-bih")

print(f"数据集 {'mit-bih'.upper()} 的摘要:")
print("标签摘要:")
print(seg_result['ann_summary'])

# 绘制心拍种类数量条形图
# label_summary = seg_result['label_summary']
# plt.figure(figsize=(10, 6))
# bars = plt.bar(label_summary['label'], label_summary['n_occurrences'])
# plt.xlabel('心拍种类标签')
# plt.ylabel('数量')
# plt.title(f'{"mit-bih".upper()} 数据集心拍种类数量分布')

annsummary = seg_result['ann_summary']
plt.figure(figsize=(10, 6))
bars = plt.bar(annsummary['symbol'], annsummary['n_occurrences'])
plt.xlabel('心拍标记')
plt.ylabel('数量')
plt.title(f'{"mit-bih".upper()} 数据集心拍标记数量分布')



# 在柱顶标上 n_occurrences 的值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), 
            ha='center', va='bottom')

plt.tight_layout()
plt.show()

