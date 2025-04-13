import matplotlib.pyplot as plt

# 五种类别

labels = ['N', 'V', 'S', 'F', 'Q']
# 概率取值
probabilities = [1, 0, 0, 0, 0]

# 绘制直方图
plt.bar(labels, probabilities)

# 在柱顶部标注值
for i, v in enumerate(probabilities):
    plt.text(i, v + 0.01, str(v), ha='center')

# 设置图表标题和标签
plt.title('One-hot label')
plt.xlabel('Categories')
plt.ylabel('Probability')

# 显示图表
plt.show()