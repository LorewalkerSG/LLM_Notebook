import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. 生成 x 轴的数据点
# 标准正态分布的绝大部分数据集中在 -4 到 4 之间
x = np.linspace(-4, 4, 1000)

# 2. 计算对应的 CDF (累积分布函数) 值
# norm.cdf 默认就是标准正态分布 (loc=0, scale=1)
y = norm.cdf(x)*x

# 3. 使用 pyplot 进行绘制
plt.figure(figsize=(8, 5))
plt.plot(x, y, color='#1f77b4', linewidth=2, label='CDF: $\Phi(x)$')
plt.plot(x,0.5*x,label='0.5x')
# 4. 图表格式化设置
plt.title('Cumulative Distribution Function of Standard Normal Distribution')
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5) # x轴
plt.axvline(0, color='black', linewidth=0.5) # y轴

# 突出显示 x=0 时，y=0.5 的关键点
plt.scatter(0, 0.5, color='red', zorder=5)
plt.annotate('$\Phi(0) = 0.5$', xy=(0, 0.5), xytext=(0.5, 0.4),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.legend()
plt.tight_layout()

# 5. 显示图表
plt.show()