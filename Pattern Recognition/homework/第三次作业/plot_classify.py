import matplotlib.pyplot as plt
import numpy as np

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 绘制第一条线
ax.plot(0.5 * np.ones(100), np.linspace(0.25, 5, 100), label='ω_3')

# 绘制第二条线
x1 = np.linspace(-5, 0.5, 100)
ax.plot(x1, 1/2 * x1, label='ω_1')

# 绘制第三条线
x2 = np.linspace(0.5, 5, 100)
ax.plot(x2, (-x2 + 1) / 2, label='ω_2')

# 添加文本标签
ax.text(-3, 3, r'$\omega_1$', fontsize=24)
ax.text(3, 3, r'$\omega_2$', fontsize=24)
ax.text(0.5, -2, r'$\omega_3$', fontsize=24)

# 设置图例
#ax.legend()
ax.set_xlabel("$x_1$",fontsize=12)
ax.set_ylabel("$x_2$",fontsize=12)
# 显示图形
plt.show()