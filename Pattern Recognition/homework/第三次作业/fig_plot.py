import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

Y1 = np.array([[0.1, 1.1, 1],[6.8, 7.1, 1],[-3.5, -4.1, 1],[2.0, 2.7, 1],[4.1, 2.8, 1],[3.1, 5.0, 1],[-0.8, -1.3, 1],[0.9, 1.2, 1],[5.0, 6.4, 1],[3.9, 4.0, 1],
               [3.0, 2.9, -1],[-0.5, -8.7, -1],[-2.9, -2.1, -1],[0.1, -5.2, -1],[4.0, -2.2, -1],[1.3, -3.7, -1],[3.4, -6.2, -1],[4.1, -3.4, -1],[5.1, -1.6, -1],[-1.9, -5.1, -1]]).astype(np.float32)

Y2 = np.array([[7.1, 4.2, 1],[-1.4, -4.3, 1],[4.5, 0.0, 1],[6.3, 1.6, 1],[4.2, 1.9, 1],[1.4, -3.2, 1],[2.4, -4.0, 1],[2.5, -6.1, 1],[8.4, 3.7, 1],[4.1, -2.2, 1],
               [2.0, 8.4, -1],[8.9, -0.2, -1],[4.2, 7.7, -1],[8.5, 3.2, -1],[6.7, 4.0, -1],[0.5, 9.2, -1],[5.3, 6.7, -1],[8.7, 6.4, -1],[7.1, 9.7, -1],[8.0, 6.3, -1]])

# 提取 Y1 中的数据
omega1_x, omega1_y = Y1[Y1[:, 2]==1][:, 0], Y1[Y1[:, 2]==1][:, 1]
omega3_x, omega3_y = -Y1[Y1[:, 2]==-1][:, 0], -Y1[Y1[:, 2]==-1][:, 1]

# 提取 Y2 中的数据
omega2_x, omega2_y = Y2[Y2[:, 2]==1][:, 0], Y2[Y2[:, 2]==1][:, 1]
omega4_x, omega4_y = -Y2[Y2[:, 2]==-1][:, 0], -Y2[Y2[:, 2]==-1][:, 1]

# 绘制散点图
plt.scatter(omega1_x, omega1_y, c='red', label=r'$\omega_1$')
plt.scatter(omega2_x, omega2_y, c='blue', label=r'$\omega_2$')
plt.scatter(omega3_x, omega3_y, c='green', label=r'$\omega_3$')
plt.scatter(omega4_x, omega4_y, c='purple', label=r'$\omega_4$')

# 补画直线 0.00533945x + 0.00474587y + 0.03688261 = 0
x = np.linspace(-10, 10, 400)
y = (-0.03688 - 0.005339*x) / 0.004746
plt.plot(x, y, 'k--', c='black',label=r'$\omega_2$和$\omega_4$的分类面')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-12, 12)
plt.ylim(-12, 12)
# 添加图例
plt.legend()

# 显示图形
plt.show()