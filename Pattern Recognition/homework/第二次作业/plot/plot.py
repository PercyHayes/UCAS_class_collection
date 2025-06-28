import numpy as np
import matplotlib.pyplot as plt

# 定义函数 p_n(x)
def p_n(x, k_n=1, n=1, V_n=1):
    # Create an array of the same shape as x with the piecewise function applied element-wise
    return np.where(x < -2, k_n / (n * V_n) * (1 / (10 * np.abs(x + 4))),
                    np.where((x > -2) & (x < 3), k_n / (n * V_n) * (1 / (10 * np.abs(x))),
                             k_n / (n * V_n) * (1 / (10 * np.abs(x - 6)))))


# 生成 x 值
x = np.linspace(-9, 9, 1000)

# 计算对应的 y 值
y = p_n(x)

# Plot the curve
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 
plt.plot(x, y,label='$p_n(x)$')

# Add vertical lines at x = -4, x = 0, and x = 6
plt.axvline(x=-4, color='red', linestyle='--', label='样本1')
plt.axvline(x=0, color='green', linestyle='--', label='样本2')
plt.axvline(x=6, color='blue', linestyle='--', label='样本3')

# Add labels and title
plt.xlabel('x')
plt.ylabel('$p_n(x)$')
plt.title('概率密度函数曲线')
plt.legend()
plt.grid(True)
plt.show()