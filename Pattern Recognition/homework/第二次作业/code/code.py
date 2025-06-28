
import numpy as np
import matplotlib.pyplot as plt


# 样本数据
data = np.array([4.6019, 5.2564, 5.2200, 3.2886, 3.7942,
                 3.2271, 4.9275, 3.2789, 5.7019, 3.9945,
                 3.8936, 6.7906, 7.1624, 4.1807, 4.9630,
                 6.9630, 4.4597, 6.7175, 5.8198, 5.0555,
                 4.6469, 6.6931, 5.7111, 4.3672, 5.3927,
                 4.1220, 5.1489, 6.5319, 5.5318, 4.2403,
                 5.3480, 4.3022, 7.0193, 3.2063, 4.3405,
                 5.7715, 4.1797, 5.0179, 5.6545, 6.2577,
                 4.0729, 4.8301, 4.5283, 4.8858, 5.3695,
                 4.3814, 5.8001, 5.4267, 4.5277, 5.2760])

# 方窗 Parzen 窗密度估计函数
def parzen_rectangular(data, x, h):
    count = 0
    n = len(data)
    for xi in data:
        if abs(xi - x) <= h/2:
            count += 1
    return count / (n * h)

# 高斯窗 Parzen 窗密度估计函数
def parzen_gaussian(data, x, h):
    n = len(data)
    sum_val = 0
    for xi in data:
        sum_val += np.exp(-(x - xi)**2 / (2 * h**2))
    return sum_val / (n * (np.sqrt(2 * np.pi) * h))

# 绘制不同窗宽下的概率密度函数曲线
def plot_density_estimates(data, window_type):
    x_vals = np.linspace(min(data) - 1, max(data) + 1, 500)
    plt.figure(figsize=(10, 6))
    for h in [0.2, 0.5, 1, 2]:
        y_vals = []
        for x in x_vals:
            if window_type == 'rectangular':
                y_vals.append(parzen_rectangular(data, x, h))
            else:
                y_vals.append(parzen_gaussian(data, x, h))

       # width = (min(data) - 1 +max(data) + 1)/ len(x_vals)
       # area = 0
       # for height in y_vals:
       #     area += height * width
       # print(f"Area under the curve for {window_type} window with width {h} is {area}")

        plt.plot(x_vals, y_vals, label=f'Window Width = {h} ({window_type} window)')
    plt.title(f'Probability Density Estimation using {window_type} Parzen Window')
    plt.xlabel('Data Values')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()
    


# 绘制方窗和高斯窗的概率密度函数曲线
plot_density_estimates(data, 'rectangular')
plot_density_estimates(data, 'gaussian')


