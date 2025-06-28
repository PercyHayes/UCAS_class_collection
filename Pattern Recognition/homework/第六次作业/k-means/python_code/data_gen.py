import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 设置均值和协方差矩阵
Sigma = np.array([[1, 0], [0, 1]])
mu1 = np.array([1, -1])
mu2 = np.array([5.5, -4.5])
mu3 = np.array([1, 4])
mu4 = np.array([6, 4.5])
mu5 = np.array([9, 0.0])

# 生成数据点
x1 = multivariate_normal.rvs(mean=mu1, cov=Sigma, size=200)
x2 = multivariate_normal.rvs(mean=mu2, cov=Sigma, size=200)
x3 = multivariate_normal.rvs(mean=mu3, cov=Sigma, size=200)
x4 = multivariate_normal.rvs(mean=mu4, cov=Sigma, size=200)
x5 = multivariate_normal.rvs(mean=mu5, cov=Sigma, size=200)

# 合并数据点
X = np.vstack((x1, x2, x3, x4, x5))

# 保存数据
np.save('./data_x.npy', X)

# 显示数据点
plt.scatter(x1[:, 0], x1[:, 1], c='r', marker='.')
plt.scatter(x2[:, 0], x2[:, 1], c='b', marker='.')
plt.scatter(x3[:, 0], x3[:, 1], c='k', marker='.')
plt.scatter(x4[:, 0], x4[:, 1], c='g', marker='.')
plt.scatter(x5[:, 0], x5[:, 1], c='m', marker='.')
plt.show()