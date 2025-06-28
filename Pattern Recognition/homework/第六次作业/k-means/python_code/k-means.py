import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# 加载数据
X = np.load('data_x.npy')

# K-means 聚类算法
def kmeans_clustering(X, k, max_iters=100):
    # 随机选择初始聚类中心
    C = X[np.random.choice(X.shape[0], k, replace=False)]
    prev_C = C.copy()
    for _ in range(max_iters):
        # 计算每个点到聚类中心的距离
        D = cdist(X, C)
        # 分配每个点到最近的聚类中心
        idx = np.argmin(D, axis=1)
        # 更新聚类中心
        for i in range(k):
            C[i] = X[idx == i].mean(axis=0)
        # 检查聚类中心是否收敛
        if np.all(C == prev_C):
            break
        prev_C = C.copy()
    return idx, C

# 设置聚类个数
k = 5

# 运行K-means聚类算法
idx, C = kmeans_clustering(X, k)

# 计算聚类精度和误差
mu = np.array([[1, -1], [5.5, -4.5], [1, 4], [6, 4.5], [9, 0.0]])

# 确定聚类中心和真实值的对应关系
correspondence = np.zeros(k, dtype=int)
for i in range(k):
    distances = np.sqrt(np.sum((C - mu[i])**2, axis=1))
    correspondence[i] = np.argmin(distances)

# 计算误差
error = np.sqrt(np.sum((C[correspondence] - mu)**2, axis=1))

# 计算聚类精度
correct = 0
for i in range(k):
    correct += np.sum(idx == correspondence[i])
accuracy = correct / X.shape[0]

# 显示结果
print('聚类中心:')
print(C)
print('误差:')
print(error)
print('聚类精度:')
print(accuracy)

# 绘制聚类结果
colors = ['r', 'b', 'k', 'g', 'm']
plt.figure()
for i in range(k):
    plt.scatter(X[idx == i, 0], X[idx == i, 1], c=colors[i], s=36)
plt.scatter(C[:, 0], C[:, 1], c='y', s=100, marker='d')
plt.show()