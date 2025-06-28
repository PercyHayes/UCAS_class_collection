from matplotlib import pyplot as plt
import numpy as np


def draw_eight_queens():
    success = np.array([141, 141, 526, 549, 642]) / 10
    avg_steps = [3.99, 6.51, 10.69, 17.01, 459.17]
    labels = ["最陡爬山法", "首选爬山法", "随机重启(最陡)", "随机重启(首选)", "模拟退火"]
    bar_width = 0.35
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置x轴的位置
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots()
    rects1 = ax1.bar(x - bar_width / 2, success, bar_width, label='成功率', color='lightblue')
    # 添加数值标签
    for rect in rects1:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2., height, f'{height:.1f}%', ha='center', va='bottom')

    # 设置y轴标签和标题
    ax1.set_ylabel('成功率/%')
    ax1.set_ylim(0, 100)
    ax1.set_title('八皇后问题算法性能比较')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")  # 旋转标签以避免重叠

    # 绘制平均步数的柱状图
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + bar_width / 2, avg_steps, bar_width, label='平均步数', color='mediumaquamarine')

    # 添加数值标签
    for rect in rects2:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2., height, '%.2f' % height, ha='center', va='bottom')

    # 设置y轴标签
    ax2.set_ylabel('平均步数')

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # 显示图表
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()

def draw_eight_code():
    success = np.array([0,0, 438]) / 10
    avg_steps = [0,0,41578.89]
    labels = ["最陡爬山法", "首选爬山法",  "模拟退火"]
    bar_width = 0.35
    plt.rcParams['font.sans-serif'] = ['SimHei']
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots()
    rects1 = ax1.bar(x - bar_width / 2, success, bar_width, label='成功率', color='lightblue')
    # 添加数值标签
    for rect in rects1:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2., height, f'{height:.1f}%', ha='center', va='bottom')

    # 设置y轴标签和标题
    ax1.set_ylabel('成功率/%')
    ax1.set_ylim(0, 100)
    ax1.set_title('八数码问题算法性能比较')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")  # 旋转标签以避免重叠

    # 绘制平均步数的柱状图
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + bar_width / 2, avg_steps, bar_width, label='平均步数', color='mediumaquamarine')

    # 添加数值标签
    for rect in rects2:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2., height, '%.2f' % height, ha='center', va='bottom')

    # 设置y轴标签
    ax2.set_ylabel('平均步数')

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    # 显示图表
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()


if __name__ == '__main__':
    #draw_eight_queens()
    draw_eight_code()
