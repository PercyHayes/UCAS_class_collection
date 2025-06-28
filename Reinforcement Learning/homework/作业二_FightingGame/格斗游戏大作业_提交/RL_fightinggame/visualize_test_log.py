import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def plot_hp_diff_distribution(log_path, method, player):
    """
    绘制血量差分布图
    Args:
        log_path: 日志文件路径
        method: 方法名称 (SARSA/DQN/Qlearning)
        player: 角色名称 (ZEN/LUD/GARNET)
    """
    # 读取JSON文件
    json_path = os.path.join(log_path, method, player, "result_dict.json")
    with open(json_path, 'r') as f:
        result_dict = json.load(f)
    
    # 获取血量差数据
    hp_diff = np.array(result_dict['hp_diff'])
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(hp_diff, bins=30,  
             alpha=0.7, color='skyblue', edgecolor='black')
    
    # 添加垂直线表示0点
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # 计算统计信息
    mean_diff = np.mean(hp_diff)
    std_diff = np.std(hp_diff)
    win_rate = np.mean(result_dict['result']) * 100
    
    # 设置标题和标签
    plt.title(f'HP Difference Distribution\n{method} - {player}', fontsize=14)
    plt.xlabel('HP Difference', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # 添加统计信息文本框
    stats_text = f'Mean: {mean_diff:.2f}\nStd: {std_diff:.2f}\nWin Rate: {win_rate:.1f}%'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    save_path = os.path.join(log_path, method, player, "hp_diff_distribution.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Distribution plot saved to: {save_path}")

def plot_zen_comparison(log_path):
    """
    绘制ZEN角色下三个算法的血量差分布对比图
    Args:
        log_path: 日志文件路径
    """
    plt.figure(figsize=(12, 8))
    
    methods = ["SARSA", "DQN", "Qlearning"]
    colors = ['steelblue', 'darkorange', 'forestgreen']
    
    stats_info = []  # 存储统计信息
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        try:
            # 读取数据
            json_path = os.path.join(log_path, method, "ZEN", "result_dict.json")
            with open(json_path, 'r') as f:
                result_dict = json.load(f)
            
            hp_diff = np.array(result_dict['hp_diff'])
            mean_diff = np.mean(hp_diff)
            win_rate = np.mean(result_dict['result']) * 100
            
            # 存储统计信息
            stats_info.append(f'{method}: Mean={mean_diff:.1f}, Win Rate={win_rate:.1f}%')
            
            # 绘制直方图 - 使用半透明效果
            plt.hist(hp_diff, bins=30,
                     alpha=0.3, color=color, edgecolor=color, 
                     linewidth=2, label=method)
            
        except FileNotFoundError:
            print(f"No data found for {method} - ZEN")
    
    # 添加垂直线表示0点
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.8, linewidth=2)
    
    # 设置标题和标签
    plt.title('HP Difference Distribution Comparison - ZEN', fontsize=16, pad=20)
    plt.xlabel('HP Difference', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    # 添加图例
    plt.legend(fontsize=12, loc='upper right')
    
    # 在图上添加统计信息
    stats_text = '\n'.join(stats_info)
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
             fontsize=11)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 美化图形
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(log_path, "zen_comparison.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"ZEN comparison plot saved to: {save_path}")

def generate_zen_summary_table(log_path):
    """
    生成ZEN角色的算法性能汇总表格
    Args:
        log_path: 日志文件路径
    """
    methods = ["SARSA", "DQN", "Qlearning"]
    
    print("\n" + "="*50)
    print("ZEN CHARACTER PERFORMANCE SUMMARY")
    print("="*50)
    print(f"{'Method':<12} {'Win Rate':<10} {'Mean HP Diff':<12}")
    print("-"*50)
    
    for method in methods:
        try:
            json_path = os.path.join(log_path, method, "ZEN", "result_dict.json")
            with open(json_path, 'r') as f:
                result_dict = json.load(f)
            
            win_rate = np.mean(result_dict['result']) * 100
            mean_diff = np.mean(result_dict['hp_diff'])
            
            print(f"{method:<12} {win_rate:<10.1f} {mean_diff:<12.1f}")
            
        except FileNotFoundError:
            print(f"{method:<12} {'N/A':<10} {'N/A':<12}")
    
    print("="*50)

def main():
    """
    主函数：执行所有可视化任务（仅ZEN角色）
    """
    # 设置参数
    log_paths = {
        "unlimited": "./test_log",           # 无限血量测试
        "limited": "./test_limithp_log"      # 限制血量测试
    }
    
    methods = ["SARSA", "DQN", "Qlearning"]
    
    for log_type, log_path in log_paths.items():
        if not os.path.exists(log_path):
            print(f"Log path not found: {log_path}")
            continue
            
        print(f"\nProcessing {log_type} HP mode logs...")
        
        # 1. 生成ZEN角色的单独分布图
        print("Generating individual distribution plots for ZEN...")
        for method in methods:
            try:
                plot_hp_diff_distribution(log_path, method, "ZEN")
            except FileNotFoundError:
                print(f"No data found for {method} - ZEN")
            except Exception as e:
                print(f"Error processing {method} - ZEN: {str(e)}")
        
        # 2. 生成ZEN角色的对比图
        print("Generating ZEN comparison plot...")
        try:
            plot_zen_comparison(log_path)
        except Exception as e:
            print(f"Error creating ZEN comparison plot: {str(e)}")
        
        # 3. 生成ZEN角色的汇总表格
        print("Generating ZEN summary table...")
        generate_zen_summary_table(log_path)

if __name__ == "__main__":
    main()