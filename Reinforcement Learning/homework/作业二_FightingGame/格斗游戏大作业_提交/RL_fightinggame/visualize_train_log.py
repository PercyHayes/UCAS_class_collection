import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def moving_average(data, window_size):
    """计算滑动平均"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def get_method_color(method):
    """获取方法对应的颜色"""
    color_map = {
        'SARSA': 'steelblue',
        'DQN': 'darkorange', 
        'QLearning': 'forestgreen'
    }
    return color_map.get(method, 'purple')

def plot_individual_metrics(log_path, method, player):
    """
    为单个方法绘制各项指标的变化曲线
    """
    # 读取JSON文件
    json_path = os.path.join(log_path, method, player, "result_dict.json")
    with open(json_path, 'r') as f:
        result_dict = json.load(f)
    
    # 创建保存目录
    save_dir = os.path.join(log_path, method, player)
    
    # 获取方法对应的颜色
    method_color = get_method_color(method)
    light_color = method_color.replace('steel', 'light').replace('dark', 'light').replace('forest', 'light')
    
    # 1. 血量差变化曲线（带滑动平均）
    hp_diff = np.array(result_dict['hp_diff'])
    episodes = np.arange(len(hp_diff))
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, hp_diff, alpha=0.5, color=light_color, label='Original')
    
    # 滑动平均
    window_size = min(20, len(hp_diff)//5)  # 动态调整窗口大小
    if len(hp_diff) > window_size:
        hp_diff_smooth = moving_average(hp_diff, window_size)
        smooth_episodes = episodes[window_size-1:]
        plt.plot(smooth_episodes, hp_diff_smooth, color=method_color, linewidth=3, 
                label=f'Moving Average (window={window_size})')
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.title(f'HP Difference Over Episodes - {method} - {player}', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('HP Difference', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hp_diff_curve.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. 胜负结果变化
    if 'result' in result_dict:
        results = np.array(result_dict['result'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, results, alpha=0.5, color=light_color, marker='o', markersize=2)
        
        # 胜率滑动平均
        if len(results) > window_size:
            win_rate_smooth = moving_average(results, window_size) * 100
            plt.plot(smooth_episodes, win_rate_smooth, color=method_color, linewidth=3, 
                    label=f'Win Rate Moving Average (window={window_size})')
        
        plt.title(f'Battle Results Over Episodes - {method} - {player}', fontsize=14)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Result (1=Win, 0=Loss)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'results_curve.pdf'), bbox_inches='tight', dpi=300)
        plt.close()
    
    # 3. Reward变化曲线（从训练log文件读取）
    try:
        log_file_path = os.path.join(log_path, method, player, "train.log")
        rewards = []
        with open(log_file_path, 'r') as f:
            for line in f:
                if 'reward:' in line:
                    # 提取reward数值
                    reward_str = line.split('reward:')[1].strip().split()[0]
                    try:
                        reward = float(reward_str)
                        rewards.append(reward)
                    except ValueError:
                        continue
        
        if rewards:
            rewards = np.array(rewards)
            reward_episodes = np.arange(len(rewards))
            
            plt.figure(figsize=(12, 6))
            plt.plot(reward_episodes, rewards, alpha=0.5, color=light_color, label='Original')
            
            # 滑动平均
            reward_window = min(20, len(rewards)//5)
            if len(rewards) > reward_window:
                reward_smooth = moving_average(rewards, reward_window)
                reward_smooth_episodes = reward_episodes[reward_window-1:]
                plt.plot(reward_smooth_episodes, reward_smooth, color=method_color, linewidth=3, 
                        label=f'Moving Average (window={reward_window})')
            
            plt.title(f'Reward Over Episodes - {method} - {player}', fontsize=14)
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Reward', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'reward_curve.pdf'), bbox_inches='tight', dpi=300)
            plt.close()
            
    except FileNotFoundError:
        print(f"No train.log found for {method} - {player}")
    
    # 4. 其他指标（如果存在）
    for key, values in result_dict.items():
        if key not in ['hp_diff', 'result'] and isinstance(values, list):
            plt.figure(figsize=(12, 6))
            values_array = np.array(values)
            plt.plot(episodes[:len(values_array)], values_array, alpha=0.5, color=light_color)
            
            # 滑动平均
            if len(values_array) > window_size:
                smooth_values = moving_average(values_array, window_size)
                smooth_episodes_adj = episodes[window_size-1:window_size-1+len(smooth_values)]
                plt.plot(smooth_episodes_adj, smooth_values, color=method_color, linewidth=3, 
                        label=f'Moving Average (window={window_size})')
            
            plt.title(f'{key.replace("_", " ").title()} Over Episodes - {method} - {player}', fontsize=14)
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel(key.replace("_", " ").title(), fontsize=12)
            if len(values_array) > window_size:
                plt.legend()
            plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{key}_curve.pdf'), bbox_inches='tight', dpi=300)
            plt.close()
    
    print(f"Individual plots saved for {method} - {player}")

def plot_hp_diff_comparison(log_path, player="ZEN"):
    """
    绘制三种方法的血量差对比图
    """
    plt.figure(figsize=(14, 8))
    
    methods = ["SARSA", "DQN", "QLearning"]
    colors = ['steelblue', 'darkorange', 'forestgreen']  # 与test可视化统一
    
    for method, color in zip(methods, colors):
        try:
            # 读取数据
            json_path = os.path.join(log_path, method, player, "result_dict.json")
            with open(json_path, 'r') as f:
                result_dict = json.load(f)
            
            hp_diff = np.array(result_dict['hp_diff'])
            episodes = np.arange(len(hp_diff))
            
            # 原始数据（半透明）
            plt.plot(episodes, hp_diff, alpha=0.5, color=color)
            
            # 滑动平均
            window_size = min(20, len(hp_diff)//5)
            if len(hp_diff) > window_size:
                hp_diff_smooth = moving_average(hp_diff, window_size)
                smooth_episodes = episodes[window_size-1:]
                plt.plot(smooth_episodes, hp_diff_smooth, color=color, linewidth=3, 
                        label=f'{method} (MA-{window_size})')
            else:
                plt.plot(episodes, hp_diff, color=color, linewidth=3, label=method)
                
        except FileNotFoundError:
            print(f"No data found for {method} - {player}")
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=2)
    plt.title(f'HP Difference Comparison Over Episodes - {player}', fontsize=16, pad=20)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('HP Difference', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(log_path, f"{player}_hp_diff_comparison.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"HP difference comparison plot saved to: {save_path}")

def plot_reward_comparison(log_path, player="ZEN"):
    """
    绘制三种方法的reward对比图
    """
    plt.figure(figsize=(14, 8))
    
    methods = ["SARSA", "DQN", "QLearning"]
    colors = ['steelblue', 'darkorange', 'forestgreen']
    
    for method, color in zip(methods, colors):
        try:
            # 从训练log文件读取reward
            log_file_path = os.path.join(log_path, method, player, "train.log")
            rewards = []
            with open(log_file_path, 'r') as f:
                for line in f:
                    if 'reward:' in line:
                        reward_str = line.split('reward:')[1].strip().split()[0]
                        try:
                            reward = float(reward_str)
                            rewards.append(reward)
                        except ValueError:
                            continue
            
            if rewards:
                rewards = np.array(rewards)
                episodes = np.arange(len(rewards))
                
                # 原始数据（半透明）
                plt.plot(episodes, rewards, alpha=0.5, color=color)
                
                # 滑动平均
                window_size = min(20, len(rewards)//5)
                if len(rewards) > window_size:
                    reward_smooth = moving_average(rewards, window_size)
                    smooth_episodes = episodes[window_size-1:]
                    plt.plot(smooth_episodes, reward_smooth, color=color, linewidth=3, 
                            label=f'{method} (MA-{window_size})')
                else:
                    plt.plot(episodes, rewards, color=color, linewidth=3, label=method)
                    
        except FileNotFoundError:
            print(f"No train.log found for {method} - {player}")
    
    plt.title(f'Reward Comparison Over Episodes - {player}', fontsize=16, pad=20)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(log_path, f"{player}_reward_comparison.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Reward comparison plot saved to: {save_path}")

if __name__ == "__main__":
    # 设置参数
    log_path = "./train_log"
    methods = ["SARSA", "DQN", "QLearning"]
    players = ["ZEN"]  # 根据实际情况调整
    
    # 为每个方法和角色生成单独的指标图
    for method in methods:
        for player in players:
            try:
                plot_individual_metrics(log_path, method, player)
            except FileNotFoundError:
                print(f"No data found for {method} - {player}")
            except Exception as e:
                print(f"Error processing {method} - {player}: {str(e)}")
    
    # 生成对比图
    for player in players:
        plot_hp_diff_comparison(log_path, player)
        plot_reward_comparison(log_path, player)
    
    print("All training log visualizations completed!")