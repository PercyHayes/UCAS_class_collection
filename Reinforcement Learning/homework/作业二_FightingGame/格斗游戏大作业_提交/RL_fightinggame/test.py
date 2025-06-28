"""
Test for game

@Author: Ruoyu Chen
"""

import os
import argparse
import numpy as np
import torch

from fightingice_env import FightingiceEnv
import json

def parse_args():
    parser = argparse.ArgumentParser(description="FightingICE game")
    # general
    parser.add_argument("--CKPT",
                        type=str,
                        default = "checkpoint/SARSA/ZEN/best-299.npy",
                        # default = "E:\NeuBCI\RL_hw2\checkpoints\DQN\ZEN\best-267.pth",
                        help="Model checkpoint path.")
    parser.add_argument("--method", type=str, 
                        default = "SARSA", 
                        choices=["DQN", "SARSA", "Qlearning"],
                        help="Choose which method to use.")
    parser.add_argument("--player", type=str, 
                        default = "ZEN", 
                        choices=["ZEN", "LUD", "GARNET"],
                        help="Choose which character to fight.")
    parser.add_argument("--Rounds",
                        type=int,
                        default = 100,
                        help="How many rounds to evaluate the outcome.")
    parser.add_argument("--log_path", type=str,
                        default="./test_log",
                        help="Choose where to save the log file.")
    
    args = parser.parse_args()
    return args

def epsilon_greedy(obs, model, method="SARSA"):
    """
    epsilon-greedy策略
    Return:
        动作值索引 [0, numActions)范围
    """
    if method == "SARSA":
        # SARSA方法使用简单的矩阵乘法
        return np.argmax(np.dot(model, obs))
    elif method == "DQN":
        # DQN方法使用神经网络
        state = torch.from_numpy(obs).float().unsqueeze(0).to(model.device)
        model.eval()
        with torch.no_grad():
            action_values = model(state)
        return np.argmax(action_values.cpu().data.numpy())
    else:
        raise ValueError(f"Unsupported method: {method}")

def main(args):
    """
    Main function for test.
    """
    env = FightingiceEnv(character=args.player, port=4242)
    env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]

    # 加载模型
    if args.method == "SARSA":
        model = np.load(args.CKPT)
    elif args.method == "DQN":
        from methods.DQN import QNetwork
        model = QNetwork(state_size=144, action_size=40)
        model.load_state_dict(torch.load(args.CKPT))
        model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(model.device)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    Win_num = 0
    result_dict = {
                'own_hp': [],
                'opp_hp': [],
                'hp_diff': [], 
                'result': []
            }
    
    for i in range(args.Rounds):
        obs = env.reset(env_args=env_args)
        reward, done, info = 0, False, None

        while not done:
            act = epsilon_greedy(obs, model, args.method)
            new_obs, reward, done, info = env.step(act)

            if not done:
                obs = new_obs
            elif info is not None:
                result_dict['own_hp'].append(info[0])
                result_dict['opp_hp'].append(info[1])
                result_dict['hp_diff'].append(info[0] - info[1])
                result_dict['result'].append(1 if info[0]>info[1] else 0)
                print(f"round {i+1} result: own hp {info[0]} vs opp hp {info[1]}, you {'win' if info[0]>info[1] else 'lose'}")
                if not os.path.exists(os.path.join(args.log_path, args.method, args.player)):
                    os.makedirs(os.path.join(args.log_path, args.method, args.player))
                with open(os.path.join(args.log_path, args.method, args.player, "test.log"), "a") as f:
                    f.write(f"round {i+1} result: own hp {info[0]} vs opp hp {info[1]}, you {'win' if info[0]>info[1] else 'lose'} hp_diff: {info[0]-info[1]}\n")
                if info[0]>info[1]: 
                    Win_num += 1

    with open(os.path.join(args.log_path, args.method, args.player, "result_dict.json"), "w") as f:
        json.dump(result_dict, f)
    print("In the {} games, AI overcomes MctsAi {} times.".format(args.Rounds, Win_num))

if __name__ == "__main__":
   
    args = parse_args()
    print(args)
    main(args)