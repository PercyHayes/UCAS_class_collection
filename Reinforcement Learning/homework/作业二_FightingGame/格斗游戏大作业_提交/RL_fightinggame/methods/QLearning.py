"""
Created by WenYaozhi 2025
"""

import numpy as np
import os
import random
import json

class QLearning():
    def __init__(self,
        args,
        environment,
        environment_args,
        numStates = 144, 
        numActions = 40, 
        decay = 0.9995, 
        epsilon = 1, 
        gamma = 0.95):
        super(QLearning, self).__init__()
        self.args = args
        self.decay = decay
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.alpha = 0.01
        self.epsilon_decay = 0.95

        self.env = environment
        self.env_args = environment_args

        self.numStates = numStates
        self.numActions = numActions

        self.transformer = np.zeros((self.numActions, self.numStates))

    def epsilon_greedy(self, obs):

        if np.random.random() > self.epsilon:
            return np.argmax(
                np.dot(self.transformer, obs)
            )
        else:
            return random.randint(0, self.numActions - 1)
    
    def epsilon_decay_step(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.1)

    def Train(self, args):
        Best = 0
        result_dict = {
                'own_hp': [],
                'opp_hp': [],
                'hp_diff': [], 
                'avg_loss': [],
                'result': []
            }
        for epoch in range(args.epochs):

            epoch_losses = []
            obs = self.env.reset(env_args=self.env_args)
            reward, done, info = 0, False, None
            total_reward = 0
            steps = 0
            
            while not done:
                act = self.epsilon_greedy(obs)
                new_obs, reward, done, info = self.env.step(act)
                total_reward += reward
                steps += 1

                if not done:
                    # 计算下一个状态的最大Q值
                    max_next_q = np.max(np.dot(self.transformer, new_obs))
                    current_q = np.dot(obs, self.transformer[act])
                    delta = reward + self.gamma * max_next_q - current_q
                    self.transformer[act] += self.alpha * delta * obs
                    loss = delta ** 2
                    epoch_losses.append(loss)
                    obs = new_obs
                    self.epsilon_decay_step()
                    
                elif info is not None:
                    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                    
                    result_dict['own_hp'].append(info[0])
                    result_dict['opp_hp'].append(info[1]) 
                    result_dict['hp_diff'].append(info[0] - info[1])
                    result_dict['avg_loss'].append(avg_loss)
                    result_dict['result'].append(1 if info[0]>info[1] else 0)
                    
                    print(f"round{epoch+1} result: own hp {info[0]} vs opp hp {info[1]}, you {'win' if info[0]>info[1] else 'lose'}, hp_diff: {info[0]-info[1]}, avg_loss: {avg_loss:.4f}, Total reward: {total_reward}")
                    if not os.path.exists(os.path.join(args.log_path, args.method, args.player)):
                        os.makedirs(os.path.join(args.log_path, args.method, args.player))
                    with open(os.path.join(args.log_path, args.method, args.player, "train.log"), "a") as f:
                        f.write(f"round{epoch+1} result: own hp {info[0]} vs opp hp {info[1]}, you {'win' if info[0]>info[1] else 'lose'}, hp_diff: {info[0]-info[1]}, avg_loss: {avg_loss:.4f}, Total reward: {total_reward}\n")
                    if not os.path.exists(os.path.join(args.save_path, args.method, args.player)):
                        os.makedirs(os.path.join(args.save_path, args.method, args.player))
                    if info[0]>info[1]: # 只在获胜时保存模型
                        np.save(
                            os.path.join(args.save_path, args.method, args.player, f"ckpt-{epoch}.npy"), 
                            self.transformer)
                        if info[0] - info[1] > Best:
                            Best = info[0] - info[1]
                            np.save(
                                os.path.join(args.save_path, args.method, args.player, f"best-{epoch}.npy"), 
                                self.transformer)
                    elif epoch % 10 == 0:
                        np.save(
                            os.path.join(args.save_path, args.method, args.player, f"temp-{epoch}.npy"), 
                            self.transformer)
        if not os.path.exists(os.path.join(args.log_path, args.method, args.player)):
            os.makedirs(os.path.join(args.log_path, args.method, args.player))
        with open(os.path.join(args.log_path, args.method, args.player, "result_dict.json"), "w") as f:
            json.dump(result_dict, f)
        print("Finished training")
        print(f"Best hp_diff: {Best}")
        return result_dict
