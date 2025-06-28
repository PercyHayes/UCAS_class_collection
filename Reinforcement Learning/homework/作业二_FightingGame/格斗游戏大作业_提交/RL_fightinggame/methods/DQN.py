"""
Created for RL homework
"""

import numpy as np
import os
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class QNetwork(nn.Module):
    # Q网络，三层MLP
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQN():
    def __init__(self,
        args,
        environment, 
        environment_args,
        numStates = 144, 
        numActions = 40, 
        gamma = 0.95,
        epsilon = 1.0,
        epsilon_min = 0.1,
        epsilon_decay = 0.995,
        learning_rate = 0.001,
        batch_size = 32,
        memory_size = 10000,
        update_target_freq = 100):
        super(DQN, self).__init__()
        self.args = args
        self.env = environment
        self.env_args = environment_args
        
        self.numStates = numStates
        self.numActions = numActions
        
        # 超参
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.update_target_freq = update_target_freq
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
        self.qnetwork_local = QNetwork(numStates, numActions).to(self.device)
        self.qnetwork_target = QNetwork(numStates, numActions).to(self.device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        self.memory = deque(maxlen=memory_size) # 经验
        
        self.t_step = 0
    
    def epsilon_greedy(self, obs):
        if np.random.random() > self.epsilon: #探索
            state = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.randint(0, self.numActions - 1)
    
    def epsilon_decay_step(self): # 衰减，有下限
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def remember(self, state, action, reward, next_state, done):
        # 确保state和next_state不是None
        if state is None:
            print("Warning: state is None, skipping memory storage")
            return
        
        if next_state is None:
            print("Warning: next_state is None, using zero state instead")
            next_state = np.zeros_like(state)
        
        # 确保state和next_state形状一致
        if hasattr(state, 'shape') and hasattr(next_state, 'shape'):
            if state.shape != next_state.shape:
                print(f"Shape mismatch in remember: state {state.shape}, next_state {next_state.shape}")
                # 如果next_state形状不对，用零状态替代
                if next_state.shape != state.shape:
                    next_state = np.zeros_like(state)
        
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self):
        # 如果经验池中的样本不足，不进行学习
        if len(self.memory) < self.batch_size:
            return 0
        
        #随机采样进行学习
        minibatch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        # 检查并处理形状不一致的问题
        valid_indices = []
        expected_shape = states[0].shape
        
        for i, (state, next_state) in enumerate(zip(states, next_states)):
            # 先检查是否为None，再检查形状
            if (state is not None and next_state is not None and 
                hasattr(state, 'shape') and hasattr(next_state, 'shape') and
                state.shape == expected_shape and next_state.shape == expected_shape):
                valid_indices.append(i)
            else:
                if state is None:
                    print(f"State is None at index {i}")
                elif next_state is None:
                    print(f"Next_state is None at index {i}")
                else:
                    print(f"Shape mismatch at index {i}: state {getattr(state, 'shape', 'None')}, next_state {getattr(next_state, 'shape', 'None')}")
        
        # 如果有效样本太少，跳过这次学习
        if len(valid_indices) < self.batch_size // 2:
            print(f"Too few valid samples: {len(valid_indices)}/{self.batch_size}")
            return 0
        
        # 只使用形状一致的样本
        if len(valid_indices) < len(minibatch):
            minibatch = [minibatch[i] for i in valid_indices[:self.batch_size]]
            states, actions, rewards, next_states, dones = zip(*minibatch)
        
        try:
            states = torch.tensor(np.vstack(states), dtype=torch.float, device=self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
            next_states = torch.tensor(np.vstack(next_states), dtype=torch.float, device=self.device)
            dones = torch.tensor(dones, dtype=torch.uint8, device=self.device)
        except Exception as e:
            print(f"Error creating tensors: {e}")
            return 0
            
        # 获取当前Q值
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 获取目标Q值
        with torch.no_grad():
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0]
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.t_step = (self.t_step + 1) % self.update_target_freq
        if self.t_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            
        return loss.item()
    
    def Train(self, args):
        Best = 0
        result_dict = {
            'own_hp': [],
            'opp_hp': [],
            'hp_diff': [], 
            'avg_loss': [],
            'result': []
        } #保存log
        
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
                
                # 检查new_obs是否为None
                if new_obs is None:
                    print(f"Warning: new_obs is None at step {steps} of epoch {epoch+1}")
                    new_obs = np.zeros_like(obs)  # 使用零状态作为替代
                
                if not done:
                    # 存储经验
                    self.remember(obs, act, reward, new_obs, False)
                    
                    # 学习
                    loss = self.learn()
                    if loss > 0:
                        epoch_losses.append(loss)
                    
                    obs = new_obs
                    self.epsilon_decay_step()
                    
                elif info is not None:
                    # 添加最后一步的经验
                    self.remember(obs, act, reward, new_obs, True)
                    
                    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                    
                    result_dict['own_hp'].append(info[0])
                    result_dict['opp_hp'].append(info[1]) 
                    result_dict['hp_diff'].append(info[0] - info[1])
                    result_dict['avg_loss'].append(avg_loss)
                    result_dict['result'].append(1 if info[0]>info[1] else 0)
                    
                    print(f"round{epoch+1} result: own hp {info[0]} vs opp hp {info[1]}, you {'win' if info[0]>info[1] else 'lose'}, hp_diff: {info[0]-info[1]}, avg_loss: {avg_loss:.4f}, Total reward: {total_reward}")
                    
                    # 保存日志
                    if not os.path.exists(os.path.join(args.log_path, args.method, args.player)):
                        os.makedirs(os.path.join(args.log_path, args.method, args.player))
                    with open(os.path.join(args.log_path, args.method, args.player, "train.log"), "a") as f:
                        f.write(f"round{epoch+1} result: own hp {info[0]} vs opp hp {info[1]}, you {'win' if info[0]>info[1] else 'lose'}, hp_diff: {info[0]-info[1]}, avg_loss: {avg_loss:.4f}, Total reward: {total_reward}\n")
                    
                    # 保存模型
                    if not os.path.exists(os.path.join(args.save_path, args.method, args.player)):
                        os.makedirs(os.path.join(args.save_path, args.method, args.player))
                    if info[0]>info[1]: # 只在获胜时保存模型
                        torch.save(self.qnetwork_local.state_dict(), 
                                os.path.join(args.save_path, args.method, args.player, f"ckpt-{epoch}.pth"))
                        if info[0] - info[1] > Best:
                            Best = info[0] - info[1]
                            torch.save(self.qnetwork_local.state_dict(), 
                                    os.path.join(args.save_path, args.method, args.player, f"best-{epoch}.pth"))
                    elif epoch % 10 == 0:
                        torch.save(self.qnetwork_local.state_dict(), 
                                os.path.join(args.save_path, args.method, args.player, f"temp-{epoch}.pth"))
        
        # 保存训练结果
        if not os.path.exists(os.path.join(args.log_path, args.method, args.player)):
            os.makedirs(os.path.join(args.log_path, args.method, args.player))
        with open(os.path.join(args.log_path, args.method, args.player, "result_dict.json"), "w") as f:
            json.dump(result_dict, f)
        
        print("Finished training")
        print(f"Best hp_diff: {Best}")
        return result_dict