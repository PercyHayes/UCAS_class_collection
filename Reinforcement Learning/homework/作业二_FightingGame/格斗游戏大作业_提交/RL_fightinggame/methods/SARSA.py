import numpy as np
import os
import random
import json
import time
class SARSA():

    def __init__(
        self, 
        environment,
        environment_args,
        numObs = 144, 
        numAct = 40, 
        gamma = 0.95,
        alpha = 0.01,
        epsilon_init = 1., 
        epsilon_limit=0.1,
        epsilon_decay=0.9995, 
    ):
        super(SARSA, self).__init__()

        self.env = environment
        self.env_args = environment_args

        self.numObs = numObs
        self.numAct = numAct

        self.gamma = gamma
        self.alpha = alpha

        self.eps = epsilon_init
        self.eps_lim = epsilon_limit
        self.eps_decay = epsilon_decay

        self.Q = np.zeros((self.numAct, self.numObs))  # Q(s)[a]

    def get_eps_greedy_action(self, obs):
        if np.random.random() > self.eps:
            return np.argmax(np.dot(self.Q, obs))       
        else:
            return random.randint(0, self.numAct - 1)   

    def Train(self, args):
        Best = 0
        result_dict = {
                'own_hp': [],
                'opp_hp': [],
                'hp_diff': [],
                'total_reward': [],
                'result': []
            }
        for epoch in range(args.epochs):
            start_time = time.time()
            obs = self.env.reset(env_args=self.env_args)
            reward, done, info = 0, False, None
            total_reward = 0
            steps = 0

            while not done:
                act = self.get_eps_greedy_action(obs)
                # TODO: or you can design with your RL algorithm to choose action [act] according to game obs [obs]
                new_obs, reward, done, info = self.env.step(act)
                total_reward += reward
                steps += 1

                if not done:
                    # TODO: (main part) learn with data (obs, act, reward, new_obs)
                    new_act = self.get_eps_greedy_action(new_obs)

                    delta = reward + self.gamma * np.dot(self.Q[new_act], new_obs) - np.dot(self.Q[act], obs)
                    self.Q[act] += self.alpha * delta * obs

                    obs = new_obs
                    self.eps = max(self.eps * self.eps_decay, self.eps_lim)
                    
                elif info is not None:
                    epoch_time = time.time() - start_time
                    result_dict['own_hp'].append(info[0])
                    result_dict['opp_hp'].append(info[1]) 
                    result_dict['hp_diff'].append(info[0] - info[1])
                    result_dict['total_reward'].append(total_reward)
                    result_dict['result'].append(1 if info[0]>info[1] else 0)

                    print(f"round{epoch+1} result: own hp {info[0]} vs opp hp {info[1]}, you {'win' if info[0]>info[1] else 'lose'}, hp_diff: {info[0]-info[1]}, Total reward: {total_reward}, Time: {epoch_time:.2f}s")
                    if not os.path.exists(os.path.join(args.log_path, args.method, args.player)):
                        os.makedirs(os.path.join(args.log_path, args.method, args.player))
                    with open(os.path.join(args.log_path, args.method, args.player, "train.log"), "a") as f:
                        f.write(f"round{epoch+1} result: own hp {info[0]} vs opp hp {info[1]}, you {'win' if info[0]>info[1] else 'lose'}, hp_diff: {info[0]-info[1]}, Total reward: {total_reward}\n")
                    if not os.path.exists(os.path.join(args.save_path, args.method, args.player)):
                        os.makedirs(os.path.join(args.save_path, args.method, args.player))
                    if info[0]>info[1]: # 只在获胜时保存模型
                        np.save(os.path.join(args.save_path, args.method, args.player, f"ckpt-{epoch}.npy"), self.Q)
                        if info[0] - info[1] > Best:
                            Best = info[0] - info[1]
                            np.save(os.path.join(args.save_path, args.method, args.player, f"best-{epoch}.npy"), self.Q)
                    elif epoch % 10 == 0:
                        np.save(os.path.join(args.save_path, args.method, args.player, f"temp-{epoch}.npy"), self.Q)
        if not os.path.exists(os.path.join(args.log_path, args.method, args.player)):
            os.makedirs(os.path.join(args.log_path, args.method, args.player))
        with open(os.path.join(args.log_path, args.method, args.player, "result_dict.json"), "w") as f:
            json.dump(result_dict, f)
        print("Finished training")
        print(f"Best hp_diff: {Best}")
        return result_dict
