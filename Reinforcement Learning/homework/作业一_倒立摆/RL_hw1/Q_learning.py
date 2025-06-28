import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import PillowWriter
import os
from Inverted_pendulum import Inverted_pendulum

matplotlib.use('Agg')  



class Q_learning(Inverted_pendulum):
    def __init__(self,gamma,iterations,lr,epsilon=0.9,distinct_num=200,step_limit=300,error_limit=None):
        super().__init__()
        # hyper parameters
        self.gamma = gamma
        self.iteraions = iterations
        self.lr = lr

        # parameters
        self.action = np.array([-3,0,3])
        self.init_state = np.array([-np.pi,0])

        self.epsilon = epsilon
        self.distinct_num = distinct_num
        
        # Q table
        self.Q_table = np.zeros([len(self.action),distinct_num,distinct_num])

        # train limits
        self.step_limit = step_limit
        if error_limit is None:
            self.error_limit = np.array([0.05, 0.01])
        else:
            self.error_limit = error_limit
        # self.error_limit = np.array([0.05, 0.01])
        # control limits of the system (angle, speed)

        self.train_history = {}

    def get_train_histor(self):
        return self.train_history

    def epsilon_greedy(self,state_index):
        """
        Epsilon greedy policy
        Parameters:
        state (np.array): current state of the system
        Returns:
        int: action index
        """
        act_index = np.argmax(self.Q_table[:, state_index[0], state_index[1]])
        if np.random.rand() > self.epsilon:
            return act_index
        else:
            return np.random.choice([0, 1, 2])
        
    def train(self, decay=0.9995):
        """
        Train the Q-learning agent
        """
        min_epslion = 0.001
        min_lr = 0.01
        optimal_reward = -1e7 
        min_error = 2 * np.pi  

        total_step = 0

        rewards = []
        steps = []
        errors = []

        # record training process
        self.train_history = {
            'reward': [],
            'average_reward': [],
            'accumulated_reward': [],
            'smooth_reward': [],
            'smooth_step': [],
            'angles': [],
            'steps': [],
            'error': [],
            'smooth_error': []
        }

        for i in range(self.iteraions):
            state = self.init_state
            count = 0
            total_reward = 0
            accumulated_reward = 0

            distinct_state = self.action_distinct(state, self.distinct_num)
            self.epsilon = max(min_epslion, self.epsilon * decay)
            self.lr = max(min_lr, self.lr * decay)
            #self.lr = self.lr * decay

            angles = []
            acts = []
            

            angles.append(state[0])
            error = np.abs(state[0])
            opt_error = np.abs(state[0])
            opt_angle = state[0]
            while count < self.step_limit:
                count += 1
                # get the action index
                greddy_act = self.epsilon_greedy(distinct_state)
                next_action = self.action[greddy_act]
                acts.append(next_action)

                reward = self.reward(state, next_action)

                new_state = self.step(state, next_action)
                error = np.abs(new_state[0])

                if (error < self.error_limit[0]) and (np.abs(new_state[1]) < self.error_limit[1]):
                    angles.append(new_state[0])
                    opt_error = error
                    opt_angle = new_state[0]
                    break
                
                new_state_distinct = self.action_distinct(new_state, self.distinct_num)
                # update the Q table
                maxQ = np.max(self.Q_table[:, new_state_distinct[0], new_state_distinct[1]])
                delta = reward + self.gamma * maxQ - self.Q_table[greddy_act][distinct_state[0]][distinct_state[1]]
                self.Q_table[greddy_act][distinct_state[0]][distinct_state[1]] += self.lr * delta

                # update the state
                state = new_state
                distinct_state = new_state_distinct
                angles.append(state[0])
                opt_error = error
                opt_angle = state[0]
                total_reward += reward
                accumulated_reward += reward * self.gamma ** count

            total_step += count
            steps.append(count)
            rewards.append(total_reward)
            errors.append(error)
            
            smooth_reward = np.mean(rewards[-100:])
            smooth_step = np.mean(steps[-100:])
            smooth_error = np.mean(errors[-100:])

            self.train_history['reward'].append(total_reward)
            self.train_history['average_reward'].append(total_reward/count)
            self.train_history['accumulated_reward'].append(accumulated_reward)
            self.train_history['smooth_reward'].append(smooth_reward)
            self.train_history['smooth_step'].append(smooth_step)

            self.train_history['angles'].append(opt_angle)
            self.train_history['steps'].append(count)
            self.train_history['error'].append(opt_error)
            self.train_history['smooth_error'].append(smooth_error)



            if count < self.step_limit:
                # save the optimal policy
                if error <= min_error and total_reward >= optimal_reward:
                    optimal_reward = total_reward
                    min_error = error
                    np.save("optimal_policy.npy", self.Q_table)
                    #print("Optimal policy saved")
                    

            # print reslts every 100 iterations
            if i % 100 == 0:
                print(f"Iteration: {i}, Steps: {count}, Average Reward: {smooth_reward}, Epsilon: {self.epsilon}, Learning Rate: {self.lr}")
                print(f"Angle: {state[0]}, Speed: {state[1]}, Error: {error}\n")
       
    def plot(self, file_path,data,filename, title,ylabel,xlabel = "Episode"):
        """
        Plot the training results
        """
        if  not os.path.exists(file_path):
            os.makedirs(file_path)
        # plot the training history
        plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.plot(np.array(data).reshape(-1))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(file_path+filename)
        plt.cla()
        plt.close()


    def plot_history(self,file_path):
        """
        Plot the training results
        """
        if  not os.path.exists(file_path):
            os.makedirs(file_path)

        # plot the training history
        self.plot(file_path, self.train_history['reward'], "reward.pdf", "reward every episode", "reward")
        self.plot(file_path, self.train_history['average_reward'], "average_reward.pdf", "average reward every episode", "average reward")
        self.plot(file_path, self.train_history['smooth_reward'], "smooth_reward.pdf", "smooth reward every episode", "smooth reward")
        self.plot(file_path, self.train_history['accumulated_reward'], "accumulated_reward.pdf", "accumulated reward every episode", "accumulated reward")
        self.plot(file_path, self.train_history['smooth_step'], "smooth_step.pdf", "smooth step every episode", "smooth step")
        self.plot(file_path, self.train_history['angles'], "angles.pdf", "angles every episode", "angles")
        self.plot(file_path, self.train_history['steps'], "steps.pdf", "steps every episode", "steps")
        self.plot(file_path, self.train_history['error'], "error.pdf", "error every episode", "error")
        self.plot(file_path, self.train_history['smooth_error'], "smooth_error.pdf", "smooth error every episode", "smooth error")


    
    def evaluate(self, table_path=None, max_steps=1000, verbose=True):
        """
        Evaluate the trained Q-learning policy
        
        Parameters:
        table_path (str): Path to the saved Q-table, uses current Q_table if None
        max_steps (int): Maximum steps for each evaluation episode
        verbose (bool): Whether to print detailed evaluation information
        
        Returns:
        tuple: (balanced, steps_to_balance, eval_errors, eval_angles)
        """
        # Load Q-table
        if table_path:
            Q_table = np.load(table_path)
        else:
            Q_table = self.Q_table
            
        # Start from fixed initial position (lowest point)
        state = self.init_state.copy()
        distinct_state = self.action_distinct(state, self.distinct_num)
            
        # Single trial data collection
        eval_errors = []
        eval_angles = []
        actions = []
        speed = []
        balanced = False
        steps_to_balance = max_steps
            
        # Execute single evaluation trial
        for step in range(max_steps):
            # Select action using Q-table (greedy policy)
            act_index = np.argmax(Q_table[:, distinct_state[0], distinct_state[1]])
            next_action = self.action[act_index]
            actions.append(next_action)

                
            # Record current state information
            eval_errors.append(np.abs(state[0]))
            eval_angles.append(state[0])
            speed.append(state[1])
                
            # Execute action and get new state
            new_state = self.step(state, next_action)
                
            # Check if balanced state achieved
            if (np.abs(new_state[0]) < self.error_limit[0] and 
                np.abs(new_state[1]) < self.error_limit[1]):
                # Record success and number of steps
                if not balanced:
                    balanced = True
                    steps_to_balance = step + 1
                    if verbose:
                        print(f"Evaluation success: Balanced at step {step+1}")
                
            # Update state
            state = new_state
            distinct_state = self.action_distinct(state, self.distinct_num)
        
        # If balance not achieved and verbose is enabled
        if not balanced and verbose:
            print(f"Evaluation fail: Failed to balance within {max_steps} steps")
                
        # Print evaluation statistics
        if verbose:
            print("\n========== Evaluation Results ==========")
            print(f"Balance status: {'Success' if balanced else 'Fail'}")
            if balanced:
                print(f"Steps to balance: {steps_to_balance}")
            print(f"Final angle: {state[0]:.2f} rad")
            print(f"Final speed: {state[1]:.2f} rad/s")
            print(f"Final error: {np.abs(state[0]):.2f} rad")
            
        # Generate evaluation plots
        self._plot_evaluation_results(eval_errors, eval_angles, balanced, steps_to_balance)
        
        return balanced, steps_to_balance, actions, speed, eval_errors, eval_angles

    def _plot_evaluation_results(self, errors, angles, balanced, steps_to_balance):
        """
        Plot evaluation result charts
        
        Parameters:
        errors: Error list from evaluation
        angles: Angle list from evaluation
        balanced: Boolean indicating if the pendulum was balanced
        steps_to_balance: Number of steps to balance
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot angle changes
        ax1.plot(angles, 'b-')
        
        # Add balance threshold
        ax1.axhline(y=0, color='g', linestyle='--', alpha=0.5, 
                    label=f'Target position (±{self.error_limit[0]})')
        ax1.axhline(y=self.error_limit[0], color='g', linestyle=':', alpha=0.3)
        ax1.axhline(y=-self.error_limit[0], color='g', linestyle=':', alpha=0.3)
        
        # Set title based on balance status
        balance_status = "Balanced" if balanced else "Not Balanced"
        if balanced:
            ax1.set_title(f'Angle Trajectory ({balance_status}, Steps: {steps_to_balance})')
            # Mark the balance point
            if steps_to_balance < len(angles):
                ax1.plot(steps_to_balance, angles[steps_to_balance], 'ro', 
                         markersize=8, label='Balance point')
        else:
            ax1.set_title(f'Angle Trajectory ({balance_status})')
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Angle (rad)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot error changes (log scale)
        ax2.plot(errors, 'g-')
        ax2.axhline(y=self.error_limit[0], color='r', linestyle='--', alpha=0.5, 
                    label=f'Error Threshold ({self.error_limit[0]})')
        ax2.set_title('Error Trajectory')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Error |θ| (rad)')
        ax2.set_yscale('log')  # Log scale
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("evaluation_results.pdf")
        plt.close()

    def create_animation(self, angles, actions, speed, save_path=None, fps=30, sample_rate=1):
        """
        Create an animation of the inverted pendulum based on evaluation results
        
        Parameters:
        angles: List of pendulum angles over time
        actions: List of actions taken by the agent
        speed: List of angular velocities of the pendulum
        save_path: Path to save the animation (default: 'pendulum_animation.gif')
        fps: Frames per second for the animation
        duration: Maximum duration in seconds (will truncate angles if needed)
        
        Returns:
        None: Saves the animation to the specified path
        """
        # Calculate pendulum coordinates for each frame
        pendulum_length = 1.0  # Visual length of the pendulum
        
        # 忽略duration参数，直接根据采样率处理所有帧
        max_frames = len(angles)
    
        # 根据采样率对数据进行采样
        sampled_angles = angles[:max_frames:sample_rate]
        sampled_actions = actions[:max_frames:sample_rate]
        sampled_speed = speed[:max_frames:sample_rate]
    
        # 保持实际输出帧率与请求的帧率一致
        effective_fps = fps
    
        print(f"Total evaluation steps: {len(angles)}")
        print(f"After sampling (1:{sample_rate}): {len(sampled_angles)} frames")
        print(f"Animation will be played at {fps} fps")
        print(f"Animation duration will be {len(sampled_angles)/fps:.1f} seconds")
        
        # Create figure and axis with improved background - increased figure size
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Inverted Pendulum Simulation', fontsize=16)
        ax.set_facecolor('#f5f5f5')  # Light gray background

        # Add reference lines for start position (-π) and balance position (0)
        start_x = pendulum_length * np.sin(-np.pi)
        start_y = pendulum_length * np.cos(-np.pi)
        balance_x = pendulum_length * np.sin(0)
        balance_y = pendulum_length * np.cos(0)

        # Draw reference lines and add labels
        ax.plot([0, start_x], [0, start_y], 'b--', alpha=0.4, label='Start Position (-π)')
        ax.plot([0, balance_x], [0, balance_y], 'g--', alpha=0.4, label='Balance Position (0)')
        ax.legend(loc='lower right', framealpha=0.8)

        # Add labels for the pendulum positions
        ax.text(start_x, start_y, "-π", fontsize=12, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='blue'))
        ax.text(balance_x, balance_y, "0", fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='green'))
        
        # Create pendulum objects with improved aesthetics
        line, = ax.plot([], [], 'k-', lw=3)  # Thicker pendulum rod
        mass = Circle((0, 0), 0.1, fc='#d13636', ec='k', zorder=5)  # Red bob with black edge
        base = Rectangle((-0.1, -0.05), 0.2, 0.05, fc='#333333', zorder=4)  # Dark base
        ax.add_patch(mass)
        ax.add_patch(base)

        # Add trajectory line to show the pendulum's path
        trajectory, = ax.plot([], [], '#ff7f0e', alpha=0.4, lw=1.5)  # Orangish trajectory
        traj_x, traj_y = [], []

        # Simplified info panel with smaller size
        info_panel = Rectangle((-1.4, 0.95), 1.0, 0.5, fc='#e0f0ff', ec='#4c72b0', alpha=0.9)
        ax.add_patch(info_panel)
        
        # Text displays for real-time information with better formatting
        step_text = ax.text(-1.35, 1.35, '', fontsize=11, fontweight='bold')
        angle_text = ax.text(-1.35, 1.25, '', fontsize=11)
        speed_text = ax.text(-1.35, 1.15, '', fontsize=11)
        action_text = ax.text(-1.35, 1.05, '', fontsize=11)
        # Balance status integrated into the info panel
        balance_text = ax.text(-1.35, 0.95, '', fontsize=11, fontweight='bold')
        
        # Initialization function
        def init():
            line.set_data([], [])
            mass.center = (0, 0)
            trajectory.set_data([], [])
            step_text.set_text('')
            angle_text.set_text('')
            speed_text.set_text('')
            action_text.set_text('')
            balance_text.set_text('')
            return line, mass, trajectory, step_text, angle_text, speed_text, action_text, balance_text
        
        # Animation function
        def animate(i):
            angle = sampled_angles[i]
            action = sampled_actions[i] if i < len(sampled_actions) else 0
            angular_speed = sampled_speed[i] if i < len(sampled_speed) else 0
            
            x = pendulum_length * np.sin(angle)
            y = pendulum_length * np.cos(angle)
            
            # Update pendulum position
            line.set_data([0, x], [0, y])
            mass.center = (x, y)
            
            # Update trajectory
            traj_x.append(x)
            traj_y.append(y)
            trajectory.set_data(traj_x, traj_y)
            
            # Update text displays with real-time information
            step_text.set_text(f'Step: {i}')
            angle_text.set_text(f'Angle: {angle:.2f} rad')
            speed_text.set_text(f'Angular Speed: {angular_speed:.2f} rad/s')
            action_text.set_text(f'Action: {action:.1f}')
            
            # Update balance status text (no color changes)
            if hasattr(self, 'error_limit') and abs(angle) < self.error_limit[0] and abs(angular_speed) < self.error_limit[1]:
                balance_text.set_text('Status: Balanced')
            else:
                balance_text.set_text('Status: Not Balanced')
            
            return line, mass, trajectory, step_text, angle_text, speed_text, action_text, balance_text
        
        # Create the animation
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(sampled_angles), interval=1000/fps, blit=True
        )       

        # Save the animation
        if save_path is None:
            save_path = 'pendulum_animation.gif'
            
        print(f"Creating animation with {len(sampled_angles)} frames...")
    
        # 使用请求的帧率(fps)而不是effective_fps
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
    
        print(f"Animation saved to {save_path}")
        # Close the figure to free memory
        plt.close(fig)
        

    

if __name__ == "__main__":
    # Create Q-learning instance
    q_learning = Q_learning(gamma=0.98, iterations=15000, lr=0.5, epsilon=0.8, 
                           distinct_num=200, step_limit=2000, error_limit=[0.01, 0.1])
    # print("start training")
    # q_learning.train(decay=0.9995)
    # Save training history
    # q_learning.plot_history(file_path="training_results/")
    
    # Evaluate
    # performance
    print("start evaluation")
    balanced, steps_to_balance, actions, speed, errors, angles = q_learning.evaluate(
        table_path="optimal_policy.npy",
        max_steps=2000,
        verbose=True
    )
    
    # 创建完整动画，不限制时长
    q_learning.create_animation(
        angles=angles,  
        actions=actions,  
        speed=speed,
        save_path="pendulum_animation_complete.gif",
        fps=30, # 不限制时长，录制所有帧
        sample_rate=4    # 每4帧取样一次，加快生成速度
    )
    