import numpy as np


class Inverted_pendulum():
    def __init__(self):
        
        # system parameters
        self.m = 0.055 # mass of the pendulum
        self.g = 9.81 # gravity
        self.l = 0.042 # length of the pendulum
        self.J = 1.91e-4 # moment of inertia
        self.b = 3e-6 # Viscous damping
        self.K = 0.0536 # motor constant
        self.R = 9.5 # resistance

        # simulation parameters
        self.dt = 0.005 # time step
        self.dim = 2 # state dimension
        self.max_angle = np.pi # maximum angle
        self.min_angle = -np.pi # minimum angle
        self.max_speed = 15 * np.pi # maximum speed
        self.min_speed = -15 * np.pi # minimum speed
        self.max_voltage = 3 # maximum voltage

    def angular_acceleration(self, theta, theta_dot, u):
        """
        Calculate the angular acceleration of the pendulum.
        Parameters:
        theta (float): angle of the pendulum
        theta_dot (float): angular velocity of the pendulum
        u (float): control input (voltage)
        Returns:
        float: angular acceleration of the pendulum
        """
        tmp = self.m * self.g * self.l * np.sin(theta) - self.b * theta_dot - (self.K ** 2)/self.R * theta_dot
        acceleration = (tmp + self.K/self.R * u) / self.J
        return acceleration
    
    def step(self, state, u):
        """
        Simulate one step of the system.
        Parameters:
        state (np.array): current state of the system
        u (float): control input (voltage)
        Returns:
        np.array: next state of the system
        """
        theta = state[0]
        theta_dot = state[1]

        # calculate angular acceleration
        alpha = self.angular_acceleration(theta, theta_dot, u)

        # update state using Euler's method
        theta_new = theta + theta_dot * self.dt
        theta_dot_new = theta_dot + alpha * self.dt

        # clip the angle and speed to the limits
        theta_dot_new = np.clip(theta_dot_new, self.min_speed, self.max_speed)

        # normalize the angle to be within [-pi, pi]
        theta_new = ((theta_new + np.pi) % (2 * np.pi)) - np.pi
        '''if theta_new < -self.max_angle:
            theta_new += np.pi  
        elif theta_new >= self.max_angle:
            theta_new -= np.pi  '''
    
    
        return np.array([theta_new, theta_dot_new])

    def reward(self, state, u):
        """
        Calculate the reward for the current state.
        Parameters:
        state (np.array): current state of the system
        Returns:
        float: reward for the current state
        """
        Q_rew = np.matrix([[5, 0], [0, 0.1]])
        R_rew = 1.
        state_matrix = np.matrix(state)
        # calculate the reward
        # print("state_matrix", state_matrix.shape)
        r = - np.dot(state_matrix, np.dot(Q_rew, state_matrix.T)) - R_rew * (u ** 2)
        return r.item()
    
    def action_distinct(self, state, num=200):
        """
        Sample the action space.
        Parameters:
        state (np.array): current state of the system
        num (int): number of discrete actions
        Returns:
        np.array: sampled action
        """
        theta = state[0]
        theta_dot = state[1]
        
        # normalize the angle and speed
        normalized_theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        
        # calculate the indices for the discrete action space
        indices_theta = int((normalized_theta + np.pi) / (2 * np.pi) * num)
        indices_theta_dot = int((theta_dot + self.max_speed) / (2 * self.max_speed) * num)
        
        # correct the indices to be within the range [0, num-1]
        indices_theta = min(max(0, indices_theta), num-1)
        indices_theta_dot = min(max(0, indices_theta_dot), num-1)
        
        return np.array([indices_theta, indices_theta_dot])
    

if __name__ == "__main__":
    # create an instance of the Inverted_pendulum class
    inverted_pendulum = Inverted_pendulum()

    # initial state
    state = np.array([0.1,1.])

    # control input (voltage)
    u = 3

    # simulate one step
    next_state = inverted_pendulum.step(state, u)

    # calculate the reward
    reward = inverted_pendulum.reward(state, u)

    # sample an action
    action = inverted_pendulum.action_distinct(state, num=200) 

    print("Next state:", next_state)
    print("Reward:", reward)
    print("Action:", action)

