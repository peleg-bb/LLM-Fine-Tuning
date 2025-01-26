import numpy as np

class Model:
    def __init__(self, num_states, num_actions, transition_probs=None, rewards=None):
        # Store environment dimensions
        self.num_states = num_states
        self.num_actions = num_actions

        # Initialize transition dynamics
        if transition_probs is None:
            self.transition_probs = np.zeros((num_states, num_actions, num_states))
        else:
            self.transition_probs = transition_probs

        # Initialize reward structure
        if rewards is None:
            self.rewards = np.zeros((num_states, num_actions))
        else:
            self.rewards = rewards

    def print_model(self):
        # Display model characteristics
        print(f"State space size: {self.num_states}")
        print(f"Action space size: {self.num_actions}")
        
        # Display non-zero transitions
        print("Non-zero transition probabilities:")
        for initial_state in range(self.num_states):
            for action in range(self.num_actions):
                for target_state in range(self.num_states):
                    prob = self.transition_probs[initial_state, action, target_state]
                    if prob != 0:
                        print(f"T[{initial_state},{action},{target_state}] = {prob}")
        
        # Display non-zero rewards
        print("Non-zero rewards:")
        for initial_state in range(self.num_states):
            for action in range(self.num_actions):
                reward = self.rewards[initial_state, action]
                if reward != 0:
                    print(f"R[{initial_state},{action}] = {reward}")