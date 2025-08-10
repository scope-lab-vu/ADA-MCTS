import gymnasium as gym
import numpy as np
import pickle
import random


class RLExperienceCollector:
    def __init__(self, env_name, model_class, transition_number=1000):
        self.env = gym.make(env_name)
        self.task = model_class(intended_prob=0.4)
        self.transition_number = transition_number
        self.exp_buffer = []
        self.state_list = []
        self.episode_buffer_bnn = []
        self.model_class_name = model_class.__name__

    def __encode_action(self, action):
        """One-hot encodes the integer action supplied."""
        a = np.zeros(self.env.action_space.n, dtype=int)
        a[action] = 1
        return a

    def __encode_state(self, state):
        """Encodes the state in a generic way. This method can be overridden by the model class if needed."""
        encode_method = f"_{self.model_class_name}__encode_state"
        if hasattr(self.task, encode_method):
            return getattr(self.task, encode_method)(state)
        return state

    def collect_experiences(self):
        action_space = list(range(self.env.action_space.n))
        # Collect valid states that are not terminal
        for state in range(self.env.observation_space.n):
            row, col = self.task.to_m(state)
            letter = self.task.desc[row, col]
            if (letter != b'H') and (letter != b'G'):  # state is not a Hole or Goal (Frozen Lake specific)
                self.state_list.append(state)

        # Collect experiences
        for state in self.state_list:
            for action in action_space:
                for _ in range(self.transition_number):
                    self.task.set_state(state, 1)
                    next_state, reward, done, _ = self.task.step(action)
                    self.episode_buffer_bnn.append(
                        np.reshape(np.array([self.__encode_state(state), self.__encode_action(action),
                                             reward, next_state, 0]), [1, 5]))

        exp_list = np.reshape(self.episode_buffer_bnn, [-1, 5])
        self.exp_buffer.extend(exp_list)


    def save_experiences(self, filename='data_buffer/exp_buffer.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.exp_buffer, f)


# Usage Example
if __name__ == "__main__":
    from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0
    # Alternatively, import your custom model class
    # from nsbridge_simulator.nsbridge_v0 import NSBridgeV0

    collector = RLExperienceCollector(env_name="FrozenLake-v1", model_class=NSFrozenLakeV0)
    collector.collect_experiences()
    collector.save_experiences('data_buffer/frozenlake_exp_buffer.pkl')