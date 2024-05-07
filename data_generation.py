'''
    self.total_steps += 1
    self.instance_steps += 1
    action = self.__e_greedy_policy(state)
    if (self.domain != 'discretegrid') and (self.domain != "frozenlake"):
        reward, next_state = self.task.perform_action(action, perturb_params=True, **self.instance_param_set)
    else:
        next_state, reward, done, _ = self.task.step(action)
        if render:
            import time

            print(self.task.t, self.instance_steps)
            self.task.render()

            time.sleep(0.1)
    # print(action)
    # print(next_state)
    ep_reward += reward
    if self.standardize_rewards:
        reward = (reward - self.reward_mean) / self.reward_std
    if self.standardize_states:
        next_state = self.__standardize_state(next_state)
    if self.run_type == "modelfree":
        self.real_buffer.add(np.reshape(np.array([state, self.__encode_action(action), reward, next_state]), [1, 4]))
'''
#from nsbridge_simulator.nsbridge_v0 import NSBridgeV0 as model

import gymnasium as gym
from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
import numpy as np
import pickle
import random
def __encode_action(action):
    """One-hot encodes the integer action supplied."""
    a = np.array([0] * 4)
    a[action] = 1
    return a

task = model()
action = [0, 1, 2, 3]
count_adhering = 0
env = gym.make("FrozenLake-v1", is_slippery=True)
# aug_state1 = np.hstack([state1, __encode_action(action), weight_set1]).reshape((1, -1))
# aug_state2 = np.hstack([state2, __encode_action(action), weight_set1]).reshape((1, -1))
exp_buffer = []
state_list = []
episode_buffer_bnn=[]
exp_list=[]
for s in range(16):
    row, col = task.to_m(s)
    #print(s, row, col)
    letter = task.desc[row, col]
    if (letter != b'H') and (letter != b'G'):  # s is not a Hole
        state_list.append(s)
temp1 = []
temp0 = []
temp2 = []
temp3 = []
#action = [0,1]
print(len(state_list))
for state in state_list:
    for a in action:
        if a in [0,1,2,3]:
            value = 1000
        else:
            value = 1000
        for i in range(value):
            #value = random.randint(1, 100)
            task.set_state(state, 1)
            next_state, reward, done, _ = task.step(a)
            if a == 0 and state == 4:
                temp0.append(task._NSFrozenLakeV0__decode_state(next_state, state, a))
            if a == 1 and state == 4:
                temp1.append(task._NSFrozenLakeV0__decode_state(next_state, state, a))
            if a == 2 and state == 4:
                temp2.append(task._NSFrozenLakeV0__decode_state(next_state, state, a))
            if a == 3 and state == 4:
                temp3.append(task._NSFrozenLakeV0__decode_state(next_state, state, a))
            #print(np.reshape(np.array([task._NSBridgeV0__encode_state(state), __encode_action(a), reward, next_state]), [1, 4]))
            episode_buffer_bnn.append(
                np.reshape(np.array([task._NSFrozenLakeV0__encode_state(state), __encode_action(a), reward, next_state, 0]),
                           [1, 5]))
            #print(task._NSFrozenLakeV0__encode_state(state))

from collections import Counter
print("state list 0", Counter(temp0))
print("state list 1", Counter(temp1))
print("state list 2", Counter(temp2))
print("state list 3", Counter(temp3))
print(len(episode_buffer_bnn))
exp_list = np.reshape(episode_buffer_bnn, [-1,5])
for trans_idx in range(len(exp_list)):
    exp_buffer.append(exp_list[trans_idx])
with open('results/{}_exp_buffer'.format("frozenlake"),'wb') as f:
    pickle.dump(exp_buffer,f)

T=task.generate_transition_matrix_parse(0.9)