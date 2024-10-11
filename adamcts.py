import math
import random
#from nsbridge_simulator.nsbridge_v0 import NSBridgeV0 as model
from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
import pickle
from BNN.BayesianNeuralNetwork import *
import autograd.numpy as np
import utils.distribution as distribution
from collections import Counter
import matplotlib.pyplot as plt
import time
from collections import Counter
from multiprocessing import Pool
from HiPMDP import HiPMDP
import logging


class Node:
    bnn_cache = {}
    basemodel_aleatoric = []
    newmodel_aleatoric = []
    basemodel_epistemic = []
    newmodel_epistemic = []
    def __init__(self, state, time, task, danger, action=None, parent=None, node_type="decision"):
        self.state = state
        self.action = action  # Action taken or outcome for chance node
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.type = node_type  # "decision" or "chance"
        self.probabilities = [] if self.type == "chance" else None
        self.possible_actions = [0, 1, 2, 3]
        self.discount_factor = 0.9999
        self.time = time
        self.np_random = np.random.RandomState()
        self.task = task
        self.danger = danger

    def get_transition(self,bnn_samples, BNN1, state, action):
        state_count = []
        samples = bnn_samples[:, 0, :]
        for j in samples.reshape(-1, 2):
            state_count.append(BNN1.task._NSFrozenLakeV0__decode_state(j, state, action))
        state_counts = Counter(state_count)  # Assuming state_count is defined somewhere in your code

        # Calculate total transitions
        total_transitions = sum(state_counts.values())

        # Initialize a list of zeros for all states
        bnn_transitions1 = [0.0] * 16

        # Fill in the transition probabilities for the states in state_counts
        for s, c in state_counts.items():
            bnn_transitions1[s] = c / total_transitions
        return bnn_transitions1

    def get_bnn_prediction(self, state, action, BNN1, weight_set1, BNN2, weight_set2, isi):
        # Check cache before BNN prediction
        cache_key = (state, action)
        if cache_key in Node.bnn_cache:
            return Node.bnn_cache[cache_key]
        # If not in cache, query the BNNs
        state1 = BNN1.task._NSFrozenLakeV0__encode_state(state)
        confidence_check = True
        EU1 = []
        EU2 = []
        aug_state1 = np.hstack([state1, self.__encode_action(action), weight_set1]).reshape((1, -1))
        aug_state2 = np.hstack([state1, self.__encode_action(action), weight_set2]).reshape((1, -1))
        _, samples1, epistemic_uncertainties1, aleatoric_uncertainties1 = BNN1.network.feed_forward_distribution(
            aug_state1)
        EU1.append(np.sum(epistemic_uncertainties1))
        _, samples2, epistemic_uncertainties2, aleatoric_uncertainties2 = BNN2.network.feed_forward_distribution(
            aug_state2)
        EU2.append(np.sum(epistemic_uncertainties2))
        transition1 = self.get_transition(samples1, BNN1, state, action)
        transition2 = self.get_transition(samples2, BNN1, state, action)
        Node.basemodel_epistemic.append(np.mean(EU1))
        Node.newmodel_epistemic.append(np.mean(EU2))
        pessimistic1 = self.pessimistic_sample(transition1, state, action, isi)
        # Store the results in the cache

        Node.bnn_cache[cache_key] = (
            transition1, samples1, pessimistic1, np.mean(EU1), np.mean(Node.basemodel_aleatoric), transition2, samples2, pessimistic1,
            np.mean(EU2), np.mean(Node.newmodel_aleatoric), confidence_check)
        return transition1, samples1, pessimistic1, np.mean(EU1), np.mean(Node.basemodel_aleatoric), transition2, samples2, pessimistic1, np.mean(EU2), np.mean(Node.newmodel_aleatoric), confidence_check
    def is_decision_node(self):
        return self.type == "decision"

    def is_chance_node(self):
        return self.type == "chance"

    def get_possible_actions(self):
        return self.possible_actions

    def get_child_with_state(self, state):
        for child in self.children:
            if np.array_equal(child.state, state):  # or any appropriate equality check for states
                return child
        return None

    def pessimistic_sample(self, w0, current_state, curren_action, isi):
        v0 = np.zeros(len(w0))
        for i, j in enumerate(w0):
            if j > 0:
                v0[i] = self.task.instant_reward_byindex(i)
        if np.all(v0 >= 0):
            return w0
        else:
            count = 0
            w_worst = np.zeros(len(w0))
            w_worst[np.argmin(v0)] = 1.0
            if not self.danger:
                return w_worst
            c = 1
            # Using numpy to find indices where rs is 1
            rs = np.array(self.task.reachable_states(current_state, curren_action))
            reachable_states = np.where(rs == 1)[0]
            d = self.task.distances_matrix(reachable_states.tolist())


            # Convert w0 and w_worst to numpy arrays before slicing
            w0_np = np.array(w0)
            w_worst_np = np.array(w_worst)
            w0_dis = w0_np[reachable_states]
            w_worst_dis = w_worst_np[reachable_states]

            wass_value = distribution.wass_dual(w0_dis, w_worst_dis, d)
            if wass_value <= c:
                return w_worst
            lbd = c / wass_value
            w = (1.0 - lbd) * w0_dis + lbd * w_worst_dis
            # Using numpy advanced indexing to assign values
            new_w = np.zeros(len(w0))
            new_w[reachable_states] = w
        return new_w.tolist()


    def is_terminal(self, bnn1, state):
        reward = self.task.instant_reward_byindex(state)
        return reward == 1 or reward == -1

    def expand(self, BNN1, weight_set1, BNN2, weight_set2, threshold, training_started, isi):
        if self.is_terminal(BNN1, self.state):
            return self
        if self.is_decision_node():
            for action in self.get_possible_actions():
                # Create chance nodes for each possible action
                child_node = Node(self.state, self.time, self.task, self.danger, action=action, parent=self, node_type="chance")
                self.children.append(child_node)
        else:  # For a chance node
            transition1, samples1, pessimistic1, epistemic_uncertainties1, aleatoric_uncertainties1, transition2, samples2, pessimistic2, epistemic_uncertainties2, aleatoric_uncertainties2, confidence_check = self.get_bnn_prediction(self.state, self.action, BNN1, weight_set1, BNN2, weight_set2, isi)
            if (np.sum(epistemic_uncertainties1) + threshold < np.sum(epistemic_uncertainties2)) or (not training_started):
                child_state = self.categorical_sample(pessimistic1, self.np_random)
            else:
                if aleatoric_uncertainties1 < aleatoric_uncertainties2:
                    gamma = 10000
                    aleatoric_difference = aleatoric_uncertainties2 - aleatoric_uncertainties1
                    likelihood = math.exp(-gamma * aleatoric_difference)
                    if aleatoric_difference < 1:
                        sampled_value = np.random.uniform(0, 1)
                        if sampled_value < likelihood:
                            child_state = self.categorical_sample(transition2, self.np_random)
                        else:
                            child_state = self.categorical_sample(pessimistic2, self.np_random)
                    else:
                        child_state = self.categorical_sample(pessimistic2, self.np_random)
                else:
                    child_state = self.categorical_sample(transition2, self.np_random)

            existing_child = self.get_child_with_state(child_state)
            if existing_child:
                child_node = existing_child
            else:
                child_node = Node(child_state, self.time, self.task, self.danger, parent=self, node_type="decision")
                self.children.append(child_node)
        return child_node

    def categorical_sample(self, prob_n, np_random):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        return (csprob_n > np_random.rand()).argmax()

    """ Simulate until a terminal state """

    def rollout(self, BNN1, weight_set1, BNN2, weight_set2, threshold, training_started, isi):
        current_state = self.state
        cumulative_reward = 0.0
        depth = 0
        done = False
        visited_pairs = set()
        while not done:
            #print("current state:", current_state)
            if self.is_terminal(BNN1, current_state):
                cumulative_reward += self.task.instant_reward_byindex(current_state)
                break
            action = random.choice(self.possible_actions)

            transition1, samples1, pessimistic1, epistemic_uncertainties1, aleatoric_uncertainties1, transition2, samples2, pessimistic2, epistemic_uncertainties2,aleatoric_uncertainties2, confidence_check = self.get_bnn_prediction(
                current_state, action, BNN1, weight_set1, BNN2, weight_set2, isi)

            if (np.sum(epistemic_uncertainties1) + threshold < np.sum(epistemic_uncertainties2)) or (not training_started):
                child_state = self.categorical_sample(pessimistic1, self.np_random)
            else:
                if aleatoric_uncertainties1 < aleatoric_uncertainties2:
                    gamma = 10000
                    aleatoric_difference = aleatoric_uncertainties2 - aleatoric_uncertainties1
                    likelihood = math.exp(-gamma * aleatoric_difference)
                    if aleatoric_difference < 1:
                        sampled_value = np.random.uniform(0, 1)
                        if sampled_value < likelihood:
                            child_state = self.categorical_sample(transition2, self.np_random)
                        else:
                            child_state = self.categorical_sample(pessimistic2, self.np_random)
                    else:
                        child_state = self.categorical_sample(pessimistic2, self.np_random)
                else:
                    child_state = self.categorical_sample(transition2, self.np_random)
            reward = self.task.instant_reward_byindex(child_state)
            cumulative_reward += pow(self.discount_factor, depth) * reward
            if reward == 1 or reward == -1:
                done = True
            depth += 1
            current_state = child_state
        return cumulative_reward

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            if self.is_decision_node:
                result = result * self.discount_factor
            self.parent.backpropagate(result)

    def uct_value(self, parent_visits, exploration_constant=math.sqrt(2)):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)

    def best_child(self, exploration_constant=math.sqrt(2)):
        return max(self.children, key=lambda c: c.uct_value(self.visits, exploration_constant))

    @staticmethod
    def __encode_action(action):
        a = np.array([0] * 4)
        a[action] = 1
        return a

class MCTS:
    def __init__(self, initial_state_coordinate, initial_state_index, bnn1, bnn2, ws1, ws2, time, task, threshold, training_started, danger):
        self.root = Node(initial_state_index, time, task, danger)
        self.task = task
        self.BNN1 = bnn1
        self.BNN2 = bnn2
        self.weight_set1 = ws1
        self.weight_set2 = ws2
        self.time = time
        self.threshold = threshold
        self.training_started = training_started
        self.isi = initial_state_index
        Node.bnn_cache = {}
        #if len(Node.basemodel_aleatoric) > 100:
        #    Node.basemodel_aleatoric = Node.basemodel_aleatoric[-100:]
        if len(Node.basemodel_epistemic) > 100:
            Node.basemodel_epistemic = Node.basemodel_epistemic[-100:]
        Node.newmodel_epistemic = []

    def is_terminal(self, state):
        reward = self.task.instant_reward_byindex(state)
        return reward == 1 or reward == -1

    def search(self, iterations):
        for _ in range(iterations):
            leaf = self.traverse(self.root)  # Traverse till you reach a leaf
            expanded_node = leaf.expand(self.BNN1, self.weight_set1, self.BNN2, self.weight_set2, self.threshold, self.training_started, self.isi)
            if expanded_node.is_chance_node():
                # If it's a chance node, expand again to get a decision node for rollout
                expanded_node = expanded_node.expand(self.BNN1, self.weight_set1, self.BNN2, self.weight_set2, self.threshold, self.training_started, self.isi)
            result = expanded_node.rollout(self.BNN1, self.weight_set1, self.BNN2, self.weight_set2, self.threshold, self.training_started, self.isi)
            expanded_node.backpropagate(result)

    def traverse(self, node):
        while node.children:
            if node.is_decision_node():
                if self.training_started:
                    node = node.best_child(math.sqrt(2))
                else:
                    node = node.best_child(math.sqrt(2))
            elif self.is_terminal(node.state):
                return node
            else: # chance node
                node = node.expand(self.BNN1, self.weight_set1, self.BNN2, self.weight_set2, self.threshold, self.training_started, self.isi)
        return node

    def best_action(self):
        #for child in self.root.children:
        #    print("values", child.action, child.value, child.visits, child.value/child.visits)
        return max(self.root.children, key=lambda c: c.visits).action

    @staticmethod
    def __encode_action(action):
        a = np.array([0] * 4)
        a[action] = 1
        return a