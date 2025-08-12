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
from HiPMDP import HiPMDP, train_model
import logging
from adamcts import MCTS


def __encode_action(action):
    """One-hot encodes the integer action supplied."""
    a = np.array([0] * 4)
    a[action] = 1
    return a

def run_task(seed, domain, network_weights1, latent_weights1, bnn_hidden_layer_size, bnn_num_hidden_layers, preset_hidden_params, run_type):
    hipmdp1 = HiPMDP(domain, preset_hidden_params,
                     run_type=run_type,
                     bnn_hidden_layer_size=bnn_hidden_layer_size,
                     bnn_num_hidden_layers=bnn_num_hidden_layers,
                     bnn_network_weights=network_weights1)
    hipmdp1._HiPMDP__initialize_BNN()
    weight_set1 = latent_weights1.reshape(latent_weights1.shape[1], )
    task = model()
    task.reset(1)
    hipmdp2 = HiPMDP(domain, preset_hidden_params, run_type=run_type, bnn_hidden_layer_size=bnn_hidden_layer_size,
                     bnn_num_hidden_layers=bnn_num_hidden_layers, bnn_network_weights=network_weights1)
    hipmdp2._HiPMDP__initialize_BNN()
    np.random.seed(seed)
    rng = np.random.default_rng()
    full_task_weights2 = rng.normal(0., 0.1, (1, 5))
    weight_set2 = full_task_weights2.reshape(full_task_weights2.shape[1], )
    reward = 0
    episode_buffer_bnn = []
    task = model()
    task.reset(1, seed)
    while not task.is_done():
        mcts_instance = MCTS(task.observe(), task.state.index, hipmdp1, hipmdp2, weight_set1, weight_set2, 0, task,
                             0.02, False, False)
        mcts_instance.search(5000)
        best_action = mcts_instance.best_action()
        next_state, reward, done, _ = task.step(best_action)
        episode_buffer_bnn.append(np.reshape(np.array([task.observe(), __encode_action(best_action), reward, next_state, 0]), [1, 5]))
        if len(episode_buffer_bnn) >= 50 and len(episode_buffer_bnn) % 5 == 0:
            exp_list = np.reshape(episode_buffer_bnn, [-1, 5])
            with open('data_buffer/{}_{}_exp_buffer'.format(domain, seed), 'wb') as f:
                pickle.dump(exp_list, f)
            train_model(seed, domain, hipmdp1, full_task_weights2, 100, 100, 0)
        task.render()
        time.sleep(0.1)
    return reward

if __name__ == '__main__':
    domain = 'frozenlake'
    with open('models/{}_network_weights_itr_2'.format(domain), 'rb') as f1:
        network_weights1 = pickle.load(f1)
    with open('models/{}_latent_weights_itr_2'.format(domain), 'rb') as f3:
        latent_weights1 = pickle.load(f3)

    bnn_hidden_layer_size = 25
    bnn_num_hidden_layers = 3
    preset_hidden_params = [{'latent_code': 1}]
    run_type = "full"
    seeds = [2, 3]

    with Pool() as pool:
        results = pool.starmap(run_task, [(seed, domain, network_weights1, latent_weights1, bnn_hidden_layer_size, bnn_num_hidden_layers, preset_hidden_params, run_type) for seed in seeds])

    # Set up logging configuration at the beginning
    logging.basicConfig(
        filename='results.log',
        filemode='a',  # Append to the file
        level=logging.INFO,  # Set the minimum level of logging to INFO
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # Create a logger
    logger = logging.getLogger(__name__)

    # Log header
    logger.info("Discounted Return, Count")

    # Log final results if needed
    for seed, reward in zip(seeds, results):
        if reward is not None:
            logger.info(f"Final Reward for Seed {seed}: {reward}")