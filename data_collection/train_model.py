import pickle
import os
import autograd.numpy as np
from BNN.BayesianNeuralNetwork import BayesianNeuralNetwork
from BNN.ExperienceReplay import ExperienceReplay

# Setup
if not os.path.isdir('./results'):
    os.mkdir('results')

domain = 'frozenlake'
num_batch_instances = 1
bnn_hidden_layer_size = 25
bnn_num_hidden_layers = 3
bnn_network_weights = None
eps_min = 0.1
grid_beta = 0.1
state_diffs = False

# Load experience buffer
with open(f'results/{domain}_exp_buffer', 'rb') as f:
    exp_buffer = pickle.load(f)
exp_buffer_np = np.vstack(exp_buffer)
inst_indices = exp_buffer_np[:, 4].astype(int)

# Group experiences by instance
exp_dict = {idx: exp_buffer_np[inst_indices == idx] for idx in range(1)}

# Prepare input and output for BNN
X = np.array([np.hstack([exp_buffer_np[tt, 0], exp_buffer_np[tt, 1]]) for tt in range(exp_buffer_np.shape[0])])
y = np.array([exp_buffer_np[tt, 3] for tt in range(exp_buffer_np.shape[0])])
num_dims = 2
num_actions = 4
num_wb = 5
if state_diffs:
    y -= X[:, :num_dims]

# Set up parameters for Bayesian Neural Network
relu = lambda x: np.maximum(x, 0.)
param_set = {
    'bnn_layer_sizes': [num_dims + num_actions + num_wb] + [bnn_hidden_layer_size] * bnn_num_hidden_layers + [num_dims * 2],
    'weight_count': num_wb,
    'num_state_dims': num_dims,
    'bnn_num_samples': 50,
    'bnn_batch_size': 32,
    'num_strata_samples': 5,
    'bnn_training_epochs': 1,
    'bnn_v_prior': 1,
    'bnn_learning_rate': 0.0005,
    'bnn_alpha': 0.5,
    'wb_num_epochs': 1,
    'wb_learning_rate': 0.0005
}

# Initialize latent weights for each instance
full_task_weights = np.random.normal(0., 0.1, (1, num_wb))

# Initialize Bayesian Neural Network
network = BayesianNeuralNetwork(param_set, nonlinearity=relu)

# Compute error before training
l2_errors = network.get_td_error(np.hstack((X, full_task_weights[inst_indices])), y, location=0.0, scale=1.0, by_dim=False)
print(f"Before training: Mean Error: {np.mean(l2_errors)}, Std Error: {np.std(l2_errors)}")

# Function to get random sample of indices
def get_random_sample(start, stop, size):
    return np.random.choice(np.arange(start, stop), size=size, replace=False)

# Train BNN and update latent weights
sample_size = 1000
for i in range(5):
    # Update BNN network weights
    network.fit_network(exp_buffer_np, full_task_weights, 0, state_diffs=state_diffs, use_all_exp=True)
    print(f'Finished BNN update {i}')

    # Compute error on random sample of transitions
    sample_indices = get_random_sample(0, X.shape[0], sample_size)
    l2_errors = network.get_td_error(np.hstack((X[sample_indices], full_task_weights[inst_indices[sample_indices]])), y[sample_indices], location=0.0, scale=1.0, by_dim=False)
    print(f"After BNN update: iter: {i}, Mean Error: {np.mean(l2_errors)}, Std Error: {np.std(l2_errors)}")

    # Update latent weights
    for inst in np.random.permutation(1):
        full_task_weights[inst, :] = network.optimize_latent_weighting_stochastic(exp_dict[inst], np.atleast_2d(full_task_weights[inst, :]), 0, state_diffs=state_diffs, use_all_exp=True)
    print(f'Finished WB update {i}')

    # Compute error after latent weight update
    l2_errors = network.get_td_error(np.hstack((X[sample_indices], full_task_weights[inst_indices[sample_indices]])), y[sample_indices], location=0.0, scale=1.0, by_dim=False)
    print(f"After Latent update: iter: {i}, Mean Error: {np.mean(l2_errors)}, Std Error: {np.std(l2_errors)}")

    # Save model weights
    with open(f'models/{domain}_network_weights_itr_{i}', 'wb') as f:
        pickle.dump(network.weights, f)
    with open(f'models/{domain}_latent_weights_itr_{i}', 'wb') as f:
        pickle.dump(full_task_weights, f)