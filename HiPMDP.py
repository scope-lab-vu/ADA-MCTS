from BNN.BayesianNeuralNetwork import *
#from nsbridge_simulator.nsbridge_v0 import NSBridgeV0 as model
from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
import pickle
class HiPMDP(object):
    """
    The Hidden Parameters-MDP
    """

    def __init__(self, domain, preset_hidden_params, run_type='full', episode_count=500, bnn_hidden_layer_size=25, bnn_num_hidden_layers=2, bnn_network_weights=None,
                 eps_min=0.15, test_inst=None, create_exp_batch=False, save_results=False,
                 grid_beta=0.23, print_output=False):
        """
        Initialize framework.
        """

        self.__initialize_params()

        # Store arguments
        self.domain = domain
        self.run_type = run_type
        self.preset_hidden_params = preset_hidden_params
        self.bnn_hidden_layer_size = bnn_hidden_layer_size
        self.bnn_num_hidden_layers = bnn_num_hidden_layers
        self.bnn_network_weights = bnn_network_weights
        self.eps_min = eps_min
        self.test_inst = test_inst
        self.create_exp_batch = create_exp_batch
        self.save_results = save_results
        self.grid_beta = grid_beta
        self.print_output = print_output
        # Set domain specific hyperparameters
        self.__set_domain_hyperparams()
        self.episode_count = episode_count

    def __initialize_params(self):
        """Initialize standard framework settings."""
        self.instance_count = 1  # number of task instances
        self.weight_count = 5  # number of latent weights
        self.eps_max = 1.0  # initial epsilon value for e-greedy policy
        self.bnn_and_latent_update_interval = 10  # Number of episodes between BNN and latent weight updates
        self.num_strata_samples = 5  # The number of samples we take from each strata of the experience buffer
        self.bnn_num_samples = 1000  # number of samples of network weights drawn to get each BNN prediction
        self.bnn_batch_size = 32
        self.bnn_v_prior = 3  # Prior variance on the BNN parameters
        self.bnn_training_epochs = 100  # number of epochs of SGD in each BNN update
        self.num_episodes_avg = 30  # number of episodes used in moving average reward to determine whether to stop DQN training
        self.wb_learning_rate = 0.0005  # latent weight learning rate
        self.bnn_alpha = 0.5  # BNN alpha divergence parameter
        self.eps_decay = 0.999  # Epsilon decay rate
        self.ddqn_batch_size = 50  # DDQN batch size
        # Prioritized experience replay hyperparameters
        self.PER_alpha = 0.2
        self.PER_beta_zero = 0.1
        self.wb_num_epochs = 100  # number of epochs of SGD in each latent weight update

    def __set_domain_hyperparams(self):
        if self.domain == 'grid':
            self.task = model(beta=self.grid_beta)
        elif self.domain == 'discretegrid':
            self.task = model(time=0)
        else:
            self.task = model()
        self.num_actions = self.task.num_actions  # number of actions
        self.num_dims = len(self.task.observe())  # number of state dimensions

    def __initialize_BNN(self):
        """Initialize the BNN and set pretrained network weights (if supplied)."""
        # Generate BNN layer sizes
        if self.run_type != 'full_linear':
            bnn_layer_sizes = [self.num_dims + self.num_actions + self.weight_count] + [
                self.bnn_hidden_layer_size] * self.bnn_num_hidden_layers + [self.num_dims * 2]
        else:
            bnn_layer_sizes = [self.num_dims + self.num_actions] + [
                self.bnn_hidden_layer_size] * self.bnn_num_hidden_layers + [self.num_dims * self.weight_count]
        # activation function
        self.bnn_learning_rate = 0.0005
        relu = lambda x: np.maximum(x, 0.0)
        # Gather parameters
        param_set = {
            'bnn_layer_sizes': bnn_layer_sizes,
            'weight_count': self.weight_count,
            'num_state_dims': self.num_dims,
            'bnn_num_samples': self.bnn_num_samples,
            'bnn_batch_size': self.bnn_batch_size,
            'num_strata_samples': self.num_strata_samples,
            'bnn_training_epochs': self.bnn_training_epochs,
            'bnn_v_prior': self.bnn_v_prior,
            'bnn_learning_rate': self.bnn_learning_rate,
            'bnn_alpha': self.bnn_alpha,
            'wb_learning_rate': self.wb_learning_rate,
            'wb_num_epochs': self.wb_num_epochs
        }
        if self.run_type != 'full_linear':
            self.network = BayesianNeuralNetwork(param_set, nonlinearity=relu)
        else:
            self.network = BayesianNeuralNetwork(param_set, nonlinearity=relu, linear_latent_weights=True)
        # Use previously trained network weights
        if self.bnn_network_weights is not None:
            self.network.weights = self.bnn_network_weights


def train_model(seed, domain, bnn2, weight_set2, best_net_work_error, best_latent_error, local_converge_count):
    with open('data_buffer/{}_{}_exp_buffer'.format(domain, seed), 'rb') as f:
        exp_buffer = pickle.load(f)
    instance_count = 1
    exp_buffer_np = np.vstack(exp_buffer)
    inst_indices = exp_buffer_np[:, 4].astype(int)
    exp_dict = {idx: exp_buffer_np[inst_indices == idx] for idx in range(instance_count)}
    X = np.array([np.hstack([exp_buffer_np[tt, 0], exp_buffer_np[tt, 1]]) for tt in range(exp_buffer_np.shape[0])])
    y = np.array([exp_buffer_np[tt, 3] for tt in range(exp_buffer_np.shape[0])])
    num_dims = 2
    num_actions = 4
    num_wb = 5
    relu = lambda x: np.maximum(x, 0.)
    state_diffs = False
    bnn_learning_rate = 0.00025
    wb_learning_rate = 0.00025
    if local_converge_count >= 2:
        tuples_list = [(0.0001, 0.0001), (0.0001, 0.0007), (0.0002, 0.0002)]
        chosen_tuple = random.choice(tuples_list)
        bnn_learning_rate, wb_learning_rate = chosen_tuple
    param_set = {
        'bnn_layer_sizes': [num_dims + num_actions + num_wb] + [bnn2.bnn_hidden_layer_size] * bnn2.bnn_num_hidden_layers + [
            num_dims * 2],
        'weight_count': num_wb,
        'num_state_dims': num_dims,
        'bnn_num_samples': 1000,
        'bnn_batch_size': 32,
        'num_strata_samples': 5,
        'bnn_training_epochs': 1,
        'bnn_v_prior': 1,
        'bnn_learning_rate': bnn_learning_rate,
        'bnn_alpha': 0.5,
        'wb_num_epochs': 1,
        'wb_learning_rate': wb_learning_rate
    }
    network_training = BayesianNeuralNetwork(param_set, nonlinearity=relu)
    network_training.weights = bnn2.network.weights
    output_network_weights = bnn2.network.weights
    output_latent_weights = weight_set2

    def get_random_sample(start, stop, size):
        indices_set = set()
        while len(indices_set) < size:
            indices_set.add(np.random.randint(start, stop))
            if len(indices_set) >= stop:
                break
        return np.array(list(indices_set))

    sample_size = 1000
    for i in range(3):
        network_training.fit_network(exp_buffer_np, weight_set2, 0, state_diffs=state_diffs, use_all_exp=True)
        if i % 1 == 0:
            sample_indices = get_random_sample(0, X.shape[0], sample_size)
            l2_errors = network_training.get_td_error(
                np.hstack((X[sample_indices], weight_set2[inst_indices[sample_indices]])), y[sample_indices],
                location=0.0, scale=1.0, by_dim=False)
            if (np.mean(l2_errors) + np.std(l2_errors)) < best_net_work_error:
                best_net_work_error = (np.mean(l2_errors) + np.std(l2_errors))
            best_latent_error = np.mean(l2_errors)
            output_network_weights = network_training.weights
            output_latent_weights = weight_set2.reshape(weight_set2.shape[1], )

    return output_network_weights, output_latent_weights, best_net_work_error, best_latent_error