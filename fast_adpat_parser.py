import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Script Parameters')

    # Define all the parameters

    # Core parameters for the run
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., cifar10, mnist)')
    parser.add_argument('--algorithm', type=str, required=True, help='Federated Learning Algorithm (e.g., dyn, fedavg, prox, moon)')
    parser.add_argument('--in_session_label_dist', type=str, required=True, help='label distribution within the session (e.g., two_shards, dirichlet)')
    parser.add_argument('--dirichlet_alpha', type=float, required=True, help='Dirichlet alpha (if in_session_label_dist is dirichlet)')
    parser.add_argument('--cross_session_label_overlap', type=float, required=True, help='overlap fraction between labels for consecutive sessions')
    parser.add_argument('--seed', type=int, required=True, help='random seed')
    parser.add_argument('--gpu_index', type=int, required=True, help='GPU index')

    # Parameters related to experimental setup/baselines
    parser.add_argument('--initial', type=int, required=True, help='Flag for proposed initial scheme (0 or 1)')
    parser.add_argument('--average', type=int, required=True, help='Flag for average baseline (0 or 1)')
    parser.add_argument('--previous', type=int, required=True, help='Flag for previous baseline (0 or 1)')
    parser.add_argument('--continuous', type=int, required=True, help='Flag for continuous baseline (0 or 1)')
    parser.add_argument('--recovery', type=int, required=True, help='Flag to measure recovery time (0 or 1)')

    # Session and Round parameters
    parser.add_argument('--num_sessions', type=int, required=True, help='total number of sessions')
    parser.add_argument('--num_clients', type=int, required=True, help='number of clients per session')
    parser.add_argument('--num_sessions_pilot', type=int, required=True, help='Number of pilot preparation sessions')
    parser.add_argument('--num_rounds_pilot', type=int, required=True, help='Number of rounds for pilot preparation')
    parser.add_argument('--num_rounds_actual', type=int, required=True, help='Number of rounds for actual training sessions')

    parser.add_argument('--lr_config_path', type=str, required=True,
                        help='Path to a JSON or YAML config file containing dataset-algorithm specific learning rates.')

    # Model and Optimization Parameters
    parser.add_argument('--momentum', type=float, required=True, help='Momentum for optimizer')
    parser.add_argument('--num_SGD_training', type=int, required=True, help='Number of local SGD training steps per round')
    parser.add_argument('--batch_size_training', type=int, required=True, help='Batch size for local training')
    parser.add_argument('--num_SGD_grad_cal', type=int, required=True, help='Number of local SGD steps for gradient calculation (if applicable to algorithm)')
    parser.add_argument('--batch_size_grad_cal', type=int, required=True, help='Batch size for gradient calculation (if applicable)')

    # Algorithm Specific Parameters (keep these if they are general to the algorithm, not just LR)
    parser.add_argument('--prox_alpha', type=float, required=True, help='Alpha Parameter for FedProx')
    parser.add_argument('--moon_mu', type=float, required=True, help='Mu Parameter for MOON')
    parser.add_argument('--moon_tau', type=float, required=True, help='Tau Parameter for MOON')
    parser.add_argument('--kl_coefficient', type=float, required=True, help='KL Divergence coefficient for continuous')
    parser.add_argument('--acg_beta', type=float, required=True, help='ACG Beta')
    parser.add_argument('--acg_lambda', type=float, required=True, help='ACG Lambda')
    parser.add_argument('--footprint_num_iteration', type=int, required=True, help='Footprint number of iterations (if applicable)')
    parser.add_argument('--similarity', type=str, required=True, help='Similarity measure (e.g., two_norm)')
    parser.add_argument('--similarity_scale', type=float, required=True, help='Similarity scale')

    args = parser.parse_args()
    return args