"""
Configuration for MCTS training and evaluation.
"""

# Default configuration
DEFAULT_CONFIG = {
    # Training configuration
    'is_training': True,
    
    # Iteration configuration
    'iteration': 1,  # default iteration for output data paths

    # Model sizes configuration
    'policy_size': 135,  # default size for policy model
    'value_size': 135,   # default size for value model

    # Policy configuration
    'policy_model_template': "mstojkov/policy-{}-iter{}",  # common template for all sizes
    'openai_api_key': "sk-placeholder",
    
    'model_endpoints': {  # or 'model_api_configs'
        135: {
            'policy_api_base': "http://101.98.36.147:42202/v1",
            'value_api_base': "http://213.181.123.66:22839/predict"
        },
        360: {
            'policy_api_base': "http://79.160.189.79:14182/v1",
            'value_api_base': "http://45.135.56.11:32637/predict"
        },
        1700: {
            'policy_api_base': "http://136.38.166.236:34733/v1",
            'value_api_base': "http://45.135.56.11:26046/predict"
        }
    },

    # Forest configuration
    'num_trees': 100,
    'batch_size': 50,
    'max_workers': 50,
    
    # Tree configuration
    'branch_factor': 3,
    'max_expansions': 20,
    'temperature': 1.0,
    'c_explore': 0.3,
    
    # Training configuration
    'target_examples_train': 10000,
    'target_examples_val': 1000,
    
    # File paths configuration
    'input_data_paths': {  # Paths from which data is read
        'train_questions_path': '../data/raw_data/data_train.txt',
        'val_questions_path': '../data/raw_data/data_val.txt',
        'test_questions_path': '../data/raw_data/data_test.txt',
    },
    'output_data_paths': {  # Paths to which data is written
        'train_policy_data_path': '../data/iter{}/{}_{}/policy_training_data.jsonl',
        'val_policy_data_path': '../data/iter{}/{}_{}/policy_validation_data.jsonl',
        'train_value_data_path': '../data/iter{}/{}_{}/value_training_data.jsonl',
        'val_value_data_path': '../data/iter{}/{}_{}/value_validation_data.jsonl',
        'evaluation_stats': '../eval/eval_stats.csv',
    },
    'stats_interval': 30,
}

def get_config(is_training=None, iteration=None, 
               policy_size=None, value_size=None, 
               num_trees=None, batch_size=None, 
               branch_factor=None, max_expansions=None, temperature=None, 
               c_explore=None):
    """
    Get configuration dictionary.
    
    Args:
        iteration (int, optional): Model iteration number for output data paths
        policy_size (int, optional): Size of policy model (135, 360, or 1700)
        value_size (int, optional): Size of value model (0, 135, 360, or 1700)
        num_trees (int, optional): Number of trees in the forest
        batch_size (int, optional): Batch size for processing
        branch_factor (int, optional): Branch factor for MCTS
        max_expansions (int, optional): Maximum number of expansions
        temperature (float, optional): Temperature parameter
        c_explore (float, optional): Exploration constant
    
    Returns:
        Complete configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    # Update training configuration
    config.update({
        'is_training': is_training or config['is_training']
    })

    # Update iteration
    iter_num = iteration or config['iteration']
    config['iteration'] = iter_num
    
    # Update model sizes
    p_size = policy_size or config['policy_size']
    if value_size is not None:
        v_size = value_size
    else:
        v_size = config['value_size']
    config['policy_size'] = p_size
    config['value_size'] = v_size
    
    # Configure policy API
    policy_api_base = config['model_endpoints'][p_size]['policy_api_base']
    config.update({
        'policy_model': config['policy_model_template'].format(p_size, iter_num - 1),
        'openai_api_base': policy_api_base
    })
    
    # Configure value API
    if v_size == 0:
        config['value_api_base'] = None
    else:
        config['value_api_base'] = config['model_endpoints'][v_size]['value_api_base']
    
    # Update forest parameters
    config.update({
        'num_trees': num_trees or config['num_trees'],
        'batch_size': batch_size or config['batch_size']
    })

    # Update tree parameters
    config.update({
        'branch_factor': branch_factor or config['branch_factor'],
        'max_expansions': max_expansions or config['max_expansions'],
        'temperature': temperature or config['temperature'],
        'c_explore': c_explore or config['c_explore']
    })
    
    # Update data paths
    for key in config['output_data_paths']:
        config['output_data_paths'][key] = config['output_data_paths'][key].format(iter_num, p_size, v_size)
        
    return config
