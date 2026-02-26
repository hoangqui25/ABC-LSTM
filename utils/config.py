import numpy as np

neuron_options = np.arange(50, 210, 10).astype(int).tolist()
dropout_options = np.arange(0.0, 0.6, 0.1).astype(float).tolist()
cfg = {
    'lb': [0, 0, 0, 0],
    'ub': [15, 15, 15, 5],
}