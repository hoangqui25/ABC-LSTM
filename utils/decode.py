import numpy as np

layer_options = [1, 2, 3]
neuron_options = np.arange(50, 250, 10).tolist()
dropout_options = [0.1, 0.2, 0.3, 0.4]
lr_options = [0.01, 0.001, 0.0001]

cfg = {
    'lb': [0, 0, 0, 0, 0, 0],
    'ub': [2, 19, 19, 19, 3, 2],
}

def decode(x):
    layers = np.round(x[0]).astype(int)
    neurons = np.round(x[1:4]).astype(int)
    dropout = np.round(x[4]).astype(int)
    lr = np.round(x[5]).astype(int)

    return {
        "layers": layer_options[layers],
        "neurons": [neuron_options[neurons[0]],
                    neuron_options[neurons[1]],
                    neuron_options[neurons[2]]],
        "dropout": dropout_options[dropout],
        "lr": lr_options[lr],
    }
