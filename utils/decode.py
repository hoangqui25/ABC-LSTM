import numpy as np

def decode(x):
    neurons = np.round(x[0:3]).astype(int)
    dropout = np.round(x[3]).astype(int)
    
    return {
        "neurons": neurons,
        "dropout": dropout,
    }
