from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input
from utils.config import neuron_options, dropout_options

        
def lstm(input_shape, params):
    neurons = params['neurons']

    dropout = params['dropout']

    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(LSTM(units=neuron_options[neurons[0]], activation='tanh', return_sequences=True))
    model.add(LSTM(units=neuron_options[neurons[1]], activation='tanh', return_sequences=True))
    model.add(LSTM(units=neuron_options[neurons[2]], activation='tanh'))
    model.add(Dropout(dropout_options[dropout]))

    model.add(Dense(units=1))
    return model
