from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input


def lstm(input_shape, decoded_params):
    layers = decoded_params['layers']
    neurons = decoded_params['neurons']
    dropout = decoded_params['dropout']

    model = Sequential()

    model.add(Input(shape=input_shape))

    for i in range(layers - 1):
        model.add(LSTM(units=neurons[i], activation='tanh', return_sequences=True))

    model.add(LSTM(units=neurons[layers-1], activation='tanh'))
    model.add(Dropout(dropout))

    model.add(Dense(units=1))
    return model
