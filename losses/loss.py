import numpy as np
import random
from models.lstm import lstm
from utils.decode import decode
from models.surrogate import Surrogate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

class Loss():
    def __init__(self, x_train, y_train, x_val, y_val, input_shape, batch_size, learning_rate):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.surrogate = Surrogate(log_file='history/training_history.csv')

    def evaluate(self, params):
        if self.surrogate.is_ready and random.random() > 0.3:
            predicted_loss = self.surrogate.predict(params)
            print(f" [Surrogate] Params: {params} -> Predicted Loss: {predicted_loss:.6f} (Skip Train)")
            return predicted_loss
        print(f" [Real Train] Running actual LSTM...")
        decoded_params = decode(params)
        model = lstm(input_shape=self.input_shape, params=decoded_params)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=20,               
            restore_best_weights=True
        )

        history = model.fit(
            x=self.x_train, 
            y=self.y_train, 
            validation_data=(self.x_val, self.y_val),
            epochs=100, 
            batch_size=self.batch_size,
            callbacks=[early_stop],
        )

        val_losses = history.history['val_loss']
        min_loss_index = np.argmin(val_losses)
        last_k = val_losses[max(0, min_loss_index - 5) : min(min_loss_index + 6, len(val_losses))]
        avg_loss = np.mean(last_k)
        std_loss = np.std(last_k)
        
        loss = avg_loss + std_loss

        if np.isnan(loss) or np.isinf(loss):
            loss = 1

        self.surrogate.save_result(params, loss)

        return loss