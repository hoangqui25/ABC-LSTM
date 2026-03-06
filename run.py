import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from models.lstm import lstm
from utils.decode import decode
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from datasets.stock import StockDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


def parse_args():
    parser = argparse.ArgumentParser(description="Test")

    parser.add_argument('--symbol', type=str, 
                        help='stock symbol to fetch')
    parser.add_argument('--start', type=str, default='2018-01-01', 
                        help='start date for fetching stock data (format: YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-01', 
                        help='end date for fetching stock data (format: YYYY-MM-DD)')
    parser.add_argument('--look-back', type=int, default=30,
                        help='number of previous days used as input for LSTM model')
    parser.add_argument('--metaheuristic', type=str,
                        choices=['abc', 'sma', 'aro', 'none'],
                        help='metaheuristic algorithm used to optimize LSTM hyperparameters')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for LSTM training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate for Adam optimizer')
    parser.add_argument('--load-dir', type=str, default='parameters',
                        help='directory to load parameters')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    stock = StockDataset(args.symbol)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data_x, data_y = stock.load_dataset(start=args.start, end=args.end, features=features, target_feature='Close')
    look_back = args.look_back

    train_x, val_x, test_x = stock.split_dataset(data_x, 0.65, 0.15)
    train_y, val_y, test_y = stock.split_dataset(data_y, 0.65, 0.15)

    # Scaling X (features)
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    train_x_scaled = scaler_x.fit_transform(train_x)
    val_x_scaled = scaler_x.transform(val_x)

    # Scaling Y (close)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_y_scaled = scaler_y.fit_transform(train_y.reshape(-1, 1)).flatten()
    val_y_scaled = scaler_y.transform(val_y.reshape(-1, 1)).flatten()

    # Train
    train_y_smoothed = savgol_filter(train_y_scaled, window_length=9, polyorder=3)
    x_train, y_train = stock.create_dataset(train_x_scaled, train_y_smoothed, look_back)
    
    # Val
    last_train_x = train_x_scaled[-look_back:]
    last_train_y = train_y_scaled[-look_back:]
    val_x_concat = np.concatenate((last_train_x, val_x_scaled), axis=0)
    val_y_concat = np.concatenate((last_train_y, val_y_scaled), axis=0)
    x_val, y_val = stock.create_dataset(val_x_concat, val_y_concat, look_back)

    params_path = os.path.join(args.load_dir, "best_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"File {params_path} not found.")

    with open(params_path, "r") as f:
        config = json.load(f)
    best_params = decode(config["best_params"])
    input_shape = (x_train.shape[1], x_train.shape[2])
    print(f"Loaded params: {best_params}")


    print("\n[Phase 1] Searching for optimal epochs...")
    
    temp_model = lstm(input_shape=input_shape, params=best_params)
    lr = best_params.get('learning_rate', args.learning_rate)
    temp_model.compile(optimizer=Adam(lr), loss='mse')

    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True)
    
    history = temp_model.fit(
        x_train, y_train, 
        validation_data=(x_val, y_val),
        epochs=100, 
        batch_size=args.batch_size,
        verbose=1
    )
    
    val_loss_history = history.history['val_loss']
    optimal_epoch = np.argmin(val_loss_history) + 1 # Plus one because index starts from zero
    print(f"--> Found Optimal Epoch: {optimal_epoch}")

    print(f"\n[Phase 2] Retraining for {optimal_epoch} epochs...")

    stock = StockDataset(args.symbol)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data_x, data_y = stock.load_dataset(start=args.start, end=args.end, features=features, target_feature='Close')
    look_back = args.look_back

    train_x, test_x = stock.split_dataset(data_x, 0.8)
    train_y, test_y = stock.split_dataset(data_y, 0.8)

    # Scaling X (features)
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    train_x_scaled = scaler_x.fit_transform(train_x)
    test_x_scaled = scaler_x.transform(test_x)

    # Scaling Y (close)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_y_scaled = scaler_y.fit_transform(train_y.reshape(-1, 1)).flatten()
    test_y_scaled = scaler_y.transform(test_y.reshape(-1, 1)).flatten()

    # Train
    train_y_smoothed = savgol_filter(train_y_scaled, window_length=9, polyorder=3)
    x_train, y_train = stock.create_dataset(train_x_scaled, train_y_smoothed, look_back)
    
    # Test
    last_train_x = train_x_scaled[-look_back:]
    last_train_y = train_y_scaled[-look_back:]
    test_x_concat = np.concatenate((last_train_x, test_x_scaled), axis=0)
    test_y_concat = np.concatenate((last_train_y, test_y_scaled), axis=0)
    x_test, y_test = stock.create_dataset(test_x_concat, test_y_concat, look_back)

    # --- Test for 10 times ---
    all_results = []
    save_fig_dir = f"figures/{args.symbol.upper()}"
    os.makedirs(save_fig_dir, exist_ok=True)

    for i in range(1, 11):
        print(f"\n>>> RUN {i}/10 (Epochs: {optimal_epoch})")
        
        # Phase 2: Retrain
        final_model = lstm(input_shape=input_shape, params=best_params)
        final_model.compile(optimizer=Adam(lr), loss='mse')
        final_model.fit(
            x_train, y_train,
            epochs=optimal_epoch,
            batch_size=args.batch_size,
            verbose=1
        )

        # Phase 3: Predict
        pred_y_scaled = final_model.predict(x_test)
    
        # Inverse transform predictions
        y_pred = scaler_y.inverse_transform(pred_y_scaled.reshape(-1, 1)).flatten()
        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Metrics
        mae = mean_absolute_error(y_test_actual, y_pred)
        mse = mean_squared_error(y_test_actual, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test_actual, y_pred)
        r2 = r2_score(y_test_actual, y_pred)
        # Metrics
        res = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "R2": float(r2),
            "Epoch_Used": int(optimal_epoch)
        }
        all_results.append(res)
        print(f"Metrics: MAE={res['MAE']:.2f}, R2={res['R2']:.4f}")

        # Plot
        plt.figure(figsize=(12,6))
        plt.title(args.symbol.upper())
        plt.plot(y_test_actual, color='cornflowerblue', label="Actual Price")
        
        label_name = "LSTM" if args.metaheuristic == 'none' else f"{args.metaheuristic.upper()}-LSTM"
        plt.plot(y_pred, color='orange', label=label_name)
        
        plt.xlabel('Date')
        plt.ylabel('Close')
        plt.legend()
        plt.savefig(f"{save_fig_dir}/{i}.png")
        plt.close()

    # --- LOGGING ---
    log_dir = f"logs/{args.symbol.upper()}"
    os.makedirs(log_dir, exist_ok=True)
    
    log_data = {
        "symbol": args.symbol,
        "optimal_epoch": int(optimal_epoch),
        "average_metrics": {
            "MAE": float(np.mean([r['MAE'] for r in all_results])),
            "RMSE": float(np.mean([r['RMSE'] for r in all_results])),
            "MAPE": float(np.mean([r['MAPE'] for r in all_results])),
            "R2": float(np.mean([r['R2'] for r in all_results]))
        },
        "runs": all_results
    }

    with open(f"{log_dir}/test_log.json", "w") as f:
        json.dump(log_data, f, indent=4)
    
    print(f"\n Done! 10 runs used the {optimal_epoch} epochs.")
