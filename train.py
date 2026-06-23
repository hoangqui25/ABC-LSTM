import os
import json
import time
import random
import argparse
import numpy as np
import tensorflow as tf
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from metaheuristics.sma import SMA
from metaheuristics.abc import ABC
from metaheuristics.aro import ARO
from losses.loss import Loss
from datasets.stock import StockDataset
from utils.decode import cfg


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Train")

    parser.add_argument('--symbol', type=str,
                        help='stock symbol to fetch')
    parser.add_argument('--start', type=str, default='2018-01-01',
                        help='start date for fetching stock data (format: YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-01',
                        help='end date for fetching stock data (format: YYYY-MM-DD)')
    parser.add_argument('--look-back', type=int, default=30,
                        help='number of previous days used as input for LSTM model')
    parser.add_argument('--metaheuristic', type=str,
                        choices=['abc', 'sma', 'aro'],
                        help='metaheuristic algorithm used to optimize LSTM hyperparameters')
    parser.add_argument('--metaheuristic-epoch', type=int, default=50,
                        help='number of iterations for metaheuristic')
    parser.add_argument('--pop-size', type=int, default=20,
                        help='population size of optimizer')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for LSTM training')
    parser.add_argument('--save-dir', type=str, default='parameters',
                        help='directory to save best parameters')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    stock = StockDataset(args.symbol)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data_x, data_y = stock.load_dataset(start=args.start, end=args.end, features=features, target_feature='Close')

    look_back = args.look_back

    train_x, val_x, _ = stock.split_dataset(data=data_x, train_ratio=0.65, val_ratio=0.15)
    train_y, val_y, _ = stock.split_dataset(data=data_y, train_ratio=0.65, val_ratio=0.15)

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

    input_shape = (x_train.shape[1], x_train.shape[2])

    # Metaheuristic
    loss = Loss(
        input_shape=input_shape,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        batch_size=args.batch_size,
    )

    if args.metaheuristic == 'abc':
        metaheuristic = ABC(
            obj_func=loss.evaluate,
            lb=cfg['lb'], ub=cfg['ub'],
            pop_size=args.pop_size,
            epochs=args.metaheuristic_epoch,
            limits=(0.2 * args.metaheuristic_epoch)
        )
    elif args.metaheuristic == 'sma':
        metaheuristic = SMA(
            obj_func=loss.evaluate,
            lb=cfg['lb'], ub=cfg['ub'],
            pop_size=args.pop_size,
            epochs=args.metaheuristic_epoch
        )
    elif args.metaheuristic == 'aro':
        metaheuristic = ARO(
            obj_func=loss.evaluate,
            lb=cfg['lb'], ub=cfg['ub'],
            pop_size=args.pop_size,
            epochs=args.metaheuristic_epoch
        )

    start = time.time()
    best_params, best_score, history = metaheuristic.solve()
    end = time.time()

    run_time = end - start

    # Result
    print("Run time: ", run_time)
    print("History: ", history)
    print("Best parameters:", best_params)
    print("Best score:", best_score)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "best_params.json")

    best_params_serializable = best_params.tolist()
    best_score_serializable = float(best_score)

    with open(save_path, "w") as f:
        json.dump({
            "best_params": best_params_serializable,
            "best_scores": best_score_serializable
        }, f, indent=4)

    print(f"Saved best parameters and best score to {save_path}")
