import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils.decode import decode

class Surrogate:
    def __init__(self, log_file):
        self.log_file = log_file

        self._ensure_dir()

        self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        self.is_ready = False

        self.load_history()

    def _ensure_dir(self):
        directory = os.path.dirname(self.log_file)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"[Surrogate] Directory created: {directory}")
            except OSError as e:
                print(f"[Surrogate] Error: Could not create directory {directory}: {e}")

    def load_history(self):
        if self._check_file_valid():
            try:
                df = pd.read_csv(self.log_file)
                if len(df) >= 20:
                    self.train_model(df)
                    print(f"[Surrogate] Restored memory from {len(df)} historical samples.")
            except Exception as e:
                print(f"[Surrogate] Warning: Failed to read history file ({e})")

    def train_model(self, df=None):
        try:
            if df is None:
                if not self._check_file_valid(): return
                df = pd.read_csv(self.log_file)

            df = df.dropna() # Remove invalid rows
            if len(df) < 100: return

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            self.model.fit(X, y)
            self.is_ready = True
        except Exception as e:
            print(f"[Surrogate] Training error: {e}")

    def predict(self, params):
        if not self.is_ready: return None
        params_array = np.array(params).flatten().reshape(1, -1)
        return self.model.predict(params_array)[0]

    def save_result(self, params, actual_loss):
        try:
            self._ensure_dir()
            decoded = decode(params)
            params_list = [decoded['layers']] + decoded['neurons'] + [decoded['dropout'], decoded['lr']]
            new_row = params_list + [actual_loss]
            df_new = pd.DataFrame([new_row])

            file_exists = os.path.exists(self.log_file)
            df_new.to_csv(self.log_file, mode='a', header=not file_exists, index=False)

            self.train_model()

        except Exception as e:
            print(f"[Surrogate] Error saving result: {e}")

    def _check_file_valid(self):
        return os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0
