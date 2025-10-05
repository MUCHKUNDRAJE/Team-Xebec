"""
CME Prediction Model Trainer
Train deep learning model on historical CME data and save for production use
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import pickle
import json
import os
from json_to_csv import JsonToCsvConverter

class DeepLSTM(nn.Module):
    """Enhanced LSTM with multiple layers and batch normalization"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(DeepLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.bn3 = nn.BatchNorm1d(hidden_size//4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout/2)
        
        self.fc4 = nn.Linear(hidden_size//4, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)
        return out

def create_sequences(data, seq_length, predict_cols_idx):
    """Create sequences for time series prediction"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length][predict_cols_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_cme_model(data_path, output_dir="models", seq_length=10):
    """
    Train CME prediction model on historical data
    
    Args:
        data_path: Path to CSV file with CME data
        output_dir: Directory to save trained models
        seq_length: Number of previous events to use for prediction
    """
    
    print("=" * 80)
    print("CME PREDICTION MODEL TRAINER")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\n[1/7] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"   ✓ Loaded {len(df)} CME events")
    print(f"   ✓ Date range: {df['time21_5'].min()} to {df['time21_5'].max()}")
    
    # Preprocess
    print("\n[2/7] Preprocessing data...")
    # Handle date parsing with errors='coerce' to catch invalid dates
    df['time21_5'] = pd.to_datetime(df['time21_5'].str.replace('Z', ''), errors='coerce')
    
    # Remove rows with invalid dates
    invalid_dates = df['time21_5'].isna().sum()
    if invalid_dates > 0:
        print(f"   ⚠ Found {invalid_dates} invalid/corrupted dates, removing them...")
        df = df.dropna(subset=['time21_5'])
    
    # Filter to reasonable date range (2000-2026)
    df = df[(df['time21_5'].dt.year >= 2000) & (df['time21_5'].dt.year <= 2026)]
    print(f"   ✓ Filtered to valid date range: {len(df)} events remain")
    
    df = df.sort_values('time21_5').reset_index(drop=True)
    
    # Extract features
    df['year'] = df['time21_5'].dt.year
    df['month'] = df['time21_5'].dt.month
    df['day'] = df['time21_5'].dt.day
    df['hour'] = df['time21_5'].dt.hour
    df['dayofyear'] = df['time21_5'].dt.dayofyear
    df['dayofweek'] = df['time21_5'].dt.dayofweek
    df['time_diff'] = df['time21_5'].diff().dt.total_seconds() / 3600
    df['time_diff'].fillna(df['time_diff'].median(), inplace=True)
    
    # Rolling statistics
    df['speed_rolling_mean'] = df['speed'].rolling(window=5, min_periods=1).mean()
    df['speed_rolling_std'] = df['speed'].rolling(window=5, min_periods=1).std().fillna(0)
    df['lat_rolling_mean'] = df['latitude'].rolling(window=5, min_periods=1).mean()
    df['lon_rolling_mean'] = df['longitude'].rolling(window=5, min_periods=1).mean()
    
    target_features = ['latitude', 'longitude', 'halfAngle', 'speed']
    time_features = ['year', 'month', 'day', 'hour', 'dayofyear', 'dayofweek', 'time_diff']
    rolling_features = ['speed_rolling_mean', 'speed_rolling_std', 'lat_rolling_mean', 'lon_rolling_mean']
    all_features = time_features + rolling_features + target_features
    
    # Impute missing values
    X = df[all_features].copy()
    imputer = SimpleImputer(strategy='mean')
    X[target_features] = imputer.fit_transform(X[target_features])
    X[rolling_features] = imputer.fit_transform(X[rolling_features])
    
    # Scale data
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print(f"   ✓ Features scaled to [0, 1]")
    
    # Create sequences
    print(f"\n[3/7] Creating sequences (length={seq_length})...")
    predict_cols_idx = [X.columns.get_loc(col) for col in target_features]
    X_seq, y_seq = create_sequences(X_scaled.values, seq_length, predict_cols_idx)
    print(f"   ✓ Created {len(X_seq)} sequences")
    
    # Split data
    train_size = int(len(X_seq) * 0.85)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    print(f"   ✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Convert to PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    # Train LSTM
    print("\n[4/7] Training Deep LSTM (4 layers, 128 units)...")
    input_size = X_train.shape[2]
    hidden_size = 128
    num_layers = 4
    output_size = len(target_features)
    
    lstm_model = DeepLSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    epochs = 200
    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(epochs):
        lstm_model.train()
        optimizer.zero_grad()
        output = lstm_model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        lstm_model.eval()
        with torch.no_grad():
            val_output = lstm_model(X_test_t)
            val_loss = criterion(val_output, y_test_t)
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = lstm_model.state_dict()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        if patience_counter >= patience:
            print(f"   ✓ Early stopping at epoch {epoch+1}")
            break
    
    lstm_model.load_state_dict(best_model_state) # type: ignore
    print(f"   ✓ Best validation loss: {best_loss:.4f}")
    
    # Train Random Forest
    print("\n[5/7] Training Random Forest (200 trees)...")
    rf_models = {}
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    for feature in target_features:
        idx = target_features.index(feature)
        rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5, 
            random_state=42, n_jobs=-1, verbose=0
        )
        rf_model.fit(X_train_flat, y_train[:, idx])
        rf_models[feature] = rf_model
    print(f"   ✓ Trained {len(rf_models)} RF models")
    
    # Train Gradient Boosting
    print("\n[6/7] Training Gradient Boosting (200 estimators)...")
    gb_models = {}
    
    for feature in target_features:
        idx = target_features.index(feature)
        gb_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, 
            random_state=42, verbose=0
        )
        gb_model.fit(X_train_flat, y_train[:, idx])
        gb_models[feature] = gb_model
    print(f"   ✓ Trained {len(gb_models)} GBM models")
    
    # Save models
    print(f"\n[7/7] Saving models to {output_dir}/...")
    
    # Save LSTM
    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'output_size': output_size
    }, f"{output_dir}/lstm_model.pth")
    print(f"   ✓ Saved LSTM model")
    
    # Save RF and GBM
    with open(f"{output_dir}/rf_models.pkl", 'wb') as f:
        pickle.dump(rf_models, f)
    with open(f"{output_dir}/gb_models.pkl", 'wb') as f:
        pickle.dump(gb_models, f)
    print(f"   ✓ Saved RF and GBM models")
    
    # Save preprocessing artifacts
    with open(f"{output_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"{output_dir}/imputer.pkl", 'wb') as f:
        pickle.dump(imputer, f)
    print(f"   ✓ Saved preprocessors")
    
    # Save metadata
    metadata = {
        'seq_length': seq_length,
        'target_features': target_features,
        'time_features': time_features,
        'rolling_features': rolling_features,
        'all_features': all_features,
        'training_date': datetime.now().isoformat(),
        'num_training_samples': len(X_train),
        'num_test_samples': len(X_test),
        'best_val_loss': float(best_loss),
        'data_date_range': {
            'start': df['time21_5'].min().isoformat(),
            'end': df['time21_5'].max().isoformat()
        },
        'statistics': {
            'avg_time_diff_hours': float(df['time_diff'].median()),
            'time_diff_std_hours': float(df['time_diff'].std())
        }
    }
    
    with open(f"{output_dir}/model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Saved metadata")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Models saved to: {os.path.abspath(output_dir)}/")
    print(f"  - lstm_model.pth")
    print(f"  - rf_models.pkl")
    print(f"  - gb_models.pkl")
    print(f"  - scaler.pkl")
    print(f"  - imputer.pkl")
    print(f"  - model_metadata.json")
    print("\nUse predict_cme.py to generate future predictions")
    print("=" * 80)

# if __name__ == "__main__":
#     # Train on your full dataset (2000-2025)
#     JsonToCsvConverter.convert()
#     train_cme_model(
#         data_path="donki_data_cleaned.csv",  # Your 25-year dataset
#         output_dir="models",
#         seq_length=10
#     )