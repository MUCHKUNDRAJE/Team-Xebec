"""
CME Future Prediction System
Load trained models and generate 30-day predictions
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
from datetime import datetime, timedelta
import os

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

def load_models(model_dir="models"):
    """Load all trained models and preprocessors"""
    
    print("Loading trained models...")
    
    # Load metadata
    with open(f"{model_dir}/model_metadata.json", 'r') as f:
        metadata = json.load(f)
    print(f"  ✓ Loaded metadata (trained on {metadata['num_training_samples']} samples)")
    
    # Load LSTM
    checkpoint = torch.load(f"{model_dir}/lstm_model.pth")
    lstm_model = DeepLSTM(
        checkpoint['input_size'],
        checkpoint['hidden_size'],
        checkpoint['num_layers'],
        checkpoint['output_size']
    )
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()
    print(f"  ✓ Loaded LSTM model")
    
    # Load RF and GBM
    with open(f"{model_dir}/rf_models.pkl", 'rb') as f:
        rf_models = pickle.load(f)
    with open(f"{model_dir}/gb_models.pkl", 'rb') as f:
        gb_models = pickle.load(f)
    print(f"  ✓ Loaded ensemble models")
    
    # Load scaler only
    with open(f"{model_dir}/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    print(f"  ✓ Loaded preprocessors")
    
    return lstm_model, rf_models, gb_models, scaler, metadata

def prepare_recent_data(data_path, metadata, scaler):
    """Prepare recent data for prediction"""
    
    print(f"\nPreparing recent data from {data_path}...")
    
    df = pd.read_csv(data_path)
    
    # Handle date parsing with error handling
    df['time21_5'] = pd.to_datetime(df['time21_5'].str.replace('Z', ''), errors='coerce')
    
    # Remove invalid dates and filter date range
    invalid_dates = df['time21_5'].isna().sum()
    if invalid_dates > 0:
        print(f"  ⚠ Removed {invalid_dates} invalid dates")
        df = df.dropna(subset=['time21_5'])
    
    df = df[(df['time21_5'].dt.year >= 2000) & (df['time21_5'].dt.year <= 2026)]
    df = df.sort_values('time21_5').reset_index(drop=True)
    
    # Extract same features as training
    df['year'] = df['time21_5'].dt.year
    df['month'] = df['time21_5'].dt.month
    df['day'] = df['time21_5'].dt.day
    df['hour'] = df['time21_5'].dt.hour
    df['dayofyear'] = df['time21_5'].dt.dayofyear
    df['dayofweek'] = df['time21_5'].dt.dayofweek
    df['time_diff'] = df['time21_5'].diff().dt.total_seconds() / 3600
    df['time_diff'] = df['time_diff'].fillna(metadata['statistics']['avg_time_diff_hours'])
    
    df['speed_rolling_mean'] = df['speed'].rolling(window=5, min_periods=1).mean()
    df['speed_rolling_std'] = df['speed'].rolling(window=5, min_periods=1).std().fillna(0)
    df['lat_rolling_mean'] = df['latitude'].rolling(window=5, min_periods=1).mean()
    df['lon_rolling_mean'] = df['longitude'].rolling(window=5, min_periods=1).mean()
    
    all_features = metadata['all_features']
    target_features = metadata['target_features']
    rolling_features = metadata['rolling_features']
    
    X = df[all_features].copy()
    
    # Fill missing values with mean (don't use saved imputer)
    from sklearn.impute import SimpleImputer
    imputer_temp = SimpleImputer(strategy='mean')
    X[target_features] = imputer_temp.fit_transform(X[target_features])
    X[rolling_features] = imputer_temp.fit_transform(X[rolling_features])
    
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    print(f"  ✓ Prepared {len(df)} events")
    print(f"  ✓ Latest event: {df['time21_5'].iloc[-1]}")
    
    return df, X_scaled, all_features, target_features

def generate_predictions(lstm_model, rf_models, gb_models, df, X_scaled, 
                        metadata, scaler, num_days=30):
    """Generate future predictions"""
    
    print(f"\nGenerating {num_days}-day predictions...")
    
    seq_length = metadata['seq_length']
    all_features = metadata['all_features']
    target_features = metadata['target_features']
    
    # Calculate prediction count
    date_range = (df['time21_5'].max() - df['time21_5'].min()).days
    avg_events_per_day = len(df) / date_range if date_range > 0 else 1
    num_predictions = max(num_days, int(avg_events_per_day * num_days))
    num_predictions = min(num_predictions, 100)
    
    last_seq = torch.tensor(X_scaled.values[-seq_length:].reshape(1, seq_length, -1), 
                           dtype=torch.float32)
    last_time = df['time21_5'].iloc[-1]
    
    future_predictions = []
    
    for i in range(num_predictions):
        # Get predictions from all models
        with torch.no_grad():
            lstm_pred = lstm_model(last_seq).numpy()[0]
        
        last_seq_flat = last_seq.numpy().reshape(1, -1)
        rf_pred = np.array([rf_models[f].predict(last_seq_flat)[0] for f in target_features])
        gb_pred = np.array([gb_models[f].predict(last_seq_flat)[0] for f in target_features])
        
        # Weighted ensemble
        ensemble_pred = 0.4 * lstm_pred + 0.3 * rf_pred + 0.3 * gb_pred
        
        # Denormalize
        pred_full = np.zeros((1, len(all_features)))
        for j, feat in enumerate(target_features):
            idx = all_features.index(feat)
            pred_full[0, idx] = ensemble_pred[j]
        
        pred_denorm = scaler.inverse_transform(pred_full)[0]
        denorm_targets = [pred_denorm[all_features.index(f)] for f in target_features]
        
        # Predict timing
        avg_time_diff = metadata['statistics']['avg_time_diff_hours']
        time_variance = metadata['statistics']['time_diff_std_hours'] * 0.5
        next_time_diff = np.random.normal(avg_time_diff, time_variance)
        next_time_diff = max(1, next_time_diff)
        next_time = last_time + timedelta(hours=next_time_diff)
        
        # Extract values
        speed_val = float(denorm_targets[3])
        lat_val = float(denorm_targets[0])
        lon_val = float(denorm_targets[1])
        half_angle_val = float(denorm_targets[2])
        
        # Impact assessment
        if speed_val > 1000:
            impact_level, severity_score = "Extreme", 10
        elif speed_val > 710:
            impact_level, severity_score = "Very High", 8
        elif speed_val > 650:
            impact_level, severity_score = "High", 6
        elif speed_val > 350:
            impact_level, severity_score = "Medium", 4
        else:
            impact_level, severity_score = "Low", 2
        
        earth_directed = abs(lon_val) < 30 and abs(lat_val) < 30
        if earth_directed:
            severity_score += 2
        
        # Confidence decay
        days_out = (next_time - df['time21_5'].iloc[-1]).days
        confidence = max(0.30, 0.90 - 0.02 * days_out)
        
        # Arrival time
        arrival_time = None
        if earth_directed and speed_val > 0:
            travel_hours = (1.5e8) / (speed_val * 3600)
            arrival_time = (next_time + timedelta(hours=travel_hours)).strftime("%Y-%m-%dT%H:%M") + "Z"
        
        # Create prediction
        prediction = {
            "prediction_id": i + 1,
            "time21_5": next_time.strftime("%Y-%m-%dT%H:%M") + "Z",
            "days_from_now": round((next_time - datetime.now()).total_seconds() / 86400, 2),
            "cme_parameters": {
                "latitude": round(lat_val, 2),
                "longitude": round(lon_val, 2),
                "halfAngle": round(half_angle_val, 2),
                "speed": round(speed_val, 2)
            },
            "impact_assessment": {
                "impact_level": impact_level,
                "severity_score": severity_score,
                "earth_directed": earth_directed,
                "estimated_arrival": arrival_time,
                "geomagnetic_storm_potential": "High" if severity_score >= 8 else "Moderate" if severity_score >= 5 else "Low"
            },
            "metadata": {
                "type": "S",
                "isMostAccurate": True,
                "catalog": "M2M_CATALOG",
                "dataLevel": "0",
                "confidence_score": round(confidence, 3),
                "prediction_method": "Deep-LSTM + RF + GBM Ensemble"
            }
        }
        
        future_predictions.append(prediction)
        
        # Update sequence
        new_row = np.zeros(len(all_features))
        new_row[all_features.index('year')] = next_time.year
        new_row[all_features.index('month')] = next_time.month
        new_row[all_features.index('day')] = next_time.day
        new_row[all_features.index('hour')] = next_time.hour
        new_row[all_features.index('dayofyear')] = next_time.timetuple().tm_yday
        new_row[all_features.index('dayofweek')] = next_time.weekday()
        new_row[all_features.index('time_diff')] = next_time_diff
        
        for feat in target_features:
            idx = all_features.index(feat)
            new_row[idx] = pred_denorm[idx]
        
        new_row_scaled = scaler.transform(new_row.reshape(1, -1))[0]
        
        new_seq = last_seq.clone()
        new_seq[0, :-1, :] = last_seq[0, 1:, :]
        new_seq[0, -1, :] = torch.tensor(new_row_scaled, dtype=torch.float32)
        last_seq = new_seq
        last_time = next_time
        
        if (next_time - df['time21_5'].iloc[-1]).days >= num_days:
            break
    
    return future_predictions

def save_predictions(predictions, output_file="cme_predictions_30days.json"):
    """Save predictions and print summary"""
    
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n✓ Saved {len(predictions)} predictions to {output_file}")
    print(f"✓ Time range: {predictions[0]['time21_5']} to {predictions[-1]['time21_5']}")
    
    # Summary
    high_impact = sum(1 for p in predictions if p['impact_assessment']['impact_level'] in ['High', 'Very High', 'Extreme'])
    earth_directed = sum(1 for p in predictions if p['impact_assessment']['earth_directed'])
    
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Predictions: {len(predictions)}")
    print(f"High Impact Events: {high_impact}")
    print(f"Earth-Directed Events: {earth_directed}")
    print(f"Average Speed: {np.mean([p['cme_parameters']['speed'] for p in predictions]):.2f} km/s")
    print(f"Average Confidence: {np.mean([p['metadata']['confidence_score'] for p in predictions]):.3f}")
    
    # Show critical events
    critical = [p for p in predictions if p['impact_assessment']['earth_directed'] 
                and p['impact_assessment']['severity_score'] >= 6]
    
    if critical:
        print(f"\n{'='*60}")
        print("CRITICAL EARTH-DIRECTED EVENTS")
        print(f"{'='*60}")
        for p in critical[:5]:  # Show top 5
            print(f"\nEvent #{p['prediction_id']}")
            print(f"  Time: {p['time21_5']}")
            print(f"  Speed: {p['cme_parameters']['speed']} km/s")
            print(f"  Impact: {p['impact_assessment']['impact_level']}")
            print(f"  Arrival: {p['impact_assessment']['estimated_arrival']}")
            print(f"  Storm Potential: {p['impact_assessment']['geomagnetic_storm_potential']}")
    
    print(f"\n{'='*60}\n")

# if __name__ == "__main__":
#     # Configuration
#     MODEL_DIR = "models"
#     DATA_PATH = "donki_data.csv"  # Your full 25-year dataset
#     OUTPUT_FILE = "cme_predictions_30days.json"
#     PREDICTION_DAYS = 30
    
#     print("=" * 60)
#     print("CME FUTURE PREDICTION SYSTEM")
#     print("=" * 60)
    
#     # Load models
#     lstm_model, rf_models, gb_models, scaler, metadata = load_models(MODEL_DIR)
    
#     # Prepare data
#     df, X_scaled, all_features, target_features = prepare_recent_data(
#         DATA_PATH, metadata, scaler
#     )
    
#     # Generate predictions
#     predictions = generate_predictions(
#         lstm_model, rf_models, gb_models, df, X_scaled,
#         metadata, scaler, num_days=PREDICTION_DAYS
#     )
    
#     # Save and summarize
#     save_predictions(predictions, OUTPUT_FILE)