from fastapi import FastAPI
from train_cme_model import train_cme_model
from predict_cme import DeepLSTM, load_models, prepare_recent_data, generate_predictions, save_predictions
import json
from fastapi.middleware.cors import CORSMiddleware
from email_sender import CMEAlertMailer
from dotenv import load_dotenv
import os
from model import AlertRequest

load_dotenv()
app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/train-cme")
def train_cme():
    # Train on your full dataset (2011-2025)
    train_cme_model(
        data_path="donki_data_cleaned.csv",
        output_dir="models",
        seq_length=10
    )
    return {"message": "CME model training initiated."}

@app.get("/predict-cme")
def predict_cme():
    # Configuration
    MODEL_DIR = "models"
    DATA_PATH = "donki_data_cleaned.csv"  # Your full 14-year dataset
    OUTPUT_FILE = "cme_predictions_30days.json"
    PREDICTION_DAYS = 30

    # Load models
    lstm_model, rf_models, gb_models, scaler, metadata = load_models(MODEL_DIR)

    # Prepare data
    df, X_scaled, all_features, target_features = prepare_recent_data(
        DATA_PATH, metadata, scaler
    )

    # Generate predictions
    predictions = generate_predictions(
        lstm_model, rf_models, gb_models, df, X_scaled,
        metadata, scaler, num_days=PREDICTION_DAYS
    )

    save_predictions(predictions, OUTPUT_FILE)
    data = []
    with open(OUTPUT_FILE, "r") as file:
        data = json.load(file)
        print(f"  ✓ Loaded predictions from {OUTPUT_FILE}")

    return {"message": "CME prediction endpoint", "data": data}

@app.post("/send-alert")
def send_alert(request: AlertRequest):

    # Load email credentials from environment variables
    SENDER_EMAIL = os.getenv("EMAIL")
    SENDER_PASSWORD = os.getenv("PASSWORD")

    # Recipient email
    RECIPIENT_EMAIL = request.email

    mailer = CMEAlertMailer(SENDER_EMAIL, SENDER_PASSWORD)

    mailer.send_alert(RECIPIENT_EMAIL, request.cme_data)

    return {"message": f"Alert sent to {RECIPIENT_EMAIL} if impact level is High or above."}